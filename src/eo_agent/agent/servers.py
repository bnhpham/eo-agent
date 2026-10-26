"""
The three stdio MCP servers of the agent, with client-side post-processing of their results.
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from agents.mcp import MCPServerStdio, create_static_tool_filter
from mcp.types import CallToolResult, TextContent

from eo_agent.agent.artifacts import ToolRecord
from eo_agent.agent.postprocess import CDSE_SOURCE, GDACS_SOURCE, PAST_EVENTS_TOOL, SELECT_TOOL, ShownPair, budget_exceeded_response, cdse_reference, clamp_pairs_argument, compact_gdacs_payload, ensure_cdse_preset, excluded_post_dates, gdacs_call_failed, pairs_used, save_cdse_image, shown_from_records, slim_imagery_response, slim_past_events_response, wrap_untrusted
from eo_agent.config import PROJECT_ROOT, Settings

logger = logging.getLogger(__name__)

ANALYSIS_SOURCE = "eo-analysis MCP server (this project): Sentinel-2 imagery selection, ML4Floods / Prithvi inference, geodesic area"
GDACS_RETRY_DELAY_S = 2.0
GDACS_RETRY_DELAYS_S = (3.0, 8.0)  # waits before the second and the third attempt of a failed GDACS search
GDACS_PARALLEL_CALLS = 2
GDACS_CACHE_S = 900
GEOCODE_SOURCE = "OpenStreetMap Nominatim via ESA EVE geocode MCP server"
GEOCODE_ALLOWED_TOOLS = ["geocode_place"]
CDSE_ALLOWED_TOOLS = ["fetch_sentinel2", "fetch_sentinel1_grd"]


class RecordingMCPServer(MCPServerStdio):
    """
    Wraps every tool result as untrusted data, converts failures into error results and records results for the UI.
    """

    source = ANALYSIS_SOURCE

    def __init__(self, recorder: list[ToolRecord], **kwargs: Any):
        super().__init__(**kwargs)
        self._recorder = recorder

    def prepare_arguments(self, tool_name: str, arguments: dict | None) -> dict | None:
        return arguments

    def precheck(self, tool_name: str, arguments: dict | None) -> Any | None:
        """Return a result to short-circuit the upstream call, or None to proceed."""
        return None

    def llm_view(self, tool_name: str, data: Any) -> Any:
        """What the model sees. The UI always gets the full record."""
        return data

    def transform(self, tool_name: str, arguments: dict, result: CallToolResult) -> Any:
        if result.structuredContent:
            return result.structuredContent
        texts = [item.text for item in result.content if isinstance(item, TextContent)]
        text = "\n".join(texts)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None, meta: dict[str, Any] | None = None) -> CallToolResult:
        
        arguments = self.prepare_arguments(tool_name, arguments)
        
        try:
            blocked = self.precheck(tool_name, arguments)
            if blocked is not None:
                is_error, data = False, blocked
            else:
                result = await super().call_tool(tool_name, arguments, meta)
                is_error = bool(result.isError)
                data = self.transform(tool_name, arguments or {}, result) if not is_error else " ".join(item.text for item in result.content if isinstance(item, TextContent))
        except Exception as exc:
            logger.warning("MCP tool %s on %s failed: %s", tool_name, self.name, exc)
            is_error, data = True, {"error": f"{type(exc).__name__}: {exc}"[:1000]}
        
        self._recorder.append(ToolRecord(server=self.name, tool=tool_name, arguments=arguments or {}, data=data, is_error=is_error))
        payload = self.llm_view(tool_name, data) if not is_error else {"error": data}
        
        return CallToolResult(content=[TextContent(type="text", text=wrap_untrusted(self.source, payload))], isError=is_error)


class AnalysisServer(RecordingMCPServer):
    """
    The custom analysis server. Enforces the per-answer pair cap and keeps bulky imagery metadata out of the model context.
    """

    def __init__(self, recorder: list[ToolRecord], max_pairs_per_answer: int, **kwargs: Any):
        super().__init__(recorder, **kwargs)
        self._max_pairs = max_pairs_per_answer
        self._shown: list[ShownPair] = []

    def set_shown(self, shown: list[ShownPair]) -> None:
        """Pairs shown in earlier answers of the current session. Set by the runtime before every turn."""
        self._shown = list(shown)

    def _remaining(self) -> int:
        return self._max_pairs - pairs_used(self._recorder)

    def prepare_arguments(self, tool_name: str, arguments: dict | None) -> dict | None:
        if tool_name != SELECT_TOOL:
            return arguments
        arguments = clamp_pairs_argument(arguments, max(self._remaining(), 1))
        # Images already shown (earlier answers and earlier calls of this answer) are skipped, unless the user explicitly asked to see them again.
        skip = {str(item) for item in arguments.get("exclude_post_dates") or []} | set(excluded_post_dates(arguments, self._shown + shown_from_records(self._recorder)))
        if skip:
            arguments["exclude_post_dates"] = sorted(skip)
        return arguments

    def precheck(self, tool_name: str, arguments: dict | None) -> Any | None:
        if tool_name == SELECT_TOOL and self._remaining() <= 0:
            return budget_exceeded_response(self._max_pairs)
        return None

    def llm_view(self, tool_name: str, data: Any) -> Any:
        if tool_name == PAST_EVENTS_TOOL:
            return slim_past_events_response(data)
        return slim_imagery_response(data) if tool_name == SELECT_TOOL else data


class GdacsServer(RecordingMCPServer):
    """
    ESA EVE GDACS server.
    
    The GDACS API is slow and often times out, so calls run at most two at a time.
    Failed searches are retried twice and good answers are cached for 15 minutes.
    """

    source = GDACS_SOURCE

    def __init__(self, recorder: list[ToolRecord], max_events: int, **kwargs: Any):
        super().__init__(recorder, **kwargs)
        self._max_events = max_events
        self._slots = asyncio.Semaphore(GDACS_PARALLEL_CALLS)
        self._cache: dict[str, tuple[float, Any]] = {}

    @staticmethod
    def _key(tool_name: str, arguments: dict | None) -> str:
        return tool_name + json.dumps(arguments or {}, sort_keys=True, default=str)

    def transform(self, tool_name: str, arguments: dict, result: CallToolResult) -> Any:
        text = "\n".join(item.text for item in result.content if isinstance(item, TextContent))
        return compact_gdacs_payload(text, self._max_events)

    def precheck(self, tool_name: str, arguments: dict | None) -> Any | None:
        cached = self._cache.get(self._key(tool_name, arguments))
        if cached is None or time.monotonic() - cached[0] > GDACS_CACHE_S:
            return None
        
        age_s = int(time.monotonic() - cached[0])
        return {**cached[1], "cache_note": f"Same search answered {age_s} s ago by GDACS, the live API was not asked again."}

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None, meta: dict[str, Any] | None = None) -> CallToolResult:
        
        key = self._key(tool_name, arguments)
        async with self._slots:
            for attempt, delay in enumerate((0.0, *GDACS_RETRY_DELAYS_S)):
                if delay:
                    await asyncio.sleep(delay)
                
                result = await super().call_tool(tool_name, arguments, meta)
                record = self._recorder[-1] if self._recorder else None
                
                failed = record is not None and record.tool == tool_name and gdacs_call_failed(record.data, record.is_error)
                if not failed:
                    if record is not None and isinstance(record.data, dict) and "events" in record.data and "cache_note" not in record.data and "error" not in record.data:
                        self._cache[key] = (time.monotonic(), record.data)
                    
                    return result
                
                if attempt < len(GDACS_RETRY_DELAYS_S):
                    self._recorder.pop()
        
        return result


class GeocodeServer(RecordingMCPServer):
    """
    ESA EVE geocode MCP server (OpenStreetMap Nominatim)
    """

    source = GEOCODE_SOURCE

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None, meta: dict[str, Any] | None = None) -> CallToolResult:
        result = await super().call_tool(tool_name, arguments, meta)
        failure = self._recorder[-1] if self._recorder else None
        
        # Nominatim rate-limits and times out now and then. One retry before the failure is reported.
        if failure is not None and failure.tool == tool_name and (failure.is_error or (isinstance(failure.data, dict) and str(failure.data.get("error", "")).startswith("Geocoding failed"))):
            self._recorder.pop()
            await asyncio.sleep(GDACS_RETRY_DELAY_S)
            result = await super().call_tool(tool_name, arguments, meta)
        
        return result


class CdseServer(RecordingMCPServer):
    """
    ESA EVE cdse MCP server (Retrieve Sentinel-1, -2, -3, -5P, Landsat and DEM imagery from the Copernicus Data Space Ecosystem (CDSE))
    """

    source = CDSE_SOURCE

    def __init__(self, recorder: list[ToolRecord], quicklook_dir: Path, **kwargs: Any):
        super().__init__(recorder, **kwargs)
        self._quicklook_dir = quicklook_dir

    def prepare_arguments(self, tool_name: str, arguments: dict | None) -> dict | None:
        arguments = ensure_cdse_preset(arguments)
        # save_path would write files wherever the model asks, so it is always dropped.
        arguments.pop("save_path", None)

        if tool_name == "fetch_sentinel2":
            arguments.setdefault("collection", "L2A")

        return arguments

    def transform(self, tool_name: str, arguments: dict, result: CallToolResult) -> Any:
        image_path, meta = None, None

        for item in result.content:
            if getattr(item, "type", None) == "image":
                image_path = save_cdse_image(item.data, item.mimeType, self._quicklook_dir, tool_name, arguments)
            elif isinstance(item, TextContent):
                try:
                    meta = json.loads(item.text)
                except json.JSONDecodeError:
                    meta = {"text": item.text[:500]}

        return cdse_reference(tool_name, arguments, image_path, meta)


def _subprocess_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(PROJECT_ROOT / "src"), env.get("PYTHONPATH")]))
    env.update(extra or {})
    return env


# GDACS, geocode, CDSE (only with credentials) and the custom analysis server, all over stdio with the current interpreter.
def build_servers(settings: Settings, recorder: list[ToolRecord]) -> list[RecordingMCPServer]:

    python = sys.executable
    servers_dir = settings.registry_dir / "servers"
    servers: list[RecordingMCPServer] = []

    gdacs_script = servers_dir / "gdacs" / "server.py"
    if gdacs_script.exists():
        servers.append(GdacsServer(recorder,
                                   settings.gdacs_max_events,
                                   name="gdacs",
                                   cache_tools_list=True,
                                   client_session_timeout_seconds=90,
                                   max_retry_attempts=1,
                                   params={"command": python,
                                           "args": [str(gdacs_script), "--transport", "stdio"],
                                           "cwd": str(gdacs_script.parent),
                                           "env": _subprocess_env()}))
    else:
        logger.error("GDACS MCP server not found at %s. Run scripts/setup_env.ps1.", gdacs_script)

    geocode_script = servers_dir / "geocode" / "server.py"
    if geocode_script.exists():
        servers.append(GeocodeServer(recorder,
                                     name="geocode",
                                     cache_tools_list=True,
                                     client_session_timeout_seconds=90,
                                     tool_filter=create_static_tool_filter(allowed_tool_names=GEOCODE_ALLOWED_TOOLS),
                                     params={"command": python,
                                             "args": [str(geocode_script), "--transport", "stdio"],
                                             "cwd": str(geocode_script.parent),
                                             "env": _subprocess_env()}))

    cdse_script = servers_dir / "cdse" / "server.py"
    if settings.has_cdse_credentials and cdse_script.exists():
        credentials = {"CDSE_CLIENT_ID": settings.cdse_client_id, "CDSE_CLIENT_SECRET": settings.cdse_client_secret.get_secret_value()}
        
        servers.append(CdseServer(recorder,
                                  settings.artifacts_dir / "quicklooks",
                                  name="cdse",
                                  cache_tools_list=True,
                                  client_session_timeout_seconds=150,
                                  tool_filter=create_static_tool_filter(allowed_tool_names=CDSE_ALLOWED_TOOLS),
                                  params={"command": python,
                                          "args": [str(cdse_script), "--transport", "stdio"],
                                          "cwd": str(cdse_script.parent),
                                          "env": _subprocess_env(credentials)}))

    servers.append(AnalysisServer(recorder,
                                  settings.max_pairs_per_answer,
                                  name="eo-analysis",
                                  cache_tools_list=True,
                                  client_session_timeout_seconds=settings.mcp_timeout_s,
                                  params={"command": python,
                                          "args": ["-m", "eo_agent.mcp_server.server"],
                                          "cwd": str(PROJECT_ROOT),
                                          "env": _subprocess_env()}))
    
    return servers
