"""
Long-lived agent runtime.

MCP stdio sessions are bound to the event loop that opened them, and Streamlit reruns the script in fresh threads.
So one asyncio loop runs in a background thread for the lifetime of the process. The MCP servers stay connected there
(the ML models stay loaded in the analysis server) and turns are submitted with run_coroutine_threadsafe.
"""

import asyncio
import logging
import threading
from collections import defaultdict
from contextlib import AsyncExitStack
from dataclasses import dataclass, field

import openai
from agents import Agent, MaxTurnsExceeded, Runner, SQLiteSession, set_default_openai_key

from eo_agent.agent.artifacts import ToolRecord, TurnArtifacts, collect
from eo_agent.agent.guardrails import UNGROUNDED_WARNING, check_area_grounding, validate_user_input
from eo_agent.agent.instructions import build_instructions
from eo_agent.agent.postprocess import ShownPair, shown_from_pairs
from eo_agent.agent.servers import AnalysisServer, RecordingMCPServer, build_servers
from eo_agent.config import Settings
from eo_agent.schemas import utc_now

logger = logging.getLogger(__name__)

STARTUP_TIMEOUT_S = 120
MAX_AGENT_TURNS = 25


@dataclass
class TurnResult:
    answer: str
    artifacts: TurnArtifacts = field(default_factory=TurnArtifacts)
    notes: list[str] = field(default_factory=list)


class AgentRuntime:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_name = settings.openai_model or None
        self.server_status: dict[str, bool] = {}
        self.startup_error: str | None = None
        self.startup_warnings: list[str] = []
        self._recorder: list[ToolRecord] = []
        self._servers: list[RecordingMCPServer] = []
        self._session_area_values: dict[str, list[float]] = defaultdict(list)
        self._session_shown: dict[str, list[ShownPair]] = defaultdict(list)
        self._turn_lock = threading.Lock()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="agent-runtime", daemon=True)
        self._stack: AsyncExitStack | None = None

    def start(self) -> None:
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        
        if self.settings.openai_api_key:
            set_default_openai_key(self.settings.openai_api_key.get_secret_value())
        else:
            self.startup_error = "OPENAI_API_KEY is not set. Add it to .env (see README)."
        
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self._connect(), self._loop).result(timeout=STARTUP_TIMEOUT_S)

    async def _connect(self) -> None:
        self._stack = AsyncExitStack()
        
        for server in build_servers(self.settings, self._recorder):
            try:
                await self._stack.enter_async_context(server)
                self._servers.append(server)
                self.server_status[server.name] = True
            except Exception as exc:
                logger.exception("MCP server %s failed to start", server.name)
                self.server_status[server.name] = False
                self.startup_warnings.append(f"MCP server '{server.name}' failed to start: {exc}")
        
        if not self._servers and not self.startup_error:
            self.startup_error = " ".join(self.startup_warnings) or "No MCP server could be started."

    def _agent(self) -> Agent:
        cdse_connected = self.server_status.get("cdse", False)
        kwargs = {"model": self.model_name} if self.model_name else {}
        return Agent(name="EO disaster monitor", instructions=build_instructions(utc_now().date(), cdse_connected, self.settings.max_pairs_per_answer), mcp_servers=list(self._servers), **kwargs)

    async def _run(self, session_id: str, text: str) -> str:
        session = SQLiteSession(session_id, str(self.settings.data_dir / "sessions.db"))
        result = await Runner.run(self._agent(), text, session=session, max_turns=MAX_AGENT_TURNS)
        return str(result.final_output or "")

    def run_turn(self, session_id: str, text: str) -> TurnResult:
        problem = validate_user_input(text)
        if problem:
            return TurnResult(answer=problem)
        
        if self.startup_error:
            return TurnResult(answer=f"The agent is not available: {self.startup_error}")
        
        with self._turn_lock:
            self._recorder.clear()
            
            for server in self._servers:
                if isinstance(server, AnalysisServer):
                    server.set_shown(self._session_shown[session_id])
            
            try:
                answer = asyncio.run_coroutine_threadsafe(self._run(session_id, text), self._loop).result()
            except MaxTurnsExceeded:
                answer = "I stopped because the task needed too many tool calls. Please narrow the question (e.g. one event, a smaller area)."
            except openai.AuthenticationError:
                answer = "OpenAI rejected the API key. Check OPENAI_API_KEY in .env."
            except openai.RateLimitError as exc:
                answer = f"OpenAI rate limit or quota exceeded: {exc}"
            except openai.APIConnectionError as exc:
                answer = f"Could not reach the OpenAI API: {exc}"
            except openai.APIError as exc:
                answer = f"The OpenAI API returned an error: {exc}"
            except Exception as exc:
                logger.exception("agent turn failed")
                answer = f"The agent failed with an unexpected error: {type(exc).__name__}: {exc}"
            artifacts = collect(list(self._recorder))

        self._session_shown[session_id].extend(shown_from_pairs(artifacts.pairs))
        area_values = self._session_area_values[session_id]
        area_values.extend(artifacts.area_values)
        report = check_area_grounding(answer, area_values)

        notes: list[str] = []
        if not report.ok:
            notes.append(UNGROUNDED_WARNING.format(values=", ".join(f"{value:g}" for value in report.ungrounded_values)))
        notes.extend(f"Tool problem: {error}" for error in artifacts.tool_errors)
        
        return TurnResult(answer=answer, artifacts=artifacts, notes=notes)

    def close(self) -> None:
        if self._stack is not None:
            try:
                asyncio.run_coroutine_threadsafe(self._stack.aclose(), self._loop).result(timeout=30)
            except Exception:
                logger.exception("error while closing MCP servers")
        
        self._loop.call_soon_threadsafe(self._loop.stop)
