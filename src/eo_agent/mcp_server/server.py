"""
Custom FastMCP server with disaster analysis tools: imagery selection, flood and burn-scar segmentation, area calculation.

Run over stdio: python -m eo_agent.mcp_server.server
All tools return JSON-serialisable Pydantic envelopes {ok, result, error}. Rasters stay on disk, only paths and metadata are returned.
"""

import logging
import sys
from collections.abc import Callable
from datetime import date
from typing import Annotated

import httpx
from fastmcp import FastMCP
from pydantic import Field, ValidationError

from eo_agent import analysis
from eo_agent.config import Settings, get_settings
from eo_agent.imagery.base import ImageryProvider, ProviderError
from eo_agent.imagery.selection import NoImageryError, load_pair, select_before_after
from eo_agent.events.search import search_past_events
from eo_agent.indices import METHODS, Thresholds, wrong_type_hint
from eo_agent.schemas import (ANALYSIS_DISASTER_TYPES, IMAGERY_DISASTER_TYPES, AreaResponse, BBox, ImageryRequest, ImageryResponse, PastEventsRequest, PastEventsResponse, SegmentationResponse, ToolError,
                              normalize_disaster_type, resolve_data_path)

logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("eo-analysis")

INSTRUCTIONS = ("Disaster analysis on Sentinel-2 imagery with spectral indices. Workflow: search_past_disasters (history), select_before_after_imagery (the main index layers of the disaster type come with the images), "
                "then the estimate_* tool that fits the question (estimate_burned_area for wildfire damage, estimate_water_loss for drought, estimate_snow_cover_change, estimate_bare_soil_change, "
                "estimate_vegetation_loss), then calculate_affected_area on the returned mask_path. Bounding boxes are WGS84 [west, south, east, north]. Dates are YYYY-MM-DD.")


def default_providers(settings: Settings) -> list[ImageryProvider]:
    from eo_agent.imagery.earthsearch import EarthSearchProvider

    providers: list[ImageryProvider] = []
    if settings.has_cdse_credentials:
        from eo_agent.imagery.cdse import CdseProvider

        providers.append(CdseProvider(settings.cdse_client_id, settings.cdse_client_secret.get_secret_value()))
    
    providers.append(EarthSearchProvider())
    
    return providers


def default_http_client() -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(90.0), follow_redirects=True, headers={"User-Agent": "eo-agent/0.1 (disaster monitoring prototype)"})


def _error(code: str, message: str, retryable: bool = False) -> ToolError:
    logger.warning("tool error %s: %s", code, message)
    return ToolError(code=code, message=message, retryable=retryable)


def _validation_message(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(part) for part in error['loc']) or 'input'}: {error['msg']}" for error in exc.errors())


def build_server(settings: Settings | None = None, providers_factory: Callable[[Settings], list[ImageryProvider]] = default_providers,
                 http_client_factory: Callable[[], httpx.Client] = default_http_client) -> FastMCP:
    
    settings = settings or get_settings()
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    thresholds = Thresholds.from_settings(settings)
    mcp = FastMCP("eo-analysis", instructions=INSTRUCTIONS)

    @mcp.tool
    def search_past_disasters(start_date: Annotated[date, Field(description="Start of the period, YYYY-MM-DD")],
                              end_date: Annotated[date, Field(description="End of the period, YYYY-MM-DD")],
                              disaster_type: Annotated[str | None, Field(description="flood, wildfire, earthquake, cyclone, volcano, drought or tsunami. Omit for all types")] = None,
                              country: Annotated[str | None, Field(description="Country name, only used for the GDACS archive (EONET has no country field, use bbox for it)")] = None,
                              bbox: Annotated[list[float] | None, Field(description="Search area [west, south, east, north] in WGS84 degrees, e.g. from geocode_place")] = None,
                              min_alert: Annotated[str | None, Field(description="Minimum GDACS alert level: Green, Orange or Red. EONET events have no alert level and are then left out")] = None,
                              rank_by: Annotated[str, Field(description="'severity' (size of the event, e.g. burned area in ha, only comparable within one disaster type) or 'date' (newest first)")] = "severity",
                              limit: Annotated[int, Field(ge=1, le=100, description="Maximum number of events returned")] = 20) -> PastEventsResponse:
        """
        Search historical disasters in a date range in the GDACS event archive and NASA EONET, ranked by size or date.

        Use this for any period in the past (e.g. "the last 10 years", "in 2018") and for rankings.
        Use search_gdacs_events for current and recent events.

        The result states what the ranking is based on, how many events have a size and which periods have no size information (coverage_notes).
        The notes must be passed on to the user: a ranking is only valid for the periods that have size information.
        """
        # Unsupport disaster type
        if disaster_type and normalize_disaster_type(disaster_type) is None:
            return PastEventsResponse(ok=False, error=_error("unsupported_disaster_type", f"Disaster type '{disaster_type}' is not known. Supported: {', '.join(IMAGERY_DISASTER_TYPES)}."))

        # Search for event
        try:
            if bbox and len(bbox) != 4:
                raise ValueError("bbox must have exactly 4 numbers [west, south, east, north]")
            request = PastEventsRequest(start_date=start_date, end_date=end_date, disaster_type=disaster_type, country=country,
                                        min_alert=min_alert, rank_by=rank_by, limit=limit,
                                        bbox=BBox(west=bbox[0], south=bbox[1], east=bbox[2], north=bbox[3]) if bbox else None)
        except ValidationError as exc:
            return PastEventsResponse(ok=False, error=_error("invalid_input", _validation_message(exc)))
        except ValueError as exc:
            return PastEventsResponse(ok=False, error=_error("invalid_input", str(exc)))

        # Return response
        try:
            with http_client_factory() as client:
                return PastEventsResponse(ok=True, result=search_past_events(request, client))
        except Exception as exc:
            logger.exception("historical search failed")
            return PastEventsResponse(ok=False, error=_error("internal_error", f"{type(exc).__name__}: {exc}"))

    @mcp.tool
    def select_before_after_imagery(disaster_type: Annotated[str, Field(description="flood, wildfire, earthquake, cyclone, volcano, drought, tsunami, or for analyses without an event snow or land_degradation (GDACS codes such as EQ or FL are also accepted)")],
                                    event_start: Annotated[date, Field(description="Event start date (e.g. GDACS fromdate), YYYY-MM-DD")],
                                    event_end: Annotated[date | None, Field(description="Event end date if known, YYYY-MM-DD")] = None,
                                    bbox: Annotated[list[float] | None, Field(description="AOI [west, south, east, north] in WGS84 degrees")] = None,
                                    lat: Annotated[float | None, Field(description="AOI centre latitude, used when bbox is not given")] = None,
                                    lon: Annotated[float | None, Field(description="AOI centre longitude, used when bbox is not given")] = None,
                                    radius_km: Annotated[float, Field(description="Half-width of the AOI square around lat/lon in km")] = 5.0,
                                    pre_days: Annotated[int, Field(description="Days before event_start to search for a pre-event scene")] = 45,
                                    post_days: Annotated[int, Field(description="Days after the event to search for a post-event scene")] = 30,
                                    max_cloud_pct: Annotated[float, Field(description="Maximum granule cloud cover in percent")] = 40.0,
                                    max_pairs: Annotated[int, Field(ge=1, description="Maximum number of before/after pairs (1 to 10) to return for this area: one pre-event scene and up to this many clear post-event dates")] = 1,
                                    event_name: Annotated[str | None, Field(description="Event name for display, e.g. the GDACS event name")] = None,
                                    baseline_date: Annotated[date | None, Field(description="Date of the BEFORE image for comparisons over a longer time (snow, land degradation, drought). "
                                                                                                       "Then event_start is the date of the AFTER image. Use the same season for both dates")] = None,
                                    baseline_window_days: Annotated[int, Field(ge=1, le=60, description="Days around baseline_date in which the before scene is searched")] = 20,
                                    exclude_post_dates: Annotated[list[date] | None, Field(description="Post-event acquisition dates to skip, e.g. dates of images that were already shown")] = None,
                                    repeat_shown: Annotated[bool, Field(description="Set true only when the user explicitly asks to see already shown images again. Handled by the agent layer, which otherwise skips already shown dates automatically")] = False) -> ImageryResponse:
        """
        Select and download the best pre-event scene and up to max_pairs clear post-event scenes (on different dates) for an area.

        Scenes are scored on AOI coverage, AOI cloud fraction and date proximity. 
        Fewer pairs than requested are returned when fewer clear scenes exist.
        
        Returns one entry per pair with a pair_id, acquisition dates, cloud fractions, file paths of analysis GeoTIFFs and true-colour PNG previews, and warnings.
        Works for every disaster type. Pass the pair_id to the estimate_* tool that fits the type, see the tool descriptions.
        Floods and the other event types are images only.
        """
        # Unsupport disaster type
        if normalize_disaster_type(disaster_type) is None:
            return ImageryResponse(ok=False, error=_error("unsupported_disaster_type", f"Before/after imagery supports {', '.join(IMAGERY_DISASTER_TYPES)}, not '{disaster_type}'."))

        # Formulate request for image providers
        try:
            if bbox and len(bbox) != 4:
                raise ValueError("bbox must have exactly 4 numbers [west, south, east, north]")
            box = BBox(west=bbox[0], south=bbox[1], east=bbox[2], north=bbox[3]) if bbox else None
            request = ImageryRequest(disaster_type=disaster_type, event_start=event_start, event_end=event_end,
                                     bbox=box, lat=lat, lon=lon, radius_km=radius_km, pre_days=pre_days, post_days=post_days,
                                     max_cloud_pct=max_cloud_pct, max_pairs=min(max_pairs, settings.max_pairs_per_answer),
                                     event_name=event_name, exclude_post_dates=exclude_post_dates or [],
                                     baseline_date=baseline_date, baseline_window_days=baseline_window_days)
        except ValidationError as exc:
            return ImageryResponse(ok=False, error=_error("invalid_input", _validation_message(exc)))
        except (ValueError, IndexError) as exc:
            return ImageryResponse(ok=False, error=_error("invalid_input", str(exc)))

        # Search for image pair
        provider_failures: list[str] = []
        for provider in providers_factory(settings):
            try:
                pairs = select_before_after(provider, request, settings.artifacts_dir, settings.max_aoi_km, thresholds=thresholds)
            except ProviderError as exc:
                provider_failures.append(f"{provider.name}: {exc}")
                continue
            except NoImageryError as exc:
                return ImageryResponse(ok=False, error=_error("no_imagery", f"{provider.name}: {exc}"))
            except ValueError as exc:
                return ImageryResponse(ok=False, error=_error("invalid_input", str(exc)))
            except Exception as exc:
                logger.exception("imagery selection failed")
                return ImageryResponse(ok=False, error=_error("internal_error", f"{type(exc).__name__}: {exc}"))
            if provider_failures:
                for pair in pairs:
                    pair.warnings.append(f"Fell back to {provider.name} because: {' | '.join(provider_failures)}")
            return ImageryResponse(ok=True, pairs=pairs)
        
        return ImageryResponse(ok=False, error=_error("provider_error", " | ".join(provider_failures), retryable=True))

    # analysis/estimation tools
    def register_estimation_tool(method_key: str) -> None:
        method = METHODS[method_key]

        def estimate(pair_id: Annotated[str, Field(description=f"pair_id returned by select_before_after_imagery for a pair of type: {', '.join(method.types)}")]) -> SegmentationResponse:
            try:
                pair = load_pair(settings.artifacts_dir, pair_id)
            except (FileNotFoundError, ValueError) as exc:
                return SegmentationResponse(ok=False, error=_error("not_found", str(exc)))
            
            if pair.disaster_type not in method.types:
                return SegmentationResponse(ok=False, error=_error("unsupported_disaster_type", wrong_type_hint(pair.disaster_type, method)))

            # Semantic segmentation
            try:
                return SegmentationResponse(ok=True, result=analysis.run_index_analysis(pair, method_key, thresholds))
            except ValueError as exc:
                return SegmentationResponse(ok=False, error=_error("invalid_input", str(exc)))
            except Exception as exc:
                logger.exception("%s failed", method.tool_name)
                return SegmentationResponse(ok=False, error=_error("inference_error", f"{type(exc).__name__}: {exc}"))

        mcp.tool(estimate, name=method.tool_name, description=method.description)

    for method_key in METHODS:
        register_estimation_tool(method_key)

    @mcp.tool
    def calculate_affected_area(mask_path: Annotated[str, Field(description="mask_path returned by one of the estimate_* tools")],
                                affected_classes: Annotated[list[int] | None, Field(description="Mask classes to count. Leave empty: the classes of the method that made the mask are used "
                                                                                             "(e.g. burned = low to high severity for NBR, vegetation loss for NDVI, water loss for AWEI)")] = None) -> AreaResponse:
        """
        Deterministically compute the area in km^2 of the affected classes of a change mask with geodesic (WGS84 ellipsoid) pixel areas.

        Returns the area, an uncertainty range from boundary pixels, the analysed and excluded (cloud, shadow, no data) area,
        the area of every class (e.g. burn severity classes), a GeoJSON of the polygons, and the label (measure) and disclaimer of the method.
        
        The result is an index-based estimate, not damage, not a forecast and not a diagnosis.
        """
        try:
            path = resolve_data_path(mask_path, settings.data_dir)
        except (FileNotFoundError, ValueError) as exc:
            return AreaResponse(ok=False, error=_error("not_found", str(exc)))

        # Compute area
        try:
            return AreaResponse(ok=True, result=analysis.calculate_area(path, affected_classes))
        except ValueError as exc:
            return AreaResponse(ok=False, error=_error("invalid_input", str(exc)))
        except Exception as exc:
            logger.exception("area calculation failed")
            return AreaResponse(ok=False, error=_error("internal_error", f"{type(exc).__name__}: {exc}"))

    return mcp


def main() -> None:
    build_server().run(transport="stdio", show_banner=False)


if __name__ == "__main__":
    main()
