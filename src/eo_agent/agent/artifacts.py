"""
Structured tool results collected during a turn for the UI (map layers, imagery panels, metrics).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from eo_agent.imagery.selection import load_pair
from eo_agent.indices import METHODS
from eo_agent.schemas import AreaResponse, AreaResult, ImageryPair, ImageryResponse, PastEvent, PastEventsResponse, SegmentationResponse, SegmentationResult


@dataclass
class ToolRecord:
    server: str
    tool: str
    arguments: dict
    data: Any
    is_error: bool = False


# The estimate_* tools of the index methods and the two ML tools.
ESTIMATE_TOOLS = {method.tool_name for method in METHODS.values()} | {"run_flood_segmentation", "run_burn_scar_segmentation"}


@dataclass
class TurnArtifacts:
    events: list[dict] = field(default_factory=list)
    pairs: list[ImageryPair] = field(default_factory=list)
    segmentations: list[SegmentationResult] = field(default_factory=list)
    areas: list[AreaResult] = field(default_factory=list)
    quicklooks: list[dict] = field(default_factory=list)
    tool_errors: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    coverage_notes: list[str] = field(default_factory=list)

    # Every km^2 figure the tools returned in this answer. The guardrail accepts only these numbers in the model's answer
    @property
    def area_values(self) -> list[float]:
        values = []
        for area in self.areas:
            values.extend([area.area_km2, area.area_km2_low, area.area_km2_high, area.analysed_km2, area.obscured_km2, area.analysed_km2 + area.obscured_km2])
            values.extend(area.class_areas_km2.values())
        layers = [layer for segmentation in self.segmentations for layer in (segmentation.pre_layer, segmentation.post_layer) if layer is not None]
        layers.extend(layer for pair in self.pairs for scene in (pair.pre, pair.post) if scene is not None for layer in scene.layers.values())
        for layer in layers:
            values.extend([layer.feature_km2, layer.assessable_km2, layer.not_assessable_km2 or 0.0, layer.assessable_km2 - layer.feature_km2, layer.assessable_km2 + (layer.not_assessable_km2 or 0.0)])
        for segmentation in self.segmentations:
            values.extend(value for key, value in segmentation.metrics.items() if key.endswith("_km2"))
        return values


def past_event_to_map_dict(event: PastEvent) -> dict:
    
    # Same shape as the compacted GDACS events, so that the map markers and popups work for both
    size = f"{event.severity_value:,.0f} {event.severity_unit}" if event.severity_value is not None and event.severity_unit else event.severity_text
    
    return {"name": event.name,
            "eventtype_name": event.disaster_type,
            "alertlevel": event.alert_level,
            "countries": event.countries,
            "continents": event.continents,
            "fromdate": event.start.isoformat(),
            "todate": event.end.isoformat() if event.end else None,
            "report_url": event.url,
            "lat": event.lat,
            "lon": event.lon,
            "source": event.source,
            "severity": size,
            "imagery_possible": event.imagery_possible}


def _parse_json(data: Any) -> Any:
    if isinstance(data, str):
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return data
    return data


def _add(response: ImageryResponse | SegmentationResponse | AreaResponse, target: list, artifacts: TurnArtifacts, tool: str) -> None:
    if isinstance(response, ImageryResponse) and response.ok:
        target.extend(response.pairs)
    elif not isinstance(response, ImageryResponse) and response.ok and response.result is not None:
        target.append(response.result)
    else:
        artifacts.tool_errors.append(f"{tool}: {response.error.message if response.error else 'unknown error'}")


# Sort the result of the agent (found events that the user requested, before/after images, semantic segmentation etc.)
def collect(records: list[ToolRecord]) -> TurnArtifacts:
    artifacts = TurnArtifacts()

    for record in records:
        artifacts.tool_calls.append(f"{record.server}.{record.tool}")
        data = _parse_json(record.data)
        
        if record.is_error:
            artifacts.tool_errors.append(f"{record.tool}: {str(data)[:300]}")
            continue
        
        try:
            # Current/Recent natural disaster (informed by GDACS)
            if record.tool == "search_gdacs_events" and isinstance(data, dict):
                artifacts.events.extend(data.get("events", []))
                if data.get("error"):
                    artifacts.tool_errors.append(f"{record.tool}: {data['error']}")
            # Natural disasters from the past
            elif record.tool == "search_past_disasters":
                response = PastEventsResponse.model_validate(data)
                if response.ok and response.result is not None:
                    artifacts.events.extend(past_event_to_map_dict(event) for event in response.result.events)
                    artifacts.coverage_notes.extend(response.result.coverage_notes)
                else:
                    artifacts.tool_errors.append(f"{record.tool}: {response.error.message if response.error else 'unknown error'}")
            # Quicklook
            elif record.tool.startswith("fetch_") and isinstance(data, dict):
                artifacts.quicklooks.append(data)
            # Before/After RS images
            elif record.tool == "select_before_after_imagery":
                _add(ImageryResponse.model_validate(data), artifacts.pairs, artifacts, record.tool)
            # Semantic segmentation
            elif record.tool in ESTIMATE_TOOLS:
                _add(SegmentationResponse.model_validate(data), artifacts.segmentations, artifacts, record.tool)
            # Evaluation of semantic segmentation (e.g. to answer "How much did the water shrink?")
            elif record.tool == "calculate_affected_area":
                _add(AreaResponse.model_validate(data), artifacts.areas, artifacts, record.tool)

        except ValidationError as exc:
            artifacts.tool_errors.append(f"{record.tool}: unexpected output format ({exc.error_count()} validation errors)")
    return artifacts


@dataclass
class MethodResult:
    segmentation: SegmentationResult
    area: AreaResult | None = None


@dataclass
class PairCard:
    pair: ImageryPair
    results: list[MethodResult] = field(default_factory=list)

    @property
    def segmentation(self) -> SegmentationResult | None:
        return self.results[0].segmentation if self.results else None

    @property
    def area(self) -> AreaResult | None:
        return self.results[0].area if self.results else None


@dataclass
class EventGroup:
    group_id: str
    title: str
    cards: list[PairCard]


def _same_file(first: str, second: str) -> bool:
    return Path(first).resolve() == Path(second).resolve()


# Segmentation result stored next to a mask on disk, for areas computed on a mask from an earlier message
def _segmentation_for_area(area: AreaResult) -> SegmentationResult | None:
    
    for path in Path(area.mask_path).parent.glob("segmentation_*.json"):
        try:
            segmentation = SegmentationResult.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue

        if _same_file(segmentation.mask_path, area.mask_path):
            return segmentation
    
    return None


# Group the pairs of one answer by event (one select call) such that streamlit UI can render it.
# Method results and areas are attached to their pair, also when the pair was created in an earlier message.
def build_groups(artifacts: TurnArtifacts, artifacts_dir: Path) -> list[EventGroup]:

    # Collect all segmentations
    segmentations: dict[str, SegmentationResult] = {}  # by mask file, a method run twice on the same pair counts once
    for segmentation in artifacts.segmentations:
        segmentations[str(Path(segmentation.mask_path).resolve())] = segmentation
    for area in artifacts.areas:
        segmentation = _segmentation_for_area(area)
        if segmentation is not None:
            segmentations.setdefault(str(Path(segmentation.mask_path).resolve()), segmentation)

    pairs = {pair.pair_id: pair for pair in artifacts.pairs}

    # For any segmentation whose pair isn't among them, it loads the pair from disk
    for segmentation in segmentations.values():
        if segmentation.pair_id not in pairs:
            try:
                pairs[segmentation.pair_id] = load_pair(artifacts_dir, segmentation.pair_id)
            except (FileNotFoundError, ValueError):
                continue

    # Build results per image pair
    groups: dict[str, EventGroup] = {}
    for pair in pairs.values():

        results = [MethodResult(segmentation, next((item for item in artifacts.areas if _same_file(item.mask_path, segmentation.mask_path)), None))
                   for segmentation in segmentations.values() if segmentation.pair_id == pair.pair_id]

        # Pairs sharing a group_id go into one EventGroup.
        # The group id corresponds to one select call, so pairs from the same event end up together.
        if pair.group_id not in groups:
            lat, lon = pair.bbox.center
            label = pair.event_name or pair.disaster_type.capitalize()
            groups[pair.group_id] = EventGroup(group_id=pair.group_id, title=f"{label} ({lat:.2f}, {lon:.2f}), event start {pair.event_start}", cards=[])
        
        groups[pair.group_id].cards.append(PairCard(pair=pair, results=results))
    
    for group in groups.values():
        group.cards.sort(key=lambda card: card.pair.pair_index)
    
    return list(groups.values())
