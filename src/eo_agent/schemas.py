"""
Pydantic models shared by the MCP server, the agent and the UI.
"""

import math
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Types that have a segmentation model (analysis) and types for which before/after imagery can be shown (all GDACS hazards).
DisasterType = Literal["flood", "wildfire"]
ImageryDisasterType = Literal["flood", "wildfire", "earthquake", "cyclone", "volcano", "drought", "tsunami", "snow", "land_degradation"]
# Types with at least one analysis method (see eo_agent.indices.METHODS_BY_TYPE). Floods and the other event types are imagery only for now.
ANALYSIS_DISASTER_TYPES: tuple[str, ...] = ("wildfire", "drought", "land_degradation", "snow")
# Analysis-only types: not GDACS/EONET events, the user names an area and two dates.
ANALYSIS_ONLY_TYPES: tuple[str, ...] = ("snow", "land_degradation")
IMAGERY_DISASTER_TYPES: tuple[str, ...] = ("flood", "wildfire", "earthquake", "cyclone", "volcano", "drought", "tsunami", "snow", "land_degradation")

# GDACS event codes and wording the model may use, mapped to the names above.
DISASTER_TYPE_ALIASES: dict[str, str] = {"fl": "flood", "floods": "flood", "flooding": "flood", "wf": "wildfire", "wildfires": "wildfire", "forest fire": "wildfire", "forest fires": "wildfire",
                                         "fire": "wildfire", "bushfire": "wildfire", "eq": "earthquake", "earthquakes": "earthquake", "quake": "earthquake", "tc": "cyclone",
                                         "tropical cyclone": "cyclone", "hurricane": "cyclone", "typhoon": "cyclone", "cyclones": "cyclone", "vo": "volcano", "volcanic eruption": "volcano",
                                         "volcanoes": "volcano", "volcanic": "volcano", "dr": "drought", "droughts": "drought", "ts": "tsunami", "tsunamis": "tsunami",
                                         "avalanche": "snow", "avalanches": "snow", "snow cover": "snow", "snowmelt": "snow", "snow melt": "snow", "winter flood": "snow", "winter flood potential": "snow",
                                         "desertification": "land_degradation", "land degradation": "land_degradation", "soil degradation": "land_degradation", "bare soil": "land_degradation"}
VISUAL_ONLY_NOTE = "No damage estimation is available for the disaster type '{name}' yet, so these images are for visual inspection only and no affected area can be measured."


# Canonical imagery disaster type for a name or GDACS code, or None when it is not a known type
def normalize_disaster_type(value: str) -> str | None:
    key = " ".join(str(value).strip().lower().replace("_", " ").replace("-", " ").split())
    key = DISASTER_TYPE_ALIASES.get(key, key)
    return key if key in IMAGERY_DISASTER_TYPES else None

ErrorCode = Literal["invalid_input", "unsupported_disaster_type", "no_imagery", "provider_error", "model_unavailable", "inference_error", "not_found", "budget_exceeded", "internal_error"]

AFFECTED_AREA_DISCLAIMER = ("Satellite-derived estimate of vegetation loss from an NDVI threshold. "
                            "This is not a confirmed burned area, destruction, damage to buildings or economic loss.")

EARTH_RADIUS_KM = 6371.0088


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class BBox(BaseModel):
    """
    WGS84 bounding box in decimal degrees.
    """

    model_config = ConfigDict(frozen=True)

    west: float = Field(ge=-180, le=180)
    south: float = Field(ge=-90, le=90)
    east: float = Field(ge=-180, le=180)
    north: float = Field(ge=-90, le=90)

    @model_validator(mode="after")
    def _check_order(self) -> "BBox":
        if self.west >= self.east:
            raise ValueError("west must be smaller than east (antimeridian-crossing boxes are not supported)")
        if self.south >= self.north:
            raise ValueError("south must be smaller than north")
        return self

    @classmethod
    def from_center(cls, lat: float, lon: float, radius_km: float) -> "BBox":
        if radius_km <= 0:
            raise ValueError("radius_km must be positive")
        dlat = math.degrees(radius_km / EARTH_RADIUS_KM)
        dlon = math.degrees(radius_km / (EARTH_RADIUS_KM * max(math.cos(math.radians(lat)), 1e-6)))
        return cls(west=max(lon - dlon, -180), south=max(lat - dlat, -90), east=min(lon + dlon, 180), north=min(lat + dlat, 90))

    def as_list(self) -> list[float]:
        return [self.west, self.south, self.east, self.north]

    @property
    def center(self) -> tuple[float, float]:
        """(lat, lon) of the box centre."""
        return (self.south + self.north) / 2, (self.west + self.east) / 2

    def size_km(self) -> tuple[float, float]:
        """Approximate (width, height) in km, measured through the box centre."""
        lat, _ = self.center
        height = math.radians(self.north - self.south) * EARTH_RADIUS_KM
        width = math.radians(self.east - self.west) * EARTH_RADIUS_KM * math.cos(math.radians(lat))
        return width, height


class Provenance(BaseModel):
    sources: list[str]
    retrieved_at: datetime = Field(default_factory=utc_now)
    coverage: BBox | None = None
    uncertainty_notes: list[str] = Field(default_factory=list)


class ToolError(BaseModel):
    code: ErrorCode
    message: str
    retryable: bool = False


class FeatureMaskInfo(BaseModel):
    """
    Index layer of one scene.

    Binary layers: 1 = feature present, 0 = not, 255 = not assessable (cloud, shadow, no data). 
    Index maps (NBR): float GeoTIFF, NaN = not assessable.
    """

    method_key: str
    index_name: str
    feature: str
    kind: Literal["binary", "index_map"] = "binary"
    rule: str = Field(description="How the layer was made, e.g. 'NDVI >= 0.5'")
    mask_path: str
    mask_png_path: str
    threshold: float | None = None
    feature_pixels: int = Field(default=0, ge=0)
    non_feature_pixels: int = Field(default=0, ge=0)
    not_assessable_pixels: int = Field(ge=0)
    feature_fraction: float | None = Field(default=None, ge=0, le=1, description="Share of the feature among the assessable pixels (binary layers)")
    feature_km2: float = Field(default=0.0, ge=0)
    assessable_km2: float = Field(ge=0, description="Area of the pixels that could be judged, cloud, shadow and no-data pixels excluded")
    not_assessable_km2: float | None = Field(default=None, ge=0, description="Area of the excluded pixels (cloud, shadow, no data)")
    mean_index: float | None = Field(default=None, description="Mean index value over the assessable pixels (index maps)")


class SceneMeta(BaseModel):
    scene_id: str
    provider: Literal["cdse", "earth-search"]
    collection: str
    acquired_at: datetime
    scene_cloud_cover_pct: float | None = Field(default=None, description="Cloud cover of the whole granule from the catalog")
    aoi_cloud_fraction: float | None = Field(default=None, ge=0, le=1, description="Cloud/shadow fraction inside the AOI")
    aoi_valid_fraction: float | None = Field(default=None, ge=0, le=1, description="Fraction of the AOI with valid data")
    tif_path: str | None = None
    png_path: str | None = None
    layers: dict[str, FeatureMaskInfo] = Field(default_factory=dict, description="Index layers of this scene by method key")

    @property
    def vegetation(self) -> FeatureMaskInfo | None:
        return self.layers.get("ndvi_vegetation_loss")


class ImageryPair(BaseModel):
    pair_id: str
    group_id: str = Field(description="Shared by all pairs of one select_before_after_imagery call (same AOI and pre-event scene)")
    pair_index: int = Field(ge=1, description="1-based position within the group, ordered by post-event date")
    event_name: str | None = None
    disaster_type: ImageryDisasterType
    event_start: date
    bbox: BBox
    crs: str
    resolution_m: float
    provider: Literal["cdse", "earth-search"]
    product: Literal["sentinel-2-l1c", "sentinel-2-l2a"]
    bands: list[str]
    pre: SceneMeta | None
    post: SceneMeta | None
    bounds_wgs84: BBox
    warnings: list[str] = Field(default_factory=list)
    provenance: Provenance


class SegmentationResult(BaseModel):
    pair_id: str
    disaster_type: ImageryDisasterType
    model_id: str
    model_revision: str | None
    device: str
    mask_path: str
    mask_png_path: str
    class_legend: dict[int, str]
    affected_class: int
    class_pixel_counts: dict[str, int]
    method_key: str | None = None
    measure: str | None = Field(default=None, description="What the km² of the affected classes mean, e.g. 'Estimated burned area'")
    affected_classes: list[int] = Field(default_factory=lambda: [2], description="Mask classes that calculate_affected_area counts by default")
    disclaimer: str | None = None
    pre_layer: FeatureMaskInfo | None = None
    post_layer: FeatureMaskInfo | None = None
    thresholds: dict[str, float] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict, description="Method specific summary figures, e.g. vegetation km2 before/after and net change in percent")
    input_product: str
    pre_scene_used: bool
    bounds_wgs84: BBox
    warnings: list[str] = Field(default_factory=list)
    provenance: Provenance


class AreaResult(BaseModel):
    mask_path: str
    affected_classes: list[int]
    area_km2: float = Field(ge=0)
    area_km2_low: float = Field(ge=0, description="Lower bound: affected pixels minus half of the boundary pixels")
    area_km2_high: float = Field(ge=0, description="Upper bound: affected pixels plus half of the boundary pixels")
    pixel_count: int = Field(ge=0)
    analysed_km2: float = Field(ge=0, description="Area with valid, cloud-free observations")
    obscured_km2: float = Field(ge=0, description="Area that could not be assessed (cloud, nodata)")
    crs: str
    pixel_size_m: tuple[float, float] | None
    method: str
    geojson_path: str | None
    measure: str = "Estimated vegetation loss"
    method_key: str | None = None
    class_areas_km2: dict[str, float] = Field(default_factory=dict, description="Geodesic area of every class of the mask (class label to km²)")
    disclaimer: str = AFFECTED_AREA_DISCLAIMER
    provenance: Provenance


class ImageryResponse(BaseModel):
    ok: bool
    pairs: list[ImageryPair] = Field(default_factory=list)
    error: ToolError | None = None


class SegmentationResponse(BaseModel):
    ok: bool
    result: SegmentationResult | None = None
    error: ToolError | None = None


class AreaResponse(BaseModel):
    ok: bool
    result: AreaResult | None = None
    error: ToolError | None = None


class ImageryRequest(BaseModel):
    """
    Validated input of select_before_after_imagery.
    """

    disaster_type: str
    event_start: date
    event_end: date | None = None
    bbox: BBox | None = None
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    radius_km: float = Field(default=5.0, gt=0)
    pre_days: int = Field(default=45, ge=1, le=365)
    post_days: int = Field(default=30, ge=1, le=365)
    max_cloud_pct: float = Field(default=40.0, ge=0, le=100)
    max_pairs: int = Field(default=1, ge=1, le=10)
    event_name: str | None = Field(default=None, max_length=120)
    exclude_post_dates: list[date] = Field(default_factory=list)
    baseline_date: date | None = Field(default=None, description="Search the before scenes around this date instead of before event_start")
    baseline_window_days: int = Field(default=20, ge=1, le=60)

    @field_validator("disaster_type")
    @classmethod
    def _normalize_type(cls, value: str) -> str:
        normalized = normalize_disaster_type(value)
        if normalized is None:
            raise UnsupportedDisasterTypeError(value)
        return normalized

    @model_validator(mode="after")
    def _check(self) -> "ImageryRequest":
        if self.bbox is None and (self.lat is None or self.lon is None):
            raise ValueError("provide either bbox or lat and lon")
        if self.event_end is not None and self.event_end < self.event_start:
            raise ValueError("event_end must not be before event_start")
        if self.baseline_date is not None and self.baseline_date >= self.event_start:
            raise ValueError("baseline_date must be before event_start")
        if self.event_start > utc_now().date():
            raise ValueError("event_start is in the future")
        return self

    def resolved_bbox(self) -> BBox:
        return self.bbox if self.bbox is not None else BBox.from_center(self.lat, self.lon, self.radius_km)


class UnsupportedDisasterTypeError(ValueError):
    def __init__(self, disaster_type: str):
        super().__init__(f"Disaster type '{disaster_type}' is not supported for before/after imagery. Supported: {', '.join(IMAGERY_DISASTER_TYPES)}.")
        self.disaster_type = disaster_type


# Resolve a path given by the LLM and refuse anything outside the data directory
def resolve_data_path(path: str | Path, data_dir: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = data_dir / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(data_dir.resolve()):
        raise ValueError(f"path '{path}' is outside the data directory")
    if not resolved.exists():
        raise FileNotFoundError(f"file '{path}' does not exist")
    return resolved


# Historical disaster search (GDACS archive and NASA EONET)
AlertLevel = Literal["Green", "Orange", "Red"]
ALERT_ORDER: dict[str, int] = {"Green": 1, "Orange": 2, "Red": 3}
SENTINEL2_FIRST_DATE = date(2015, 7, 1)  # Sentinel-2A launched June 2015, imagery exists from mid-2015 on


class PastEvent(BaseModel):
    """
    One disaster event from a historical source in a common shape.
    """

    event_id: str = Field(description="gdacs-<id> or eonet-<id>")
    source: Literal["gdacs", "eonet"]
    also_in: list[str] = Field(default_factory=list, description="Ids of the same event in other sources (merged duplicates)")
    disaster_type: ImageryDisasterType
    name: str
    countries: list[str] = Field(default_factory=list)
    continents: list[str] = Field(default_factory=list)
    lat: float | None = None
    lon: float | None = None
    bbox: list[float] | None = Field(default=None, description="[west, south, east, north] footprint when the source has one")
    start: date
    end: date | None = None
    alert_level: AlertLevel | None = Field(default=None, description="GDACS alert level. EONET has none")
    severity_value: float | None = Field(default=None, description="Size of the event as reported by the source, e.g. burned area in ha for wildfires")
    severity_unit: str | None = None
    severity_text: str | None = None
    url: str | None = None
    imagery_possible: bool = Field(description="False for events before July 2015, Sentinel-2 imagery does not exist for them")


class PastEventsRequest(BaseModel):
    """
    Validated input of search_past_disasters.
    """

    start_date: date
    end_date: date
    disaster_type: str | None = None
    country: str | None = Field(default=None, max_length=80)
    bbox: BBox | None = None
    min_alert: AlertLevel | None = None
    rank_by: Literal["severity", "date"] = "severity"
    limit: int = Field(default=20, ge=1, le=100)

    @field_validator("disaster_type")
    @classmethod
    def _normalize_type(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        normalized = normalize_disaster_type(value)
        if normalized is None:
            raise UnsupportedDisasterTypeError(value)
        return normalized

    @model_validator(mode="after")
    def _check(self) -> "PastEventsRequest":
        if self.end_date < self.start_date:
            raise ValueError("end_date must not be before start_date")
        if self.start_date > utc_now().date():
            raise ValueError("start_date is in the future")
        if self.end_date > utc_now().date():
            self.end_date = utc_now().date()
        return self


class PastEventsResult(BaseModel):
    events: list[PastEvent]
    total_matched: int
    returned: int
    ranking_basis: str | None = Field(default=None, description="What the order is based on, e.g. 'burned area in ha'")
    events_with_severity: int
    source_counts: dict[str, int]
    coverage_notes: list[str] = Field(default_factory=list)
    period_start: date
    period_end: date
    provenance: Provenance


class PastEventsResponse(BaseModel):
    ok: bool
    result: PastEventsResult | None = None
    error: ToolError | None = None
