"""
NASA EONET v3 events (https://eonet.gsfc.nasa.gov/docs/v3).

No API key. Curated from many sources, coverage is uneven and most events before 2023 have no size.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import date

import httpx

from eo_agent.events.common import EONET_CATEGORY, EONET_TYPE, HECTARES_PER_ACRE, SourceError, to_date
from eo_agent.geo.regions import continent_from_lat_lon
from eo_agent.schemas import SENTINEL2_FIRST_DATE, BBox, PastEvent

EONET_EVENTS_URL = "https://eonet.gsfc.nasa.gov/api/v3/events"
YEAR_WORKERS = 4


def _ring_bounds(rings: list) -> tuple[float, float, float, float] | None:
    
    points = [point for ring in rings for point in ring if len(point) >= 2]
    if not points:
        return None
    lons, lats = [p[0] for p in points], [p[1] for p in points]
    
    return min(lons), min(lats), max(lons), max(lats)


# Magnitude of one geometry with acres converted to hectares.
def _size(geometry: dict) -> tuple[float, str] | None:
    
    value, unit = geometry.get("magnitudeValue"), geometry.get("magnitudeUnit")
    if value is None or unit is None:
        return None
    
    if unit == "acres":
        return float(value) * HECTARES_PER_ACRE, "ha"
    if unit == "hectare":
        return float(value), "ha"
    
    return float(value), str(unit)


def parse_eonet_event(event: dict) -> PastEvent | None:
    
    categories = [category.get("id") for category in event.get("categories", [])]
    disaster_type = next((EONET_TYPE[category] for category in categories if category in EONET_TYPE), None)
    geometries = event.get("geometry") or []
    dates = sorted(d for d in (to_date(g.get("date")) for g in geometries) if d is not None)
    
    if disaster_type is None or not dates:
        return None

    bbox, lat, lon = None, None, None
    polygons = [(_ring_bounds(g["coordinates"]), g) for g in geometries if g.get("type") == "Polygon" and g.get("coordinates")]
    polygons = [(bounds, g) for bounds, g in polygons if bounds is not None]
    if polygons:
        bounds = max((b for b, _ in polygons), key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        bbox = [round(v, 5) for v in bounds]
        lon, lat = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
    else:
        point = next((g for g in geometries if g.get("type") == "Point" and len(g.get("coordinates") or []) >= 2), None)
        if point is not None:
            lon, lat = point["coordinates"][0], point["coordinates"][1]

    sizes = [size for size in (_size(g) for g in geometries) if size is not None]
    severity = max(sizes, key=lambda item: item[0]) if sizes else None
    sources = event.get("sources") or []
    continent = continent_from_lat_lon(lat, lon) if lat is not None else None

    return PastEvent(event_id=f"eonet-{event.get('id')}",
                     source="eonet", 
                     also_in=[f"{source.get('id')}:{source.get('url')}" for source in sources if source.get("id") and source.get("url")],
                     disaster_type=disaster_type, name=str(event.get("title") or "")[:200],
                     continents=[continent] if continent else [],
                     lat=lat,
                     lon=lon,
                     bbox=bbox,
                     start=dates[0],
                     end=to_date(event.get("closed")) or dates[-1],
                     severity_value=severity[0] if severity else None, severity_unit=severity[1] if severity else None,
                     url=event.get("link"),
                     imagery_possible=dates[0] >= SENTINEL2_FIRST_DATE)


def _fetch_year(client: httpx.Client, start: date, end: date, categories: str, bbox: BBox | None) -> list[PastEvent]:
    
    params = {"status": "all", "category": categories, "start": start.isoformat(), "end": end.isoformat(), "limit": 20000}
    if bbox is not None:
        params["bbox"] = f"{bbox.west},{bbox.north},{bbox.east},{bbox.south}"  # EONET order: west, north, east, south
    
    try:
        response = client.get(EONET_EVENTS_URL, params=params)
        response.raise_for_status()
        events = response.json().get("events", [])
    except (httpx.HTTPError, ValueError) as exc:
        raise SourceError(f"EONET request failed: {exc}") from exc
    
    return [event for event in (parse_eonet_event(raw) for raw in events) if event is not None]


# EONET events in the period. One request per calendar year keeps each response small, the years are fetched in parallel.
def fetch_eonet(client: httpx.Client, start: date, end: date, disaster_type: str | None = None, bbox: BBox | None = None) -> list[PastEvent]:
    
    categories = EONET_CATEGORY.get(disaster_type) if disaster_type else ",".join(EONET_CATEGORY.values())
    if categories is None:
        return []
    
    ranges = [(max(start, date(year, 1, 1)), min(end, date(year, 12, 31))) for year in range(start.year, end.year + 1)]
    with ThreadPoolExecutor(max_workers=YEAR_WORKERS) as pool:
        batches = list(pool.map(lambda span: _fetch_year(client, span[0], span[1], categories, bbox), ranges))
    
    seen: dict[str, PastEvent] = {}
    for batch in batches:
        for event in batch:
            seen.setdefault(event.event_id, event)  # an event spanning New Year is returned for both years
    
    return list(seen.values())
