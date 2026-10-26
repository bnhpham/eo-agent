"""
GDACS event archive.

The API holds history (earthquakes from 2000, floods from 2005, wildfires from 2021), the ESA MCP tool just does not send date filters.
"""

from datetime import date

import httpx

from eo_agent.events.common import GDACS_CODE, GDACS_TYPE, SourceError, to_date
from eo_agent.geo.regions import continent_from_lat_lon, continents_for
from eo_agent.schemas import SENTINEL2_FIRST_DATE, PastEvent

GDACS_SEARCH_URL = "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH"
PAGE_SIZE = 100  # the API returns at most 100 events per call
MAX_PAGES = 40


def _number(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_gdacs_feature(feature: dict) -> PastEvent | None:
    
    properties = feature.get("properties") or {}
    disaster_type = GDACS_TYPE.get(properties.get("eventtype"))
    start = to_date(properties.get("fromdate"))

    if disaster_type is None or start is None or properties.get("eventid") is None:
        return None
    
    geometry = feature.get("geometry") or {}
    coordinates = geometry.get("coordinates") if geometry.get("type") == "Point" else None
    lon, lat = (coordinates[0], coordinates[1]) if coordinates and len(coordinates) >= 2 else (None, None)
    affected = properties.get("affectedcountries") or []
    countries = [country["countryname"] for country in affected if country.get("countryname")] or ([properties["country"]] if properties.get("country") else [])
    
    iso3 = [country["iso3"] for country in affected if country.get("iso3")] or ([properties["iso3"]] if properties.get("iso3") else [])
    continents = continents_for(iso3) or ([continent_from_lat_lon(lat, lon)] if lat is not None and continent_from_lat_lon(lat, lon) else [])
    
    severity = properties.get("severitydata") or {}
    alert = str(properties.get("alertlevel") or "").capitalize()
    bbox = feature.get("bbox")

    return PastEvent(event_id=f"gdacs-{properties['eventtype']}{properties['eventid']}",
                     source="gdacs",
                     disaster_type=disaster_type,
                     name=str(properties.get("name") or properties.get("eventname") or "")[:200],
                     countries=countries,
                     continents=continents,
                     lat=lat,
                     lon=lon,
                     bbox=[float(v) for v in bbox] if bbox and len(bbox) == 4 else None,
                     start=start,
                     end=to_date(properties.get("todate")),
                     alert_level=alert if alert in ("Green", "Orange", "Red") else None,
                     severity_value=_number(severity.get("severity")),
                     severity_unit=severity.get("severityunit") or None,
                     severity_text=(str(severity.get("severitytext"))[:200] if severity.get("severitytext") else None),
                     url=(properties.get("url") or {}).get("report"),
                     imagery_possible=start >= SENTINEL2_FIRST_DATE)


# All GDACS events in the period, newest first. Pages through the 100-events-per-call limit.
def fetch_gdacs(client: httpx.Client, start: date, end: date, disaster_type: str | None = None, country: str | None = None) -> list[PastEvent]:

    params: dict[str, str | int] = {"fromdate": start.isoformat(), "todate": end.isoformat(), "pagesize": PAGE_SIZE}
    if disaster_type:
        params["eventlist"] = GDACS_CODE[disaster_type]
    if country:
        params["country"] = country
    
    events: list[PastEvent] = []
    for page in range(1, MAX_PAGES + 1):
        
        try:
            response = client.get(GDACS_SEARCH_URL, params={**params, "pagenumber": page})
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise SourceError(f"GDACS archive request failed: {exc}") from exc
        
        if not response.content.strip():  # 204: no (more) events
            break
        
        try:
            features = response.json().get("features", [])
        except ValueError as exc:
            raise SourceError("GDACS archive returned a non-JSON response") from exc
        
        events.extend(event for event in (parse_gdacs_feature(feature) for feature in features) if event is not None)
        
        if len(features) < PAGE_SIZE:
            break
    
    return events
