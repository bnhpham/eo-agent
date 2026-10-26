"""
Historical disaster search over the GDACS archive and NASA EONET with coverage-aware ranking.

A ranking is only as good as the size information behind it. The result therefore states what the order is based on, how many events have a size,
and which years have none, so that the agent cannot present an incomplete ranking as a complete one.
"""

from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import httpx

from eo_agent.events.common import GDACS_FIRST_YEAR, SourceError
from eo_agent.events.eonet import fetch_eonet
from eo_agent.events.gdacs_archive import fetch_gdacs
from eo_agent.events.merge import merge_events
from eo_agent.schemas import ALERT_ORDER, ANALYSIS_ONLY_TYPES, BBox, PastEvent, PastEventsRequest, PastEventsResult, Provenance

GDACS_SOURCE = "GDACS - Global Disaster Alert and Coordination System, event archive (gdacs.org)"
SPARSE_SIZE_SHARE = 0.25  # a year counts as "mostly without size" when fewer than a quarter of its events have one
EONET_SOURCE = "NASA EONET v3 - Earth Observatory Natural Event Tracker (eonet.gsfc.nasa.gov)"


def _attempt(label: str, notes: list[str], call: Callable[[], list[PastEvent]]) -> list[PastEvent]:
    try:
        return call()
    except SourceError as exc:
        notes.append(f"{label} could not be queried ({exc}). Results come from the other source only.")
        return []


def _inside(event: PastEvent, bbox: BBox) -> bool:
    return event.lat is not None and event.lon is not None and bbox.west <= event.lon <= bbox.east and bbox.south <= event.lat <= bbox.north


def _year_ranges(years: list[int]) -> str:
    """
    [2016, 2017, 2018, 2020] -> '2016-2018, 2020'.
    """

    parts, index = [], 0
    while index < len(years):
        end = index
        while end + 1 < len(years) and years[end + 1] == years[end] + 1:
            end += 1
        parts.append(str(years[index]) if end == index else f"{years[index]}-{years[end]}")
        index = end + 1
    return ", ".join(parts)


def _severity_description(disaster_type: str, unit: str) -> str:
    if disaster_type == "wildfire" and unit == "ha":
        return "burned area in hectares"
    return f"severity value as reported by the source, unit {unit}" if unit else "severity value as reported by the source (the source gives no unit)"


# Order the events and say what the order is based on. Sizes are only comparable within one disaster type and unit.
def rank_events(events: list[PastEvent], request: PastEventsRequest) -> tuple[list[PastEvent], str | None]:
    
    newest = sorted(events, key=lambda event: event.start, reverse=True)
    if request.rank_by == "date":
        return newest, "date, newest first"
    
    by_alert = sorted(newest, key=lambda event: ALERT_ORDER.get(event.alert_level or "", 0), reverse=True)
    if request.disaster_type is None:
        return by_alert, "GDACS alert level, then date. Sizes of different disaster types cannot be compared"
    
    units = Counter(event.severity_unit for event in events if event.severity_value is not None)
    if not units:
        return by_alert, None
    unit = units.most_common(1)[0][0]

    sized = sorted((event for event in events if event.severity_value is not None and event.severity_unit == unit), key=lambda event: event.severity_value, reverse=True)
    sized_ids = {event.event_id for event in sized}
    return [*sized, *(event for event in by_alert if event.event_id not in sized_ids)], _severity_description(request.disaster_type, unit)


def coverage_notes(events: list[PastEvent], request: PastEventsRequest, ranking_basis: str | None) -> list[str]:
    notes: list[str] = []
    types = [request.disaster_type] if request.disaster_type else sorted(GDACS_FIRST_YEAR)
    
    for disaster_type in types:
        first_year = GDACS_FIRST_YEAR.get(disaster_type)
        if first_year and request.start_date.year < first_year:
            notes.append(f"The GDACS archive has no {disaster_type} events before {first_year}.")
    
    if request.rank_by == "severity" and request.disaster_type and events:
        per_year = Counter(event.start.year for event in events)
        sized = Counter(event.start.year for event in events if event.severity_value is not None)
        sparse = sorted(year for year, count in per_year.items() if sized[year] / count < SPARSE_SIZE_SHARE)
        
        if sparse:
            events_in_sparse_years = sum(per_year[year] for year in sparse)
            with_size = sum(sized[year] for year in sparse)
            notes.append(f"Most events of {_year_ranges(sparse)} have no size (only {with_size} of {events_in_sparse_years} have one), so the ranking is incomplete for these years. "
                         "Events without a size are listed after the sized ones.")
        
        if ranking_basis is None:
            notes.append("None of the events has a size, so they are ordered by alert level and date only.")
    
    old = sum(1 for event in events if not event.imagery_possible)
    if old:
        notes.append(f"{old} of the events are before July 2015. Sentinel-2 imagery does not exist for them, only the event information is available.")
    if request.start_date.year < 2018:
        notes.append("Sentinel-2 coverage is sparse before 2018 (a single satellite until 2017, fewer L2A scenes), before/after imagery for these years may be incomplete.")
    
    return notes


def search_past_events(request: PastEventsRequest, client: httpx.Client) -> PastEventsResult:
    
    if request.disaster_type in ANALYSIS_ONLY_TYPES:
        note = (f"'{request.disaster_type}' is an analysis type, not an event type: GDACS and EONET have no events for it. "
                f"Ask the user for the area and the two dates, "
                f"then call select_before_after_imagery with disaster_type='{request.disaster_type}' and baseline_date.")
        
        return PastEventsResult(events=[],
                                total_matched=0,
                                returned=0,
                                ranking_basis=None,
                                events_with_severity=0,
                                source_counts={"gdacs": 0, "eonet": 0, "merged_duplicates": 0},
                                coverage_notes=[note],
                                period_start=request.start_date,
                                period_end=request.end_date,
                                provenance=Provenance(sources=[], coverage=request.bbox))
    
    notes: list[str] = []
    eonet_skipped = request.country is not None and request.bbox is None
    if eonet_skipped:
        notes.append("EONET has no country field, so it was not searched. Pass a bbox (e.g. from geocode_place) to include it.")
    if request.disaster_type == "tsunami":
        notes.append("EONET has no tsunami category, only the GDACS archive was searched.")
    
    with ThreadPoolExecutor(max_workers=2) as pool:
        gdacs_future = pool.submit(_attempt, "The GDACS archive", notes, lambda: fetch_gdacs(client, request.start_date, request.end_date, request.disaster_type, request.country))
        eonet_future = None if eonet_skipped or request.disaster_type == "tsunami" else pool.submit(_attempt, "EONET", notes, lambda: fetch_eonet(client, request.start_date, request.end_date, request.disaster_type, request.bbox))
        gdacs_events = gdacs_future.result()
        eonet_events = eonet_future.result() if eonet_future else []

    if request.bbox is not None:
        gdacs_events = [event for event in gdacs_events if _inside(event, request.bbox)]
        eonet_events = [event for event in eonet_events if _inside(event, request.bbox)]

    if request.min_alert is not None:
        if eonet_events:
            notes.append("EONET events have no alert level and are left out because min_alert is set.")
        eonet_events = []
        gdacs_events = [event for event in gdacs_events if event.alert_level and ALERT_ORDER[event.alert_level] >= ALERT_ORDER[request.min_alert]]

    merged, duplicates = merge_events(gdacs_events, eonet_events)
    if duplicates:
        notes.append(f"{duplicates} EONET events were duplicates of GDACS events and were merged.")
    
    ranked, basis = rank_events(merged, request)
    notes = [*coverage_notes(merged, request, basis), *notes]
    returned = ranked[: request.limit]
    if len(ranked) > len(returned):
        notes.append(f"{len(ranked)} events matched. Only the first {len(returned)} in the order above are returned.")
    
    sources = [GDACS_SOURCE] + ([] if eonet_skipped or request.disaster_type == "tsunami" else [EONET_SOURCE])
    
    return PastEventsResult(events=returned,
                            total_matched=len(ranked),
                            returned=len(returned),
                            ranking_basis=basis,
                            events_with_severity=sum(1 for event in merged if event.severity_value is not None),
                            source_counts={"gdacs": sum(1 for e in merged if e.source == "gdacs"),
                                           "eonet": sum(1 for e in merged if e.source == "eonet"),
                                           "merged_duplicates": duplicates},
                            coverage_notes=notes,
                            period_start=request.start_date,
                            period_end=request.end_date,
                            provenance=Provenance(sources=sources,
                                                  coverage=request.bbox,
                                                  uncertainty_notes=["Both archives are incomplete: GDACS lists events of alert level Green to Red, EONET is curated from many sources with uneven coverage."]))
