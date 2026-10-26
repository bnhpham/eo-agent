"""
Merge GDACS and EONET events.

EONET also ingests GDACS records, so the same event can come from both.
"""

import re
from eo_agent.events.common import haversine_km
from eo_agent.schemas import PastEvent

MAX_DAYS_APART = 3
MAX_KM_APART = 50.0


# GDACS event ids in the source links of an EONET event, e.g. .../report.aspx?eventid=1029987&eventtype=WF.
def _gdacs_ids_from_eonet(event: PastEvent) -> set[str]:
    ids = set()
    for link in event.also_in:
        if link.startswith("GDACS:"):
            match = re.search(r"eventid=(\d+)", link)
            if match:
                ids.add(match.group(1))
    return ids


def _same_event(gdacs: PastEvent, eonet: PastEvent) -> bool:
    if gdacs.disaster_type != eonet.disaster_type or abs((gdacs.start - eonet.start).days) > MAX_DAYS_APART:
        return False
    if None in (gdacs.lat, gdacs.lon, eonet.lat, eonet.lon):
        return False
    return haversine_km(gdacs.lat, gdacs.lon, eonet.lat, eonet.lon) <= MAX_KM_APART


# Returns the merged lis. GDACS records win and the number of EONET duplicates that were dropped.
def merge_events(gdacs: list[PastEvent], eonet: list[PastEvent]) -> tuple[list[PastEvent], int]:
    
    by_numeric_id = {event.event_id.removeprefix("gdacs-")[2:]: event for event in gdacs}
    kept: list[PastEvent] = []
    dropped = 0
    
    for event in eonet:
        linked = _gdacs_ids_from_eonet(event)
        match = next((by_numeric_id[i] for i in linked if i in by_numeric_id), None)
        if match is None:
            match = next((g for g in gdacs if _same_event(g, event)), None)
        if match is None:
            kept.append(event)
        else:
            match.also_in.append(event.event_id)
            dropped += 1

    return [*gdacs, *kept], dropped
