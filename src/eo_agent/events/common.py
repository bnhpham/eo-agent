"""
Shared constants and helpers of the historical event sources.
"""

import math
from datetime import date, datetime

HECTARES_PER_ACRE = 0.40468564224

GDACS_CODE = {"flood": "FL", "wildfire": "WF", "earthquake": "EQ", "cyclone": "TC", "volcano": "VO", "drought": "DR", "tsunami": "TS"}
GDACS_TYPE = {code: name for name, code in GDACS_CODE.items()}

EONET_CATEGORY = {"wildfire": "wildfires", "flood": "floods", "volcano": "volcanoes", "earthquake": "earthquakes", "cyclone": "severeStorms", "drought": "drought"}
EONET_TYPE = {category: name for name, category in EONET_CATEGORY.items()}

# First year with events per type in the GDACS archive, as measured against the live API. Types missing here were only checked from 2010 on.
GDACS_FIRST_YEAR = {"wildfire": 2021, "earthquake": 2000, "flood": 2005}


class SourceError(RuntimeError):
    """
    A historical source could not be queried. The search continues with the other source.
    """


def to_date(value: object) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except ValueError:
        return None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((phi2 - phi1) / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))
