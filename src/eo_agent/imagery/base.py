"""
Common types for analysis-grade Sentinel-2 imagery providers.
"""

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal, Protocol

import numpy as np
from affine import Affine
from pyproj import CRS, Transformer

from eo_agent.indices import ANALYSIS_BANDS
from eo_agent.schemas import ANALYSIS_DISASTER_TYPES, BBox

Product = Literal["sentinel-2-l1c", "sentinel-2-l2a"]

# All index methods (NBR, NDVI, AWEI, NDSI, BSI) work from the same seven bands, so one download of the 10 m grid serves every method.
# Event types without a method only need true-colour imagery for visual inspection.
VISUAL_BANDS = ["B02", "B03", "B04"]
RESOLUTION_M_ANALYSIS = 10.0


def bands_for(disaster_type: str) -> list[str]:
    return list(ANALYSIS_BANDS) if disaster_type in ANALYSIS_DISASTER_TYPES else list(VISUAL_BANDS)


def resolution_for(disaster_type: str) -> float:
    return RESOLUTION_M_ANALYSIS


# Reflectance is stored as uint16 = reflectance * 10000 with the processing-baseline offset removed. 0 is nodata.
REFLECTANCE_SCALE = 10000.0


class ProviderError(RuntimeError):
    """
    Raised when an imagery backend fails (HTTP error, auth, quota).
    """


@dataclass(frozen=True)
class TargetGrid:
    """
    North-up UTM grid shared by the pre and post scenes so that they align pixel by pixel.
    """

    crs: CRS
    transform: Affine
    width: int
    height: int

    @classmethod
    def for_bbox(cls, bbox: BBox, resolution_m: float) -> "TargetGrid":
        lat, lon = bbox.center
        zone = int(math.floor((lon + 180) / 6)) + 1
        epsg = (32600 if lat >= 0 else 32700) + zone
        crs = CRS.from_epsg(epsg)
        to_utm = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)
        xs, ys = to_utm.transform([bbox.west, bbox.east, bbox.east, bbox.west], [bbox.south, bbox.south, bbox.north, bbox.north])
        left = math.floor(min(xs) / resolution_m) * resolution_m
        right = math.ceil(max(xs) / resolution_m) * resolution_m
        bottom = math.floor(min(ys) / resolution_m) * resolution_m
        top = math.ceil(max(ys) / resolution_m) * resolution_m
        width = int(round((right - left) / resolution_m))
        height = int(round((top - bottom) / resolution_m))
        return cls(crs=crs, transform=Affine(resolution_m, 0, left, 0, -resolution_m, top), width=width, height=height)

    @property
    def resolution_m(self) -> float:
        return abs(self.transform.a)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(left, bottom, right, top) in grid CRS units."""
        left, top = self.transform.c, self.transform.f
        return left, top - self.height * self.resolution_m, left + self.width * self.resolution_m, top

    def coarsened(self, max_pixels: int = 256) -> "TargetGrid":
        """Same extent at lower resolution, used for cheap cloud screening."""
        factor = max(1, math.ceil(max(self.width, self.height) / max_pixels))
        left, _, _, top = self.bounds
        resolution = self.resolution_m * factor
        return TargetGrid(crs=self.crs, transform=Affine(resolution, 0, left, 0, -resolution, top), width=math.ceil(self.width / factor), height=math.ceil(self.height / factor))


@dataclass
class SceneCandidate:
    provider: Literal["cdse", "earth-search"]
    product: Product
    acquired_at: datetime
    scene_ids: list[str]
    cloud_cover_pct: float | None
    payload: list[Any] = field(default_factory=list, repr=False)

    @property
    def acquired_date(self) -> date:
        return self.acquired_at.date()


@dataclass
class FetchedScene:
    """
    Spectral bands (uint16 reflectance * 10000) plus boolean cloud and valid masks on a TargetGrid.
    """

    candidate: SceneCandidate
    grid: TargetGrid
    bands: list[str]
    reflectance: np.ndarray
    cloud: np.ndarray
    valid: np.ndarray

    @property
    def valid_fraction(self) -> float:
        return float(self.valid.mean())

    @property
    def cloud_fraction(self) -> float:
        """Cloud fraction among pixels that have data."""
        count = int(self.valid.sum())
        return float((self.cloud & self.valid).sum() / count) if count else 1.0


class ImageryProvider(Protocol):
    name: Literal["cdse", "earth-search"]

    def product_for(self, disaster_type: str) -> Product: ...

    def search(self, bbox: BBox, start: date, end: date, product: Product, max_cloud_pct: float) -> list[SceneCandidate]: ...

    def screen(self, candidate: SceneCandidate, grid: TargetGrid) -> tuple[float, float]:
        """(cloud_fraction, valid_fraction) of the AOI, computed cheaply on a coarse grid."""
        ...

    def fetch(self, candidate: SceneCandidate, grid: TargetGrid, bands: list[str]) -> FetchedScene: ...


def group_by_day(candidates: list[SceneCandidate]) -> list[SceneCandidate]:
    """Merge granules acquired on the same UTC day into one candidate (the AOI may span several tiles)."""
    by_day: dict[date, SceneCandidate] = {}
    for candidate in sorted(candidates, key=lambda item: item.acquired_at):
        existing = by_day.get(candidate.acquired_date)
        if existing is None:
            by_day[candidate.acquired_date] = SceneCandidate(provider=candidate.provider, product=candidate.product, acquired_at=candidate.acquired_at,
                                                             scene_ids=list(candidate.scene_ids), cloud_cover_pct=candidate.cloud_cover_pct, payload=list(candidate.payload))
            continue
        existing.scene_ids.extend(candidate.scene_ids)
        existing.payload.extend(candidate.payload)
        if candidate.cloud_cover_pct is not None:
            existing.cloud_cover_pct = candidate.cloud_cover_pct if existing.cloud_cover_pct is None else min(existing.cloud_cover_pct, candidate.cloud_cover_pct)
    return list(by_day.values())
