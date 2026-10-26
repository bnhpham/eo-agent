"""
Deterministic, geodesically correct area of segmentation masks.

Pixel areas are measured on the WGS84 ellipsoid, never in projected units.
Geographic rasters (EPSG:4326 and similar) get an exact geodesic area per row, because all pixels in a row share it.
Projected rasters (e.g. UTM) get the nominal pixel area divided by the projection's areal scale factor.
The factor is evaluated per block of pixels because it varies slowly across an AOI.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS, Geod, Proj, Transformer

GEOD = Geod(ellps="WGS84")
SCALE_BLOCK_PX = 32


@dataclass(frozen=True)
class AreaStats:
    area_km2: float
    area_km2_low: float
    area_km2_high: float
    pixel_count: int
    boundary_pixel_count: int
    analysed_km2: float
    obscured_km2: float
    crs: str
    pixel_size_m: tuple[float, float] | None
    method: str


def pixel_area_grid_m2(shape: tuple[int, int], transform: Affine, crs: CRS) -> np.ndarray:
    """Ellipsoidal area in m² of every pixel of a north-up raster."""
    if transform.b != 0 or transform.d != 0:
        raise ValueError("rotated rasters are not supported")
    height, width = shape
    if crs.is_geographic:
        return _geographic_pixel_areas(height, width, transform, crs)
    if crs.is_projected:
        return _projected_pixel_areas(height, width, transform, crs)
    raise ValueError(f"unsupported CRS {crs.to_string()}")


def _geographic_pixel_areas(height: int, width: int, transform: Affine, crs: CRS) -> np.ndarray:
    to_wgs84 = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
    row_areas = np.empty(height, dtype=np.float64)
    for row in range(height):
        x0, y0 = transform @ (0, row)
        x1, y1 = transform @ (1, row + 1)
        lons, lats = to_wgs84.transform([x0, x1, x1, x0], [y0, y0, y1, y1])
        area, _ = GEOD.polygon_area_perimeter(lons, lats)
        row_areas[row] = abs(area)
    return np.repeat(row_areas[:, None], width, axis=1)


def _projected_pixel_areas(height: int, width: int, transform: Affine, crs: CRS) -> np.ndarray:
    unit_factor = crs.axis_info[0].unit_conversion_factor
    nominal_area = abs(transform.a * transform.e) * unit_factor**2

    block_rows = np.arange(0, height, SCALE_BLOCK_PX) + min(SCALE_BLOCK_PX, height) / 2
    block_cols = np.arange(0, width, SCALE_BLOCK_PX) + min(SCALE_BLOCK_PX, width) / 2
    cols, rows = np.meshgrid(block_cols, block_rows)
    xs, ys = transform @ (cols, rows)
    to_wgs84 = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
    lons, lats = to_wgs84.transform(xs, ys)
    factors = Proj(crs).get_factors(lons, lats)
    block_areas = nominal_area / np.asarray(factors.areal_scale, dtype=np.float64)

    expanded = np.repeat(np.repeat(block_areas, SCALE_BLOCK_PX, axis=0), SCALE_BLOCK_PX, axis=1)
    return expanded[:height, :width]


def boundary_mask(affected: np.ndarray) -> np.ndarray:
    """Affected pixels with at least one 4-connected neighbour that is not affected."""
    padded = np.pad(affected, 1, mode="constant", constant_values=False)
    interior = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    return affected & ~interior


def compute_mask_area(mask: np.ndarray, transform: Affine, crs: CRS, affected_classes: list[int], invalid_classes: list[int]) -> AreaStats:
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    pixel_areas = pixel_area_grid_m2(mask.shape, transform, crs)

    affected = np.isin(mask, affected_classes)
    invalid = np.isin(mask, invalid_classes)
    boundary = boundary_mask(affected)

    area_m2 = float(pixel_areas[affected].sum())
    boundary_m2 = float(pixel_areas[boundary].sum())
    pixel_size = None
    if crs.is_projected:
        unit_factor = crs.axis_info[0].unit_conversion_factor
        pixel_size = (abs(transform.a) * unit_factor, abs(transform.e) * unit_factor)

    method = ("Geodesic per-row pixel area on the WGS84 ellipsoid (pyproj.Geod)" if crs.is_geographic
              else "Nominal pixel area corrected by the projection areal scale factor (pyproj), i.e. area on the WGS84 ellipsoid")
    return AreaStats(area_km2=area_m2 / 1e6,
                     area_km2_low=max(area_m2 - boundary_m2 / 2, 0.0) / 1e6,
                     area_km2_high=(area_m2 + boundary_m2 / 2) / 1e6,
                     pixel_count=int(affected.sum()),
                     boundary_pixel_count=int(boundary.sum()),
                     analysed_km2=float(pixel_areas[~invalid].sum()) / 1e6,
                     obscured_km2=float(pixel_areas[invalid].sum()) / 1e6,
                     crs=crs.to_string(),
                     pixel_size_m=pixel_size,
                     method=method)


def class_areas_km2(mask: np.ndarray, transform: Affine, crs: CRS, classes: list[int]) -> dict[int, float]:
    """Geodesic area in km² of each given class of a mask."""
    pixel_areas = pixel_area_grid_m2(mask.shape, transform, crs)
    return {value: float(pixel_areas[mask == value].sum()) / 1e6 for value in classes}


def area_from_geotiff(path: Path, affected_classes: list[int], invalid_classes: list[int]) -> AreaStats:
    with rasterio.open(path) as dataset:
        if dataset.crs is None:
            raise ValueError(f"{path.name} has no CRS")
        mask = dataset.read(1)
        return compute_mask_area(mask, dataset.transform, CRS.from_user_input(dataset.crs), affected_classes, invalid_classes)
