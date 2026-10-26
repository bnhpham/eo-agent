"""
Raster I/O and display helpers (GeoTIFF, PNG overlays, GeoJSON).
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from rasterio.crs import CRS as RioCRS
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.warp import calculate_default_transform, reproject, transform_bounds
from shapely.geometry import shape

from eo_agent.schemas import BBox

WEB_MERCATOR = RioCRS.from_epsg(3857)
WGS84 = RioCRS.from_epsg(4326)


def write_geotiff(path: Path, data: np.ndarray, transform: Affine, crs, nodata: float | None = None, band_names: list[str] | None = None) -> Path:
    if data.ndim == 2:
        data = data[None]
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {"driver": "GTiff", "height": data.shape[1], "width": data.shape[2], "count": data.shape[0], "dtype": data.dtype,
               "crs": crs, "transform": transform, "nodata": nodata, "compress": "deflate", "tiled": True, "blockxsize": 256, "blockysize": 256}
    
    if data.dtype.kind == "f":
        profile["predictor"] = 3
    elif data.dtype.itemsize > 1:
        profile["predictor"] = 2
    
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(data)
        if band_names:
            dataset.descriptions = tuple(band_names)
    
    return path


def read_geotiff(path: Path) -> tuple[np.ndarray, Affine, RioCRS, list[str]]:
    with rasterio.open(path) as dataset:
        names = [name or f"band{i + 1}" for i, name in enumerate(dataset.descriptions)]
        return dataset.read(), dataset.transform, dataset.crs, names


def wgs84_bounds(shape_hw: tuple[int, int], transform: Affine, crs) -> BBox:
    height, width = shape_hw
    left, top = transform @ (0, 0)
    right, bottom = transform @ (width, height)
    west, south, east, north = transform_bounds(crs, WGS84, min(left, right), min(top, bottom), max(left, right), max(top, bottom), densify_pts=21)
    return BBox(west=west, south=south, east=east, north=north)


def true_color(stack: np.ndarray, band_names: list[str], valid: np.ndarray | None = None) -> np.ndarray:
    """Percentile-stretched uint8 RGB image (H, W, 3) from B04/B03/B02."""
    rgb = np.stack([stack[band_names.index(name)] for name in ("B04", "B03", "B02")], axis=-1).astype(np.float32)
    if valid is None:
        valid = np.all(np.isfinite(rgb), axis=-1) & (rgb.sum(axis=-1) > 0)
    if not valid.any():
        return np.zeros(rgb.shape, dtype=np.uint8)
    low, high = np.percentile(rgb[valid], [2, 98])
    scaled = np.clip((rgb - low) / max(high - low, 1e-6), 0, 1)
    image = (scaled * 255).astype(np.uint8)
    image[~valid] = 0
    return image


def mask_to_rgba(mask: np.ndarray, colors: dict[int, tuple[int, int, int, int]]) -> np.ndarray:
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    for value, color in colors.items():
        rgba[mask == value] = color
    return rgba


def save_display_png(image: np.ndarray, transform: Affine, crs, path: Path, categorical: bool = False, max_side: int | None = None) -> BBox:
    """
    Warp an (H, W, C) uint8 image to Web Mercator and save it as PNG for a Leaflet ImageOverlay.

    Leaflet stretches overlays linearly in Web Mercator, so the image must be on that grid to line up exactly.
    Returns the WGS84 bounds to pass to folium. max_side downsamples the image first, used for the light map overlays.
    """
    height, width, channels = image.shape
    if max_side and max(height, width) > max_side:
        scale = max(height, width) / max_side
        new_width, new_height = round(width / scale), round(height / scale)
        image = np.asarray(Image.fromarray(np.ascontiguousarray(image)).resize((new_width, new_height), Image.NEAREST if categorical else Image.BILINEAR))
        transform = transform * Affine.scale(width / new_width, height / new_height)
        height, width = new_height, new_width
    if channels == 3:
        alpha = np.where(image.sum(axis=-1) > 0, 255, 0).astype(np.uint8)
        image = np.concatenate([image, alpha[..., None]], axis=-1)
    left, top = transform @ (0, 0)
    right, bottom = transform @ (width, height)
    dst_transform, dst_width, dst_height = calculate_default_transform(crs, WEB_MERCATOR, width, height, left=min(left, right), bottom=min(top, bottom), right=max(left, right), top=max(top, bottom))
    warped = np.zeros((4, dst_height, dst_width), dtype=np.uint8)
    resampling = Resampling.nearest if categorical else Resampling.bilinear
    for band in range(4):
        reproject(source=np.ascontiguousarray(image[..., band]), destination=warped[band], src_transform=transform, src_crs=crs, dst_transform=dst_transform, dst_crs=WEB_MERCATOR, resampling=resampling)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.moveaxis(warped, 0, -1), mode="RGBA").save(path, optimize=True)
    bounds = wgs84_bounds((dst_height, dst_width), dst_transform, WEB_MERCATOR)
    path.with_suffix(".bounds.json").write_text(bounds.model_dump_json(), encoding="utf-8")
    return bounds


def polygonize_mask(mask: np.ndarray, transform: Affine, crs, classes: list[int], path: Path, min_pixels: int = 4) -> Path | None:
    """Vectorise the given mask classes to a WGS84 GeoJSON. Returns None when nothing is affected."""
    selected = np.isin(mask, classes)
    if not selected.any():
        return None
    pixel_area = abs(transform.a * transform.e)
    records = [{"geometry": shape(geometry), "class": int(value)} for geometry, value in shapes(selected.astype(np.uint8), mask=selected, transform=transform, connectivity=8) if value == 1]
    frame = gpd.GeoDataFrame(records, crs=crs)
    frame = frame[frame.geometry.area >= min_pixels * pixel_area]
    if frame.empty:
        return None
    frame = frame.to_crs(epsg=4326)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_file(path, driver="GeoJSON")
    return path
