"""
Index layers of single scenes: Compute a method's mask or index map, write GeoTIFF and PNG overlays, and summarise it.
"""

from pathlib import Path

import numpy as np
from pyproj import CRS

from eo_agent.geo.area import compute_mask_area
from eo_agent.geo.raster import mask_to_rgba, read_geotiff, save_display_png, write_geotiff
from eo_agent.indices import FEATURE, METHODS, IndexMethod, Thresholds, feature_mask, nbr_to_rgba, reflectance_bands, threshold_values
from eo_agent.schemas import FeatureMaskInfo, SceneMeta
from eo_agent.vegetation import MASK_NOT_ASSESSABLE

DISPLAY_MAX_SIDE = 1280
MAP_MAX_SIDE = 640
MASK_BAND_NAMES = ["CLOUD", "VALID"]  # the two flag bands that follow the spectral bands in a saved scene


# Light-weight copy of a display PNG for map overlays, so many overlays stay cheap to embed in a folium map
def map_overlay_path(png_path: Path | str) -> Path:
    path = Path(png_path)
    return path.with_name(f"{path.stem}_map{path.suffix}")


# Compute the layer of one method for one scene and write <stem>_<layer>.tif and its PNG overlays into the directory
def build_layer(method: IndexMethod, reflectance: np.ndarray, bands: list[str], valid: np.ndarray, cloud: np.ndarray,
                transform, crs, directory: Path, stem: str, thresholds: Thresholds) -> FeatureMaskInfo:

    # Name the spectral bands and compute the index of the method (e.g. NDVI) for every pixel.
    # Only pixels that are valid and not covered by cloud count as assessable.
    r = reflectance_bands(reflectance, bands)
    assessable = valid & ~cloud
    index = method.index_fn(r, thresholds)
    pyproj_crs = CRS.from_user_input(crs)
    png_path = directory / f"{stem}_{method.layer_name}.png"
    rule = method.rule_fn(thresholds)

    # Binary method: Threshold the index into a mask (feature, no feature, not assessable) and count its pixels and km²
    if method.kind == "binary":
        mask = feature_mask(method, index, r, assessable, thresholds)
        tif_path = write_geotiff(directory / f"{stem}_{method.layer_name}.tif", mask, transform, crs, nodata=MASK_NOT_ASSESSABLE, band_names=[method.feature_slug])
        overlay, categorical = mask_to_rgba(mask, method.mask_colors), True
        area = compute_mask_area(mask, transform, pyproj_crs, [FEATURE], [MASK_NOT_ASSESSABLE])
        feature_pixels, non_feature_pixels = int((mask == FEATURE).sum()), int((mask == 0).sum())

        # The fraction is relative to the assessable pixels only, so clouds do not dilute it
        usable = feature_pixels + non_feature_pixels
        stats = dict(feature_pixels=feature_pixels,
                     non_feature_pixels=non_feature_pixels,
                     not_assessable_pixels=int((mask == MASK_NOT_ASSESSABLE).sum()),
                     feature_fraction=round(feature_pixels / usable, 4) if usable else None,
                     feature_km2=round(area.area_km2, 4),
                     threshold=_main_threshold(method, thresholds))
    # Index map method (NBR): Keep the continuous index values, pixels that are not assessable become NaN
    else:
        layer = np.where(assessable & np.isfinite(index), index, np.nan).astype(np.float32)
        tif_path = write_geotiff(directory / f"{stem}_{method.layer_name}.tif", layer, transform, crs, nodata=float("nan"), band_names=[method.index_name])
        overlay, categorical = nbr_to_rgba(layer), False

        # compute_mask_area needs a mask, so finite pixels become 1 and NaN pixels become not assessable
        usable_mask = np.where(np.isfinite(layer), 1, MASK_NOT_ASSESSABLE).astype(np.uint8)
        area = compute_mask_area(usable_mask, transform, pyproj_crs, [1], [MASK_NOT_ASSESSABLE])
        stats = dict(not_assessable_pixels=int((~np.isfinite(layer)).sum()), mean_index=round(float(np.nanmean(layer)), 4) if np.isfinite(layer).any() else None, threshold=_main_threshold(method, thresholds))

    # Full-size PNG for the image panels and a smaller copy for the map. Categorical masks must not be smoothed when downsampled
    save_display_png(overlay, transform, crs, png_path, categorical=categorical, max_side=DISPLAY_MAX_SIDE)
    save_display_png(overlay, transform, crs, map_overlay_path(png_path), categorical=categorical, max_side=MAP_MAX_SIDE)

    # Summary of the layer with the file paths, the rule used and the assessable and not assessable area
    return FeatureMaskInfo(method_key=method.key, index_name=method.index_name, feature=method.feature, kind="binary" if method.kind == "binary" else "index_map", rule=rule,
                           mask_path=str(tif_path), mask_png_path=str(png_path), assessable_km2=round(area.analysed_km2, 4), not_assessable_km2=round(area.obscured_km2, 4), **stats)


def _main_threshold(method: IndexMethod, thresholds: Thresholds) -> float:
    return next(iter(threshold_values(method.key, thresholds).values()))


# The scene's layer of a method: the automatic one if it exists, otherwise computed now from the saved scene bands and written next to the scene.
def ensure_layer(scene: SceneMeta, method_key: str, thresholds: Thresholds) -> FeatureMaskInfo:

    # Reuse the layer that was created together with the scene, if its file is still on disk
    existing = scene.layers.get(method_key)
    if existing is not None and Path(existing.mask_path).exists():
        return existing
    if not scene.tif_path or not Path(scene.tif_path).exists():
        raise ValueError("the scene file is no longer on disk, select the imagery again")

    # Read the saved scene back. The spectral bands come first, followed by the CLOUD and VALID flag bands
    stack, transform, crs, names = read_geotiff(Path(scene.tif_path))
    bands = [name for name in names if name not in MASK_BAND_NAMES]
    count = len(bands)
    path = Path(scene.tif_path)

    # Build the layer from the saved bands. stack[count] is CLOUD and stack[count + 1] is VALID, the layer is written next to the scene
    return build_layer(METHODS[method_key], stack[:count], bands, stack[count + 1].astype(bool), stack[count].astype(bool), transform, crs, path.parent, path.stem, thresholds)
