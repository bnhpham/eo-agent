"""
Analysis of a pre/post pair: NDVI vegetation loss (active) and the dormant ML segmentation pipelines.

The ML runners are kept for later but are not registered as MCP tools.

Artifacts are named after the method (mask_<method_key>.*), not the disaster type,
so another method such as an NBR burned area can write next to the NDVI result of the same pair.
"""

from pathlib import Path

import numpy as np
from pyproj import CRS

from eo_agent.geo.area import area_from_geotiff, class_areas_km2, compute_mask_area
from eo_agent.geo.raster import mask_to_rgba, polygonize_mask, read_geotiff, save_display_png, wgs84_bounds, write_geotiff
from eo_agent.imagery.base import REFLECTANCE_SCALE
from eo_agent.imagery.selection import DISPLAY_MAX_SIDE, MAP_MAX_SIDE, MASK_BAND_NAMES, map_overlay_path
from eo_agent.models.registry import ML4FLOODS_REPO, PRITHVI_REPO, ModelRegistry
from eo_agent.indices import METHODS, Thresholds, binary_change_classes, net_loss_pct, severity_classes, threshold_values, wrong_type_hint
from eo_agent.layers import ensure_layer
from eo_agent.schemas import AreaResult, ImageryPair, Provenance, SegmentationResult
from eo_agent.vegetation import MASK_NOT_ASSESSABLE, VEGETATION

NOT_ASSESSABLE, NOT_AFFECTED, AFFECTED, PRE_EXISTING, PRE_UNKNOWN = 0, 1, 2, 3, 4

FLOOD_LEGEND = {NOT_ASSESSABLE: "not assessable (cloud or no data in post-event scene)", NOT_AFFECTED: "no water detected",
                AFFECTED: "flooded (water after the event, land before)", PRE_EXISTING: "permanent or pre-existing water (water before and after)",
                PRE_UNKNOWN: "water after the event, pre-event state unknown (no clear pre-event pixel)"}
BURN_LEGEND = {NOT_ASSESSABLE: "not assessable (cloud or no data in post-event scene)", NOT_AFFECTED: "no burn scar detected",
               AFFECTED: "new burn scar (burned after the event, not before)", PRE_EXISTING: "burn scar already present before the event",
               PRE_UNKNOWN: "burn scar after the event, pre-event state unknown (no clear pre-event pixel)"}
FLOOD_COLORS = {NOT_ASSESSABLE: (128, 128, 128, 90), AFFECTED: (0, 112, 255, 190), PRE_EXISTING: (0, 0, 139, 130), PRE_UNKNOWN: (135, 206, 250, 160)}
BURN_COLORS = {NOT_ASSESSABLE: (128, 128, 128, 90), AFFECTED: (255, 69, 0, 200), PRE_EXISTING: (139, 0, 0, 130), PRE_UNKNOWN: (255, 165, 0, 160)}

ML4FLOODS_SOURCE = "ML4Floods WF2_unetv2_bgriswirs (SpaceML / ISP Universitat de València, CC BY-NC 4.0), huggingface.co/isp-uv-es/ml4floods"
PRITHVI_SOURCE = "Prithvi-EO-2.0-300M-BurnScars (IBM / NASA, Apache-2.0), huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars"


def _load_scene(tif_path: str, expected_bands: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, object, object]:
    
    # Read the saved scene and check that it has the expected bands followed by the CLOUD and VALID flag bands
    stack, transform, crs, names = read_geotiff(Path(tif_path))
    if names[: len(expected_bands)] != expected_bands or names[len(expected_bands):] != MASK_BAND_NAMES:
        raise ValueError(f"{Path(tif_path).name} has bands {names}, expected {[*expected_bands, *MASK_BAND_NAMES]}")
    
    # Split the stack into the reflectance bands and the two flag bands
    reflectance =stack[: len(expected_bands)]
    cloud = stack[len(expected_bands)].astype(bool)
    valid = stack[len(expected_bands) + 1].astype(bool)

    return reflectance, cloud, valid, transform, crs


# Combine per-date detections into the change legend. *_clear marks pixels that are valid and cloud-free
def change_classes(post_detected: np.ndarray, post_clear: np.ndarray, pre_detected: np.ndarray | None, pre_clear: np.ndarray | None) -> np.ndarray:

    # Start with every pixel as not affected. Only detections on clear post-event pixels count
    mask = np.full(post_detected.shape, NOT_AFFECTED, dtype=np.uint8)
    detected = post_detected & post_clear

    # Without a pre-event scene, new features cannot be separated from pre-existing ones
    if pre_detected is None or pre_clear is None:
        mask[detected] = PRE_UNKNOWN
    else:
        # New: detected after but not before. Pre-existing: detected on both dates. Unknown: the pre-event pixel is cloudy or has no data
        mask[detected & pre_clear & ~pre_detected] = AFFECTED
        mask[detected & pre_clear & pre_detected] = PRE_EXISTING
        mask[detected & ~pre_clear] = PRE_UNKNOWN
    
    # Pixels without a clear post-event view are not assessable, this overrides the classes above
    mask[~post_clear] = NOT_ASSESSABLE
    return mask


def _finish(pair: ImageryPair, mask: np.ndarray, transform, crs, legend: dict[int, str], colors: dict, model_id: str,
            revision: str | None, device: str, thresholds: dict[str, float], warnings: list[str], source: str, notes: list[str],
            method_key: str, metrics: dict[str, float] | None = None, extra: dict | None = None) -> SegmentationResult:
    
    # The change mask is written next to the post-event scene and named after the method
    directory = Path(pair.post.tif_path).parent
    stem = f"mask_{method_key}"
    mask_path = write_geotiff(directory / f"{stem}.tif", mask, transform, crs, nodata=None, band_names=["change_class"])
    
    # Coloured PNG for the image panels and a smaller copy for the map
    png_path = directory / f"{stem}.png"
    overlay = mask_to_rgba(mask, colors)
    save_display_png(overlay, transform, crs, png_path, categorical=True, max_side=DISPLAY_MAX_SIDE)
    save_display_png(overlay, transform, crs, map_overlay_path(png_path), categorical=True, max_side=MAP_MAX_SIDE)
    
    # Number of pixels per class, keyed by the legend text
    counts = {legend[value]: int((mask == value).sum()) for value in legend}
    
    # Warn when the result is less trustworthy: no pre-event scene or too many cloudy pixels
    if pair.pre is None:
        warnings.append("No usable pre-event scene, so pre-existing features cannot be separated from event impact.")
    clear_fraction = float((mask != NOT_ASSESSABLE).mean())
    if clear_fraction < 0.7:
        warnings.append(f"Only {clear_fraction:.0%} of the AOI could be assessed in the post-event scene (clouds or missing data).")
    
    # Collect everything in one result, including the warnings of the pair and the provenance
    result =SegmentationResult(pair_id=pair.pair_id, disaster_type=pair.disaster_type, model_id=model_id, model_revision=revision,
                               device=device, mask_path=str(mask_path), mask_png_path=str(png_path), class_legend=legend, 
                               affected_class=AFFECTED, class_pixel_counts=counts, thresholds=thresholds, metrics=metrics or {},
                               input_product=pair.product, pre_scene_used=pair.pre is not None, method_key=method_key, **(extra or {}),
                               bounds_wgs84=wgs84_bounds(mask.shape, transform, crs), warnings=[*pair.warnings, *warnings],
                               provenance=Provenance(sources=[source, *pair.provenance.sources], coverage=pair.bbox, uncertainty_notes=notes))
    
    # Stored next to the mask, so that calculate_area can later find the legend, the measure and the disclaimer of this mask
    (directory / f"segmentation_{method_key}.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result


def run_flood_segmentation(pair: ImageryPair, registry: ModelRegistry, water_threshold: float = 0.5, cloud_threshold: float = 0.5) -> SegmentationResult:
    from eo_agent.models.ml4floods import BANDS, CLASS_CLOUD, CLASS_INVALID, CLASS_WATER, MODEL_ID, classify

    # Only flood pairs with a post-event scene can be mapped
    if pair.disaster_type != "flood":
        raise ValueError(f"pair {pair.pair_id} was selected for {pair.disaster_type}, not flood")
    if pair.post is None:
        raise ValueError("the pair has no post-event scene, so flood extent cannot be mapped")
    model = registry.ml4floods()

    # Water and clear pixels of one scene. Clear means valid and neither cloud nor invalid according to the model
    def detect(tif_path: str):
        digital_numbers, _, valid, transform, crs = _load_scene(tif_path, BANDS)
        classes = classify(digital_numbers, model.predict_probabilities(digital_numbers), water_threshold, cloud_threshold)
        clear = valid & (classes != CLASS_CLOUD) & (classes != CLASS_INVALID)
        return classes == CLASS_WATER, clear, transform, crs

    # Detect water on the post-event scene and, if there is one, on the pre-event scene. Then combine both dates
    post_water, post_clear, transform, crs = detect(pair.post.tif_path)
    pre_water, pre_clear = (None, None)
    if pair.pre is not None:
        pre_water, pre_clear, _, _ = detect(pair.pre.tif_path)
    mask = change_classes(post_water, post_clear, pre_water, pre_clear)

    # Notes shown as uncertainty notes in the provenance
    notes = ["Water is detected by an ML model per scene, flooded = water after the event on pixels that were land before.",
             "Clouds are masked with the ML model cloud output (probability and brightness thresholds).",
             "Optical imagery cannot see water under clouds, dense vegetation or in narrow urban streets, so flood extent is likely underestimated."]
    
    return _finish(pair, mask, transform, crs, FLOOD_LEGEND, FLOOD_COLORS, MODEL_ID, registry.paths.revision(ML4FLOODS_REPO), str(model.device),
                   {"water_probability": water_threshold, "cloud_probability": cloud_threshold}, [], ML4FLOODS_SOURCE, notes, "ml4floods_flood_extent")


def run_burn_scar_segmentation(pair: ImageryPair, registry: ModelRegistry) -> SegmentationResult:
    from eo_agent.models.prithvi import BANDS, CLASS_BURNED, MODEL_ID

    # Only wildfire pairs with a post-event scene can be mapped
    if pair.disaster_type != "wildfire":
        raise ValueError(f"pair {pair.pair_id} was selected for {pair.disaster_type}, not wildfire")
    if pair.post is None:
        raise ValueError("the pair has no post-event scene, so burn scars cannot be mapped")
    model = registry.prithvi()

    # Burned and clear pixels of one scene. The model needs reflectance scaled to 0-1, clear means valid and not cloud
    def detect(tif_path: str):
        reflectance, cloud, valid, transform, crs = _load_scene(tif_path, BANDS)
        burned = model.predict(reflectance.astype(np.float32) / REFLECTANCE_SCALE) == CLASS_BURNED
        return burned, valid & ~cloud, transform, crs

    # Detect burn scars on the post-event scene and, if there is one, on the pre-event scene. Then combine both dates
    post_burned, post_clear, transform, crs = detect(pair.post.tif_path)
    pre_burned, pre_clear = (None, None)
    if pair.pre is not None:
        pre_burned, pre_clear, _, _ = detect(pair.pre.tif_path)
    mask = change_classes(post_burned, post_clear, pre_burned, pre_clear)

    # Notes shown as uncertainty notes in the provenance
    notes = ["Burn scars are detected by Prithvi per scene, new burn scar = burned after the event on pixels not burned before.",
             "Clouds and shadows are masked with the Sentinel-2 L2A scene classification (SCL).",
             "Published test IoU of the model for burn scars is about 0.88 on HLS data. Accuracy on Sentinel-2 L2A input is not separately validated here.",
             "Open water and other very dark surfaces (e.g. reservoirs, deep shadow) can be misclassified as burn scar. Check the mask against the imagery."]
    
    return _finish(pair, mask, transform, crs, BURN_LEGEND, BURN_COLORS, MODEL_ID, registry.paths.revision(PRITHVI_REPO), str(model.device), {}, [], PRITHVI_SOURCE, notes, "prithvi_burn_scar")


# Source text of each index method, shown in the provenance of its result
METHOD_SOURCES = {
    "ndvi_vegetation_loss": "NDVI vegetation mask (Sentinel-2 B08 near infrared and B04 red), method of the forest loss notebook by B. Pham, github.com/bnhpham/forest-loss-analysis",
    "nbr_burned_area": "Normalized Burn Ratio and dNBR burn severity classes (Key and Benson 2006, USGS), Sentinel-2 bands B8A and B12 as recommended by UN-SPIDER",
    "awei_water_loss": "Automated Water Extraction Index (Feyisa et al. 2014, Remote Sensing of Environment 140), Sentinel-2 bands B02, B03, B08, B11, B12",
    "ndsi_snow_cover_change": "Normalized Difference Snow Index (Sentinel-2 B03 and B11), thresholds after Dietz et al. 2012 and Hagolle et al. 2017 (MAJA)",
    "bsi_bare_soil_change": "Bare Soil Index (Sentinel-2 B02, B04, B08, B11), formula as in the Sentinel Hub custom scripts",
}


# Feature area before and after on the pixels that are assessable on both dates, so that clouds do not bias the difference
def _binary_metrics(method, pre: np.ndarray, post: np.ndarray, transform, pyproj_crs) -> dict[str, float]:

    # Pixels that are assessable on both dates, everything else is set to not assessable on both masks
    both = (pre != MASK_NOT_ASSESSABLE) & (post != MASK_NOT_ASSESSABLE)
    before = compute_mask_area(np.where(both, pre, MASK_NOT_ASSESSABLE).astype(np.uint8), transform, pyproj_crs, [VEGETATION], [MASK_NOT_ASSESSABLE])
    after = compute_mask_area(np.where(both, post, MASK_NOT_ASSESSABLE).astype(np.uint8), transform, pyproj_crs, [VEGETATION], [MASK_NOT_ASSESSABLE])
    slug = method.feature_slug
    
    # Net loss compares the total feature size before and after
    return {f"{slug}_before_km2": round(before.area_km2, 4), f"{slug}_after_km2": round(after.area_km2, 4), "assessable_on_both_dates_km2": round(before.analysed_km2, 4),
            f"net_{slug}_loss_pct": round(net_loss_pct(before.pixel_count, after.pixel_count), 4)}


def _severity_metrics(pre: np.ndarray, post: np.ndarray, burned_dnbr: float, transform, pyproj_crs) -> dict[str, float]:
    # dNBR = NBR before - NBR after, only on pixels that have a value on both dates
    both = np.isfinite(pre) & np.isfinite(post)
    dnbr = (pre - post)[both]

    # Burned pixels are those with a dNBR at or above the burned threshold
    burned = dnbr[dnbr >= burned_dnbr]
    
    # Area of the pixels that are assessable on both dates
    usable = compute_mask_area(np.where(both, 1, MASK_NOT_ASSESSABLE).astype(np.uint8), transform, pyproj_crs, [1], [MASK_NOT_ASSESSABLE])
    
    # The means are 0.0 if there are no usable or burned pixels
    return {"mean_nbr_before": round(float(pre[both].mean()), 4) if both.any() else 0.0, "mean_nbr_after": round(float(post[both].mean()), 4) if both.any() else 0.0,
            "mean_dnbr_of_burned_pixels": round(float(burned.mean()), 4) if burned.size else 0.0, "assessable_on_both_dates_km2": round(usable.analysed_km2, 4)}


# Compute change between the before and after image of a pair with one index method (NBR, NDVI, AWEI, NDSI or BSI)
def run_index_analysis(pair: ImageryPair, method_key: str, thresholds: Thresholds | None = None) -> SegmentationResult:

    # The method must fit the disaster type and both scenes must exist, otherwise there is no change to compare
    thresholds = thresholds or Thresholds()
    method = METHODS[method_key]
    if pair.disaster_type not in method.types:
        raise ValueError(wrong_type_hint(pair.disaster_type, method))
    if pair.post is None:
        raise ValueError("the pair has no post-event scene, so the change cannot be determined")
    if pair.pre is None:
        raise ValueError("the pair has no usable pre-event scene, so the change cannot be determined")

    # The index layer of each date is reused if it exists, otherwise it is computed from the saved scene
    pre_layer, post_layer = ensure_layer(pair.pre, method_key, thresholds), ensure_layer(pair.post, method_key, thresholds)
    
    # Band 0 of a layer file holds the mask (binary methods) or the index values (NBR)
    pre_stack, transform, crs, _ = read_geotiff(Path(pre_layer.mask_path))
    post_stack, _, _, _ = read_geotiff(Path(post_layer.mask_path))
    pre, post = pre_stack[0], post_stack[0]
    pyproj_crs = CRS.from_user_input(crs)
    
    # Binary methods compare the two masks pixel by pixel, NBR is classified into burn severity from dNBR
    if method.kind == "binary":
        classes = binary_change_classes(pre, post)
        metrics = _binary_metrics(method, pre, post, transform, pyproj_crs)
    else:
        classes = severity_classes(pre, post, thresholds.nbr_burned_dnbr)
        metrics = _severity_metrics(pre, post, thresholds.nbr_burned_dnbr, transform, pyproj_crs)

    # Notes for the provenance: the rule that was applied and the caveats of the method
    rule = method.rule_fn(thresholds)
    notes = [f"Rule: {rule}.", *method.caveats]
    if method.kind == "binary":
        notes.append(f"The net figure compares totals like the notebook. A {method.feature} gain elsewhere cancels losses in it, the pixel-wise change classes do not.")
    
    # Extra fields of the result for the UI and for calculate_area. No ML model is used, so there is no model revision or device
    extra = {"measure": method.measure, "affected_classes": list(method.affected_classes), "disclaimer": method.disclaimer, "pre_layer": pre_layer, "post_layer": post_layer}
    
    return _finish(pair, classes, transform, crs, method.legend, method.change_colors, f"{method.index_name} {method.feature} analysis ({rule})", None, "none (no ML model)",
                   threshold_values(method_key, thresholds), [], METHOD_SOURCES[method_key], notes, method_key, metrics, extra)


# Shortcut for the NDVI method
def run_vegetation_loss_estimation(pair: ImageryPair, thresholds: Thresholds | None = None) -> SegmentationResult:
    return run_index_analysis(pair, "ndvi_vegetation_loss", thresholds)


# The segmentation result stored next to a change mask: it knows the legend, the default affected classes, the measure and the disclaimer
def _segmentation_for_mask(mask_path: Path) -> SegmentationResult | None:

    # Look for the stored result whose mask is the same file
    for path in mask_path.parent.glob("segmentation_*.json"):
        try:
            result =SegmentationResult.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if Path(result.mask_path).resolve() == mask_path.resolve():
            return result
    return None


# Deterministic geodesic area of the affected classes of a change mask.
# Without classes, the defaults of the method that made the mask are used.
def calculate_area(mask_path: Path, affected_classes: list[int] | None = None) -> AreaResult:

    # The stored result of the method gives the legend and the default affected classes. Without it the flood legend is assumed
    segmentation = _segmentation_for_mask(mask_path)
    if affected_classes is None:
        affected_classes = list(segmentation.affected_classes) if segmentation else [AFFECTED]
    
    # The requested classes must exist in the legend, and class 0 (not assessable) can never be affected
    allowed = set(segmentation.class_legend) if segmentation else set(FLOOD_LEGEND)
    if not affected_classes or any(value not in allowed for value in affected_classes):
        raise ValueError(f"affected_classes must be a non-empty subset of {sorted(allowed)}")
    if NOT_ASSESSABLE in affected_classes:
        raise ValueError("class 0 (not assessable) cannot be counted as affected")
    
    # Area of the affected classes with its uncertainty range, and the outline of the affected pixels as GeoJSON
    stats = area_from_geotiff(mask_path, affected_classes, [NOT_ASSESSABLE])
    stack, transform, crs, _ = read_geotiff(mask_path)
    geojson_path = polygonize_mask(stack[0], transform, crs, affected_classes, mask_path.with_suffix(".affected.geojson"))
    
    # Area of every class, for the class table in the UI
    legend = segmentation.class_legend if segmentation else FLOOD_LEGEND
    per_class = class_areas_km2(stack[0], transform, CRS.from_user_input(crs), [value for value in sorted(allowed) if value != NOT_ASSESSABLE])
    
    # Notes for the provenance
    notes = [f"Uncertainty range counts half of the {stats.boundary_pixel_count} boundary pixels as possibly misclassified. It does not include method error.",
             f"{stats.obscured_km2:.2f} km² of the AOI could not be assessed (cloud or no data), the true affected area may be larger."]
    
    # Measure, method key and disclaimer are only taken over if the mask comes from one of the index methods
    fields = {"measure": segmentation.measure, "method_key": segmentation.method_key, "disclaimer": segmentation.disclaimer} if segmentation and segmentation.measure else {}
    return AreaResult(mask_path=str(mask_path), affected_classes=affected_classes, area_km2=round(stats.area_km2, 4), area_km2_low=round(stats.area_km2_low, 4),
                      area_km2_high=round(stats.area_km2_high, 4), pixel_count=stats.pixel_count, analysed_km2=round(stats.analysed_km2, 4), obscured_km2=round(stats.obscured_km2, 4),
                      crs=CRS.from_user_input(crs).to_string(), pixel_size_m=stats.pixel_size_m, method=stats.method, geojson_path=str(geojson_path) if geojson_path else None,
                      class_areas_km2={legend[value]: round(area, 4) for value, area in per_class.items()}, **fields,
                      provenance=Provenance(sources=["eo-agent deterministic geodesic area calculation (pyproj, WGS84 ellipsoid)"], uncertainty_notes=notes))
