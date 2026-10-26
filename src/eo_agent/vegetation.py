"""
NDVI vegetation masks and vegetation change, no I/O.

Formula: NDVI = (NIR - RED) / (NIR + RED + 1e-6)
A pixel is vegetation (1) when NDVI >= threshold (0.5), otherwise 0.

For Sentinel-2 L2A, band B08 as near infrared and B04 as red.

Cloud, shadow and no-data pixels are not assessable (255) instead of silently counting as "no vegetation".
Change between two dates is computed pixel by pixel instead of from pixel totals only.
"""

import numpy as np

NDVI_EPSILON = 1e-6

# Mask values of one scene.
VEGETATION, NO_VEGETATION, MASK_NOT_ASSESSABLE = 1, 0, 255

# Classes of the change mask between two dates. Class 2 is the "affected" class of calculate_affected_area.
NOT_ASSESSABLE, UNCHANGED_NO_VEGETATION, VEGETATION_LOSS, UNCHANGED_VEGETATION, VEGETATION_GAIN = 0, 1, 2, 3, 4

LOSS_LEGEND = {NOT_ASSESSABLE: "not assessable (cloud, shadow or no data on at least one date)", UNCHANGED_NO_VEGETATION: "no vegetation on both dates",
               VEGETATION_LOSS: "vegetation loss (vegetation before, no vegetation after)", UNCHANGED_VEGETATION: "vegetation on both dates",
               VEGETATION_GAIN: "vegetation gain (no vegetation before, vegetation after)"}
LOSS_COLORS = {NOT_ASSESSABLE: (128, 128, 128, 90), VEGETATION_LOSS: (255, 69, 0, 210), VEGETATION_GAIN: (144, 238, 144, 150)}
# Binary look: green = vegetation (1), near black = no vegetation (0), grey = not assessable.
VEGETATION_MASK_COLORS = {VEGETATION: (34, 139, 34, 255), NO_VEGETATION: (25, 25, 25, 255), MASK_NOT_ASSESSABLE: (170, 170, 170, 255)}


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    return (nir - red) / (nir + red + NDVI_EPSILON)


# uint8 mask: 1 where NDVI >= threshold, 0 elsewhere, 255 where the pixel is not assessable
def vegetation_mask(ndvi_values: np.ndarray, assessable: np.ndarray, threshold: float) -> np.ndarray:
    mask = (ndvi_values >= threshold).astype(np.uint8)
    mask[~assessable] = MASK_NOT_ASSESSABLE
    return mask


# Vegetation mask of one scene from its band stack, B08 (near infrared) and B04 (red) are required
def scene_vegetation_mask(reflectance: np.ndarray, bands: list[str], valid: np.ndarray, cloud: np.ndarray, threshold: float) -> np.ndarray:
    for name in ("B08", "B04"):
        if name not in bands:
            raise ValueError(f"band {name} is required for NDVI, scene has {bands}")
    values = ndvi(reflectance[bands.index("B08")], reflectance[bands.index("B04")])
    return vegetation_mask(values, valid & ~cloud, threshold)


# Change classes from two vegetation masks. A pixel must be assessable on both dates, otherwise it is not assessable
def vegetation_change_classes(pre_mask: np.ndarray, post_mask: np.ndarray) -> np.ndarray:
    if pre_mask.shape != post_mask.shape:
        raise ValueError("pre and post masks must have the same shape")
    
    both = (pre_mask != MASK_NOT_ASSESSABLE) & (post_mask != MASK_NOT_ASSESSABLE)
    classes = np.full(pre_mask.shape, NOT_ASSESSABLE, dtype=np.uint8)
    classes[both & (pre_mask == NO_VEGETATION) & (post_mask == NO_VEGETATION)] = UNCHANGED_NO_VEGETATION
    classes[both & (pre_mask == VEGETATION) & (post_mask == NO_VEGETATION)] = VEGETATION_LOSS
    classes[both & (pre_mask == VEGETATION) & (post_mask == VEGETATION)] = UNCHANGED_VEGETATION
    classes[both & (pre_mask == NO_VEGETATION) & (post_mask == VEGETATION)] = VEGETATION_GAIN
    
    return classes


# Net vegetation loss in percent of the vegetation before (positive = loss, negative = net gain).
# Gains elsewhere cancel losses
def net_change_pct(before_pixels: int, after_pixels: int) -> float:
    return 100.0 * (before_pixels - after_pixels) / (before_pixels + NDVI_EPSILON)
