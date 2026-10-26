"""
Spectral index methods for Sentinel-2 before/after analysis, no I/O.

Five methods share one pipeline. Four are binary feature methods (a pixel has the feature or not) and one grades a burn severity:

| key | index | answers | applies to |
|---|---|---|---|
| nbr_burned_area | NBR = (B8A - B12) / (B8A + B12), dNBR = before - after | burned area and burn severity | wildfire |
| ndvi_vegetation_loss | NDVI = (B08 - B04) / (B08 + B04) | vegetation extent and its loss | wildfire, drought, land degradation |
| awei_water_loss | AWEI (Feyisa et al. 2014) | surface water extent and its loss | drought |
| ndsi_snow_cover_change | NDSI = (B03 - B11) / (B03 + B11) | snow cover and its change | snow |
| bsi_bare_soil_change | BSI = ((B11 + B04) - (B08 + B02)) / ((B11 + B04) + (B08 + B02)) | bare soil extent and its gain | land degradation, drought |

Reflectance is 0 to 1 (digital numbers / 10000). Scene masks use 1 (feature), 0 (no feature) and 255 (not assessable: cloud, shadow, no data).
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from eo_agent.vegetation import LOSS_COLORS, MASK_NOT_ASSESSABLE, VEGETATION_MASK_COLORS, vegetation_change_classes

EPSILON = 1e-6
REFLECTANCE_SCALE = 10000.0
ANALYSIS_BANDS = ["B02", "B03", "B04", "B08", "B8A", "B11", "B12"]  # one download serves every method

FEATURE, NO_FEATURE = 1, 0
# Classes of the change mask between two dates (binary methods). Class 0 is never counted.
NOT_ASSESSABLE, NONE_ON_BOTH, LOSS, ON_BOTH, GAIN = 0, 1, 2, 3, 4
# Classes of the NBR burn severity mask.
SEVERITY_UNBURNED, SEVERITY_LOW, SEVERITY_MODERATE_LOW, SEVERITY_MODERATE_HIGH, SEVERITY_HIGH, SEVERITY_REGROWTH = 1, 2, 3, 4, 5, 6
USGS_DNBR_EDGES = (0.27, 0.44, 0.66)  # low | moderate-low | moderate-high | high, the lower bound of "low" is the configurable burned threshold (0.1)
USGS_REGROWTH_BELOW = -0.1


@dataclass(frozen=True)
class Thresholds:
    ndvi: float = 0.5
    nbr_burned_dnbr: float = 0.1
    awei: float = -0.15
    awei_variant: str = "nsh"
    ndsi: float = 0.4
    ndsi_min_green: float = 0.3
    bsi: float = 0.0
    bsi_max_ndvi: float = 0.2

    @classmethod
    def from_settings(cls, settings) -> "Thresholds":
        return cls(ndvi=settings.ndvi_threshold, nbr_burned_dnbr=settings.nbr_burned_dnbr, awei=settings.awei_threshold, awei_variant=settings.awei_variant,
                   ndsi=settings.ndsi_threshold, ndsi_min_green=settings.ndsi_min_green, bsi=settings.bsi_threshold, bsi_max_ndvi=settings.bsi_max_ndvi)


# Band stack of digital numbers to a dict of float32 reflectance arrays
def reflectance_bands(stack: np.ndarray, bands: list[str]) -> dict[str, np.ndarray]:
    return {name: stack[index].astype(np.float32) / REFLECTANCE_SCALE for index, name in enumerate(bands)}


def _require(reflectance: dict[str, np.ndarray], *names: str) -> None:
    missing = [name for name in names if name not in reflectance]
    if missing:
        raise ValueError(f"bands {missing} are required, the scene has {sorted(reflectance)}")


# ==============================================================================
# Spectral indices
# ==============================================================================

def index_ndvi(r: dict[str, np.ndarray], t: Thresholds) -> np.ndarray:
    _require(r, "B08", "B04")
    return (r["B08"] - r["B04"]) / (r["B08"] + r["B04"] + EPSILON)

def index_nbr(r: dict[str, np.ndarray], t: Thresholds) -> np.ndarray:
    _require(r, "B8A", "B12")
    return (r["B8A"] - r["B12"]) / (r["B8A"] + r["B12"] + EPSILON)

def index_awei(r: dict[str, np.ndarray], t: Thresholds) -> np.ndarray:
    """AWEInsh (default, urban and dark surfaces) or AWEIsh (shadow-prone terrain), Feyisa et al. 2014 with Sentinel-2 bands."""
    _require(r, "B02", "B03", "B08", "B11", "B12")
    if t.awei_variant == "sh":
        return r["B02"] + 2.5 * r["B03"] - 1.5 * (r["B08"] + r["B11"]) - 0.25 * r["B12"]
    return 4.0 * (r["B03"] - r["B11"]) - (0.25 * r["B08"] + 2.75 * r["B12"])

def index_ndsi(r: dict[str, np.ndarray], t: Thresholds) -> np.ndarray:
    _require(r, "B03", "B11")
    return (r["B03"] - r["B11"]) / (r["B03"] + r["B11"] + EPSILON)

def index_bsi(r: dict[str, np.ndarray], t: Thresholds) -> np.ndarray:
    _require(r, "B02", "B04", "B08", "B11")
    numerator = (r["B11"] + r["B04"]) - (r["B08"] + r["B02"])
    return numerator / ((r["B11"] + r["B04"]) + (r["B08"] + r["B02"]) + EPSILON)


# ==============================================================================
# Masks based spectral indices
# ==============================================================================

def mask_ndvi(index: np.ndarray, r: dict[str, np.ndarray], t: Thresholds) -> np.ndarray:
    return index >= t.ndvi

def mask_awei(index: np.ndarray, r: dict[str, np.ndarray], t: Thresholds) -> np.ndarray:
    return index > t.awei

def mask_ndsi(index: np.ndarray, r: dict[str, np.ndarray], t: Thresholds) -> np.ndarray:
    return (index >= t.ndsi) & (r["B03"] >= t.ndsi_min_green)  # the brightness test keeps dark, wet or shaded surfaces out

def mask_bsi(index: np.ndarray, r: dict[str, np.ndarray], t: Thresholds) -> np.ndarray:
    return (index >= t.bsi) & (index_ndvi(r, t) < t.bsi_max_ndvi)  # the NDVI ceiling keeps dry vegetation out


# uint8 mask: 1 where the feature is present, 0 where not, 255 where the pixel is not assessable
def feature_mask(method: "IndexMethod", index: np.ndarray, r: dict[str, np.ndarray], assessable: np.ndarray, t: Thresholds) -> np.ndarray:
    mask = method.mask_fn(index, r, t).astype(np.uint8)
    mask[~assessable | ~np.isfinite(index)] = MASK_NOT_ASSESSABLE
    return mask


# Change classes of two feature masks, pixels that are not assessable on both dates are class 0.
def binary_change_classes(pre_mask: np.ndarray, post_mask: np.ndarray) -> np.ndarray:
    return vegetation_change_classes(pre_mask, post_mask)


# USGS burn severity classes from dNBR = NBR(before) - NBR(after).
# NaN means not assessable. dNBR >= burned_dnbr counts as burned.
def severity_classes(pre_nbr: np.ndarray, post_nbr: np.ndarray, burned_dnbr: float) -> np.ndarray:

    if pre_nbr.shape != post_nbr.shape:
        raise ValueError("pre and post NBR maps must have the same shape")
    
    dnbr = pre_nbr - post_nbr
    classes = np.full(dnbr.shape, NOT_ASSESSABLE, dtype=np.uint8)
    valid = np.isfinite(dnbr)
    low_edge, moderate_low_edge, high_edge = USGS_DNBR_EDGES

    classes[valid & (dnbr >= USGS_REGROWTH_BELOW) & (dnbr < burned_dnbr)] = SEVERITY_UNBURNED
    classes[valid & (dnbr < USGS_REGROWTH_BELOW)] = SEVERITY_REGROWTH
    classes[valid & (dnbr >= burned_dnbr) & (dnbr < low_edge)] = SEVERITY_LOW
    classes[valid & (dnbr >= low_edge) & (dnbr < moderate_low_edge)] = SEVERITY_MODERATE_LOW
    classes[valid & (dnbr >= moderate_low_edge) & (dnbr < high_edge)] = SEVERITY_MODERATE_HIGH
    classes[valid & (dnbr >= high_edge)] = SEVERITY_HIGH

    return classes


# Net loss of the feature in percent of the pixels before (positive = loss, negative = net gain).
# Gains elsewhere cancel losses.
def net_loss_pct(before_pixels: int, after_pixels: int) -> float:
    return 100.0 * (before_pixels - after_pixels) / (before_pixels + EPSILON)


# Colour map of an NBR layer: dark red (burned, below -0.25) to yellow (about 0) to dark green (dense healthy vegetation, 0.6 and above), grey where not assessable
def nbr_to_rgba(nbr: np.ndarray) -> np.ndarray:
    stops = np.array([-0.25, 0.0, 0.3, 0.6])
    colors = np.array([[139, 0, 0], [255, 222, 121], [140, 200, 100], [0, 100, 0]], dtype=np.float32)
    clipped = np.nan_to_num(nbr, nan=0.0)
    rgb = np.stack([np.interp(clipped, stops, colors[:, channel]) for channel in range(3)], axis=-1).astype(np.uint8)
    rgba = np.concatenate([rgb, np.full(nbr.shape + (1,), 255, dtype=np.uint8)], axis=-1)
    rgba[~np.isfinite(nbr)] = (170, 170, 170, 255)
    return rgba


def binary_change_legend(feature: str) -> dict[int, str]:
    return {NOT_ASSESSABLE: "not assessable (cloud, shadow or no data on at least one date)",
            NONE_ON_BOTH: f"no {feature} on both dates",
            LOSS: f"{feature} loss ({feature} before, no {feature} after)",
            ON_BOTH: f"{feature} on both dates",
            GAIN: f"{feature} gain (no {feature} before, {feature} after)"}


SEVERITY_LEGEND = {NOT_ASSESSABLE: "not assessable (cloud, shadow or no data on at least one date)",
                   SEVERITY_UNBURNED: "unburned (dNBR below the burned threshold)",
                   SEVERITY_LOW: "low burn severity",
                   SEVERITY_MODERATE_LOW: "moderate-low burn severity",
                   SEVERITY_MODERATE_HIGH: "moderate-high burn severity",
                   SEVERITY_HIGH: "high burn severity",
                   SEVERITY_REGROWTH: "regrowth or greening (dNBR below -0.1)"}
SEVERITY_COLORS = {NOT_ASSESSABLE: (128, 128, 128, 90),
                   SEVERITY_LOW: (255, 235, 59, 200),
                   SEVERITY_MODERATE_LOW: (255, 165, 0, 215),
                   SEVERITY_MODERATE_HIGH: (255, 69, 0, 225),
                   SEVERITY_HIGH: (139, 0, 0, 235),
                   SEVERITY_REGROWTH: (144, 238, 144, 150)}


@dataclass(frozen=True)
class IndexMethod:
    key: str
    tool_name: str
    layer_name: str  # short name in file names, e.g. post_ndvi.tif
    kind: str  # "binary" or "severity"
    index_name: str
    feature: str  # what the binary mask marks, "burn severity" for NBR
    feature_slug: str  # used in metric names, e.g. vegetation -> vegetation_before_km2
    measure: str  # label of the km² result
    types: tuple[str, ...]
    affected_classes: tuple[int, ...]
    legend: dict[int, str]
    change_colors: dict[int, tuple[int, int, int, int]]
    mask_colors: dict[int, tuple[int, int, int, int]]
    rule_fn: Callable[[Thresholds], str]
    index_fn: Callable[[dict[str, np.ndarray], Thresholds], np.ndarray]
    mask_fn: Callable[[np.ndarray, dict[str, np.ndarray], Thresholds], np.ndarray] | None
    disclaimer: str
    caveats: tuple[str, ...]
    description: str


def _mask_colors(present: tuple[int, int, int, int]) -> dict[int, tuple[int, int, int, int]]:
    return {FEATURE: present, NO_FEATURE: (25, 25, 25, 255), MASK_NOT_ASSESSABLE: (170, 170, 170, 255)}


METHODS: dict[str, IndexMethod] = {method.key: method for method in (
    IndexMethod(
        key="nbr_burned_area", tool_name="estimate_burned_area", layer_name="nbr", kind="severity", index_name="NBR", feature="burn severity", feature_slug="burned",
        measure="Estimated burned area", types=("wildfire",), affected_classes=(SEVERITY_LOW, SEVERITY_MODERATE_LOW, SEVERITY_MODERATE_HIGH, SEVERITY_HIGH),
        legend=SEVERITY_LEGEND, change_colors=SEVERITY_COLORS, mask_colors={}, rule_fn=lambda t: f"NBR = (B8A - B12) / (B8A + B12), burned where dNBR >= {t.nbr_burned_dnbr}",
        index_fn=index_nbr, mask_fn=None,
        disclaimer="Satellite-derived estimate of burned area and burn severity from the difference of the Normalized Burn Ratio (dNBR, USGS severity classes). "
                   "This is not damage to buildings, infrastructure or economic loss.",
        caveats=("Healthy vegetation has high near infrared and low shortwave infrared reflectance, burned areas the opposite, so dNBR rises with burn severity.",
                 "dNBR classes follow the USGS standard. Severity also depends on the vegetation before the fire, and smoke, clouds, ash washed away and fast regrowth change the result.",
                 "Bare or sparsely vegetated land and dry grassland give small dNBR even when burned."),
        description="Estimate burned area and burn severity of a WILDFIRE pair with the Normalized Burn Ratio (NBR = (B8A - B12) / (B8A + B12)). "
                    "dNBR = NBR before - NBR after, burned where dNBR >= 0.1, classified into USGS severity classes (low, moderate-low, moderate-high, high). "
                    "This is the method to use for wildfire damage, burned area and burn severity. Returns a severity mask and NBR maps; call calculate_affected_area with the returned mask_path for the km² per class."),
    IndexMethod(
        key="ndvi_vegetation_loss", tool_name="estimate_vegetation_loss", layer_name="ndvi", kind="binary", index_name="NDVI", feature="vegetation", feature_slug="vegetation",
        measure="Estimated vegetation loss", types=("wildfire", "drought", "land_degradation"), affected_classes=(LOSS,), legend=binary_change_legend("vegetation"),
        change_colors=LOSS_COLORS, mask_colors=VEGETATION_MASK_COLORS, rule_fn=lambda t: f"NDVI >= {t.ndvi}", index_fn=index_ndvi, mask_fn=mask_ndvi,
        disclaimer="Satellite-derived estimate of vegetation loss from an NDVI threshold. This is not a confirmed burned area, destruction, damage to buildings or economic loss.",
        caveats=("NDVI = (B08 - B04) / (B08 + B04), a pixel is vegetation when NDVI >= the threshold. Vegetation loss = vegetation before and none after, only where both dates are clear.",
                 "Harvest, drought, season and clouds missed by the cloud mask also lower NDVI. Sparse or dry vegetation below the threshold is not counted, so fire losses in dry scrub are underestimated."),
        description="Estimate vegetation loss between the before and after image of a pair from binary NDVI vegetation masks (NDVI = (B08 - B04) / (B08 + B04), vegetation where NDVI >= 0.5). "
                    "Use it for vegetation loss or vegetation stress (wildfire, drought, land degradation). For wildfire burned area and severity use estimate_burned_area instead. "
                    "Returns a change mask; call calculate_affected_area with the returned mask_path for the km² of vegetation loss."),
    IndexMethod(
        key="awei_water_loss", tool_name="estimate_water_loss", layer_name="awei", kind="binary", index_name="AWEI", feature="water", feature_slug="water",
        measure="Estimated water loss", types=("drought",), affected_classes=(LOSS,), legend=binary_change_legend("water"),
        change_colors={NOT_ASSESSABLE: (128, 128, 128, 90), LOSS: (255, 140, 0, 215), GAIN: (0, 112, 255, 200), ON_BOTH: (0, 0, 139, 110)},
        mask_colors=_mask_colors((0, 112, 255, 255)), rule_fn=lambda t: f"AWEI{t.awei_variant} > {t.awei}", index_fn=index_awei, mask_fn=mask_awei,
        disclaimer="Satellite-derived estimate of the loss of open surface water from the Automated Water Extraction Index. This is not a drought severity index, "
                   "and not a measure of impact on people, agriculture or economy.",
        caveats=("AWEI separates open water from dark surfaces. Water that is shallow, turbid, frozen, shaded or under cloud and vegetation is missed.",
                 "Water extent of lakes and reservoirs also changes with the season and with dam operation, so a loss is only a drought signal when the dates are in the same season.",
                 "The threshold is -0.15 and not the 0 of the original paper. Older Sentinel-2 L2A scenes keep a little haze over dark water, so with 0 a large reservoir was almost missed in a 2021 scene. A lower threshold can add shadows and wet soil.",
                 "Terrain shadows can be mistaken for water (use AWEI_VARIANT=sh in mountains), and snow and ice are also flagged as water."),
        description="Estimate the loss of open surface water (lakes, reservoirs, rivers) between the before and after image of a DROUGHT pair with the Automated Water Extraction Index (AWEI, water where AWEI is above its threshold, default -0.15). "
                    "Water loss = water before and none after. Returns a change mask; call calculate_affected_area with the returned mask_path for the km² of water loss."),
    IndexMethod(
        key="ndsi_snow_cover_change", tool_name="estimate_snow_cover_change", layer_name="ndsi", kind="binary", index_name="NDSI", feature="snow", feature_slug="snow",
        measure="Snow-covered area on the later date", types=("snow",), affected_classes=(ON_BOTH, GAIN), legend=binary_change_legend("snow"),
        change_colors={NOT_ASSESSABLE: (128, 128, 128, 90), LOSS: (255, 165, 0, 215), GAIN: (0, 230, 255, 215), ON_BOTH: (200, 225, 255, 215)},
        mask_colors=_mask_colors((200, 230, 255, 255)), rule_fn=lambda t: f"NDSI >= {t.ndsi} and green reflectance >= {t.ndsi_min_green}", index_fn=index_ndsi, mask_fn=mask_ndsi,
        disclaimer="Satellite-derived snow cover from the Normalized Difference Snow Index. This is snow cover only, not snow depth, snow water equivalent, snowpack stability, "
                   "an avalanche risk assessment or a flood forecast.",
        caveats=("NDSI = (B03 - B11) / (B03 + B11), snow where NDSI >= the threshold and the pixel is bright in the green band.",
                 "Snow under forest canopy, in terrain shadow or under cloud is missed, and thin cloud can still be confused with snow.",
                 "Avalanche risk and winter flood potential depend on slope, snow depth, snowpack layering, temperature and rain. Snow cover is only one input."),
        description="Map snow cover and its change between a before and an after date with the Normalized Difference Snow Index (NDSI = (B03 - B11) / (B03 + B11), snow where NDSI >= 0.4 and bright). "
                    "Use it for snow cover, snow melt and as an input for winter flood potential or avalanche context, for a SNOW pair. It does NOT give an avalanche risk or a flood forecast. "
                    "Returns a change mask; call calculate_affected_area with the returned mask_path for the snow-covered km² on the later date."),
    IndexMethod(
        key="bsi_bare_soil_change", tool_name="estimate_bare_soil_change", layer_name="bsi", kind="binary", index_name="BSI", feature="bare soil", feature_slug="bare_soil",
        measure="Estimated bare soil gain", types=("land_degradation", "drought"), affected_classes=(GAIN,), legend=binary_change_legend("bare soil"),
        change_colors={NOT_ASSESSABLE: (128, 128, 128, 90), LOSS: (144, 238, 144, 150), GAIN: (205, 92, 0, 215), ON_BOTH: (222, 184, 135, 110)},
        mask_colors=_mask_colors((205, 133, 63, 255)), rule_fn=lambda t: f"BSI >= {t.bsi} and NDVI < {t.bsi_max_ndvi}", index_fn=index_bsi, mask_fn=mask_bsi,
        disclaimer="Satellite-derived estimate of the gain of bare soil from the Bare Soil Index between two dates. Two dates are a comparison, not a trend, "
                   "and not a diagnosis of land degradation or desertification.",
        caveats=("BSI = ((B11 + B04) - (B08 + B02)) / ((B11 + B04) + (B08 + B02)), bare soil where BSI >= the threshold and NDVI is below its ceiling (0.2). BSI alone flagged most dry Mediterranean scrub as bare soil and almost none of a pale desert, so the NDVI ceiling is added.",
                 "Rock, sand, built-up land, fallow and freshly harvested or ploughed fields also count as bare soil. Soil moisture and the season change BSI.",
                 "Land degradation and desertification are long-term processes: they need several years at the same season, not one before/after pair."),
        description="Estimate the gain of bare soil between a before and an after date with the Bare Soil Index (BSI, bare soil where BSI >= its threshold and NDVI < 0.2), "
                    "for LAND_DEGRADATION or desertification questions (also for drought). Bare soil gain = no bare soil before and bare soil after. "
                    "Returns a change mask; call calculate_affected_area with the returned mask_path for the km² of bare soil gain."),
)}

METHODS_BY_TYPE: dict[str, tuple[str, ...]] = {"wildfire": ("nbr_burned_area", "ndvi_vegetation_loss"), "drought": ("awei_water_loss", "ndvi_vegetation_loss", "bsi_bare_soil_change"),
                                                "land_degradation": ("bsi_bare_soil_change", "ndvi_vegetation_loss"), "snow": ("ndsi_snow_cover_change",)}
# Layers that are created together with the before/after images. Everything else is computed when the agent calls the tool.
AUTO_METHODS: dict[str, tuple[str, ...]] = {"wildfire": ("nbr_burned_area", "ndvi_vegetation_loss"), "drought": ("awei_water_loss",), "snow": ("ndsi_snow_cover_change",),
                                            "land_degradation": ("bsi_bare_soil_change",)}


# The thresholds a method used, stored with every result. The first entry is the main threshold
def threshold_values(method_key: str, t: Thresholds) -> dict[str, float]:
    return {"ndvi_vegetation_loss": {"ndvi": t.ndvi},
            "nbr_burned_area": {"nbr_burned_dnbr": t.nbr_burned_dnbr},
            "awei_water_loss": {"awei": t.awei},
            "ndsi_snow_cover_change": {"ndsi": t.ndsi, "ndsi_min_green": t.ndsi_min_green},
            "bsi_bare_soil_change": {"bsi": t.bsi, "bsi_max_ndvi": t.bsi_max_ndvi}}[method_key]


def tools_for_type(disaster_type: str) -> list[str]:
    return [METHODS[key].tool_name for key in METHODS_BY_TYPE.get(disaster_type, ())]


def method_by_tool(tool_name: str) -> IndexMethod | None:
    return next((method for method in METHODS.values() if method.tool_name == tool_name), None)


# Why a tool does not serve a pair, and which tool does
def wrong_type_hint(disaster_type: str, method: IndexMethod) -> str:
    right = tools_for_type(disaster_type)
    if right:
        return f"{method.tool_name} does not apply to a {disaster_type} pair. For {disaster_type} use: {', '.join(right)}."
    
    served = ", ".join(sorted(set(method.types)))
    return f"{method.tool_name} serves {served} pairs only. This pair is a {disaster_type}: imagery only, no damage estimation is available for it yet."
