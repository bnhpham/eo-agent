"""
Plain-language captions for the UI.
"""

from eo_agent.indices import METHODS
from eo_agent.schemas import FeatureMaskInfo

LEGENDS = {
    "ndvi_vegetation_loss": "NDVI masks: green = vegetation (pixel value 1), black = no vegetation (0), grey = excluded (clouds, shadows or no data).",
    "awei_water_loss": "AWEI masks: blue = water (1), black = no water (0), grey = excluded (clouds, shadows or no data).",
    "ndsi_snow_cover_change": "NDSI masks: light blue = snow (1), black = no snow (0), grey = excluded (clouds, shadows or no data).",
    "bsi_bare_soil_change": "BSI masks: brown = bare soil (1), black = no bare soil (0), grey = excluded (clouds, shadows or no data).",
    "nbr_burned_area": "NBR maps: dark green = dense healthy vegetation, yellow = sparse or dry, dark red = burned, bare or water, grey = excluded (clouds, shadows or no data).",
}


def layer_legend(method_key: str) -> str:
    return LEGENDS.get(method_key, "")


# Caption of a scene layer. Binary masks give three numbers that add up (feature, no feature, excluded), index maps give the mean value
def layer_caption(label: str, layer: FeatureMaskInfo) -> str:
    
    excluded = f", excluded {layer.not_assessable_km2:.2f} km² (clouds, shadows, no data)" if layer.not_assessable_km2 is not None else ""
    if layer.kind == "index_map":
        mean = f"mean {layer.mean_index:.2f} over {layer.assessable_km2:.2f} km²" if layer.mean_index is not None else "no usable pixels"
        return f"{label} {layer.index_name} map ({layer.rule}): {mean}{excluded}"
    
    title = f"{label} {layer.feature} mask ({layer.rule}): "
    if layer.feature_fraction is None:
        return title + "no usable pixels (everything is cloud, shadow or no data)"
    
    share = layer.feature_fraction
    return title + f"{layer.feature} {layer.feature_km2:.2f} km² ({share:.0%}), no {layer.feature} {layer.assessable_km2 - layer.feature_km2:.2f} km² ({1 - share:.0%}){excluded}"


# Legend of the change mask of a method: (rgba colour as drawn in the mask, label) of every coloured class and the labels of the classes that have no colour (transparent)
def change_legend(method_key: str) -> tuple[list[tuple[tuple[int, int, int, int], str]], list[str]]:

    method = METHODS.get(method_key)
    if method is None:
        return [], []
    
    coloured = [(method.change_colors[value], label) for value, label in method.legend.items() if value in method.change_colors]
    transparent = [label for value, label in method.legend.items() if value not in method.change_colors]
    return coloured, transparent


# Kept for callers that only deal with the NDVI vegetation mask.
vegetation_caption = layer_caption
