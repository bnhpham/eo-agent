"""
Folium map of one answer: GDACS event markers, AOIs, pre/post imagery overlays and mask overlays.

No streamlit import here, so the map can be built and tested on its own.
"""

from pathlib import Path

import folium

from eo_agent.agent.artifacts import EventGroup
from eo_agent.imagery.selection import map_overlay_path
from eo_agent.schemas import BBox

ALERT_COLORS = {"Red": "red", "Orange": "orange", "Green": "green"}
GROUP_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#008080", "#9a6324", "#800000", "#808000", "#000075"]


def _bounds(box: BBox) -> list[list[float]]:
    return [[box.south, box.west], [box.north, box.east]]


def _overlay_bounds(png_path: str, fallback: BBox) -> list[list[float]]:
    # save_display_png writes the exact Web Mercator bounds of each overlay next to it.
    sidecar = Path(png_path).with_suffix(".bounds.json")
    if sidecar.exists():
        return _bounds(BBox.model_validate_json(sidecar.read_text(encoding="utf-8")))
    return _bounds(fallback)


# Light overlay copy if available, otherwise the display PNG
def _map_png(png_path: str | None) -> str | None:
    if not png_path:
        return None
    
    light = map_overlay_path(png_path)
    if light.exists():
        return str(light)
    
    return png_path if Path(png_path).exists() else None


def _add_overlay(fmap: folium.Map | folium.FeatureGroup, png_path: str | None, fallback: BBox, name: str, show: bool, opacity: float = 1.0) -> None:
    image = _map_png(png_path)
    if image:
        folium.raster_layers.ImageOverlay(image=image, bounds=_overlay_bounds(image, fallback), name=name, opacity=opacity, show=show).add_to(fmap)


# Interactive map.
# Satellite imagery is the default basemap, OpenStreetMap stays available in the layer control.
def build_map(events: list[dict], groups: list[EventGroup]) -> folium.Map:
    fmap = folium.Map(location=[20, 0], zoom_start=2, control_scale=True, tiles=None)
    folium.TileLayer("Esri.WorldImagery", name="Esri World Imagery", show=True).add_to(fmap)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", show=False).add_to(fmap)
    corners: list[list[float]] = []

    event_layer = folium.FeatureGroup(name="GDACS events")
    for event in events:
        if event.get("lat") is None:
            continue
        alert = f" | alert {event['alertlevel']}" if event.get("alertlevel") else ""
        size = f"<br>size {event['severity']}" if event.get("severity") else ""
        link = f"<br><a href='{event['report_url']}' target='_blank'>{(event.get('source') or 'gdacs').upper()} report</a>" if event.get("report_url") else ""
        popup = f"<b>{event.get('name')}</b><br>{event.get('eventtype_name')}{alert}{size}<br>{', '.join(event.get('countries') or [])}<br>{event.get('fromdate')} to {event.get('todate')}{link}"
        folium.CircleMarker([event["lat"], event["lon"]], radius=8, color=ALERT_COLORS.get(event.get("alertlevel"), "gray"), fill=True, fill_opacity=0.8,
                            popup=folium.Popup(popup, max_width=300), tooltip=event.get("name")).add_to(event_layer)
        corners.append([event["lat"], event["lon"]])
    event_layer.add_to(fmap)

    for group_index, group in enumerate(groups):
        
        color = GROUP_COLORS[group_index % len(GROUP_COLORS)]
        first = group.cards[0].pair
        folium.Rectangle(_bounds(first.bbox), color=color, weight=3, fill=False, tooltip=f"{group_index + 1}. {group.title}").add_to(fmap)
        corners.extend(_bounds(first.bbox))

        for card_position, card in enumerate(group.cards):
            pair = card.pair
            label = f"{group_index + 1}.{pair.pair_index}"
            
            # Pairs of one event share the AOI, so only the first post-event image (and any analysed one) is visible at start.
            visible = card_position == 0 or card.segmentation is not None
            if card_position == 0 and pair.pre:
                _add_overlay(fmap, pair.pre.png_path, pair.bounds_wgs84, f"{label} pre-event {pair.pre.acquired_at:%Y-%m-%d}", show=False)
            if pair.post:
                _add_overlay(fmap, pair.post.png_path, pair.bounds_wgs84, f"{label} post-event {pair.post.acquired_at:%Y-%m-%d}", show=visible)
            
            # Index layers of every method (masks and maps) are available as hidden layers.
            layers = {}

            # Pre-event RS image
            for scene, role in ((pair.pre if card_position == 0 else None, "pre"), (pair.post, "post")):
                if scene is not None:
                    layers.update({(layer.mask_png_path, role): (layer, scene.acquired_at) for layer in scene.layers.values()})
            # Post-event RS image
            for result in card.results:
                for layer, scene, role in ((result.segmentation.pre_layer, pair.pre, "pre"), (result.segmentation.post_layer, pair.post, "post")):
                    if layer is not None and scene is not None and (card_position == 0 or role == "post"):
                        layers.setdefault((layer.mask_png_path, role), (layer, scene.acquired_at))
            # Index images (e.g., NDVI or NBR)
            for (_, role), (layer, acquired_at) in layers.items():
                kind = "map" if layer.kind == "index_map" else "mask"
                name = f"{label} {role}-event {layer.index_name if layer.kind == 'index_map' else layer.feature} {kind} {acquired_at:%Y-%m-%d}"
                _add_overlay(fmap, layer.mask_png_path, pair.bounds_wgs84, name, show=False)
            # Change masks (comparison between pre-event and post-event segmentation)
            for result in card.results:
                segmentation = result.segmentation
                name = segmentation.measure.removeprefix("Estimated ").lower() if segmentation.measure else f"{segmentation.disaster_type} mask"
                _add_overlay(fmap, segmentation.mask_png_path, segmentation.bounds_wgs84, f"{label} {name}", show=True, opacity=0.75)
    
    if corners:
        fmap.fit_bounds(corners, padding=(20, 20), max_zoom=13)
    
    folium.LayerControl(collapsed=True).add_to(fmap)
    return fmap
