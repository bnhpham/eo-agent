"""
Streamlit UI to chat with the EO agent

Every assistant answer gets its own interface directly below it (interactive map and before/after imagery).

Run: streamlit run app/streamlit_app.py
"""

import shutil
import sys
import uuid
from datetime import date
from pathlib import Path

import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eo_agent.agent.artifacts import EventGroup, MethodResult, PairCard, TurnArtifacts, build_groups
from eo_agent.agent.runtime import AgentRuntime
from eo_agent.captions import change_legend, layer_caption, layer_legend
from eo_agent.indices import METHODS
from eo_agent.config import get_settings
from eo_agent.map_view import build_map
from eo_agent.schemas import AreaResult, FeatureMaskInfo, ImageryPair, SceneMeta, SegmentationResult

APP_TITLE = "Natural Disaster Monitoring"

EXAMPLES = ["What natural disasters are currently happening in South America?",
            "Show me recent floods in Europe.",
            "Show satellite imagery before and after the wildfire in [location].",
            "Estimate the damage of this wildfire.",
            "How much did the water surface of the reservoir shrink between 2021 and 2023?",
            "How has the snow cover in the Alps changed since last year?", 
            "Is there more bare soil in this region than 5 years ago?"]

DEFAULT_SIDEBAR_BACKGROUND = {"light": "#f0f2f6", "dark": "#262730"}

st.set_page_config(page_title=APP_TITLE, page_icon="🛰️", layout="wide")


# Start everything
@st.cache_resource(show_spinner="Starting agent and MCP servers...") # starts the agent and MCP servers once per process and reuses them on every rerun
def get_runtime() -> AgentRuntime:
    runtime = AgentRuntime(get_settings())
    runtime.start()
    return runtime


def init_state() -> None:
    st.session_state.setdefault("session_id", uuid.uuid4().hex)
    st.session_state.setdefault("messages", [])


# ==============================================================================
# Sidebar
# ==============================================================================

def sidebar_background() -> str:
    configured = st.get_option("theme.secondaryBackgroundColor")
    if configured:
        return configured
    try:
        return DEFAULT_SIDEBAR_BACKGROUND[st.context.theme.type]
    except Exception:
        return DEFAULT_SIDEBAR_BACKGROUND["light"]


def sidebar(runtime: AgentRuntime | None, startup_error: str | None) -> None:
    settings = get_settings()
    # The sidebar does not scroll with the answers. The sticky rule additionally keeps the title in place if the sidebar itself scrolls.
    st.html(f"<style>.st-key-sidebar_title {{position: sticky; top: 0; z-index: 100; background: {sidebar_background()};padding-bottom: 0.5rem;}}"
            ".st-key-sidebar_title h1 {font-size: 1.7rem; line-height: 1.2; padding: 0.25rem 0;}</style>")
    with st.sidebar.container(key="sidebar_title"):
        st.title(APP_TITLE)

    # Current service status
    st.sidebar.header("Status")
    st.sidebar.write(("✅" if settings.openai_api_key else "❌") + " OpenAI API key")
    st.sidebar.write(("✅" if settings.has_cdse_credentials else "⚠️") + " CDSE credentials" + ("" if settings.has_cdse_credentials else " (Earth Search fallback)"))
    if runtime is not None:
        for name, connected in runtime.server_status.items():
            st.sidebar.write(("✅" if connected else "❌") + f" MCP: {name}")
        st.sidebar.caption(f"Model: {runtime.model_name or 'SDK default'}")
        for warning in runtime.startup_warnings:
            st.sidebar.warning(warning)
    if startup_error:
        st.sidebar.error(startup_error)
    
    st.sidebar.divider()

    # Buttons
    if st.sidebar.button("New chat"):
        for key in ("session_id", "messages"):
            st.session_state.pop(key, None)
        st.rerun()
    if st.sidebar.button("Clear downloaded imagery"):
        shutil.rmtree(settings.artifacts_dir, ignore_errors=True)
        st.rerun()
    
    st.sidebar.divider()

    # Example questions
    st.sidebar.caption("Example questions")
    for example in EXAMPLES:
        st.sidebar.caption(f"• {example}")


# ==============================================================================
# Remote sensing image
# ==============================================================================

# Show 1 single remote sensing image
def show_scene(scene: SceneMeta | None, label: str, event_start: date) -> None:

    if scene is None:
        st.info(f"No usable {label.lower()} scene")
    elif scene.png_path and Path(scene.png_path).exists():

        cloud = f"{scene.aoi_cloud_fraction:.0%}" if scene.aoi_cloud_fraction is not None else "n/a"
        offset = (scene.acquired_at.date() - event_start).days
        timing = f"{abs(offset)} days {'after' if offset >= 0 else 'before'} event start"

        # Show image with caption under it
        st.image(scene.png_path, caption=f"{label}: {scene.acquired_at:%Y-%m-%d %H:%M} UTC ({timing}), AOI cloud {cloud}", width="stretch")
    else:
        st.info(f"{label} image is no longer on disk")


# ==============================================================================
# RS indices (e.g., NDVI, NBR, AWEI, NDSI BSI) for semantic segmentation
# ==============================================================================

def show_layer(layer: FeatureMaskInfo | None, label: str) -> None:
    if layer is None:
        return
    
    if Path(layer.mask_png_path).exists():
        st.image(layer.mask_png_path, caption=layer_caption(label, layer), width="stretch")
    else:
        st.info(f"{label} {layer.index_name} layer is no longer on disk")


# Index layer / before-and-after mask
def render_layer_pair(pre: FeatureMaskInfo | None, post: FeatureMaskInfo | None, method_key: str) -> None:
    
    columns = st.columns(2)
    with columns[0]:
        show_layer(pre, "Before")
    with columns[1]:
        show_layer(post, "After")
    
    legend = layer_legend(method_key)
    if legend:
        st.caption(legend)


# ==============================================================================
# Methods for rendering elements appearing in the chat
# ==============================================================================

# Show metrics (before/after comparison)
def render_metrics(metrics: dict[str, float]) -> None:

    for key, value in metrics.items():
        
        # Before-and-after of surface area size in km^2
        if key.endswith("_before_km2") and f"{key.removesuffix('_before_km2')}_after_km2" in metrics:
            slug = key.removesuffix("_before_km2")
            st.metric(f"{slug.replace('_', ' ').capitalize()} before / after", f"{value:.2f} / {metrics[f'{slug}_after_km2']:.2f} km²")
        # Net loss in %
        elif key.startswith("net_") and key.endswith("_loss_pct"):
            slug = key.removeprefix("net_").removesuffix("_loss_pct").replace("_", " ")
            st.metric(f"Net {slug} loss", f"{value:.2f} %", help=f"Totals compared like in the notebook. A {slug} gain elsewhere cancels losses.")
        # Delta Normalized Burn Ratio (dNBR)
        elif key == "mean_dnbr_of_burned_pixels":
            st.metric("Mean dNBR of burned pixels", f"{value:.2f}", help="dNBR = NBR before - NBR after. USGS: 0.1-0.27 low, 0.27-0.44 moderate-low, 0.44-0.66 moderate-high, above 0.66 high severity.")


# Colours of the change mask (labels come from the code, not from tool output)
def render_change_legend(method_key: str) -> None:

    coloured, transparent = change_legend(method_key)
    items = [f'<div><span style="display:inline-block;width:14px;height:14px;margin-right:8px;vertical-align:middle;border:1px solid #888;background:rgba({r},{g},{b},{a / 255:.2f})"></span>{label}</div>'
             for (r, g, b, a), label in coloured]
    if transparent:
        items.append(f'<div style="color:#666">No colour (transparent): {", ".join(transparent)}</div>')
    if items:
        st.markdown('<div style="font-size:0.85rem">' + "".join(items) + "</div>", unsafe_allow_html=True)


# Table with the number of pixels of each class and, once the area tool ran, its km^2
def render_class_table(segmentation: SegmentationResult, area: AreaResult | None) -> None:

    names = list(segmentation.class_pixel_counts)
    table = {"class": names, "pixels": list(segmentation.class_pixel_counts.values())}

    if area is not None:
        km2 = []
        for name in names:
            if name in area.class_areas_km2:
                km2.append(f"{area.class_areas_km2[name]:.2f}")
            else:
                km2.append(f"{area.obscured_km2:.2f}" if name.startswith("not assessable") else "-")
        table["km²"] = km2
    
    with st.expander("Classes", expanded=True):
        st.table(table)


# Result of one analysis method: change mask, class table, metrics and warnings
def render_result(result: MethodResult, pair: ImageryPair, shown_methods: set[str]) -> None:
    
    segmentation, area = result.segmentation, result.area
    title = segmentation.measure.removeprefix("Estimated ").capitalize() if segmentation.measure else segmentation.model_id
    st.markdown(f"**{title}** ({segmentation.model_id})")
    
    if segmentation.method_key not in shown_methods:
        render_layer_pair(segmentation.pre_layer, segmentation.post_layer, segmentation.method_key or "")
    mask_column, number_column = st.columns(2)

    # On the left side: difference mask
    with mask_column:
        # Change mask showing increase/decrease of surface areas
        if Path(segmentation.mask_png_path).exists():
            st.image(segmentation.mask_png_path, caption=f"Change between the two dates ({segmentation.model_id})", width="stretch")
        # Legend for describing the mask's pixels
        render_change_legend(segmentation.method_key or "")
        # Class table showing pixel count increase/decrease
        render_class_table(segmentation, area)
    # On the right side: metrics (before/after comparison)
    with number_column:
        render_metrics(segmentation.metrics)

    for warning in dict.fromkeys(w for w in segmentation.warnings if w not in pair.warnings):
        st.warning(warning)
    
    if area:
        st.caption("Some pixels may be misclassified, mostly along the edges of an area, so the km² values are estimates.")
        st.caption(area.disclaimer)


# Put everything together
def render_card(card: PairCard) -> None:

    # Orignal remote sensing (RS) images before/after
    pair = card.pair
    images = st.columns(2)
    with images[0]:
        show_scene(pair.pre, "Before", pair.event_start)
    with images[1]:
        show_scene(pair.post, "After", pair.event_start)

    # Index layers that were created together with the images, then one block per method that was run
    pre_layers, post_layers = (pair.pre.layers if pair.pre else {}), (pair.post.layers if pair.post else {})
    shown_methods = {key for key in METHODS if key in pre_layers or key in post_layers}

    # Semantic segmentation of each RS image
    for key in (key for key in METHODS if key in shown_methods):
        render_layer_pair(pre_layers.get(key), post_layers.get(key), key)

    # Comparison between semantic segmentations
    for result in card.results:
        render_result(result, pair, shown_methods)

    # Source    
    st.caption(f"Source: {pair.provider} {pair.product}, {pair.resolution_m:.0f} m, {pair.crs}. Pair {pair.pair_index}, id {pair.pair_id}")


# Label of a tab (pair index, post-event date and cloud cover
def tab_label(card: PairCard) -> str:
    pair = card.pair
    if pair.post is None:
        return f"{pair.pair_index}. no post-event scene"
    
    cloud = f", cloud {pair.post.aoi_cloud_fraction:.0%}" if pair.post.aoi_cloud_fraction is not None else ""
    
    return f"{pair.pair_index}. {pair.post.acquired_at:%d %b %Y}{cloud}"


# Show multiple before/after image pairs (if agent picked more than one)
def render_group(group: EventGroup, group_index: int) -> None:
    st.markdown(f"**{group_index + 1}. {group.title}**")

    for warning in dict.fromkeys(w for card in group.cards for w in card.pair.warnings):
        st.warning(warning)
    
    for tab, card in zip(st.tabs([tab_label(card) for card in group.cards]), group.cards):
        with tab:
            render_card(card)


# Collapsed sources and provenance expander
def render_provenance(groups: list[EventGroup], artifacts: TurnArtifacts) -> None:
    
    items = [card.pair for group in groups for card in group.cards] + [result.segmentation for group in groups for card in group.cards for result in card.results] + artifacts.areas
    if not items:
        return
    
    with st.expander("Sources and provenance"):
        sources = dict.fromkeys(source for item in items for source in item.provenance.sources)
        notes = dict.fromkeys(note for item in items for note in item.provenance.uncertainty_notes)
        st.write(f"Retrieved {max(item.provenance.retrieved_at for item in items):%Y-%m-%d %H:%M} UTC")
        
        for source in sources:
            st.write(f"- {source}")
        for note in notes:
            st.caption(note)


# ==============================================================================
# Rendering 1 chat reply
# ==============================================================================

# Map and imagery of one answer. Rendered below that answer, never in a side panel.
def render_interface(artifacts: TurnArtifacts, key: str) -> None:
    
    groups = build_groups(artifacts, get_settings().artifacts_dir)

    # Folium map
    if artifacts.events or groups:
        st_folium(build_map(artifacts.events, groups), height=480, use_container_width=True, returned_objects=[], key=f"map_{key}")

    # Caveat about how complete or reliable the search is
    for note in artifacts.coverage_notes:
        st.caption(f"Coverage note: {note}")

    # Before-and-after RS images + analysis (e.g. semantic segmentation with comparison)
    for group_index, group in enumerate(groups):
        render_group(group, group_index)

    # Quicklook: Cheap, small preview of an area without triggering the full before/after pipeline (user can see what an area looks like right now)
    for quicklook in artifacts.quicklooks:
        if quicklook.get("quicklook_path") and Path(quicklook["quicklook_path"]).exists():
            st.image(quicklook["quicklook_path"], caption=f"CDSE quicklook {quicklook.get('collection')} {quicklook.get('start_date')} to {quicklook.get('end_date')} ({quicklook.get('caveat')})")

    # Sources and provenance
    render_provenance(groups, artifacts)


# Draw the content of 1 single chat message
def render_body(message: dict, index: int) -> None:
    st.markdown(message["content"])
    
    for note in message.get("notes", []):
        st.caption(note)

    if message.get("artifacts") is not None:
        render_interface(message["artifacts"], key=str(index))


def render_message(message: dict, index: int) -> None:
    with st.chat_message(message["role"]):
        render_body(message, index)


def main() -> None:
    init_state()
    runtime, startup_error = None, None
    try:
        runtime = get_runtime()
        startup_error = runtime.startup_error
    except Exception as exc:
        startup_error = f"Agent runtime failed to start: {exc}"
    
    sidebar(runtime, startup_error)

    # Read user input
    prompt = st.chat_input("Ask about current disasters, imagery or affected areas")
    messages = st.session_state.messages
    if not messages and not prompt:
        st.caption("Ask about current disasters, before/after satellite imagery or affected areas. Example questions are listed in the sidebar.")
    for index, message in enumerate(messages):
        render_message(message, index)

    # Chat between user & agent
    if prompt:
        messages.append({"role": "user", "content": prompt})
        render_message(messages[-1], len(messages) - 1)
        
        with st.chat_message("assistant"):
            if runtime is None or startup_error:
                reply = {"role": "assistant", "content": f"The agent is not available: {startup_error}", "notes": [], "artifacts": None}
            else:
                with st.spinner("Querying tools... (satellite downloads and model inference can take several minutes)"):
                    result = runtime.run_turn(st.session_state.session_id, prompt)
                reply = {"role": "assistant", "content": result.answer, "notes": result.notes, "artifacts": result.artifacts}
            
            messages.append(reply)
            render_body(reply, len(messages) - 1)

main()
