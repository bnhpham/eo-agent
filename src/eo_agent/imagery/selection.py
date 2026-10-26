"""
Select and download the best pre- and post-event Sentinel-2 scenes for an AOI.
"""

import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from eo_agent.geo.raster import save_display_png, true_color, wgs84_bounds, write_geotiff
from eo_agent.imagery.base import FetchedScene, ImageryProvider, SceneCandidate, TargetGrid, bands_for, group_by_day, resolution_for
from eo_agent.indices import AUTO_METHODS, METHODS, Thresholds
from eo_agent.layers import DISPLAY_MAX_SIDE, MAP_MAX_SIDE, MASK_BAND_NAMES, build_layer, map_overlay_path  # noqa: F401  (re-exported for other modules)
from eo_agent.schemas import ANALYSIS_DISASTER_TYPES, SENTINEL2_FIRST_DATE, VISUAL_ONLY_NOTE, ImageryPair, ImageryRequest, Provenance, SceneMeta, utc_now

SCREEN_TOP_K = 4
MIN_VALID_FRACTION = 0.5
CLOUDY_WARNING_FRACTION = 0.3
FETCH_WORKERS = 3
SENTINEL2_FIRST_ACQUISITION = date(2015, 6, 23)
SPARSE_COVERAGE_BEFORE_YEAR = 2018
SEASON_WARNING_DAYS = 45
PROVIDER_SOURCES = {"cdse": "Copernicus Data Space Ecosystem, Sentinel Hub Catalog and Process API (sh.dataspace.copernicus.eu)",
                    "earth-search": "Element84 Earth Search STAC, Sentinel-2 L2A COGs on AWS (earth-search.aws.element84.com)"}


class NoImageryError(RuntimeError):
    pass


@dataclass(frozen=True)
class ScreenedCandidate:
    candidate: SceneCandidate
    cloud_fraction: float
    valid_fraction: float
    days_from_reference: int

    @property
    def score(self) -> float:
        # The lower, the better. Cloud and missing coverage dominate, closeness in time breaks ties.
        return self.cloud_fraction + 2.0 * (1.0 - self.valid_fraction) + self.days_from_reference / 60.0


# Decides which date ranges to search for the before and after satellite scenes.
# Returns (pre window, post window, post reference date).
# Both windows are inclusive ((start, end) date ranges).
def event_windows(request: ImageryRequest, today: date) -> tuple[tuple[date, date], tuple[date, date], date]:
    
    if request.baseline_date is not None:
        span = timedelta(days=request.baseline_window_days)
        pre_window = (request.baseline_date - span, min(request.baseline_date + span, request.event_start - timedelta(days=1)))
    else:
        pre_window = (request.event_start - timedelta(days=request.pre_days), request.event_start - timedelta(days=1))
    
    # Burn scars are best seen once the fire is out.
    # For every other type, such as floods, it is the event start (floods are best seen near their peak).
    post_reference = request.event_end if (request.disaster_type == "wildfire" and request.event_end) else request.event_start
    post_window = (post_reference, min(post_reference + timedelta(days=request.post_days), today))
    
    return pre_window, post_window, post_reference


def rank_candidates(screened: list[ScreenedCandidate]) -> list[ScreenedCandidate]:
    usable = [item for item in screened if item.valid_fraction >= MIN_VALID_FRACTION]
    return sorted(usable, key=lambda item: (item.score, item.days_from_reference))


# Up to `count` best scenes, ordered by acquisition date.
# Only scenes clear enough are kept, the single best one is returned as a fallback.
def choose_scenes(provider: ImageryProvider, candidates: list[SceneCandidate], reference: date, coarse_grid: TargetGrid, label: str, warnings: list[str], count: int = 1, allow_cloudy_fallback: bool = True) -> list[ScreenedCandidate]:

    if not candidates:
        warnings.append(f"No {label}-event Sentinel-2 acquisitions found in the search window below the cloud-cover limit.")
        return []
    
    nearest = sorted(candidates, key=lambda item: (abs((item.acquired_date - reference).days), item.cloud_cover_pct or 0.0))[:max(SCREEN_TOP_K, 2 * count)]
    screened = [ScreenedCandidate(candidate, *provider.screen(candidate, coarse_grid), abs((candidate.acquired_date - reference).days)) for candidate in nearest]
    ranked = rank_candidates(screened)
    if not ranked:
        warnings.append(f"All screened {label}-event acquisitions cover less than {MIN_VALID_FRACTION:.0%} of the AOI.")
        return []
    
    clear = [item for item in ranked if item.cloud_fraction <= CLOUDY_WARNING_FRACTION]
    if not clear and not allow_cloudy_fallback:
        warnings.append(f"The remaining {label}-event scenes are mostly cloud/shadow covered inside the AOI and were not used.")
        return []
    if not clear:
        best = ranked[0]
        warnings.append(f"Best {label}-event scene ({best.candidate.acquired_date}) is {best.cloud_fraction:.0%} cloud/shadow covered inside the AOI.")
        return [best]
    
    chosen = clear[:count]
    if count > 1 and len(chosen) < count:
        warnings.append(f"Only {len(chosen)} clear {label}-event scene(s) found, {count} were requested.")
    
    return sorted(chosen, key=lambda item: item.candidate.acquired_at)


def _save_scene(scene: FetchedScene, directory: Path, stem: str, disaster_type: str | None = None, thresholds: Thresholds | None = None) -> SceneMeta:
    
    # Stack the reflectance bands together with the cloud and valid masks into one array. 
    # This is the analysis file that the estimate_* tools read later.
    stack = np.concatenate([scene.reflectance, scene.cloud[None].astype(np.uint16), scene.valid[None].astype(np.uint16)])
    tif_path = write_geotiff(directory / f"{stem}.tif", stack, scene.grid.transform, scene.grid.crs, nodata=None, band_names=[*scene.bands, *MASK_BAND_NAMES])

    # true-colour preview from the reflectance as PNG
    png_path = directory / f"{stem}.png"
    preview = true_color(scene.reflectance, scene.bands, scene.valid)
    save_display_png(preview, scene.grid.transform, scene.grid.crs, png_path, max_side=DISPLAY_MAX_SIDE)
    save_display_png(preview, scene.grid.transform, scene.grid.crs, map_overlay_path(png_path), max_side=MAP_MAX_SIDE) # smaller copy for Folium map
   
    # The primary index layers of the disaster type are created together with the images.
    layers = {key: build_layer(METHODS[key], scene.reflectance, scene.bands, scene.valid, scene.cloud, scene.grid.transform, scene.grid.crs, directory, stem, thresholds or Thresholds())
              for key in AUTO_METHODS.get(disaster_type or "", ())}
    
    candidate = scene.candidate

    # Return metadata
    return SceneMeta(scene_id=",".join(candidate.scene_ids),
                     provider=candidate.provider,
                     collection=candidate.product,
                     acquired_at=candidate.acquired_at,
                     scene_cloud_cover_pct=candidate.cloud_cover_pct,
                     aoi_cloud_fraction=round(scene.cloud_fraction, 4),
                     aoi_valid_fraction=round(scene.valid_fraction, 4),
                     tif_path=str(tif_path),
                     png_path=str(png_path),
                     layers=layers)


# Distance (days) of two dates within a year (365 days), ignoring the year, so circular range of 0 to 182 (365/2= 182.5)
def season_offset_days(first: date, second: date) -> int:
    difference = abs(first.timetuple().tm_yday - second.timetuple().tm_yday)
    return min(difference, 365 - difference)


# One shared pre-event scene and up to request.max_pairs clear post-event dates for one AOI, as one pair per post-event date
def select_before_after(provider: ImageryProvider, request: ImageryRequest, artifacts_dir: Path, max_aoi_km: float, today: date | None = None, thresholds: Thresholds | None = None) -> list[ImageryPair]:
    today = today or utc_now().date()

    # ==============================================================================
    # 1. Validate the request
    # ==============================================================================

    # The AOI must not be larger than max_aoi_km, with 1% tolerance for rounding
    bbox = request.resolved_bbox()
    width_km, height_km = bbox.size_km()
    if max(width_km, height_km) > max_aoi_km * 1.01:  # a 10 km radius is a 20 km box, rounding must not reject it
        raise ValueError(f"AOI is {width_km:.1f} x {height_km:.1f} km, larger than the {max_aoi_km:.0f} km limit. Use a smaller bbox or radius.")

    # Sentinel-2 imagery does not exist before July 2015, so an earlier event or baseline date raises an error
    disaster_type = request.disaster_type
    grid = TargetGrid.for_bbox(bbox, resolution_for(disaster_type))
    coarse_grid = grid.coarsened()
    if request.baseline_date is not None and request.baseline_date < SENTINEL2_FIRST_DATE:
        raise NoImageryError(f"Sentinel-2 imagery does not exist before July 2015 (baseline date {request.baseline_date}). Choose a baseline from July 2015 on.")
    if request.event_start < SENTINEL2_FIRST_DATE:
        raise NoImageryError(f"Sentinel-2 imagery does not exist for events before July 2015 (event start {request.event_start}). Only the event information can be shown, no before/after images.")

    # If the post-event window hasn't started yet (the event is too recent), it raises an error
    pre_window, post_window, post_reference = event_windows(request, today)
    pre_window = (max(pre_window[0], SENTINEL2_FIRST_ACQUISITION), pre_window[1])
    
    archive_notes: list[str] = []  # added to the pairs, but not to the error message when nothing is found
    if request.event_start.year < SPARSE_COVERAGE_BEFORE_YEAR:
        archive_notes.append("Sentinel-2 coverage before 2018 is sparse (a single satellite until 2017, fewer L2A scenes), so the scenes may be further from the event and fewer dates are available.")
        if provider.name == "earth-search":
            archive_notes.append("The public Earth Search archive has very few scenes before 2018. CDSE credentials give a much better archive for older events.")
    if post_window[1] < post_window[0]:
        raise NoImageryError("The post-event window has not started yet, so no post-event imagery can exist.")

    # ==============================================================================
    # 2. Setup
    # ==============================================================================

    # The product and the bands depend on the disaster type. The warnings are shown in the UI above the images
    product = provider.product_for(disaster_type)
    bands = bands_for(disaster_type)
    warnings: list[str] = []

    # ==============================================================================
    # 3. Search for scenes
    # ==============================================================================

    # One catalogue search covers both windows. Same-day tiles are merged, then the scenes are split into pre and post candidates
    candidates = group_by_day(provider.search(bbox, pre_window[0], post_window[1], product, request.max_cloud_pct))
    pre_candidates = [item for item in candidates if pre_window[0] <= item.acquired_date <= pre_window[1]]
    post_candidates = [item for item in candidates if post_window[0] <= item.acquired_date <= post_window[1]]
    
    # Drop post-event dates that the user has already seen in an earlier answer
    skipped = [item for item in post_candidates if item.acquired_date in request.exclude_post_dates]
    if skipped:
        post_candidates = [item for item in post_candidates if item.acquired_date not in request.exclude_post_dates]
        warnings.append(f"Skipped {len(skipped)} post-event date(s) that were already shown.")
        if not post_candidates:
            raise NoImageryError("No further post-event dates beyond the ones already shown were found for this area and time window. Try a different area, a longer post_days window or another event.")
    
    # ==============================================================================
    # 4. Choose scenes
    # ==============================================================================

    # One pre-event scene closest to the event start. It is None if no usable scene was found
    pre = (choose_scenes(provider, pre_candidates, request.event_start, coarse_grid, "pre", warnings) or [None])[0]
    
    # Without exclusions a cloudy best scene is still better than nothing. When the user already saw the best dates only clear scenes are worth adding.
    posts = choose_scenes(provider, post_candidates, post_reference, coarse_grid, "post", warnings, request.max_pairs, allow_cloudy_fallback=not request.exclude_post_dates)
    if request.exclude_post_dates and not posts:
        raise NoImageryError("No further clear post-event dates beyond the ones already shown were found for this area and time window. Try a different area, a longer post_days window or another event.")
    
    # Without any scene there is nothing to show, the collected warnings explain why
    if pre is None and not posts:
        raise NoImageryError(" ".join(warnings) or "No usable imagery found.")

    # ==============================================================================
    # 5. Download and save the scenes
    # ==============================================================================

    # The archive notes go first. All pairs of this call share one group_id, which groups them in the UI
    warnings[:0] = archive_notes
    group_id = f"{disaster_type}-{request.event_start:%Y%m%d}-{uuid.uuid4().hex[:8]}"
    
    # One folder per pair. If there is no post-event scene, one folder is still needed for the pre-event scene
    pair_count = max(len(posts), 1)
    directories =[artifacts_dir / f"{group_id}-p{index + 1}" for index in range(pair_count)]
    
    # The primary index layers of the disaster type are created with the images, the event types without a method are images only.
    thresholds = thresholds or Thresholds()
    
    # The pre-event scene is downloaded once and saved in the folder of the first pair
    pre_meta = _save_scene(provider.fetch(pre.candidate, grid, bands), directories[0], "pre", disaster_type, thresholds) if pre else None
    
    # The post-event scenes are downloaded in parallel, then each is saved in its own pair folder
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        fetched = list(pool.map(lambda item: provider.fetch(item.candidate, grid, bands), posts))
    post_metas = [_save_scene(scene, directory, "post", disaster_type, thresholds) for scene, directory in zip(fetched, directories)] or [None]

    # ==============================================================================
    # 6. Warnings and provenance
    # ==============================================================================

    # Disaster types without an analysis method only get images
    if disaster_type not in ANALYSIS_DISASTER_TYPES:
        warnings.append(VISUAL_ONLY_NOTE.format(name=disaster_type))
        if disaster_type == "earthquake":
            warnings.append("Earthquake damage is usually not visible at 10 m resolution, so the images may show no obvious change.")
    notes = ["Pre/post scenes are single-day acquisitions (same-day tiles mosaicked).", "AOI cloud fraction comes from the Sentinel-2 L2A scene classification (SCL).",
             "All pairs of one call share the same pre-event scene and AOI."]
    provenance = Provenance(sources=[PROVIDER_SOURCES[provider.name], "Copernicus Sentinel-2 data"], coverage=bbox, uncertainty_notes=notes)
    
    # WGS84 bounds of the grid, used to place the images on the Folium map
    bounds = wgs84_bounds((grid.height, grid.width), grid.transform, grid.crs)

    # ==============================================================================
    # 7. Build the pairs
    # ==============================================================================

    pairs = []
    for index, (post_meta, directory) in enumerate(zip(post_metas, directories)):
        # Each pair gets its own copy of the warnings, so the seasonal warning only appears on the pair it belongs to
        pair_warnings = list(warnings)
        
        # The seasonal warning only makes sense if both scenes exist and the disaster type is analysed
        if disaster_type in ANALYSIS_DISASTER_TYPES and pre_meta is not None and post_meta is not None:
            offset = season_offset_days(pre_meta.acquired_at.date(), post_meta.acquired_at.date())
            if offset > SEASON_WARNING_DAYS:
                pair_warnings.append(f"The before scene ({pre_meta.acquired_at.date()}) and the after scene ({post_meta.acquired_at.date()}) are {offset} days apart in the season. "
                                     "Vegetation, snow, water and soil change with the season, so part of the difference may be seasonal and not an effect of the event.")
        
        pair = ImageryPair(pair_id=directory.name, group_id=group_id, pair_index=index + 1, event_name=request.event_name, disaster_type=disaster_type, event_start=request.event_start,
                           bbox=bbox, crs=grid.crs.to_string(), resolution_m=grid.resolution_m, provider=provider.name, product=product, bands=bands, pre=pre_meta, post=post_meta,
                           bounds_wgs84=bounds, warnings=pair_warnings, provenance=provenance)
        
        # metadata.json is what load_pair reads later to recover the pair from its pair_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "metadata.json").write_text(pair.model_dump_json(indent=2), encoding="utf-8")
        pairs.append(pair)
    
    return pairs


def load_pair(artifacts_dir: Path, pair_id: str) -> ImageryPair:
    
    if not pair_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("invalid pair_id")
    
    path = artifacts_dir / pair_id / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"unknown pair_id '{pair_id}'. Call select_before_after_imagery first.")
    
    return ImageryPair.model_validate_json(path.read_text(encoding="utf-8"))
