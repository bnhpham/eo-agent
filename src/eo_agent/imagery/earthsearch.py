"""
Element84 Earth Search provider: public Sentinel-2 L2A Cloud-Optimized GeoTIFFs on AWS, no credentials required.

Used when CDSE credentials are missing or CDSE fails.
"""

from datetime import date, datetime

import numpy as np
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from eo_agent.imagery.base import REFLECTANCE_SCALE, FetchedScene, Product, ProviderError, SceneCandidate, TargetGrid
from eo_agent.imagery.scl import scl_masks
from eo_agent.schemas import BBox

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
ASSET_KEYS = {"B02": "blue", "B03": "green", "B04": "red", "B08": "nir", "B8A": "nir08", "B11": "swir16", "B12": "swir22", "SCL": "scl"}
GDAL_ENV = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "AWS_NO_SIGN_REQUEST": "YES", "GDAL_HTTP_MAX_RETRY": "3", "GDAL_HTTP_RETRY_DELAY": "1",
            "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif", "GDAL_HTTP_MULTIPLEX": "YES", "VSI_CACHE": "TRUE"}


class EarthSearchProvider:
    name = "earth-search"

    def __init__(self, url: str = EARTH_SEARCH_URL):
        self._url = url

    def product_for(self, disaster_type: str) -> Product:
        return "sentinel-2-l2a"

    def search(self, bbox: BBox, start: date, end: date, product: Product, max_cloud_pct: float) -> list[SceneCandidate]:
        try:
            client = Client.open(self._url)
            search = client.search(collections=[COLLECTION], bbox=bbox.as_list(), datetime=f"{start.isoformat()}T00:00:00Z/{end.isoformat()}T23:59:59Z",
                                   query={"eo:cloud_cover": {"lte": max_cloud_pct}}, max_items=300)
            items = list(search.items())
        except Exception as exc:
            raise ProviderError(f"Earth Search catalog request failed: {exc}") from exc
        candidates = []
        for item in items:
            acquired_at = item.datetime or datetime.fromisoformat(item.properties["datetime"].replace("Z", "+00:00"))
            candidates.append(SceneCandidate(provider="earth-search", product="sentinel-2-l2a", acquired_at=acquired_at, scene_ids=[item.id],
                                             cloud_cover_pct=item.properties.get("eo:cloud_cover"), payload=[item]))
        return candidates

    @staticmethod
    def _read_asset(item, band: str, grid: TargetGrid, resampling: Resampling) -> tuple[np.ndarray, float, float]:
        asset = item.assets.get(ASSET_KEYS[band])
        if asset is None:
            raise ProviderError(f"scene {item.id} has no asset for band {band}")
        raster_bands = asset.extra_fields.get("raster:bands") or [{}]
        scale = float(raster_bands[0].get("scale", 1.0))
        offset = float(raster_bands[0].get("offset", 0.0))
        # Earth Search already removed the baseline 04.00+ BOA offset from these COGs but still advertises it in raster:bands.
        if item.properties.get("earthsearch:boa_offset_applied"):
            offset = 0.0
        with rasterio.open(asset.href) as source, WarpedVRT(source, crs=grid.crs, transform=grid.transform, width=grid.width, height=grid.height, resampling=resampling, src_nodata=0, nodata=0) as vrt:
            return vrt.read(1), scale, offset

    def _scl(self, items: list, grid: TargetGrid) -> np.ndarray:
        scl = np.zeros((grid.height, grid.width), dtype=np.uint8)
        for item in items:
            tile_scl, _, _ = self._read_asset(item, "SCL", grid, Resampling.nearest)
            empty = scl == 0
            scl[empty] = tile_scl[empty]
        return scl

    def screen(self, candidate: SceneCandidate, grid: TargetGrid) -> tuple[float, float]:
        try:
            with rasterio.Env(**GDAL_ENV):
                scl = self._scl(candidate.payload, grid)
        except rasterio.RasterioIOError as exc:
            raise ProviderError(f"Earth Search read failed: {exc}") from exc
        cloud, valid = scl_masks(scl)
        count = int(valid.sum())
        return (float(cloud.sum() / count) if count else 1.0), float(valid.mean())

    def fetch(self, candidate: SceneCandidate, grid: TargetGrid, bands: list[str]) -> FetchedScene:
        reflectance = np.zeros((len(bands), grid.height, grid.width), dtype=np.uint16)
        scl = np.zeros((grid.height, grid.width), dtype=np.uint8)
        try:
            with rasterio.Env(**GDAL_ENV):
                # Mosaic same-day tiles: a pixel is filled by the first tile that has data for it.
                for item in candidate.payload:
                    tile_scl, _, _ = self._read_asset(item, "SCL", grid, Resampling.nearest)
                    fill = (scl == 0) & (tile_scl != 0)
                    if not fill.any():
                        continue
                    scl[fill] = tile_scl[fill]
                    for index, band in enumerate(bands):
                        digital_numbers, scale, offset = self._read_asset(item, band, grid, Resampling.bilinear)
                        surface_reflectance = digital_numbers.astype(np.float32) * scale + offset
                        scaled = np.clip(np.round(surface_reflectance * REFLECTANCE_SCALE), 1, 65535).astype(np.uint16)
                        reflectance[index][fill] = np.where(digital_numbers[fill] > 0, scaled[fill], 0)
        except rasterio.RasterioIOError as exc:
            raise ProviderError(f"Earth Search read failed: {exc}") from exc
        cloud, valid = scl_masks(scl)
        valid &= np.all(reflectance > 0, axis=0)
        reflectance[:, ~valid] = 0
        return FetchedScene(candidate=candidate, grid=grid, bands=bands, reflectance=reflectance, cloud=cloud & valid, valid=valid)
