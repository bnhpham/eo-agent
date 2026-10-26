"""
Copernicus Data Space Ecosystem (CDSE) provider: Sentinel Hub Catalog API (search) and Process API (analysis-grade bands).

Uses the same OAuth client credentials as the ESA EVE CDSE MCP server.
Process API DN values are harmonized by Sentinel Hub, so DN = 10000 * reflectance across processing baselines.
"""

import threading
import time
from datetime import date, datetime

import numpy as np
import requests
from rasterio.io import MemoryFile

from eo_agent.imagery.base import FetchedScene, Product, ProviderError, SceneCandidate, TargetGrid
from eo_agent.imagery.scl import scl_masks
from eo_agent.schemas import BBox

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CATALOG_URL = "https://sh.dataspace.copernicus.eu/catalog/v1/search"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"
MAX_PROCESS_PX = 2500
TIMEOUT_S = 120


def _band_evalscript(bands: list[str], sample_type: str) -> str:
    inputs = ", ".join(f'"{band}"' for band in bands)
    outputs = ", ".join(f"s.{band}" for band in bands)
    return ("//VERSION=3\n"
            f"function setup() {{ return {{ input: [{{ bands: [{inputs}], units: \"DN\" }}], output: {{ bands: {len(bands)}, sampleType: \"{sample_type}\" }} }}; }}\n"
            f"function evaluatePixel(s) {{ return [{outputs}]; }}\n")


class CdseProvider:
    name = "cdse"

    def __init__(self, client_id: str, client_secret: str, session: requests.Session | None = None):
        self._client_id = client_id
        self._client_secret = client_secret
        self._session = session or requests.Session()
        self._token: str | None = None
        self._token_expiry = 0.0
        self._lock = threading.Lock()

    def product_for(self, disaster_type: str) -> Product:
        # NDVI and true-colour images use surface reflectance (L2A). 
        return "sentinel-2-l2a"

    def _access_token(self) -> str:
        with self._lock:
            if self._token and time.monotonic() < self._token_expiry - 60:
                return self._token
            try:
                response = self._session.post(TOKEN_URL, data={"grant_type": "client_credentials", "client_id": self._client_id, "client_secret": self._client_secret}, timeout=30)
            except requests.RequestException as exc:
                raise ProviderError(f"CDSE authentication request failed: {exc}") from exc
            if response.status_code != 200:
                raise ProviderError(f"CDSE authentication failed (HTTP {response.status_code}). Check CDSE_CLIENT_ID and CDSE_CLIENT_SECRET.")
            payload = response.json()
            self._token = payload["access_token"]
            self._token_expiry = time.monotonic() + float(payload.get("expires_in", 600))
            return self._token

    def _post(self, url: str, body: dict, accept: str = "application/geo+json") -> requests.Response:
        headers = {"Authorization": f"Bearer {self._access_token()}", "Accept": accept}
        try:
            response = self._session.post(url, json=body, headers=headers, timeout=TIMEOUT_S)
        except requests.RequestException as exc:
            raise ProviderError(f"CDSE request failed: {exc}") from exc
        if response.status_code == 429:
            raise ProviderError("CDSE rate limit or monthly quota exceeded (HTTP 429)")
        if response.status_code >= 400:
            raise ProviderError(f"CDSE API error HTTP {response.status_code}: {response.text[:300]}")
        return response

    def search(self, bbox: BBox, start: date, end: date, product: Product, max_cloud_pct: float) -> list[SceneCandidate]:
        # Cloud screening relies on the L2A scene classification, so candidates always come from the L2A catalog.
        body = {"bbox": bbox.as_list(), "datetime": f"{start.isoformat()}T00:00:00Z/{end.isoformat()}T23:59:59Z", "collections": ["sentinel-2-l2a"], "limit": 100,
                "filter": {"op": "<=", "args": [{"property": "eo:cloud_cover"}, max_cloud_pct]}, "filter-lang": "cql2-json"}
        candidates: list[SceneCandidate] = []
        for _ in range(10):
            payload = self._post(CATALOG_URL, body).json()
            for feature in payload.get("features", []):
                properties = feature.get("properties", {})
                acquired_at = datetime.fromisoformat(properties["datetime"].replace("Z", "+00:00"))
                candidates.append(SceneCandidate(provider="cdse", product=product, acquired_at=acquired_at, scene_ids=[feature["id"]], cloud_cover_pct=properties.get("eo:cloud_cover")))
            next_token = payload.get("context", {}).get("next")
            if next_token is None:
                break
            body["next"] = next_token
        return candidates

    def _process(self, collection: str, grid: TargetGrid, day: date, bands: list[str], sample_type: str) -> np.ndarray:
        if grid.width > MAX_PROCESS_PX or grid.height > MAX_PROCESS_PX:
            raise ProviderError(f"AOI grid {grid.width}x{grid.height} px exceeds the Process API limit of {MAX_PROCESS_PX} px")
        left, bottom, right, top = grid.bounds
        body = {"input": {"bounds": {"bbox": [left, bottom, right, top], "properties": {"crs": f"http://www.opengis.net/def/crs/EPSG/0/{grid.crs.to_epsg()}"}},
                          "data": [{"type": collection,
                                    "dataFilter": {"timeRange": {"from": f"{day.isoformat()}T00:00:00Z", "to": f"{day.isoformat()}T23:59:59Z"}, "mosaickingOrder": "leastCC"},
                                    "processing": {"harmonizeValues": True, "upsampling": "BILINEAR"}}]},
                "output": {"width": grid.width, "height": grid.height, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
                "evalscript": _band_evalscript(bands, sample_type)}
        response = self._post(PROCESS_URL, body, accept="image/tiff")
        with MemoryFile(response.content) as memfile, memfile.open() as dataset:
            return dataset.read()

    def screen(self, candidate: SceneCandidate, grid: TargetGrid) -> tuple[float, float]:
        """(cloud_fraction, valid_fraction) of the AOI from the L2A scene classification."""
        scl = self._process("sentinel-2-l2a", grid, candidate.acquired_date, ["SCL"], "UINT8")[0]
        cloud, valid = scl_masks(scl)
        count = int(valid.sum())
        return (float(cloud.sum() / count) if count else 1.0), float(valid.mean())

    def fetch(self, candidate: SceneCandidate, grid: TargetGrid, bands: list[str]) -> FetchedScene:
        day = candidate.acquired_date
        if candidate.product == "sentinel-2-l2a":
            stack = self._process("sentinel-2-l2a", grid, day, [*bands, "SCL", "dataMask"], "UINT16")
            reflectance, scl, data_mask = stack[: len(bands)], stack[len(bands)], stack[len(bands) + 1]
        else:
            stack = self._process("sentinel-2-l1c", grid, day, [*bands, "dataMask"], "UINT16")
            reflectance, data_mask = stack[: len(bands)], stack[len(bands)]
            scl = self._process("sentinel-2-l2a", grid, day, ["SCL"], "UINT8")[0]
        cloud, valid = scl_masks(scl)
        # Harmonisation clips deep clear water to 0 in some bands. Only pixels without any signal are no data, like in the Earth Search provider.
        valid &= data_mask.astype(bool) & np.any(reflectance > 0, axis=0)
        reflectance = np.where(valid, np.maximum(reflectance, 1), 0).astype(np.uint16)
        return FetchedScene(candidate=candidate, grid=grid, bands=bands, reflectance=reflectance, cloud=cloud & valid, valid=valid)
