"""
Checkpoint locations, availability checks and lazy, cached model loading.
"""

import json
import threading
from dataclasses import dataclass
from pathlib import Path

ML4FLOODS_REPO = "isp-uv-es/ml4floods"
ML4FLOODS_SUBFOLDER = "models/WF2_unetv2_bgriswirs"
ML4FLOODS_FILES = ["model.pt", "config.json"]
PRITHVI_REPO = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars"
PRITHVI_FILES = ["Prithvi_EO_V2_300M_BurnScars.pt", "burn_scars_config.yaml", "config.json", "inference.py"]


class ModelUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckpointPaths:
    ml4floods_weights: Path
    prithvi_checkpoint: Path
    prithvi_config: Path
    manifest: Path

    @classmethod
    def under(cls, models_dir: Path) -> "CheckpointPaths":
        return cls(ml4floods_weights=models_dir / "ml4floods" / ML4FLOODS_SUBFOLDER / "model.pt",
                   prithvi_checkpoint=models_dir / "prithvi" / PRITHVI_FILES[0],
                   prithvi_config=models_dir / "prithvi" / PRITHVI_FILES[1],
                   manifest=models_dir / "manifest.json")

    def revision(self, repo_id: str) -> str | None:
        if not self.manifest.exists():
            return None
        return json.loads(self.manifest.read_text(encoding="utf-8")).get(repo_id, {}).get("revision")

    def status(self) -> dict[str, bool]:
        return {"ml4floods": self.ml4floods_weights.exists(), "prithvi": self.prithvi_checkpoint.exists() and self.prithvi_config.exists()}


def resolve_device(preference: str):
    import torch

    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(preference)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ModelUnavailableError("TORCH_DEVICE requests CUDA but CUDA is not available")
    return device


class ModelRegistry:
    """Loads each model at most once per process. Loading is lazy so the MCP server starts quickly."""

    def __init__(self, paths: CheckpointPaths, device_preference: str = "auto", prithvi_half_precision: bool = False):
        self.paths = paths
        self._device_preference = device_preference
        self._prithvi_half_precision = prithvi_half_precision
        self._models: dict[str, object] = {}
        self._lock = threading.Lock()

    @property
    def device(self):
        return resolve_device(self._device_preference)

    def ml4floods(self):
        with self._lock:
            if "ml4floods" not in self._models:
                if not self.paths.ml4floods_weights.exists():
                    raise ModelUnavailableError(f"ML4Floods checkpoint missing at {self.paths.ml4floods_weights}. Run scripts/download_models.py.")
                from eo_agent.models.ml4floods import ML4FloodsModel

                try:
                    self._models["ml4floods"] = ML4FloodsModel(self.paths.ml4floods_weights, self.device)
                except Exception as exc:
                    raise ModelUnavailableError(f"ML4Floods checkpoint could not be loaded: {exc}") from exc
            return self._models["ml4floods"]

    def prithvi(self):
        with self._lock:
            if "prithvi" not in self._models:
                if not (self.paths.prithvi_checkpoint.exists() and self.paths.prithvi_config.exists()):
                    raise ModelUnavailableError(f"Prithvi checkpoint or config missing in {self.paths.prithvi_checkpoint.parent}. Run scripts/download_models.py.")
                try:
                    from eo_agent.models.prithvi import PrithviBurnScarModel

                    self._models["prithvi"] = PrithviBurnScarModel(self.paths.prithvi_config, self.paths.prithvi_checkpoint, self.device, self._prithvi_half_precision)
                except ImportError as exc:
                    raise ModelUnavailableError(f"terratorch is not installed or failed to import: {exc}") from exc
                except Exception as exc:
                    raise ModelUnavailableError(f"Prithvi checkpoint could not be loaded: {exc}") from exc
            return self._models["prithvi"]
