"""
Prithvi-EO-2.0-300M-BurnScars (ibm-nasa-geospatial) inference through terratorch.

Mirrors the official inference.py shipped with the checkpoint:
  LightningInferenceModel.from_config, reflect-pad to 512 px windows, datamodule test_transform and aug (normalization), argmax.
Pretrained weights are loaded unchanged. Nothing is trained here.
"""

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml

MODEL_ID = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars"
BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]  # HLS BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2
IMG_SIZE = 512
CLASS_NOT_BURNED, CLASS_BURNED = 0, 1


def write_inference_config(source_config: Path, target_config: Path) -> Path:
    """Copy the upstream config with backbone_pretrained disabled.

    The fine-tuned checkpoint already contains the backbone weights.
    Without this, terratorch would download another ~1.2 GB copy of the base Prithvi weights.
    """
    config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    config["model"]["init_args"]["model_args"]["backbone_pretrained"] = False
    config["data"]["init_args"]["num_workers"] = 0
    target_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return target_config


class PrithviBurnScarModel:
    def __init__(self, config_path: Path, checkpoint_path: Path, device: torch.device, half_precision: bool = False):
        from terratorch.cli_tools import LightningInferenceModel

        inference_config = write_inference_config(config_path, config_path.with_name("burn_scars_config_inference.yaml"))
        self._lightning = LightningInferenceModel.from_config(str(inference_config), str(checkpoint_path))
        self.model = self._lightning.model
        self.datamodule = self._lightning.datamodule
        self.device = device
        self.model.to(device).eval()
        self.half_precision = half_precision and device.type == "cuda"

    @torch.no_grad()
    def predict(self, reflectance: np.ndarray) -> np.ndarray:
        """(6, H, W) surface reflectance in 0..1 (nodata = 0) to (H, W) uint8 classes {0: not burned, 1: burn scar}."""
        if reflectance.shape[0] != len(BANDS):
            raise ValueError(f"expected {len(BANDS)} bands ordered {BANDS}")
        height, width = reflectance.shape[-2:]
        pad_h = (IMG_SIZE - height % IMG_SIZE) % IMG_SIZE
        pad_w = (IMG_SIZE - width % IMG_SIZE) % IMG_SIZE
        padded = np.pad(reflectance.astype(np.float32), ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
        output = np.zeros(padded.shape[-2:], dtype=np.uint8)
        autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if self.half_precision else nullcontext()
        for row in range(0, padded.shape[1], IMG_SIZE):
            for col in range(0, padded.shape[2], IMG_SIZE):
                window = padded[:, row:row + IMG_SIZE, col:col + IMG_SIZE]
                sample = self.datamodule.test_transform(image=window.transpose(1, 2, 0))
                sample["image"] = sample["image"].unsqueeze(0)
                batch = self.datamodule.aug(sample)["image"].to(self.device)
                with autocast:
                    logits = self.model(batch).output
                output[row:row + IMG_SIZE, col:col + IMG_SIZE] = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        return output[:height, :width]
