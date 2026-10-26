"""
ML4Floods WorldFloods v2 multi-output UNet (isp-uv-es/ml4floods, WF2_unetv2_bgriswirs), re-implemented in plain PyTorch.

The architecture, normalization, padding and post-processing mirror the ml4floods package:
  ml4floods/models/architectures/unets.py, layer_factory.double_conv
  ml4floods/models/model_setup.py (normalization, padded_predict, tiling)
  ml4floods/models/postprocess.py get_pred_mask_v2, ml4floods/data/create_gt.py get_brightness
Pretrained weights are loaded unchanged with strict key matching. Nothing is trained here.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn

MODEL_ID = "isp-uv-es/ml4floods/models/WF2_unetv2_bgriswirs"
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]

# SENTINEL2_NORMALIZATION rows (mean, std) in raw L1C digital numbers for B2, B3, B4, B8, B11, B12.
NORMALIZATION_MEAN = np.array([3758.07467509, 3238.08247208, 3418.90147615, 3981.96268494, 2391.66101119, 1790.32497137], dtype=np.float32)
NORMALIZATION_STD = np.array([2794.09579088, 2549.4940614, 2811.78109878, 2500.47885249, 1500.02533014, 1241.9817628], dtype=np.float32)
BRIGHTNESS_THRESHOLD = 3500.0
MODULE_SHAPE = 8
TILE_SIZE = 512
TILE_OVERLAP = 32

CLASS_INVALID, CLASS_LAND, CLASS_WATER, CLASS_CLOUD = 0, 1, 2, 3


def double_conv(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, n_channels: int, n_class: int):
        super().__init__()
        self.dconv_down1 = double_conv(n_channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)
        self.maxpool = nn.MaxPool2d(2)
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        x = self.dconv_down4(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self.dconv_up3(torch.cat([x, conv3], dim=1))
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self.dconv_up2(torch.cat([x, conv2], dim=1))
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self.dconv_up1(torch.cat([x, conv1], dim=1))
        return self.conv_last(x)


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # model.pt is the state dict of the Lightning module, which holds the UNet as `network`.
    return {key.removeprefix("network."): value for key, value in state.items()}


class ML4FloodsModel:
    def __init__(self, weights_path: Path, device: torch.device):
        self.device = device
        self.network = UNet(len(BANDS), 2)
        self.network.load_state_dict(load_state_dict(weights_path), strict=True)
        self.network.to(device).eval()
        self._mean = torch.from_numpy(NORMALIZATION_MEAN)[None, :, None, None].to(device)
        self._std = torch.from_numpy(NORMALIZATION_STD)[None, :, None, None].to(device)

    @torch.no_grad()
    def _predict_padded(self, tile: torch.Tensor) -> torch.Tensor:
        height, width = tile.shape[-2:]
        pad_bottom = (-height) % MODULE_SHAPE
        pad_right = (-width) % MODULE_SHAPE
        if pad_bottom or pad_right:
            tile = nn.functional.pad(tile, (0, pad_right, 0, pad_bottom), mode="reflect")
        normalized = (tile - self._mean) / (self._std + 1e-6)
        return torch.sigmoid(self.network(normalized))[..., :height, :width]

    @torch.no_grad()
    def predict_probabilities(self, digital_numbers: np.ndarray) -> np.ndarray:
        """(6, H, W) L1C-like DN (reflectance * 10000) to (2, H, W) probabilities: [cloud, water]."""
        if digital_numbers.shape[0] != len(BANDS):
            raise ValueError(f"expected {len(BANDS)} bands ordered {BANDS}")
        image = torch.from_numpy(digital_numbers.astype(np.float32))[None].to(self.device)
        _, _, height, width = image.shape
        output = torch.zeros((2, height, width), dtype=torch.float32)
        for row in range(0, height, TILE_SIZE):
            for col in range(0, width, TILE_SIZE):
                # Predict each tile with surrounding context and keep only its centre, as ml4floods predbytiles does.
                row0, col0 = max(row - TILE_OVERLAP, 0), max(col - TILE_OVERLAP, 0)
                row1, col1 = min(row + TILE_SIZE + TILE_OVERLAP, height), min(col + TILE_SIZE + TILE_OVERLAP, width)
                prediction = self._predict_padded(image[..., row0:row1, col0:col1])[0].cpu()
                tile_height, tile_width = min(TILE_SIZE, height - row), min(TILE_SIZE, width - col)
                output[:, row:row + tile_height, col:col + tile_width] = prediction[:, row - row0:row - row0 + tile_height, col - col0:col - col0 + tile_width]
        return output.numpy()


def classify(digital_numbers: np.ndarray, probabilities: np.ndarray, th_water: float = 0.5, th_cloud: float = 0.5) -> np.ndarray:
    """get_pred_mask_v2: {0: invalid, 1: land, 2: water, 3: cloud}. Clouds need high probability and high brightness."""
    output = np.full(probabilities.shape[-2:], CLASS_LAND, dtype=np.uint8)
    output[probabilities[1] > th_water] = CLASS_WATER
    rgb = digital_numbers[[BANDS.index("B04"), BANDS.index("B03"), BANDS.index("B02")]].astype(np.float32)
    brightness = np.sqrt(np.sum(rgb**2, axis=0))
    output[(probabilities[0] > th_cloud) & (brightness > BRIGHTNESS_THRESHOLD)] = CLASS_CLOUD
    output[np.all(digital_numbers == 0, axis=0)] = CLASS_INVALID
    return output
