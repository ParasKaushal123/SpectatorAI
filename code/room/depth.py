"""Metric depth for virtual pinhole views. Backend: MoGe-2 (Microsoft, metric scale), or a test stand-in."""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


class MogeDepth:
    def __init__(self, device: str = "cuda", model: str = "Ruicheng/moge-2-vitl-normal", batch: int = 16,
                 resolution_level: int = 9):
        import torch
        from moge.model.v2 import MoGeModel
        self.torch = torch
        self.dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.model = MoGeModel.from_pretrained(model).to(self.dev).eval()
        self.batch, self.level = batch, resolution_level

    def __call__(self, images: list[np.ndarray], fov_deg: float, poses=None, clip=None) -> list[np.ndarray]:
        """images: RGB uint8 (S,S,3). Returns metric depth (S,S) float32, 0 where invalid."""
        torch = self.torch
        out = []
        for i in range(0, len(images), self.batch):
            x = torch.from_numpy(np.stack(images[i:i + self.batch])).to(self.dev).permute(0, 3, 1, 2).float() / 255
            with torch.no_grad():
                r = self.model.infer(x, fov_x=fov_deg, resolution_level=self.level,
                                     use_fp16=self.dev.type == "cuda")
            d = r["depth"].float().cpu().numpy()
            d[~np.isfinite(d)] = 0
            out += list(d.astype(np.float32))
        return out


class CallableDepth:
    """Test stand-in: fn(images, fov_deg, poses) -> depths."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, images, fov_deg, poses=None, clip=None):
        return self.fn(images, fov_deg, poses, clip)
