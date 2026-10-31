"""Hand detectors. Output per image: array (n, 6) = x1, y1, x2, y2, score, p_right (1 = right hand)."""
from __future__ import annotations

from typing import Protocol

import numpy as np


class HandDetector(Protocol):
    def detect(self, images: list[np.ndarray], imgsz: int, conf: float) -> list[np.ndarray]: ...


class WiLoRDetector:
    """WiLoR's YOLO hand detector (trained on 2M+ in-the-wild images, predicts handedness)."""

    def __init__(self, device: str = "cuda"):
        from .wilor_loader import load_wilor
        self.pipe = load_wilor(device)

    def detect(self, images, imgsz, conf):
        if not images:
            return []
        half = self.pipe._ps4v4_device.type == "cuda"
        res = self.pipe.hand_detector(images, conf=conf, imgsz=imgsz, verbose=False, half=half)
        out = []
        for r in res:
            b = r.boxes
            if b is None or len(b) == 0:
                out.append(np.zeros((0, 6)))
                continue
            out.append(np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None],
                                       b.cls.cpu().numpy()[:, None]], 1))
        return out


def make_detector(backend: str, device: str) -> HandDetector:
    if backend == "wilor":
        return WiLoRDetector(device)
    if backend == "fake":
        from .fake import FakeDetector
        return FakeDetector()
    raise ValueError(f"unknown detector backend {backend}")
