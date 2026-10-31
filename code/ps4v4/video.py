"""Sequential video reading in batches (decode order = frame index)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoMeta:
    path: str
    width: int
    height: int
    fps: float
    n_frames: int

    @property
    def diag(self) -> float:
        return float(np.hypot(self.width, self.height))


def video_meta(path: str, max_seconds: float | None = None) -> VideoMeta:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if max_seconds:
        n = min(n, int(round(max_seconds * fps)))
    return VideoMeta(path, w, h, float(fps), n)


def iter_batches(meta: VideoMeta, batch: int = 16, start: int = 0) -> Iterator[tuple[int, list[np.ndarray]]]:
    """Yield (first_frame_index, [BGR frames]) until meta.n_frames."""
    cap = cv2.VideoCapture(meta.path)
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    fi = start
    while fi < meta.n_frames:
        frames = []
        while len(frames) < batch and fi + len(frames) < meta.n_frames:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        if not frames:
            break
        yield fi, frames
        fi += len(frames)
    cap.release()


def iter_frames(meta: VideoMeta) -> Iterator[tuple[int, np.ndarray]]:
    for f0, frames in iter_batches(meta, 32):
        for k, f in enumerate(frames):
            yield f0 + k, f


def read_frame(meta: VideoMeta, index: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(meta.path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, f = cap.read()
    cap.release()
    return f if ok else None
