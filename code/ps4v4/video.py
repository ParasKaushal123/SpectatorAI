"""Sequential video reading in batches (decode order = frame index)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Iterator

import cv2
import numpy as np

_frames: dict[tuple, np.ndarray] = {}
_cache_lock = threading.Lock()


def _key(meta):
    p = Path(meta.path).resolve()
    st = p.stat()
    return str(p), st.st_size, st.st_mtime_ns, meta.n_frames


def cache_video(meta, max_bytes: int = 8 * 1024**3) -> bool:
    """Opt-in full decode, bounded to 8 GiB. Cached frames are read-only."""
    key = _key(meta)
    with _cache_lock:
        if key in _frames:
            return True
        if meta.n_frames * meta.height * meta.width * 3 > max_bytes:
            return False
        try:
            frames = np.empty((meta.n_frames, meta.height, meta.width, 3), np.uint8)
        except MemoryError:
            return False
        cap = cv2.VideoCapture(meta.path)
        try:
            for i in range(meta.n_frames):
                ok, frame = cap.read()
                if not ok:
                    return False
                frames[i] = frame
        finally:
            cap.release()
        frames.flags.writeable = False
        _frames[key] = frames
    return True


def clear_video_cache(path: str) -> None:
    with _cache_lock:
        for key in list(_frames):
            if key[0] == str(Path(path).resolve()):
                del _frames[key]


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
    cached = _frames.get(_key(meta))
    if cached is not None:
        for fi in range(start, meta.n_frames, batch):
            yield fi, list(cached[fi:fi + batch])
        return
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
    cached = _frames.get(_key(meta))
    if cached is not None:
        return cached[index] if 0 <= index < len(cached) else None
    cap = cv2.VideoCapture(meta.path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, f = cap.read()
    cap.release()
    return f if ok else None
