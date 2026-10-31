"""Fake models for tests: coloured blobs stand in for hands.

Green blob = left hand, blue = right hand, red = junk (a static object), yellow = another
person's hand (small, far, arm from the top). The fake detector misses hands, flips labels and
fires on junk and on the other person's hand with hand-like scores. The fake 3D model reads the
blob colour to pose the hand like the real thing would look: the wearer's forearms run down and
outwards toward the camera, the other person's arm comes from the top and runs away, junk gets a
random pose.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..geometry import FisheyeCamera

CANON = np.array([[0, 0, 0], [.02, -.02, 0], [.035, -.04, 0], [.045, -.06, 0], [.05, -.08, 0], [.02, -.07, 0],
                  [.022, -.1, 0], [.023, -.12, 0], [.024, -.14, 0], [0, -.075, 0], [0, -.105, 0], [0, -.13, 0],
                  [0, -.15, 0], [-.02, -.07, 0], [-.02, -.1, 0], [-.02, -.12, 0], [-.02, -.135, 0],
                  [-.04, -.06, 0], [-.04, -.08, 0], [-.04, -.095, 0], [-.04, -.11, 0]]) + [0, 0.075, 0]


def _masks(img: np.ndarray) -> dict:
    b, g, r = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
    return {0: (g > 110) & (g > r + 60) & (g > b + 60),
            1: (b > 110) & (b > r + 60) & (b > g + 60),
            2: (r > 110) & (r > g + 60) & (r > b + 60),
            3: (r > 150) & (g > 150) & (b < 80)}


def colour_blobs(img: np.ndarray, min_area: int = 120) -> list[tuple[float, float, float, float, int]]:
    """(x1,y1,x2,y2,kind) with kind 0=left/green, 1=right/blue, 2=junk/red, 3=other person/yellow."""
    masks = _masks(img)
    out = []
    for kind, m in masks.items():
        n, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
        for k in range(1, n):
            x, y, w, h, a = st[k]
            if a >= min_area:
                out.append((x, y, x + w, y + h, kind))
    return out


class FakeDetector:
    def __init__(self, miss=0.25, flip=0.12, junk_score=(0.45, 0.85)):
        self.miss, self.flip, self.junk_score = miss, flip, junk_score

    def detect(self, images, imgsz, conf):
        out = []
        for img in images:
            rng = np.random.default_rng(int(img[::17, ::17].sum()) % (2 ** 32))
            rows = []
            for x1, y1, x2, y2, kind in colour_blobs(img, min_area=60):
                if kind == 3:
                    s, cls = rng.uniform(0.5, 0.9), float(rng.random() > 0.5)
                elif kind == 2:
                    s, cls = rng.uniform(*self.junk_score), float(rng.random() > 0.5)
                else:
                    if rng.random() < self.miss:
                        continue
                    s, cls = rng.uniform(0.35, 0.95), float(kind)
                    if rng.random() < self.flip:
                        cls = 1 - cls
                if s >= conf:
                    j = rng.normal(0, 3, 4)
                    rows.append([x1 + j[0], y1 + j[1], x2 + j[2], y2 + j[3], s, cls])
            out.append(np.array(rows, float).reshape(-1, 6))
        return out


def _rot(axis: str, deg: float) -> np.ndarray:
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class FakeReconstructor:
    faces = None

    def reconstruct(self, meta, cam: FisheyeCamera, requests: dict) -> dict:
        from ..video import iter_frames
        jobs = {}
        for side, (frames, boxes) in requests.items():
            for i, (f, b) in enumerate(zip(frames, boxes)):
                jobs.setdefault(int(f), []).append((side, i, b))
        out = {s: {"frames": np.asarray(fr), "joints": np.full((len(fr), 21, 3), np.nan),
                   "kp2d": np.full((len(fr), 21, 2), np.nan), "valid": np.zeros(len(fr), bool)}
               for s, (fr, _) in requests.items()}
        if not jobs:
            out["faces"] = None
            return out
        last = max(jobs)
        for f, img in iter_frames(meta):
            if f > last:
                break
            for side, i, b in jobs.get(f, []):
                x1, y1 = int(max(b[0], 0)), int(max(b[1], 0))
                x2, y2 = int(min(b[2], meta.width)), int(min(b[3], meta.height))
                crop = img[y1:y2, x1:x2]
                kind = 2
                if crop.size:
                    cnt = {k: int(m.sum()) for k, m in _masks(crop).items()}
                    if max(cnt.values()) > 0:
                        kind = max(cnt, key=cnt.get)
                rng = np.random.default_rng(f * 7 + i)
                c = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
                side_px = max(b[2] - b[0], b[3] - b[1])
                z = 0.19 * cam.fx / max(side_px, 1.0)
                ray = cam.rays(c)
                pos = ray / ray[2] * z
                hand = CANON * (np.array([-1, 1, 1]) if side == "left" else 1)
                if kind in (0, 1):                                      # forearm points to its own shoulder,
                    sh = np.array([0.2 if kind == 0 else 0.8, 1.6]) * [meta.width, meta.height]   # below the image
                    d = sh - c
                    phi = np.degrees(np.arctan2(-d[0], d[1]))
                    R = _rot("z", phi) @ _rot("x", -30)                 # ... and runs back toward the camera
                elif kind == 3:
                    R = _rot("z", 180 + rng.normal(0, 10)) @ _rot("x", 50)   # arm from the top, running away
                else:
                    R = _rot("z", rng.uniform(0, 360)) @ _rot("x", rng.uniform(-80, 80))
                Jh = (hand - hand.mean(0)) @ R.T
                J = Jh + pos + rng.normal(0, 0.004, (21, 3))
                J[:, 2] += rng.normal(0, 0.02)                           # depth noise
                if rng.random() < 0.02:
                    J += [0, 0, 0.5]                                     # occasional wild depth outlier
                out[side]["joints"][i] = J
                out[side]["kp2d"][i] = cam.project(J)
                out[side]["valid"][i] = True
        out["faces"] = None
        return out
