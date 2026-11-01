"""Camera models used by the world step (numpy, OpenCV axes: x right, y down, z forward)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MeiCamera:
    """Unified (Mei) camera, ViPE's model for wide-angle / fisheye video: intrinsics [fx, fy, cx, cy, xi]."""
    fx: float
    fy: float
    cx: float
    cy: float
    xi: float
    width: int
    height: int

    def scaled(self, width: int, height: int) -> "MeiCamera":
        sx, sy = width / self.width, height / self.height
        return MeiCamera(self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy, self.xi, width, height)

    @property
    def f_centre(self) -> float:
        """Focal length that matches a pinhole/equidistant model near the image centre."""
        return float(np.sqrt(self.fx * self.fy) / (1.0 + self.xi))

    def project(self, P: np.ndarray) -> np.ndarray:
        P = np.asarray(P, float)
        n = np.linalg.norm(P, axis=-1)
        d = np.maximum(P[..., 2] + self.xi * n, 1e-9)
        return np.stack([self.fx * P[..., 0] / d + self.cx, self.fy * P[..., 1] / d + self.cy], -1)

    def rays(self, uv: np.ndarray) -> np.ndarray:
        """Unit viewing rays; NaN where the pixel is outside the model's valid domain."""
        uv = np.asarray(uv, float)
        u = (uv[..., 0] - self.cx) / self.fx
        v = (uv[..., 1] - self.cy) / self.fy
        r2 = u * u + v * v
        disc = 1 + (1 - self.xi ** 2) * r2
        with np.errstate(invalid="ignore"):
            q = np.sqrt(disc)
        factor = (self.xi + q) / (1 + r2)
        P = np.stack([u * factor, v * factor, factor - self.xi], -1)
        P = P / np.maximum(np.linalg.norm(P, axis=-1, keepdims=True), 1e-12)
        P[disc < 0] = np.nan
        return P


@dataclass
class EquidistantCamera:
    """PS4's camera (Kannala-Brandt with k = 0): r = f * theta."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_ps4(cls, d: dict) -> "EquidistantCamera":
        return cls(d["fx"], d["fy"], d["cx"], d["cy"], int(d["width"]), int(d["height"]))

    @property
    def f_centre(self) -> float:
        return float(np.sqrt(self.fx * self.fy))

    def rays(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, float)
        mx, my = (uv[..., 0] - self.cx) / self.fx, (uv[..., 1] - self.cy) / self.fy
        th = np.hypot(mx, my)
        s = np.where(th > 1e-12, np.sin(th) / np.maximum(th, 1e-12), 1.0)
        return np.stack([mx * s, my * s, np.cos(th)], -1)

    def project(self, P: np.ndarray) -> np.ndarray:
        P = np.asarray(P, float)
        r = np.hypot(P[..., 0], P[..., 1])
        th = np.arctan2(r, P[..., 2])
        s = np.where(r > 1e-12, th / np.maximum(r, 1e-12), 0.0)
        return np.stack([self.fx * P[..., 0] * s + self.cx, self.fy * P[..., 1] * s + self.cy], -1)


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a to unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        axis = np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))
