"""Fisheye camera model (Kannala-Brandt / OpenCV equidistant), virtual pinhole crops, rotations.

Camera axes everywhere: x right, y down, z forward (OpenCV).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class FisheyeCamera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k: np.ndarray = field(default_factory=lambda: np.zeros(4))   # k1..k4
    source: str = "unknown"

    # ---- construction ---------------------------------------------------
    @classmethod
    def from_fov(cls, width: int, height: int, dfov_deg: float) -> "FisheyeCamera":
        f = (math.hypot(width, height) / 2) / math.radians(dfov_deg / 2)     # equidistant: r = f*theta
        return cls(width, height, f, f, width / 2, height / 2, np.zeros(4), source=f"fov_guess_{dfov_deg:.0f}")

    @classmethod
    def from_dict(cls, d: dict, width: int | None = None, height: int | None = None) -> "FisheyeCamera":
        w0, h0 = d.get("width", width), d.get("height", height)
        sx, sy = (width or w0) / w0, (height or h0) / h0
        k = np.array([d.get(f"k{i}", 0.0) for i in range(1, 5)], float)
        return cls(int(width or w0), int(height or h0), d["fx"] * sx, d["fy"] * sy, d["cx"] * sx, d["cy"] * sy, k,
                   source=d.get("source", "calibration"))

    def to_dict(self) -> dict:
        return {"width": self.width, "height": self.height, "fx": self.fx, "fy": self.fy, "cx": self.cx,
                "cy": self.cy, **{f"k{i + 1}": float(v) for i, v in enumerate(self.k)}, "model": "KB4",
                "source": self.source}

    def to_egohand_json(self, path: str) -> None:
        """EgoHand's camera format (model KB4 = OpenCV equidistant fisheye)."""
        with open(path, "w") as fh:
            json.dump({"model": "KB4", "image_width": self.width, "image_height": self.height,
                       "camera_matrix": {"matrix_3x3": self.K.tolist()},
                       "distortion_coefficients": [float(v) for v in self.k]}, fh, indent=2)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    # ---- projection -----------------------------------------------------
    def project(self, P: np.ndarray) -> np.ndarray:
        """3D points (...,3) in camera coordinates -> pixels (...,2). Works for wide angles (> 90 deg)."""
        P = np.asarray(P, float)
        x, y, z = P[..., 0], P[..., 1], P[..., 2]
        r = np.hypot(x, y)
        theta = np.arctan2(r, z)
        t2 = theta * theta
        k1, k2, k3, k4 = self.k
        theta_d = theta * (1 + k1 * t2 + k2 * t2 ** 2 + k3 * t2 ** 3 + k4 * t2 ** 4)
        scale = np.where(r > 1e-12, theta_d / np.maximum(r, 1e-12), 1.0 / np.maximum(z, 1e-12))
        return np.stack([self.fx * x * scale + self.cx, self.fy * y * scale + self.cy], -1)

    def rays(self, uv: np.ndarray) -> np.ndarray:
        """Pixels (...,2) -> unit viewing rays (...,3)."""
        uv = np.asarray(uv, float)
        mx, my = (uv[..., 0] - self.cx) / self.fx, (uv[..., 1] - self.cy) / self.fy
        theta_d = np.hypot(mx, my)
        theta = theta_d.copy()
        k1, k2, k3, k4 = self.k
        if np.any(self.k != 0):
            for _ in range(10):                                     # Newton: solve theta_d(theta) = theta_d
                t2 = theta * theta
                f = theta * (1 + k1 * t2 + k2 * t2 ** 2 + k3 * t2 ** 3 + k4 * t2 ** 4) - theta_d
                df = 1 + 3 * k1 * t2 + 5 * k2 * t2 ** 2 + 7 * k3 * t2 ** 3 + 9 * k4 * t2 ** 4
                theta = theta - f / np.where(np.abs(df) < 1e-9, 1e-9, df)
        s = np.where(theta_d > 1e-12, np.sin(theta) / np.maximum(theta_d, 1e-12), 1.0)
        return np.stack([mx * s, my * s, np.cos(theta)], -1)

    def normalized(self, uv: np.ndarray) -> np.ndarray:
        """Pixels -> normalised pinhole coordinates (x/z, y/z)."""
        r = self.rays(uv)
        return r[..., :2] / np.maximum(r[..., 2:3], 1e-6)


# ---- virtual pinhole crops (perspective-correct view centred on a hand) --------
def look_rotation(ray: np.ndarray) -> np.ndarray:
    """Rotation R (3x3, rows = virtual camera axes in real-camera coordinates) with virtual z = ray."""
    z = ray / np.linalg.norm(ray)
    x = np.cross([0.0, 1.0, 0.0], z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


@dataclass
class VirtualView:
    R: np.ndarray       # real -> virtual rotation
    f: float            # virtual focal length (px)
    size: int           # square image size (px)

    @property
    def c(self) -> float:
        return self.size / 2.0

    def to_real_rays(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, float)
        v = np.stack([(uv[..., 0] - self.c) / self.f, (uv[..., 1] - self.c) / self.f, np.ones(uv.shape[:-1])], -1)
        return v @ self.R          # R^T v for row vectors

    def project_real(self, P: np.ndarray) -> np.ndarray:
        """3D points in real-camera coords -> virtual pixels."""
        Q = np.asarray(P, float) @ self.R.T
        return np.stack([self.f * Q[..., 0] / Q[..., 2] + self.c, self.f * Q[..., 1] / Q[..., 2] + self.c], -1)


def make_virtual_view(cam: FisheyeCamera, box: np.ndarray, size: int, fov_scale: float) -> VirtualView:
    """Virtual pinhole camera looking at the box centre; its FOV covers fov_scale x the box's angular size."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    ray = cam.rays(np.array([cx, cy]))
    edge = cam.rays(np.array([[box[0], cy], [box[2], cy], [cx, box[1]], [cx, box[3]]]))
    ang = max(np.arccos(np.clip(edge @ ray, -1, 1)).max() * 2, math.radians(4))
    half = min(ang * fov_scale / 2, math.radians(80))
    f = (size / 2) / math.tan(half)
    return VirtualView(look_rotation(ray), f, size)


def render_virtual(img: np.ndarray, cam: FisheyeCamera, view: VirtualView) -> np.ndarray:
    """Resample the fisheye image into the virtual pinhole view."""
    s = view.size
    u, v = np.meshgrid(np.arange(s, dtype=np.float64), np.arange(s, dtype=np.float64))
    rays = view.to_real_rays(np.stack([u, v], -1))
    uv = cam.project(rays).astype(np.float32)
    return cv2.remap(img, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


# ---- 3D helpers -------------------------------------------------------------
def solve_translation(X: np.ndarray, uv_norm: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Translation t so that (X + t) projects to normalised coords uv_norm (weighted linear least squares)."""
    u, v = uv_norm[:, 0], uv_norm[:, 1]
    n = len(X)
    A = np.zeros((2 * n, 3)); b = np.zeros(2 * n)
    A[0::2, 0] = 1; A[0::2, 2] = -u; b[0::2] = u * X[:, 2] - X[:, 0]
    A[1::2, 1] = 1; A[1::2, 2] = -v; b[1::2] = v * X[:, 2] - X[:, 1]
    if w is not None:
        ww = np.repeat(np.sqrt(np.clip(w, 1e-6, None)), 2)
        A, b = A * ww[:, None], b * ww
    return np.linalg.lstsq(A, b, rcond=None)[0]


def kabsch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation R minimising |R a_i - b_i| for unit vectors (rows)."""
    H = a.T @ b
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def kabsch_ransac(a: np.ndarray, b: np.ndarray, thresh_rad: float, iters: int = 200,
                  rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Rotation R with b ~ R a from unit rays, robust to outliers. All hypotheses are scored at once."""
    rng = rng or np.random.default_rng(0)
    n = len(a)
    if n < 3:
        return np.eye(3), np.zeros(n, bool)
    idx = np.stack([rng.choice(n, 3, replace=False) for _ in range(iters)])     # (iters, 3)
    Hm = np.einsum("kni,knj->kij", a[idx], b[idx])
    U, _, Vt = np.linalg.svd(Hm)
    d = np.sign(np.linalg.det(np.einsum("kji,kjl->kil", Vt, np.transpose(U, (0, 2, 1)))))
    D = np.repeat(np.eye(3)[None], iters, 0)
    D[:, 2, 2] = d
    R = np.einsum("kji,kjl,klm->kim", Vt, D, np.transpose(U, (0, 2, 1)))       # V D U^T
    cos = np.einsum("kij,nj,ni->kn", R, a, b)
    inl = cos > math.cos(thresh_rad)
    best = inl[np.argmax(inl.sum(1))]
    Rf = kabsch(a[best], b[best]) if best.sum() >= 3 else np.eye(3)
    return Rf, best


def hand_frame(J: np.ndarray) -> np.ndarray:
    """Hand orientation from 21 joints (...,21,3): rotation whose columns are
    x = little->index knuckle, y = wrist->middle knuckle, z = palm normal. Model-agnostic."""
    J = np.asarray(J, float)
    y = J[..., 9, :] - J[..., 0, :]
    x0 = J[..., 5, :] - J[..., 17, :]
    y = y / np.maximum(np.linalg.norm(y, axis=-1, keepdims=True), 1e-9)
    z = np.cross(x0, y)
    z = z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-9)
    x = np.cross(y, z)
    return np.stack([x, y, z], -1)


def quat_continuity(q: np.ndarray) -> np.ndarray:
    q = q.copy()
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q
