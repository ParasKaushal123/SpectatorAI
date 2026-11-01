"""Virtual pinhole views cut out of a fisheye frame (depth models and TSDF fusion need pinhole images)."""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# (yaw, pitch) in degrees, camera axes: x right, y down, z forward. Centre, left, right, up, down.
DEFAULT_DIRS = ((0, 0), (-60, 0), (60, 0), (0, -50), (0, 50))


def view_rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """R_cv: virtual-camera axes -> fisheye-camera axes (yaw about camera y, then pitch about the new x)."""
    return Rotation.from_euler("YX", [yaw_deg, -pitch_deg], degrees=True).as_matrix()


class VirtualViews:
    def __init__(self, cam, size: int = 384, fov_deg: float = 90.0, dirs=DEFAULT_DIRS):
        self.size, self.fov = int(size), float(fov_deg)
        f = (size / 2) / math.tan(math.radians(fov_deg) / 2)
        self.K = np.array([[f, 0, (size - 1) / 2], [0, f, (size - 1) / 2], [0, 0, 1.0]])
        u, v = np.meshgrid(np.arange(size, dtype=float), np.arange(size, dtype=float))
        rays_v = np.stack([(u - self.K[0, 2]) / f, (v - self.K[1, 2]) / f, np.ones_like(u)], -1)
        self.R, self.maps, self.valid = [], [], []
        for yaw, pitch in dirs:
            R = view_rotation(yaw, pitch)
            rc = rays_v @ R.T                                           # rays in the fisheye camera frame
            uv = cam.project(rc)
            ok = np.isfinite(uv).all(-1) & (uv[..., 0] >= 0) & (uv[..., 0] <= cam.width - 1) & \
                (uv[..., 1] >= 0) & (uv[..., 1] <= cam.height - 1)
            # stay inside the lens' valid domain: the ray must come back unchanged
            back = cam.rays(np.where(ok[..., None], uv, 0))
            n = rc / np.linalg.norm(rc, axis=-1, keepdims=True)
            ok &= np.isfinite(back).all(-1) & ((back * n).sum(-1) > 0.9999)
            mx = np.where(ok, uv[..., 0], -1).astype(np.float32)
            my = np.where(ok, uv[..., 1], -1).astype(np.float32)
            self.R.append(R)
            self.maps.append((mx, my))
            self.valid.append(ok)

    def __len__(self):
        return len(self.R)

    def images(self, frame: np.ndarray) -> list[np.ndarray]:
        return [cv2.remap(frame, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) for mx, my in self.maps]

    def masks(self, mask: np.ndarray) -> list[np.ndarray]:
        """Remap a full-resolution bool mask (True = keep) into every view (outside the lens = False)."""
        m = mask.astype(np.uint8)
        return [(cv2.remap(m, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0) & ok
                for (mx, my), ok in zip(self.maps, self.valid)]
