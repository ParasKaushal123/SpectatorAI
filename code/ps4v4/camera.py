"""M1: camera intrinsics and per-frame head rotation.

Intrinsics, in order of preference: a calibration JSON, GeoCalib's focal estimate
(combined with an equidistant fisheye model, which matches a pinhole focal at the
image centre), or a field-of-view guess.

Head rotation: background features are tracked frame to frame with hand boxes masked
out; their viewing rays give the rotation between frames (RANSAC Kabsch). Head
translation is ignored: hands are ~0.4 m away, head translation is a few cm.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from .geometry import FisheyeCamera, kabsch_ransac
from .video import VideoMeta, iter_frames, read_frame

log = logging.getLogger(__name__)


# ---- intrinsics -------------------------------------------------------------
def estimate_intrinsics(meta: VideoMeta, cfg) -> FisheyeCamera:
    c = cfg.camera
    if c.calibration_json:
        with open(c.calibration_json) as fh:
            cam = FisheyeCamera.from_dict(json.load(fh), meta.width, meta.height)
        log.info("intrinsics from calibration file: f=%.1f", cam.fx)
        return cam
    if c.use_geocalib:
        try:
            f, gravity = _geocalib_estimate(meta, c.geocalib_frames, cfg.device)
            cam = FisheyeCamera(meta.width, meta.height, f, f, meta.width / 2, meta.height / 2, np.zeros(4),
                                source="geocalib_focal+equidistant")
            log.info("intrinsics from GeoCalib: f=%.1f", f)
            cam.gravity_samples = gravity
            return cam
        except Exception as exc:                                   # noqa: BLE001 - fall back on any failure
            log.warning("GeoCalib failed (%s); using FOV guess", exc)
    cam = FisheyeCamera.from_fov(meta.width, meta.height, c.fallback_dfov_deg)
    log.info("intrinsics from FOV guess %.0f deg: f=%.1f", c.fallback_dfov_deg, cam.fx)
    return cam


@lru_cache(maxsize=2)
def geocalib_model(device: str):
    from geocalib import GeoCalib
    return GeoCalib().to(device).eval()


def _geocalib_focal(meta: VideoMeta, n_frames: int, device: str) -> float:
    return _geocalib_estimate(meta, n_frames, device)[0]


def _geocalib_estimate(meta: VideoMeta, n_frames: int, device: str):
    import torch

    dev = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    model = geocalib_model(dev)
    focals = []
    gravity = []
    for idx in np.linspace(meta.n_frames * 0.05, meta.n_frames * 0.95, n_frames).astype(int):
        img = read_frame(meta, int(idx))
        if img is None:
            continue
        t = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255).to(dev)
        res = model.calibrate(t, camera_model="simple_divisional")
        focals.append(float(res["camera"].f.reshape(-1)[0]))
        try:
            # GeoCalib represents up; the scene schema requires down.
            g = -res["gravity"].vec3d.detach().cpu().numpy().reshape(-1, 3)[0]
            if np.isfinite(g).all():
                gravity.append({"frame": int(idx), "down": g.tolist()})
        except (KeyError, AttributeError, TypeError):
            pass
    if not focals:
        raise RuntimeError("no frames calibrated")
    return float(np.median(focals)), gravity


# ---- head rotation ----------------------------------------------------------
def estimate_head_rotation(meta: VideoMeta, cam: FisheyeCamera, boxes_per_frame: list[np.ndarray], cfg) -> dict:
    """Returns {'R': (N,3,3) camera_t -> reference(frame 0) rotations, 'inliers': (N,) counts}."""
    hc = cfg.camera.head_rotation
    N = meta.n_frames
    C = np.repeat(np.eye(3)[None], N, 0)
    inl_count = np.zeros(N, int)
    if not hc.enabled:
        return {"R": C, "inliers": inl_count}
    scale = 0.5
    prev_gray, prev_mask = None, None
    rng = np.random.default_rng(0)
    for fi, frame in iter_frames(meta):
        gray = cv2.cvtColor(cv2.resize(frame, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
        mask = _background_mask(gray.shape, boxes_per_frame[fi] if fi < len(boxes_per_frame) else [],
                                scale, hc.hand_mask_scale)
        if prev_gray is not None:
            R_rel, n_in = _relative_rotation(prev_gray, gray, prev_mask, cam, scale, hc, rng)
            inl_count[fi] = n_in
            C[fi] = C[fi - 1] @ R_rel.T
        else:
            C[fi] = np.eye(3)
        prev_gray, prev_mask = gray, mask
        if fi + 1 >= N:
            break
    if hc.smooth_sigma > 0 and N > 3:
        q = Rotation.from_matrix(C).as_quat()
        for i in range(1, N):
            if np.dot(q[i], q[i - 1]) < 0:
                q[i] = -q[i]
        q = gaussian_filter1d(q, hc.smooth_sigma, axis=0, mode="nearest")
        C = Rotation.from_quat(q / np.linalg.norm(q, axis=1, keepdims=True)).as_matrix()
    return {"R": C, "inliers": inl_count}


def _background_mask(shape, boxes, scale: float, grow: float) -> np.ndarray:
    m = np.full(shape, 255, np.uint8)
    for b in boxes:
        cx, cy = (b[0] + b[2]) / 2 * scale, (b[1] + b[3]) / 2 * scale
        hw, hh = (b[2] - b[0]) * scale * grow / 2, (b[3] - b[1]) * scale * grow / 2
        cv2.rectangle(m, (int(cx - hw), int(cy - hh)), (int(cx + hw), int(cy + hh)), 0, -1)
    return m


def _relative_rotation(g0, g1, mask0, cam: FisheyeCamera, scale: float, hc, rng) -> tuple[np.ndarray, int]:
    p0 = cv2.goodFeaturesToTrack(g0, maxCorners=hc.max_corners, qualityLevel=0.01, minDistance=8, mask=mask0)
    if p0 is None or len(p0) < hc.min_inliers:
        return np.eye(3), 0
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, winSize=(21, 21), maxLevel=3)
    pb, stb, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, winSize=(21, 21), maxLevel=3)
    ok = (st.ravel() == 1) & (stb.ravel() == 1) & (np.linalg.norm((pb - p0).reshape(-1, 2), axis=1) < 1.0)
    if ok.sum() < hc.min_inliers:
        return np.eye(3), int(ok.sum())
    a = cam.rays(p0.reshape(-1, 2)[ok] / scale)
    b = cam.rays(p1.reshape(-1, 2)[ok] / scale)
    R, inl = kabsch_ransac(a, b, np.radians(hc.ransac_deg), iters=int(hc.get("ransac_iters", 200)), rng=rng)
    if inl.sum() < hc.min_inliers:
        return np.eye(3), int(inl.sum())
    return R, int(inl.sum())
