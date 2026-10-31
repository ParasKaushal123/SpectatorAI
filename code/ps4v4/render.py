"""M6a: side-by-side render.

Left: the original frame with each hand's skeleton (re-projected through the fisheye model),
solid when observed, thin when infilled, plus a state badge per side.
Right: both hands in the head-stable 3D frame (mesh when the backend provides one). The view centre\nfollows the hands slowly (smoothed over a few seconds), with a fading trail of each wrist.
"""
from __future__ import annotations

import logging
import math
import subprocess

import cv2
import numpy as np

from . import HAND_CONNECTIONS, SIDES, STATES
from .video import VideoMeta, iter_frames

log = logging.getLogger(__name__)
COL = {"left": (60, 200, 60), "right": (0, 140, 255)}            # BGR
GREY = (150, 150, 150)


def _view(t: float, size: int, tgt: np.ndarray, radius: float):
    az = math.radians(12 * math.sin(2 * math.pi * t / 16))            # slow, gentle orbit
    el = math.radians(24)
    dist = max(0.35, 1.05 * radius / math.tan(math.radians(26)))
    pos = tgt + dist * np.array([math.sin(az) * math.cos(el), -math.sin(el), -math.cos(az) * math.cos(el)])
    f = (tgt - pos) / np.linalg.norm(tgt - pos)
    r = np.cross(f, [0.0, -1.0, 0.0]); r /= np.linalg.norm(r)
    d = np.cross(f, r)
    M = np.stack([r, d, f], 1)

    def proj(P):
        Q = (np.asarray(P) - pos) @ M
        z = np.maximum(Q[..., 2], 1e-3)
        return np.stack([size / 2 + 0.9 * size * Q[..., 0] / z, size / 2 + 0.9 * size * Q[..., 1] / z], -1), Q
    return proj, M


def follow_targets(seq: dict, fps: float, sigma_s: float) -> np.ndarray:
    """Per-frame view centre: the hands' mean position, smoothed over a few seconds, so the view follows
    the hands slowly (they stay near the centre) while their own motion still shows."""
    from scipy.ndimage import gaussian_filter1d
    W = np.stack([seq[s]["J_head"][:, 0] for s in SIDES])              # (2, N, 3)
    ok = np.isfinite(W).all(-1)
    num = np.where(ok[..., None], W, 0).sum(0)
    den = ok.sum(0).astype(float)
    sig = max(1.0, sigma_s * fps)
    num = gaussian_filter1d(num, sig, axis=0, mode="nearest")
    den = gaussian_filter1d(den, sig, mode="nearest")
    tgt = np.full_like(num, np.nan)
    good = den > 1e-3
    tgt[good] = num[good] / den[good, None]
    if not good.any():
        return np.tile([0.0, 0.15, 0.4], (len(den), 1))
    idx = np.arange(len(den))
    for k in range(3):                                                  # long gaps: hold the nearest centre
        tgt[:, k] = np.interp(idx, idx[good], tgt[good, k])
    return tgt


def draw_3d(hands: list, size: int, t: float, tgt: np.ndarray, radius: float, faces, trails: dict | None = None,
            ss: int = 2) -> np.ndarray:
    S = size * ss                                                       # supersample, then downscale: smooth edges
    img = np.full((S, S, 3), 24, np.uint8)
    proj, _ = _view(t, S, tgt, radius)
    gy = tgt[1] + 0.22                                                  # floor plane under the hands
    g = radius * 1.4
    for a in np.linspace(-g, g, 13):
        for p0, p1 in (([tgt[0] + a, gy, tgt[2] - g], [tgt[0] + a, gy, tgt[2] + g]),
                       ([tgt[0] - g, gy, tgt[2] + a], [tgt[0] + g, gy, tgt[2] + a])):
            uv, _ = proj(np.array([p0, p1]))
            cv2.line(img, tuple(uv[0].astype(int)), tuple(uv[1].astype(int)), (50, 50, 50), ss, cv2.LINE_AA)
    for side, J, V, src in hands:                                       # soft shadow + drop line: depth at a glance
        w = J[0]
        if not np.isfinite(w).all():
            continue
        uv, _ = proj(np.array([w, [w[0], gy, w[2]]]))
        cv2.ellipse(img, tuple(uv[1].astype(int)), (10 * ss, 4 * ss), 0, 0, 360, (36, 36, 36), -1, cv2.LINE_AA)
        cv2.line(img, tuple(uv[0].astype(int)), tuple(uv[1].astype(int)), (70, 70, 70), ss, cv2.LINE_AA)
    if trails:                                                          # fading light trail of each wrist
        for side, P in trails.items():
            P = P[np.isfinite(P).all(1)]
            if len(P) < 2:
                continue
            uv, _ = proj(P)
            n = len(uv)
            for i in range(1, n):
                a = i / n
                c = tuple(int(24 + (v - 24) * a) for v in COL[side])
                cv2.line(img, tuple(uv[i - 1].astype(int)), tuple(uv[i].astype(int)), c, max(1, int(3 * ss * a)),
                         cv2.LINE_AA)
    polys, cols, depth = [], [], []
    for side, J, V, src in hands:
        base = np.array(COL[side], float) * (1.0 if src == 1 else 0.6 if src == 2 else 0.35)
        if faces is not None and V is not None and np.isfinite(V).all():
            uv, Q = proj(V)
            tri = Q[faces]
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
            L = np.array([-0.35, -0.75, -0.55]); L /= np.linalg.norm(L)          # key light, upper left, front
            diff = np.abs(n @ L)
            rim = np.clip(1 - np.abs(n[:, 2]), 0, 1) ** 3                       # thin rim light on the silhouette
            shade = np.clip(0.28 + 0.72 * diff + 0.25 * rim, 0, 1.15)
            polys += list((uv[faces]).astype(np.int32)); depth += list(tri[:, :, 2].mean(1))
            cols += np.clip(shade[:, None] * base[None], 0, 255).astype(int).tolist()
        else:
            uv, _ = proj(J)
            dashed = src == 3
            for a, b in HAND_CONNECTIONS:
                if dashed:
                    _dashed(img, uv[a], uv[b], tuple(int(c) for c in base), 2 * ss, 6 * ss)
                else:
                    cv2.line(img, tuple(uv[a].astype(int)), tuple(uv[b].astype(int)), tuple(int(c) for c in base),
                             3 * ss, cv2.LINE_AA)
            for p in uv:
                cv2.circle(img, tuple(p.astype(int)), 2 * ss, (235, 235, 235), -1, cv2.LINE_AA)
    for k in np.argsort(depth)[::-1]:
        cv2.fillConvexPoly(img, polys[k], cols[k], cv2.LINE_AA)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA) if ss > 1 else img
    cv2.putText(img, "3D hands (head rotation removed), view follows the hands", (10, size - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (205, 205, 205), 1, cv2.LINE_AA)
    return img


class _Writer:
    """Raw frames straight into x264 (one high-quality encode). Falls back to OpenCV mp4v without ffmpeg."""

    def __init__(self, path: str, fps: float, size: tuple, crf: int):
        self.p, self.vw = None, None
        try:
            self.p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                       "-s", f"{size[0]}x{size[1]}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264",
                                       "-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", "veryfast", path],
                                      stdin=subprocess.PIPE)
        except FileNotFoundError:
            self.vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, frame: np.ndarray):
        if self.p is not None:
            self.p.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            self.vw.write(frame)

    def release(self):
        if self.p is not None:
            self.p.stdin.close(); self.p.wait()
        else:
            self.vw.release()


def _dashed(img, p0, p1, col, thick=1, dash=6):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    n = max(1, int(np.linalg.norm(p1 - p0) / dash))
    for i in range(0, n, 2):
        a, b = p0 + (p1 - p0) * i / n, p0 + (p1 - p0) * min(i + 1, n) / n
        cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), col, thick, cv2.LINE_AA)


def render(meta: VideoMeta, seq: dict, faces, out_path: str, cfg, title: str = "", rejected: dict | None = None) -> str:
    rc = cfg.render
    RH = rc.height
    lw = int(round(meta.width * RH / meta.height)) // 2 * 2
    s = RH / meta.height
    tgts = follow_targets(seq, meta.fps, rc.follow_sigma_s)
    radius = float(rc.view_radius_m)
    n_trail = max(2, int(rc.trail_s * meta.fps))
    n_max = meta.n_frames if not rc.max_seconds else min(meta.n_frames, int(rc.max_seconds * meta.fps))
    vw = _Writer(out_path, meta.fps, (lw + RH, RH), rc.crf)
    for t, frame in iter_frames(meta):
        if t >= n_max:
            break
        left = cv2.resize(frame, (lw, RH), interpolation=cv2.INTER_AREA)
        if rc.show_rejected and rejected:
            for b, why in rejected.get(t, []):                  # tracklets M3A decided are not the wearer's hands
                bb = (np.asarray(b) * s).astype(int)
                cv2.rectangle(left, (bb[0], bb[1]), (bb[2], bb[3]), GREY, 1, cv2.LINE_AA)
                cv2.putText(left, f"x {why}", (bb[0], bb[1] - 4 if bb[1] > 40 else bb[3] + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, GREY, 1,
                            cv2.LINE_AA)
        hands3d = []
        for i, side in enumerate(SIDES):
            d = seq[side]
            src = int(d["src"][t])
            state = STATES[int(d["state"][t])]
            label = f"{side.upper()}: " + ("OUT OF VIEW" if src == 0 and state == "OUT_OF_VIEW" else
                                           "not reconstructed" if src == 0 else
                                           "infilled" if src == 2 else
                                           "out of view (estimated)" if src == 3 else state.lower())
            cv2.putText(left, label, (10 + i * (lw // 2), RH - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        COL[side] if src else GREY, 2, cv2.LINE_AA)
            if src == 0:
                continue
            kp = d["kp2d"][t] * s
            if src == 3:                                          # out-of-view guess: faded, dashed, never solid
                faded = tuple(int(0.55 * c) for c in COL[side])
                for a, b in HAND_CONNECTIONS:
                    if np.isfinite(kp[[a, b]]).all():
                        _dashed(left, kp[a], kp[b], faded)
                hands3d.append((side, d["J_head"][t], None, src))
                continue
            thick = 2 if src == 1 else 1
            for a, b in HAND_CONNECTIONS:
                if np.isfinite(kp[[a, b]]).all():
                    cv2.line(left, tuple(kp[a].astype(int)), tuple(kp[b].astype(int)), COL[side], thick, cv2.LINE_AA)
            for k, p in enumerate(kp):
                if np.isfinite(p).all():
                    c = (255, 255, 255) if d["joint_in_image"][t, k] else GREY
                    cv2.circle(left, tuple(p.astype(int)), 2, c, -1, cv2.LINE_AA)
            V = d["verts_head"][t] if d.get("verts_head") is not None else None
            hands3d.append((side, d["J_head"][t], V, src))
        cv2.putText(left, f"{title}  t={t / meta.fps:6.2f}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        trails = {sd: seq[sd]["J_head"][max(0, t - n_trail):t + 1, 0] for sd in SIDES}
        vw.write(np.hstack([left, draw_3d(hands3d, RH, t / meta.fps, tgts[t], radius, faces, trails,
                                          rc.supersample)]))
    vw.release()
    log.info("M6 rendered %s", out_path)
    return out_path
