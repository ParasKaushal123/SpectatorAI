"""World render: the original frame next to a 3D view of the room (map points), the worker's head
(camera frustum + path), gaze ray, body axis and both hands with light trails. The 3D view follows the
worker slowly from behind and above."""
from __future__ import annotations

import logging
import math

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from ps4v4 import HAND_CONNECTIONS
from ps4v4.render import _Writer

log = logging.getLogger(__name__)
COL = {"left": (60, 200, 60), "right": (0, 140, 255)}                 # BGR
HEAD = (255, 220, 120)
GAZE = (60, 230, 255)
BODY = (200, 200, 200)
BG = 22


class _View:
    def __init__(self, pos, target, size, f_frac=0.85):
        fwd = target - pos
        fwd /= np.linalg.norm(fwd)
        r = np.cross(fwd, [0, 0, 1.0])
        r /= max(np.linalg.norm(r), 1e-9)
        d = np.cross(fwd, r)                                            # image "down"
        self.M = np.stack([r, d, fwd], 1)
        self.pos, self.size, self.f = pos, size, f_frac * size

    def proj(self, P):
        Q = (np.asarray(P, float) - self.pos) @ self.M
        z = Q[..., 2]
        uv = np.stack([self.size / 2 + self.f * Q[..., 0] / np.maximum(z, 1e-3),
                       self.size / 2 + self.f * Q[..., 1] / np.maximum(z, 1e-3)], -1)
        return uv, z


def _line(img, view, a, b, col, th=2):
    uv, z = view.proj(np.array([a, b]))
    if (z > 0.05).all() and np.isfinite(uv).all():
        cv2.line(img, tuple(np.round(uv[0]).astype(int)), tuple(np.round(uv[1]).astype(int)), col, th, cv2.LINE_AA)


def _splat(img, view, P, C, max_dist):
    if not len(P):
        return
    uv, z = view.proj(P)
    S = img.shape[0]
    ok = (z > 0.05) & (z < max_dist) & (uv[:, 0] >= 0) & (uv[:, 0] < S - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < S - 1)
    if not ok.any():
        return
    u, v, zz, c = uv[ok, 0].astype(int), uv[ok, 1].astype(int), z[ok], C[ok]
    shade = np.clip(1.15 - zz / max_dist, 0.35, 1.0)[:, None]
    c = (c * shade).astype(np.uint8)
    o = np.argsort(-zz)                                                 # far first: near points overwrite
    for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1)):
        img[v[o] + dv, u[o] + du] = c[o]


_LIGHT = np.array([0.35, -0.45, 1.0]) / np.linalg.norm([0.35, -0.45, 1.0])     # from above, slightly in front


def _mesh_polys(view, V, faces, base):
    """Shaded triangles of one hand mesh: (polygons in pixels, colours, depths) for painter's-order drawing."""
    uv, z = view.proj(V)
    tri = V[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    to_cam = view.pos - tri.mean(1)
    to_cam /= np.maximum(np.linalg.norm(to_cam, axis=1, keepdims=True), 1e-12)
    diff = np.abs(n @ _LIGHT)
    rim = np.clip(1 - np.abs((n * to_cam).sum(1)), 0, 1) ** 3                 # thin bright silhouette
    shade = np.clip(0.30 + 0.70 * diff + 0.30 * rim, 0, 1.2)
    zf = z[faces]
    ok = (zf > 0.02).all(1)
    cols = np.clip(shade[:, None] * np.asarray(base, float)[None], 0, 255).astype(int)
    return uv[faces][ok], cols[ok], zf[ok].mean(1)


def render_world(video: str, hands: dict, world: dict, out_path: str, height: int = 720, title: str = "",
                 max_points: int = 200_000, trail_s: float = 1.0, head_trail_s: float = 4.0) -> str:
    N, fps = world["N"], world["fps"]
    cap = cv2.VideoCapture(video)
    W0, H0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    lw = int(round(W0 * height / H0)) // 2 * 2
    S = height
    s = lw / W0
    # map points (BGR 0-255), subsampled
    P, C = world["points"], world["colors"]
    if len(P) > max_points:
        k = np.random.default_rng(0).choice(len(P), max_points, replace=False)
        P, C = P[k], C[k]
    C = (np.clip(C, 0, 1)[:, ::-1] * 255).astype(np.float32) if len(C) else np.zeros((0, 3), np.float32)
    # follow camera: behind-above the worker, looking at head + hands, smoothed over a few seconds
    hp, f = world["head_p"], world["facing"]
    wr = np.stack([world["hands"][sd][:, 0] for sd in ("left", "right")])
    tgt = hp.copy()
    for k in range(N):
        w = wr[:, k][np.isfinite(wr[:, k]).all(1)]
        if len(w):
            tgt[k] = 0.5 * hp[k] + 0.5 * w.mean(0)
    sig = max(1.0, 2.0 * fps)
    tgt = gaussian_filter1d(tgt, sig, axis=0, mode="nearest")
    fs = gaussian_filter1d(f, sig, axis=0, mode="nearest")
    fs /= np.maximum(np.linalg.norm(fs, axis=1, keepdims=True), 1e-9)
    floor_z = (np.percentile(P[:, 2], 3) if len(P) > 1000 else float(np.min(world["pelvis"][:, 2]) - 0.45))
    nt, nh = max(2, int(trail_s * fps)), max(2, int(head_trail_s * fps))
    corners = None
    cam = world["cam"]
    if hasattr(cam, "rays"):
        cr = cam.rays(np.array([[0, 0], [W0 - 1, 0], [W0 - 1, H0 - 1], [0, H0 - 1]], float))
        if np.isfinite(cr).all():
            corners = cr
    if corners is None:
        corners = np.array([[-1, -0.6, 1], [1, -0.6, 1], [1, 0.6, 1], [-1, 0.6, 1.0]])
        corners /= np.linalg.norm(corners, axis=1, keepdims=True)
    mesh, faces = world.get("mesh") or {}, world.get("faces")
    vw = _Writer(out_path, fps, (lw + S, S), 18)
    for t in range(N):
        ok, frame = cap.read()
        if not ok:
            break
        left = cv2.resize(frame, (lw, S), interpolation=cv2.INTER_AREA)
        for sd in ("left", "right"):
            if hands[sd]["src"][t] in (1, 2):
                kp = hands[sd]["kp2d"][t] * s
                for a, b in HAND_CONNECTIONS:
                    if np.isfinite(kp[[a, b]]).all():
                        cv2.line(left, tuple(kp[a].astype(int)), tuple(kp[b].astype(int)), COL[sd], 2, cv2.LINE_AA)
        cv2.putText(left, f"{title}  t={t / fps:6.2f}s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                    cv2.LINE_AA)
        # 3D view
        img = np.full((S, S, 3), BG, np.uint8)
        az = math.radians(25 * math.sin(2 * math.pi * t / (20 * fps)))
        back = np.array([math.cos(az) * -fs[t, 0] - math.sin(az) * -fs[t, 1],
                         math.sin(az) * -fs[t, 0] + math.cos(az) * -fs[t, 1], 0.0])
        right = np.cross(fs[t], [0, 0, 1.0])
        view = _View(tgt[t] + 1.35 * back + 0.35 * right + np.array([0, 0, 0.95]), tgt[t], S)
        gz = floor_z
        for a in np.linspace(-2, 2, 17):                                # floor grid around the worker
            _line(img, view, [tgt[t, 0] + a, tgt[t, 1] - 2, gz], [tgt[t, 0] + a, tgt[t, 1] + 2, gz], (45, 45, 45), 1)
            _line(img, view, [tgt[t, 0] - 2, tgt[t, 1] + a, gz], [tgt[t, 0] + 2, tgt[t, 1] + a, gz], (45, 45, 45), 1)
        _splat(img, view, P, C, 5.0)
        # head path
        a0 = max(0, t - nh)
        path = hp[a0:t + 1]
        for i in range(1, len(path)):
            c = tuple(int(BG + (v - BG) * i / len(path)) for v in HEAD)
            _line(img, view, path[i - 1], path[i], c, 2)
        # body axis
        _line(img, view, world["neck"][t], world["pelvis"][t], BODY, 3)
        _line(img, view, world["shoulders"][t, 0], world["shoulders"][t, 1], BODY, 3)
        _line(img, view, world["neck"][t], hp[t], BODY, 2)
        _line(img, view, world["pelvis"][t], world["pelvis"][t] + 0.3 * f[t], (150, 150, 150), 2)
        # head = camera frustum + gaze ray
        R, c0 = world["head_R"][t], hp[t]
        cw = c0 + 0.13 * corners @ R.T
        for i in range(4):
            _line(img, view, c0, cw[i], HEAD, 2)
            _line(img, view, cw[i], cw[(i + 1) % 4], HEAD, 2)
        _line(img, view, c0, c0 + 0.8 * world["gaze"][t], GAZE, 2)
        # hands (shaded meshes when PS4 exported them, else skeletons) + wrist light trails
        polys, cols, depths = [], [], []
        for sd in ("left", "right"):
            J = world["hands"][sd]
            src = world["src"][sd]
            tr = J[max(0, t - nt):t + 1, 0]
            for i in range(1, len(tr)):
                if np.isfinite(tr[i - 1]).all() and np.isfinite(tr[i]).all():
                    c = tuple(int(BG + (v - BG) * i / len(tr)) for v in COL[sd])
                    _line(img, view, tr[i - 1], tr[i], c, max(1, int(3 * i / len(tr))))
            if src[t] > 0 and np.isfinite(J[t]).all():
                col = COL[sd] if src[t] in (1, 2) else tuple(int(0.45 * v) for v in COL[sd])
                V = mesh[sd][t] if mesh.get(sd) is not None else None
                if faces is not None and V is not None and src[t] in (1, 2) and np.isfinite(V).all():
                    P_, C_, Z_ = _mesh_polys(view, V, faces, col)          # real hand surface (MANO mesh)
                    polys.append(P_); cols.append(C_); depths.append(Z_)
                else:                                                      # estimated / no mesh: skeleton
                    for a, b in HAND_CONNECTIONS:
                        _line(img, view, J[t, a], J[t, b], col, 2)
        if polys:                                                          # both hands, far triangles first
            P_, C_, Z_ = np.concatenate(polys), np.concatenate(cols), np.concatenate(depths)
            for k in np.argsort(-Z_):
                cv2.fillConvexPoly(img, P_[k].astype(np.int32), tuple(int(c) for c in C_[k]), cv2.LINE_AA)
        cv2.putText(img, f"room view ({world['backend']}): head, gaze, body axis, hands", (10, S - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
        vw.write(np.hstack([left, img]))
    vw.release()
    cap.release()
    log.info("world render: %s", out_path)
    return out_path
