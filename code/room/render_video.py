"""Fly-through video of the room twin with the worker's hands (meshes), head and gaze replayed inside it.
Ray casting on the CPU (Open3D/Embree): no OpenGL or display needed on Colab."""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from ps4v4 import HAND_CONNECTIONS
from ps4v4.render import _Writer

log = logging.getLogger("room")
COL = {"left": np.array([60, 200, 60.0]), "right": np.array([255, 140, 0.0])}     # RGB
HEAD = (120, 220, 255)
GAZE = (255, 230, 60)


def load_motion(slam_dir: str) -> dict:
    """world_hands.jsonl (+ world_hands_mesh.npz) of the target clip, in its room frame."""
    rows = [json.loads(l) for l in open(Path(slam_dir) / "world_hands.jsonl")]
    N = len(rows)
    m = {"N": N, "fps": 1.0 / max(rows[1]["t"] - rows[0]["t"], 1e-6) if N > 1 else 30.0,
         "head": np.array([r["head"]["position"] for r in rows], float),
         "quat": np.array([r["head"]["quat_xyzw"] for r in rows], float),
         "gaze": np.array([r["head"]["gaze_dir"] for r in rows], float),
         "joints": {s: np.full((N, 21, 3), np.nan) for s in ("left", "right")},
         "src": {s: np.zeros(N, int) for s in ("left", "right")}, "faces": None,
         "verts": {"left": None, "right": None}}
    code = {"observed": 1, "infilled": 2, "estimated": 3}
    for t, r in enumerate(rows):
        for h in r["hands"]:
            m["joints"][h["side"]][t] = h["joints"]
            m["src"][h["side"]][t] = code.get(h["source"], 1)
    p = Path(slam_dir) / "world_hands_mesh.npz"
    if p.exists():
        d = np.load(p)
        m["faces"] = d["faces"].astype(np.int32)
        for s in ("left", "right"):
            if f"{s}_frames" in d and len(d[f"{s}_frames"]):
                V = np.full((N, d[f"{s}_rel"].shape[1], 3), np.nan, np.float32)
                fr = d[f"{s}_frames"]
                keep = fr < N
                V[fr[keep]] = d[f"{s}_rel"][keep].astype(np.float32) + d[f"{s}_wrist"][keep][:, None]
                m["verts"][s] = V
    return m


def _look_at(eye, target):
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    up = np.array([0, 0, 1.0])
    if abs(f @ up) > 0.995:
        up = np.array([0, 1.0, 0])
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    return np.stack([r, d, f], 1)                                       # camera -> room (columns x, y, z)


def camera_path(m: dict, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Eye + target per frame: over the shoulder -> left -> right -> top -> close-up -> worker's eye -> back.
    Orbit around the hands' working area, always on the worker's side of the bench (where the room was filmed)."""
    H = np.nanmedian(m["head"][:n], 0)
    W = np.concatenate([m["joints"][s][:n, 0] for s in ("left", "right")])
    W = W[np.isfinite(W).all(1)]
    C = np.median(W, 0) if len(W) > 10 else H + np.array([0, 0.45, -0.35])
    back = H[:2] - C[:2]
    back = back / np.linalg.norm(back) if np.linalg.norm(back) > 0.05 else np.array([0, -1.0])
    R_eye = float(np.linalg.norm(H[:2] - C[:2]))
    h_eye = float(H[2] - C[2])
    # (time fraction, orbit angle deg, radius m, height above the hands m)
    keys = [(0.00, 0, 0.80, 0.50), (0.17, -55, 1.00, 0.55), (0.37, 55, 1.00, 0.55), (0.52, 0, 0.35, 1.20),
            (0.66, -25, 0.55, 0.30), (0.82, 0, max(R_eye, 0.25), max(h_eye, 0.2)), (1.00, 0, 0.80, 0.50)]
    T = np.array([k[0] for k in keys])
    eye, tgt = np.zeros((n, 3)), np.repeat(C[None], n, 0)
    for i in range(n):
        x = i / max(n - 1, 1)
        j = min(np.searchsorted(T, x, side="right") - 1, len(keys) - 2)
        a = (x - T[j]) / (T[j + 1] - T[j])
        a = 0.5 - 0.5 * math.cos(math.pi * a)                           # ease in / out
        th, R, h = [(1 - a) * keys[j][k] + a * keys[j + 1][k] for k in (1, 2, 3)]
        c, s_ = math.cos(math.radians(th)), math.sin(math.radians(th))
        dxy = np.array([c * back[0] - s_ * back[1], s_ * back[0] + c * back[1]])
        eye[i] = [C[0] + R * dxy[0], C[1] + R * dxy[1], C[2] + h]
    return eye, tgt


class RoomRenderer:
    def __init__(self, mesh, width: int = 1280, height: int = 720, fov_deg: float = 70.0):
        import open3d as o3d
        self.o3d = o3d
        self.W, self.H = width, height
        self.f = (width / 2) / math.tan(math.radians(fov_deg) / 2)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        self.tri = np.asarray(mesh.triangles)
        self.vcol = (np.clip(np.asarray(mesh.vertex_colors), 0, 1) * 255).astype(np.float32)
        u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
        self.dirs_cam = np.stack([(u - width / 2) / self.f, (v - height / 2) / self.f, np.ones_like(u)], -1)
        self.dirs_cam = (self.dirs_cam / np.linalg.norm(self.dirs_cam, axis=-1, keepdims=True)).astype(np.float32)
        g = np.linspace(38, 18, height, dtype=np.float32)[:, None, None]
        self.bg = np.repeat(np.repeat(g, width, 1), 3, 2)

    def rays(self, eye, Rc):
        d = self.dirs_cam @ Rc.T.astype(np.float32)
        o = np.broadcast_to(np.asarray(eye, np.float32), d.shape)
        return self.o3d.core.Tensor(np.concatenate([o, d], -1).astype(np.float32))

    def render(self, eye, Rc, hands: list) -> tuple[np.ndarray, np.ndarray]:
        """RGB uint8 + depth along the ray. hands: [(V (n,3), faces, rgb colour)]."""
        o3d = self.o3d
        rays = self.rays(eye, Rc)
        r = self.scene.cast_rays(rays)
        t = r["t_hit"].numpy()
        hit = np.isfinite(t)
        img = self.bg.copy()
        pid = r["primitive_ids"].numpy()[hit]
        uv = r["primitive_uvs"].numpy()[hit]
        tri = self.tri[pid]
        w0 = 1 - uv[:, 0] - uv[:, 1]
        col = self.vcol[tri[:, 0]] * w0[:, None] + self.vcol[tri[:, 1]] * uv[:, :1] + self.vcol[tri[:, 2]] * uv[:, 1:]
        img[hit] = col
        depth = np.where(hit, t, np.inf)
        if hands:
            hs = o3d.t.geometry.RaycastingScene()
            cols = []
            for V, F, c in hands:
                hs.add_triangles(o3d.core.Tensor(np.ascontiguousarray(V, np.float32)),
                                 o3d.core.Tensor(np.ascontiguousarray(F, np.uint32)))
                cols.append(c)
            rh = hs.cast_rays(rays)
            th = rh["t_hit"].numpy()
            front = np.isfinite(th) & (th < depth)
            if front.any():
                gid = rh["geometry_ids"].numpy()[front]
                n = rh["primitive_normals"].numpy()[front]
                d = rays.numpy()[..., 3:][front]
                lam = np.abs((n * d).sum(1))
                base = np.asarray(cols, np.float32)[gid.astype(int)]
                img[front] = np.clip(base * (0.35 + 0.75 * lam[:, None]), 0, 255)
                depth[front] = th[front]
        return img.astype(np.uint8), depth

    def project(self, P, eye, Rc):
        X = (np.asarray(P, float) - eye) @ Rc
        z = X[..., 2]
        uv = np.stack([self.f * X[..., 0] / np.maximum(z, 1e-6) + self.W / 2,
                       self.f * X[..., 1] / np.maximum(z, 1e-6) + self.H / 2], -1)
        return uv, z


def _line(img, R, eye, Rc, a, b, col, th=2):
    uv, z = R.project(np.array([a, b]), eye, Rc)
    if (z > 0.05).all() and np.isfinite(uv).all() and np.abs(uv).max() < 1e5:
        cv2.line(img, tuple(uv[0].astype(int)), tuple(uv[1].astype(int)), col, th, cv2.LINE_AA)


def render_room_video(mesh_path: str, slam_dir: str, video: str, out_path: str, seconds: float = 120.0,
                      width: int = 1280, height: int = 720, inset: bool = True, title: str = "") -> dict:
    import open3d as o3d
    t0 = time.time()
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    m = load_motion(slam_dir)
    n = min(m["N"], int(round(seconds * m["fps"])))
    eye, tgt = camera_path(m, n)
    R = RoomRenderer(mesh, width, height)
    cap = cv2.VideoCapture(video) if inset else None
    vw = _Writer(out_path, m["fps"], (width, height), 20)
    corners = np.array([[-1, -0.6, 1], [1, -0.6, 1], [1, 0.6, 1], [-1, 0.6, 1.0]]) * 0.09
    trail = int(m["fps"])
    used_mesh = 0
    for t in range(n):
        Rc = _look_at(eye[t], tgt[t])
        hands = []
        for s in ("left", "right"):
            V = m["verts"][s][t] if m["verts"][s] is not None else None
            if m["faces"] is not None and V is not None and np.isfinite(V).all() and m["src"][s][t] in (1, 2):
                hands.append((V, m["faces"], COL[s]))
        used_mesh += bool(hands)
        img, _ = R.render(eye[t], Rc, hands)
        img = np.ascontiguousarray(img[..., ::-1])                     # BGR for OpenCV / the writer
        meshed = {s for s in ("left", "right") if m["verts"][s] is not None and t < len(m["verts"][s])
                  and np.isfinite(m["verts"][s][t]).all() and m["src"][s][t] in (1, 2)}
        for s in ("left", "right"):
            J = m["joints"][s]
            col = tuple(int(c) for c in COL[s][::-1])
            tr = J[max(0, t - trail):t + 1, 0]
            for i in range(1, len(tr)):
                if np.isfinite(tr[i - 1]).all() and np.isfinite(tr[i]).all():
                    _line(img, R, eye[t], Rc, tr[i - 1], tr[i], col, max(1, int(3 * i / len(tr))))
            if s not in meshed and m["src"][s][t] > 0 and np.isfinite(J[t]).all():     # no mesh: skeleton
                dim = col if m["src"][s][t] in (1, 2) else tuple(int(0.5 * c) for c in col)
                for a, b in HAND_CONNECTIONS:
                    _line(img, R, eye[t], Rc, J[t, a], J[t, b], dim, 2)
        hp = m["head"][t]
        if np.isfinite(hp).all() and np.isfinite(m["quat"][t]).all():
            Rh = Rotation.from_quat(m["quat"][t]).as_matrix()
            cw = hp + corners @ Rh.T
            for i in range(4):
                _line(img, R, eye[t], Rc, hp, cw[i], HEAD[::-1], 2)
                _line(img, R, eye[t], Rc, cw[i], cw[(i + 1) % 4], HEAD[::-1], 2)
            _line(img, R, eye[t], Rc, hp, hp + 0.6 * m["gaze"][t], GAZE[::-1], 2)
        if cap is not None:
            ok, fr = cap.read()
            if ok:
                iw = width // 4
                ih = int(round(fr.shape[0] * iw / fr.shape[1]))
                small = cv2.resize(fr, (iw, ih), interpolation=cv2.INTER_AREA)
                img[height - ih - 12:height - 12, width - iw - 12:width - 12] = small
                cv2.rectangle(img, (width - iw - 13, height - ih - 13), (width - 12, height - 12), (230, 230, 230), 1)
        cv2.putText(img, f"{title}  t={t / m['fps']:6.2f}s", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (235, 235, 235), 2, cv2.LINE_AA)
        vw.write(img)
        if t and t % 300 == 0:
            log.info("render %d/%d frames (%.0f s)", t, n, time.time() - t0)
    vw.release()
    if cap is not None:
        cap.release()
    return {"frames": n, "frames_with_hand_mesh": int(used_mesh), "seconds": round(time.time() - t0, 1),
            "size": [width, height]}
