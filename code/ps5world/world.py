"""Assemble the world view: hands, head (camera), gaze and body axis in one gravity-aligned room frame.

Steps (the HaWoR / EgoAllo recipe, adapted to our fisheye camera and to PS4's hands):
 1. re-lift PS4 hands into the SLAM camera: keep PS4's hand shape, re-aim the wrist along the SLAM
    camera's own ray for the same pixel, and rescale its distance by the ratio of the two focal lengths
    (a hand's distance from its apparent size is proportional to the focal length)
 2. camera -> world with the SLAM poses
 3. gravity: GeoCalib gravity if available, refined by the dominant horizontal plane of the map
    (bench / floor); fallback: the head's average up axis
 4. re-express everything in a frame with Z up, Y = the worker's average facing, origin at the first head
    position
 5. body axis: a torso under the head along gravity, facing the head's heading smoothed over ~1 s
    (the body turns slower than the head); gaze = the camera's forward axis
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .cameras import EquidistantCamera, rotation_between

log = logging.getLogger(__name__)
SIDES = ("left", "right")


def relift_hands(hands: dict, ps4_cam: EquidistantCamera, cam) -> tuple[dict, float, dict]:
    """PS4 camera-frame hands -> SLAM camera frame. Returns ({side: (N,21,3)}, scale, {side: (N,V,3) or None}).
    The hand mesh (if PS4 exported one) gets exactly the joints' move: rotate about the wrist + new wrist position."""
    same = isinstance(cam, EquidistantCamera) and abs(cam.fx - ps4_cam.fx) < 1e-6
    s = 1.0 if same else cam.f_centre / ps4_cam.f_centre
    out, mesh = {}, {}
    for side in SIDES:
        J, kp, src = hands[side]["J_cam"], hands[side]["kp2d"], hands[side]["src"]
        V = hands[side].get("V_cam")
        Jn = np.full_like(J, np.nan)
        Vn = np.full_like(V, np.nan) if V is not None else None
        for t in np.nonzero(src > 0)[0]:
            j = J[t]
            if not np.isfinite(j).all():
                continue
            d = float(np.linalg.norm(j[0]))
            r_p = j[0] / max(d, 1e-9)
            if same:
                Q, w_new = np.eye(3), j[0]
            else:
                r_v = cam.rays(kp[t, 0]) if np.isfinite(kp[t, 0]).all() else np.full(3, np.nan)
                if not np.isfinite(r_v).all():
                    r_v = r_p
                Q, w_new = rotation_between(r_p, r_v), s * d * r_v
            Jn[t] = (j - j[0]) @ Q.T + w_new
            if Vn is not None and np.isfinite(V[t, 0]).all():
                Vn[t] = (V[t] - j[0]) @ Q.T + w_new
        out[side] = Jn
        mesh[side] = Vn
    return out, float(s), mesh


def to_world(J_cam: np.ndarray, R_wc: np.ndarray, t_wc: np.ndarray) -> np.ndarray:
    return np.einsum("nij,nkj->nki", R_wc, J_cam) + t_wc[:, None, :]


# ---- gravity ---------------------------------------------------------------------------------------
def _ransac_planes(P: np.ndarray, k: int = 3, iters: int = 400, thr: float = 0.02, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    planes, pts = [], P.copy()
    for _ in range(k):
        if len(pts) < 500:
            break
        best_in = None
        for _ in range(iters):
            a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
            n = np.cross(b - a, c - a)
            if np.linalg.norm(n) < 1e-9:
                continue
            n /= np.linalg.norm(n)
            inl = np.abs((pts - a) @ n) < thr
            if best_in is None or inl.sum() > best_in.sum():
                best_in = inl
        if best_in is None or best_in.sum() < 0.08 * len(P):
            break
        q = pts[best_in]
        _, _, vt = np.linalg.svd(q - q.mean(0), full_matrices=False)    # refine on inliers
        planes.append((vt[-1], int(best_in.sum())))
        pts = pts[~best_in]
    return planes


def estimate_up(R_wc: np.ndarray, points: np.ndarray, gravity_world: np.ndarray | None) -> tuple[np.ndarray, str]:
    """Unit 'up' vector in the SLAM world frame and how it was found."""
    head_up = -R_wc[:, :, 1].mean(0)                              # camera -y, averaged over the clip
    head_up /= np.linalg.norm(head_up)
    ref, how = (gravity_world, "geocalib") if gravity_world is not None else (head_up, "head average")
    if len(points) >= 2000:
        P = points[np.isfinite(points).all(1)]
        d = np.linalg.norm(P - np.median(P, 0), axis=1)
        P = P[d < np.percentile(d, 90)]
        if len(P) > 30000:
            P = P[np.random.default_rng(0).choice(len(P), 30000, replace=False)]
        scale = float(np.median(np.linalg.norm(P - P.mean(0), axis=1)))
        planes = _ransac_planes(P, thr=max(0.01, 0.02 * scale))
        lim = np.cos(np.radians(15 if gravity_world is not None else 40))
        best = max(planes, key=lambda p: abs(p[0] @ ref), default=None)
        if best is not None and abs(best[0] @ ref) > lim:
            n = best[0] * np.sign(best[0] @ ref)
            return n, how + " + map plane"
    return ref, how


def geocalib_gravity(video: str, R_wc: np.ndarray, n: int = 8, device: str = "cuda") -> np.ndarray | None:
    """Mean gravity ('up') direction in the world frame from GeoCalib on a few frames; None if unavailable."""
    try:
        import cv2
        import torch
        from geocalib import GeoCalib
        dev = device if torch.cuda.is_available() else "cpu"
        model = GeoCalib().to(dev)
        cap = cv2.VideoCapture(video)
        N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ups = []
        for idx in np.linspace(0.05 * N, 0.95 * N, n).astype(int):
            if idx >= len(R_wc):
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, img = cap.read()
            if not ok:
                continue
            x = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255).to(dev)
            res = model.calibrate(x, camera_model="simple_divisional")
            g = res["gravity"]
            v = g.vec3d if hasattr(g, "vec3d") else g
            v = np.asarray(v.detach().cpu() if hasattr(v, "detach") else v, float).reshape(-1)[:3]
            v = R_wc[idx] @ (v / np.linalg.norm(v))
            ups.append(v)
        cap.release()
        if len(ups) < 3:
            return None
        U = np.array(ups)
        head_up = -R_wc[:, :, 1].mean(0)
        U *= np.sign(U @ head_up)[:, None]                         # GeoCalib's sign convention: align with the head's up
        m = U.mean(0)
        m /= np.linalg.norm(m)
        if np.degrees(np.arccos(np.clip(U @ m, -1, 1))).max() > 20:  # inconsistent: don't trust it
            return None
        return m
    except Exception as exc:                                         # noqa: BLE001 - optional
        log.info("GeoCalib gravity unavailable (%s)", exc)
        return None


# ---- assembly ------------------------------------------------------------------------------------------
def build_world(hands: dict, ps4_cam: dict, traj: dict, fps: float, gravity_world: np.ndarray | None = None,
                body: dict | None = None) -> dict:
    body = body or {}
    N = hands["n_frames"]
    R_wc, t_wc = traj["R_wc"][:N], traj["t_wc"][:N]
    pcam = EquidistantCamera.from_ps4(ps4_cam)
    J_slam, scale, V_slam = relift_hands(hands, pcam, traj["cam"])
    up, up_how = estimate_up(R_wc, traj["points"], gravity_world)
    # frame: Z up, Y = mean horizontal facing, origin = first head position
    fwd = R_wc[:, :, 2]
    fh = fwd - np.outer(fwd @ up, up)
    yax = fh.mean(0)
    if np.linalg.norm(yax) < 1e-6:
        yax = np.cross(up, [1.0, 0, 0])
    yax -= (yax @ up) * up
    yax /= np.linalg.norm(yax)
    xax = np.cross(yax, up)
    A = np.stack([xax, yax, up])                                    # world -> aligned
    o = t_wc[0].copy()
    tf = lambda P: (np.asarray(P) - o) @ A.T                         # noqa: E731
    head_R = np.einsum("ij,njk->nik", A, R_wc)                      # camera -> aligned
    head_p = tf(t_wc)
    hands_w = {s: tf(to_world(J_slam[s], R_wc, t_wc)) for s in SIDES}
    faces = hands.get("faces")
    mesh_w = {s: (tf(to_world(V_slam[s], R_wc, t_wc)).astype(np.float32) if V_slam.get(s) is not None
                  and faces is not None else None) for s in SIDES}
    pts = tf(traj["points"]) if len(traj["points"]) else np.zeros((0, 3))
    # body axis
    sig = max(1.0, body.get("facing_smooth_s", 1.0) * fps)
    f = head_R[:, :, 2].copy()
    f[:, 2] = 0
    f = gaussian_filter1d(f, sig, axis=0, mode="nearest")
    f /= np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-9)
    hp = gaussian_filter1d(head_p, max(1.0, 0.5 * fps), axis=0, mode="nearest")
    neck = hp - np.array([0, 0, body.get("neck_drop_m", 0.22)])
    pelvis = hp - np.array([0, 0, body.get("pelvis_drop_m", 0.62)])
    right = np.cross(f, [0, 0, 1.0])
    sh = body.get("shoulder_half_m", 0.19)
    return {"N": N, "fps": fps, "head_R": head_R, "head_p": head_p, "gaze": head_R[:, :, 2],
            "valid": traj["valid"][:N], "facing": f, "neck": neck, "pelvis": pelvis,
            "shoulders": np.stack([neck - sh * right, neck + sh * right], 1),
            "hands": hands_w, "mesh": mesh_w, "faces": faces, "src": {s: hands[s]["src"] for s in SIDES}, "points": pts,
            "colors": traj["colors"], "scale": scale, "up_method": up_how, "backend": traj["backend"],
            "cam": traj["cam"], "align": A, "origin": o}


def checks(world: dict, traj: dict, ps4_head_R: np.ndarray | None) -> dict:
    """Automatic sanity numbers (no labels needed)."""
    N = world["N"]
    out = {"backend": world["backend"], "up_method": world["up_method"], "hand_scale_factor": round(world["scale"], 3),
           "frames": N, "frames_with_pose": int(world["valid"].sum())}
    cam = world["cam"]
    out["camera"] = {k: round(float(getattr(cam, k)), 3) for k in ("fx", "fy", "cx", "cy") if hasattr(cam, k)}
    if hasattr(cam, "xi"):
        out["camera"]["xi"] = round(float(cam.xi), 4)
    p = world["head_p"]
    out["head_path_m"] = round(float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()), 2)
    out["head_max_from_start_m"] = round(float(np.linalg.norm(p - p[0], axis=1).max()), 2)
    for s in SIDES:
        W = world["hands"][s][:, 0]
        ok = np.isfinite(W).all(1) & (world["src"][s] > 0) & (world["src"][s] < 3)
        if ok.any():
            d = np.linalg.norm(W[ok] - p[ok], axis=1)
            out[f"{s}_hand_to_head_median_m"] = round(float(np.median(d)), 3)
    if ps4_head_R is not None and traj["backend"] != "rotation":
        R = traj["R_wc"][:N]
        rel = np.einsum("ji,njk->nik", R[0], R)                          # camera_t -> camera_0 (SLAM)
        C = np.asarray(ps4_head_R)[:N]
        M = np.einsum("nji,njk->nik", rel, C)
        ang = np.degrees(np.arccos(np.clip((np.trace(M, axis1=1, axis2=2) - 1) / 2, -1, 1)))
        out["rotation_vs_ps4_median_deg"] = round(float(np.median(ang[world["valid"][:len(ang)]])), 2)
        k = max(1, int(round(world["fps"])))                             # over 1 s windows (PS4 drifts slowly)
        if len(C) > k:
            dS = np.einsum("nji,njk->nik", rel[:-k], rel[k:])
            dP = np.einsum("nji,njk->nik", C[:-k], C[k:])
            M1 = np.einsum("nji,njk->nik", dS, dP)
            a1 = np.degrees(np.arccos(np.clip((np.trace(M1, axis1=1, axis2=2) - 1) / 2, -1, 1)))
            out["rotation_1s_vs_ps4_median_deg"] = round(float(np.median(a1)), 2)
    out["map_points"] = int(len(world["points"]))
    return out
