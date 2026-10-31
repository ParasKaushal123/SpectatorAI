"""M5: sequence step. Turns noisy per-frame 3D into stable, believable hand motion.

Per side:
 1. move joints into the head-stable frame (head rotation removed)
 2. reject implausible frames: depth range, Hampel outliers on the wrist, impossible speed
 3. fix bone lengths to the clip median (a hand does not change size)
 4. decompose: wrist position + hand orientation (quaternion) + finger joints in the hand frame
 5. confidence-weighted local-quadratic smoothing with a window that shrinks during fast motion
    (clear frames repair blurred ones; fast moves stay sharp), infilling gaps up to infill_max_s
 6. compose back; label every frame observed / infilled / none
 7. (v4) out-of-view estimate: a short gap between two segments of the same hand is bridged by a
    smooth wrist path (cubic Hermite in the head-stable frame, kept within arm's reach), labelled
    "estimated". It is a guess, never counted as observed, and drawn faded.
v4 also: smoothing never mixes two separate segments, and the depth gate is relative to the
wearer's typical hand distance (robust to a wrong focal length).
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.spatial.transform import Rotation

from . import BONES, SIDES
from .geometry import hand_frame, quat_continuity

log = logging.getLogger(__name__)
SRC_NONE, SRC_OBSERVED, SRC_INFILLED, SRC_ESTIMATED = 0, 1, 2, 3


def hampel_mask(x: np.ndarray, window: int, k: float) -> np.ndarray:
    """True where x (n,d) deviates > k robust sigmas from its windowed median (any dimension)."""
    n = len(x)
    bad = np.zeros(n, bool)
    h = window // 2
    for i in range(n):
        seg = x[max(0, i - h): i + h + 1]
        med = np.median(seg, 0)
        mad = 1.4826 * np.median(np.abs(seg - med), 0) + 1e-3        # 1 mm floor
        bad[i] = np.any(np.abs(x[i] - med) > k * mad)
    return bad


def _bone_fix(L: np.ndarray, lengths: dict) -> np.ndarray:
    out = L.copy()
    for a, b in BONES:
        d = L[:, b] - L[:, a]
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        out[:, b] = out[:, a] + d * lengths[(a, b)]
    return out


def _local_quadratic(t_query: float, ts: np.ndarray, ys: np.ndarray, ws: np.ndarray) -> np.ndarray | None:
    """Weighted quadratic (linear if few points) fit around t_query; returns value at t_query."""
    if ws.sum() <= 1e-9 or len(ts) < 2:
        return None
    dt = ts - t_query
    order = 2 if len(ts) >= 5 else 1
    A = np.stack([dt ** p for p in range(order + 1)], 1)
    sw = np.sqrt(ws)[:, None]
    coef, *_ = np.linalg.lstsq(A * sw, ys * sw, rcond=None)
    return coef[0]


def smooth_side(J: np.ndarray, conf: np.ndarray, valid: np.ndarray, present: np.ndarray, fps: float, cfg,
                verts: np.ndarray | None = None) -> dict:
    """J (N,21,3) head-stable joints; returns smoothed joints, quaternions, source labels."""
    sc = cfg.sequence
    N = len(J)
    seg = np.cumsum(np.diff(np.concatenate([[0], present.astype(int)])) == 1)   # id of each presence run
    w = np.where(valid, np.clip(conf, 0, 1), 0.0)
    w[w < sc.min_conf] = 0.0
    # 2) gates ---------------------------------------------------------------------
    idx = np.nonzero(w > 0)[0]
    if len(idx) >= 5:
        bad = hampel_mask(J[idx, 0], sc.hampel_window, sc.hampel_k)
        w[idx[bad]] = 0.0
        for _ in range(3):                                         # impossible speed between kept frames
            idx = np.nonzero(w > 0)[0]
            if len(idx) < 3:
                break
            step = np.linalg.norm(np.diff(J[idx, 0], axis=0), axis=1) / (np.diff(idx) / fps)
            fast = np.nonzero(step > sc.v_max_mps)[0]
            if not len(fast):
                break
            for f in fast:                                         # drop the less trusted end of each jump
                a, b = idx[f], idx[f + 1]
                w[a if w[a] < w[b] else b] = 0.0
    keep = w > 0
    out = {"J": np.full((N, 21, 3), np.nan), "quat": np.full((N, 4), np.nan), "src": np.zeros(N, np.int8),
           "verts": None}
    if keep.sum() < 3:
        return out
    # 3-4) decomposition + fixed bones -------------------------------------------
    R = hand_frame(J[keep])                                        # columns = hand axes
    p = J[keep, 0]
    L = np.einsum("nji,nkj->nki", R, J[keep] - p[:, None])         # R^T (x - p)
    lengths = {bn: float(np.median(np.linalg.norm(L[:, bn[1]] - L[:, bn[0]], axis=1))) for bn in BONES}
    L = _bone_fix(L, lengths)
    q = quat_continuity(Rotation.from_matrix(R).as_quat())
    kidx = np.nonzero(keep)[0]
    feats = np.concatenate([p, q, L.reshape(len(L), -1)], 1)       # 3 + 4 + 63
    wk = w[keep]
    # adaptive window from the wrist speed around each frame
    speed = np.zeros(len(kidx))
    if len(kidx) > 1:
        sp = np.linalg.norm(np.diff(p, axis=0), axis=1) / (np.diff(kidx) / fps)
        speed[1:] = sp; speed[0] = sp[0]
    h_of = lambda s: int(round(sc.h_max - (sc.h_max - sc.h_min) * min(1.0, s / sc.speed_ref_mps)))
    infill_max = int(round(sc.infill_max_s * fps))
    # 5) evaluate at every observed frame and every short-gap frame ----------------
    targets = list(kidx)
    for a, b in zip(kidx[:-1], kidx[1:]):
        if 1 < b - a <= infill_max + 1 and present[a + 1:b].all():
            targets += list(range(a + 1, b))
    pos = np.searchsorted(kidx, np.array(sorted(targets)))
    Vlocal = None
    if verts is not None:
        Vlocal = np.einsum("nji,nkj->nki", R, verts[keep].astype(float) - p[:, None])
    for t, ps in zip(sorted(targets), pos):
        near = min(ps, len(kidx) - 1)
        h = h_of(speed[near])
        is_gap = keep[t] == 0
        if is_gap:                                                 # reach both sides of the gap
            left = kidx[kidx < t].max()
            right = kidx[kidx > t].min()
            h = max(h, t - left + 2, right - t + 2)
        sel = np.nonzero((np.abs(kidx - t) <= h) & (seg[kidx] == seg[t]))[0]
        tri = (1 - (np.abs(kidx[sel] - t) / (h + 1)) ** 3) ** 3
        y = _local_quadratic(t, kidx[sel].astype(float), feats[sel], wk[sel] * tri)
        if y is None:
            continue
        qq = y[3:7] / max(np.linalg.norm(y[3:7]), 1e-9)
        Rs = Rotation.from_quat(qq).as_matrix()
        Ls = _bone_fix(y[7:].reshape(1, 21, 3), lengths)[0]
        out["J"][t] = Ls @ Rs.T + y[:3]
        out["quat"][t] = qq
        out["src"][t] = SRC_INFILLED if is_gap else SRC_OBSERVED
        if Vlocal is not None:
            if out["verts"] is None:
                out["verts"] = np.full((N, Vlocal.shape[1], 3), np.nan, np.float32)
            src_i = kidx[np.argmin(np.abs(kidx - t))]            # nearest observed mesh, re-posed rigidly
            out["verts"][t] = Vlocal[np.searchsorted(kidx, src_i)] @ Rs.T + y[:3]
    return out


def estimate_gaps(J: np.ndarray, quat: np.ndarray, src: np.ndarray, fps: float, sc) -> int:
    """Bridge gaps of <= est_max_s between two observed segments of one hand (in place).
    Wrist: cubic Hermite from the last seen position/velocity to the next one, within arm's reach.
    Hand: the nearer end's hand shape, rotated by the slerp between the two end orientations.
    Returns the number of frames filled."""
    obs = np.nonzero(src > 0)[0]
    if len(obs) < 2:
        return 0
    max_gap = int(round(sc.est_max_s * fps))
    vmax = sc.v_max_mps / fps
    filled = 0
    for a, b in zip(obs[:-1], obs[1:]):
        gap = b - a
        if gap <= 1 or gap > max_gap:
            continue
        pa, pb = J[a, 0], J[b, 0]
        va = (J[a, 0] - J[a - 2, 0]) / 2 if a >= 2 and src[a - 2] > 0 else np.zeros(3)
        vb = (J[b + 2, 0] - J[b, 0]) / 2 if b + 2 < len(src) and src[b + 2] > 0 else np.zeros(3)
        va = va * min(1.0, vmax / max(np.linalg.norm(va), 1e-9))      # keep the tangents physical
        vb = vb * min(1.0, vmax / max(np.linalg.norm(vb), 1e-9))
        qa, qb = quat[a], (quat[b] if np.dot(quat[a], quat[b]) >= 0 else -quat[b])
        Ra, Rb = Rotation.from_quat(qa), Rotation.from_quat(qb)
        rel = (Ra.inv() * Rb).as_rotvec()
        La, Lb = Ra.inv().apply(J[a] - pa), Rb.inv().apply(J[b] - pb)  # hand shape in its own frame
        for t in range(a + 1, b):
            u = (t - a) / gap
            h00, h10 = 2 * u ** 3 - 3 * u ** 2 + 1, u ** 3 - 2 * u ** 2 + u
            h01, h11 = -2 * u ** 3 + 3 * u ** 2, u ** 3 - u ** 2
            p = h00 * pa + h10 * gap * va + h01 * pb + h11 * gap * vb
            r = np.linalg.norm(p)
            if r > sc.reach_max_m:                                   # stay within arm's reach of the head
                p = p * sc.reach_max_m / r
            Rt = Ra * Rotation.from_rotvec(rel * u)
            J[t] = Rt.apply(La if u < 0.5 else Lb) + p
            quat[t] = Rt.as_quat()
            src[t] = SRC_ESTIMATED
            filled += 1
    return filled


def run_sequence(meta, cam, tracks: dict, recon: dict, head_R: np.ndarray, cfg, z_ref: float | None = None) -> dict:
    sc = cfg.sequence
    N, fps = meta.n_frames, meta.fps
    C = head_R if sc.head_stable else np.repeat(np.eye(3)[None], N, 0)
    out = {"head_R": C}
    for side in SIDES:
        r = recon[side]
        J_cam = r["joints"]
        z = J_cam[:, 0, 2]
        dist = np.linalg.norm(np.nan_to_num(J_cam[:, 0], nan=1e3), axis=1)
        valid = r["valid"] & np.isfinite(z) & (z >= sc.z_range_m[0]) & (z <= sc.z_range_m[1])
        if z_ref:
            valid &= (dist >= sc.rel_range[0] * z_ref) & (dist <= sc.rel_range[1] * z_ref)
        J_h = np.einsum("nij,nkj->nki", C, np.nan_to_num(J_cam))          # camera -> head-stable
        verts_h = None
        if r.get("verts") is not None:
            verts_h = np.einsum("nij,nkj->nki", C, np.nan_to_num(r["verts"].astype(float)))
        present = tracks[side]["state"] != 0
        s = smooth_side(J_h, r["conf"], valid, present, fps, cfg, verts_h)
        n_est = estimate_gaps(s["J"], s["quat"], s["src"], fps, sc) if sc.estimate_out_of_view else 0
        Jc = np.einsum("nji,nkj->nki", C, np.nan_to_num(s["J"]))         # head-stable -> camera (C^T)
        Jc[np.isnan(s["J"][:, 0, 0])] = np.nan
        kp2d = np.full((N, 21, 2), np.nan)
        ok = ~np.isnan(Jc[:, 0, 0])
        kp2d[ok] = cam.project(Jc[ok])
        inside = (kp2d[..., 0] >= 0) & (kp2d[..., 0] < meta.width) & (kp2d[..., 1] >= 0) & (kp2d[..., 1] < meta.height)
        vc = None
        if s["verts"] is not None:
            vc = np.einsum("nji,nkj->nki", C, np.nan_to_num(s["verts"].astype(float))).astype(np.float32)
            vc[np.isnan(s["verts"][:, 0, 0])] = np.nan
        out[side] = {"J_head": s["J"], "J_cam": Jc, "quat_head": s["quat"], "src": s["src"], "kp2d": kp2d,
                     "joint_in_image": inside & ok[:, None], "verts_head": s["verts"], "verts_cam": vc,
                     "state": tracks[side]["state"], "box": tracks[side]["box"], "conf": recon[side]["conf"]}
        log.info("M5 %s: observed %d, infilled %d, estimated %d, none %d", side, (s["src"] == 1).sum(),
                 (s["src"] == 2).sum(), n_est, (s["src"] == 0).sum())
    return out
