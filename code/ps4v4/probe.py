"""M3P: probe 3D. Run the hand model on a sample of each tracklet's frames to get evidence
the 2D detector cannot give:

  * wrist depth          -> the wearer's hands are within arm's reach; another person's are far
  * forearm direction    -> which side the arm comes from (Betancourt et al. 2016: arm angle to
                            the bottom edge separates left from right far better than x position)
  * forearm toward me?   -> the wearer's forearm runs back toward the camera; a person facing the
                            wearer has the forearm running away from it
  * keypoints vs box     -> a hand model fitted to junk rarely fills the detector box

The probe results are reused by M4 when the side matches, so they are not wasted work.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from .models.reconstructor import make_reconstructor
from .models.verifier import make_verifier, square_crop
from .video import iter_frames
from .proposals import box_iou
from .tracklets import SRC_DET

log = logging.getLogger(__name__)


def pick_probe_frames(tr: dict, stride: int, max_n: int) -> np.ndarray:
    det = np.nonzero(tr["src"] == SRC_DET)[0]
    if len(det) == 0:
        return det
    pick = det[::stride]
    if len(pick) > max_n:
        pick = pick[np.linspace(0, len(pick) - 1, max_n).round().astype(int)]
    return np.unique(np.concatenate([pick, det[-1:]]))


def tracklet_side_guess(tr: dict) -> str:
    p = tr["p_right"][tr["src"] == SRC_DET]
    return "right" if (np.nanmean(p) if len(p) else 0.5) >= 0.5 else "left"


def run_probe(meta, cam, tracklets: list[dict], cfg, cache_dir) -> dict:
    """Returns {tracklet_id: {"side": str, "idx": (k,) indices into the tracklet, "frames", "joints",
    "kp2d", "verts", "valid"}}."""
    pc = cfg.probe
    t0 = time.time()
    req = {"left": ([], []), "right": ([], [])}
    owners = {"left": [], "right": []}
    for tr in tracklets:
        if (tr["src"] == SRC_DET).sum() < pc.min_det:
            continue
        idx = pick_probe_frames(tr, pc.stride, pc.max_per_tracklet)
        side = tracklet_side_guess(tr)
        for i in idx:
            req[side][0].append(int(tr["frames"][i])); req[side][1].append(tr["boxes"][i])
            owners[side].append((tr["id"], int(i)))
    requests = {s: (np.array(f, int), np.array(b, float).reshape(-1, 4)) for s, (f, b) in req.items()}
    pcfg = cfg
    if cfg.recon.backend == "egohand":                     # EgoHand takes one box per frame and side: probe with WiLoR
        from .config import Config, deep_update
        pcfg = Config(deep_update(cfg, {"recon": {"backend": "wilor"}}))
    rec = make_reconstructor(pcfg, cache_dir).reconstruct(meta, cam, requests) if any(
        len(v[0]) for v in requests.values()) else {"left": None, "right": None}
    out: dict = {}
    for side in ("left", "right"):
        r = rec.get(side)
        for k, (tid, i) in enumerate(owners[side]):
            d = out.setdefault(tid, {"side": side, "idx": [], "joints": [], "kp2d": [], "verts": [], "valid": []})
            d["idx"].append(i)
            d["joints"].append(r["joints"][k]); d["kp2d"].append(r["kp2d"][k]); d["valid"].append(r["valid"][k])
            d["verts"].append(r["verts"][k] if "verts" in r else None)
    for d in out.values():
        d["idx"] = np.array(d["idx"], int)
        d["joints"], d["kp2d"], d["valid"] = np.array(d["joints"]), np.array(d["kp2d"]), np.array(d["valid"])
        d["verts"] = np.array(d["verts"], np.float32) if d["verts"][0] is not None else None
    _verify(meta, tracklets, out, cfg)
    n = sum(len(v[0]) for v in requests.values())
    log.info("M3P probe: %d crops from %d tracklets in %.0f s", n, len(out), time.time() - t0)
    return {"probe": out, "faces": rec.get("faces")}


def _verify(meta, tracklets: list[dict], out: dict, cfg) -> None:
    """Adds p_hand / other_name per probe frame (semantic check: hand vs knee, foot, wood, ...)."""
    ver = make_verifier(cfg)
    by_id = {tr["id"]: tr for tr in tracklets}
    for d in out.values():
        d["p_hand"] = np.full(len(d["idx"]), np.nan)
        d["other"] = [""] * len(d["idx"])
    if ver is None:
        return
    jobs: dict[int, list] = {}
    for tid, d in out.items():
        tr = by_id[tid]
        for k, i in enumerate(d["idx"]):
            jobs.setdefault(int(tr["frames"][i]), []).append((tid, k, tr["boxes"][i]))
    if not jobs:
        return
    crops, keys = [], []
    last = max(jobs)
    for f, img in iter_frames(meta):
        if f > last:
            break
        for tid, k, b in jobs.get(f, []):
            crops.append(square_crop(img, b, cfg.probe.crop_pad)); keys.append((tid, k))
    p, other = ver.score(crops)
    for (tid, k), ph, o in zip(keys, p, other):
        out[tid]["p_hand"][k] = ph
        out[tid]["other"][k] = o


# ---- per-frame evidence from a probe result ---------------------------------------
def probe_features(joints: np.ndarray, kp2d: np.ndarray, box: np.ndarray, W: int, H: int) -> dict:
    """Evidence from one reconstructed frame (camera coordinates: x right, y down, z forward)."""
    wrist, knuckles = joints[0], joints[[5, 9, 13, 17]].mean(0)
    d3 = wrist - knuckles                                  # continues into the forearm
    d3 = d3 / max(np.linalg.norm(d3), 1e-9)
    w2, k2 = kp2d[0], kp2d[[5, 9, 13, 17]].mean(0)
    d2 = w2 - k2
    n2 = np.linalg.norm(d2)
    arm_dx = float(d2[0] / n2) if n2 > 3 else 0.0         # +: forearm runs to the right (right arm)
    exit_y, arm_exit = np.nan, 0.0
    if n2 > 3:
        u = d2 / n2
        ts = [tt for tt in ((W - w2[0]) / u[0] if u[0] > 1e-6 else np.inf,
                            -w2[0] / u[0] if u[0] < -1e-6 else np.inf,
                            (H - w2[1]) / u[1] if u[1] > 1e-6 else np.inf,
                            -w2[1] / u[1] if u[1] < -1e-6 else np.inf) if tt >= 0]
        if ts:
            ex = w2 + u * min(ts)                           # where the forearm leaves the image
            exit_y = float(ex[1] / H)                       # 0 = top
            # side vote: a left forearm leaves through the left edge or the bottom left of the centre, a right
            # one through the right edge or bottom right. (The raw left/right lean of the forearm is not used:
            # a left hand reaching forward-left has its forearm leaning RIGHT, toward the body.)
            if ex[0] <= 1 and ex[1] < H - 1:
                arm_exit = -1.0
            elif ex[0] >= W - 1 and ex[1] < H - 1:
                arm_exit = 1.0
            elif ex[1] >= H - 1:
                arm_exit = float(np.clip((ex[0] - W / 2) / (0.25 * W), -1, 1))
    k = kp2d[np.isfinite(kp2d).all(1)]
    if len(k):
        kb = np.array([k[:, 0].min(), k[:, 1].min(), k[:, 0].max(), k[:, 1].max()])
        c, s = (kb[:2] + kb[2:]) / 2, (kb[2:] - kb[:2]) * 0.6 + 4
        kp_iou = box_iou(np.concatenate([c - s, c + s]), box)
    else:
        kp_iou = 0.0
    return {"z": float(wrist[2]), "dist": float(np.linalg.norm(wrist)), "toward_z": float(d3[2]),
            "arm_dx": arm_dx, "arm_exit": arm_exit, "exit_y": exit_y, "kp_iou": float(kp_iou)}
