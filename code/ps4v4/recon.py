"""M4: per-frame 3D on tracked crops only, plus a per-frame confidence.

confidence = evidence (detector score / flow ratio / SAM score) x visible fraction of the box x sharpness
"""
from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from . import SIDES, STATES
from .models.reconstructor import make_reconstructor
from .tracklets import SRC_DET, SRC_FLOW, SRC_SAM
from .video import VideoMeta, iter_frames

log = logging.getLogger(__name__)
EVIDENCE_WEIGHT = {SRC_DET: 1.0, SRC_FLOW: 0.7, SRC_SAM: 0.8}


def build_requests(tracks: dict, cfg, only: dict | None = None) -> dict:
    """Frames to reconstruct per side: tracked with real evidence in an allowed state."""
    run_states = [STATES.index(s) for s in cfg.recon.run_states]
    req = {}
    for side in SIDES:
        t = tracks[side]
        ok = np.isin(t["state"], run_states) & np.isin(t["src"], list(EVIDENCE_WEIGHT)) & np.isfinite(t["box"][:, 0])
        if only is not None:
            mask = np.zeros_like(ok)
            mask[only.get(side, [])] = True
            ok &= mask
        frames = np.nonzero(ok)[0]
        req[side] = (frames, t["box"][frames])
    return req


def crop_sharpness(meta: VideoMeta, requests: dict, blur_ref: float, floor: float) -> dict:
    """Laplacian-variance sharpness of each requested hand crop, mapped to [floor, 1].
    Blur lowers trust in a frame; it never removes the frame on its own."""
    jobs: dict[int, list] = {}
    for side, (frames, boxes) in requests.items():
        for i, (f, b) in enumerate(zip(frames, boxes)):
            jobs.setdefault(int(f), []).append((side, i, b))
    out = {s: np.ones(len(fr)) for s, (fr, _) in requests.items()}
    if not jobs:
        return out
    last = max(jobs)
    for f, frame in iter_frames(meta):
        for side, i, b in jobs.get(f, []):
            x1, y1 = int(max(b[0], 0)), int(max(b[1], 0))
            x2, y2 = int(min(b[2], meta.width)), int(min(b[3], meta.height))
            if x2 - x1 < 8 or y2 - y1 < 8:
                out[side][i] = floor
                continue
            crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            s = 128.0 / max(crop.shape)                     # compare crops at a common scale
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            out[side][i] = float(np.clip(cv2.Laplacian(crop, cv2.CV_64F).var() / blur_ref, floor, 1.0))
        if f >= last:
            break
    return out


def visible_fraction(boxes: np.ndarray, W: int, H: int) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros(0)
    iw = np.clip(np.minimum(boxes[:, 2], W) - np.maximum(boxes[:, 0], 0), 0, None)
    ih = np.clip(np.minimum(boxes[:, 3], H) - np.maximum(boxes[:, 1], 0), 0, None)
    area = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-6)
    return iw * ih / area


def _prefill_from_probe(tracks: dict, req: dict, probe: dict | None, tracklets_by_id: dict | None):
    """Split requests into frames already reconstructed by the probe (same tracklet, same side) and the rest."""
    pre = {s: {} for s in SIDES}
    if not probe:
        return req, pre
    lookup = {}
    for tid, p in probe.items():
        tr = tracklets_by_id[tid]
        for k, i in enumerate(p["idx"]):
            lookup[(tid, int(tr["frames"][i]))] = (p["side"], k)
    todo = {}
    for side, (frames, boxes) in req.items():
        keep = []
        for j, f in enumerate(frames):
            tid = int(tracks[side]["tid"][f])
            hit = lookup.get((tid, int(f)))
            if hit is not None and hit[0] == side:
                p = probe[tid]
                k = hit[1]
                pre[side][int(f)] = (p["joints"][k], p["kp2d"][k], p["valid"][k],
                                     None if p["verts"] is None else p["verts"][k])
            else:
                keep.append(j)
        keep = np.array(keep, int)
        todo[side] = (frames[keep], boxes[keep])
    return todo, pre


def run_recon(meta: VideoMeta, cam, tracks: dict, cfg, cache_dir, only: dict | None = None,
              probe: dict | None = None, tracklets_by_id: dict | None = None) -> dict:
    """Returns per side full-length arrays: joints (N,21,3), kp2d (N,21,2), valid (N,), conf (N,), verts?."""
    t0 = time.time()
    req = build_requests(tracks, cfg, only)
    todo, pre = _prefill_from_probe(tracks, req, probe if cfg.recon.backend != "egohand" else None, tracklets_by_id)
    if any(len(v[0]) for v in todo.values()):
        part = make_reconstructor(cfg, cache_dir).reconstruct(meta, cam, todo)
    else:
        part = {s: {"joints": np.zeros((0, 21, 3)), "kp2d": np.zeros((0, 21, 2)), "valid": np.zeros(0, bool)}
                for s in SIDES} | {"faces": None}
    rec = {"faces": part.get("faces")}
    for side in SIDES:                                       # stitch probe results and new results in request order
        frames = req[side][0]
        n = len(frames)
        has_verts = "verts" in part[side] or any(v[3] is not None for v in pre[side].values())
        r = {"joints": np.full((n, 21, 3), np.nan), "kp2d": np.full((n, 21, 2), np.nan), "valid": np.zeros(n, bool)}
        if has_verts:
            r["verts"] = np.full((n, 778, 3), np.nan, np.float32)
        pos = {int(f): j for j, f in enumerate(frames)}
        for j, f in enumerate(todo[side][0]):
            q = pos[int(f)]
            r["joints"][q], r["kp2d"][q], r["valid"][q] = part[side]["joints"][j], part[side]["kp2d"][j], \
                part[side]["valid"][j]
            if has_verts and "verts" in part[side]:
                r["verts"][q] = part[side]["verts"][j]
        for f, (J, kp, v, V) in pre[side].items():
            q = pos[f]
            r["joints"][q], r["kp2d"][q], r["valid"][q] = J, kp, v
            if has_verts and V is not None:
                r["verts"][q] = V
        rec[side] = r
    log.info("M4: reused %s probe frames", {s: len(pre[s]) for s in SIDES})
    sharp = crop_sharpness(meta, req, cfg.recon.blur_ref, cfg.recon.blur_floor)
    N = meta.n_frames
    out = {"faces": rec.get("faces")}
    for side in SIDES:
        frames, boxes = req[side]
        r, t = rec[side], tracks[side]
        full = {"joints": np.full((N, 21, 3), np.nan), "kp2d": np.full((N, 21, 2), np.nan),
                "valid": np.zeros(N, bool), "conf": np.zeros(N)}
        if "verts" in r:
            full["verts"] = np.full((N, 778, 3), np.nan, np.float32)
        if len(frames):
            ev = np.array([EVIDENCE_WEIGHT[int(s)] for s in t["src"][frames]]) * np.clip(t["score"][frames], 0, 1)
            conf = ev * visible_fraction(boxes, meta.width, meta.height) * sharp[side] * r["valid"]
            full["joints"][frames], full["kp2d"][frames] = r["joints"], r["kp2d"]
            full["valid"][frames], full["conf"][frames] = r["valid"], conf
            if "verts" in r:
                full["verts"][frames] = r["verts"]
        out[side] = full
    log.info("M4 recon (%s): %s frames in %.0f s", cfg.recon.backend,
             {s: len(req[s][0]) for s in SIDES}, time.time() - t0)
    return out


def merge_recon(base: dict, update: dict, frames: dict) -> dict:
    """Overwrite `base` with `update` on the given frames per side (used after identity flips)."""
    for side, fr in frames.items():
        for k in ("joints", "kp2d", "valid", "conf", "verts"):
            if k in base[side] and k in update[side]:
                base[side][k][fr] = update[side][k][fr]
    return base
