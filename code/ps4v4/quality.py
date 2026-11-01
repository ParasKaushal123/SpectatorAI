"""MQ: clip quality. Which parts of the clip are usable for hand data, which are not, and why.

Per frame it labels five lanes (the frontend draws one row per lane):
  left_hand / right_hand  visible | partial | low_confidence | gap_filled | out_of_view
  image                   sharp | blurred                     (whole-frame sharpness, relative to this clip)
  head                    stable | fast_turn                  (head angular speed from M1H)
  body                    still | moving                      (background motion NOT explained by head rotation:
                                                               walking, leaning, standing up)
  usable                  usable | unusable                   (+ the reasons, per segment)
A frame is usable when at least one hand is clearly visible, the frame is sharp, the head is not turning
fast and the body is not moving. Short blips (< min_segment_s) are merged into their neighbours.

Output: quality.json = overall score (0-100) + grade, sub-scores, usable share, segments with
start/end times per lane, and 10 Hz timeline arrays for charts. Everything is derived from PS4's own
outputs plus one light pass over the video (quarter resolution) for sharpness and background motion.
"""
from __future__ import annotations

import logging
import time

import cv2
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from . import SIDES, STATES
from .video import iter_frames

log = logging.getLogger(__name__)
S = {n: i for i, n in enumerate(STATES)}
HAND_LABELS = ("visible", "partial", "low_confidence", "gap_filled", "out_of_view")


# ---- per-frame signals -------------------------------------------------------------------
def video_signals(meta, cam, head_R: np.ndarray, boxes: list[np.ndarray], scale: float = 0.25) -> dict:
    """One quarter-resolution pass: sharpness (Laplacian variance) and background motion that head
    rotation does not explain (px per frame at full resolution)."""
    N = meta.n_frames
    sharp = np.zeros(N)
    resid = np.full(N, np.nan)
    prev = None
    for t, frame in iter_frames(meta):
        if t >= N:
            break
        g = cv2.cvtColor(cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY)
        sharp[t] = cv2.Laplacian(g, cv2.CV_64F).var()
        if prev is not None:
            mask = np.full(g.shape, 255, np.uint8)
            for b in boxes[t - 1]:                                       # hands are not background
                if np.isfinite(b).all():
                    c, h = (b[:2] + b[2:]) / 2 * scale, (b[2:] - b[:2]) * scale * 0.9
                    cv2.rectangle(mask, tuple((c - h).astype(int)), tuple((c + h).astype(int)), 0, -1)
            p0 = cv2.goodFeaturesToTrack(prev, maxCorners=250, qualityLevel=0.01, minDistance=6, mask=mask)
            if p0 is not None and len(p0) >= 20:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p0, None, winSize=(15, 15), maxLevel=3)
                ok = st.ravel() == 1
                if ok.sum() >= 15:
                    a = p0.reshape(-1, 2)[ok] / scale
                    b = p1.reshape(-1, 2)[ok] / scale
                    R_delta = head_R[t].T @ head_R[t - 1]               # same convention as the tracker
                    pred = cam.project(cam.rays(a) @ R_delta.T)
                    resid[t] = float(np.median(np.linalg.norm(b - pred, axis=1)))
        prev = g
    return {"sharpness": sharp, "residual_px": resid}


def head_speed_dps(head_R: np.ndarray, fps: float) -> np.ndarray:
    N = len(head_R)
    sp = np.zeros(N)
    if N > 1:
        Rd = np.einsum("nji,njk->nik", head_R[1:], head_R[:-1])
        ang = np.degrees(np.arccos(np.clip((np.trace(Rd, axis1=1, axis2=2) - 1) / 2, -1, 1)))
        sp[1:] = ang * fps
        sp[0] = sp[1]
    return median_filter(sp, size=5, mode="nearest")


def hand_lane(seq_side: dict, qc) -> np.ndarray:
    """Per-frame hand label index into HAND_LABELS."""
    src, state, conf = seq_side["src"], seq_side["state"], seq_side["conf"]
    inside = seq_side["joint_in_image"].mean(1) if seq_side.get("joint_in_image") is not None else np.ones(len(src))
    lab = np.full(len(src), HAND_LABELS.index("out_of_view"))
    obs = src == 1
    partial = obs & ((inside < qc.partial_joints_in_image) | np.isin(state, [S["PARTIAL"], S["EXITING"]]))
    lowc = obs & ~partial & (conf < qc.low_confidence)
    lab[obs] = HAND_LABELS.index("visible")
    lab[lowc] = HAND_LABELS.index("low_confidence")
    lab[partial] = HAND_LABELS.index("partial")
    lab[src == 2] = HAND_LABELS.index("gap_filled")
    tracked_no3d = (src == 0) & (state != S["OUT_OF_VIEW"])               # tracked but no usable 3D
    lab[tracked_no3d] = HAND_LABELS.index("low_confidence")
    return lab


# ---- segments ----------------------------------------------------------------------------
def _merge_short(lab: np.ndarray, min_len: int) -> np.ndarray:
    """Repeatedly merge the shortest run below min_len frames into its longer neighbour."""
    lab = np.asarray(lab).copy()
    for _ in range(len(lab)):
        runs = _runs(lab)
        if len(runs) < 2:
            break
        lens = [e - s for s, e, _ in runs]
        k = int(np.argmin(lens))
        if lens[k] >= min_len:
            break
        s0, e0, _ = runs[k]
        left = runs[k - 1] if k > 0 else None
        right = runs[k + 1] if k + 1 < len(runs) else None
        if left and right and left[2] == right[2]:
            v = left[2]
        elif left and right:
            v = left[2] if (left[1] - left[0]) >= (right[1] - right[0]) else right[2]
        else:
            v = (left or right)[2]
        lab[s0:e0] = v
    return lab


def _runs(lab: np.ndarray):
    starts = np.r_[0, np.nonzero(np.diff(lab))[0] + 1]
    ends = np.r_[starts[1:], len(lab)]
    return [(int(s), int(e), lab[s]) for s, e in zip(starts, ends)]


def segments(lane: str, lab: np.ndarray, names, fps: float, detail=None) -> list[dict]:
    out = []
    for s, e, v in _runs(lab):
        d = {"lane": lane, "label": names[int(v)] if names is not None else str(v),
             "start_frame": s, "end_frame": e - 1, "start_s": round(s / fps, 2), "end_s": round(e / fps, 2),
             "duration_s": round((e - s) / fps, 2)}
        if detail is not None:
            d.update(detail(s, e))
        out.append(d)
    return out


# ---- main ---------------------------------------------------------------------------------
def assess_clip(meta, cam, head: dict, seq: dict, cfg) -> dict:
    qc = cfg.quality
    t0 = time.time()
    N, fps = meta.n_frames, meta.fps
    min_len = max(1, int(round(qc.min_segment_s * fps)))
    boxes = [[seq[s]["box"][t] for s in SIDES] for t in range(N)]
    sig = video_signals(meta, cam, head["R"], boxes)
    # image sharpness relative to this clip (and an absolute floor for clips that are blurred throughout)
    sh = sig["sharpness"]
    ref = np.percentile(sh, 75) if N else 1.0
    blurred = (sh < qc.blur_relative * ref) | (sh < qc.blur_absolute)
    image = _merge_short(blurred.astype(int), min_len)
    # head and body motion
    hs = head_speed_dps(head["R"][:N], fps)
    fast = _merge_short((hs > qc.head_fast_dps).astype(int), min_len)
    res = sig["residual_px"]
    res = np.where(np.isfinite(res), res, np.nanmedian(res) if np.isfinite(res).any() else 0.0)
    res_s = uniform_filter1d(res, size=max(3, int(round(0.5 * fps))), mode="nearest")
    body_thr = qc.body_motion_frac_diag * meta.diag
    moving = _merge_short((res_s > body_thr).astype(int), min_len)
    # hands
    hand = {s: _merge_short(hand_lane(seq[s], qc), min_len) for s in SIDES}
    good = {s: np.isin(hand[s], [HAND_LABELS.index("visible"), HAND_LABELS.index("partial")]) for s in SIDES}
    any_hand = good["left"] | good["right"]
    usable_raw = any_hand & (image == 0) & (fast == 0) & (moving == 0)
    usable = _merge_short(usable_raw.astype(int), min_len)

    def why(s, e):
        if usable[s]:
            return {}
        r = []
        if not any_hand[s:e].mean() > 0.5:
            r.append("no hand clearly visible")
        if image[s:e].mean() > 0.5:
            r.append("motion blur")
        if fast[s:e].mean() > 0.5:
            r.append("head turning fast")
        if moving[s:e].mean() > 0.5:
            r.append("body moving")
        return {"reasons": r or ["mixed"]}

    # scores (0..1)
    w = {HAND_LABELS.index("visible"): 1.0, HAND_LABELS.index("partial"): 0.6,
         HAND_LABELS.index("low_confidence"): 0.35, HAND_LABELS.index("gap_filled"): 0.3,
         HAND_LABELS.index("out_of_view"): 0.0}
    best = np.maximum(np.vectorize(w.get)(hand["left"]), np.vectorize(w.get)(hand["right"])) if N else np.zeros(0)
    conf = np.concatenate([seq[s]["conf"][seq[s]["src"] == 1] for s in SIDES])
    sub = {"hand_visibility": float(best.mean()) if N else 0.0,
           "both_hands": float((good["left"] & good["right"]).mean()) if N else 0.0,
           "image_clarity": float(1 - (image == 1).mean()) if N else 0.0,
           "camera_stability": float(((fast == 0) & (moving == 0)).mean()) if N else 0.0,
           "tracking_confidence": float(np.clip(np.mean(conf), 0, 1)) if len(conf) else 0.0}
    weights = dict(qc.score_weights)
    score = 100 * sum(weights[k] * sub[k] for k in weights) / sum(weights.values())
    grade = "A" if score >= 80 else "B" if score >= 65 else "C" if score >= 50 else "D"
    useg = [r for r in _runs(usable) if r[2] == 1]
    longest = max((e - s for s, e, _ in useg), default=0) / fps
    # segments per lane
    segs = []
    for s in SIDES:
        segs += segments(f"{s}_hand", hand[s], HAND_LABELS, fps)
    segs += segments("image", image, ("sharp", "blurred"), fps)
    segs += segments("head", fast, ("stable", "fast_turn"), fps)
    segs += segments("body", moving, ("still", "moving"), fps)
    segs += segments("usable", usable, ("unusable", "usable"), fps, why)
    # 10 Hz timeline for charts
    step = max(1, int(round(fps / qc.timeline_hz)))
    idx = np.arange(0, N, step)
    timeline = {"t": np.round(idx / fps, 2).tolist(),
                "head_speed_dps": np.round(hs[idx], 1).tolist(),
                "body_motion_px": np.round(res_s[idx], 2).tolist(),
                "sharpness_rel": np.round(sh[idx] / max(ref, 1e-9), 3).tolist(),
                "left_hand": [HAND_LABELS[int(v)] for v in hand["left"][idx]],
                "right_hand": [HAND_LABELS[int(v)] for v in hand["right"][idx]],
                "left_conf": np.round(seq["left"]["conf"][idx], 3).tolist(),
                "right_conf": np.round(seq["right"]["conf"][idx], 3).tolist(),
                "usable": usable[idx].astype(bool).tolist()}
    hands_summary = {s: {lab: round(float((hand[s] == i).mean()), 3) for i, lab in enumerate(HAND_LABELS)}
                     for s in SIDES}
    out = {"version": 1, "fps": fps, "frames": N, "duration_s": round(N / fps, 2),
           "score": round(float(score), 1), "grade": grade,
           "subscores": {k: round(v, 3) for k, v in sub.items()},
           "usable": {"share": round(float(usable.mean()), 3) if N else 0.0,
                      "seconds": round(float(usable.sum() / fps), 2), "longest_segment_s": round(longest, 2),
                      "segments": [{"start_s": round(s / fps, 2), "end_s": round(e / fps, 2)} for s, e, _ in useg]},
           "hands": hands_summary,
           "issues": {"blurred_s": round(float((image == 1).sum() / fps), 2),
                      "fast_head_s": round(float((fast == 1).sum() / fps), 2),
                      "body_moving_s": round(float((moving == 1).sum() / fps), 2),
                      "left_out_of_view_s": round(float((hand["left"] == 4).sum() / fps), 2),
                      "right_out_of_view_s": round(float((hand["right"] == 4).sum() / fps), 2)},
           "segments": segs, "timeline": timeline,
           "thresholds": {k: v for k, v in dict(qc).items() if k != "score_weights"} | {"score_weights": weights}}
    log.info("MQ quality: score %.0f (%s), usable %.0f%% in %.0f s", score, grade, 100 * out["usable"]["share"],
             time.time() - t0)
    return out
