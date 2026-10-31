"""M2: hand proposals per frame from the detector on raw + enhanced frames.

Only boxes with score >= proposals.min_score survive; the tracker never sees weaker ones.
Output: list (one per frame) of arrays (n, 6) = x1, y1, x2, y2, score, p_right.
"""
from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from .models.detector import make_detector
from .video import VideoMeta, iter_batches

log = logging.getLogger(__name__)

_CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
_GAMMA = (255 * (np.arange(256) / 255) ** 0.6).astype(np.uint8)


def enhance(bgr: np.ndarray) -> np.ndarray:
    """Contrast (CLAHE on L) plus gamma lift for dark frames."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l = lab[..., 0]
    if l.mean() < 90:
        l = cv2.LUT(l, _GAMMA)
    lab[..., 0] = _CLAHE.apply(l)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def merge(dets: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy cluster by IoU: box = score-weighted mean, score = max, p_right = score-weighted mean."""
    if len(dets) == 0:
        return np.zeros((0, 6))
    dets = dets[np.argsort(-dets[:, 4])]
    used = np.zeros(len(dets), bool)
    out = []
    for i in range(len(dets)):
        if used[i]:
            continue
        group = [j for j in range(i, len(dets)) if not used[j] and box_iou(dets[i, :4], dets[j, :4]) > iou_thr]
        used[group] = True
        g = dets[group]
        w = g[:, 4]
        out.append([*(g[:, :4] * w[:, None]).sum(0) / w.sum(), g[:, 4].max(), (g[:, 5] * w).sum() / w.sum()])
    return np.array(out)


def run_proposals(meta: VideoMeta, cfg) -> dict:
    pc = cfg.proposals
    det = make_detector(pc.backend, cfg.device)
    diag = meta.diag
    per_frame: list[np.ndarray] = []
    t0 = time.time()
    for f0, frames in iter_batches(meta, pc.batch):
        raw = det.detect(frames, pc.imgsz, pc.min_score)
        enh = [np.zeros((0, 6))] * len(frames)
        if pc.enhanced_pass == "auto":                  # only where it can help: dark frames or < 2 strong hands
            need = [k for k, (f, r) in enumerate(zip(frames, raw))
                    if (r[:, 4] >= 0.5).sum() < 2 or cv2.cvtColor(f[::8, ::8], cv2.COLOR_BGR2GRAY).mean() < 90]
            if need:
                res = det.detect([enhance(frames[k]) for k in need], pc.imgsz, pc.min_score)
                for k, r in zip(need, res):
                    enh[k] = r
        elif pc.enhanced_pass:
            enh = det.detect([enhance(f) for f in frames], pc.imgsz, pc.min_score)
        for a, b in zip(raw, enh):
            d = merge(np.concatenate([a, b]) if len(a) + len(b) else np.zeros((0, 6)), pc.nms_iou)
            if len(d):
                side = np.maximum(d[:, 2] - d[:, 0], d[:, 3] - d[:, 1]) / diag
                d = d[(d[:, 4] >= pc.min_score) & (side >= pc.min_box_frac) & (side <= pc.max_box_frac)]
            per_frame.append(d)
    dt = time.time() - t0
    log.info("M2 proposals: %d frames in %.0f s", len(per_frame), dt)
    return {"proposals": per_frame, "seconds": dt}
