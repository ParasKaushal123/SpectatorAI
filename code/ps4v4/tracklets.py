"""M3: tracklets. Link hand proposals into short, side-free tracks.

v3 decided "which side" and "is it a hand" online, frame by frame, inside the tracker. A wrong
decision (junk acquired, a real hand blocked, a swap at a crossing) could not be undone.
v4 only links boxes here; all decisions move to M3A, which sees each tracklet as a whole.

Linking (ByteTrack-style, with head-rotation-compensated prediction):
  1. strong proposals (>= high_score) are matched to active tracklets first
  2. weak proposals (>= low_score) can only continue an existing tracklet
  3. unmatched tracklets coast on optical flow for <= max_coast_s, then end
  4. only proposals >= start_score start a tracklet
  5. when two tracklets overlap (hands crossing / touching) both are cut, and cut again when they
     separate. Identity through a crossing is decided later from arm direction and motion,
     never by box matching inside the overlap.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import FisheyeCamera
from .proposals import box_iou
from .video import VideoMeta, iter_frames

log = logging.getLogger(__name__)
SRC_NONE, SRC_DET, SRC_FLOW, SRC_SAM, SRC_PRED = 0, 1, 2, 3, 4


def center(b: np.ndarray) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def bsize(b: np.ndarray) -> float:
    return float(max(b[2] - b[0], b[3] - b[1], 1.0))


def box_from(c: np.ndarray, w: float, h: float) -> np.ndarray:
    return np.array([c[0] - w / 2, c[1] - h / 2, c[0] + w / 2, c[1] + h / 2])


@dataclass
class _Live:
    id: int
    crossing: bool
    frames: list = field(default_factory=list)
    boxes: list = field(default_factory=list)
    score: list = field(default_factory=list)
    p_right: list = field(default_factory=list)
    src: list = field(default_factory=list)
    resid: list = field(default_factory=list)          # scene-relative image motion (px) per frame
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    coast: int = 0
    calm: int = 0                                        # frames without overlap (to leave crossing mode)

    @property
    def box(self) -> np.ndarray:
        return self.boxes[-1]

    def add(self, t, box, score, p_right, src, resid):
        self.frames.append(t); self.boxes.append(np.asarray(box, float)); self.score.append(float(score))
        self.p_right.append(float(p_right)); self.src.append(int(src)); self.resid.append(np.asarray(resid, float))

    def pop(self):
        return (self.frames.pop(), self.boxes.pop(), self.score.pop(), self.p_right.pop(), self.src.pop(),
                self.resid.pop())

    def finish(self) -> dict | None:
        while self.src and self.src[-1] == SRC_PRED:      # no evidence at the tail: drop it
            self.pop()
        if not self.frames:
            return None
        return {"id": self.id, "crossing": self.crossing, "frames": np.array(self.frames, int),
                "boxes": np.array(self.boxes, float), "score": np.array(self.score),
                "p_right": np.array(self.p_right), "src": np.array(self.src, np.int8),
                "resid": np.array(self.resid, float).reshape(-1, 2)}


class TrackletBuilder:
    def __init__(self, meta: VideoMeta, cam: FisheyeCamera, head_R: np.ndarray, cfg):
        self.meta, self.cam, self.head_R, self.c = meta, cam, head_R, cfg.tracklets
        self.max_coast = max(1, int(round(self.c.max_coast_s * meta.fps)))
        self.live: list[_Live] = []
        self.done: list[dict] = []
        self.next_id = 0

    # ---- geometry ----------------------------------------------------------
    def _compensate(self, c: np.ndarray, t: int) -> np.ndarray:
        """Where an image point of frame t-1 lands in frame t if it were static in the scene."""
        if not self.c.head_compensation or t == 0:
            return c
        R_delta = self.head_R[t].T @ self.head_R[t - 1]
        return self.cam.project(R_delta @ self.cam.rays(c))

    def _predict(self, tr: _Live, t: int) -> np.ndarray:
        b = tr.box
        c = self._compensate(center(b), t) + 0.7 * tr.vel
        return box_from(c, b[2] - b[0], b[3] - b[1])

    def _cost(self, pred: np.ndarray, d: np.ndarray, speed: float = 0.0) -> float:
        iou = box_iou(pred, d[:4])
        dist = np.linalg.norm(center(pred) - center(d[:4])) / bsize(pred)
        ratio = bsize(d[:4]) / bsize(pred)
        lo, hi = self.c.gate_scale
        gate = self.c.gate_dist + min(speed / bsize(pred), 1.0)       # fast hands (wiping) get a wider gate
        if not ((iou > 0.1 or dist < gate) and lo <= ratio <= hi):
            return 1e6
        return (1 - iou) + 0.5 * dist

    def _match(self, idx: list[int], preds: dict, props: np.ndarray, cand: list[int], live=None) -> dict:
        if not idx or not cand:
            return {}
        sp = {i: float(np.linalg.norm(live[i].vel)) if live is not None else 0.0 for i in idx}
        C = np.array([[self._cost(preds[i], props[j], sp[i]) for j in cand] for i in idx])
        r, c = linear_sum_assignment(C)
        return {idx[a]: cand[b] for a, b in zip(r, c) if C[a, b] < 1e5}

    # ---- life cycle ----------------------------------------------------------
    def _new(self, crossing: bool) -> _Live:
        tr = _Live(self.next_id, crossing)
        self.next_id += 1
        self.live.append(tr)
        return tr

    def _end(self, tr: _Live):
        self.live.remove(tr)
        res = tr.finish()
        if res is not None:
            self.done.append(res)

    def _split(self, tr: _Live, crossing: bool):
        """Cut a tracklet at the current frame: the current observation starts a new tracklet."""
        last = tr.pop()
        vel = tr.vel.copy()
        self._end(tr)
        nt = self._new(crossing)
        nt.add(*last)
        nt.vel = vel

    def _update_crossings(self, t: int):
        act = [tr for tr in self.live if tr.frames and tr.frames[-1] == t]
        over = {tr.id: 0.0 for tr in act}
        for i, a in enumerate(act):
            for b in act[i + 1:]:
                o = box_iou(a.box, b.box)
                over[a.id] = max(over[a.id], o); over[b.id] = max(over[b.id], o)
        for tr in act:
            o = over[tr.id]
            if not tr.crossing and o > self.c.cross_iou_enter:
                self._split(tr, crossing=True)
            elif tr.crossing:
                tr.calm = tr.calm + 1 if o < self.c.cross_iou_exit else 0
                if tr.calm >= 2:
                    self._split(tr, crossing=False)

    # ---- main loop -----------------------------------------------------------
    def run(self, proposals: list[np.ndarray]) -> list[dict]:
        t0 = time.time()
        c = self.c
        prev_gray, scale = None, 0.5
        for t, frame in iter_frames(self.meta):
            if t >= self.meta.n_frames:
                break
            gray = cv2.cvtColor(cv2.resize(frame, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
            props = proposals[t] if t < len(proposals) else np.zeros((0, 6))
            props = props[props[:, 4] >= c.low_score] if len(props) else np.zeros((0, 6))
            live = list(self.live)
            preds = {i: self._predict(tr, t) for i, tr in enumerate(live)}
            high = [j for j in range(len(props)) if props[j, 4] >= c.high_score]
            low = [j for j in range(len(props)) if props[j, 4] < c.high_score]
            m1 = self._match(list(range(len(live))), preds, props, high, live)
            rest = [i for i in range(len(live)) if i not in m1]
            m2 = self._match(rest, preds, props, low, live)
            match = {**m1, **m2}
            used = set(match.values())
            for i, tr in enumerate(live):
                if i in match:
                    d = props[match[i]]
                    comp = self._compensate(center(tr.box), t)
                    resid = center(d[:4]) - comp
                    tr.vel = 0.5 * tr.vel + 0.5 * resid
                    tr.coast = 0
                    tr.add(t, d[:4], d[4], d[5], SRC_DET, resid)
                else:
                    self._coast(tr, preds[i], prev_gray, gray, t, scale)
            # start new tracklets from strong, unexplained proposals
            for j in np.argsort(-props[:, 4]) if len(props) else []:
                if j in used or props[j, 4] < c.start_score:
                    continue
                b = props[j, :4]
                if any(box_iou(b, tr.box) > 0.5 for tr in self.live if tr.frames and tr.frames[-1] == t):
                    continue                                   # duplicate of a live tracklet
                nt = self._new(crossing=False)
                nt.add(t, b, props[j, 4], props[j, 5], SRC_DET, np.zeros(2))
                used.add(j)
            self._update_crossings(t)
            prev_gray = gray
        for tr in list(self.live):
            self._end(tr)
        if self.c.split_on_handedness:
            self.done = split_on_handedness(self.done, self.c.handedness_run, self.next_id)
        self.done.sort(key=lambda d: (d["frames"][0], d["id"]))
        log.info("M3 tracklets: %d tracklets over %d frames in %.0f s", len(self.done), self.meta.n_frames,
                 time.time() - t0)
        return self.done

    def _coast(self, tr: _Live, pred, prev_gray, gray, t: int, scale: float):
        tr.coast += 1
        W, H = self.meta.width, self.meta.height
        off = min(pred[2], W - pred[0], pred[3], H - pred[1]) < 0.25 * bsize(pred)   # mostly outside
        if tr.coast > self.max_coast or off:
            self._end(tr)
            return
        res = None
        if self.c.propagate == "flow" and prev_gray is not None:
            res = _flow_box(prev_gray, gray, tr.box, self.c.flow_min_points, self.c.flow_min_inlier, scale)
        if res is not None:
            box, conf, src = res[0] / scale, res[1], SRC_FLOW
        else:
            box, conf, src = pred, 0.0, SRC_PRED
        comp = self._compensate(center(tr.box), t)
        tr.add(t, box, conf, np.nan, src, center(box) - comp)


def _handedness_cut(tr: dict, run: int, gap: int = 4) -> int | None:
    """Index where the detector's left/right call flips and STAYS flipped: the `run` detections before
    (skipping `gap` transition frames) are >= 85% one side, the `run` after >= 85% the other. That is a
    track that slid from one hand onto the other (hands close, fast head turn), not label noise."""
    det = np.nonzero(tr["src"] == SRC_DET)[0]
    if len(det) < 2 * run + 2 * gap:
        return None
    lab = (tr["p_right"][det] > 0.5).astype(float)
    best, best_j = 0.0, None
    for j in range(run + gap, len(det) - run - gap + 1):
        a = lab[j - gap - run:j - gap].mean()
        b = lab[j + gap:j + gap + run].mean()
        d = abs(a - b)
        if d >= 0.7 and min(a, 1 - a) <= 0.15 and min(b, 1 - b) <= 0.15 and d > best:
            best, best_j = d, j
    return int(det[best_j]) if best_j is not None else None


def split_on_handedness(done: list[dict], run: int, next_id: int) -> list[dict]:
    out = []
    todo = list(done)
    while todo:
        tr = todo.pop()
        i = _handedness_cut(tr, run)
        if i is None or i == 0:
            out.append(tr)
            continue
        a = {k: (v[:i] if isinstance(v, np.ndarray) else v) for k, v in tr.items()}
        b = {k: (v[i:] if isinstance(v, np.ndarray) else v) for k, v in tr.items()}
        b["id"] = next_id
        next_id += 1
        todo += [a, b]
    return out


def _flow_box(g0, g1, box_full: np.ndarray, min_pts: int, min_inlier: float, scale: float = 0.5):
    """Follow the hand's own texture with pyramidal LK; returns (box in scaled coords, confidence)."""
    b = box_full * scale
    H, W = g0.shape
    x1, y1 = int(max(b[0] + 0.1 * (b[2] - b[0]), 0)), int(max(b[1] + 0.1 * (b[3] - b[1]), 0))
    x2, y2 = int(min(b[2] - 0.1 * (b[2] - b[0]), W)), int(min(b[3] - 0.1 * (b[3] - b[1]), H))
    if x2 - x1 < 6 or y2 - y1 < 6:
        return None
    mask = np.zeros_like(g0); mask[y1:y2, x1:x2] = 255
    p0 = cv2.goodFeaturesToTrack(g0, maxCorners=80, qualityLevel=0.01, minDistance=4, mask=mask)
    if p0 is None or len(p0) < min_pts:
        return None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, winSize=(15, 15), maxLevel=3)
    pb, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, winSize=(15, 15), maxLevel=3)
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm((pb - p0).reshape(-1, 2), axis=1) < 1.0)
    ratio = ok.mean()
    if ok.sum() < min_pts or ratio < min_inlier:
        return None
    d = np.median((p1 - p0).reshape(-1, 2)[ok], axis=0)
    return b + np.array([d[0], d[1], d[0], d[1]]), float(ratio)


def build_tracklets(meta, cam, head_R, proposals, cfg) -> list[dict]:
    return TrackletBuilder(meta, cam, head_R, cfg).run(proposals)
