"""End-to-end test on a synthetic clip with fake models (no GPU, no downloads).

It reproduces the failures seen in the v3 renders (10 s, 30 fps, 960x540):
  * background pans (head rotation)
  * LEFT hand (green) exits through the left edge at ~2.3 s and re-enters at ~4.7 s
  * JUNK (red, static in the scene, hand-like detector scores) while the left hand is out   -> render 2/5/6
  * RIGHT hand (blue) hidden mid-frame 3.0-3.8 s (cloth/occluder) and comes back mid-frame -> render 2
  * the wearer's KNEE (moving, bottom centre, hand-like scores) 2.7-4.4 s                -> hand_tracking_8
  * ANOTHER PERSON's hand (yellow, small, far, arm from the top) 5.5-9.5 s                  -> render 4
  * the hands CROSS at ~7.5 s and stay on the "wrong" halves until the end                 -> render 3/5
  * RIGHT hand motion-blurred 5.5-6.5 s
  * the fake detector misses 25% of hands and flips 12% of labels; fake 3D has depth noise/outliers
Run:  python tests/test_synthetic.py
"""
from __future__ import annotations

import json
import logging
import math
import os
import pickle
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ps4v4 import STATES  # noqa: E402
from ps4v4.config import load_config  # noqa: E402
from ps4v4.evaluate import evaluate  # noqa: E402
from ps4v4.pipeline import run_clip  # noqa: E402

W, H, FPS, SECS = 960, 540, 30, 10


def pan(t):
    return int(200 + 120 * math.sin(0.6 * t))


def _cross(t, a, b):                                 # move from a to b between 7.0 and 8.0 s
    u = min(max((t - 7.0) / 1.0, 0.0), 1.0)
    u = u * u * (3 - 2 * u)
    return a + (b - a) * u


def left_pos(t):
    if t < 1.5:
        return 260 + 20 * math.sin(3 * t), 380
    if t < 2.6:                                      # slides out through the left edge
        return 260 - (t - 1.5) * 330, 380
    if t < 4.6:
        return None
    if t < 5.4:                                      # comes back in from the left edge
        return -60 + (t - 4.6) * 380, 390
    if t < 7.0:
        return 244 + 25 * math.sin(2 * t), 390
    return _cross(t, 244 + 25 * math.sin(14), 700), 390 - 40 * math.sin(math.pi * min(max(t - 7, 0), 1))


def right_pos(t):
    if t < 7.0:
        return 690 + 60 * math.sin(1.3 * t), 360 + 40 * math.cos(0.9 * t)
    x7, y7 = 690 + 60 * math.sin(9.1), 360 + 40 * math.cos(6.3)
    return _cross(t, x7, 260), y7


def right_hidden(t):
    return 3.0 < t < 3.8


def junk_pos(t):                                     # static in the scene: moves with the background
    return (670 - pan(t), 200) if 3.0 < t < 4.4 else None


def knee_pos(t):                                     # the wearer's knee: bottom centre, moves on its own
    return (470 + 25 * math.sin(4 * t), 505) if 2.7 < t < 4.4 else None


def other_pos(t):                                    # another worker's hand: small, top of the frame
    return (200 + 30 * math.sin(3 * t), 70) if 5.5 < t < 9.5 else None


def make_video(path: str):
    rng = np.random.default_rng(0)
    tex = cv2.GaussianBlur(rng.integers(30, 120, (H + 100, W + 400, 3), dtype=np.uint8), (0, 0), 2)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(SECS * FPS):
        t = i / FPS
        ox = pan(t)
        img = tex[50:50 + H, ox:ox + W].copy()
        lp, rp, jp, op = left_pos(t), right_pos(t), junk_pos(t), other_pos(t)
        if jp:
            cv2.ellipse(img, (int(jp[0]), int(jp[1])), (45, 45), 0, 0, 360, (0, 0, 220), -1)
        kp_ = knee_pos(t)
        if kp_:
            cv2.ellipse(img, (int(kp_[0]), int(kp_[1])), (70, 55), 0, 0, 360, (0, 0, 220), -1)
        if op:
            cv2.ellipse(img, (int(op[0]), int(op[1])), (22, 28), 0, 0, 360, (0, 230, 230), -1)
        if lp:
            cv2.ellipse(img, (int(lp[0]), int(lp[1])), (55, 75), 15, 0, 360, (0, 220, 0), -1)
        if not right_hidden(t):
            cv2.ellipse(img, (int(rp[0]), int(rp[1])), (58, 78), -15, 0, 360, (220, 0, 0), -1)
        else:                                        # an occluder (cloth) covers the right hand
            cv2.rectangle(img, (int(rp[0]) - 90, int(rp[1]) - 110), (int(rp[0]) + 90, int(rp[1]) + 110),
                          (128, 128, 128), -1)
        if 5.5 < t < 6.5:                            # motion blur on the right hand
            k = np.zeros((1, 25), np.float32); k[0, :] = 1 / 25
            x0, x1 = int(rp[0] - 120), int(rp[0] + 120)
            img[:, max(x0, 0):x1] = cv2.filter2D(img[:, max(x0, 0):x1], -1, k)
        vw.write(img)
    vw.release()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    tmp = Path(tempfile.mkdtemp())
    video = str(tmp / "synthetic.mp4")
    make_video(video)
    cfg = load_config(ROOT / "configs" / "default.yaml", {
        "device": "cpu", "paths": {"cache_root": str(tmp / "cache")},
        "camera": {"use_geocalib": False},
        "proposals": {"backend": "fake", "imgsz": 640},
        "probe": {"verifier": "fake"},
        "recon": {"backend": "fake"},
        "render": {"enabled": True, "height": 360},
    } | json.loads(os.environ.get("PS4_TEST_SET", "{}")))   # e.g. '{"camera": {"fallback_dfov_deg": 100}}' 
    stats = run_clip(video, cfg, clip_name="synthetic")
    seq = pickle.load(open(tmp / "cache" / "synthetic" / "sequence.pkl", "rb"))
    n = SECS * FPS
    ts = np.arange(n) / FPS

    def centre(side, f):
        b = seq[side]["box"][f]
        return None if not np.isfinite(b).all() else ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

    def hits(fn):
        k = 0
        for f in range(n):
            p = fn(ts[f])
            for side in ("left", "right"):
                c = centre(side, f)
                if p and c and seq[side]["src"][f] in (1, 2) and math.hypot(c[0] - p[0], c[1] - p[1]) < 50:
                    k += 1
        return k

    junk_hits, other_hits, knee_hits = hits(junk_pos), hits(other_pos), hits(knee_pos)
    # side correctness on observed frames (frames where the two hands overlap are ambiguous: skipped)
    wrong_side = checked = 0
    for f in range(n):
        lp, rp = left_pos(ts[f]), (None if right_hidden(ts[f]) else right_pos(ts[f]))
        if lp and rp and math.hypot(lp[0] - rp[0], lp[1] - rp[1]) < 140:
            continue
        for side, truth, other in (("left", lp, rp), ("right", rp, lp)):
            c = centre(side, f)
            if c and seq[side]["src"][f] == 1 and truth:
                d_own = math.hypot(c[0] - truth[0], c[1] - truth[1])
                d_oth = math.hypot(c[0] - other[0], c[1] - other[1]) if other else 1e9
                wrong_side += d_own > d_oth; checked += 1
    after_cross = [f for f in range(n) if ts[f] > 8.2]
    cross_ok = np.mean([seq["left"]["src"][f] == 1 and centre("left", f)[0] > W / 2 for f in after_cross
                        if centre("left", f)] or [0])
    gone = (ts > 2.75) & (ts < 4.5)
    left_out_share = float(np.mean(seq["left"]["state"][gone] == STATES.index("OUT_OF_VIEW")))
    back = np.nonzero((ts > 4.75) & (seq["left"]["src"] == 1))[0]
    reentry_delay_s = (back[0] / FPS - 4.75) if len(back) else None
    rback = np.nonzero((ts > 3.8) & (seq["right"]["src"] == 1))[0]
    right_back_s = (rback[0] / FPS - 3.8) if len(rback) else None
    est = int((seq["left"]["src"] == 3).sum() + (seq["right"]["src"] == 3).sum())
    jumps = sum(stats[s]["depth_jumps_over_10cm"] for s in ("left", "right"))
    rows = []
    for f in range(0, n, 6):
        for side, fn in (("left", left_pos), ("right", right_pos)):
            p = fn(ts[f])
            hidden = side == "right" and right_hidden(ts[f])
            if p is None or p[0] < -40 or hidden:
                rows.append(dict(clip="synthetic", frame=f, side=side, state="out", x1="", y1="", x2="", y2=""))
            else:
                rows.append(dict(clip="synthetic", frame=f, side=side, state="visible",
                                 x1=p[0] - 70, y1=p[1] - 80, x2=p[0] + 70, y2=p[1] + 80))
    import pandas as pd
    pd.DataFrame(rows).to_csv(tmp / "labels.csv", index=False)
    m = evaluate(str(tmp / "labels.csv"), str(tmp / "cache" / "synthetic" / "hands.jsonl"))
    table = pd.read_csv(tmp / "cache" / "synthetic" / "tracklets.csv")

    print("\n=== TRACKLETS ===")
    print(table[["id", "start", "end", "n_det", "crossing", "score", "hand", "side", "spread", "label",
                 "reasons"]].to_string(index=False))
    print("\n=== RESULTS ===")
    print(f"junk frames tracked as a hand          : {junk_hits}")
    print(f"other person's hand tracked            : {other_hits}")
    print(f"knee tracked as a hand                 : {knee_hits}")
    print(f"wrong side (outside overlaps)          : {wrong_side} / {checked}")
    print(f"left hand on the right half after cross: {cross_ok:.0%} of frames labelled LEFT")
    print(f"left OUT_OF_VIEW while gone            : {left_out_share:.0%}")
    print(f"left re-entry delay (edge)             : {reentry_delay_s}")
    print(f"right recovery after mid-frame occlusion: {right_back_s}")
    print(f"estimated (out of view) frames         : {est}")
    print(f"depth jumps > 10 cm                    : {jumps}")
    print(f"labelled-frame metrics                 : {m}")
    print(f"render                                 : {tmp / 'cache' / 'synthetic' / 'render.mp4'}")
    assert junk_hits == 0, "junk tracked as hand"
    assert other_hits == 0, "other person's hand tracked"
    assert knee_hits == 0, "knee tracked as a hand"
    assert wrong_side <= 0.01 * checked, "left/right swapped"
    assert cross_ok > 0.8, "identity lost through the crossing"
    assert left_out_share > 0.9, "left hand not held out of view"
    assert reentry_delay_s is not None and reentry_delay_s < 0.5, "edge re-entry not recovered within 0.5 s"
    assert right_back_s is not None and right_back_s < 0.5, "mid-frame re-entry not recovered within 0.5 s"
    assert jumps == 0, "depth jumps"
    assert m["side_accuracy"] >= 0.98 and m["recall"] > 0.85 and m["false_hand_rate"] < 0.05, m
    print("ALL CHECKS PASSED")
    return tmp


if __name__ == "__main__":
    main()
