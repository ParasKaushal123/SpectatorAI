"""Regression (hand_tracking_4): a long left-hand tracklet that overlaps a stronger left tracklet for
16 frames, and right-hand tracklets elsewhere, was rejected entirely ("side taken") in v4.0.
v4.1 cuts tracklets at overlaps, so only the overlapping frames are contested.
Run: python tests/test_assign_overlap.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ps4v4.assign import solve_labels, split_at_overlaps  # noqa: E402
from ps4v4.config import load_config  # noqa: E402


def tr(i, a, b, x):
    fr = np.arange(a, b + 1); n = len(fr)
    return {"id": i, "crossing": False, "frames": fr, "boxes": np.tile([x, 100, x + 200, 300.0], (n, 1)),
            "score": np.full(n, 0.7), "p_right": np.full(n, 0.5), "src": np.ones(n, np.int8), "resid": np.zeros((n, 2))}


ac = load_config(ROOT / "configs" / "default.yaml").assign
T = [tr(1, 9, 137, 900), tr(6, 50, 517, 300), tr(32, 371, 684, 1200), tr(40, 502, 899, 350)]
F = [{"n_det": 129, "hand": 0.86, "side": 1.2}, {"n_det": 446, "hand": 0.27, "side": -0.04},
     {"n_det": 314, "hand": 0.25, "side": 0.73}, {"n_det": 398, "hand": 1.0, "side": -0.56}]
P, PF, par = split_at_overlaps(T, F, ac)
L = solve_labels(PF, P, 30.0, ac, par)
left_frames = sum(len(pc["frames"]) for pc, l, p in zip(P, L, par) if T[p]["id"] == 6 and l == "left")
print("tracklet 6 frames labelled left:", left_frames, "of", len(T[1]["frames"]))
assert left_frames >= 440
print("PASSED")
