"""Regression (hand_tracking_2): a LEFT hand resting forward-left has its forearm leaning slightly RIGHT
(toward the body). v4.0 read that lean as "right" and flipped both hands. The side vote now uses where
the forearm leaves the image (bottom-left of centre = left arm).
Run: python tests/test_side_cue.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ps4v4.probe import probe_features  # noqa: E402

W, H = 1920, 1080


def hand(wrist, lean_dx):
    """21 keypoints: knuckles 60 px above the wrist, forearm leaning by lean_dx (px per 60 px down)."""
    kp = np.tile(np.array(wrist, float), (21, 1))
    kp[[5, 9, 13, 17]] = np.array(wrist) + [-lean_dx, -60]
    J = np.zeros((21, 3)); J[:, 2] = 0.5; J[[5, 9, 13, 17], 1] = -0.08
    return J, kp


J, kp = hand((440, 600), lean_dx=8)               # left hand at 23% of the width, forearm leaning right
f = probe_features(J, kp, np.array([380, 520, 520, 660.0]), W, H)
print("resting left hand:", round(f["arm_dx"], 2), "lean (old cue says right) ->", round(f["arm_exit"], 2), "exit vote")
assert f["arm_dx"] > 0 and f["arm_exit"] < -0.3
J, kp = hand((1600, 500), lean_dx=-20)            # right hand on the right, forearm leaning left toward the body
f = probe_features(J, kp, np.array([1540, 420, 1680, 580.0]), W, H)
assert f["arm_exit"] > 0.3
print("PASSED")
