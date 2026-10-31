"""Pick frames worth hand-labelling (hard cases oversampled) and write a labels template.

Usage:
    python tools/sample_label_frames.py --clip clips/demo_01.mp4 --cache cache/demo_01 --out labels/demo_01 --n 100

Writes <out>/frames/<frame>.jpg (PRIVATE: footage) and <out>/labels_template.csv with one row
per frame and side. Fill state (visible/partial/out) and the box in any image tool.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def pick_frames(n_frames: int, tracks: dict | None, n: int, seed: int = 0) -> list[int]:
    rng = np.random.default_rng(seed)
    picks: set[int] = set()
    if tracks is not None:
        for side in ("left", "right"):
            st = tracks[side]["state"]
            change = np.nonzero(np.diff(st) != 0)[0]                  # exits, re-entries, partial spells
            for c in rng.permutation(change)[: n // 4]:
                picks.add(int(np.clip(c + rng.integers(-3, 4), 0, n_frames - 1)))
            low = np.nonzero((st != 0) & (tracks[side]["score"] < 0.5))[0]
            for c in rng.permutation(low)[: n // 8]:
                picks.add(int(c))
    while len(picks) < n:
        picks.add(int(rng.integers(0, n_frames)))
    return sorted(picks)[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--cache", default=None, help="v4 cache dir of this clip (to oversample hard frames)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.clip)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tracks = None
    if a.cache and (Path(a.cache) / "tracks_final.pkl").exists():
        tracks = pickle.load(open(Path(a.cache) / "tracks_final.pkl", "rb"))
    frames = pick_frames(n_frames, tracks, a.n)
    out = Path(a.out); (out / "frames").mkdir(parents=True, exist_ok=True)
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if ok:
            cv2.imwrite(str(out / "frames" / f"{f:06d}.jpg"), img)
    clip = Path(a.clip).stem
    rows = [{"clip": clip, "frame": f, "side": s, "state": "", "x1": "", "y1": "", "x2": "", "y2": ""}
            for f in frames for s in ("left", "right")]
    pd.DataFrame(rows).to_csv(out / "labels_template.csv", index=False)
    print(f"wrote {len(frames)} frames and {out / 'labels_template.csv'}")


if __name__ == "__main__":
    main()
