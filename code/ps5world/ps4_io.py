"""Read a PS4 v4 result folder (hands.jsonl, camera.json, head_rot.pkl, stats.json)."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

SIDES = ("left", "right")
SOURCES = {"none": 0, "observed": 1, "infilled": 2, "estimated": 3}


def load_hands(folder: str | Path) -> dict:
    """Per side: kp2d (N,21,2) full-res px, J_cam (N,21,3) PS4 camera metres, src (N,), box (N,4)."""
    folder = Path(folder)
    rows = [json.loads(line) for line in open(folder / "hands.jsonl")]
    N = max(r["frame"] for r in rows) + 1
    out = {"n_frames": N, "fps": None}
    if len(rows) > 1:
        out["fps"] = round(1.0 / max(rows[1]["t"] - rows[0]["t"], 1e-6), 3)
    for s in SIDES:
        out[s] = {"kp2d": np.full((N, 21, 2), np.nan), "J_cam": np.full((N, 21, 3), np.nan),
                  "src": np.zeros(N, np.int8), "box": np.full((N, 4), np.nan)}
    for r in rows:
        t = r["frame"]
        for h in r["hands"]:
            d = out[h["side"]]
            if "bbox" in h:
                d["box"][t] = h["bbox"]
            src = SOURCES.get(h.get("source", "none"), 0)
            if src and "kp3d_cam" in h:
                d["src"][t] = src
                d["kp2d"][t] = np.array(h["kp2d"], float)
                d["J_cam"][t] = np.array(h["kp3d_cam"], float)
    load_mesh(folder, out)
    return out


def load_mesh(folder: str | Path, hands: dict) -> bool:
    """Adds hands[side]["V_cam"] (N,778,3, NaN where no mesh) and hands["faces"] from hands_mesh.npz (PS4 >= 4.5).
    Without the file the world view falls back to skeletons."""
    p = Path(folder) / "hands_mesh.npz"
    hands["faces"] = None
    if not p.exists():
        return False
    d = np.load(p)
    N = hands["n_frames"]
    for s in SIDES:
        fr, rel = d[f"{s}_frames"], d[f"{s}_rel"].astype(np.float32)
        V = np.full((N, rel.shape[1] if len(rel) else 778, 3), np.nan, np.float32)
        keep = fr < N
        fr, rel = fr[keep], rel[keep]
        wrist = hands[s]["J_cam"][fr, 0]
        ok = np.isfinite(wrist).all(1)
        V[fr[ok]] = rel[ok] + wrist[ok, None].astype(np.float32)
        hands[s]["V_cam"] = V
    hands["faces"] = d["faces"].astype(np.int32)
    return True


def trim_hands(hands: dict, n: int) -> dict:
    """Keep only the first n frames (when the world step runs on part of a clip, e.g. --max-seconds 60)."""
    n = min(int(n), hands["n_frames"])
    out = {**hands, "n_frames": n}
    for s in SIDES:
        out[s] = {k: v[:n] for k, v in hands[s].items()}
    return out


def load_camera(folder: str | Path) -> dict:
    return json.load(open(Path(folder) / "camera.json"))


def load_head_rotation(folder: str | Path) -> np.ndarray | None:
    p = Path(folder) / "head_rot.pkl"
    if not p.exists():
        return None
    return pickle.load(open(p, "rb"))["R"]                     # (N,3,3) camera_t -> camera_0


def hand_boxes(hands: dict, grow: float = 1.5) -> dict[int, list[np.ndarray]]:
    """frame -> list of hand boxes (full-res px), grown to cover the wrist and the hand's edges."""
    out: dict[int, list] = {}
    for s in SIDES:
        B = hands[s]["box"]
        for t in np.nonzero(np.isfinite(B).all(1))[0]:
            b = B[t]
            c, half = (b[:2] + b[2:]) / 2, (b[2:] - b[:2]) * grow / 2
            out.setdefault(int(t), []).append(np.concatenate([c - half, c + half]))
    return out
