"""M6b: exports. hands.jsonl (per frame), robot CSV (wrist pose + gripper), stats."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import SIDES, STATES

SRC_NAME = {0: "none", 1: "observed", 2: "infilled", 3: "estimated"}   # estimated = out of view, guessed


def _r(a, n):
    return np.round(a, n).tolist()


def export_jsonl(meta, seq: dict, path: str) -> None:
    with open(path, "w") as fh:
        for t in range(meta.n_frames):
            hands = []
            for side in SIDES:
                d = seq[side]
                src = int(d["src"][t])
                h = {"side": side, "state": STATES[int(d["state"][t])], "source": SRC_NAME[src],
                     "conf": round(float(d["conf"][t]), 3)}
                if np.isfinite(d["box"][t]).all():
                    h["bbox"] = _r(d["box"][t], 1)
                if src:
                    h.update(kp2d=_r(d["kp2d"][t], 1), kp3d_cam=_r(d["J_cam"][t], 4), kp3d_head=_r(d["J_head"][t], 4),
                             quat_head_xyzw=_r(d["quat_head"][t], 5),
                             joint_in_image=d["joint_in_image"][t].astype(int).tolist())
                hands.append(h)
            fh.write(json.dumps({"frame": t, "t": round(t / meta.fps, 4), "hands": hands}) + "\n")


def export_mesh(seq: dict, faces, path: str) -> bool:
    """hands_mesh.npz: MANO faces + per side the frames with a mesh and the 778 vertices relative to the wrist
    joint of kp3d_cam (camera frame, metres, float16 = 0.1 mm steps near the wrist). Absolute vertex position =
    rel + kp3d_cam[wrist] from hands.jsonl. Used by the PS5 world view to draw real hand surfaces."""
    if faces is None or all(seq[s].get("verts_cam") is None for s in SIDES):
        return False
    out = {"faces": np.asarray(faces, np.int32)}
    for s in SIDES:
        V, J, src = seq[s].get("verts_cam"), seq[s]["J_cam"], seq[s]["src"]
        if V is None:                                                   # this hand was never reconstructed
            out[f"{s}_frames"], out[f"{s}_rel"] = np.zeros(0, np.int32), np.zeros((0, 778, 3), np.float16)
            continue
        ok = (src > 0) & np.isfinite(V[:, 0, 0]) & np.isfinite(J[:, 0, 0])
        idx = np.nonzero(ok)[0]
        out[f"{s}_frames"] = idx.astype(np.int32)
        out[f"{s}_rel"] = (V[idx] - J[idx, :1]).astype(np.float16)
    np.savez_compressed(path, **out)
    return True


def export_robot_csv(meta, seq: dict, path: str) -> None:
    """Wrist pose in the head-stable frame (x right, y down, z forward at frame 0; metres) + gripper aperture."""
    rows = []
    for side in SIDES:
        d = seq[side]
        for t in np.nonzero(d["src"] > 0)[0]:
            J = d["J_head"][t]
            hand_len = np.linalg.norm(J[9] - J[0]) + 1e-6
            q = d["quat_head"][t]
            rows.append({"t": t / meta.fps, "frame": int(t), "side": side, "x": J[0, 0], "y": J[0, 1], "z": J[0, 2],
                         "qx": q[0], "qy": q[1], "qz": q[2], "qw": q[3],
                         "grip": float(np.clip(np.linalg.norm(J[4] - J[8]) / hand_len, 0, 2)),
                         "source": SRC_NAME[int(d["src"][t])]})
    columns = ["t", "frame", "side", "x", "y", "z", "qx", "qy", "qz", "qw", "grip", "source"]
    pd.DataFrame(rows, columns=columns).sort_values(["frame", "side"]).round(5).to_csv(path, index=False)


def stability_stats(meta, seq: dict, timings: dict) -> dict:
    """Automatic metrics (no labels needed)."""
    out = {"frames": meta.n_frames, "fps_video": meta.fps, "seconds": {k: round(v, 1) for k, v in timings.items()}}
    for side in SIDES:
        d = seq[side]
        src, state = d["src"], d["state"]
        J = d["J_head"]
        ok = (src > 0) & (src < 3)                     # measured frames only
        acc = np.linalg.norm(J[2:, 0] - 2 * J[1:-1, 0] + J[:-2, 0], axis=1) * 1000
        acc = acc[ok[2:] & ok[1:-1] & ok[:-2]]
        dz = np.abs(np.diff(d["J_cam"][:, 0, 2]))
        dz = dz[ok[1:] & ok[:-1]]
        runs = int(np.sum(np.diff(np.concatenate([[0], ok.astype(int), [0]])) == 1))
        out[side] = {
            "share_observed": round(float((src == 1).mean()), 3),
            "share_infilled": round(float((src == 2).mean()), 3),
            "share_estimated_out_of_view": round(float((src == 3).mean()), 3),
            "share_out_of_view": round(float((state == 0).mean()), 3),
            "segments": runs,
            "wrist_accel_median_mm_per_frame2": round(float(np.median(acc)), 3) if len(acc) else None,
            "depth_jumps_over_10cm": int((dz > 0.10).sum()),
            "state_counts": {STATES[i]: int((state == i).sum()) for i in range(len(STATES))},
        }
    return out
