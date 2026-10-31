"""M7: accuracy on hand-labelled frames, comparable across v1-v4 outputs.

Labels CSV (one row per labelled frame and side):
    clip,frame,side,state,x1,y1,x2,y2
    state = visible | partial | out   (box empty when out)
Predictions: any hands.jsonl from v1 (kp2d), v2 (bbox/kp2d) or v3/v4 (bbox/kp2d + source; "estimated" frames count as out of view).
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .proposals import box_iou


def load_predictions(jsonl_path: str) -> dict[int, list[dict]]:
    """frame -> [{'side', 'box'}] for hands the system claims are present."""
    preds = defaultdict(list)
    with open(jsonl_path) as fh:
        for line in fh:
            r = json.loads(line)
            for h in r.get("hands", []):
                if h.get("source") in ("none", "estimated"):        # estimated = claimed out of view
                    continue
                if "bbox" in h and np.isfinite(h["bbox"]).all():          # detector/tracker box when available
                    box = np.array(h["bbox"], float)
                elif "kp2d" in h:                                          # v1: box around the keypoints
                    k = np.array(h["kp2d"], float)
                    k = k[np.isfinite(k).all(1)]
                    if not len(k):
                        continue
                    box = np.array([k[:, 0].min(), k[:, 1].min(), k[:, 0].max(), k[:, 1].max()])
                else:
                    continue
                preds[int(r["frame"])].append({"side": h["side"], "box": box})
    return preds


def evaluate(labels_csv: str, jsonl_path: str, clip: str | None = None, iou_thr: float = 0.3) -> dict:
    lab = pd.read_csv(labels_csv)
    if clip is not None and "clip" in lab:
        lab = lab[lab["clip"] == clip]
    preds = load_predictions(jsonl_path)
    n_lab = n_found = n_side_ok = n_pred = n_false = n_out = n_out_ok = 0
    by_state = defaultdict(lambda: [0, 0])
    for frame, g in lab.groupby("frame"):
        gt = [r for _, r in g.iterrows() if r["state"] in ("visible", "partial")]
        outs = [r for _, r in g.iterrows() if r["state"] == "out"]
        pr = preds.get(int(frame), [])
        n_pred += len(pr)
        n_lab += len(gt)
        matched_pred = set()
        if gt and pr:
            C = np.array([[1 - box_iou(np.array([r.x1, r.y1, r.x2, r.y2], float), p["box"]) for p in pr] for r in gt])
            ri, ci = linear_sum_assignment(C)
            for i, j in zip(ri, ci):
                if 1 - C[i, j] >= iou_thr:
                    n_found += 1
                    matched_pred.add(j)
                    by_state[gt[i]["state"]][0] += 1
                    n_side_ok += pr[j]["side"] == gt[i]["side"]
        for r in gt:
            by_state[r["state"]][1] += 1
        n_false += len(pr) - len(matched_pred)
        for r in outs:
            n_out += 1
            n_out_ok += not any(p["side"] == r["side"] for p in pr)
    div = lambda a, b: round(a / b, 3) if b else None
    return {"labelled_hands": n_lab, "recall": div(n_found, n_lab), "side_accuracy": div(n_side_ok, n_found),
            "false_hand_rate": div(n_false, n_pred), "out_of_view_correct": div(n_out_ok, n_out),
            "recall_by_state": {k: div(v[0], v[1]) for k, v in by_state.items()}}


def compare(labels_csv: str, systems: dict[str, str], clip: str | None = None) -> pd.DataFrame:
    """systems = {'v1': path/to/hands.jsonl, 'v2': ..., 'v3': ...} -> one table for the slides."""
    rows = []
    for name, path in systems.items():
        m = evaluate(labels_csv, path, clip)
        rows.append({"system": name, **{k: v for k, v in m.items() if k != "recall_by_state"},
                     **{f"recall_{k}": v for k, v in m["recall_by_state"].items()}})
    return pd.DataFrame(rows)
