"""Label-free accuracy report over PS4 results (and SLAM results if present). Runs in a few minutes on CPU.

Without hand-labelled frames there is no ground truth, so this measures what CAN be checked automatically, the
way tracking papers report "physical plausibility" and "temporal consistency":

  coverage        share of frames each hand is tracked (observed)
  identity        left/right swaps per minute (a hand jumps to where the OTHER hand just was)
  teleports       wrist jumps > 25 cm between consecutive frames (> 7.5 m/s: impossible) per minute
  jitter          median wrist acceleration, mm/frame^2 (head-stable frame; lower = smoother)
  anatomy         hand size (wrist -> middle knuckle, adult 7-11 cm), bone-length stability over time (CV %)
  reach           share of tracked frames with the wrist 15-90 cm from the camera (an arm's length)
  rejection       candidate tracks rejected as not-the-wearer, and the top reasons (tracklets.csv)
  quality         usable share + grade (quality.json, if present)
  SLAM            head-rotation agreement SLAM vs PS4 (deg), hand-to-head distance (world_stats.json, if present)

Usage:  python tools/accuracy_report.py <results_dir> [<results_dir> ...] [--slam <ps5_world_dir>] [--out report]
        each results_dir contains one folder per clip with hands.jsonl (+ stats.json, tracklets.csv, quality.json)
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def load(path: Path):
    rows = [json.loads(l) for l in open(path)]
    n = len(rows)
    fps = 1.0 / max(rows[1]["t"] - rows[0]["t"], 1e-6) if n > 1 else 30.0
    J = {s: np.full((n, 21, 3), np.nan) for s in ("left", "right")}       # camera frame (reach, anatomy)
    H = {s: np.full((n, 21, 3), np.nan) for s in ("left", "right")}       # head-stable frame (motion)
    obs = {s: np.zeros(n, bool) for s in ("left", "right")}
    for t, r in enumerate(rows):
        for h in r["hands"]:
            s = h["side"]
            if h.get("source") == "observed" and "kp3d_cam" in h:
                obs[s][t] = True
                J[s][t] = h["kp3d_cam"]
                H[s][t] = h.get("kp3d_head", h["kp3d_cam"])
    return n, fps, J, H, obs


def clip_metrics(d: Path) -> dict:
    n, fps, J, H, obs = load(d / "hands.jsonl")
    minutes = n / fps / 60
    m = {"clip": d.name, "minutes": round(minutes, 2)}
    swaps = 0
    both = obs["left"] & obs["right"]
    WL, WR = H["left"][:, 0], H["right"][:, 0]
    for t in np.nonzero(both[1:] & both[:-1])[0] + 1:                     # identity swap test
        keep = np.linalg.norm(WL[t] - WL[t - 1]) + np.linalg.norm(WR[t] - WR[t - 1])
        cross = np.linalg.norm(WL[t] - WR[t - 1]) + np.linalg.norm(WR[t] - WL[t - 1])
        if cross < 0.5 * keep and keep > 0.10:
            swaps += 1
    m["lr_swaps_per_min"] = round(swaps / max(minutes, 1e-9), 2)
    tele, acc, size, cv, reach_ok, reach_n = 0, [], [], [], 0, 0
    for s in ("left", "right"):
        o = obs[s]
        m[f"{s}_tracked"] = round(float(o.mean()), 3)
        W = H[s][:, 0]
        k = np.nonzero(o[1:] & o[:-1])[0] + 1
        tele += int((np.linalg.norm(W[k] - W[k - 1], axis=1) > 0.25).sum())
        k2 = np.nonzero(o[2:] & o[1:-1] & o[:-2])[0] + 2
        if len(k2):
            acc.append(np.linalg.norm(W[k2] - 2 * W[k2 - 1] + W[k2 - 2], axis=1) * 1000)
        Jc = J[s][o]
        if len(Jc) > 10:
            size.append(np.linalg.norm(Jc[:, 9] - Jc[:, 0], axis=1) * 100)
            L = np.stack([np.linalg.norm(Jc[:, a] - Jc[:, b], axis=1) for a, b in BONES], 1)
            cv.append(np.median(L.std(0) / np.maximum(L.mean(0), 1e-6)) * 100)
            dist = np.linalg.norm(Jc[:, 0], axis=1)
            reach_ok += int(((dist > 0.15) & (dist < 0.90)).sum())
            reach_n += len(dist)
    m["teleports_per_min"] = round(tele / max(minutes, 1e-9), 2)
    m["jitter_mm"] = round(float(np.median(np.concatenate(acc))), 2) if acc else None
    hs = np.concatenate(size) if size else np.zeros(0)
    m["hand_size_cm"] = round(float(np.median(hs)), 1) if len(hs) else None
    m["hand_size_plausible"] = round(float(((hs > 7) & (hs < 11)).mean()), 3) if len(hs) else None
    m["bone_cv_pct"] = round(float(np.mean(cv)), 1) if cv else None
    m["reach_plausible"] = round(reach_ok / reach_n, 3) if reach_n else None
    tc = d / "tracklets.csv"
    if tc.exists():
        import pandas as pd
        t = pd.read_csv(tc)
        if "label" in t:
            rej = t[t["label"] == "reject"]
            m["candidates"] = int(len(t))
            m["rejected_share"] = round(len(rej) / max(len(t), 1), 3)
            reasons = Counter(r.strip() for rs in rej.get("reasons", []).fillna("") for r in str(rs).split(";") if r.strip())
            m["top_reject_reasons"] = ", ".join(f"{k} ({v})" for k, v in reasons.most_common(3))
    q = d / "quality.json"
    if q.exists():
        qq = json.load(open(q))
        m["usable_share"] = qq.get("usable", {}).get("share")
        m["grade"] = qq.get("grade")
    return m


def slam_metrics(d: Path) -> dict | None:
    p = d / "world_stats.json"
    if not p.exists():
        return None
    s = json.load(open(p))
    return {"clip": d.name, "backend": s.get("backend"), "rot_agree_deg": s.get("rotation_1s_vs_ps4_median_deg"),
            "left_hand_to_head_m": s.get("left_hand_to_head_median_m"),
            "right_hand_to_head_m": s.get("right_hand_to_head_median_m"), "up_method": s.get("up_method")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+")
    ap.add_argument("--slam", nargs="*", default=[])
    ap.add_argument("--out", default="accuracy_report")
    a = ap.parse_args(argv)
    import pandas as pd
    rows = []
    for r in a.results:
        for d in sorted(Path(r).iterdir()):
            if (d / "hands.jsonl").exists():
                try:
                    rows.append(clip_metrics(d))
                    print("ok  ", d.name, flush=True)
                except Exception as exc:  # noqa: BLE001
                    print("skip", d.name, exc, flush=True)
    df = pd.DataFrame(rows).drop_duplicates("clip", keep="last")
    sl = [x for s in a.slam for d in sorted(Path(s).iterdir()) if d.is_dir() for x in [slam_metrics(d)] if x]
    w = df["minutes"].sum()
    wavg = lambda c: float((df[c] * df["minutes"]).sum() / w) if c in df and df[c].notna().all() else float(df[c].mean())  # noqa: E731
    summary = {
        "clips": int(len(df)), "minutes": round(float(w), 1),
        "hands_tracked_share": round((wavg("left_tracked") + wavg("right_tracked")) / 2, 3),
        "lr_swaps_per_min": round(float((df["lr_swaps_per_min"] * df["minutes"]).sum() / w), 2),
        "teleports_per_min": round(float((df["teleports_per_min"] * df["minutes"]).sum() / w), 2),
        "jitter_mm_median": round(float(df["jitter_mm"].median()), 2),
        "hand_size_cm_median": round(float(df["hand_size_cm"].median()), 1),
        "hand_size_plausible": round(wavg("hand_size_plausible"), 3),
        "bone_cv_pct_mean": round(float(df["bone_cv_pct"].mean()), 1),
        "reach_plausible": round(wavg("reach_plausible"), 3),
    }
    if "rejected_share" in df:
        summary["candidates_total"] = int(df["candidates"].sum())
        summary["rejected_share"] = round(float(df["rejected_share"].mean()), 3)
    if "usable_share" in df and df["usable_share"].notna().any():
        summary["usable_share_mean"] = round(float(df["usable_share"].mean()), 3)
    if sl:
        s = pd.DataFrame(sl)
        summary["slam_clips"] = len(s)
        summary["slam_rot_agree_deg_median"] = round(float(pd.to_numeric(s["rot_agree_deg"]).median()), 2)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_clip.csv", index=False)
    if sl:
        pd.DataFrame(sl).to_csv(out / "slam.csv", index=False)
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    lines = [
        f"{summary['clips']} clips, {summary['minutes']} min of real factory video",
        f"both hands tracked {100 * summary['hands_tracked_share']:.0f}% of frames on average",
        f"left/right swaps: {summary['lr_swaps_per_min']} per minute; impossible jumps: {summary['teleports_per_min']} per minute",
        f"hand size {summary['hand_size_cm_median']} cm (adult 7-11 cm: {100 * summary['hand_size_plausible']:.0f}% of frames)",
        f"wrist within arm's reach in {100 * summary['reach_plausible']:.0f}% of tracked frames",
        f"bone lengths stable over time: {summary['bone_cv_pct_mean']}% variation; jitter {summary['jitter_mm_median']} mm/frame^2",
    ]
    if "rejected_share" in summary:
        lines.append(f"{summary['candidates_total']} candidate tracks, {100 * summary['rejected_share']:.0f}% rejected as not the wearer's hand")
    if "slam_rot_agree_deg_median" in summary:
        lines.append(f"SLAM vs our head tracking agree within {summary['slam_rot_agree_deg_median']} deg ({summary['slam_clips']} clips)")
    (out / "summary.md").write_text("# Accuracy report (label-free)\n\n" + "\n".join(f"- {x}" for x in lines) + "\n")
    print("\n".join(lines))
    print(f"\nsaved: {out}/summary.md, summary.json, per_clip.csv")


if __name__ == "__main__":
    main()
