"""Warm full/fast comparison on the same dashboard-transcoded clip; no footage in reports."""
import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import yaml

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE.parents[0]))
from ps4v4.config import load_config, deep_update
from ps4v4.pipeline import run_clip


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--clip", required=True)
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--out", default="cache/benchmark")
    a = p.parse_args()
    import shutil, subprocess
    import cv2
    def transcode(src, dst, seconds):
        cap = cv2.VideoCapture(str(src))
        fps = min(cap.get(5) or 30, 30)
        cap.release()
        vf = f"scale=w='min(1280,iw)':h='min(720,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2,fps={fps}"
        subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                        "-t", str(seconds), "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)], check=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    video = out / "video.mp4"
    transcode(Path(a.clip), video, min(a.seconds, 60))
    fast = yaml.safe_load((CODE / "configs/profiles/fast.yaml").read_text())
    run_root = out / ("run-" + uuid.uuid4().hex[:10])
    runs = {}
    for name, extra in (("warmup", fast), ("full", {}), ("fast", fast)):
        overrides = deep_update(extra, {"paths": {"cache_root": str(run_root / name)},
                                      "max_seconds": 2 if name == "warmup" else a.seconds})
        cfg = load_config(CODE / "configs/default.yaml", overrides)
        t0 = time.monotonic()
        stats = run_clip(str(video), cfg, clip_name="clip")
        runs[name] = {"wall_s": round(time.monotonic() - t0, 2), "stages": stats["seconds"],
                      "tracklets": stats["tracklets"], "quality": stats.get("quality", {}).get("score")}
    rows = {k: [json.loads(s) for s in (run_root / k / "clip/hands.jsonl").read_text().splitlines()]
            for k in ("full", "fast")}
    agreements, distances, swaps = [], [], 0
    for full, fast_row in zip(rows["full"], rows["fast"]):
        fh, qh = ({h["side"]: h for h in row["hands"]} for row in (full, fast_row))
        for side, other in (("left", "right"), ("right", "left")):
            x, y = fh[side], qh[side]
            obs = x["source"] == "observed", y["source"] == "observed"
            agreements.append(obs[0] == obs[1])
            if all(obs):
                d = np.linalg.norm(np.array(x["kp3d_head"][0]) - y["kp3d_head"][0])
                distances.append(d * 1000)
                if fh[other]["source"] == "observed":
                    alt = np.linalg.norm(np.array(fh[other]["kp3d_head"][0]) - y["kp3d_head"][0])
                    swaps += int(alt + .015 < d)
    agree = float(np.mean(agreements))
    median = float(np.median(distances)) if distances else None
    report = {"runs": runs, "side_presence_agreement": agree, "median_wrist_difference_mm": median,
              "suspected_swapped_frames": swaps,
              "quality_difference": runs["fast"]["quality"] - runs["full"]["quality"],
              "pass": runs["fast"]["wall_s"] <= 80 and agree >= .97 and median is not None
                      and median <= 15 and swaps == 0,
              "note": "Suspected swaps are a proximity check; inspect ambiguous crossings before acceptance."}
    (out / "benchmark.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
