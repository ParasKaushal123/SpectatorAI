"""PS5 world view for ONE clip, given the video file and its PS4 result folder.
(For a folder of clips + a folder of PS4 results, use run_slam.py.)

Examples:
    python run_world.py --clip clip.mp4 --ps4 results/clip --out world/clip                 # ViPE if installed
    python run_world.py --clip clip.mp4 --ps4 results/clip --out world/clip --backend rotation
    python run_world.py --clip clip.mp4 --ps4 results/clip --out world/clip --max-seconds 60 --slam-fps 15

Outputs in --out: world_render.mp4, scene.rrd (open with `rerun scene.rrd`), world_hands.jsonl,
trajectory.csv, world_stats.json, vipe/ (raw SLAM artifacts, reused on re-runs).
"""
from __future__ import annotations

import argparse
import json
import logging

from ps5world.runner import run_world


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--ps4", required=True, help="PS4 v4 result folder of this clip")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="auto", choices=["auto", "vipe", "rotation"])
    ap.add_argument("--slam-fps", type=float, default=15.0, help="frames per second fed to SLAM")
    ap.add_argument("--slam-width", type=int, default=768, help="SLAM input width (640 on 8 GB GPUs)")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--no-geocalib", action="store_true", help="skip GeoCalib gravity")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--no-rerun", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    stats = run_world(a.clip, a.ps4, a.out, a.backend, a.slam_fps, a.slam_width, a.max_seconds,
                      not a.no_geocalib, not a.no_render, not a.no_rerun)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
