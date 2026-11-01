"""Robot replay of a SLAM world view: two Franka Panda arms copy the worker's hands (inverse kinematics, no training).

    python run_robot.py --world <run_slam out>/<clip>/world_hands.jsonl --web <room web folder> --seconds 120

Writes robot.glb, robot.json, robot_motion.bin (for viewer.html) and robot_traj.json (joint angles + copyable grades)
into the web folder. Needs: pip install mujoco mink daqp trimesh scipy, and git (downloads the Panda model once).
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", required=True, help="world_hands.jsonl of the clip")
    ap.add_argument("--web", required=True, help="the room viewer folder (unzipped room_web.zip)")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--cache", default="~/.cache/spectator_robot", help="where the Panda model is downloaded")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    from robot.retarget import retarget
    s = retarget(a.world, a.web, a.seconds, a.cache)
    viewer = Path(__file__).resolve().parent / "room" / "viewer.html"
    if (Path(a.web) / "viewer.html").exists():
        shutil.copy(viewer, Path(a.web) / "viewer.html")             # viewer with the robot toggle
    print(json.dumps(s, indent=1))
    print(f"done -> {a.web}  (open with open_viewer.bat)")


if __name__ == "__main__":
    main()
