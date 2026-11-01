"""Real detectors see no hands in the warm-up clip; outputs must still succeed."""
import json
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from ps4v4.config import load_config
from ps4v4.pipeline import run_clip


def main():
    with tempfile.TemporaryDirectory(dir=CODE.parent / "dashboard/data", prefix="empty-clip-test-") as folder:
        root = Path(folder)
        video = root / "empty.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
        for _ in range(20):
            writer.write(np.full((240, 320, 3), 100, np.uint8))
        writer.release()
        cfg = load_config(CODE / "configs/default.yaml", {
            "device": "cpu", "paths": {"cache_root": str(root / "cache")},
            "camera": {"use_geocalib": False}, "video": {"ram_cache": True},
            "proposals": {"backend": "fake"}, "probe": {"verifier": "fake"},
            "recon": {"backend": "fake"}, "render": {"enabled": False}})
        stats = run_clip(str(video), cfg)
        out = root / "cache/empty"
        csv = pd.read_csv(out / "robot_wrist_gripper.csv")
        assert csv.empty and {"frame", "side", "x", "y", "z", "source"} <= set(csv.columns)
        assert stats["left"]["share_observed"] == stats["right"]["share_observed"] == 0
        assert len((out / "hands.jsonl").read_text().splitlines()) == 20
        assert len(json.loads((out / "scene.json").read_text())["head_quat_xyzw"]) == 20
        assert (out / "quality.json").is_file()
        print("EMPTY-CLIP CHECK PASSED: no-hand warm-up completes and exports valid outputs")


if __name__ == "__main__":
    main()
