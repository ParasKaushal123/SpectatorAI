"""Create a synthetic-only workspace for browser QA. No model downloads or real footage."""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO / "code/tests"))
from test_synthetic import make_video
from ps4v4.config import load_config
from ps4v4.pipeline import run_clip
from dashboard.backend import clips as C
from dashboard.backend.media import transcode


def main():
    d = C.DATA / "workspaces/bbbbbbbbbbbb"
    if (d / "job.json").exists():
        print("Synthetic QA workspace already exists")
        return
    d.mkdir(parents=True, exist_ok=True)
    raw = d / "synthetic.mp4"
    make_video(str(raw))
    meta = transcode(raw, d / "video.mp4")
    raw.unlink()
    cfg = load_config(REPO / "code/configs/default.yaml", {
        "device": "cpu", "camera": {"use_geocalib": False}, "paths": {"cache_root": str(d)},
        "proposals": {"backend": "fake", "imgsz": 640}, "probe": {"verifier": "fake"},
        "recon": {"backend": "fake"}, "render": {"enabled": True, "height": 360}})
    start = time.monotonic()
    stats = run_clip(str(d / "video.mp4"), cfg, clip_name="out")
    clip = C.build_clip_json(C.Clip("workspace__bbbbbbbbbbbb", "Synthetic QA", "pretrained", "workspace", d / "out"))
    (d / "clip.json").write_text(json.dumps(clip))
    (d / "job.json").write_text(json.dumps({
        "id": "bbbbbbbbbbbb", "name": "Synthetic QA", "created": time.time(), "synthetic": True,
        "status": "ready", "progress": 1, "video": meta, "quality": clip["quality"],
        "stage_seconds": stats["seconds"], "timings": {"total": round(time.monotonic() - start, 2)},
        "render": {"status": "done", "progress": 1}}))
    print("Open http://127.0.0.1:8000/#/workspace/bbbbbbbbbbbb")


if __name__ == "__main__":
    main()
