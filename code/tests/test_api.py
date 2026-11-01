"""API end to end with stand-in models (no GPU): upload -> progress -> result -> files -> delete.
Run: python tests/test_api.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

tmp = Path(tempfile.mkdtemp())
os.environ["PS4_JOBS_DIR"] = str(tmp / "jobs")
os.environ["PS4_API_TOKEN"] = "secret"
os.environ["PS4_MAX_SECONDS"] = "6"
os.environ["PS4_OVERRIDES"] = json.dumps({
    "device": "cpu", "camera": {"use_geocalib": False}, "proposals": {"backend": "fake", "imgsz": 640},
    "probe": {"verifier": "fake"}, "recon": {"backend": "fake"}, "render": {"height": 360}})

from fastapi.testclient import TestClient  # noqa: E402

import test_synthetic  # noqa: E402
from server.app import app  # noqa: E402


def main():
    video = str(tmp / "clip.mp4")
    test_synthetic.make_video(video)                                   # 10 s synthetic clip
    c = TestClient(app)
    c.__enter__()
    H = {"X-API-Key": "secret"}
    assert c.get("/health").json()["ok"]
    assert c.get("/jobs").status_code == 401, "auth not enforced"
    with open(video, "rb") as fh:
        job = c.post("/jobs", headers=H, files={"file": ("my clip.mp4", fh, "video/mp4")}, data={"profile": "fast", "render": "after"}).json()
    print("created:", job["id"], job["video"], "truncated:", job["truncated"])
    assert job["truncated"] and job["processed_seconds"] == 6
    stages = set()
    for _ in range(600):
        j = c.get(f"/jobs/{job['id']}", headers=H).json()
        stages.add(j["stage"])
        if j["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    print("stages seen:", sorted(stages))
    assert j["status"] == "done", j.get("traceback")
    res = c.get(f"/jobs/{job['id']}/result", headers=H).json()
    q = res["quality"]
    assert "scene.json" in res["files"]
    assert j["profile"] == "fast" and j["stage_seconds"]
    scene = c.get(f"/jobs/{job['id']}/files/scene.json", headers=H).json()
    assert len(scene["head_quat_xyzw"]) == 180 and scene["gravity_source"] == "camera_y"
    for _ in range(600):
        j = c.get(f"/jobs/{job['id']}", headers=H).json()
        if j["render"]["status"] in ("done", "failed"):
            break
        time.sleep(.2)
    assert j["render"]["status"] == "done", j["render"]
    res = c.get(f"/jobs/{job['id']}/result", headers=H).json()
    print("files:", res["files"])
    print("quality:", q["score"], q["grade"], q["usable"], q["issues"])
    assert {"render.mp4", "hands.jsonl", "quality.json", "input.mp4"} <= set(res["files"])
    r = c.get(f"/jobs/{job['id']}/files/render.mp4?key=secret")            # key in the query, as <video src> does
    assert r.status_code == 200 and len(r.content) > 1000
    r = c.get(f"/jobs/{job['id']}/files/render.mp4", headers=H | {"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100, "video seeking (range requests) not supported"
    assert c.delete(f"/jobs/{job['id']}", headers=H).json()["deleted"] == job["id"]
    assert c.get(f"/jobs/{job['id']}", headers=H).status_code == 404
    from server.app import _save, sweep_expired
    expired = {"id": "a" * 12, "status": "queued", "created_at": time.time() - 7 * 3600}
    (tmp / "jobs" / expired["id"]).mkdir()
    _save(expired)
    sweep_expired()
    assert not (tmp / "jobs" / expired["id"]).exists()
    assert c.post("/jobs", headers=H, data={"profile": "invalid"}, files={"file": ("x.mp4", b"x")}).status_code == 400
    c.__exit__(None, None, None)
    print("API CHECKS PASSED (fast profile, scene, background render, Range, auth, TTL)")


if __name__ == "__main__":
    main()
