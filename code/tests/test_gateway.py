"""Real HTTP fake-model Colab at :8765 + local gateway, through exports and cleanup."""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "code/tests"))
import cv2
import httpx
from fastapi.testclient import TestClient
import test_synthetic


def main():
    with tempfile.TemporaryDirectory(dir=REPO / "dashboard", prefix="gateway-test-") as td:
        root = Path(td)
        env = dict(os.environ, PS4_JOBS_DIR=str(root / "remote"), PS4_API_KEY="synthetic-test-only",
                   PS4_MAX_SECONDS="2", PS4_OVERRIDES=json.dumps({"device": "cpu",
                    "camera": {"use_geocalib": False}, "proposals": {"backend": "fake", "imgsz": 640},
                    "probe": {"verifier": "fake"}, "recon": {"backend": "fake"}}))
        log = (root / "remote.log").open("w")
        server = subprocess.Popen([sys.executable, "-m", "uvicorn", "server.app:app", "--host", "127.0.0.1",
                                   "--port", "8765"], cwd=REPO / "code", env=env, stdout=log, stderr=log)
        try:
            for _ in range(120):
                if server.poll() is not None:
                    raise AssertionError((root / "remote.log").read_text())
                try:
                    if httpx.get("http://127.0.0.1:8765/health").json()["warm"]:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(.5)
            else:
                raise AssertionError("Fake server did not warm up: " + (root / "remote.log").read_text())
            os.environ["PS4_API_URL"] = "http://127.0.0.1:8765"
            os.environ["PS4_API_KEY"] = "synthetic-test-only"
            from dashboard.backend import clips as C, colab, workspace as W
            C.DATA, C.WEB, C.JOBS = root / "local", root / "local/web", root / "local/jobs"
            colab.SETTINGS = C.DATA / "settings.json"
            W.ROOT = C.DATA / "workspaces"
            from dashboard.backend.app import app
            video = root / "synthetic.mp4"
            test_synthetic.make_video(str(video))
            with TestClient(app) as client:
                assert client.get("/api/colab/status").json()["connected"]
                with video.open("rb") as fh:
                    response = client.post("/api/workspaces", files={"file": ("synthetic.mp4", fh)})
                assert response.status_code == 200, response.text
                wid = response.json()["id"]
                states = {"uploaded"}
                for _ in range(900):
                    job = client.get(f"/api/workspaces/{wid}").json()
                    states.add(job["status"])
                    assert job["status"] != "failed", job
                    if job["status"] == "ready" and job.get("remote_deleted"):
                        break
                    time.sleep(.2)
                else:
                    raise AssertionError(job)
                clip = client.get(f"/api/workspaces/{wid}/clip").json()
                assert clip["scene"]["head_quat_xyzw"] and clip["quality"]["segments"]
                assert job["render"]["status"] == "done", job
                for url in ("video", "export/render"):
                    r = client.get(f"/api/workspaces/{wid}/{url}", headers={"Range": "bytes=0-99"})
                    assert r.status_code == 206 and len(r.content) == 100
                assert httpx.get("http://127.0.0.1:8765/jobs", headers={"X-API-Key": "synthetic-test-only"}).json() == []
                # Real WebM -> MP4 conversion, then decode the result.
                webm = root / "view.webm"
                subprocess.run([C.FFMPEG, "-y", "-loglevel", "error", "-i", str(video), "-t", "1",
                                "-c:v", "libvpx-vp9", str(webm)], check=True)
                with webm.open("rb") as fh:
                    r = client.post(f"/api/workspaces/{wid}/export/view", files={"file": ("view.webm", fh)})
                assert r.status_code == 200, r.text
                cap = cv2.VideoCapture(str(W.directory(wid) / "view.mp4"))
                assert cap.read()[0], "Export MP4 is not decodable"
                cap.release()
                assert client.delete(f"/api/workspaces/{wid}").status_code == 200
                assert client.get(f"/api/workspaces/{wid}").status_code == 404
                assert client.delete("/api/data").status_code == 200
                print("states:", sorted(states))
                print("timings:", job["timings"])
                print("GATEWAY CHECKS PASSED: scene, quality, local video, render, remote deletion, view MP4, delete-all")
        finally:
            if os.name == "nt":
                # Windows venv python.exe is a launcher; terminate its server descendants too.
                subprocess.run(["taskkill", "/PID", str(server.pid), "/T", "/F"], capture_output=True)
            else:
                server.terminate()
            server.wait(timeout=20)
            log.close()


if __name__ == "__main__":
    main()
