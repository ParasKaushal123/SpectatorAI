"""Live upload: trim/resample the clip, run code/run.py on CPU in a background thread, report progress."""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import cv2

from .clips import FFMPEG, JOBS, ROOT

CODE = ROOT / "code"
MAX_SECONDS = 10
SAMPLE_FPS = 8
LONG_SIDE = 960
PHONE_HFOV_DEG = 68.0           # typical phone main camera (~26 mm equivalent)
SEC_PER_FRAME_CPU = float(os.environ.get("SPECTATOR_SEC_PER_FRAME", "2.6"))   # ETA only; measured 2.6 s/frame on this PC (CPU)

# progress = fraction done when this log line appears
STAGES = [("=== ", 0.05, "Reading video"), ("intrinsics", 0.08, "Camera model"), ("M2", 0.10, "Detecting hands"),
          ("M3 tracklets", 0.40, "Linking tracks"), ("M3P probe", 0.65, "3D probe + wearer check"),
          ("M3A assign", 0.70, "Left / right / reject"), ("M4 recon", 0.88, "3D hands (WiLoR)"),
          ("M5 ", 0.95, "Smoothing + infill"), ("done:", 1.0, "Done")]

_lock = threading.Lock()
_threads = {}
_processes = {}
_cancelled = set()
_running = threading.Semaphore(1)       # one CPU job at a time


def _save(job: dict) -> None:
    d = JOBS / job["id"]
    with _lock:
        if job["id"] in _cancelled or not d.is_dir():
            return
        tmp = d / "job.tmp"
        tmp.write_text(json.dumps(job, indent=1))
        tmp.replace(d / "job.json")


def load(job_id: str) -> dict | None:
    if not re.fullmatch(r"[a-f0-9]{10}", job_id):
        return None
    p = JOBS / job_id / "job.json"
    return json.loads(p.read_text()) if p.is_file() else None


def _phone_calibration(w: int, h: int, path: Path) -> None:
    """Pinhole camera written as the pipeline's KB4 model: theta_d = tan(theta) series => r = f*tan(theta)."""
    f = (w / 2) / math.tan(math.radians(PHONE_HFOV_DEG / 2))
    path.write_text(json.dumps({"width": w, "height": h, "fx": f, "fy": f, "cx": w / 2, "cy": h / 2,
                                "k1": 1 / 3, "k2": 2 / 15, "k3": 17 / 315, "k4": 62 / 2835,
                                "source": f"phone_pinhole_hfov{PHONE_HFOV_DEG:.0f}"}))


def create(upload_path: Path, filename: str, camera: str, model: dict) -> dict:
    jid = uuid.uuid4().hex[:10]
    d = JOBS / jid
    d.mkdir(parents=True)
    job = {"id": jid, "filename": filename, "camera": camera, "status": "queued", "progress": 0.0,
           "stage": "Queued", "created": time.time(), "variant": model["variant"], "model_label": model["label"],
           "error": None, "eta_s": None}
    _save(job)
    thread = threading.Thread(target=_run, args=(job, upload_path), daemon=True)
    _threads[jid] = thread
    thread.start()
    return job


def _run(job: dict, upload_path: Path) -> None:
    d = JOBS / job["id"]
    log_path = d / "run.log"
    try:
        with _running:
            if job["id"] in _cancelled:
                return
            job.update(status="running", stage="Preparing video", progress=0.02, started=time.time())
            _save(job)
            inp = d / "input.mp4"
            vf = (f"fps={SAMPLE_FPS},scale='if(gt(iw,ih),min({LONG_SIDE},iw),-2)':'if(gt(iw,ih),-2,min({LONG_SIDE},ih))'")
            r = subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(upload_path), "-t", str(MAX_SECONDS),
                                "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(inp)],
                               capture_output=True, text=True)
            if r.returncode != 0 or not inp.is_file():
                (d / "ffmpeg.log").write_text(r.stderr or "")
                raise RuntimeError("Could not read this file as a video. Try an .mp4 or .mov straight from a phone or camera.")
            cap = cv2.VideoCapture(str(inp))
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if n < 5:
                raise RuntimeError("Video is too short (need at least half a second).")
            job.update(frames=n, width=w, height=h, eta_s=round(n * SEC_PER_FRAME_CPU + 20))
            sets = ["device=cpu", f"max_seconds={MAX_SECONDS}", "camera.use_geocalib=false", "render.enabled=false",
                    "proposals.imgsz=640", f"paths.cache_root={(d / 'out').as_posix()}"]
            if job["camera"] == "phone":
                _phone_calibration(w, h, d / "calib.json")
                sets.append(f"camera.calibration_json={(d / 'calib.json').as_posix()}")
            cmd = [sys.executable, "run.py", "--clip", str(inp), "--set", *sets]
            job.update(stage="Starting pipeline", progress=0.04)
            _save(job)
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
                   "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", "1")}
            with open(log_path, "w") as lf:
                p = subprocess.Popen(cmd, cwd=CODE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     env=env, bufsize=1)
                _processes[job["id"]] = p
                if job["id"] in _cancelled:
                    p.terminate()
                for line in p.stdout:
                    lf.write(line)
                    lf.flush()
                    for key, frac, label in STAGES:
                        if key in line and frac > job["progress"]:
                            job.update(progress=frac, stage=label)
                            _save(job)
                rc = p.wait()
            if rc != 0:
                tail = log_path.read_text(errors="replace").strip().splitlines()[-6:]
                raise RuntimeError("Pipeline failed: " + " | ".join(tail)[-600:])
            out = next((d / "out").rglob("hands.jsonl"), None)
            if out is None:
                raise RuntimeError("Pipeline finished but wrote no hands.jsonl")
            (out.parent / "model.json").write_text(json.dumps({"variant": job["variant"], "label": job["model_label"]}))
            job.update(status="done", progress=1.0, stage="Done", finished=time.time(),
                       clip_key=f"job__{job['id']}")
    except Exception as exc:  # noqa: BLE001
        job.update(status="error", error=str(exc), stage="Failed")
    finally:
        _save(job)
        _processes.pop(job["id"], None)
        try:
            upload_path.unlink(missing_ok=True)
        except OSError:
            pass


def cancel_all():
    for jid in list(_threads):
        _cancelled.add(jid)
    for process in list(_processes.values()):
        if process.poll() is None:
            process.terminate()
    for thread in list(_threads.values()):
        thread.join(timeout=20)
    if any(t.is_alive() for t in _threads.values()):
        raise RuntimeError("CPU preview is stopping; retry deletion in a moment")
    _threads.clear()
