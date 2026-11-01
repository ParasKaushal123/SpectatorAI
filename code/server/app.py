"""HTTP API around the PS4 pipeline + clip quality, meant to run on Colab (A100) behind a tunnel, with the
frontend running on your own computer.

    POST   /jobs                      upload a clip (multipart field "file"), returns the job
    GET    /jobs                      all jobs, newest first
    GET    /jobs/{id}                 status: queued | running | done | failed, current stage, progress 0-1
    GET    /jobs/{id}/result          when done: stats.json + quality.json + list of files
    GET    /jobs/{id}/files/{name}    input.mp4, render.mp4, hands.jsonl, robot_wrist_gripper.csv, stats.json,
                                      quality.json, tracklets.csv (videos support seeking)
    DELETE /jobs/{id}                 delete the job and all its files
    GET    /health                    GPU, queue length, limits

Auth: if PS4_API_TOKEN is set, send it as header "X-API-Key" or as query "?key=" (for <video src>).
Jobs run one at a time on the GPU (models stay loaded between jobs). Settings via environment:
PS4_JOBS_DIR (default /content/jobs), PS4_MAX_SECONDS (default 90; longer clips are cut),
PS4_CONFIG (default configs/default.yaml), PS4_OVERRIDES (JSON dict merged into the config).
"""
from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import threading
import time
import traceback
import uuid
import re
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from .activity import TransferMiddleware, activity

CODE_DIR = Path(__file__).resolve().parents[1]
JOBS = Path(os.environ.get("PS4_JOBS_DIR", "/content/jobs"))
TOKEN = os.environ.get("PS4_API_KEY") or os.environ.get("PS4_API_TOKEN") or None
TTL_H = float(os.environ.get("PS4_JOB_TTL_H", "6"))
_warm = False
_warm_error = None
_render_queue = queue.Queue()

MAX_SECONDS = float(os.environ.get("PS4_MAX_SECONDS", "90"))
CONFIG = os.environ.get("PS4_CONFIG", str(CODE_DIR / "configs" / "default.yaml"))
OVERRIDES = json.loads(os.environ.get("PS4_OVERRIDES", "{}"))
FILES = ("input.mp4", "render.mp4", "hands.jsonl", "robot_wrist_gripper.csv", "stats.json", "quality.json",
         "tracklets.csv", "scene.json")
STAGES = {"M1": "Camera model", "M2": "Detecting hands", "M1H": "Head motion", "M3": "Linking tracks",
          "M3P": "Checking candidates", "M3A": "Deciding left / right", "M4": "3D hands", "M5": "Smoothing",
          "M6": "Rendering video", "MQ": "Rating the clip"}
WEIGHTS = {"M1": 0.03, "M2": 0.30, "M1H": 0.15, "M3": 0.10, "M3P": 0.07, "M3A": 0.01, "M4": 0.10, "M5": 0.02,
           "M6": 0.15, "MQ": 0.07}

log = logging.getLogger("ps4api")
JOBS.mkdir(parents=True, exist_ok=True)
_lock = threading.RLock()
_queue: "queue.Queue[str]" = queue.Queue()


# ---- job store (one job.json per job, so jobs survive a server restart) ---------------------
def _jpath(jid: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{12}", jid):
        raise HTTPException(404, "no such job")
    return JOBS / jid / "job.json"


def _load(jid: str) -> dict:
    p = _jpath(jid)
    if not p.exists():
        raise HTTPException(404, "no such job")
    return json.loads(p.read_text())


def _save(job: dict) -> None:
    with _lock:                                                           # atomic: readers never see half a file
        p = _jpath(job["id"])
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(job, indent=1))
        os.replace(tmp, p)


def _update(jid: str, **kw) -> dict:
    with _lock:
        job = json.loads(_jpath(jid).read_text())
        job.update(kw)
        _save(job)
    return job


def _out_dir(jid: str) -> Path:
    _jpath(jid)  # validate identifiers before joining paths (also on Windows)
    return JOBS / jid / "out" / "clip"


# ---- worker ----------------------------------------------------------------------------------
def job_config(jid, profile, render=False):
    from ps4v4.config import load_config, deep_update
    import yaml
    over = {}
    if profile == "fast":
        over = yaml.safe_load((CODE_DIR / "configs/profiles/fast.yaml").read_text())
    over = deep_update(over, OVERRIDES)
    over = deep_update(over, {"paths": {"cache_root": str(JOBS / jid / "out")}, "max_seconds": MAX_SECONDS})
    if render:
        over = deep_update(over, {"render": {"enabled": True, "height": 720, "crf": 23, "supersample": 1}})
    return load_config(CONFIG, over)


def _run(jid: str) -> None:
    from ps4v4.config import load_config, deep_update
    from ps4v4.pipeline import run_clip

    job = _update(jid, status="running", started_at=time.time(), stage="Starting", progress=0.0)
    done = set()
    started = {}
    seconds = {}

    def progress(module: str, state: str):
        if state == "start":
            started[module] = time.monotonic()
        elif state == "done":
            seconds[module] = round(time.monotonic() - started.get(module, time.monotonic()), 3)
        if state in ("done", "cached"):
            done.add(module)
        frac = sum(WEIGHTS.get(m, 0) for m in done)
        _update(jid, stage=STAGES.get(module, module) + ("" if state != "start" else "…"),
                progress=round(min(frac, 0.99), 3), stage_seconds=seconds)

    try:
        cfg = job_config(jid, job.get("profile", "full"))
        stats = run_clip(str(JOBS / jid / "input.mp4"), cfg, clip_name="clip", progress=progress)
        from ps4v4.models.wilor_loader import model_info
        stats["model"] = model_info()
        (_out_dir(jid) / "stats.json").write_text(json.dumps(stats))
        _update(jid, model=stats["model"], status="done", stage="Done", progress=1.0, finished_at=time.time(),
                quality=stats.get("quality"), seconds=round(time.time() - job["started_at"], 1))
        if job.get("render_after"):
            queue_render(jid)
    except Exception as exc:                                              # noqa: BLE001 - report, keep serving
        log.exception("job %s failed", jid)
        _update(jid, status="failed", stage="Failed", error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc()[-4000:], finished_at=time.time())


def _worker() -> None:
    warm_up()
    while True:
        jid = _queue.get()
        try:
            if _jpath(jid).is_file():
                _run(jid)
        except Exception:
            log.exception("worker could not run %s", jid)
        finally:
            _queue.task_done()


def queue_render(jid):
    with _lock:
        job = _load(jid)
        if job["status"] != "done":
            raise HTTPException(409, "job is not done")
        if job.get("render", {}).get("status") in ("queued", "running", "done"):
            return job
        if (_out_dir(jid) / "render.mp4").is_file():
            return _update(jid, render={"status": "done", "progress": 1, "error": None})
        job = _update(jid, render={"status": "queued", "progress": 0, "error": None})
        _render_queue.put(jid)
        return job


def _renderer():
    # Read immutable cached outputs directly: no MQ rerun or stats overwrite during rendering.
    from ps4v4.cache import ClipCache
    from ps4v4.render import render
    from ps4v4.video import video_meta
    while True:
        jid = _render_queue.get()
        try:
            with _lock:
                if not _jpath(jid).is_file():
                    continue
                _update(jid, render={"status": "running", "progress": .05, "error": None})
            cfg = job_config(jid, _load(jid).get("profile", "full"), render=True)
            cache = ClipCache(JOBS / jid / "out", "clip", cfg)
            recon = cache.load_pickle("recon.pkl")
            render(video_meta(str(JOBS / jid / "input.mp4"), MAX_SECONDS), cache.load_pickle("sequence.pkl"),
                   recon.get("faces"), str(cache.path("render.mp4")), cfg, _load(jid)["name"],
                   cache.load_pickle("assign.pkl")["rejected"],
                   progress=lambda p: _update(jid, render={"status": "running", "progress": round(p, 3), "error": None}))
            _update(jid, render={"status": "done", "progress": 1, "error": None})
        except Exception as exc:
            log.exception("render failed")
            if _jpath(jid).exists():
                _update(jid, render={"status": "failed", "progress": 0, "error": str(exc)})
        finally:
            _render_queue.task_done()


def sweep_expired(now=None):
    now = now or time.time()
    with _lock:
        for p in JOBS.glob("*/job.json"):
            job = json.loads(p.read_text())
            active = job.get("status") == "running" or job.get("render", {}).get("status") == "running"
            if not active and now - job.get("created_at", now) > TTL_H * 3600:
                shutil.rmtree(p.parent)


def _sweeper():
    while True:
        try:
            sweep_expired()
        except Exception:
            log.exception("TTL sweep failed")
        time.sleep(60)


def warm_up():
    global _warm, _warm_error
    try:
        from ps4v4.config import load_config
        from ps4v4.models.wilor_loader import load_wilor
        from ps4v4.models.verifier import make_verifier
        from ps4v4.camera import geocalib_model
        from ps4v4.pipeline import run_clip
        import cv2
        import numpy as np
        cfg = load_config(CONFIG, OVERRIDES)
        if cfg.proposals.backend != "fake":
            load_wilor(cfg.device)
            if cfg.probe.verifier == "clip" and make_verifier(cfg) is None:
                raise RuntimeError("CLIP warm-up failed")
            if cfg.camera.use_geocalib:
                geocalib_model(cfg.device)
        with tempfile.TemporaryDirectory(dir=JOBS, prefix="warm-") as td:
            video = str(Path(td) / "warm.mp4")
            out = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
            for i in range(20):
                frame = np.full((240, 320, 3), 100, np.uint8)
                cv2.circle(frame, (90 + i, 170), 35, (40, 220, 40), -1)
                out.write(frame)
            out.release()
            cfg = job_config("warmup", "fast")
            cfg["paths"] = {"cache_root": str(Path(td) / "cache")}
            cfg["max_seconds"] = 2
            cfg["render"] = dict(cfg.render) | {"enabled": False}
            run_clip(video, cfg)
        _warm = True
    except Exception as exc:
        _warm_error = str(exc)
        log.exception("warm-up failed; first job may be cold")


_started = False


@asynccontextmanager
async def lifespan(app):
    global _started
    if not _started:
        _started = True
        for p in sorted(JOBS.glob("*/job.json")):
            j = json.loads(p.read_text())
            if j.get("status") in ("queued", "running"):
                _update(j["id"], status="queued")
                _queue.put(j["id"])
            elif j.get("render", {}).get("status") in ("queued", "running"):
                _update(j["id"], render={"status": "none"})
                queue_render(j["id"])
        for fn in (_worker, _renderer, _sweeper):
            threading.Thread(target=fn, daemon=True).start()
    yield


# ---- API -------------------------------------------------------------------------------------
app = FastAPI(title="PS4 hand tracking API", version="1.1", lifespan=lifespan)
app.add_middleware(TransferMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def auth(x_api_key: str | None = Header(default=None), key: str | None = Query(default=None)) -> None:
    if TOKEN and (x_api_key or key) != TOKEN:
        raise HTTPException(401, "missing or wrong API key")


@app.get("/health")
def health():
    gpu = None
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        pass
    return {"ok": True, "gpu": gpu, "queued": _queue.qsize(), "max_seconds": MAX_SECONDS, "auth": bool(TOKEN), "warm": _warm, "warm_error": _warm_error, **activity.snapshot()}


@app.post("/jobs", dependencies=[Depends(auth)])
async def create_job(file: UploadFile = File(...), name: str | None = Form(default=None),
                     profile: str = Form("fast"), render: str = Form("none")):
    if profile not in ("fast", "full") or render not in ("none", "after"):
        raise HTTPException(400, "profile must be fast/full; render must be none/after")
    import cv2
    jid = uuid.uuid4().hex[:12]
    d = JOBS / jid
    d.mkdir(parents=True)
    dst = d / "input.mp4"
    with open(dst, "wb") as fh:
        total = 0
        while chunk := file.file.read(1024 * 1024):
            total += len(chunk)
            if total > 100 * 1024 * 1024:
                fh.close()
                shutil.rmtree(d, ignore_errors=True)
                raise HTTPException(413, "upload exceeds 100 MB")
            fh.write(chunk)
    cap = cv2.VideoCapture(str(dst))
    ok = cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not ok or n <= 0:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, "not a readable video")
    dur = n / fps
    job = {"profile": profile, "render_after": render == "after", "stage_seconds": {},
           "render": {"status": "none", "progress": 0, "error": None}, "id": jid, "name": name or Path(file.filename or "clip.mp4").stem, "status": "queued", "stage": "Queued",
           "progress": 0.0, "created_at": time.time(), "video": {"duration_s": round(dur, 2), "fps": fps,
                                                                   "width": w, "height": h, "frames": n},
           "processed_seconds": round(min(dur, MAX_SECONDS), 2), "truncated": dur > MAX_SECONDS}
    _save(job)
    _queue.put(jid)
    return job


@app.get("/jobs", dependencies=[Depends(auth)])
def list_jobs():
    jobs = [json.loads(p.read_text()) for p in JOBS.glob("*/job.json")]
    return sorted(jobs, key=lambda j: j.get("created_at", 0), reverse=True)


@app.get("/jobs/{jid}", dependencies=[Depends(auth)])
def get_job(jid: str):
    job = _load(jid)
    if job["status"] == "queued":
        job["queue_position"] = list(_queue.queue).index(jid) + 1 if jid in list(_queue.queue) else None
    return job


@app.get("/jobs/{jid}/result", dependencies=[Depends(auth)])
def get_result(jid: str):
    job = _load(jid)
    if job["status"] != "done":
        raise HTTPException(409, f"job is {job['status']}")
    out = _out_dir(jid)
    read = lambda f: json.loads((out / f).read_text()) if (out / f).exists() else None   # noqa: E731
    files = [f for f in FILES if (JOBS / jid / f).exists() or (out / f).exists()]
    return {"job": job, "stats": read("stats.json"), "quality": read("quality.json"), "files": files}


@app.get("/jobs/{jid}/files/{name}", dependencies=[Depends(auth)])
def get_file(jid: str, name: str):
    _load(jid)
    if name not in FILES:
        raise HTTPException(404, "unknown file")
    p = JOBS / jid / name if name == "input.mp4" else _out_dir(jid) / name
    if not p.exists():
        raise HTTPException(404, "file not ready")
    media = {".mp4": "video/mp4", ".json": "application/json", ".jsonl": "application/x-ndjson",
             ".csv": "text/csv"}[p.suffix]
    return FileResponse(p, media_type=media, filename=name if p.suffix != ".mp4" else None)


@app.delete("/jobs/{jid}", dependencies=[Depends(auth)])
def delete_job(jid: str):
    with _lock:
        job = _load(jid)
        if job["status"] == "running" or job.get("render", {}).get("status") == "running":
            raise HTTPException(409, "job is running")
        shutil.rmtree(JOBS / jid, ignore_errors=True)
        return {"deleted": jid}


@app.post("/jobs/{jid}/render", dependencies=[Depends(auth)])
def render_job(jid: str):
    return queue_render(jid)
