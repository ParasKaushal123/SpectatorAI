"""Persistent workspace queue. Videos stay local; credentials stay server-side."""
import json
import queue
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

import httpx
from fastapi import HTTPException
from . import clips as C, colab
from .media import transcode

ROOT = C.DATA / "workspaces"
_lock = threading.RLock()
_queue = queue.Queue()
_cancelled = set()
_started = False
OUTPUTS = ("hands.jsonl", "stats.json", "quality.json", "scene.json", "tracklets.csv", "robot_wrist_gripper.csv")


def directory(wid):
    if not re.fullmatch(r"[a-f0-9]{12}", wid):
        raise HTTPException(404, "No such workspace")
    return ROOT / wid


def load(wid):
    p = directory(wid) / "job.json"
    if not p.is_file():
        raise HTTPException(404, "No such workspace")
    return json.loads(p.read_text())


def all_jobs():
    if not ROOT.exists():
        return []
    return sorted([json.loads(p.read_text()) for p in ROOT.glob("*/job.json")],
                  key=lambda j: j["created"], reverse=True)


def update(wid, **kw):
    with _lock:
        if wid in _cancelled:
            raise InterruptedError("Workspace deleted")
        job = load(wid)
        job.update(kw)
        p = directory(wid) / "job.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(job, indent=1))
        tmp.replace(p)
        return job


def create(fh, filename):
    created = time.time()
    wid = uuid.uuid4().hex[:12]
    d = directory(wid)
    d.mkdir(parents=True)
    suffix = Path(filename).suffix.lower()
    if suffix not in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
        shutil.rmtree(d)
        raise HTTPException(400, "Choose an mp4, mov, avi, mkv or webm video")
    original = d / ("original" + suffix)
    size = 0
    with original.open("wb") as out:
        while chunk := fh.read(1024 * 1024):
            size += len(chunk)
            if size > 500 * 1024 * 1024:
                out.close()
                shutil.rmtree(d)
                raise HTTPException(413, "Video exceeds 500 MB")
            out.write(chunk)
    settings = colab.settings()
    job = {"id": wid, "name": Path(filename).name, "created": created, "status": "uploaded",
           "stage": "Queued locally", "progress": 0, "timings": {}, "stage_seconds": {}, "error": None,
           "profile": settings["profile"], "auto_render": settings["auto_render"],
           "render": {"status": "none"}, "remote_id": None}
    tmp = d / "job.tmp"
    tmp.write_text(json.dumps(job))
    tmp.replace(d / "job.json")
    _queue.put(wid)
    return job


def retry(wid):
    j = load(wid)
    if j["status"] != "failed":
        raise HTTPException(409, "Only failed workspaces can be retried")
    old_remote = j.get("remote_id")
    j = update(wid, status="uploaded", error=None, traceback=None, stage="Queued locally", progress=0,
               remote_id=None, render={"status": "none"}, started=time.time(), timings={})
    if old_remote:
        threading.Thread(target=_remote_delete, args=(wid, old_remote), daemon=True).start()
    _queue.put(wid)
    return j


def _check(wid):
    if wid in _cancelled:
        raise InterruptedError("Workspace deleted")


def _remote(wid, method, path, **kw):
    for attempt in range(3):
        _check(wid)
        try:
            return colab.request(method, path, **kw)
        except (ConnectionError, httpx.TransportError):
            colab.discover(force=True)
            if attempt == 2:
                raise
            time.sleep(1)


def _fetch(wid, jid, name, dest):
    for attempt in range(3):
        _check(wid)
        try:
            colab.download(jid, name, dest)
            return
        except (ConnectionError, httpx.TransportError):
            colab.discover(force=True)
            if attempt == 2:
                raise


def _remote_delete(wid, jid):
    try:
        colab.request("DELETE", f"/jobs/{jid}")
        if wid not in _cancelled and load(wid).get("remote_id") == jid:
            update(wid, remote_id=None, remote_deleted=True)
        return True
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            if wid not in _cancelled and load(wid).get("remote_id") == jid:
                update(wid, remote_id=None, remote_deleted=True)
            return True
    except (ConnectionError, httpx.TransportError):
        pass
    return False


def _render_fetch(wid, jid):
    try:
        while True:
            _check(wid)
            try:
                remote = _remote(wid, "GET", f"/jobs/{jid}").json()
                render = remote.get("render", {"status": "none"})
                update(wid, render=render if render["status"] != "done" else {"status": "fetching"})
                if render["status"] == "done":
                    _fetch(wid, jid, "render.mp4", directory(wid) / "out/render.mp4")
                    update(wid, render=render)
                    break
                if render["status"] in ("failed", "none"):
                    break
            except (ConnectionError, httpx.TransportError):
                update(wid, render={"status": "waiting", "error": "Waiting for Colab"})
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    update(wid, render={"status": "failed", "error": "Remote job expired"})
                    return
                raise
            time.sleep(1.5)
        while not _remote_delete(wid, jid):
            _check(wid)
            time.sleep(2)
    except InterruptedError:
        _remote_delete(wid, jid)
    except Exception as exc:
        if wid not in _cancelled:
            update(wid, render={"status": "failed", "error": str(exc)})


def _run(wid):
    d = directory(wid)
    job = load(wid)
    timings = job.get("timings", {})
    begin = job.get("started", job["created"])
    update(wid, started=begin)
    def mark(state, **extra):
        return update(wid, **({"status": state, "stage": state.capitalize()} | extra))
    try:
        video = d / "video.mp4"
        if not video.is_file():
            t = time.monotonic()
            mark("preparing", progress=.02)
            original = next(d.glob("original.*"))
            meta = transcode(original, video)
            original.unlink()
            timings["preparing"] = round(time.monotonic() - t, 3)
            update(wid, video=meta, timings=timings)
        if job.get("remote_id") and job["status"] == "ready":
            threading.Thread(target=_render_fetch, args=(wid, job["remote_id"]), daemon=True).start()
            return
        remote_id = job.get("remote_id")
        while True:
            _check(wid)
            if not colab.discover()["connected"]:
                mark("waiting", stage="Waiting for Colab", progress=.05)
                # Let the next upload prepare its local preview while the GPU is offline.
                _queue.put(wid)
                time.sleep(2)
                return
            if remote_id:
                try:
                    _remote(wid, "GET", f"/jobs/{remote_id}")
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code != 404:
                        raise
                    remote_id = None
            if not remote_id:
                t = time.monotonic()
                mark("uploading", progress=.06)
                with video.open("rb") as fh:
                    wrapper = colab.ProgressFile(fh, lambda x: update(wid, upload_progress=x, progress=.06 + x * .09))
                    remote = _remote(wid, "POST", "/jobs", files={"file": ("video.mp4", wrapper, "video/mp4")},
                                     data={"name": job["name"], "profile": job["profile"],
                                           "render": "after" if job["auto_render"] else "none"}).json()
                remote_id = remote["id"]
                timings["uploading"] = round(time.monotonic() - t, 3)
                update(wid, remote_id=remote_id, timings=timings)
            mark("processing")
            t = time.monotonic()
            try:
                while True:
                    remote = _remote(wid, "GET", f"/jobs/{remote_id}").json()
                    update(wid, stage=remote.get("stage"), progress=.15 + .7 * remote["progress"],
                           stage_seconds=remote.get("stage_seconds", {}), queue_position=remote.get("queue_position"),
                           render=remote.get("render", {"status": "none"}))
                    if remote["status"] == "failed":
                        update(wid, traceback=remote.get("traceback"))
                        raise RuntimeError(remote.get("error", "Pipeline failed"))
                    if remote["status"] == "done":
                        break
                    time.sleep(1.5)
                break
            except (ConnectionError, httpx.TransportError):
                colab.discover(force=True)
                time.sleep(2)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    remote_id = None
                else:
                    raise
        timings["processing"] = round(time.monotonic() - t, 3)
        mark("fetching", progress=.9)
        t = time.monotonic()
        (d / "out").mkdir(exist_ok=True)
        for name in OUTPUTS:
            _fetch(wid, remote_id, name, d / "out" / name)
        variant = json.loads((d / "out/stats.json").read_text()).get("model", {}).get("variant", "pretrained")
        clip = C.Clip("workspace__" + wid, job["name"], variant, "workspace", d / "out")
        compact = C.build_clip_json(clip)
        (d / "clip.json").write_text(json.dumps(compact, separators=(",", ":")))
        timings["fetching"] = round(time.monotonic() - t, 3)
        timings["total"] = round(time.time() - begin, 3)
        mark("ready", progress=1, timings=timings, quality=compact.get("quality"))
        threading.Thread(target=_render_fetch, args=(wid, remote_id), daemon=True).start()
    except InterruptedError:
        if load_safe := locals().get("remote_id", job.get("remote_id")):
            _remote_delete(wid, load_safe)
    except Exception as exc:
        if wid not in _cancelled:
            update(wid, status="failed", error=str(exc), stage="Failed")


def _worker():
    while True:
        wid = _queue.get()
        try:
            if wid not in _cancelled and (directory(wid) / "job.json").exists():
                _run(wid)
        except Exception:
            import logging
            logging.exception("Workspace worker failed")
        finally:
            if wid in _cancelled:
                shutil.rmtree(directory(wid), ignore_errors=True)
            _queue.task_done()


def delete(wid):
    with _lock:
        job = load(wid)
        _cancelled.add(wid)
        if job.get("remote_id"):
            # Active remote work cannot be deleted yet. Retry until it stops (or TTL removes it).
            def cleanup():
                while not _remote_delete(wid, job["remote_id"]):
                    colab.discover(force=True)
                    time.sleep(2)
            threading.Thread(target=cleanup, daemon=True).start()
        shutil.rmtree(directory(wid), ignore_errors=True)


def start():
    global _started
    if _started:
        return
    _started = True
    ROOT.mkdir(parents=True, exist_ok=True)
    for job in reversed(all_jobs()):
        if job["status"] not in ("ready", "failed") or job.get("remote_id"):
            _queue.put(job["id"])
    threading.Thread(target=_worker, daemon=True).start()
    def sweep():
        while True:
            cutoff = time.time() - colab.settings()["retention_h"] * 3600
            for job in all_jobs():
                if job["status"] in ("ready", "failed") and job["created"] < cutoff:
                    delete(job["id"])
            time.sleep(60)
    threading.Thread(target=sweep, daemon=True).start()
