"""Local gateway API. No Colab credentials are returned to the browser."""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, Body
from fastapi.responses import FileResponse
from . import clips as C, colab, workspace as W, jobs as J

router = APIRouter(prefix="/api")


@router.get("/colab/status")
def connection(force: bool = False):
    return colab.status(force)


@router.post("/colab/url")
def configure(values: dict = Body(...)):
    allowed = {k: v for k, v in values.items() if k in ("url", "profile", "auto_render", "retention_h")}
    try:
        colab.save_settings(allowed)
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return colab.status(force=True)


@router.post("/workspaces")
def upload_workspace(file: UploadFile = File(...)):
    return W.create(file.file, file.filename or "video.mp4")


@router.get("/workspaces")
def workspaces():
    return W.all_jobs()


@router.get("/workspaces/{wid}")
def workspace(wid: str):
    return W.load(wid)


@router.post("/workspaces/{wid}/retry")
def retry(wid: str):
    return W.retry(wid)


@router.post("/workspaces/{wid}/cpu")
def cpu_preview(wid: str):
    from ps4v4.models.wilor_loader import model_info
    job = W.load(wid)
    video = W.directory(wid) / "video.mp4"
    if not video.exists():
        raise HTTPException(409, "Video is still being prepared")
    C.JOBS.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=C.JOBS, suffix=".mp4", delete=False) as fh:
        dst = Path(fh.name)
    shutil.copyfile(video, dst)
    return J.create(dst, job["name"], "fisheye", model_info())


def file_response(path, name=None):
    if not path.is_file():
        raise HTTPException(404, "File is not ready")
    return FileResponse(path, filename=name,
                        media_type="video/mp4" if path.suffix == ".mp4" else None)


@router.get("/workspaces/{wid}/clip")
def clip(wid: str):
    W.load(wid)
    return file_response(W.directory(wid) / "clip.json")


@router.get("/workspaces/{wid}/video")
def video(wid: str):
    W.load(wid)
    return file_response(W.directory(wid) / "video.mp4")


@router.get("/workspaces/{wid}/export/render")
def rendered(wid: str):
    W.load(wid)
    return file_response(W.directory(wid) / "out/render.mp4", "render.mp4")


@router.post("/workspaces/{wid}/export/render")
def request_render(wid: str):
    j = W.load(wid)
    if (W.directory(wid) / "out/render.mp4").exists():
        return j
    if not j.get("remote_id"):
        # With auto-render off the remote data is already deleted; reprocess on Retry to render.
        raise HTTPException(409, "Enable auto-render in Settings and upload again to generate a render")
    colab.request("POST", f"/jobs/{j['remote_id']}/render")
    return W.update(wid, render={"status": "queued"})


@router.post("/workspaces/{wid}/export/view")
def view(wid: str, file: UploadFile = File(...)):
    W.load(wid)
    d = W.directory(wid)
    src = d / "view.webm"
    try:
        size = 0
        with src.open("wb") as fh:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > 500 * 1024 * 1024:
                    raise HTTPException(413, "Recording exceeds 500 MB")
                fh.write(chunk)
        target = d / "view.tmp.mp4"
        subprocess.run([C.FFMPEG, "-y", "-loglevel", "error", "-i", str(src), "-an", "-c:v", "libx264",
                        "-preset", "veryfast", "-crf", "23", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)],
                       check=True, capture_output=True, timeout=180)
        target.replace(d / "view.mp4")
    except subprocess.CalledProcessError as exc:
        raise HTTPException(400, "Could not decode the browser recording") from exc
    finally:
        src.unlink(missing_ok=True)
    return {"url": f"/api/workspaces/{wid}/download/view.mp4"}


@router.get("/workspaces/{wid}/download/{name}")
def download(wid: str, name: str):
    W.load(wid)
    if name not in (*W.OUTPUTS, "view.mp4", "render.mp4"):
        raise HTTPException(404, "Unknown download")
    d = W.directory(wid)
    return file_response(d / name if name == "view.mp4" else d / "out" / name, name)


@router.delete("/workspaces/{wid}")
def delete(wid: str):
    W.delete(wid)
    return {"deleted": wid}


@router.delete("/data")
def delete_all():
    for j in W.all_jobs():
        W.delete(j["id"])
    try:
        J.cancel_all()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    for target in (C.WEB, C.JOBS):
        if target.exists():
            shutil.rmtree(target)
    return {"deleted": "all workspace, CPU preview, browser video and export data"}
