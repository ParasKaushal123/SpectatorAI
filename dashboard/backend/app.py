"""SpectatorAI dashboard API. Run: python -m uvicorn dashboard.backend.app:app --port 8000 (from the repo root)."""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import clips as C
from . import jobs as J
from . import workspace as W
from .routes import router as workspace_router
from .dataset import router as dataset_router

sys.path.insert(0, str(C.ROOT / "code"))
from ps4v4.models.wilor_loader import model_info  # noqa: E402

app = FastAPI(title="SpectatorAI")
app.add_middleware(GZipMiddleware, minimum_size=2000)
app.include_router(workspace_router)
app.include_router(dataset_router)
SYNTHETIC = Path(__file__).with_name("synthetic_test.json")
TRAINING = C.RESULTS / "training"


@app.on_event("startup")
def _warm():
    W.start()
    threading.Thread(target=C.warm_all, daemon=True).start()


@app.get("/api/model")
def api_model():
    """Which model a new upload would run with."""
    return model_info()


@app.get("/api/clips")
def api_clips():
    return [c.summary() for c in C.list_clips()]


def _clip(key: str) -> C.Clip:
    c = C.get_clip(key)
    if c is None:
        raise HTTPException(404, f"no clip {key}")
    return c


@app.get("/api/clips/{key}")
def api_clip(key: str):
    return FileResponse(C.clip_json_path(_clip(key)), media_type="application/json")


@app.get("/api/clips/{key}/video")
def api_video(key: str):
    return FileResponse(C.video_path(_clip(key)), media_type="video/mp4")


@app.get("/api/clips/{key}/download/{name}")
def api_download(key: str, name: str):
    if name not in ("hands.jsonl", "robot_wrist_gripper.csv", "stats.json"):
        raise HTTPException(404)
    p = _clip(key).dir / name
    if not p.is_file():
        raise HTTPException(404)
    return FileResponse(p, filename=f"{key}_{name}")


# ---- metrics ------------------------------------------------------------------------------------
def _clip_metrics(c: C.Clip) -> dict:
    s = C.load_stats(c.dir)
    L, R, tr = s.get("left", {}), s.get("right", {}), s.get("tracklets", {})
    acc = [x.get("wrist_accel_median_mm_per_frame2") for x in (L, R) if x.get("wrist_accel_median_mm_per_frame2")]
    return {"key": c.key, "name": c.name, "variant": c.variant, "origin": c.origin, "model_label": c.model_label,
            "frames": s.get("frames"), "fps": s.get("fps_video"),
            "observed_left": L.get("share_observed"), "observed_right": R.get("share_observed"),
            "infilled": round((L.get("share_infilled", 0) + R.get("share_infilled", 0)) / 2, 3),
            "estimated": round((L.get("share_estimated_out_of_view", 0) + R.get("share_estimated_out_of_view", 0)) / 2, 3),
            "out_of_view": round((L.get("share_out_of_view", 0) + R.get("share_out_of_view", 0)) / 2, 3),
            "jitter_mm": round(sum(acc) / len(acc), 2) if acc else None,
            "depth_jumps": L.get("depth_jumps_over_10cm", 0) + R.get("depth_jumps_over_10cm", 0),
            "tracklets_total": tr.get("total"), "tracklets_rejected": tr.get("rejected"),
            "tracklets_left": tr.get("left"), "tracklets_right": tr.get("right"),
            "camera_source": s.get("camera", {}).get("source"), "z_ref_m": s.get("z_ref_m"),
            "seconds": s.get("seconds", {})}


def _aggregate(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    frames = sum(r["frames"] or 0 for r in rows)
    tot = sum(r["tracklets_total"] or 0 for r in rows)
    rej = sum(r["tracklets_rejected"] or 0 for r in rows)
    jit = [r["jitter_mm"] for r in rows if r["jitter_mm"] is not None]
    obs = [v for r in rows for v in (r["observed_left"], r["observed_right"]) if v is not None]
    return {"clips": len(rows), "frames": frames, "minutes": round(sum((r["frames"] or 0) / (r["fps"] or 30)
                                                                      for r in rows) / 60, 1),
            "tracklets_total": tot, "tracklets_rejected": rej, "reject_rate": round(rej / tot, 3) if tot else None,
            "depth_jumps": sum(r["depth_jumps"] for r in rows),
            "depth_jumps_per_1k_frames": round(1000 * sum(r["depth_jumps"] for r in rows) / frames, 2) if frames else None,
            "jitter_mm_median": round(sorted(jit)[len(jit) // 2], 2) if jit else None,
            "observed_mean": round(sum(obs) / len(obs), 3) if obs else None,
            "estimated_mean": round(sum(r["estimated"] for r in rows) / len(rows), 3),
            "infilled_mean": round(sum(r["infilled"] for r in rows) / len(rows), 3)}


@app.get("/api/metrics")
def api_metrics():
    a100 = [_clip_metrics(c) for c in C.list_clips() if c.origin == "a100"]
    by = {v: [r for r in a100 if r["variant"] == v] for v in ("pretrained", "finetuned")}
    pairs = sorted({r["name"] for r in by["pretrained"]} & {r["name"] for r in by["finetuned"]})
    return {"clips": a100, "aggregate": {v: _aggregate(rows) for v, rows in by.items()}, "paired_clips": pairs,
            "synthetic_test": json.loads(SYNTHETIC.read_text()) if SYNTHETIC.is_file() else None}


# ---- USP evidence: rejection reasons + concrete clip moments -------------------------------------
REASON_GROUPS = [("Another person's hand", ("far", "arm away", "too close")),
                 ("Body parts (knee, foot, leg, head)", ("looks like knee", "looks like foot", "looks like leg", "looks like head")),
                 ("Objects (wood, surface, tool, cloth)", ("looks like wood", "looks like surface", "looks like tool", "looks like cloth", "static")),
                 ("Lying on a forearm", ("on a forearm",)),
                 ("Second left / right in frame", ("side taken",)),
                 ("Flicker / too weak", ("no 3d", "short", "weak", "kp/box"))]


def _group(reasons: str) -> str | None:
    r = reasons.lower()
    for name, keys in REASON_GROUPS:          # first matching group wins (order = specificity)
        if any(k in r for k in keys):
            return name
    return None


@app.get("/api/evidence")
def api_evidence():
    counts = {name: 0 for name, _ in REASON_GROUPS}
    best: dict[str, tuple] = {}

    def keep(tag, score, clip, frame, note):
        if tag not in best or score > best[tag][0]:
            best[tag] = (score, clip, frame, note)

    for c in C.list_clips():
        if c.origin != "a100" or c.variant != "pretrained":
            continue
        p = c.dir / "tracklets.csv"
        fps = C.load_stats(c.dir).get("fps_video") or 30.0
        if p.is_file():
            with open(p, newline="") as fh:
                for row in csv.DictReader(fh):
                    label, reasons = row.get("label", ""), row.get("reasons") or ""
                    n, start = int(float(row.get("n") or 0)), int(float(row.get("start") or 0))
                    if label == "reject":
                        g = _group(reasons)
                        if g:
                            counts[g] += 1
                        if "far" in reasons or "arm away" in reasons:
                            keep("other_person", n, c, start, f"rejected: {reasons}")
                        if any(k in reasons for k in ("knee", "foot", "leg", "wood", "surface")):
                            keep("objects", n, c, start, f"rejected: {reasons}")
                    elif row.get("crossing") == "True" and label in ("left", "right"):
                        keep("crossing", n, c, start, f"{label} hand through a crossing")
        hj = C.clip_json_path(c)
        d = json.loads(hj.read_text())
        for side in ("left", "right"):
            src, run, a = d["hands"][side]["src"], 0, 0
            for i, s in enumerate(src + [0]):
                if s == 3:
                    a, run = (i, 1) if run == 0 else (a, run + 1)
                else:
                    if run:
                        keep("estimated", run, c, a, f"{side} hand estimated out of view for {run / fps:.1f} s")
                    run = 0
        s = C.load_stats(c.dir)
        both = min(s.get("left", {}).get("share_observed", 0), s.get("right", {}).get("share_observed", 0))
        keep("metric3d", both, c, 0, "both hands measured most of the clip")
    moments = {k: {"clip_key": v[1].key, "clip": v[1].name, "frame": v[2],
                   "t": round(v[2] / (C.load_stats(v[1].dir).get("fps_video") or 30.0), 1), "note": v[3]}
               for k, v in best.items()}
    return {"reasons": [{"group": k, "count": v} for k, v in counts.items()], "moments": moments}


# ---- training logs ------------------------------------------------------------------------------
def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


@app.get("/api/training")
def api_training():
    """Reads whatever the team drops in results/training/: training.json (setup), dataset.json, *.csv loss logs."""
    if not TRAINING.is_dir():
        return {"available": False}
    out: dict = {"available": False, "setup": None, "dataset": None, "curves": []}
    for name, key in (("training.json", "setup"), ("setup.json", "setup"), ("dataset.json", "dataset")):
        p = TRAINING / name
        if p.is_file():
            out[key] = json.loads(p.read_text())
    for p in sorted(TRAINING.glob("*.csv")):
        with open(p, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            continue
        cols = list(rows[0].keys())
        x = next((c for c in ("epoch", "step", "global_step", "iter", "iteration") if c in cols), None)
        ys = [c for c in cols if "loss" in c.lower() or "mpjpe" in c.lower() or "pa_" in c.lower()]
        if not ys:
            continue
        pts = [{"x": _num(r[x]) if x else i, **{y: _num(r[y]) for y in ys}} for i, r in enumerate(rows)]
        out["curves"].append({"file": p.name, "x": x or "row", "series": ys, "points": pts[:: max(1, len(pts) // 600)]})
    out["available"] = bool(out["setup"] or out["dataset"] or out["curves"])
    return out


# ---- live upload --------------------------------------------------------------------------------
@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...), camera: str = Form("phone")):
    if camera not in ("phone", "fisheye"):
        raise HTTPException(400, "camera must be phone or fisheye")
    C.JOBS.mkdir(parents=True, exist_ok=True)
    fd, tmpname = tempfile.mkstemp(suffix=Path(file.filename or "x.mp4").suffix or ".mp4", dir=C.JOBS)
    os.close(fd)
    tmp = Path(tmpname)
    with open(tmp, "wb") as fh:
        shutil.copyfileobj(file.file, fh)
    if tmp.stat().st_size > 300 * 1024 * 1024:
        tmp.unlink()
        raise HTTPException(413, "File over 300 MB; trim it to 10 s first.")
    return J.create(tmp, file.filename or "upload.mp4", camera, model_info())


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    job = J.load(job_id)
    if job is None:
        raise HTTPException(404)
    return job


# ---- frontend -----------------------------------------------------------------------------------
PREVIEW_ASSETS = C.DATA / "design-reference" / "landing" / "assets"
if PREVIEW_ASSETS.is_dir():
    app.mount("/api/preview-assets", StaticFiles(directory=PREVIEW_ASSETS), name="preview-assets")

DIST = C.DASHBOARD / "frontend" / "dist"
if DIST.is_dir():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="ui")
