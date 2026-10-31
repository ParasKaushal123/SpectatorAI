"""Adapter: PS4 pipeline outputs (hands.jsonl + stats.json) -> compact JSON + browser video for the dashboard.

Where results live (anything holding a hands.jsonl):
    results/<clip>/                 pretrained-WiLoR runs (A100, ps4v4.2)
    results/pretrained/<clip>/      same
    results/finetuned/<clip>/       runs with our fine-tuned checkpoint
    data/jobs/<id>/out/<clip>/      live uploads (model recorded in job.json)
A model.json next to hands.jsonl ({"variant": "...", "label": "..."}) overrides the guess.
Raw video for the overlay: raw_clips/<clip>.mp4, or raw_clips/mapping.json {"<clip>": "<file>.mp4"}.
Without it, the left (video) panel of render.mp4 is used, which already has the pipeline's overlay drawn in.
"""
from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RAW = ROOT / "raw_clips"
DATA = ROOT / "data"
WEB = DATA / "web"
JOBS = DATA / "jobs"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

LABELS = {"pretrained": "Pretrained WiLoR (baseline)", "finetuned": "Fine-tuned WiLoR (ours)"}
SIDES = ("left", "right")
SRC_CODE = {"none": 0, "observed": 1, "infilled": 2, "estimated": 3}


@dataclass
class Clip:
    key: str            # unique id used in URLs: <variant>__<name> or job__<id>
    name: str
    variant: str        # pretrained | finetuned
    origin: str         # a100 | upload
    dir: Path

    @property
    def model_label(self) -> str:
        m = self.dir / "model.json"
        if m.is_file():
            return json.loads(m.read_text()).get("label", LABELS.get(self.variant, self.variant))
        return LABELS.get(self.variant, self.variant)

    def summary(self) -> dict:
        stats = load_stats(self.dir)
        return {"key": self.key, "name": self.name, "variant": self.variant, "origin": self.origin,
                "model_label": self.model_label, "frames": stats.get("frames"), "fps": stats.get("fps_video"),
                "has_raw": raw_video(self) is not None}


def _variant_of(d: Path) -> str:
    m = d / "model.json"
    if m.is_file():
        return json.loads(m.read_text()).get("variant", "pretrained")
    return "finetuned" if "finetuned" in d.parts[-2:] else "pretrained"


def list_clips() -> list[Clip]:
    out: list[Clip] = []
    if RESULTS.is_dir():
        for hj in sorted(RESULTS.rglob("hands.jsonl")):
            d = hj.parent
            if "training" in d.relative_to(RESULTS).parts or not (d / "stats.json").is_file():
                continue
            v = _variant_of(d)
            out.append(Clip(f"{v}__{d.name}", d.name, v, "a100", d))
    if JOBS.is_dir():
        for jf in sorted(JOBS.glob("*/job.json")):
            job = json.loads(jf.read_text())
            if job.get("status") != "done":
                continue
            for hj in (jf.parent / "out").rglob("hands.jsonl"):
                out.append(Clip(f"job__{job['id']}", job.get("filename", job["id"]), job.get("variant", "pretrained"),
                                "upload", hj.parent))
    return out


def get_clip(key: str) -> Clip | None:
    return next((c for c in list_clips() if c.key == key), None)


def load_stats(d: Path) -> dict:
    p = d / "stats.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def raw_video(c: Clip) -> Path | None:
    if c.origin == "upload":
        p = c.dir.parents[1] / "input.mp4"
        return p if p.is_file() else None
    mp = RAW / "mapping.json"
    if mp.is_file():
        name = json.loads(mp.read_text()).get(c.name)
        if name and (RAW / name).is_file():
            return RAW / name
    p = RAW / f"{c.name}.mp4"
    return p if p.is_file() else None


# ---- compact clip JSON --------------------------------------------------------------------------
def build_clip_json(c: Clip) -> dict:
    stats = load_stats(c.dir)
    cam = stats.get("camera") or (json.loads((c.dir / "camera.json").read_text())
                                  if (c.dir / "camera.json").is_file() else {})
    fps = stats.get("fps_video") or 30.0
    t, sides = [], {s: {"src": [], "conf": [], "state": [], "kp2d": [], "kp3d": []} for s in SIDES}
    with open(c.dir / "hands.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            t.append(r["t"])
            by = {h["side"]: h for h in r["hands"]}
            for s in SIDES:
                h, o = by.get(s, {}), sides[s]
                o["src"].append(SRC_CODE.get(h.get("source", "none"), 0))
                o["conf"].append(h.get("conf", 0.0))
                o["state"].append(h.get("state", "OUT_OF_VIEW"))
                k2, k3 = h.get("kp2d"), h.get("kp3d_head")
                o["kp2d"].append([v for p in k2 for v in p] if k2 else None)
                o["kp3d"].append([v for p in k3 for v in p] if k3 else None)
    return {**c.summary(), "fps": fps, "n_frames": len(t), "width": cam.get("width"), "height": cam.get("height"),
            "camera_source": cam.get("source"), "overlay_baked": raw_video(c) is None, "t": t, "hands": sides,
            "stats": stats}


# ---- browser video ----------------------------------------------------------------------------
_locks: dict[str, threading.Lock] = {}
_glock = threading.Lock()


def _lock(key: str) -> threading.Lock:
    with _glock:
        return _locks.setdefault(key, threading.Lock())


def web_dir(c: Clip) -> Path:
    d = WEB / c.key
    d.mkdir(parents=True, exist_ok=True)
    return d


def clip_json_path(c: Clip) -> Path:
    p = web_dir(c) / "clip.json"
    with _lock(c.key + "json"):
        src = c.dir / "hands.jsonl"
        if not p.is_file() or p.stat().st_mtime < src.stat().st_mtime:
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(build_clip_json(c), separators=(",", ":")))
            tmp.replace(p)
    return p


def video_path(c: Clip) -> Path:
    """H.264 video matching hands.jsonl frame for frame: raw clip if we have it, else render.mp4's video panel."""
    p = web_dir(c) / "video.mp4"
    with _lock(c.key + "video"):
        if p.is_file():
            return p
        stats = load_stats(c.dir)
        n, fps = stats.get("frames"), stats.get("fps_video") or 30.0
        raw = raw_video(c)
        if raw is not None:
            src, vf = raw, "scale=-2:720"
        else:
            src = c.dir / "render.mp4"
            cam = stats.get("camera", {})
            lw = int(round(cam.get("width", 1920) * 720 / cam.get("height", 1080))) // 2 * 2
            vf = f"crop={lw}:720:0:0"            # render.mp4 = [video panel | 3D panel], 720 px high
        cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", str(src)]
        if n:
            cmd += ["-frames:v", str(n)]
        cmd += ["-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "27", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-r", str(fps), str(p.with_suffix(".tmp.mp4"))]
        subprocess.run(cmd, check=True)
        p.with_suffix(".tmp.mp4").replace(p)
    return p


def warm_all() -> None:
    for c in list_clips():
        try:
            clip_json_path(c)
            video_path(c)
        except Exception as exc:  # noqa: BLE001
            print(f"[warm] {c.key}: {exc}")
