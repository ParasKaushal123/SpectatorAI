"""Orchestrates M1-M6 for one clip with per-module caching (resume after disconnects).

v4 flow:  M1 camera -> M2 proposals -> M1H head rotation -> M3 tracklets -> M3P probe 3D
          -> M3A hand/side decision -> M4 full 3D -> M5 sequence -> M6 outputs
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from . import SIDES
from .cache import ClipCache
from .camera import estimate_head_rotation, estimate_intrinsics
from .config import MODULE_ORDER
from .export import export_jsonl, export_mesh, export_robot_csv, stability_stats
from .geometry import FisheyeCamera
from .assign import run_assign
from .probe import run_probe
from .proposals import run_proposals
from .quality import assess_clip
from .recon import run_recon
from .render import render
from .scene import scene_metadata
from .sequence import run_sequence
from .tracklets import build_tracklets
from .video import video_meta, cache_video, clear_video_cache

log = logging.getLogger(__name__)


def run_clip(video_path: str, cfg, clip_name: str | None = None, start_from: str | None = None,
             progress=None) -> dict:
    try:
        return _run_clip(video_path, cfg, clip_name, start_from, progress)
    finally:
        clear_video_cache(video_path)


def _run_clip(video_path: str, cfg, clip_name: str | None = None, start_from: str | None = None,
              progress=None) -> dict:
    """progress: optional callback(module_name, "start" | "done" | "cached") for live status (the API server)."""
    clip_name = clip_name or Path(video_path).stem.replace("Copy of ", "").replace(" ", "_")
    meta = video_meta(video_path, cfg.max_seconds)
    if cfg.get("video", {}).get("ram_cache", False):
        cache_video(meta)
    cache = ClipCache(cfg.paths.cache_root, clip_name, cfg)
    if start_from:
        cache.invalidate_from(start_from, MODULE_ORDER)
    timings: dict[str, float] = {}
    log.info("=== %s: %dx%d @ %.1f fps, %d frames ===", clip_name, meta.width, meta.height, meta.fps, meta.n_frames)

    def stage(name: str, fn, load):
        if cache.is_fresh(name):
            log.info("%s: cached", name)
            if progress:
                progress(name, "cached")
            return load()
        if progress:
            progress(name, "start")
        t0 = time.time()
        result = fn()
        timings[name] = time.time() - t0
        cache.mark_done(name)
        if progress:
            progress(name, "done")
        return result

    # M1 intrinsics
    cam = stage("M1", lambda: _save_cam(cache, estimate_intrinsics(meta, cfg)),
                lambda: FisheyeCamera.from_dict(cache.load_json("camera.json"), meta.width, meta.height))
    # M2 proposals
    props = stage("M2", lambda: _save(cache, "proposals.pkl", run_proposals(meta, cfg)["proposals"]),
                  lambda: cache.load_pickle("proposals.pkl"))
    # M1H head rotation (hand proposals are masked out of the background)
    head = stage("M1H", lambda: _save(cache, "head_rot.pkl", estimate_head_rotation(
        meta, cam, [p[:, :4] for p in props], cfg)), lambda: cache.load_pickle("head_rot.pkl"))
    # M3 tracklets (no decisions yet)
    tls = stage("M3", lambda: _save(cache, "tracklets.pkl", build_tracklets(meta, cam, head["R"], props, cfg)),
                lambda: cache.load_pickle("tracklets.pkl"))
    # M3P probe 3D on a sample of each tracklet
    probe = stage("M3P", lambda: _save(cache, "probe.pkl", run_probe(meta, cam, tls, cfg, cache.dir)),
                  lambda: cache.load_pickle("probe.pkl"))
    # M3A: is it the wearer's hand, and which side
    def _assign():
        res = run_assign(meta, tls, probe["probe"], cfg)
        import pandas as pd
        pd.DataFrame(res["table"]).to_csv(cache.path("tracklets.csv"), index=False)
        _save(cache, "tracks.pkl", res["tracks"])
        return _save(cache, "assign.pkl", res)
    asg = stage("M3A", _assign, lambda: cache.load_pickle("assign.pkl"))
    tracks = asg["tracks"]
    # M4 full 3D on accepted hands (probe frames reused)
    def _recon():
        by_id = {tr["id"]: tr for tr in tls}
        r = run_recon(meta, cam, tracks, cfg, cache.dir, probe=probe["probe"], tracklets_by_id=by_id)
        if r.get("faces") is None:
            r["faces"] = probe.get("faces")
        return _save(cache, "recon.pkl", r)
    recon = stage("M4", _recon, lambda: cache.load_pickle("recon.pkl"))
    # M5 sequence
    seq = stage("M5", lambda: _save(cache, "sequence.pkl", run_sequence(meta, cam, tracks, recon, head["R"], cfg,
                                                                        asg["z_ref"])),
                lambda: cache.load_pickle("sequence.pkl"))
    # M6 outputs (always cheap except render)
    cache.save_json("scene.json", scene_metadata(meta, cam, head, seq, cache.load_json("camera.json")))
    export_jsonl(meta, seq, str(cache.path("hands.jsonl")))
    export_robot_csv(meta, seq, str(cache.path("robot_wrist_gripper.csv")))
    export_mesh(seq, recon.get("faces"), str(cache.path("hands_mesh.npz")))     # hand surfaces for the world view
    if cfg.render.enabled:
        stage("M6", lambda: render(meta, seq, recon.get("faces"), str(cache.path("render.mp4")), cfg, clip_name,
                                     asg["rejected"]),
              lambda: str(cache.path("render.mp4")))
    # MQ clip quality: which parts are usable and why
    if cfg.quality.enabled:
        q = stage("MQ", lambda: _save_json(cache, "quality.json", assess_clip(meta, cam, head, seq, cfg)),
                  lambda: cache.load_json("quality.json"))
    stats = stability_stats(meta, seq, timings)
    if cfg.quality.enabled:
        stats["quality"] = {k: q[k] for k in ("score", "grade", "subscores")} | {"usable_share": q["usable"]["share"]}
    stats["camera"] = cam.to_dict()
    stats["head_rotation_inlier_median"] = float(_median(head["inliers"]))
    stats["z_ref_m"] = round(float(asg["z_ref"]), 3)
    labs = list(asg["labels"].values())
    stats["tracklets"] = {"total": len(labs), "left": sum("left" in l for l in labs),
                          "right": sum("right" in l for l in labs), "rejected": labs.count("reject")}
    cache.save_json("stats.json", stats)
    log.info("done: %s", {s: stats[s]["share_observed"] for s in SIDES})
    return stats


def _save(cache: ClipCache, name: str, obj):
    cache.save_pickle(name, obj)
    return obj


def _save_json(cache: ClipCache, name: str, obj):
    cache.save_json(name, obj)
    return obj


def _save_cam(cache: ClipCache, cam: FisheyeCamera) -> FisheyeCamera:
    cache.save_json("camera.json", cam.to_dict() | {"gravity_samples": getattr(cam, "gravity_samples", [])})
    return cam


def _median(x):
    import numpy as np
    x = np.asarray(x)
    x = x[x > 0]
    return np.median(x) if len(x) else 0.0
