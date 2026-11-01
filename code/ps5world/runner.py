"""Run the world step for one clip (used by run_world.py and run_slam.py)."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from . import backends
from .export import export_csv, export_jsonl, export_mesh, export_rerun
from .ps4_io import hand_boxes, load_camera, load_hands, load_head_rotation, trim_hands
from .render_world import render_world
from .world import build_world, checks, geocalib_gravity

log = logging.getLogger("ps5world")
REQUIRED_PS4 = ("hands.jsonl", "camera.json", "head_rot.pkl")


def missing_ps4_files(ps4: str | Path) -> list[str]:
    return [f for f in REQUIRED_PS4 if not (Path(ps4) / f).exists()]


# SLAM settings tried in order. Bundle adjustment can diverge (energy -> NaN, then a singular camera matrix)
# on hard stretches: fast head turns, a frame mostly covered by hands/arms, blur. Fixed intrinsics are the
# usual cure; ViPE at a lower frame rate (bigger motion per step, fewer keyframes) is the last SLAM try.
SLAM_ATTEMPTS = [
    {"name": "refine_intrinsics", "optimize_intrinsics": True, "robust": True, "intrinsics_damping": 100.0},
    {"name": "fixed_intrinsics", "optimize_intrinsics": False, "robust": True, "intrinsics_damping": 1.0},
    {"name": "fixed_intrinsics_10fps", "optimize_intrinsics": False, "robust": True, "intrinsics_damping": 1.0,
     "fps": 10.0},
]


def _vipe_with_retries(clip, out, hands, cam4, N, fps, slam_fps, slam_width, max_seconds, timings, attempts):
    import shutil
    vout = out / "vipe"
    if (vout / "pose" / "clip.npz").exists():                       # reuse a finished SLAM run
        try:
            traj = backends.read_vipe(vout, N, (cam4["width"], cam4["height"]))
            attempts.append({"name": "reused", "ok": True})
            return traj
        except Exception as exc:  # noqa: BLE001
            log.warning("previous SLAM output unusable (%s); running again", exc)
    for a in SLAM_ATTEMPTS:
        shutil.rmtree(vout, ignore_errors=True)
        t1 = time.time()
        try:
            backends.run_vipe(clip, out, hand_boxes(hands), a.get("fps", slam_fps), slam_width, max_seconds,
                              video_fps=fps, optimize_intrinsics=a["optimize_intrinsics"], robust=a["robust"],
                              intrinsics_damping=a["intrinsics_damping"])
            traj = backends.read_vipe(vout, N, (cam4["width"], cam4["height"]))
            attempts.append({"name": a["name"], "ok": True, "seconds": round(time.time() - t1, 1)})
            timings["slam_s"] = round(time.time() - t1, 1)
            log.info("SLAM ok with %s", a["name"])
            return traj
        except Exception as exc:  # noqa: BLE001 - try the next setting
            msg = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
            attempts.append({"name": a["name"], "ok": False, "seconds": round(time.time() - t1, 1), "error": msg})
            log.warning("SLAM attempt %s failed after %.0f s: %s", a["name"], time.time() - t1, msg)
            _free_gpu()
    return None


def _free_gpu():
    try:
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _video_frames(clip) -> int:
    import cv2
    cap = cv2.VideoCapture(str(clip))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    cap.release()
    return max(n, 0)


def run_world(clip: str, ps4: str, out: str, backend: str = "auto", slam_fps: float = 15.0, slam_width: int = 768,
              max_seconds: float | None = None, geocalib: bool = True, render: bool = True,
              rerun: bool = True) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    miss = missing_ps4_files(ps4)
    if miss:
        raise FileNotFoundError(f"PS4 folder {ps4} is missing {miss}")
    t0 = time.time()
    hands = load_hands(ps4)
    cam4 = load_camera(ps4)
    head_R = load_head_rotation(ps4)
    fps = hands["fps"] or 30.0
    n_video = _video_frames(clip)
    n_keep = hands["n_frames"]
    if max_seconds:
        n_keep = min(n_keep, int(round(max_seconds * fps)))
    if n_video:                                                    # PS4 result longer than the video file
        n_keep = min(n_keep, n_video)
    hands = trim_hands(hands, n_keep)                              # every per-frame array, not just the count
    if head_R is not None:
        head_R = head_R[:n_keep]
    N = hands["n_frames"]
    if backend == "auto":
        backend = "vipe" if backends.have_vipe() else "rotation"
    log.info("%s: %d frames, backend %s", Path(clip).name, N, backend)
    timings = {}
    attempts: list[dict] = []
    traj = None
    if backend == "vipe":
        traj = _vipe_with_retries(clip, out, hands, cam4, N, fps, slam_fps, slam_width, max_seconds, timings,
                                  attempts)
        if traj is None:
            if head_R is None:
                raise RuntimeError(f"SLAM failed in every attempt and there is no head_rot.pkl: {attempts}")
            log.warning("SLAM failed in every attempt; using the rotation-only backend")
            backend = "rotation"
    if traj is None:
        if head_R is None:
            raise FileNotFoundError("rotation backend needs head_rot.pkl in the PS4 result folder")
        traj = backends.rotation_only(head_R, cam4, N)
    g = geocalib_gravity(clip, traj["R_wc"]) if geocalib else None
    world = build_world(hands, cam4, traj, fps, g)
    stats = checks(world, traj, head_R)
    export_jsonl(world, str(out / "world_hands.jsonl"))
    export_csv(world, str(out / "trajectory.csv"))
    stats["hand_meshes"] = export_mesh(world, str(out / "world_hands_mesh.npz"))
    if rerun:
        t1 = time.time()
        export_rerun(world, str(out / "scene.rrd"), clip)
        timings["rerun_s"] = round(time.time() - t1, 1)
    if render:
        t1 = time.time()
        render_world(clip, hands, world, str(out / "world_render.mp4"), title=Path(clip).stem.replace("Copy of ", ""))
        timings["render_s"] = round(time.time() - t1, 1)
    timings["total_s"] = round(time.time() - t0, 1)
    stats["seconds"] = timings
    stats["slam_attempts"] = attempts
    stats["clip"] = str(clip)
    stats["ps4_folder"] = str(ps4)
    json.dump(stats, open(out / "world_stats.json", "w"), indent=2)
    return stats
