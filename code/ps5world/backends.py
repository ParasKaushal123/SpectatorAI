"""Camera trajectory backends.

vipe      NVIDIA ViPE (Huang et al. 2025, Apache-2.0): learned dense bundle adjustment (DROID-style) with
          metric depth priors; `wide_angle` preset = unified (Mei) camera model for fisheye video, intrinsics
          from GeoCalib then refined. Moving things are removed from tracking with GroundingDINO + SAM masks
          ("person", "hand", "arm"); we add PS4's own hand boxes (grown, plus a strip to the bottom edge
          for the forearm) to that mask, the HaWoR recipe: hands never drive the camera estimate.
rotation  fallback without SLAM: PS4's head rotation (background KLT + RANSAC), head position fixed.

Every backend returns: R_wc (N,3,3), t_wc (N,3) camera->world for EVERY video frame (interpolated),
valid (N,) frames with a real estimate, cam (camera model at full resolution), points/colors (map).
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .cameras import EquidistantCamera, MeiCamera

log = logging.getLogger(__name__)


# ---- shared helpers -----------------------------------------------------------------
def interpolate_poses(inds: np.ndarray, T: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sparse cam->world 4x4 poses at frame indices `inds` -> rotation/translation for frames 0..n-1."""
    order = np.argsort(inds)
    inds, T = np.asarray(inds)[order], np.asarray(T)[order]
    R = Rotation.from_matrix(T[:, :3, :3])
    q = np.clip(np.arange(n), inds[0], inds[-1])
    R_all = Slerp(inds, R)(q).as_matrix() if len(inds) > 1 else np.repeat(T[:1, :3, :3], n, 0)
    t_all = np.stack([np.interp(q, inds, T[:, k, 3]) for k in range(3)], 1)
    valid = np.zeros(n, bool)
    valid[(np.arange(n) >= inds[0]) & (np.arange(n) <= inds[-1])] = True
    return R_all, t_all, valid


def _downscale_video(src: str, dst: str, width: int, max_seconds: float | None) -> None:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-vf", f"scale={width}:-2", "-an", "-c:v", "libx264", "-crf", "17", "-preset", "veryfast", dst]
    subprocess.run(cmd, check=True)


class VipeDiverged(RuntimeError):
    """SLAM ran but its result is unusable (NaN poses or intrinsics)."""


# ---- ViPE --------------------------------------------------------------------------------
class _BoxMaskProcessor:
    """ViPE stream processor: removes PS4 hand boxes (and the forearm strip below them) from the
    static-scene mask that SLAM tracks. frame.mask is True where the scene is static."""

    n_passes_required = 1

    def __init__(self, boxes: dict, scale: float, forearm: bool = True, min_static: float = 0.30):
        self.boxes, self.scale, self.forearm, self.min_static = boxes, scale, forearm, min_static
        self.relaxed = 0                                     # frames where our mask was reduced (too little left)

    def update_fps(self, f):
        return f

    def update_frame_size(self, s):
        return s

    def update_attributes(self, prev):
        from vipe.streams.base import FrameAttribute
        return prev | {FrameAttribute.MASK}

    def update_iterator(self, it, pass_idx):
        for i, fr in enumerate(it):
            yield self(i, fr)

    def __call__(self, frame_idx, frame):
        import torch
        H, W = frame.size()
        m = frame.mask if frame.mask is not None else torch.ones((H, W), dtype=torch.bool, device=frame.device)
        boxes = self.boxes.get(int(frame.raw_frame_idx), [])
        if not boxes:
            return frame
        # Too little static scene left makes bundle adjustment singular (energy -> NaN, singular K).
        # So: boxes + forearm strips; if that leaves < min_static, boxes only; if still too little, ViPE's mask.
        for forearm in ((True, False) if self.forearm else (False,)):
            mm = self._apply(m.clone(), boxes, H, W, forearm)
            if float(mm.float().mean()) >= self.min_static:
                frame.mask = mm
                if not forearm and self.forearm:
                    self.relaxed += 1
                return frame
        self.relaxed += 1
        frame.mask = m
        return frame

    def _apply(self, m, boxes, H, W, forearm):
        for b in boxes:
            x1, y1, x2, y2 = [int(round(v * self.scale)) for v in b]
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
            if x2 > x1 and y2 > y1:
                m[y1:y2, x1:x2] = False
                if forearm:                                  # the forearm runs from the hand toward the body
                    m[y1:H, x1:x2] = False
        return m


def run_vipe(video: str, work: str | Path, boxes: dict, fps_target: float = 15.0, width: int = 768,
             max_seconds: float | None = None, phrases=("person", "hand", "arm"), pipeline: str = "wide_angle",
             video_fps: float = 30.0, optimize_intrinsics: bool = True, robust: bool = True,
             intrinsics_damping: float = 100.0) -> Path:
    """Runs ViPE on a downscaled copy of the clip. Returns the ViPE output folder.

    robust: Huber kernel on the flow residuals (a few bad matches can't blow up bundle adjustment).
    intrinsics_damping: x ViPE's default damping on the intrinsics update (keeps focal / xi from running away).
    optimize_intrinsics=False keeps the GeoCalib intrinsics fixed (the safe retry when BA diverges)."""
    from vipe import make_pipeline
    from vipe.config import parse_typed_config
    from vipe.pipeline.default import DefaultAnnotationPipeline
    from vipe.streams.base import ProcessedVideoStream
    from vipe.streams.raw_mp4_stream import RawMp4Stream

    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    small = work / "clip_small.mp4"
    if not small.exists():
        _downscale_video(video, str(small), width, max_seconds)
    import cv2
    cap = cv2.VideoCapture(str(small))
    sw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    cap = cv2.VideoCapture(str(video))
    full_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    step = max(1, int(round(video_fps / fps_target)))
    out = work / "vipe"
    overrides = [f"pipeline={pipeline}", f"pipeline.output.path={out}", "pipeline.output.save_artifacts=true",
                 "pipeline.output.save_viz=false", "pipeline.output.save_slam_map=true",
                 "pipeline.init.instance.phrases=[" + ",".join(phrases) + "]",
                 f"pipeline.slam.optimize_intrinsics={str(optimize_intrinsics).lower()}"]
    ba = _ba_keys(overrides)                                   # only set keys this ViPE version has (strict schema)
    if robust and "robust_kernel" in ba:
        overrides += ["pipeline.slam.ba.robust_kernel=huber", "pipeline.slam.ba.robust_kernel_threshold=3.0"]
    if intrinsics_damping and intrinsics_damping != 1.0 and "intrinsics_damping_scale" in ba:
        overrides += [f"pipeline.slam.ba.intrinsics_damping_scale={intrinsics_damping}"]
    if (robust and "robust_kernel" not in ba) or (intrinsics_damping != 1.0 and "intrinsics_damping_scale" not in ba):
        log.info("this ViPE version has no robust kernel / intrinsics damping options; running without them")
    args = parse_typed_config("default", hydra_args=overrides)
    pipe = make_pipeline(args.pipeline)
    if boxes and isinstance(pipe, DefaultAnnotationPipeline):
        base_cls = type(pipe)
        proc = _BoxMaskProcessor(boxes, sw / full_w)

        class _Masked(base_cls):                               # add our hand masks after ViPE's own masks
            def _add_init_processors(self, video_stream):
                return ProcessedVideoStream(base_cls._add_init_processors(self, video_stream), [proc])

        pipe.__class__ = _Masked
    stream = ProcessedVideoStream(RawMp4Stream(small, seek_range=range(0, -1, step), name="clip"), [])
    log.info("ViPE: %s, every %d frame(s), width %d, pipeline %s, optimize_intrinsics=%s, robust=%s",
             video, step, sw, pipeline, optimize_intrinsics, robust)
    pipe.run(stream)
    if boxes and isinstance(pipe, DefaultAnnotationPipeline) and proc.relaxed:
        log.info("hand mask reduced on %d frames (too little static scene left)", proc.relaxed)
    (out / "step.txt").write_text(str(step))
    return out


def _ba_keys(overrides) -> set:
    try:
        from vipe.config.parse import parse_untyped_config
        cfg = parse_untyped_config("default", hydra_args=list(overrides))
        ba = cfg.pipeline.slam.get("ba")
        return set(ba.keys()) if ba is not None else set()
    except Exception:  # noqa: BLE001
        return set()


def read_vipe(out: str | Path, n_frames: int, full_size: tuple[int, int], name: str = "clip") -> dict:
    """Reads ViPE artifacts (pose = OpenCV cam->world 4x4 per processed frame; intrinsics = [fx,fy,cx,cy,xi])."""
    out = Path(out)
    step = int((out / "step.txt").read_text()) if (out / "step.txt").exists() else 1
    P = np.load(out / "pose" / f"{name}.npz")
    if not np.isfinite(P["data"]).all():
        raise VipeDiverged(f"ViPE poses contain NaN/inf ({int((~np.isfinite(P['data'])).any((1, 2)).sum())} frames)")
    inds = P["inds"] * step                                     # processed index -> original frame index
    R, t, valid = interpolate_poses(inds, P["data"], n_frames)
    I = np.load(out / "intrinsics" / f"{name}.npz")["data"]
    k = np.median(I, 0)
    if not np.isfinite(k).all() or k[0] <= 0 or k[1] <= 0:
        raise VipeDiverged(f"ViPE intrinsics invalid: {k}")
    small = _small_size(out, name)
    if len(k) >= 5:
        cam = MeiCamera(*k[:5], *small).scaled(*full_size)
    else:                                                       # pinhole preset: xi = 0
        cam = MeiCamera(*k[:4], 0.0, *small).scaled(*full_size)
    pts, cols = np.zeros((0, 3)), np.zeros((0, 3))
    mp = out / "vipe" / f"{name}_slam_map.pt"
    if mp.exists():
        import torch
        d = torch.load(mp, map_location="cpu", weights_only=False)
        pts = d["dense_disp_xyz"].numpy().astype(np.float32)
        cols = d["dense_disp_rgb"].numpy().astype(np.float32)
    return {"R_wc": R, "t_wc": t, "valid": valid, "cam": cam, "points": pts, "colors": cols, "backend": "vipe",
            "n_keyposes": int(len(inds)), "step": step}


def _small_size(out: Path, name: str) -> tuple[int, int]:
    import cv2
    for p in (out.parent / "clip_small.mp4", out / "rgb" / f"{name}.mp4"):
        if p.exists():
            cap = cv2.VideoCapture(str(p))
            wh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if wh[0] > 0:
                return wh
    raise FileNotFoundError("cannot find the processed video size for the ViPE intrinsics")


# ---- fallback: rotation only ----------------------------------------------------------------
def rotation_only(head_R: np.ndarray, ps4_cam: dict, n_frames: int) -> dict:
    """PS4 head rotation (camera_t -> camera_0) as camera->world rotation; head position fixed at the origin."""
    R = np.asarray(head_R)[:n_frames]
    if len(R) < n_frames:
        R = np.concatenate([R, np.repeat(R[-1:], n_frames - len(R), 0)])
    cam = EquidistantCamera.from_ps4(ps4_cam)
    return {"R_wc": R, "t_wc": np.zeros((n_frames, 3)), "valid": np.ones(n_frames, bool), "cam": cam,
            "points": np.zeros((0, 3)), "colors": np.zeros((0, 3)), "backend": "rotation", "n_keyposes": n_frames,
            "step": 1}


def have_vipe() -> bool:
    try:
        import vipe  # noqa: F401,F811
        return shutil.which("ffmpeg") is not None
    except Exception:  # noqa: BLE001
        return False
