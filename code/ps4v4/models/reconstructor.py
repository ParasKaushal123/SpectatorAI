"""Per-frame 3D hand reconstruction backends (M4).

Interface: reconstruct(meta, cam, requests) where requests = {side: (frames (n,), boxes (n,4))}.
Returns {side: {"frames", "joints" (n,21,3 camera metres), "kp2d" (n,21,2 px), "valid" (n,),
               "verts" (n,V,3) or absent}} plus an optional top-level "faces".
"""
from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from ..geometry import FisheyeCamera, make_virtual_view, render_virtual, solve_translation
from ..video import VideoMeta, iter_batches

log = logging.getLogger(__name__)

# EgoHand SOMA-25 (wrist, thumb 4, four fingers x 5 incl. metacarpal base) -> 21-joint layout
SOMA25_TO_21 = [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24]


def make_reconstructor(cfg, cache_dir: Path):
    backend = cfg.recon.backend
    if backend == "wilor":
        return WiLoRReconstructor(cfg)
    if backend == "egohand":
        return EgoHandReconstructor(cfg, cache_dir)
    if backend == "fake":
        from .fake import FakeReconstructor
        return FakeReconstructor()
    raise ValueError(f"unknown recon backend {backend}")


def square_box(box: np.ndarray, pad: float) -> np.ndarray:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    s = max(box[2] - box[0], box[3] - box[1]) * pad / 2
    return np.array([cx - s, cy - s, cx + s, cy + s])


# =============================================================================
class WiLoRReconstructor:
    """WiLoR MANO regression on perspective-correct virtual crops (fisheye-aware)."""

    RESCALE = 2.5     # WiLoR's crop = 2.5 x box

    def __init__(self, cfg):
        from .wilor_loader import load_wilor
        self.cfg = cfg
        self.pipe = load_wilor(cfg.device)

    @property
    def faces(self):
        return np.asarray(self.pipe.wilor_model.mano.faces, np.int32)

    def reconstruct(self, meta: VideoMeta, cam: FisheyeCamera, requests: dict) -> dict:
        wc = self.cfg.recon.wilor
        jobs: dict[int, list] = {}
        for side, (frames, boxes) in requests.items():
            for i, (f, b) in enumerate(zip(frames, boxes)):
                jobs.setdefault(int(f), []).append((side, i, square_box(b, self.cfg.recon.box_pad)))
        out = {s: _empty(len(fr), with_verts=True) | {"frames": np.asarray(fr)} for s, (fr, _) in requests.items()}
        pending, meta_list = [], []

        def flush():
            if not pending:
                return
            o = self._run_model(pending)
            for k, (side, i, view, is_right) in enumerate(meta_list):
                res = self._postprocess(o, k, view, is_right, cam)
                if res is not None:
                    J, V = res
                    out[side]["joints"][i], out[side]["verts"][i] = J, V
                    out[side]["kp2d"][i] = cam.project(J)
                    out[side]["valid"][i] = True
            pending.clear(); meta_list.clear()

        for f0, frames in iter_batches(meta, 16):
            for k, img in enumerate(frames):
                for side, i, box in jobs.get(f0 + k, []):
                    is_right = side == "right"
                    if wc.virtual_crop:
                        view = make_virtual_view(cam, box, wc.virtual_size, wc.virtual_fov_scale)
                        vimg = render_virtual(img, cam, view)
                        s = wc.virtual_size / wc.virtual_fov_scale
                        c = wc.virtual_size / 2
                        patch = self._patch(vimg, np.array([c - s / 2, c - s / 2, c + s / 2, c + s / 2]), is_right)
                    else:
                        view = None
                        patch = self._patch(img, box, is_right)
                    pending.append(patch)
                    meta_list.append((side, i, view, is_right))
            if len(pending) >= wc.batch:
                flush()
        flush()
        out["faces"] = self.faces
        return out

    # ---- internals ------------------------------------------------------
    def _patch(self, img, box, is_right):
        from wilor_mini.utils import utils as wutils
        H, W = img.shape[:2]
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        size = self.RESCALE * max(box[2] - box[0], box[3] - box[1])
        half = size / 2 + 2
        xa, ya = max(int(math.floor(cx - half)), 0), max(int(math.floor(cy - half)), 0)
        xb, yb = min(int(math.ceil(cx + half)), W), min(int(math.ceil(cy + half)), H)
        crop, lx, ly = (img[ya:yb, xa:xb], cx - xa, cy - ya) if (xb - xa > 2 and yb - ya > 2) else \
            (np.zeros((4, 4, 3), np.uint8), 2.0, 2.0)
        ds = size / 256 / 2
        if ds > 1.1:
            crop = cv2.GaussianBlur(crop, (0, 0), (ds - 1) / 2)            # anti-alias, as WiLoR does
        patch, _ = wutils.generate_image_patch_cv2(crop, lx, ly, size, size, 256, 256, not is_right, 1.0, 0,
                                                   border_mode=cv2.BORDER_CONSTANT)
        return patch.astype(np.float32), (cx, cy, size, W, H)

    def _run_model(self, items):
        import torch
        dev, dt = self.pipe._ps4v4_device, self.pipe._ps4v4_dtype
        with torch.no_grad():
            x = torch.from_numpy(np.stack([p for p, _ in items])).to(dev, dtype=dt)   # BGR patches (WiLoR flips)
            o = self.pipe.wilor_model(x)
        o = {k: v.float().cpu().numpy() for k, v in o.items()}
        o["_geom"] = [g for _, g in items]
        return o

    def _postprocess(self, o, k, view, is_right, cam: FisheyeCamera):
        from wilor_mini.utils import utils as wutils
        cx, cy, size, W, H = o["_geom"][k]
        kp3d, verts = o["pred_keypoints_3d"][k].copy(), o["pred_vertices"][k].copy()
        pcam = o["pred_cam"][k:k + 1].copy()
        pcam[:, 1] *= (1 if is_right else -1)
        if not is_right:                                    # left hand = mirrored right MANO
            kp3d[:, 0] *= -1; verts[:, 0] *= -1
        f_s = 5000 / 256 * max(W, H)
        t_full = wutils.cam_crop_to_full(pcam, np.array([[cx, cy]]), size, np.array([[W, H]]), f_s)
        kp2d = wutils.perspective_projection(kp3d[None], t_full, np.array([[f_s, f_s]]),
                                             np.array([[W / 2, H / 2]]))[0]
        if not np.isfinite(kp2d).all():
            return None
        if view is not None:                                # virtual pinhole camera
            uvn = (kp2d - view.c) / view.f
            t = solve_translation(kp3d, uvn)
            Jv, Vv = kp3d + t, verts + t
            return Jv @ view.R, Vv @ view.R                 # rotate back to the real camera
        t = solve_translation(kp3d, cam.normalized(kp2d))
        return kp3d + t, verts + t


# =============================================================================
class EgoHandReconstructor:
    """NVIDIA EgoHand (Oct 2026) run in its own environment on our tracked boxes,
    with its built-in perspective cropping for distorted (KB4) cameras."""

    def __init__(self, cfg, cache_dir: Path):
        self.cfg = cfg
        self.work = Path(cache_dir) / "egohand"
        self.faces = None

    def reconstruct(self, meta: VideoMeta, cam: FisheyeCamera, requests: dict) -> dict:
        ec = self.cfg.recon.egohand
        if self.work.exists():
            shutil.rmtree(self.work)
        stem = "clip"
        vid_dir, box_dir, out_dir = self.work / "video", self.work / "bboxes", self.work / "out"
        for d in (vid_dir, box_dir / stem, out_dir):
            d.mkdir(parents=True, exist_ok=True)
        video = vid_dir / f"{stem}.mp4"
        os.symlink(os.path.abspath(meta.path), video)
        cam_json = self.work / "camera.json"
        cam.to_egohand_json(str(cam_json))
        for side, (frames, boxes) in requests.items():
            d = {f"{int(f):06d}": [float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2),
                                   float(max(b[2] - b[0], b[3] - b[1]) * self.cfg.recon.box_pad)]
                 for f, b in zip(frames, boxes)}
            with open(box_dir / stem / f"{stem}_{side}_bbox.json", "w") as fh:
                json.dump(d, fh)
        cmd = [ec.python, "inference/scripts/run_egohand_inference_pipeline.py", "--input", str(video),
               "--bbox_detector_config", "inference/hand_detector_2d/configs/saved_bboxes.yaml",
               "--output_dir", str(out_dir), "--intrinsics_json", str(cam_json), "--perspective_cropping",
               *list(ec.extra_args)]
        if self.cfg.max_seconds:
            cmd += ["--max_seconds", str(self.cfg.max_seconds)]
        env = dict(os.environ, BBOX_DIR=str(box_dir), PYTHONPATH=ec.repo_dir, PYOPENGL_PLATFORM="egl")
        log.info("running EgoHand: %s", " ".join(cmd))
        subprocess.run(cmd, cwd=ec.repo_dir, env=env, check=True)

        out = {}
        for side, (frames, _) in requests.items():
            res = _empty(len(frames), with_verts=False) | {"frames": np.asarray(frames)}
            p = out_dir / stem / f"{side}_results.npz"
            if p.exists():
                z = np.load(p)
                if int(z["num_frames"]) > 0:
                    idx = {int(f): i for i, f in enumerate(frames)}
                    J = z["joints"][:, SOMA25_TO_21] + z["cam_t"][:, None, :]
                    for row, name in enumerate(z["frame_names"]):
                        i = idx.get(int(name))
                        if i is not None and np.isfinite(J[row]).all():
                            res["joints"][i] = J[row]
                            res["kp2d"][i] = cam.project(J[row])
                            res["valid"][i] = True
            out[side] = res
        out["faces"] = None
        return out


def _empty(n: int, with_verts: bool) -> dict:
    d = {"joints": np.full((n, 21, 3), np.nan), "kp2d": np.full((n, 21, 2), np.nan), "valid": np.zeros(n, bool)}
    if with_verts:
        d["verts"] = np.full((n, 778, 3), np.nan, np.float32)
    return d
