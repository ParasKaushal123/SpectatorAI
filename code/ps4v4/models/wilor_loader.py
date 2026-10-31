"""Shared WiLoR-mini loader (detector + reconstruction model), loaded once per process."""
from __future__ import annotations

import functools
import logging


@functools.lru_cache(maxsize=2)
def load_wilor(device: str = "cuda"):
    import torch

    # old WiLoR / YOLO checkpoints need weights_only=False on PyTorch >= 2.6
    if not getattr(torch.load, "_ps4v4_patched", False):
        orig = torch.load

        def patched(*a, **k):
            k.setdefault("weights_only", False)
            return orig(*a, **k)

        patched._ps4v4_patched = True
        torch.load = patched
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    pipe = WiLorHandPose3dEstimationPipeline(device=dev, dtype=dtype, verbose=False)
    try:
        from ultralytics.utils import LOGGER
        LOGGER.setLevel(logging.ERROR)
    except Exception:  # noqa: BLE001
        pass
    pipe._ps4v4_device, pipe._ps4v4_dtype = dev, dtype
    return pipe
