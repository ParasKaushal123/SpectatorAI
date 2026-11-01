"""Shared WiLoR-mini loader (detector + reconstruction model), loaded once per process.

Fine-tuned weights: if $SPECTATOR_WEIGHTS_DIR (default: <repo>/weights) holds a WiLoR checkpoint
(wilor*.ckpt / *.pth / *.pt other than detector.pt), it is loaded over the pretrained WiLoR weights;
a detector.pt there replaces the YOLO hand detector. Otherwise pretrained WiLoR is used.
`model_info()` says which one is running.
"""
from __future__ import annotations

import functools
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

_INFO: dict = {"variant": "pretrained", "label": "Pretrained WiLoR (baseline)", "wilor_ckpt": None, "detector": None}


def weights_dir() -> Path:
    return Path(os.environ.get("SPECTATOR_WEIGHTS_DIR", Path(__file__).resolve().parents[3] / "weights"))


def find_finetuned() -> tuple[Path | None, Path | None]:
    d = weights_dir()
    if not d.is_dir():
        return None, None
    det = d / "detector.pt"
    cands = sorted(p for p in d.iterdir() if p.suffix in (".ckpt", ".pth", ".pt") and p.name != "detector.pt")
    return (cands[0] if cands else None), (det if det.is_file() else None)


def model_info() -> dict:
    """Which model the pipeline uses (or will use) in this process."""
    ckpt, det = find_finetuned()
    if ckpt or det:
        return {"variant": "finetuned", "label": "Fine-tuned WiLoR (ours)",
                "wilor_ckpt": ckpt.name if ckpt else None, "detector": det.name if det else None}
    return dict(_INFO)


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
    options = {}
    model_cache = os.getenv("SPECTATOR_MODEL_CACHE")
    if model_cache:
        options["wilor_pretrained_dir"] = str(Path(model_cache) / "wilor")
    pipe = WiLorHandPose3dEstimationPipeline(device=dev, dtype=dtype, verbose=False, **options)
    _load_finetuned(pipe, dev, dtype)
    try:
        from ultralytics.utils import LOGGER
        LOGGER.setLevel(logging.ERROR)
    except Exception:  # noqa: BLE001
        pass
    pipe._ps4v4_device, pipe._ps4v4_dtype = dev, dtype
    return pipe


def _load_finetuned(pipe, dev, dtype) -> None:
    import torch

    ckpt, det = find_finetuned()
    if ckpt:
        sd = torch.load(str(ckpt), map_location="cpu")
        sd = sd.get("state_dict", sd.get("model", sd)) if isinstance(sd, dict) else sd
        sd = {k.removeprefix("module.").removeprefix("wilor_model."): v for k, v in sd.items()}
        missing, unexpected = pipe.wilor_model.load_state_dict(sd, strict=False)
        pipe.wilor_model.to(dev, dtype=dtype)
        log.info("fine-tuned WiLoR weights from %s (%d missing, %d unexpected keys)", ckpt.name, len(missing),
                 len(unexpected))
    if det:
        from ultralytics import YOLO
        pipe.hand_detector = YOLO(str(det))
        log.info("fine-tuned hand detector from %s", det.name)
    info = model_info()
    _INFO.update(info)
    log.info("MODEL: %s", info["label"])
