"""Semantic check: "is this crop a hand, or a knee / foot / wood / head / tool ...?"

The hand detector is a single-class detector: anything skin-coloured and roughly hand-shaped can
score high, and nothing in it knows what a knee or a foot looks like. A zero-shot image-text model
(CLIP) does: it compares each crop against short descriptions of hands and of the things we saw
being mistaken for hands, and returns P(hand). Used as one more piece of evidence per tracklet.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import cv2
import numpy as np

log = logging.getLogger(__name__)

HAND_PROMPTS = ["a close-up photo of a human hand", "a hand holding a tool", "a hand wearing a work glove",
                "a hand holding a cloth", "a hand with a wristwatch"]
OTHER_PROMPTS = {"knee": ["a photo of a person's knee", "a knee in trousers seen from above"],
                 "foot": ["a photo of a bare foot", "a foot wearing a sandal", "a shoe on a floor"],
                 "leg": ["a person's leg"],
                 "head": ["the top of a person's head", "a human face"],
                 "wood": ["a piece of wood", "a wooden chair frame", "a wooden plank"],
                 "tool": ["a power tool", "a metal machine part"],
                 "surface": ["a stone slab surface", "a concrete floor", "a table top"],
                 "cloth": ["a cloth rag", "a plastic bag"]}


class ClipVerifier:
    def __init__(self, model_id: str, device: str = "cuda"):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.dev = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.model = CLIPModel.from_pretrained(model_id).to(self.dev).eval()
        self.proc = CLIPProcessor.from_pretrained(model_id)
        texts = HAND_PROMPTS + [p for v in OTHER_PROMPTS.values() for p in v]
        self.names = ["hand"] * len(HAND_PROMPTS) + [k for k, v in OTHER_PROMPTS.items() for _ in v]
        self.text = self.proc(text=texts, return_tensors="pt", padding=True).to(self.dev)

    def score(self, crops_bgr: list[np.ndarray], batch: int = 128) -> tuple[np.ndarray, list[str]]:
        """P(hand) per crop and the most likely non-hand class name."""
        torch = self.torch
        p_hand, other = [], []
        for i in range(0, len(crops_bgr), batch):
            ims = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops_bgr[i:i + batch]]
            with torch.no_grad():                      # full forward: stable API across transformers versions
                x = self.proc(images=ims, return_tensors="pt").to(self.dev)
                out = self.model(input_ids=self.text["input_ids"], attention_mask=self.text["attention_mask"],
                                 pixel_values=x["pixel_values"])
                prob = out.logits_per_image.softmax(-1).float().cpu().numpy()
            names = np.array(self.names)
            ph = prob[:, names == "hand"].sum(1)
            p_hand += ph.tolist()
            nh = np.where(names != "hand")[0]
            other += [str(names[nh[j]]) for j in prob[:, nh].argmax(1)]
        return np.array(p_hand), other


class FakeVerifier:
    """Tests: reads the blob colour (green/blue/yellow = hand, red = junk)."""

    def score(self, crops_bgr, batch: int = 128):
        from .fake import _masks
        out, other = [], []
        for c in crops_bgr:
            cnt = {k: int(m.sum()) for k, m in _masks(c).items()}
            kind = max(cnt, key=cnt.get) if max(cnt.values()) > 0 else 2
            out.append(0.1 if kind == 2 else 0.9)
            other.append("wood")
        return np.array(out), other


@lru_cache(maxsize=2)
def _clip_verifier(model_id, device):
    return ClipVerifier(model_id, device)


def make_verifier(cfg):
    pc = cfg.probe
    if pc.verifier == "clip":
        try:
            return _clip_verifier(pc.clip_model, cfg.device)
        except Exception as exc:                                    # noqa: BLE001 - optional evidence
            log.warning("CLIP verifier unavailable (%s); continuing without it", exc)
            return None
    if pc.verifier == "fake":
        return FakeVerifier()
    return None


def square_crop(img: np.ndarray, box: np.ndarray, pad: float, size: int = 224) -> np.ndarray:
    H, W = img.shape[:2]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    s = max(box[2] - box[0], box[3] - box[1]) * pad / 2
    x1, y1, x2, y2 = int(max(cx - s, 0)), int(max(cy - s, 0)), int(min(cx + s, W)), int(min(cy + s, H))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return np.zeros((size, size, 3), np.uint8)
    return cv2.resize(img[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_AREA)
