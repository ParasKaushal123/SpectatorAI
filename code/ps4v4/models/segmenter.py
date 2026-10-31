"""Optional SAM 2 box-prompted segmenter, used by the tracker to follow partial hands."""
from __future__ import annotations

import cv2
import numpy as np


class Sam2BoxSegmenter:
    def __init__(self, model_id: str, device: str = "cuda"):
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        dev = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.predictor = SAM2ImagePredictor.from_pretrained(model_id, device=dev)
        self._frame_id = None

    def segment(self, frame_bgr: np.ndarray, frame_id: int, box: np.ndarray) -> tuple[np.ndarray | None, float]:
        """Mask box (x1,y1,x2,y2) of the object inside `box`, and SAM 2's quality score."""
        if self._frame_id != frame_id:
            self.predictor.set_image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            self._frame_id = frame_id
        masks, scores, _ = self.predictor.predict(box=np.asarray(box, float), multimask_output=False)
        m = masks[0] > 0
        if m.sum() < 50:
            return None, 0.0
        ys, xs = np.nonzero(m)
        return np.array([xs.min(), ys.min(), xs.max(), ys.max()], float), float(scores[0])
