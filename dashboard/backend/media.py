"""Shared dashboard transcode: bounded dimensions, frame rate, duration, and no audio."""
import subprocess
from pathlib import Path
import cv2
from .clips import FFMPEG


def transcode(src: Path, dst: Path, seconds=60):
    cap = cv2.VideoCapture(str(src))
    fps = min(cap.get(cv2.CAP_PROP_FPS) or 30, 30)
    cap.release()
    # 1280x720 bounding box also handles portrait inputs without upscaling.
    vf = f"scale=w='min(1280,iw)':h='min(720,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2,fps={fps}"
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(src), "-t", str(seconds),
                    "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)],
                   check=True, capture_output=True, timeout=180)
    cap = cv2.VideoCapture(str(dst))
    meta = {"width": int(cap.get(3)), "height": int(cap.get(4)), "fps": cap.get(5),
            "frames": int(cap.get(7))}
    cap.release()
    if not meta["frames"] or not meta["fps"]:
        raise ValueError("No readable video frames")
    meta["duration_s"] = meta["frames"] / meta["fps"]
    return meta
