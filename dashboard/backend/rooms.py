"""Read-only access to imported room twins. No processing or model loading."""
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from . import clips as C

router = APIRouter(prefix="/api/rooms")
ROOT = C.RESULTS / "room_twin"
WEB_FILES = {"room.glb", "motion.json", "hands_left.bin", "hands_right.bin",
             "robot.glb", "robot.json", "robot_motion.bin", "robot_traj.json"}


def directory(clip: str) -> Path:
    if not re.fullmatch(r"[\w-]+", clip) or not (ROOT / clip / "web/motion.json").is_file():
        raise HTTPException(404, "Room not found; expected dashboard/results/room_twin/<clip>/web/motion.json")
    path = (ROOT / clip).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise HTTPException(404, "Room not found")
    return path


def original(clip: str) -> Path | None:
    # Deliberately do not substitute a pre-rendered output for the original video.
    for suffix in (".mp4", ".MP4", ".mov", ".webm", ".m4v"):
        path = C.RAW / (clip + suffix)
        if path.is_file() and path.resolve().is_relative_to(C.RAW.resolve()):
            return path
    return None


def serve(path: Path, media_type=None, download=False):
    if not path.is_file():
        raise HTTPException(404, f"Missing {path.name}; expected {path}")
    return FileResponse(path, media_type=media_type, filename=path.name if download else None)


@router.get("")
def rooms():
    if not ROOT.is_dir():
        return []
    return [{"id": p.name, "has_video": original(p.name) is not None,
             "has_render": (p / "room_video.mp4").is_file(), "has_robot": (p / "web/robot.json").is_file()}
            for p in sorted(ROOT.iterdir()) if p.is_dir() and (p / "web/motion.json").is_file()
            and p.resolve().is_relative_to(ROOT.resolve())]


@router.get("/{clip}/web/{name}")
def web(clip: str, name: str):
    if name not in WEB_FILES:
        raise HTTPException(404, "Unknown room asset")
    return serve(directory(clip) / "web" / name,
                 "model/gltf-binary" if name.endswith(".glb") else
                 "application/json" if name.endswith(".json") else "application/octet-stream")


@router.get("/{clip}/video")
def video(clip: str):
    directory(clip)
    path = original(clip)
    if path is None:
        raise HTTPException(404, f"Missing original video; expected dashboard/raw_clips/{clip}.mp4")
    return serve(path, "video/quicktime" if path.suffix.lower() == ".mov" else
                 "video/webm" if path.suffix.lower() == ".webm" else "video/mp4")


@router.get("/{clip}/render")
def render(clip: str):
    return serve(directory(clip) / "room_video.mp4", "video/mp4", download=True)


@router.get("/{clip}/slam")
def slam(clip: str):
    return serve(directory(clip) / "room_video.mp4", "video/mp4")


@router.get("/{clip}/stats")
def stats(clip: str):
    return serve(directory(clip) / "room_stats.json", "application/json")
