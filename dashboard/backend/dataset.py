"""Read-only video catalogue for the two local outputs folders."""
from hashlib import sha256
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from .clips import ROOT

router = APIRouter(prefix="/api/dataset")
OUTPUT_ROOT = ROOT.parent / "outputs"
FOLDERS = ("ps4v5_FINAL_renders", "ps4v5_P2_renders")
EXTENSIONS = {".mp4", ".webm", ".mov", ".m4v"}


def videos():
    for folder in FOLDERS:
        root = (OUTPUT_ROOT / folder).resolve()
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in EXTENSIONS or not path.is_file():
                continue
            if not path.resolve().is_relative_to(root):
                continue
            key = sha256((folder + "/" + path.relative_to(root).as_posix()).encode()).hexdigest()[:24]
            yield path, {"key": key, "id": path.stem, "folder": folder, "name": path.name,
                         "url": f"/api/dataset/videos/{key}"}


@router.get("")
def catalogue():
    groups = {}
    for _, video in videos():
        groups.setdefault(video["id"], {"id": video["id"], "videos": []})["videos"].append(video)
    return sorted(groups.values(), key=lambda row: row["id"].casefold())


@router.get("/videos/{key}")
def play(key: str):
    for path, video in videos():
        if video["key"] == key:
            return FileResponse(path, media_type="video/webm" if path.suffix.lower() == ".webm" else
                                "video/quicktime" if path.suffix.lower() == ".mov" else "video/mp4")
    raise HTTPException(404, "Dataset video not found")
