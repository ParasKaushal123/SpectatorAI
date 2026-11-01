"""Web bundle for the room twin: room.glb + the worker's motion (head, gaze, hand meshes) + a three.js viewer.
Open with:  python -m http.server 8000  (inside the folder), then http://localhost:8000/viewer.html"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .render_video import camera_path, load_motion

Q = 5e-5                                    # hand vertex offsets from the wrist: int16 steps of 0.05 mm (+-1.6 m)


def export_web(room_dir: str, slam_dir: str, out_dir: str, seconds: float = 120.0) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(room_dir) / "room.glb", out / "room.glb")
    m = load_motion(slam_dir)
    n = min(m["N"], int(round(seconds * m["fps"])))
    eye, tgt = camera_path(m, n)
    r4 = lambda a: np.round(np.nan_to_num(np.asarray(a, float), nan=0.0), 4).tolist()      # noqa: E731
    meta = {"fps": round(float(m["fps"]), 4), "n": n, "quant": Q,
            "head": r4(m["head"][:n]), "quat": r4(m["quat"][:n]), "gaze": r4(m["gaze"][:n]),
            "focus": r4(tgt[0]), "eye0": r4(eye[0]), "faces": None, "hands": {}}
    if m["faces"] is not None:
        meta["faces"] = m["faces"].reshape(-1).astype(int).tolist()
    for s in ("left", "right"):
        J = m["joints"][s][:n]
        src = m["src"][s][:n]
        h = {"src": src.astype(int).tolist(), "joints": r4(J.reshape(n, -1)), "mesh": False}
        V = m["verts"][s]
        if V is not None and m["faces"] is not None:
            V = V[:n]
            has = np.isfinite(V).all((1, 2)) & np.isin(src, (1, 2))
            w = np.where(has[:, None], V[:, 0], 0.0)                       # anchor = first vertex
            rel = np.where(has[:, None, None], V - w[:, None], 0.0)
            q = np.clip(np.round(rel / Q), -32767, 32767).astype("<i2")
            q.tofile(out / f"hands_{s}.bin")
            h.update(mesh=True, has_mesh=has.astype(int).tolist(), anchor=r4(w), verts=int(V.shape[1]))
        meta["hands"][s] = h
    json.dump(meta, open(out / "motion.json", "w"), separators=(",", ":"))
    shutil.copy(Path(__file__).with_name("viewer.html"), out / "viewer.html")
    shutil.copytree(Path(__file__).with_name("web_lib"), out / "lib", dirs_exist_ok=True)     # three.js r160, offline
    (out / "open_viewer.bat").write_text("@echo off\r\ncd /d %~dp0\r\nstart \"\" http://localhost:8000/viewer.html\r\n"
                                         "python -m http.server 8000\r\n")
    return {"frames": n, "files": sorted(p.name for p in out.iterdir()),
            "mb": round(sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6, 1)}
