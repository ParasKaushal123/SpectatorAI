"""World exports: world_hands.jsonl, trajectory.csv, and a Rerun recording (scene.rrd) to scrub in 3D."""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from ps4v4 import HAND_CONNECTIONS

log = logging.getLogger(__name__)
SRC = {0: "none", 1: "observed", 2: "infilled", 3: "estimated"}


def _r(a, n=4):
    return np.round(np.asarray(a, float), n).tolist()


def export_jsonl(world: dict, path: str) -> None:
    """One line per frame. Frame: Z up, Y = worker's average facing, X right, metres, origin = first head position."""
    q = Rotation.from_matrix(world["head_R"]).as_quat()
    with open(path, "w") as fh:
        for t in range(world["N"]):
            row = {"frame": t, "t": round(t / world["fps"], 4), "pose_valid": bool(world["valid"][t]),
                   "head": {"position": _r(world["head_p"][t]), "quat_xyzw": _r(q[t], 5),
                            "gaze_dir": _r(world["gaze"][t])},
                   "body": {"neck": _r(world["neck"][t]), "pelvis": _r(world["pelvis"][t]),
                            "facing": _r(world["facing"][t])},
                   "hands": []}
            for s in ("left", "right"):
                src = int(world["src"][s][t])
                J = world["hands"][s][t]
                if src and np.isfinite(J).all():
                    row["hands"].append({"side": s, "source": SRC[src], "joints": _r(J)})
            fh.write(json.dumps(row) + "\n")


def export_mesh(world: dict, path: str) -> bool:
    """world_hands_mesh.npz (for the web viewer): faces, and per side the frames with a mesh, the world wrist
    (float32) and the 778 vertices relative to it (float16). Same room frame as world_hands.jsonl."""
    if world.get("faces") is None or not any(world["mesh"].get(s) is not None for s in ("left", "right")):
        return False
    out = {"faces": np.asarray(world["faces"], np.int32), "fps": np.float32(world["fps"])}
    for s in ("left", "right"):
        V = world["mesh"].get(s)
        if V is None:
            continue
        idx = np.nonzero(np.isfinite(V[:, :, 0]).all(1))[0]
        w = world["hands"][s][idx, 0].astype(np.float32)
        out[f"{s}_frames"] = idx.astype(np.int32)
        out[f"{s}_wrist"] = w
        out[f"{s}_rel"] = (V[idx] - w[:, None]).astype(np.float16)
    np.savez_compressed(path, **out)
    return True


def export_csv(world: dict, path: str) -> None:
    rows = []
    for t in range(world["N"]):
        r = {"frame": t, "t": t / world["fps"]}
        r.update({f"head_{k}": v for k, v in zip("xyz", world["head_p"][t])})
        r.update({f"gaze_{k}": v for k, v in zip("xyz", world["gaze"][t])})
        r.update({f"pelvis_{k}": v for k, v in zip("xyz", world["pelvis"][t])})
        r["facing_yaw_deg"] = float(np.degrees(np.arctan2(world["facing"][t, 0], world["facing"][t, 1])))
        for s in ("left", "right"):
            w = world["hands"][s][t, 0]
            ok = world["src"][s][t] > 0 and np.isfinite(w).all()
            r.update({f"{s}_wrist_{k}": (v if ok else np.nan) for k, v in zip("xyz", w)})
            r[f"{s}_source"] = SRC[int(world["src"][s][t])]
        rows.append(r)
    pd.DataFrame(rows).round(5).to_csv(path, index=False)


def _set_time(rr, t: int, fps: float):
    if hasattr(rr, "set_time"):
        rr.set_time("frame", sequence=t)
        rr.set_time("time", duration=t / fps)
    else:                                                               # older rerun-sdk
        rr.set_time_sequence("frame", t)
        rr.set_time_seconds("time", t / fps)


def export_rerun(world: dict, path: str, video: str | None = None, image_every: int = 2, image_width: int = 480,
                 max_points: int = 400_000) -> str | None:
    """Rerun recording: map, head (camera with frustum), gaze, body axis, hands, trails, and the video frame."""
    try:
        import rerun as rr
    except ImportError:
        log.warning("rerun-sdk not installed: skipping scene.rrd (pip install rerun-sdk)")
        return None
    rr.init("ps5_world", spawn=False)
    rr.save(path)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    P, C = world["points"], world["colors"]
    if len(P):
        if len(P) > max_points:
            k = np.random.default_rng(0).choice(len(P), max_points, replace=False)
            P, C = P[k], C[k]
        rr.log("world/map", rr.Points3D(P, colors=(np.clip(C, 0, 1) * 255).astype(np.uint8), radii=0.004),
               static=True)
    rr.log("world/head_path", rr.LineStrips3D([world["head_p"]], colors=[[120, 220, 255]], radii=0.003), static=True)
    cap = None
    W0 = H0 = 0
    if video:
        import cv2
        cap = cv2.VideoCapture(video)
        W0, H0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam = world["cam"]
    f_img = getattr(cam, "f_centre", 500.0)
    colors = {"left": [60, 200, 60], "right": [255, 140, 0]}
    for t in range(world["N"]):
        _set_time(rr, t, world["fps"])
        rr.log("world/head", rr.Transform3D(translation=world["head_p"][t], mat3x3=world["head_R"][t]))
        if t == 0 and W0:
            sc = image_width / W0
            rr.log("world/head/camera", rr.Pinhole(focal_length=f_img * sc, width=image_width,
                                                   height=int(H0 * sc), camera_xyz=rr.ViewCoordinates.RDF,
                                                   image_plane_distance=0.15), static=True)
        rr.log("world/gaze", rr.Arrows3D(origins=[world["head_p"][t]], vectors=[0.8 * world["gaze"][t]],
                                         colors=[[255, 230, 60]]))
        rr.log("world/body", rr.LineStrips3D([[world["head_p"][t], world["neck"][t], world["pelvis"][t]],
                                              list(world["shoulders"][t])], colors=[[200, 200, 200]], radii=0.01))
        for s in ("left", "right"):
            J = world["hands"][s][t]
            src = int(world["src"][s][t])
            if src and np.isfinite(J).all():
                col = colors[s] if src in (1, 2) else [int(0.45 * v) for v in colors[s]]
                rr.log(f"world/hands/{s}", rr.LineStrips3D([[J[a], J[b]] for a, b in HAND_CONNECTIONS],
                                                           colors=[col], radii=0.003))
                tr = world["hands"][s][max(0, t - int(world["fps"])):t + 1, 0]
                tr = tr[np.isfinite(tr).all(1)]
                if len(tr) > 1:
                    rr.log(f"world/trails/{s}", rr.LineStrips3D([tr], colors=[colors[s]], radii=0.002))
            else:
                rr.log(f"world/hands/{s}", rr.Clear(recursive=False))
        if cap is not None:
            ok, frame = cap.read()
            if ok and t % image_every == 0:
                import cv2
                sm = cv2.resize(frame, (image_width, int(H0 * image_width / W0)), interpolation=cv2.INTER_AREA)
                img = rr.Image(cv2.cvtColor(sm, cv2.COLOR_BGR2RGB))
                rr.log("world/head/camera", img.compress(jpeg_quality=70) if hasattr(img, "compress") else img)
    if cap is not None:
        cap.release()
    log.info("rerun recording: %s", path)
    return path
