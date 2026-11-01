"""Room twin geometry: keyframes -> virtual pinhole views -> metric depth (scaled to the SLAM map) -> masks ->
multi-clip registration -> TSDF fusion -> cleaned, coloured mesh in the target clip's room frame."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ps5world import backends
from ps5world.ps4_io import hand_boxes, load_hands, trim_hands

from .views import VirtualViews

log = logging.getLogger("room")


@dataclass
class ClipInput:
    name: str
    video: str          # the clip video (full resolution)
    ps4: str            # PS4 result folder (hands.jsonl, camera.json)
    slam: str           # run_slam output folder (vipe/, clip_small.mp4, world_frame.json)


DEFAULTS = dict(kf_fps=2.0, max_seconds=None, view_size=384, view_fov=90.0, blur_rel=0.45, max_depth=3.5,
                min_depth=0.20, edge_rel=0.05, body_radius=0.28, voxel=0.012, sdf_trunc=0.05, crop_radius=3.0,
                crop_z=(-2.0, 1.2), min_cluster=0.002, web_triangles=500_000, scale_min_points=30)


# ---- per clip: views, depth, poses --------------------------------------------------------------------------
def _frame_count(video: str) -> tuple[int, float, int, int]:
    cap = cv2.VideoCapture(video)
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return n, fps, w, h


def _keep_mask(shape, boxes) -> np.ndarray:
    """True = static scene. Hand boxes and the forearm strip below each box are removed (as for SLAM)."""
    H, W = shape
    m = np.ones((H, W), bool)
    for b in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
        if x2 > x1 and y2 > y1:
            m[y1:H, x1:x2] = False
    return m


def collect_clip(clip: ClipInput, depth_fn, cfg: dict) -> dict:
    """Keyframe views of one clip with metric depth and view->room poses (room frame of this clip)."""
    t0 = time.time()
    wf = json.load(open(Path(clip.slam) / "world_frame.json"))
    A, o = np.array(wf["align"]), np.array(wf["origin"])
    n_vid, fps, W, H = _frame_count(clip.video)
    hands = load_hands(clip.ps4)
    n = min(n_vid, hands["n_frames"])
    if cfg["max_seconds"]:
        n = min(n, int(round(cfg["max_seconds"] * fps)))
    hands = trim_hands(hands, n)
    traj = backends.read_vipe(Path(clip.slam) / "vipe", n, (W, H))
    cam = traj["cam"]
    views = VirtualViews(cam, cfg["view_size"], cfg["view_fov"])
    R_room = np.einsum("ij,njk->nik", A, traj["R_wc"])                 # camera -> room
    p_room = (traj["t_wc"] - o) @ A.T
    P = traj["points"]
    if len(P) > 300_000:
        P = P[np.random.default_rng(0).choice(len(P), 300_000, replace=False)]
    P_room = (P - o) @ A.T if len(P) else np.zeros((0, 3))
    boxes = hand_boxes(hands)
    step = max(1, int(round(fps / cfg["kf_fps"])))
    cap = cv2.VideoCapture(clip.video)
    rgb, keep, poses, sharp, frames = [], [], [], [], []
    for t in range(n):
        if t % step:
            if not cap.grab():
                break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        if not traj["valid"][t]:
            continue
        g = cv2.cvtColor(cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        sharp.append(cv2.Laplacian(g, cv2.CV_32F).var())
        km = _keep_mask((H, W), boxes.get(t, []))
        rgb.append([cv2.cvtColor(v, cv2.COLOR_BGR2RGB) for v in views.images(frame)])
        keep.append(views.masks(km))
        poses.append([np.r_[np.c_[R_room[t] @ Rv, p_room[t]], [[0, 0, 0, 1]]] for Rv in views.R])
        frames.append(t)
    cap.release()
    sharp = np.array(sharp)
    good = sharp >= cfg["blur_rel"] * np.percentile(sharp, 70) if len(sharp) else np.zeros(0, bool)
    log.info("%s: %d keyframes, %d sharp, %d views each", clip.name, len(frames), int(good.sum()), len(views))
    rgb = [r for r, g_ in zip(rgb, good) if g_]
    keep = [k for k, g_ in zip(keep, good) if g_]
    poses = [p for p, g_ in zip(poses, good) if g_]
    frames = [f for f, g_ in zip(frames, good) if g_]
    flat_rgb = [im for r in rgb for im in r]
    flat_pose = [p for ps in poses for p in ps]
    depth = depth_fn(flat_rgb, cfg["view_fov"], flat_pose, clip=clip.name)
    nv = len(views)
    # metric scale: depth model vs. the SLAM map, per keyframe (robust), clamped around the clip median
    ratios = []
    for i in range(len(frames)):
        rs = []
        for v in range(nv):
            rs += list(_map_ratios(P_room, flat_pose[i * nv + v], views.K, depth[i * nv + v], keep[i][v], cfg))
        ratios.append(np.median(rs) if len(rs) >= cfg["scale_min_points"] else np.nan)
    ratios = np.array(ratios, float)
    med = float(np.nanmedian(ratios)) if np.isfinite(ratios).any() else 1.0
    if not np.isfinite(ratios).any():
        log.warning("%s: no SLAM map points to scale the depth; using the depth model's own metric scale", clip.name)
    ratios = np.where(np.isfinite(ratios), np.clip(ratios, 0.8 * med, 1.25 * med), med)
    out = {"name": clip.name, "rgb": [], "depth": [], "pose": [], "K": views.K, "S": views.size,
           "frames": frames, "scale_median": med, "scale_spread": float(np.nanstd(ratios) / med) if med else 0.0,
           "head": p_room, "facing": None, "seconds": 0.0}
    for i in range(len(frames)):
        p_head = p_room[frames[i]]
        for v in range(nv):
            d = depth[i * nv + v] * ratios[i]
            d = _clean_depth(d, keep[i][v], flat_pose[i * nv + v], views.K, p_head, cfg)
            out["rgb"].append(flat_rgb[i * nv + v])
            out["depth"].append(d)
            out["pose"].append(flat_pose[i * nv + v])
    out["seconds"] = round(time.time() - t0, 1)
    log.info("%s: depth scale x%.3f (spread %.1f%%), %.0f s", clip.name, med, 100 * out["scale_spread"],
             out["seconds"])
    return out


def _map_ratios(P_room, T, K, d, keep, cfg):
    if not len(P_room):
        return np.zeros(0)
    X = (P_room - T[:3, 3]) @ T[:3, :3]                                # room -> view
    z = X[:, 2]
    ok = z > cfg["min_depth"]
    X, z = X[ok], z[ok]
    S = d.shape[0]
    u = np.round(K[0, 0] * X[:, 0] / z + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * X[:, 1] / z + K[1, 2]).astype(int)
    ok = (u >= 0) & (u < S) & (v >= 0) & (v < S)
    u, v, z = u[ok], v[ok], z[ok]
    dp = d[v, u]
    ok = (dp > 0) & (dp < cfg["max_depth"]) & keep[v, u]
    return z[ok] / dp[ok]


def _clean_depth(d, keep, T, K, p_head, cfg):
    d = d.astype(np.float32).copy()
    d[~np.isfinite(d)] = 0
    d[(d < cfg["min_depth"]) | (d > cfg["max_depth"]) | ~keep] = 0
    k = np.ones((3, 3), np.uint8)
    hi, lo = cv2.dilate(d, k), cv2.erode(np.where(d > 0, d, 1e3).astype(np.float32), k)
    d[(hi - lo) > cfg["edge_rel"] * np.maximum(d, 1e-3)] = 0           # flying pixels at depth edges
    # the wearer's own body / lap under the head (not part of the room)
    S = d.shape[0]
    u, v = np.meshgrid(np.arange(S), np.arange(S))
    ray = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(d)], -1)
    Pw = (ray * d[..., None]) @ T[:3, :3].T + T[:3, 3]
    dxy = np.hypot(Pw[..., 0] - p_head[0], Pw[..., 1] - p_head[1])
    body = (d > 0) & (dxy < cfg["body_radius"]) & (Pw[..., 2] < p_head[2] - 0.10) & (Pw[..., 2] > p_head[2] - 1.6)
    d[body] = 0
    return d


# ---- registration of a second clip into the target clip's room frame ------------------------------------------
def _cloud(c: dict, stride: int = 4):
    import open3d as o3d
    K = c["K"]
    pts, cols = [], []
    for im, d, T in zip(c["rgb"], c["depth"], c["pose"]):
        dd = d[::stride, ::stride]
        v, u = np.nonzero(dd > 0)
        z = dd[v, u]
        u, v = u * stride, v * stride
        X = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)
        pts.append(X @ T[:3, :3].T + T[:3, 3])
        cols.append(im[v, u] / 255.0)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(pts)))
    pc.colors = o3d.utility.Vector3dVector(np.concatenate(cols))
    return pc


def register(src: dict, dst: dict) -> dict:
    """Similarity transform G (src room -> dst room). Both rooms are gravity-aligned (Z up) and start at the same
    seat, so a yaw sweep + ICP is enough; scale is refined at the end. Returns G and quality numbers."""
    import open3d as o3d
    reg = o3d.pipelines.registration
    a = _cloud(src).voxel_down_sample(0.02)
    b = _cloud(dst).voxel_down_sample(0.02)
    for pc in (a, b):
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=30))
    best = None
    for yaw in range(0, 360, 30):
        c, s_ = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        G0 = np.eye(4)
        G0[:2, :2] = [[c, -s_], [s_, c]]
        r = reg.registration_icp(a, b, 0.15, G0, reg.TransformationEstimationPointToPlane(),
                                 reg.ICPConvergenceCriteria(max_iteration=40))
        if best is None or r.fitness > best.fitness + 1e-6 or (abs(r.fitness - best.fitness) < 1e-6 and
                                                               r.inlier_rmse < best.inlier_rmse):
            best = r
    G = best.transformation
    for thr in (0.06, 0.03):
        G = reg.registration_icp(a, b, thr, G, reg.TransformationEstimationPointToPlane(),
                                 reg.ICPConvergenceCriteria(max_iteration=60)).transformation
    rs = reg.registration_icp(a, b, 0.03, G, reg.TransformationEstimationPointToPoint(with_scaling=True),
                              reg.ICPConvergenceCriteria(max_iteration=60))
    Gs = rs.transformation
    scale = float(np.cbrt(np.linalg.det(Gs[:3, :3])))
    if 0.9 <= scale <= 1.1:
        G = Gs
    else:
        scale = 1.0
    ev = reg.evaluate_registration(a, b, 0.02, G)
    Rn = G[:3, :3] / np.cbrt(np.linalg.det(G[:3, :3]))
    tilt = float(np.degrees(np.arccos(np.clip(Rn[2, 2], -1, 1))))
    yaw = float(np.degrees(np.arctan2(Rn[1, 0], Rn[0, 0])))
    return {"G": G, "fitness": float(ev.fitness), "rmse_m": float(ev.inlier_rmse), "scale": scale,
            "tilt_deg": tilt, "yaw_deg": yaw, "shift_m": float(np.linalg.norm(G[:3, 3]))}


# ---- fusion and mesh -------------------------------------------------------------------------------------------
def fuse(clips: list[dict], transforms: list[np.ndarray], cfg: dict):
    """TSDF fusion (Open3D VoxelBlockGrid, CPU) of all views; returns a legacy coloured triangle mesh."""
    import open3d as o3d
    import open3d.core as o3c
    dev = o3c.Device("CPU:0")
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"), attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)), voxel_size=cfg["voxel"], block_resolution=8,
        block_count=cfg.get("block_count", 120_000), device=dev)
    mult = cfg["sdf_trunc"] / cfg["voxel"]
    for c, G in zip(clips, transforms):
        K = o3c.Tensor(c["K"], o3c.float64)
        for im, d, T in zip(c["rgb"], c["depth"], c["pose"]):
            if not (d > 0).any():
                continue
            dt = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(d, np.float32)))
            ct = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(im, np.float32) / 255.0))
            E = o3c.Tensor(np.linalg.inv(G @ T), o3c.float64)                  # room -> view
            fb = vbg.compute_unique_block_coordinates(dt, K, E, 1.0, cfg["max_depth"], trunc_voxel_multiplier=mult)
            vbg.integrate(fb, dt, ct, K, K, E, 1.0, cfg["max_depth"], trunc_voxel_multiplier=mult)
    mesh = vbg.extract_triangle_mesh(weight_threshold=cfg.get("weight_threshold", 2.0)).to_legacy()
    return mesh


def clean_mesh(mesh, cfg: dict):
    import open3d as o3d
    V = np.asarray(mesh.vertices)
    keep = (np.hypot(V[:, 0], V[:, 1]) <= cfg["crop_radius"]) & (V[:, 2] >= cfg["crop_z"][0]) & \
        (V[:, 2] <= cfg["crop_z"][1])
    mesh.remove_vertices_by_mask(~keep)
    cl, n_tri, _ = mesh.cluster_connected_triangles()
    cl, n_tri = np.asarray(cl), np.asarray(n_tri)
    if len(n_tri):
        small = n_tri[cl] < max(200, cfg["min_cluster"] * len(cl))
        mesh.remove_triangles_by_mask(small)
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()
    return mesh


def export_glb(mesh, path: str) -> None:
    import trimesh
    col = (np.clip(np.asarray(mesh.vertex_colors), 0, 1) * 255).astype(np.uint8)
    tm = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                         vertex_colors=np.c_[col, np.full(len(col), 255, np.uint8)], process=False)
    tm.export(path)


def build_room(clips: list[ClipInput], target: str, out_dir: str, depth_fn, cfg: dict | None = None) -> dict:
    """clips: same seat; target: the clip whose room frame (world_hands.jsonl) everything is put in."""
    import open3d as o3d
    cfg = {**DEFAULTS, **(cfg or {})}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    order = sorted(clips, key=lambda c: c.name != target)              # target first
    data = [collect_clip(c, depth_fn, cfg) for c in order]
    transforms, regs = [np.eye(4)], {}
    used = [data[0]]
    for c in data[1:]:
        r = register(c, data[0])
        ok = r["fitness"] >= 0.30 and r["rmse_m"] <= 0.02 and r["tilt_deg"] <= 5.0
        regs[c["name"]] = {k: v for k, v in r.items() if k != "G"} | {"used": bool(ok)}
        log.info("register %s -> %s: fitness %.2f, rmse %.1f mm, scale %.3f, tilt %.1f deg, yaw %.1f deg -> %s",
                 c["name"], data[0]["name"], r["fitness"], 1000 * r["rmse_m"], r["scale"], r["tilt_deg"],
                 r["yaw_deg"], "used" if ok else "NOT used")
        if ok:
            transforms.append(r["G"])
            used.append(c)
    mesh = clean_mesh(fuse(used, transforms, cfg), cfg)
    o3d.io.write_triangle_mesh(str(out / "room_mesh.ply"), mesh)
    web = mesh
    if len(mesh.triangles) > cfg["web_triangles"]:
        web = mesh.simplify_quadric_decimation(cfg["web_triangles"])
        web.compute_vertex_normals()
    export_glb(web, str(out / "room.glb"))
    V = np.asarray(mesh.vertices)
    stats = {"target": target, "clips": [c["name"] for c in data], "clips_used": [c["name"] for c in used],
             "registration": regs, "views": {c["name"]: len(c["depth"]) for c in data},
             "depth_scale": {c["name"]: round(c["scale_median"], 4) for c in data},
             "depth_scale_spread": {c["name"]: round(c["scale_spread"], 4) for c in data},
             "triangles": int(len(mesh.triangles)), "web_triangles": int(len(web.triangles)),
             "bbox_min": V.min(0).round(3).tolist() if len(V) else None,
             "bbox_max": V.max(0).round(3).tolist() if len(V) else None,
             "seconds": round(time.time() - t0, 1), "config": {k: v for k, v in cfg.items()}}
    json.dump(stats, open(out / "room_stats.json", "w"), indent=1, default=float)
    return stats
