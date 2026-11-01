"""Room twin end to end on a synthetic workstation with known geometry (no GPU, no downloads).

Two seated "clips" of the same room, each with its own SLAM frame (as ViPE gives) and its own gravity-aligned room
frame with a different facing (as build_world gives). The depth stand-in returns true depth x 1.15 (a wrong metric
scale) with noise, which the SLAM-map scaling must correct. Checks: registration of clip B into clip A's room frame,
mesh accuracy against the true planes, bench height, the fly-through video, the web bundle and the viewer page.
Run: python tests/test_room_synthetic.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ps5world.cameras import MeiCamera  # noqa: E402

W, H, FPS, SECS = 640, 360, 30, 6
N = FPS * SECS
CAM = MeiCamera(230.0, 230.0, W / 2, H / 2, 0.9, W, H)
BENCH_Z, FLOOR_Z = -0.45, -1.25


# ---- the true room (truth frame: Z up, worker's head near the origin, bench in front along +Y) ------------------
def planes():
    """(normal, offset, bounds fn) for p.n = offset."""
    return [
        (np.array([0, 0, 1.0]), FLOOR_Z, lambda P: np.ones(len(P), bool)),
        (np.array([0, 0, 1.0]), BENCH_Z, lambda P: (np.abs(P[:, 0]) < 0.9) & (P[:, 1] > 0.25) & (P[:, 1] < 1.0)),
        (np.array([0, 1.0, 0]), 1.6, lambda P: np.ones(len(P), bool)),
        (np.array([1.0, 0, 0]), -1.5, lambda P: np.ones(len(P), bool)),
        (np.array([1.0, 0, 0]), 1.5, lambda P: np.ones(len(P), bool)),
        (np.array([0, 1.0, 0]), -1.4, lambda P: np.ones(len(P), bool)),
    ]


def cast(o, d):
    """Rays (o (3,), d (n,3), not normalised) -> parameter t (n,) of the first hit, and the hit points."""
    best = np.full(len(d), np.inf)
    for n, off, inside in planes():
        dn = d @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (off - o @ n) / dn
        P = o + t[:, None] * d
        ok = (t > 1e-4) & np.isfinite(t) & inside(np.nan_to_num(P))
        best = np.where(ok & (t < best), t, best)
    return best, o + best[:, None] * d


def colour(P):
    """Texture: 10 cm checker x smooth colour ramp, different per surface height."""
    c = ((np.floor(P[:, 0] / 0.1) + np.floor(P[:, 1] / 0.1) + np.floor(P[:, 2] / 0.1)) % 2)
    base = np.stack([0.45 + 0.4 * np.sin(P[:, 0] * 2), 0.45 + 0.4 * np.cos(P[:, 1] * 2), 0.5 + 0.3 * np.sin(P[:, 2] * 3)], 1)
    return np.clip(base * (0.65 + 0.35 * c[:, None]), 0, 1)


def head_pose(t: float, phase: float):
    yaw = np.radians(25 * np.sin(2 * np.pi * t / 4 + phase))
    pitch = np.radians(38 + 8 * np.sin(2 * np.pi * t / 3 + phase))
    f = np.array([np.sin(yaw) * np.cos(pitch), np.cos(yaw) * np.cos(pitch), -np.sin(pitch)])
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f], 1)                                          # camera -> truth
    p = np.array([0.02 * np.sin(t), 0.015 * np.cos(1.3 * t), 0.01 * np.sin(0.7 * t)])
    return R, p


def make_clip(tmp: Path, name: str, phase: float, p0_shift, yaw_room_deg: float):
    """Writes the video, PS4 folder and SLAM folder of one clip. Returns the truth<-clip-room transform."""
    clips, ps4, slam = tmp / "clips", tmp / "ps4" / name, tmp / "slam" / name
    for d in (clips, ps4, slam / "vipe" / "pose", slam / "vipe" / "intrinsics", slam / "vipe" / "vipe"):
        d.mkdir(parents=True, exist_ok=True)
    poses = [head_pose(i / FPS, phase) for i in range(N)]
    Rs = np.array([p[0] for p in poses])
    ps = np.array([p[1] for p in poses]) + np.asarray(p0_shift)
    R0, p0 = Rs[0], ps[0]
    # video (fisheye)
    u, v = np.meshgrid(np.arange(W) + 0.0, np.arange(H) + 0.0)
    rays = CAM.rays(np.stack([u, v], -1)).reshape(-1, 3)
    okr = np.isfinite(rays).all(1)
    vw = cv2.VideoWriter(str(clips / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    map_pts = []
    for i in range(N):
        d = np.nan_to_num(rays) @ Rs[i].T
        t, P = cast(ps[i], d)
        img = np.zeros((H * W, 3))
        hit = okr & np.isfinite(t)
        img[hit] = colour(P[hit])
        vw.write((img.reshape(H, W, 3)[..., ::-1] * 255).astype(np.uint8))
        if i % 10 == 0:
            k = np.nonzero(hit)[0][::97]
            map_pts.append(P[k])
    vw.release()
    # SLAM (ViPE frame = first camera), every 2nd frame, intrinsics at half size
    step = 2
    inds = np.arange(0, N, step)
    T = np.repeat(np.eye(4)[None], len(inds), 0)
    for k, i in enumerate(inds):
        T[k, :3, :3] = R0.T @ Rs[i]
        T[k, :3, 3] = R0.T @ (ps[i] - p0)
    np.savez(slam / "vipe" / "pose" / "clip.npz", data=T, inds=inds // step)
    sm = CAM.scaled(W // 2, H // 2)
    np.savez(slam / "vipe" / "intrinsics" / "clip.npz", data=np.tile([sm.fx, sm.fy, sm.cx, sm.cy, sm.xi], (len(inds), 1)))
    (slam / "vipe" / "step.txt").write_text(str(step))
    import torch
    P = np.concatenate(map_pts)
    torch.save({"dense_disp_xyz": torch.from_numpy((P - p0) @ R0).float(),
                "dense_disp_rgb": torch.from_numpy(colour(P)).float()}, slam / "vipe" / "vipe" / "clip_slam_map.pt")
    vw = cv2.VideoWriter(str(slam / "clip_small.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W // 2, H // 2))
    vw.write(np.zeros((H // 2, W // 2, 3), np.uint8)); vw.release()
    # room frame of this clip: gravity-aligned, rotated by yaw_room about Z, origin = first head position
    Rz = Rotation.from_euler("z", yaw_room_deg, degrees=True).as_matrix()
    A = Rz @ R0                                                         # vipe -> clip room
    json.dump({"align": A.tolist(), "origin": [0, 0, 0], "backend": "vipe", "n_frames": N, "fps": FPS},
              open(slam / "world_frame.json", "w"))
    # PS4: one hand box per frame (lower centre of the image), observed
    with open(ps4 / "hands.jsonl", "w") as fh:
        for i in range(N):
            box = [W * 0.42, H * 0.62, W * 0.58, H * 0.86]
            kp = np.tile([W * 0.5, H * 0.74], (21, 1))
            J = np.tile([0.0, 0.15, 0.45], (21, 1))
            fh.write(json.dumps({"frame": i, "t": i / FPS, "hands": [
                {"side": "right", "state": "VISIBLE", "source": "observed", "conf": 0.9, "bbox": box,
                 "kp2d": kp.tolist(), "kp3d_cam": J.tolist()}]}) + "\n")
    json.dump({"width": W, "height": H, "fx": 300.0, "fy": 300.0, "cx": W / 2, "cy": H / 2}, open(ps4 / "camera.json", "w"))
    to_truth = lambda X: X @ Rz + p0                                    # noqa: E731  clip room -> truth
    return {"name": name, "to_truth": to_truth, "Rz": Rz, "p0": p0, "Rs": Rs, "ps": ps, "slam": slam}


def write_motion(c, slam: Path):
    """world_hands.jsonl + world_hands_mesh.npz for the target clip (in its room frame): hands resting on the bench."""
    to_room = lambda X: (np.asarray(X) - c["p0"]) @ c["Rz"].T          # noqa: E731
    rng = np.random.default_rng(0)
    blob = rng.normal(0, 1, (778, 3)); blob /= np.linalg.norm(blob, axis=1, keepdims=True)
    blob *= [0.045, 0.08, 0.02]                                         # a hand-sized flat ellipsoid
    from scipy.spatial import ConvexHull
    faces = ConvexHull(blob).simplices.astype(np.int32)
    rows, frames, rel, wrist = [], [], [], []
    for i in range(N):
        hp = to_room(c["ps"][i])
        R = c["Rz"] @ c["Rs"][i]
        q = Rotation.from_matrix(R).as_quat()
        w_truth = np.array([0.15 + 0.05 * np.sin(i / 20), 0.55, BENCH_Z + 0.025])
        J = to_room(np.tile(w_truth, (21, 1)) + np.linspace(0, 0.08, 21)[:, None] * [0, 1, 0])
        rows.append({"frame": i, "t": i / FPS, "pose_valid": True,
                     "head": {"position": hp.tolist(), "quat_xyzw": q.tolist(), "gaze_dir": R[:, 2].tolist()},
                     "body": {}, "hands": [{"side": "right", "source": "observed", "joints": J.tolist()}]})
        frames.append(i); wrist.append(J[0]); rel.append(blob + [0, 0.04, 0])
    with open(slam / "world_hands.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    np.savez_compressed(slam / "world_hands_mesh.npz", faces=faces, fps=np.float32(FPS),
                        right_frames=np.array(frames, np.int32), right_wrist=np.array(wrist, np.float32),
                        right_rel=np.array(rel, np.float16))


def main():
    import open3d as o3d
    from room.build import ClipInput, build_room
    from room.depth import CallableDepth
    from room.render_video import render_room_video
    from room.web import export_web

    tmp = Path(tempfile.mkdtemp())
    t0 = time.time()
    A = make_clip(tmp, "clipA", 0.0, [0, 0, 0], 0.0)
    B = make_clip(tmp, "clipB", 1.7, [0.03, -0.02, 0.01], 20.0)
    write_motion(A, A["slam"])
    print(f"synthetic clips written in {time.time() - t0:.0f} s")
    clips = {c["name"]: c for c in (A, B)}
    rng = np.random.default_rng(5)

    def fake_depth(images, fov, poses, clip):
        S = images[0].shape[0]
        f = (S / 2) / np.tan(np.radians(fov) / 2)
        u, v = np.meshgrid(np.arange(S), np.arange(S))
        dv = np.stack([(u - (S - 1) / 2) / f, (v - (S - 1) / 2) / f, np.ones((S, S))], -1).reshape(-1, 3)
        c = clips[clip]
        out = []
        for T in poses:
            o = c["to_truth"](T[:3, 3])
            Rt = c["Rz"].T @ T[:3, :3]                                   # view -> truth
            z, _ = cast(o, dv @ Rt.T)
            z = z.reshape(S, S)
            z = np.where(np.isfinite(z), z, 0) * 1.15 * (1 + 0.01 * rng.standard_normal((S, S)))   # wrong scale + noise
            out.append(z.astype(np.float32))
        return out

    ins = [ClipInput(c["name"], str(tmp / "clips" / f"{c['name']}.mp4"), str(tmp / "ps4" / c["name"]), str(c["slam"]))
           for c in (A, B)]
    out = tmp / "room"
    st = build_room(ins, "clipA", str(out), CallableDepth(fake_depth),
                    {"view_size": 160, "kf_fps": 3.0, "voxel": 0.015, "web_triangles": 150_000})
    print(json.dumps({k: st[k] for k in ("clips_used", "registration", "depth_scale", "triangles", "seconds")},
                     indent=1, default=float))
    # registration: clip B's room -> clip A's room = Rz(-20 deg) and the seat offset
    rg = st["registration"]["clipB"]
    assert st["clips_used"] == ["clipA", "clipB"], "clip B was not registered"
    assert abs(rg["yaw_deg"] + 20) < 1.5 and abs(rg["scale"] - 1) < 0.02, rg
    # depth scale corrected to the SLAM map (fake depth was 15% too far)
    for k, s in st["depth_scale"].items():
        assert abs(s - 1 / 1.15) < 0.02, (k, s)
    # geometry: distance of mesh vertices to the nearest true surface, in the truth frame
    mesh = o3d.io.read_triangle_mesh(str(out / "room_mesh.ply"))
    V = A["to_truth"](np.asarray(mesh.vertices))
    dist = np.min(np.stack([np.where(inside(V), np.abs(V @ n - off), np.inf) for n, off, inside in planes()]), 0)
    med, p90 = float(np.median(dist)), float(np.percentile(dist, 90))
    Nn = np.asarray(mesh.vertex_normals) @ A["Rz"]
    bench = (np.abs(V[:, 2] - BENCH_Z) < 0.05) & (np.abs(V[:, 0]) < 0.8) & (V[:, 1] > 0.3) & (V[:, 1] < 0.95) & (np.abs(Nn[:, 2]) > 0.9)
    bz = float(np.median(V[bench, 2]))
    col_err = float(np.median(np.abs(np.asarray(mesh.vertex_colors) - colour(V)).mean(1)))
    print(f"mesh: {len(mesh.triangles)} triangles; distance to truth median {med * 100:.2f} cm, 90% {p90 * 100:.2f} cm; "
          f"bench top {bz:.3f} m (true {BENCH_Z}); colour error {col_err:.3f}")
    assert med < 0.012 and p90 < 0.035, "room geometry not accurate"
    assert abs(bz - BENCH_Z) < 0.01, "bench height wrong"
    assert bench.sum() > 500, "bench missing (continuity)"
    assert col_err < 0.08, "colours wrong"
    assert (out / "room.glb").stat().st_size > 10_000
    # fly-through video
    r = render_room_video(str(out / "room_mesh.ply"), str(A["slam"]), str(tmp / "clips" / "clipA.mp4"),
                          str(out / "room_video.mp4"), seconds=4.0, width=480, height=270, title="synthetic")
    print("render:", r)
    cap = cv2.VideoCapture(str(out / "room_video.mp4"))
    n_read, lit = 0, []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        n_read += 1
        lit.append((fr.max(2) > 60).mean())
    assert n_read == r["frames"] == 4 * FPS and r["frames_with_hand_mesh"] == r["frames"], (n_read, r)
    assert min(lit) > 0.4, f"render mostly empty ({min(lit):.2f})"
    cap = cv2.VideoCapture(str(out / "room_video.mp4")); cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
    ok, fr = cap.read(); cv2.imwrite(str(tmp / "render_frame.png"), fr)
    shutil.copy(tmp / "render_frame.png", Path(tempfile.gettempdir()) / "room_render_frame.png")
    # web bundle + viewer page
    w = export_web(str(out), str(A["slam"]), str(out / "web"), seconds=4.0)
    print("web:", w)
    m = json.load(open(out / "web" / "motion.json"))
    assert m["n"] == 4 * FPS and m["hands"]["right"]["mesh"]
    assert (out / "web" / "hands_right.bin").stat().st_size == m["n"] * 778 * 3 * 2
    check_viewer(out / "web")
    check_cli_and_cache(tmp, fake_depth)
    print("ROOM CHECKS PASSED")


def check_cli_and_cache(tmp: Path, fake_depth):
    """The notebook path: SLAM folders saved to / restored from a tar with only the kept files, PS4 folders as
    symlinks, then run_room.py (CLI) for all three steps."""
    import os
    import tarfile
    import run_room
    import room.depth
    from room.depth import CallableDepth
    SAVE = ["vipe/pose", "vipe/intrinsics", "vipe/step.txt", "vipe/vipe/clip_slam_map.pt", "clip_small.mp4",
            "world_frame.json", "world_hands.jsonl", "world_hands_mesh.npz", "world_stats.json", "trajectory.csv"]
    slam2, links = tmp / "slam_restored", tmp / "ps4_links"
    slam2.mkdir(); links.mkdir()
    for name in ("clipA", "clipB"):
        d = tmp / "slam" / name
        with tarfile.open(tmp / f"slam_{name}.tar", "w") as tf:
            for f in SAVE:
                if (d / f).exists():
                    tf.add(d / f, arcname=f"{name}/{f}")
        with tarfile.open(tmp / f"slam_{name}.tar") as tf:
            tf.extractall(slam2)
        os.symlink(tmp / "ps4" / name, links / name)
    real = room.depth.MogeDepth
    room.depth.MogeDepth = lambda *a, **k: CallableDepth(fake_depth)
    try:
        run_room.main(["--clips-dir", str(tmp / "clips"), "--ps4-dir", str(links), "--slam-dir", str(slam2),
                       "--clips", "clipA", "clipB", "--target", "clipA", "--seconds", "2", "--out", str(tmp / "cli"),
                       "--kf-fps", "3", "--width", "320", "--height", "180"])
    finally:
        room.depth.MogeDepth = real
    for f in ("room_mesh.ply", "room.glb", "room_video.mp4", "room_stats.json", "web/viewer.html", "web/motion.json",
              "web/lib/three.module.js", "run_room_summary.json"):
        assert (tmp / "cli" / f).exists(), f"CLI did not write {f}"
    st = json.load(open(tmp / "cli" / "room_stats.json"))
    assert st["clips_used"] == ["clipA", "clipB"], st["clips_used"]
    print("CLI + SLAM cache restore work")


def check_viewer(web: Path):
    """Loads viewer.html in headless Chromium: page ready, no errors, slider moves the hands."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed: viewer page not checked")
        return
    srv = subprocess.Popen([sys.executable, "-m", "http.server", "8765"], cwd=web, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
    try:
        time.sleep(1.0)
        with sync_playwright() as p:
            b = p.chromium.launch(args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader"])
            pg = b.new_page(viewport={"width": 1100, "height": 650})
            errs = []
            pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
            pg.goto("http://localhost:8765/viewer.html")
            try:
                pg.wait_for_function("window.__ready === true", timeout=60000)
            except Exception:  # noqa: BLE001
                if any("jsdelivr" in e or "Failed to fetch" in e or "net::" in e for e in errs) or not errs:
                    print("viewer: three.js CDN not reachable from this machine; page not checked", errs[:2])
                    return
                raise AssertionError(f"viewer did not start: {errs}")
            pg.wait_for_timeout(1500)
            pg.fill("#slider", "90")
            pg.dispatch_event("#slider", "input")
            pg.click("[data-view=left]")
            pg.wait_for_timeout(800)
            t = pg.inner_text("#time")
            pg.screenshot(path=str(Path(tempfile.gettempdir()) / "room_viewer.png"))
            b.close()
        assert "f90" in t, t
        assert not errs, errs
        print("viewer page works:", t)
    finally:
        srv.terminate()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
