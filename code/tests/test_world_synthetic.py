"""World step on a synthetic scene with known truth (no GPU, no ViPE needed).

A worker walks 1.2 m and turns 60 degrees while looking down at a bench; one hand rests on the bench.
"SLAM" artifacts are written in ViPE's exact format (pose/clip.npz cam->world OpenCV, intrinsics
[fx, fy, cx, cy, xi], vipe/clip_slam_map.pt) in ViPE's world frame (= the first camera). PS4's hands are
simulated with a WRONG focal length (x1.3), as GeoCalib gave us on real clips.
Checks: the resting hand stays still in the room (< 1 cm), its position matches the truth, gravity is
recovered, the head path length is right, the torso sits under the head, the render and the Rerun
recording are written.
Run: python tests/test_world_synthetic.py
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ps4v4.models.fake import CANON  # noqa: E402
from ps5world import backends  # noqa: E402
from ps5world.cameras import EquidistantCamera, MeiCamera  # noqa: E402
from ps5world.export import export_csv, export_jsonl, export_rerun  # noqa: E402
from ps5world.ps4_io import load_camera, load_hands  # noqa: E402
from ps5world.render_world import render_world  # noqa: E402
from ps5world.world import build_world, checks  # noqa: E402

W, H, FPS, SECS = 640, 360, 30, 6
N = FPS * SECS


def cam_pose(t: float):
    """True camera->world (Z up): walks 1.2 m along +x, turns 60 deg, pitched 40 deg down."""
    u = min(t / SECS, 1.0)
    pos = np.array([1.2 * u, 0.0, 1.25 + 0.02 * math.sin(6 * t)])
    yaw = math.radians(90 + 60 * u)                                     # heading in the xy plane
    pitch = math.radians(-40)
    fwd = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)])
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    return np.stack([right, down, fwd], 1), pos


def main():
    tmp = Path(tempfile.mkdtemp())
    true_cam = MeiCamera(260.0, 260.0, W / 2, H / 2, 0.9, W, H)          # wide-angle, full resolution
    # MEI round trip
    P = np.random.default_rng(0).normal(size=(500, 3)); P[:, 2] = np.abs(P[:, 2]) + 0.2
    rt = true_cam.rays(true_cam.project(P))
    assert np.allclose(rt, P / np.linalg.norm(P, axis=1, keepdims=True), atol=1e-6), "MEI round trip"
    # truth
    Rs, ts = zip(*[cam_pose(i / FPS) for i in range(N)])
    Rs, ts = np.array(Rs), np.array(ts)
    hand_w = (CANON - CANON.mean(0)) @ np.diag([1, 1, 1.0]) + np.array([0.3, 0.55, 0.80])   # resting on the bench
    rng = np.random.default_rng(1)
    floor = np.c_[rng.uniform(-2, 3, 40000), rng.uniform(-2, 3, 40000), np.zeros(40000)]
    bench = np.c_[rng.uniform(-0.6, 1.6, 15000), rng.uniform(0.3, 1.0, 15000), np.full(15000, 0.75)]
    wall = np.c_[rng.uniform(-2, 3, 8000), np.full(8000, 3.0), rng.uniform(0, 2.5, 8000)]
    pts_true = np.concatenate([floor, bench, wall])
    # ViPE world = first camera frame
    R0, t0 = Rs[0], ts[0]
    to_slam = lambda X: (X - t0) @ R0                                    # noqa: E731  (R0^T (X - t0))
    step = 2
    vout = tmp / "world" / "vipe"
    for d in ("pose", "intrinsics", "vipe"):
        (vout / d).mkdir(parents=True, exist_ok=True)
    inds = np.arange(0, N, step) // step
    T = np.repeat(np.eye(4)[None], len(inds), 0)
    for k, i in enumerate(range(0, N, step)):
        T[k, :3, :3] = R0.T @ Rs[i]
        T[k, :3, 3] = to_slam(ts[i])
    np.savez(vout / "pose" / "clip.npz", data=T, inds=inds)
    small = true_cam.scaled(W // 2, H // 2)
    np.savez(vout / "intrinsics" / "clip.npz", data=np.tile([small.fx, small.fy, small.cx, small.cy, small.xi],
                                                            (len(inds), 1)), inds=inds)
    import torch
    torch.save({"dense_disp_xyz": torch.from_numpy(to_slam(pts_true)).float(),
                "dense_disp_rgb": torch.full((len(pts_true), 3), 0.6), "dense_disp_packinfo": torch.zeros(1, 1, 2),
                "dense_disp_frame_inds": [0], "device": "cpu"}, vout / "vipe" / "clip_slam_map.pt")
    (vout / "step.txt").write_text(str(step))
    # videos: full-res clip and the downscaled copy ViPE would have used
    clip = str(tmp / "clip.mp4")
    for path, (w, h) in ((clip, (W, H)), (str(tmp / "world" / "clip_small.mp4"), (W // 2, H // 2))):
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
        for i in range(N):
            img = np.full((h, w, 3), 90, np.uint8)
            cv2.putText(img, str(i), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            vw.write(img)
        vw.release()
    # PS4 outputs with a wrong focal length (x1.3)
    ps4 = tmp / "ps4"
    ps4.mkdir()
    f_ps4 = 1.3 * true_cam.f_centre
    pcam = EquidistantCamera(f_ps4, f_ps4, W / 2, H / 2, W, H)
    json.dump({"width": W, "height": H, "fx": f_ps4, "fy": f_ps4, "cx": W / 2, "cy": H / 2}, open(ps4 / "camera.json", "w"))
    with open(ps4 / "hands.jsonl", "w") as fh:
        for i in range(N):
            Jc = (hand_w - ts[i]) @ Rs[i]                                 # true camera coordinates
            kp = true_cam.project(Jc)
            d_ps4 = np.linalg.norm(Jc[0]) * f_ps4 / true_cam.f_centre      # PS4's distance (from hand size)
            Jp = (Jc - Jc[0]) + d_ps4 * pcam.rays(kp[0])
            hand = {"side": "right", "state": "VISIBLE", "source": "observed", "conf": 0.9,
                    "bbox": [*kp.min(0), *kp.max(0)], "kp2d": kp.tolist(), "kp3d_cam": Jp.tolist()}
            fh.write(json.dumps({"frame": i, "t": i / FPS, "hands": [hand]}) + "\n")
    # run the world step
    hands, cam4 = load_hands(ps4), load_camera(ps4)
    traj = backends.read_vipe(vout, N, (W, H))
    world = build_world(hands, cam4, traj, FPS)
    st = checks(world, traj, None)
    out = tmp / "world"
    export_jsonl(world, str(out / "world_hands.jsonl"))
    export_csv(world, str(out / "trajectory.csv"))
    rrd = export_rerun(world, str(out / "scene.rrd"), clip)
    render_world(clip, hands, world, str(out / "world_render.mp4"), height=360, title="synthetic")
    # checks against the truth
    A, o = world["align"], world["origin"]
    wr = world["hands"]["right"][:, 0]
    jitter = float(np.linalg.norm(wr - wr.mean(0), axis=1).max())
    truth = (to_slam(hand_w[0]) - o) @ A.T
    err = float(np.linalg.norm(wr.mean(0) - truth))
    up_true = R0.T @ np.array([0, 0, 1.0])                                 # true up in the ViPE frame
    up_err = float(np.degrees(np.arccos(np.clip(A[2] @ up_true, -1, 1))))
    path_true = float(np.linalg.norm(np.diff(ts, axis=0), axis=1).sum())
    head_below = float(np.median(world["head_p"][:, 2] - world["pelvis"][:, 2]))
    print(json.dumps(st, indent=1))
    print(f"resting hand max drift in the room : {jitter * 100:.2f} cm")
    print(f"hand position error vs truth       : {err * 100:.2f} cm   (PS4 focal was 30% off)")
    print(f"gravity error                      : {up_err:.2f} deg ({world['up_method']})")
    print(f"head path {st['head_path_m']} m vs true {path_true:.2f} m; pelvis {head_below:.2f} m under the head")
    print(f"render / rerun                     : {(out / 'world_render.mp4').exists()} / {rrd is not None}")
    assert jitter < 0.01, "resting hand moves in the room"
    assert err < 0.02, "hand placed in the wrong spot"
    assert up_err < 3, "gravity wrong"
    assert abs(st["head_path_m"] - path_true) / path_true < 0.03, "head path length wrong"
    assert 0.5 < head_below < 0.8
    assert (out / "world_render.mp4").exists()
    # fallback backend (no SLAM): PS4 head rotation only, head position fixed
    head_R = np.einsum("ji,njk->nik", R0, Rs)                             # camera_t -> camera_0
    tr2 = backends.rotation_only(head_R, cam4, N)
    w2 = build_world(hands, cam4, tr2, FPS)
    render_world(clip, hands, w2, str(out / "world_render_rotation.mp4"), height=360, title="rotation only")
    assert w2["backend"] == "rotation" and np.isfinite(w2["hands"]["right"]).all()
    print("rotation-only fallback runs")
    check_slam_retries(tmp, vout, clip, ps4, head_R)
    check_hand_meshes(tmp, vout, clip, ps4)
    print("ALL WORLD CHECKS PASSED")


def check_hand_meshes(tmp, good_vout, clip, ps4):
    """PS4 hands_mesh.npz -> world mesh moved exactly like the joints (rigid), rendered, exported for the web."""
    import shutil
    from ps4v4.export import export_mesh as ps4_export_mesh
    from ps5world import runner
    from ps5world.ps4_io import load_hands as lh
    pm = tmp / "ps4_mesh"
    shutil.copytree(ps4, pm)
    h = lh(pm)
    assert h["faces"] is None, "no mesh file yet"
    rng = np.random.default_rng(3)
    jidx = rng.integers(0, 21, 778)
    off = rng.normal(0, 0.006, (778, 3))                                # a 'skin' around the joints, rigid
    seq = {}
    for s in ("left", "right"):
        J = h[s]["J_cam"]
        V = np.full((len(J), 778, 3), np.nan, np.float32)
        ok = np.isfinite(J[:, 0, 0])
        V[ok] = J[ok][:, jidx] + off
        seq[s] = {"verts_cam": V, "J_cam": J, "src": h[s]["src"]}
    faces = np.array([[i, i + 1, i + 2] for i in range(0, 775, 3)], np.int32)
    assert ps4_export_mesh(seq, faces, str(pm / "hands_mesh.npz"))
    h2 = lh(pm)
    t0 = int(np.nonzero(h2["right"]["src"])[0][0])
    assert np.abs(h2["right"]["V_cam"][t0] - seq["right"]["verts_cam"][t0]).max() < 5e-4, "mesh file round trip"
    for be in ("vipe", "rotation"):
        w = tmp / f"mesh_{be}"
        if be == "vipe":                                                # reuse the synthetic SLAM output
            shutil.copytree(good_vout, w / "vipe")
            shutil.copy(good_vout.parent / "clip_small.mp4", w / "clip_small.mp4")
        st = runner.run_world(clip, str(pm), str(w), be, geocalib=False, render=True, rerun=False)
        assert st["hand_meshes"], "world mesh not exported"
        d = np.load(w / "world_hands_mesh.npz")
        rel_w = d["right_rel"].astype(np.float32)
        fr = d["right_frames"]
        rel_c = seq["right"]["verts_cam"][fr] - seq["right"]["J_cam"][fr, :1]
        err = np.abs(np.linalg.norm(rel_w, axis=2) - np.linalg.norm(rel_c, axis=2)).max()
        print(f"hand mesh ({be}): {len(fr)} frames, rigid to {err * 1000:.2f} mm")
        assert len(fr) > 0.9 * N and err < 2e-3, "mesh not moved rigidly with the hand"
        assert (w / "world_render.mp4").stat().st_size > 10000
    print("hand meshes in the world view work")


def check_slam_retries(tmp, good_vout, clip, ps4, head_R):
    """SLAM divergence (exception or NaN output) -> next setting; all fail -> rotation-only, never a crash."""
    import pickle
    import shutil
    from ps5world import runner
    pickle.dump({"R": head_R}, open(ps4 / "head_rot.pkl", "wb"))
    backup = tmp / "good_vipe"
    shutil.copytree(good_vout, backup)
    calls = []

    def fake_vipe(video, work, boxes, fps, width, max_s, video_fps=30.0, optimize_intrinsics=True, robust=True,
                  intrinsics_damping=1.0):
        calls.append((optimize_intrinsics, fps))
        out = Path(work) / "vipe"
        if len(calls) == 1:                                   # the real failure: singular camera matrix
            raise RuntimeError("linalg.inv: The diagonal element 1 is zero, the input matrix is singular.")
        shutil.copytree(backup, out, dirs_exist_ok=True)
        shutil.copy(good_vout.parent / "clip_small.mp4", Path(work) / "clip_small.mp4")
        if len(calls) == 2:                                   # finishes, but bundle adjustment went to NaN
            d = np.load(out / "pose" / "clip.npz")
            T = d["data"].copy(); T[5:, 0, 3] = np.nan
            np.savez(out / "pose" / "clip.npz", data=T, inds=d["inds"])
        return out

    def fake_ok(video, work, *a, **k):
        out = Path(work) / "vipe"
        shutil.copytree(backup, out, dirs_exist_ok=True)
        shutil.copy(good_vout.parent / "clip_small.mp4", Path(work) / "clip_small.mp4")
        return out

    real = backends.run_vipe
    backends.run_vipe = fake_vipe
    try:
        w = tmp / "retry"
        st = runner.run_world(clip, str(ps4), str(w), "vipe", geocalib=False, render=False, rerun=False)
        names = [(a["name"], a["ok"]) for a in st["slam_attempts"]]
        print("SLAM attempts:", names)
        assert st["backend"] == "vipe" and names == [("refine_intrinsics", False), ("fixed_intrinsics", False),
                                                     ("fixed_intrinsics_10fps", True)], names
        assert calls[1][0] is False and calls[2][1] == 10.0
        calls.clear()
        backends.run_vipe = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("always diverges"))
        st = runner.run_world(clip, str(ps4), str(tmp / "retry2"), "vipe", geocalib=False, render=False, rerun=False)
        assert st["backend"] == "rotation" and not any(a["ok"] for a in st["slam_attempts"])
        print("SLAM retries + rotation fallback work")
        # the real Colab case: PS4 result covers the whole clip, the world step only the first part (--max-seconds),
        # with everything on (render, Rerun) and both backends
        backends.run_vipe = fake_ok
        half = N / FPS / 2
        for be in ("vipe", "rotation"):
            st = runner.run_world(clip, str(ps4), str(tmp / f"part_{be}"), be, max_seconds=half, geocalib=True,
                                  render=True, rerun=True)
            wl = sum(1 for _ in open(tmp / f"part_{be}" / "world_hands.jsonl"))
            print(f"partial clip ({be}): {wl} frames of {N}, backend {st['backend']}")
            assert st["backend"] == be and wl == int(round(half * FPS))
            assert (tmp / f"part_{be}" / "world_render.mp4").exists()
        print("partial clip (max_seconds) works")
    finally:
        backends.run_vipe = real


if __name__ == "__main__":
    main()
