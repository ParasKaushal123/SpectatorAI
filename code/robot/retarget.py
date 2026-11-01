"""Human hands (SLAM world view) -> two Franka Panda arms, by inverse kinematics. No training.

Input : world_hands.jsonl (room frame: metres, Z up) from run_slam.py.
Output: robot.glb (Panda link meshes) + robot.json + robot_motion.bin (per frame, per link: position + quaternion)
        written next to the room viewer (web/), plus robot_traj.json (joint angles, copyable flags, summary).

Each wrist drives one arm: tool point = between thumb tip and index tip (pulled toward the palm), tool axis = wrist
-> middle knuckle, fingers close across the knuckles; gripper opening = thumb-index distance. Every frame is graded:
1 copyable, 2 out of reach (pose error), 3 too fast for the arm, 0 no hand data.
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("robot")
MENAGERIE = "https://github.com/google-deepmind/mujoco_menagerie"
VEL_LIMIT = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])     # Panda joint speed limits, rad/s
HOME = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
TCP_OFFSET = 0.1034                                                     # hand origin -> between the fingertips


def get_panda(cache: str | Path) -> Path:
    """Downloads only the Franka Panda folder of MuJoCo Menagerie (~36 MB, Apache-2.0) once."""
    cache = Path(cache)
    xml = cache / "mujoco_menagerie" / "franka_emika_panda" / "panda.xml"
    if xml.exists():
        return xml
    cache.mkdir(parents=True, exist_ok=True)
    d = cache / "mujoco_menagerie"
    subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse", MENAGERIE, str(d)], check=True)
    subprocess.run(["git", "-C", str(d), "sparse-checkout", "set", "franka_emika_panda"], check=True)
    assert xml.exists(), "Panda model download failed"
    return xml


def _yaw_quat(yaw: float) -> list:
    return [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]        # wxyz about Z


def build_model(panda_xml: Path, bases: dict):
    """Two Pandas (prefixes left/, right/) with a 'tcp' site between the fingertips. bases: side -> (pos, yaw)."""
    import mujoco
    spec = mujoco.MjSpec()
    for side, (pos, yaw) in bases.items():
        child = mujoco.MjSpec.from_file(str(panda_xml))
        child.body("hand").add_site(name="tcp", pos=[0, 0, TCP_OFFSET])
        fr = spec.worldbody.add_frame(pos=list(map(float, pos)), quat=_yaw_quat(yaw))
        spec.attach(child, prefix=f"{side}/", frame=fr)
    return spec.compile()


def load_world(path: str | Path, seconds: float | None):
    rows = [json.loads(l) for l in open(path)]
    fps = 1.0 / max(rows[1]["t"] - rows[0]["t"], 1e-6) if len(rows) > 1 else 30.0
    n = len(rows) if not seconds else min(len(rows), int(round(seconds * fps)))
    J = {s: np.full((n, 21, 3), np.nan) for s in ("left", "right")}
    src = {s: np.zeros(n, int) for s in ("left", "right")}
    code = {"observed": 1, "infilled": 2, "estimated": 3}
    head = np.array([r["head"]["position"] for r in rows[:n]], float)
    neck, facing = [], []
    for t, r in enumerate(rows[:n]):
        for h in r["hands"]:
            J[h["side"]][t] = h["joints"]
            src[h["side"]][t] = code.get(h["source"], 1)
        b = r.get("body") or {}
        neck.append(b.get("neck", (np.array(r["head"]["position"]) - [0, 0, 0.22]).tolist()))
        f = b.get("facing") or r["head"]["gaze_dir"]
        facing.append([f[0], f[1], 0.0])
    return {"fps": fps, "n": n, "J": J, "src": src, "head": head, "neck": np.array(neck, float),
            "facing": np.array(facing, float)}


def hand_targets(J: np.ndarray, src: np.ndarray, fps: float):
    """Per frame: tool position (n,3), tool rotation (n,3,3), gripper opening m (n,), valid (n,)."""
    from scipy.ndimage import gaussian_filter1d
    n = len(J)
    valid = (src > 0) & np.isfinite(J).all((1, 2))
    pos, R, grip = np.full((n, 3), np.nan), np.full((n, 3, 3), np.nan), np.zeros(n)
    for t in np.nonzero(valid)[0]:
        j = J[t]
        palm = j[[0, 5, 9, 13, 17]].mean(0)
        pinch = 0.5 * (j[4] + j[8])
        pos[t] = 0.5 * palm + 0.5 * pinch
        z = j[9] - j[0]
        z /= max(np.linalg.norm(z), 1e-9)
        y = j[5] - j[17]
        y -= (y @ z) * z
        if np.linalg.norm(y) < 1e-6:
            y = np.cross(z, [0, 0, 1.0])
        y /= np.linalg.norm(y)
        R[t] = np.stack([np.cross(y, z), y, z], 1)
        grip[t] = np.clip(np.linalg.norm(j[4] - j[8]) - 0.01, 0.0, 0.08)
    if valid.sum() > 3:                                                # smooth positions / grip inside valid runs
        idx = np.nonzero(valid)[0]
        runs = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
        for r in runs:
            if len(r) > 4:
                pos[r] = gaussian_filter1d(pos[r], max(1.0, 0.07 * fps), axis=0, mode="nearest")
                grip[r] = gaussian_filter1d(grip[r], max(1.0, 0.07 * fps), mode="nearest")
    return pos, R, grip, valid


def place_bases(W: dict, pos: dict) -> dict:
    """Each arm's base sits beside and slightly behind the worker's head, at the height of the work. 'Forward' is the
    direction from the head to where the hands work (more reliable than the gaze, which mostly points down)."""
    head = np.nanmedian(W["head"], 0)
    P = {s: p[np.isfinite(p).all(1)] for s, p in pos.items()}
    allp = np.concatenate([p for p in P.values() if len(p)] or [head[None] + [0, 0.4, -0.4]])
    work = np.median(allp, 0)
    f = np.r_[work[:2] - head[:2], 0.0]
    if np.linalg.norm(f) < 0.05:
        f = np.nanmedian(W["facing"], 0)
    f = f / max(np.linalg.norm(f), 1e-9)
    right = np.cross(f, [0, 0, 1.0])
    z = float(work[2]) - 0.15
    yaw = float(np.arctan2(f[1], f[0]))
    out = {}
    for s, sgn in (("left", -1.0), ("right", 1.0)):
        out[s] = (np.array([*(head[:2] + sgn * 0.25 * right[:2] - 0.05 * f[:2]), z]), yaw)
    return out


def retarget(world_jsonl: str, out_dir: str, seconds: float | None = 120.0, cache: str = "~/.cache/spectator_robot",
             iters: int = 6, pos_tol: float = 0.03, rot_tol_deg: float = 35.0) -> dict:
    import mink
    import mujoco
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    W = load_world(world_jsonl, seconds)
    n, fps = W["n"], W["fps"]
    tg = {s: hand_targets(W["J"][s], W["src"][s], fps) for s in ("left", "right")}
    bases = place_bases(W, {s: tg[s][0] for s in tg})
    model = build_model(get_panda(Path(cache).expanduser()), bases)
    data = mujoco.MjData(model)
    cfg = mink.Configuration(model)
    q = cfg.q.copy()
    adr = {s: [model.jnt_qposadr[model.joint(f"{s}/joint{i}").id] for i in range(1, 8)] for s in ("left", "right")}
    fadr = {s: [model.jnt_qposadr[model.joint(f"{s}/finger_joint{i}").id] for i in (1, 2)] for s in ("left", "right")}
    for s in ("left", "right"):
        q[adr[s]] = HOME
        q[fadr[s]] = 0.04
    cfg.update(q)
    tasks = {s: mink.FrameTask(f"{s}/tcp", "site", position_cost=5.0, orientation_cost=0.25, lm_damping=0.1)
             for s in ("left", "right")}
    posture = mink.PostureTask(model, cost=1e-3)
    posture.set_target(q.copy())
    limits = [mink.ConfigurationLimit(model)]
    sid = {s: model.site(f"{s}/tcp").id for s in ("left", "right")}
    Q = np.zeros((n, model.nq))
    grade = {s: np.zeros(n, np.int8) for s in ("left", "right")}
    err = {s: np.full(n, np.nan) for s in ("left", "right")}
    last = {s: None for s in ("left", "right")}
    dt = 1.0 / fps
    for t in range(n):
        active = []
        for s in ("left", "right"):
            p, R, g, v = tg[s]
            if v[t]:
                last[s] = (p[t], R[t], g[t])
            if last[s] is not None:
                tasks[s].set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(last[s][1]), last[s][0]))
                active.append(tasks[s])
        for _ in range(iters if active else 0):
            vel = mink.solve_ik(cfg, active + [posture], dt, solver="daqp", damping=1e-3, limits=limits)
            cfg.integrate_inplace(vel, dt)
        q_now = cfg.q.copy()
        for s in ("left", "right"):                                     # gripper opening = thumb-index distance
            q_now[fadr[s]] = (last[s][2] / 2) if last[s] is not None else 0.04
        cfg.update(q_now)
        Q[t] = cfg.q
        for s in ("left", "right"):
            if not tg[s][3][t]:
                continue
            data.qpos[:] = cfg.q
            mujoco.mj_kinematics(model, data)
            pe = np.linalg.norm(data.site_xpos[sid[s]] - tg[s][0][t])
            Rs = data.site_xmat[sid[s]].reshape(3, 3)
            re = np.degrees(np.arccos(np.clip((np.trace(Rs.T @ tg[s][1][t]) - 1) / 2, -1, 1)))
            err[s][t] = pe
            grade[s][t] = 1 if (pe <= pos_tol and re <= rot_tol_deg) else 2
    for s in ("left", "right"):                                         # too fast for the arm
        v = np.abs(np.diff(Q[:, adr[s]], axis=0)) * fps
        fast = np.r_[False, (v > VEL_LIMIT).any(1)]
        grade[s][(grade[s] == 1) & fast] = 3
    summary = {}
    for s in ("left", "right"):
        g = grade[s]
        tracked = int((g > 0).sum())
        summary[s] = {"tracked_frames": tracked, "copyable_share": round(float((g == 1).sum() / max(tracked, 1)), 3),
                      "out_of_reach_share": round(float((g == 2).sum() / max(tracked, 1)), 3),
                      "too_fast_share": round(float((g == 3).sum() / max(tracked, 1)), 3),
                      "median_pos_error_cm": round(float(np.nanmedian(err[s]) * 100), 2) if tracked else None}
    both = np.r_[grade["left"][grade["left"] > 0], grade["right"][grade["right"] > 0]]
    summary["copyable_share"] = round(float((both == 1).mean()), 3) if len(both) else 0.0
    json.dump({"fps": fps, "n": n, "joint_names": [model.joint(i).name for i in range(model.njnt)],
               "qpos": np.round(Q, 5).tolist(), "grade": {s: grade[s].tolist() for s in grade},
               "bases": {s: {"pos": bases[s][0].round(4).tolist(), "yaw": bases[s][1]} for s in bases},
               "summary": summary}, open(out / "robot_traj.json", "w"))
    export_web(model, Q, grade, fps, summary, out)
    summary["seconds"] = round(time.time() - t0, 1)
    log.info("robot: %d frames, copyable %.0f%% (%s), %.0f s", n, 100 * summary["copyable_share"],
             {s: summary[s]["copyable_share"] for s in ("left", "right")}, summary["seconds"])
    return summary


def export_web(model, Q: np.ndarray, grade: dict, fps: float, summary: dict, out: Path) -> None:
    """robot.glb: one node per link (visual meshes in the link frame); robot_motion.bin: float32 [n][links][7]
    = position xyz + quaternion xyzw in the room frame; robot.json: names, grades, summary."""
    import mujoco
    import trimesh
    data = mujoco.MjData(model)
    bodies = []
    scene = trimesh.Scene()
    for b in range(1, model.nbody):
        parts = []
        for g in np.nonzero(model.geom_bodyid == b)[0]:
            if model.geom_group[g] != 2 or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = model.geom_dataid[g]
            va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
            V = model.mesh_vert[va:va + vn].astype(float)
            F = model.mesh_face[fa:fa + fn]
            Rg = np.zeros(9)
            mujoco.mju_quat2Mat(Rg, model.geom_quat[g])
            V = V @ Rg.reshape(3, 3).T + model.geom_pos[g]
            rgba = model.mat_rgba[model.geom_matid[g]] if model.geom_matid[g] >= 0 else model.geom_rgba[g]
            col = np.tile((np.clip(rgba, 0, 1) * 255).astype(np.uint8), (len(V), 1))
            parts.append(trimesh.Trimesh(V, F, vertex_colors=col, process=False))
        if not parts:
            continue
        name = model.body(b).name.replace("/", "__")
        scene.add_geometry(trimesh.util.concatenate(parts), node_name=name, geom_name=name + "_mesh")
        bodies.append((b, name))
    scene.export(str(out / "robot.glb"))
    n = len(Q)
    M = np.zeros((n, len(bodies), 7), np.float32)
    for t in range(n):
        data.qpos[:] = Q[t]
        mujoco.mj_kinematics(model, data)
        for k, (b, _) in enumerate(bodies):
            w, x, y, z = data.xquat[b]
            M[t, k] = [*data.xpos[b], x, y, z, w]
    M.tofile(out / "robot_motion.bin")
    json.dump({"fps": fps, "n": n, "links": [nm for _, nm in bodies], "grade": {s: grade[s].tolist() for s in grade},
               "summary": summary}, open(out / "robot.json", "w"), separators=(",", ":"))
