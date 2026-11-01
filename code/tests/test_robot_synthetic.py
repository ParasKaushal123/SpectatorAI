"""Robot replay on synthetic hands (needs: mujoco mink daqp trimesh, git for the Panda model; no GPU).
Hands move on the bench in front of the worker, then one hand reaches 1.6 m away (must be graded out of reach).
Run: python tests/test_robot_synthetic.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# a right hand in its own frame: x across knuckles (index -> little = -x), y up, z = fingers forward (metres)
HAND = np.array([[0, 0, 0],
                 [0.025, -0.01, 0.02], [0.04, -0.015, 0.04], [0.05, -0.02, 0.06], [0.055, -0.025, 0.075],
                 [0.02, 0, 0.09], [0.022, 0, 0.125], [0.022, -0.005, 0.15], [0.02, -0.01, 0.17],
                 [0.0, 0, 0.095], [0, 0, 0.13], [0, -0.005, 0.155], [0, -0.01, 0.175],
                 [-0.018, 0, 0.09], [-0.02, 0, 0.12], [-0.02, -0.005, 0.145], [-0.02, -0.01, 0.16],
                 [-0.035, 0, 0.08], [-0.04, 0, 0.1], [-0.04, -0.005, 0.12], [-0.04, -0.01, 0.135]])


def hand_world(c, yaw, mirror):
    H = HAND * [(-1 if mirror else 1), 1, 1]
    # fingers point forward (+Y) and down 30 deg, palm down
    pitch = np.radians(-30)
    R0 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0.0]])                 # hand z -> world +Y, hand y -> world -Z...
    R0 = np.array([[1, 0, 0], [0, 0, 1.0], [0, 1.0, 0]]) @ np.diag([1, -1, 1])
    Rp = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    Ry = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return H @ (Ry @ Rp @ R0).T + c


def make_world(path: Path, fps=30, secs=8):
    n = fps * secs
    with open(path, "w") as fh:
        for t in range(n):
            s = t / fps
            hands = []
            for side, x0, mirror in (("left", -0.18, True), ("right", 0.18, False)):
                c = np.array([x0 + 0.06 * np.sin(s * 1.3 + x0), 0.42 + 0.06 * np.cos(s), -0.40 + 0.03 * np.sin(2 * s)])
                if side == "right" and 5.0 <= s < 6.0:
                    c = np.array([0.2, 1.6, -0.3])                         # reach far beyond the arm
                if side == "left" and 2.0 <= s < 2.5:
                    continue                                              # gap: no hand data
                hands.append({"side": side, "source": "observed",
                              "joints": hand_world(c, 0.2 * np.sin(s), mirror).round(5).tolist()})
            fh.write(json.dumps({"frame": t, "t": t / fps, "pose_valid": True,
                                 "head": {"position": [0, 0, 0], "quat_xyzw": [0, 0, 0, 1], "gaze_dir": [0, 0.7, -0.7]},
                                 "body": {"neck": [0, 0, -0.22], "pelvis": [0, 0, -0.62], "facing": [0, 1, 0]},
                                 "hands": hands}) + "\n")
    return n


def main():
    import trimesh
    from robot.retarget import retarget
    from room.web import export_web
    tmp = Path(tempfile.mkdtemp())
    slam = tmp / "slam"
    slam.mkdir()
    n = make_world(slam / "world_hands.jsonl")
    room = tmp / "room"
    room.mkdir()
    bench = trimesh.creation.box([1.6, 0.8, 0.04])
    bench.apply_translation([0, 0.5, -0.47])
    bench.visual.vertex_colors = [150, 120, 90, 255]
    bench.export(room / "room.glb")
    export_web(str(room), str(slam), str(room / "web"), seconds=None or 8.0)
    t0 = time.time()
    s = retarget(str(slam / "world_hands.jsonl"), str(room / "web"), seconds=8.0, cache=str(tmp / "cache"))
    print(json.dumps(s, indent=1), f"({time.time() - t0:.0f} s incl. download)")
    g = json.load(open(room / "web" / "robot.json"))
    R = np.array(g["grade"]["right"])
    far = R[150:180]
    assert (far == 2).mean() > 0.8, f"far reach not graded out of reach: {np.bincount(far)}"
    normal = np.r_[R[30:140], R[200:]]
    assert (normal == 1).mean() > 0.6, f"normal work not copyable enough: {np.bincount(normal)}"
    L = np.array(g["grade"]["left"])
    assert (L[62:74] == 0).all(), "gap frames should be 'no hand'"
    assert s["left"]["median_pos_error_cm"] < 1.5, s["left"]
    links = len(g["links"])
    assert (room / "web" / "robot_motion.bin").stat().st_size == n * links * 7 * 4
    shutil_viewer(room / "web")
    check_viewer(room / "web")
    print("ROBOT CHECKS PASSED")


def shutil_viewer(web: Path):
    import shutil
    shutil.copy(ROOT / "room" / "viewer.html", web / "viewer.html")


def check_viewer(web: Path):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed: viewer not checked")
        return
    srv = subprocess.Popen([sys.executable, "-m", "http.server", "8766"], cwd=web, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
    try:
        time.sleep(1)
        with sync_playwright() as p:
            b = p.chromium.launch(args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader"])
            pg = b.new_page(viewport={"width": 1100, "height": 650})
            errs = []
            pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
            pg.goto("http://localhost:8766/viewer.html")
            pg.wait_for_function("window.__ready === true", timeout=60000)
            pg.wait_for_timeout(1200)
            pg.fill("#slider", "160")
            pg.dispatch_event("#slider", "input")
            pg.click("[data-view=right]")
            pg.wait_for_timeout(800)
            sc = pg.inner_text("#score")
            pg.screenshot(path=str(Path(tempfile.gettempdir()) / "robot_viewer.png"))
            b.close()
        print("viewer:", sc)
        assert "robot-copyable" in sc and "out of reach" in sc, sc
        assert not errs, errs
    finally:
        srv.terminate()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    main()
