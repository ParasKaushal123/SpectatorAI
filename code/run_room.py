"""Room twin for one workstation: 3D room from several seated clips + fly-through video + web viewer.

Needs, per clip: the video, its PS4 result folder (hands.jsonl, camera.json, hands_mesh.npz for meshes) and its
run_slam output folder (vipe/, clip_small.mp4, world_frame.json, world_hands.jsonl, world_hands_mesh.npz).

    python run_room.py --clips-dir <videos> --ps4-dir <ps4 results> --slam-dir <run_slam out> \
        --clips MTL-SKL00089-006 MTL-SKL00089-007 --target MTL-SKL00089-007 --seconds 120 --out <out>
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def find_video(clips_dir: Path, name: str) -> Path | None:
    clean = lambda p: p.stem.replace("Copy of ", "").strip().replace(" ", "_")        # noqa: E731
    hits = [p for p in clips_dir.glob("*.mp4") if clean(p) == name]
    return hits[0] if hits else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips-dir", required=True)
    ap.add_argument("--ps4-dir", required=True)
    ap.add_argument("--slam-dir", required=True)
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=120.0, help="length of the video / web motion (target clip)")
    ap.add_argument("--steps", nargs="+", default=["build", "render", "web"], choices=["build", "render", "web"])
    ap.add_argument("--depth", default="moge", choices=["moge"])
    ap.add_argument("--kf-fps", type=float, default=2.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    import room
    from room.build import ClipInput, build_room
    print(f"room code version {room.VERSION}", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    clips = []
    for name in a.clips:
        v = find_video(Path(a.clips_dir), name)
        slam = Path(a.slam_dir) / name
        ps4 = Path(a.ps4_dir) / name
        miss = [str(p) for p in (slam / "world_frame.json", slam / "vipe" / "pose" / "clip.npz", ps4 / "hands.jsonl")
                if not p.exists()]
        if v is None or miss:
            raise SystemExit(f"{name}: missing {'video' if v is None else ''} {miss}")
        clips.append(ClipInput(name, str(v), str(ps4), str(slam)))
    target = next(c for c in clips if c.name == a.target)
    t0 = time.time()
    summary = {}
    if "build" in a.steps:
        from room.depth import MogeDepth
        st = build_room(clips, a.target, str(out), MogeDepth(), {"kf_fps": a.kf_fps})
        summary["build"] = {k: st[k] for k in ("clips_used", "registration", "depth_scale", "triangles", "seconds")}
        print(json.dumps(summary["build"], indent=1, default=float), flush=True)
    if "render" in a.steps:
        from room.render_video import render_room_video
        summary["render"] = render_room_video(str(out / "room_mesh.ply"), target.slam, target.video,
                                              str(out / "room_video.mp4"), a.seconds, a.width, a.height,
                                              title=a.target)
        print("render:", summary["render"], flush=True)
    if "web" in a.steps:
        from room.web import export_web
        summary["web"] = export_web(str(out), target.slam, str(out / "web"), a.seconds)
        print("web:", summary["web"], flush=True)
    summary["total_s"] = round(time.time() - t0, 1)
    json.dump(summary, open(out / "run_room_summary.json", "w"), indent=1, default=float)
    print(f"done in {summary['total_s'] / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
