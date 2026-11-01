"""SLAM / world view from a folder of clips and the folder of their PS4 results. PS4 is NOT re-run.

    python run_slam.py --clips-dir <folder with the .mp4 files> --ps4-dir <ps4v4_results folder> --clip <name>

The clip is found by name in --clips-dir, ignoring any "Copy of " prefixes
("Copy of Copy of MTL-SKL00089-011.mp4" matches --clip MTL-SKL00089-011).
Its PS4 results must be in <ps4-dir>/<name>/ (hands.jsonl, camera.json, head_rot.pkl).
Results go to <out-dir>/<name>/ (default out-dir: a "ps5_world" folder next to --ps4-dir).

Examples:
    python run_slam.py --clips-dir D:/personal/mytronD --ps4-dir D:/personal/mytronD/ps4v4_results --clip mytronlabs-000003-008
    python run_slam.py ... --clip mytronlabs-000003-008 --max-seconds 60          # quick first try
    python run_slam.py ... --list                                                  # which clips are ready
    python run_slam.py ... --all                                                   # every clip with PS4 results
    python run_slam.py ... --clip X --backend rotation                             # no SLAM install needed
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
from pathlib import Path

from ps5world.runner import missing_ps4_files, run_world

log = logging.getLogger("run_slam")


def clean_name(p: Path) -> str:
    return p.stem.replace("Copy of ", "").strip().replace(" ", "_")


def find_clips(clips_dir: Path) -> dict[str, Path]:
    vids = sorted(list(clips_dir.glob("*.mp4")) + list(clips_dir.glob("*.MP4")))
    return {clean_name(p): p for p in vids}


def main():
    import ps5world
    print(f"ps5world code version {ps5world.VERSION}", flush=True)   # hand meshes: PS4 hands_mesh.npz
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips-dir", required=True, help="folder with the clip videos")
    ap.add_argument("--ps4-dir", required=True, help="folder with one PS4 result folder per clip")
    ap.add_argument("--out-dir", default=None, help="default: <parent of ps4-dir>/ps5_world")
    ap.add_argument("--clip", nargs="*", default=[], help="clip name(s), without 'Copy of' and .mp4")
    ap.add_argument("--all", action="store_true", help="every clip that has PS4 results")
    ap.add_argument("--list", action="store_true", help="only list clips and whether they are ready")
    ap.add_argument("--force", action="store_true", help="re-run clips that are already done")
    ap.add_argument("--backend", default="auto", choices=["auto", "vipe", "rotation"])
    ap.add_argument("--slam-fps", type=float, default=15.0)
    ap.add_argument("--slam-width", type=int, default=768, help="use 640 on 8 GB GPUs")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--no-geocalib", action="store_true")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--no-rerun", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    clips_dir, ps4_dir = Path(a.clips_dir), Path(a.ps4_dir)
    out_dir = Path(a.out_dir) if a.out_dir else ps4_dir.parent / "ps5_world"
    clips = find_clips(clips_dir)
    if not clips:
        raise SystemExit(f"no .mp4 files in {clips_dir}")
    if a.list:
        for name, p in clips.items():
            miss = missing_ps4_files(ps4_dir / name)
            done = (out_dir / name / "world_stats.json").exists()
            print(f"{name:32s} {'PS4 ok' if not miss else 'PS4 missing ' + ','.join(miss):36s} "
                  f"{'world done' if done else ''}")
        return
    names = [n for n in clips if not missing_ps4_files(ps4_dir / n)] if a.all else list(a.clip)
    if not names:
        raise SystemExit("give --clip <name> (see --list) or --all")
    results = {}
    for name in names:
        if name not in clips:
            close = [n for n in clips if name.lower() in n.lower()]
            print(f"[skip] {name}: no video in {clips_dir}" + (f" (did you mean {close[:3]}?)" if close else ""))
            continue
        miss = missing_ps4_files(ps4_dir / name)
        if miss:
            print(f"[skip] {name}: {ps4_dir / name} is missing {miss}")
            continue
        out = out_dir / name
        new_mesh = (ps4_dir / name / "hands_mesh.npz").exists() and not (out / "world_hands_mesh.npz").exists()
        if (out / "world_stats.json").exists() and not a.force and not new_mesh:
            print(f"[done] {name}: already in {out} (use --force to re-run)")
            continue
        t0 = time.time()
        try:
            st = run_world(str(clips[name]), str(ps4_dir / name), str(out), a.backend, a.slam_fps, a.slam_width,
                           a.max_seconds, not a.no_geocalib, not a.no_render, not a.no_rerun)
            results[name] = st
            print(f"[ok]   {name}: {(time.time() - t0) / 60:.1f} min, backend {st['backend']} -> {out}")
        except Exception as exc:                                     # noqa: BLE001 - keep going with other clips
            traceback.print_exc()
            print(f"[fail] {name}: {exc}")
    if results:
        print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
