# PS5 world view (SLAM) on your own computer

You need the clip videos and their PS4 result folders (each with `hands.jsonl`, `camera.json`,
`head_rot.pkl`). PS4 is not run again.

## Option A: full SLAM (ViPE), via WSL2 Ubuntu

ViPE compiles CUDA code that targets Linux, so on Windows it runs inside WSL2.

1. In Windows PowerShell (admin), once: `wsl --install -d Ubuntu-24.04`, then restart and create a user.
   Keep the normal Windows NVIDIA driver up to date (it provides the GPU inside WSL).
2. In Ubuntu, go to this folder (Windows drives are under `/mnt/`):
   ```bash
   cd /mnt/d/personal/mytron/ps4-hand-tracking/code
   bash setup_slam_wsl.sh            # ~30-60 min the first time (downloads + compiling)
   ```
3. Every time after:
   ```bash
   source ~/ps5env/bin/activate
   cd /mnt/d/personal/mytron/ps4-hand-tracking/code
   python run_slam.py --clips-dir /mnt/d/personal/mytronD --ps4-dir /mnt/d/personal/mytronD/ps4v4_results --list
   python run_slam.py --clips-dir /mnt/d/personal/mytronD --ps4-dir /mnt/d/personal/mytronD/ps4v4_results \
       --clip mytronlabs-000003-008 --max-seconds 60 --slam-width 640
   ```
   Drop `--max-seconds 60` for the whole clip once the first minute looks right.

On an 8 GB GPU keep `--slam-width 640`. The first run also downloads ViPE's model weights.

## Option B: no SLAM install (rotation only), Windows directly

Head turns, gaze, body axis and hands are correct; head position is held fixed (fine for seated clips).
In the PS4 virtual environment:
```powershell
pip install rerun-sdk
python run_slam.py --clips-dir D:\personal\mytronD --ps4-dir D:\personal\mytronD\ps4v4_results --clip mytronlabs-000003-008 --backend rotation
```

## Results

`<parent of ps4-dir>/ps5_world/<clip>/` (or `--out-dir`):
`world_render.mp4`, `scene.rrd` (`rerun scene.rrd`), `world_hands.jsonl`, `trajectory.csv`, `world_stats.json`.
A finished clip is skipped on re-runs (use `--force`); a finished SLAM step (`vipe/`) is reused.

Check `world_stats.json`: `backend` should be `vipe`; `rotation_1s_vs_ps4_median_deg` around 2 or less;
hand-to-head distance 0.3-0.8 m.
