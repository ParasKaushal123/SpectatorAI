# PS4 + PS5: 3D hand tracking and a room-fixed world view from egocentric factory video

This code takes head-mounted fisheye video of a factory worker and tracks both of the wearer's hands in 3D. For every frame it outputs the hand's 21 joints and the wrist pose. It also gives a robot-ready wrist + gripper trajectory and a side-by-side render.

It rejects other people's hands, knees, feet, forearms and objects that look like hands. It keeps left and right correct, including when the hands cross. Short out-of-view gaps are filled with an estimate that is clearly labelled as one.

## Folder structure

```
code/
├── run.py                  command line: run the pipeline on one or more clips
├── configs/default.yaml    every threshold and setting
├── ps4v4/                  the pipeline (Python package)
│   ├── pipeline.py         runs the modules in order, caches each one (resumes after a disconnect)
│   ├── camera.py           M1  intrinsics (calibration / GeoCalib / FOV guess) + head rotation
│   ├── proposals.py        M2  hand boxes from the WiLoR detector
│   ├── tracklets.py        M3  links boxes into short tracks (no decisions yet)
│   ├── probe.py            M3P 3D + CLIP check on a sample of each track
│   ├── assign.py           M3A decides LEFT / RIGHT / REJECT per track
│   ├── recon.py            M4  3D hand (WiLoR on perspective-corrected crops)
│   ├── sequence.py         M5  smoothing, gap filling, out-of-view estimate
│   ├── render.py, export.py  M6  video, hands.jsonl, robot CSV, stats
│   ├── quality.py          MQ  clip rating: usable vs unusable parts (hands out of view, blur, head/body motion)
│   ├── evaluate.py         accuracy against hand-labelled frames
│   ├── geometry.py, video.py, cache.py, config.py
│   └── models/             detector, 3D reconstructor, CLIP verifier, test stand-ins
├── run_slam.py             command line: PS5 world view from a clips folder + the PS4 results folder
├── run_world.py            command line: PS5 world view for one clip file + its PS4 result folder
├── setup_slam_wsl.sh       one-time SLAM setup in WSL2 Ubuntu (see docs/SLAM_LOCAL.md)
├── ps5world/               PS5: SLAM + world assembly
│   ├── backends.py         ViPE (wide-angle camera, hands/people masked) or rotation-only fallback
│   ├── cameras.py          unified (Mei) and equidistant fisheye camera models
│   ├── world.py            re-lift PS4 hands into the SLAM camera, gravity, Z-up room frame, body axis, checks
│   ├── render_world.py     side-by-side video: frame + 3D room view following the worker
│   └── export.py           world_hands.jsonl, trajectory.csv, Rerun recording (scene.rrd)
├── notebooks/
│   ├── ps4v4_colab.ipynb             Colab: a few clips
│   ├── ps4v4_colab_all_clips.ipynb   Colab: the whole dataset, split over 2 accounts (PART = 1 / 2)
│   ├── ps5_world_slam.ipynb          Colab: PS5 world view for one clip
│   └── ps4_api_server_colab.ipynb    Colab: pipeline API server + public tunnel for the local frontend
├── server/app.py           HTTP API (FastAPI) around the pipeline + clip rating, run on Colab for the frontend
├── tests/                  synthetic end-to-end test + regression tests (no GPU, no downloads)
├── tools/                  chumpy patch, label-frame sampler
├── docs/API.md             API + quality.json reference for the frontend
├── docs/DESIGN.md          design notes: why each module exists, what each fix solved
├── requirements.txt
└── setup_windows.ps1       one-time local setup on Windows
```

## Run on Google Colab (recommended: A100)

1. Zip this `code/` folder as `ps4v4.zip`.
2. Upload `ps4v4.zip` to `MyDrive/mytronD/` in Google Drive, next to the videos.
3. Open `notebooks/ps4v4_colab_all_clips.ipynb` in Colab and pick an A100 runtime.
4. Run the cells in order:
   - Cell 1 installs everything.
   - Cell 2 unpacks the code, runs a 1-minute self-test, and downloads the model weights.
   - Cell 3 processes the clips.
5. Results are saved to `MyDrive/mytronD/ps4v4_results/<clip>/` as each clip finishes.

If Colab disconnects, run the cells again; finished clips are skipped.

## Run locally

```bash
pip install -r requirements.txt
pip install --no-deps git+https://github.com/warmshao/WiLoR-mini
pip install --no-build-isolation --no-deps git+https://github.com/mattloper/chumpy
python tools/patch_chumpy.py
pip install git+https://github.com/cvg/GeoCalib        # optional

python run.py --clip "path/to/clips/*.mp4"                           # full pipeline, cached per module
python run.py --clip clip.mp4 --set max_seconds=30                   # quick check
python run.py --clip clip.mp4 --from M3A --set assign.accept_bias=0.3   # re-decide only, seconds
```

On Windows, run `powershell -ExecutionPolicy Bypass -File setup_windows.ps1` once instead of the install lines.

## Outputs (per clip)

| File | Content |
|---|---|
| `render.mp4` | The video with each hand's skeleton next to a 3D view. Rejected candidates are shown in grey with the reason. |
| `hands.jsonl` | One line per frame. Each hand has: side, state, box, 2D and 3D joints, orientation, and `source` (observed / infilled / estimated). |
| `robot_wrist_gripper.csv` | Wrist position and orientation (head-stable frame, metres) plus gripper opening, per frame. |
| `stats.json` | Share of frames tracked, smoothness, depth jumps, time per module. |
| `tracklets.csv` | Every candidate track with its evidence and decision (for tuning). |
| `quality.json` | Clip rating: score, usable segments, per-lane timestamped segments (see `docs/API.md`). |

## PS5 world view (SLAM)

`run_world.py` places the worker's head (as a camera), gaze, body axis and both hands in one gravity-aligned room frame (Z up, metres). It takes the clip and its PS4 result folder:

```bash
# setup (Linux / WSL2): see docs/SLAM_LOCAL.md or run setup_slam_wsl.sh
python run_slam.py --clips-dir <videos> --ps4-dir <ps4v4_results> --list
python run_slam.py --clips-dir <videos> --ps4-dir <ps4v4_results> --clip mytronlabs-000003-008 --max-seconds 60
python run_slam.py --clips-dir <videos> --ps4-dir <ps4v4_results> --clip mytronlabs-000003-008 --backend rotation  # no SLAM install
```

How it works:

1. **Camera tracking:** [ViPE](https://github.com/nv-tlabs/vipe) runs with its `wide_angle` preset, a unified camera model for fisheye. Its intrinsics start from GeoCalib and are refined during tracking. It runs at 15 fps and the poses are interpolated to every frame.
2. **Hands masked out of tracking:** ViPE's own "person / hand / arm" masks are combined with PS4's hand boxes, which are grown and extended to the bottom edge to cover the forearm.
3. **Hands re-lifted into the SLAM camera:** each hand keeps PS4's shape. Its wrist is re-aimed along the SLAM camera's ray for the same pixel, and its distance is rescaled by the ratio of the two focal lengths. This also corrects GeoCalib's focal error.
4. **Gravity:** GeoCalib gravity, refined by the dominant horizontal plane of the map (bench or floor).
5. **Room frame:** Z up, Y = the worker's average facing direction, origin = the first head position.
6. **Body axis:** a torso under the head along gravity, facing the head's smoothed heading.
7. **Gaze:** the camera's forward axis.

| Output | Content |
|---|---|
| `world_render.mp4` | The frame next to a 3D room view: map points, head frustum and path, gaze ray, torso, hands with light trails |
| `scene.rrd` | Rerun recording; open with `rerun scene.rrd` and scrub the timeline |
| `world_hands.jsonl` | Per frame: head pose, gaze, neck/pelvis/facing, 21 world joints per hand |
| `trajectory.csv` | Per frame: head, gaze, pelvis, facing yaw, both wrists |
| `world_stats.json` | Backend, gravity method, hand scale factor, head path, hand-to-head distance, agreement with PS4's head rotation |

## Clip rating and the API for the frontend

Every run also writes `quality.json`. It holds:
- an overall score (0–100) and a grade (A–D), plus sub-scores;
- the usable share of the clip, and the usable segments;
- per-lane timestamped segments: each hand (visible / partial / low_confidence / gap_filled / out_of_view), image (sharp / blurred), head (stable / fast_turn), body (still / moving), and usable / unusable with the reasons;
- 10 Hz arrays for charts.

To serve the pipeline to a frontend on your own computer, run `notebooks/ps4_api_server_colab.ipynb` on a Colab A100. It starts `server/app.py` and opens a public HTTPS tunnel, then prints a URL and an API key. Endpoints and schemas are in `docs/API.md`; `tools/api_client.py` tests the API from the command line.

## Tests

```bash
python tests/test_synthetic.py       # end-to-end on a generated clip with stand-in models
python tests/test_assign_overlap.py
python tests/test_side_cue.py
python tests/test_world_synthetic.py  # PS5 world step on a synthetic scene with known truth
python tests/test_api.py              # API: upload -> progress -> result -> files -> delete
```

## Data

Videos, extracted frames and outputs derived from footage must not be committed. The repository `.gitignore` excludes them.
