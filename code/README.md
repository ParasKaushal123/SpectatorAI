# PS4: 3D hand tracking from egocentric factory video

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
│   ├── evaluate.py         accuracy against hand-labelled frames
│   ├── geometry.py, video.py, cache.py, config.py
│   └── models/             detector, 3D reconstructor, CLIP verifier, test stand-ins
├── notebooks/
│   ├── ps4v4_colab.ipynb             Colab: a few clips
│   └── ps4v4_colab_all_clips.ipynb   Colab: the whole dataset, split over 2 accounts (PART = 1 / 2)
├── tests/                  synthetic end-to-end test + regression tests (no GPU, no downloads)
├── tools/                  chumpy patch, label-frame sampler
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

## Tests

```bash
python tests/test_synthetic.py       # end-to-end on a generated clip with stand-in models
python tests/test_assign_overlap.py
python tests/test_side_cue.py
```

## Data

Videos, extracted frames and outputs derived from footage must not be committed. The repository `.gitignore` excludes them.
