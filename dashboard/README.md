# SpectatorAI local workspace

Drop a video to transcode it locally, send it to Colab, and open synchronized video, 3D hands,
head orientation, wrist trails, quality lanes, and exports. Results work after Colab shuts down.
Showcase preserves the original story sections and explorer.

## One-cell connection on this computer

The local environment and matching credentials are prepared. Follow [START_HERE.md](../START_HERE.md):
open the updated `CONNECT_COLAB.ipynb`, choose GPU and press its single Play button.
First use asks for Drive access and the private `SpectatorAI-Colab.zip` from the repo root.
The ZIP and pretrained/package caches are saved under `MyDrive/SpectatorAI-Private`; later starts reuse them.
Google may request Drive access on a fresh runtime. Your videos are never written to Drive by this helper.
A fresh runtime still needs dependency installation/model loading; startup is not instant.

The cell requests runtime release after five idle minutes, on user interruption and on startup failure.
Queued/running processing, background rendering and complete streamed uploads/downloads prevent idle shutdown.
Health/job polling does not reset the idle timer. This releases the allocation rather than only stopping Python.
If Google's release API fails, the cell instructs you to Disconnect and delete runtime manually.
Startup exceptions are printed before runtime release and saved (with credentials redacted) to
`MyDrive/SpectatorAI-Private/startup-error.txt`, so automatic shutdown cannot hide the underlying error.

The bundle contains private connection settings but no footage, and is gitignored. Notebook code has no keys.
To update the generated artifacts, run `.venv/Scripts/python.exe tools/build_connect_notebook.py`
and `.venv/Scripts/python.exe tools/prepare_connection.py`. The notebook checks a setup version and prompts for a replacement ZIP when an update is required.
`code/tests/test_connect_notebook.py` validates one-cell setup reuse, wheel/install caching, idle protection,
and release on success/failure without accessing Google or using a GPU.

The original instructions below remain for fresh installations and advanced Drive/Secrets setup.

## Setup

1. Install Python 3.12, Node.js/npm and Git. From the repo root run `setup.bat` once (internet required).
   It creates `.venv`, installs CPU PyTorch and dependencies, caches WiLoR/CLIP and builds the frontend.
2. Copy `dashboard/.env.example` to `dashboard/.env`. Set `PS4_API_KEY` and `PS4_DISCOVERY_TOPIC` to
   long random values; put the same values in Colab Secrets. Never commit `.env`.
3. Optional: set `PS4_DRIVE_URL_FILE` to your Drive desktop path to `mytronD/api_url.json`.
   `PS4_API_URL` overrides discovery. Settings supports an override; clearing it restores discovery.
4. Zip the **contents of `code/`** as `ps4v4.zip`, upload to `MyDrive/mytronD/`, open
   `code/notebooks/ps4_api_server_colab.ipynb`, select an A100, enable Secrets access and Run all.
   The notebook publishes only the URL to ntfy and Drive. The key stays in Secrets.
5. Run `run.bat`. Open <http://127.0.0.1:8000>. Wait for the connection pill to say ready, then drop a
   30–60 s video. Processing starts automatically. Offline uploads wait locally until Colab connects.

After setup, the app/fonts/CPU fallback/saved workspaces work offline; Colab and discovery need internet.
`run.sh` supports Linux/macOS. The real 30 s / 1–2 minute target is **not measured yet**. Fake-model
tests validate integration, not A100 throughput or real-footage accuracy. Quality remains uncalibrated.

## Files and data

| Location (relative to repo root) | Purpose |
|---|---|
| `code/` | Only pipeline copy, Colab API, fast profile, CLI, notebooks and tests |
| `dashboard/frontend/` | React/TS/Three.js shell, original palette and bundled fonts |
| `dashboard/backend/` | Local FastAPI gateway, discovery, atomic job store and queue |
| `dashboard/results/<clip>/` | Optional existing A100 results; read-only in Library/Showcase |
| `dashboard/results/finetuned/<clip>/` | Paired fine-tuned comparison results |
| `dashboard/results/training/` | Original setup/dataset JSON and CSV loss curves |
| `dashboard/raw_clips/` | Optional original videos; mapping.json maps clip names to filenames |
| `weights/` | Optional WiLoR checkpoint and/or detector.pt |
| `dashboard/data/workspaces/<id>/` | Local video, job/timings, clip.json, outputs and view export |
| `dashboard/data/jobs/` | Existing 10 s / 8 fps CPU fallback |
| `dashboard/data/settings.json` | URL override, profile, auto-render, retention (default 24 h) |

The source dashboard had no bundled results in this checkout. No real footage was copied.
Media/results/.env/weights/exports/node_modules/dist are gitignored. Never commit or publish footage,
frames, screenshots or footage-derived outputs. Delete event data from Drive, Colab and source copies
after the event. Delete all data clears generated workspaces, CPU jobs, browser caches and exports;
read-only results and original source files remain for their owner to remove deliberately.

## Workspace

- Video is the clock. Slider seeks by frame; Left/Right steps a frame, Shift+Left/Right moves 1 s,
  Space plays/pauses; speeds 0.25/0.5/1×. Quality lanes seek on click and show reasons on hover.
- Left-drag orbits, right-drag or Shift-drag pans, wheel zooms, double-click focuses. Follow, Reset,
  Wearer's eye, Behind, Top and Side are available. Joints are instanced; bone buffers update in place
  using a frame ref and `useFrame`. Laptop 60 fps acceptance remains pending.
- Observed hands are solid; infilled hands translucent; estimated hands dashed/faint. Estimates are
  guesses. Older results without scene/quality still open with defaults.
- Rendered video downloads the pipeline MP4, generated after data is ready on a separate thread.
  Export my 3D view records the selected frames at 1× with video plus current 3D camera, then converts
  WebM to H.264/yuv420p MP4 locally. Controls lock during capture; time/speed are restored afterward.
  MediaRecorder tries VP9, VP8 then WebM. Use Chrome or Edge.
- With auto-render off, only data and browser-view exports are prepared. Enable it before uploading
  for the pipeline MP4. Data exports: hands JSONL, robot CSV, stats, quality and scene JSON.
- Room is a placeholder: PS5 SLAM runs separately. PS4 supplies head rotation, not room translation.

## Local API

The browser calls only this gateway; the Colab key is never sent to browser JS or API responses.

| Method | Route | Result |
|---|---|---|
| GET | `/api/colab/status?force=true` | connection, GPU, warm, queue, source, key_present and settings |
| POST | `/api/colab/url` | JSON: url, profile (fast/full), auto_render, retention_h |
| POST | `/api/workspaces` | multipart file; automatically queues a workspace |
| GET | `/api/workspaces` | jobs, newest first |
| GET | `/api/workspaces/{id}` | stage/progress/queue, timings, render state, error/traceback |
| POST | `/api/workspaces/{id}/retry` | retries failed processing using retained video |
| POST | `/api/workspaces/{id}/cpu` | starts 10 s CPU preview; returns CPU job ID |
| GET | `/api/workspaces/{id}/clip` | compact hands, scene and quality |
| GET | `/api/workspaces/{id}/video` | local H.264 with Range support |
| GET/POST | `/api/workspaces/{id}/export/render` | download MP4 / request render while job remains remote |
| POST | `/api/workspaces/{id}/export/view` | multipart WebM → MP4; returns download URL |
| GET | `/api/workspaces/{id}/download/{name}` | hands.jsonl, robot_wrist_gripper.csv, stats.json, quality.json, scene.json, tracklets.csv, render.mp4, view.mp4 |
| DELETE | `/api/workspaces/{id}` | removes locally and schedules remote cleanup |
| DELETE | `/api/data` | removes generated workspaces, CPU previews, web caches and exports |

States: uploaded → preparing → waiting (offline) → uploading → processing → fetching → ready; failed
retains error details. Timings record local stages/total; stage_seconds mirrors server modules.
Render readiness is independent. Unfinished jobs and render downloads resume after restart.
Remote jobs are deleted after fetching data and optional render; Colab TTL is six hours.

Original `/api/model`, `/api/clips` and video/download subroutes, `/api/metrics`, `/api/evidence`,
`/api/training`, `/api/upload`, `/api/jobs/{id}` remain. CPU camera defaults remain guessed calibration:
phone 68° HFOV, fisheye 150°.

## Checks

From the repo root:

```powershell
python code/tests/test_synthetic.py
python code/tests/test_side_cue.py
python code/tests/test_assign_overlap.py
python code/tests/test_api.py
python code/tests/test_gateway.py
npm run build --prefix dashboard/frontend
```

Gateway tests start fake-model HTTP on port 8765, upload synthetic video, verify scene/quality,
local seeking, render fetch, remote deletion, decodable view MP4 and delete-all. No real footage,
GPU or model downloads. Keep port 8765 free. `test_browser.mjs` tests Chrome and real MediaRecorder;
its header documents the optional Playwright install and synthetic fixture. Python checks here used
3.11; the fresh-clone Python 3.12 Windows installer still needs a clean-machine run.

## A100 acceptance: M5 pending

After notebook cells 1–3 run and `/health` says `warm: true`, add a Colab cell:

```python
%cd /content/ps4v4
!python tools/bench_profiles.py --clip "/content/drive/MyDrive/mytronD/hand_tracking_5.mp4" --seconds 30 --out /content/bench30
```

Use the actual filename of a verified clip (`hand_tracking_5` or `mytronlabs-000003-…`). The tool
transcodes once, warms the process, runs full and fast separately, and prints module times, presence
agreement, wrist distance, accepted counts, suspected swaps and quality difference. Benchmark JSON
contains numbers; its associated videos/outputs must stay private. The proximity-based swap check
requires local inspection of ambiguous crossings.

Then upload the same **30 s** clip three times through the dashboard, fast profile, auto-render on,
warm A100, no other queued work. Record each workspace's `timings.total` (right panel or job API).
The middle time is p50. Acceptance: p50 ≤120 s.

| Measurement | Full | Fast | Required |
|---|---|---|---|
| Pipeline wall time (30 s) | pending | pending | Fast ≤80 s |
| Side presence agreement | — | pending | ≥97% |
| Median wrist difference | — | pending | ≤15 mm |
| New side swaps | — | pending | 0 |
| Quality score difference | — | pending | report |
| Drop-to-ready run 1 | — | pending | report |
| Drop-to-ready run 2 | — | pending | report |
| Drop-to-ready run 3 | — | pending | report |
| Drop-to-ready p50 | — | pending | ≤120 s |

If speed misses the budget, change only the fast profile and rerun agreement checks, in order:

1. proposals.imgsz=832.
2. Add an opt-in 15 fps head-motion mode with interpolation (not enabled in this implementation).
3. probe.stride=6.

Keep CLIP and tuned decisions intact. Fallbacks are not applied without measurements. Colab credentials,
real footage and an A100 are required to complete M5; no performance numbers have been invented.


To reproduce the browser test with generated data (no real footage):

```powershell
python code/tools/create_demo.py
python -m uvicorn dashboard.backend.app:app --host 127.0.0.1 --port 8000
# In another terminal at the repo root:
npm install --prefix dashboard/data/browser-tools playwright --no-audit --no-fund
node code/tests/test_browser.mjs
```

Browser tests use installed Chrome on Windows. They cover presets, orbit/pan/zoom, slider/frame stepping,
playback, lane seeking, real MediaRecorder export/download, navigation and mobile overflow. No screenshots
are committed; the test's synthetic screenshot stays in ignored dashboard/data/.

Local verification (1 Nov 2026): baseline synthetic test passes with 98.8% recall, 100% side accuracy,
zero junk/other-person/knee detections and zero depth jumps; fast API and gateway end-to-end checks pass;
scene/cache invariants, side-cue and overlap regressions pass; production TypeScript/Vite build passes;
Chrome browser test passes including MP4 capture. These are synthetic checks, not real accuracy evidence.
A100 timing, real footage agreement, clean Python 3.12 installation, laptop 60 fps and manual Windows
Media Player/VLC verification remain unmeasured.


## Frontend design update

The app now opens on the supplied cream landing-page layout. Workspace uses the supplied dark
prototype: a 64 px top bar, 216/66 px collapsible navigation, split viewers, resizable docked timeline,
and a 352 px inspector with Frame / Clip / Quality tabs. Actual workspace data drives confidence,
wrist position/speed, quality and timings. Unsupported sample activity/gaze metrics are not fabricated.
Use the camera-view menu for Reset, Wearer’s eye, Behind, Top and Side; the Camera toggle shows head/gaze.
Exports are available from the top bar and the navigation’s Exports button. Compare opens landing results;
the original Showcase remains at `#/showcase`.

Supplied preview media is served locally from ignored `dashboard/data/design-reference/landing/assets/`
under `/api/preview-assets/`. The reference archives, preview footage and QA screenshots are not committed
or published. They illustrate the design; their prototype sample metrics are not accuracy measurements.
