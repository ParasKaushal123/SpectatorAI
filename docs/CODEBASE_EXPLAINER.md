# SpectatorAI codebase explainer

## Part 1: What the system is

SpectatorAI takes **head-mounted factory video** and turns it into **robot-ready data**:

1. **Hands:** the wearer's two hands as 3D skeletons and meshes in metres, with left/right kept correct and other people's hands ignored.
2. **Quality:** which seconds of the clip are usable (hands visible, sharp, head steady, body still), and why the rest aren't.
3. **World:** the head, gaze, body and hands placed in a fixed, gravity-aligned room (via SLAM).
4. **Room twin:** a 3D copy of the workstation built from the video.
5. **Robot:** two robot arms that copy the hands, with a "how much a robot can copy" score.
6. **Website:** a local dashboard. You upload a video, a Colab GPU processes it, and you explore the results in 3D.

**Two machines are involved:**
- **Google Colab (GPU):** the heavy models.
- **Your laptop:** the website, the stored results and the robot step.

They talk over an HTTPS tunnel, and the API key never reaches the browser.

```
ps4-hand-tracking/
├── code/        all pipelines (Python): hand tracking, quality, SLAM, room, robot, Colab API server
├── dashboard/   the website: backend (FastAPI gateway) + frontend (React/three.js)
├── tools/       repo-level helpers: build the Colab connect notebook and private bundle, generate fixtures
├── CONNECT_COLAB.ipynb   one-cell notebook that starts the GPU server for the website
├── setup.bat / run.bat / run.sh   install + start the website
├── README.md / ARCHITECTURE.md    user guide / technical reference
└── .gitignore   keeps footage, results, weights and credentials out of git
```

### The data flow in one paragraph

You add a video on the website. The **gateway** shrinks it (≤ 720p, ≤ 30 fps, ≤ 60 s) and uploads it to the **Colab API server**. The server runs **`ps4v4.pipeline.run_clip`** (the fast profile): camera → detect → head rotation → tracklets → probe → assign → 3D → smoothing → quality → scene. The gateway polls for progress, downloads the results, builds a compact clip JSON, and the browser shows the **Workspace**. The SLAM, room twin and robot steps are **offline workflows** run from notebooks or the command line. Their results are imported into `dashboard/results/room_twin/<clip>/` and shown on the **Room** page.

---

## Part 2: The codebase in detail

### 2.1 `code/ps4v4/`: hand tracking (the core)

The modules run in this order. The table gives the file, its job, and the main function.

| File | Job | Main function / notes |
|---|---|---|
| `pipeline.py` | Orchestrator | `run_clip(video, cfg, clip_name, start_from, progress)` runs M1 → MQ through `stage()`. Each stage is cached and reports start, done or cached. |
| `cache.py` | Per-module cache | A hash of the module's config section plus everything upstream. Change a value and only that module and the ones after it re-run. |
| `config.py` | Config | `load_config(yaml, overrides)` returns a dict with attribute access. `MODULE_ORDER` and the config sections per module live here. |
| `video.py` | Decoding | `video_meta`, `iter_batches`, `iter_frames`, `read_frame`, plus the optional RAM cache |
| `camera.py` | M1 + M1H | `estimate_intrinsics` (GeoCalib, then fallbacks); `estimate_head_rotation` (KLT + RANSAC on the background) |
| `proposals.py` | M2 | `run_proposals`: detector + enhanced pass + IoU merge |
| `tracklets.py` | M3 | `build_tracklets`: matching, coasting, crossing cuts, `split_on_handedness` |
| `probe.py` | M3P | `run_probe`: 3D samples, forearm features (`exit_y`, `arm_exit`, `toward_z`), CLIP scores |
| `assign.py` | M3A | `tracklet_features` → `score_tracklet` (hand and side log-odds, reasons, close-work rule) → `split_at_overlaps` → `solve_labels` → `build_side_tracks`. Writes `tracklets.csv`. |
| `recon.py` | M4 | `run_recon`: WiLoR on perspective-corrected crops; probe frames reused; gives joints + MANO vertices |
| `sequence.py` | M5 | `run_sequence`: range checks, outliers, smoothing, infill, out-of-view estimates |
| `quality.py` | MQ | `assess_clip`: lanes, segments, score, `quality.json` |
| `scene.py` | Browser scene | `scene_metadata`: head quaternions, gravity, hand centre |
| `export.py` | M6 data | `export_jsonl`, `export_robot_csv`, `export_mesh`, `stability_stats` |
| `render.py` | M6 video | `render`: side-by-side video with a 3D panel; `_Writer` streams frames into ffmpeg/x264 |
| `geometry.py` | Maths | Rays, rotations, IoU and similar helpers |
| `evaluate.py` | Accuracy | Scoring against hand-labelled frames |

**`ps4v4/models/`:** thin wrappers, so the pipeline doesn't depend on any one model.

| File | Wraps |
|---|---|
| `detector.py` | WiLoR YOLO hand detector |
| `reconstructor.py` | WiLoR 3D hand (MANO) |
| `verifier.py` | CLIP "is this a hand?" check |
| `wilor_loader.py` | Loads WiLoR once; uses a fine-tuned checkpoint from `weights/` if present |
| `segmenter.py` | Optional segmentation |
| `fake.py` | Stand-ins so the tests run without a GPU or downloads |

**`code/configs/`:**
- `default.yaml` holds every threshold, one section per module.
- `profiles/fast.yaml` holds the dashboard speed settings.

**`code/run.py`:** the command line for PS4. Usage: `--clip <video> --set key=value --from M3A`.

### 2.2 `code/ps5world/`: SLAM world view

| File | Job |
|---|---|
| `runner.py` | `run_world(clip, ps4_folder, out, backend, …)`: loads the PS4 results, trims them to the processed length, runs SLAM through the retry ladder (`SLAM_ATTEMPTS`), builds the world, then writes the exports, `world_frame.json` and stats |
| `backends.py` | `run_vipe` (ViPE with our hand-box mask processor, mask guard, robust and fixed-intrinsics options); `read_vipe` (poses, intrinsics and map, with NaN checks); `rotation_only` fallback |
| `world.py` | `relift_hands` (joints + mesh into the SLAM camera); gravity estimate (GeoCalib + RANSAC planes); `build_world` (room frame, body axis, gaze); `checks` |
| `cameras.py` | `MeiCamera` (ViPE's fisheye model) and `EquidistantCamera` (PS4's), with `project` / `rays` |
| `ps4_io.py` | Reads the PS4 folder: `load_hands`, `load_mesh`, `trim_hands`, `hand_boxes`, `load_head_rotation` |
| `render_world.py` | `world_render.mp4`: frame + 3D room view with shaded hand meshes |
| `export.py` | `world_hands.jsonl`, `world_hands_mesh.npz`, `trajectory.csv`, Rerun `scene.rrd` |

**Command line:**
- `run_slam.py`: clips folder + PS4 results folder.
- `run_world.py`: one clip.

### 2.3 `code/room/`: room twin

| File | Job |
|---|---|
| `views.py` | `VirtualViews`: cuts 5 pinhole views from a fisheye frame (remap tables, lens-validity mask) |
| `depth.py` | `MogeDepth` (MoGe-2 metric depth, batched); `CallableDepth` (test stand-in) |
| `build.py` | `collect_clip` (keyframes, masks, depth scaled to the SLAM map, cleaning); `register` (clip → target room by yaw sweep + ICP + scale); `fuse` (TSDF VoxelBlockGrid); `clean_mesh`; `build_room` (the whole flow + `room_stats.json`) |
| `render_video.py` | `load_motion`, `camera_path`, `RoomRenderer` (Open3D ray casting of room + hand meshes), `render_room_video` |
| `web.py` | `export_web`: `room.glb`, `motion.json`, `hands_*.bin`, the viewer and bundled three.js |
| `viewer.html` | Standalone viewer: orbit, pan, presets, slider, robot toggle |

**Command line:** `run_room.py`, with `--steps build render web`.

### 2.4 `code/robot/`: robot replay

**`retarget.py`:**

| Function | Job |
|---|---|
| `get_panda` | Sparse download of the Panda model |
| `build_model` | Two arms + TCP sites |
| `load_world` | Reads `world_hands.jsonl` |
| `hand_targets` | Tool pose and gripper opening from the hand joints |
| `place_bases` | Where the two arm bases go |
| `retarget` | mink IK per frame + grades + summary |
| `export_web` | `robot.glb` (link meshes), `robot_motion.bin` (link poses), `robot.json` |

**Command line:** `run_robot.py --world world_hands.jsonl --web <room web folder>`.

### 2.5 `code/server/`: GPU API (runs on Colab)

- **`app.py` (FastAPI):** jobs stored as `job.json`; one GPU worker thread; a separate render thread.
- **Endpoints:**

  | Method | Path |
  |---|---|
  | GET | `/health` |
  | POST | `/jobs` (upload; `profile`, `render`) |
  | GET | `/jobs`, `/jobs/{id}`, `/jobs/{id}/result`, `/jobs/{id}/files/{name}` (Range requests) |
  | POST | `/jobs/{id}/render` |
  | DELETE | `/jobs/{id}` |

- **Also:** warm-up at start, TTL sweep of old jobs, API-key auth.
- **`activity.py`:** counts real transfers, so idle shutdown never interrupts an upload or download.

### 2.6 `code/tools/` and `code/notebooks/`

**Tools:**

| File | Role |
|---|---|
| `colab_start.py` | Cached install and start inside Colab |
| `colab_session.py` | Session management |
| `idle_guard.py` | Releases the GPU after 5 idle minutes |
| `bench_profiles.py` | Fast vs full profile: timing + agreement |
| `api_client.py` | Command-line client for the API |
| `create_demo.py` | Demo data |
| `patch_chumpy.py` | Fix for the old `chumpy` dependency |
| `sample_label_frames.py` | Frames for hand labelling |

**Notebooks:**

| Notebook | Purpose |
|---|---|
| `ps4v4_colab_all_clips.ipynb` | Batch PS4 over two accounts; `MESH_FOR` mode |
| `ps4v4_closework_colab.ipynb` | Close-work clips |
| `ps4v4_colab.ipynb` | A few test clips |
| `ps5_world_slam.ipynb` | SLAM |
| `room_twin_colab.ipynb` | SLAM + room + video + web |
| `ps4_api_server_colab.ipynb` | Manual API server |

### 2.7 `code/tests/`

Each stage has a synthetic test with known ground truth, so the logic can be checked on any computer:

| Test | Covers |
|---|---|
| `test_synthetic` | PS4 end to end |
| `test_assign_overlap`, `test_side_cue` | M3A decisions |
| `test_empty_clip` | A clip with no hands |
| `test_scene_video` | `scene.json` |
| `test_world_synthetic` | SLAM geometry, retries, partial clips, meshes |
| `test_room_synthetic` | Room accuracy, registration, video, web, CLI |
| `test_robot_synthetic` | Grades and viewer |
| `test_api`, `test_gateway`, `test_connect_notebook` | Server, gateway, Colab notebook |
| `test_browser.mjs` | Website UI |

### 2.8 `dashboard/`: the website

**`backend/` (FastAPI, runs on your laptop):**

| File | Job |
|---|---|
| `app.py` | Creates the app; metrics, evidence and training routes; serves the built frontend |
| `routes.py` | Colab status/URL, workspaces (upload, retry, CPU preview, clip, video, exports, downloads, delete), delete all data |
| `workspace.py` | Persistent queue: transcode → upload → poll → fetch → build clip → background render → delete the remote job |
| `colab.py` | Tunnel discovery from `.env` settings; authenticated streaming client; the key stays server-side |
| `media.py` | ffmpeg transcode |
| `rooms.py` | Room assets, original video, render, stats |
| `dataset.py` | Rendered-video catalogue |
| `clips.py` | Turns pipeline results into compact clip JSON + browser video |
| `jobs.py` | Offline CPU preview |

**`frontend/src/` (React + TypeScript + Tailwind + react-three-fiber):**

| File | Job |
|---|---|
| `App.tsx` | Shell and pages |
| `api.ts`, `workspaceApi.ts` | API calls |
| `Landing.tsx`, `Showcase.tsx`, `components/Story.tsx`, `components/Sections.tsx` | Project story pages |
| `components/WorkspacePlayer.tsx`, `Workspace3D.tsx`, `Timeline.tsx`, `VideoOverlay.tsx`, `HandViewer3D.tsx`, `ClipPlayer.tsx` | Workspace: synced video + 3D hands + quality lanes |
| `components/RoomPage.tsx`, `RoomViewport.tsx`, `RoomScene.tsx`, `roomData.ts` | Room: room mesh, hand meshes, head and gaze, cameras, shared slider |
| `components/RobotReplay.tsx` | Robot viewport, grade lanes, scores |
| `components/Dataset.tsx` | Dataset page |
| `src/__fixtures__/` | A synthetic room used by tests |

**Other folders:**
- `dashboard/results/`, `raw_clips/` and `data/` are local only and gitignored.
- `.env` holds the private connection settings; `.env.example` is the template.

### 2.9 Repo-level `tools/`

| File | Job |
|---|---|
| `prepare_connection.py` | Creates the private `dashboard/.env` and `SpectatorAI-Colab.zip` with matching credentials |
| `build_connect_notebook.py` | Builds `CONNECT_COLAB.ipynb` |
| `generate_room_fixture.py` | Builds the synthetic room for frontend tests |

---

## Part 3: How to read the code (suggested order)

1. `code/configs/default.yaml`: every knob, grouped by module.
2. `code/ps4v4/pipeline.py`: the whole PS4 flow in about 140 lines.
3. `code/ps4v4/assign.py`: where the important decisions are made.
4. `code/ps5world/runner.py` → `world.py`: how hands enter the room.
5. `code/room/build.py`, then `code/robot/retarget.py`.
6. `code/server/app.py` → `dashboard/backend/workspace.py` → `dashboard/frontend/src/components/RoomPage.tsx`.

## Part 4: Where outputs live

| Output | Folder | Key files |
|---|---|---|
| PS4 per clip | `<results>/<clip>/` | `hands.jsonl`, `hands_mesh.npz`, `quality.json`, `stats.json`, `tracklets.csv`, `render.mp4`, `robot_wrist_gripper.csv`, caches (`*.pkl`) |
| SLAM per clip | `<ps5_world>/<clip>/` | `world_hands.jsonl`, `world_hands_mesh.npz`, `world_frame.json`, `world_stats.json`, `trajectory.csv`, `world_render.mp4`, `scene.rrd` |
| Room / robot | `dashboard/results/room_twin/<clip>/` | `room_stats.json`, `room_video.mp4`, `web/` (`room.glb`, `motion.json`, `hands_*.bin`, `robot.*`) |
| Workspaces | `dashboard/data/workspaces/<id>/` | Gitignored |
