# SpectatorAI: final technical documentation

> Egocentric factory video → the wearer's metric 3D hands → a usable/unusable rating → a room-fixed world (SLAM)
> → a 3D workstation twin → a robot replay. All of it is viewable in a local web dashboard.

| | |
|---|---|
| Repository | `ps4-hand-tracking/` (`code/` = pipelines, `dashboard/` = website, `tools/` = setup helpers) |
| Compute | Google Colab A100 for the GPU stages; a laptop for the dashboard and the robot step |
| Code versions | PS4 v4.5 (hand tracking + MQ), PS5 world 1.4 (SLAM), room 1.0, robot 1.0 |

---

## 1. Problem and goal

Robots can learn manipulation from video of people working, and head-mounted ("egocentric") cameras are the cheapest way to record it. Raw factory head-cam video has seven problems:

1. **Other people's hands** are in view.
2. **Hand look-alikes:** knees, feet, gloves, cloth, wood and tools get detected as hands.
3. **Left/right swaps** when the hands cross or overlap.
4. **A fisheye lens:** strong distortion where the hands usually are.
5. **Hands leaving the frame**, often for seconds at a time.
6. **No fixed reference frame:** everything is relative to a moving head.
7. **Unusable stretches** (blur, fast head turns, walking) that should not be used for training.

**Goal:** a pipeline that outputs **only the wearer's hands**, in **metres**, with **correct sides**, **labelled guesses**, a **per-second usability rating**, and a **room-fixed world** that can drive a **robot**.

---

## 2. System architecture

```
                         ┌─────────────────────── Google Colab (A100) ───────────────────────┐
  head-cam video ──────► │ 1  PS4 v4 hand tracking   (ps4v4/, run.py)                        │
                         │ 2  MQ clip rating         (ps4v4/quality.py)                      │
                         │ 3  PS5 SLAM world view    (ps5world/, run_slam.py)                │
                         │ 4  Room twin              (room/, run_room.py)                    │
                         │    API server             (server/app.py, FastAPI + tunnel)       │
                         └──────────────┬──────────────────────────────────────────┬─────────┘
                                        │ HTTPS tunnel (API key)                   │ Drive files
                         ┌──────────────▼──────────────┐              ┌───────────▼───────────┐
                         │ Local gateway (FastAPI)     │              │ 5 Robot replay        │
                         │ dashboard/backend           │◄─────────────┤ robot/, run_robot.py  │
                         │ transcode, queue, cache     │  web files   │ (laptop CPU, seconds) │
                         └──────────────┬──────────────┘              └───────────────────────┘
                                        ▼
                         React + three.js website (dashboard/frontend):
                         Workspace · Library · Dataset · Room (+ robot replay) · Showcase
```

| Stage | Input | Output | Where it runs | Time |
|---|---|---|---|---|
| 1 PS4 | video | `hands.jsonl`, `hands_mesh.npz`, robot CSV, `render.mp4`, `tracklets.csv` | Colab GPU | ~25 min per 3-min clip (4 in parallel); fast profile ~1–2 min per 30 s |
| 2 MQ | stage 1 caches | `quality.json` | Colab (CPU part) | seconds |
| 3 SLAM | video + stage 1 | `world_hands.jsonl`, `world_hands_mesh.npz`, `world_frame.json`, … | Colab GPU | ~7 min per 3-min clip |
| 4 Room | 2 clips + stage 3 | `room.glb`, `room_video.mp4`, `web/` | Colab GPU | ~15 min + video |
| 5 Robot | `world_hands.jsonl` | `robot.glb`, `robot_motion.bin`, grades | laptop | ~5–10 s |

---

## 3. Stage 1: PS4 v4 hand tracking

### 3.1 Principle: track first, decide later

Older versions decided "hand? left or right?" on every frame, so junk became hands and sides swapped. v4 works in four steps:

1. Keep **weak** detections.
2. Link them into **tracklets** without deciding anything.
3. **Probe** each tracklet in 3D and with a semantic check.
4. Decide **once per tracklet** from all the evidence together.

Every decision is logged with its reasons in `tracklets.csv`.

### 3.2 Modules (`code/ps4v4/`)

| ID | File | Purpose | Key method |
|---|---|---|---|
| M1 | `camera.py` | Intrinsics | GeoCalib estimates the focal length on a few frames; equidistant fisheye model. Fallbacks: calibration JSON, then a 150° field-of-view guess. GeoCalib gravity samples are kept for the scene's up direction. |
| M2 | `proposals.py` | Hand boxes | WiLoR's YOLO detector at score ≥ 0.15. A CLAHE/gamma "enhanced" pass runs on dark or difficult frames; the passes are merged by IoU clustering. |
| M1H | `camera.py` | Head rotation | KLT features on the background (hand boxes masked), then RANSAC Kabsch on fisheye rays, then smoothing. Gives camera_t → frame-0 rotation. |
| M3 | `tracklets.py` | Tracklets | ByteTrack-style matching with a speed-aware gate; head-rotation compensation; optical-flow coasting ≤ 0.5 s. **Cut at overlaps** (crossings), **split on sustained handedness flips** (15-detection runs, ≥ 85%). |
| M3P | `probe.py` | Evidence | WiLoR 3D on every 3rd frame (≤ 40 per tracklet). Features: wrist distance, forearm exit point on the image border, forearm direction (`toward_z`), keypoint/box IoU. CLIP crop score: hand vs knee/foot/wood/head/tool. |
| M3A | `assign.py` | Decide | Hand log-odds and side log-odds per frame, then a global labelling (§3.3) |
| M4 | `recon.py` | 3D hands | WiLoR on **perspective-corrected virtual crops** per hand (fisheye-correct). Probe frames are reused. Gives 21 joints + MANO vertices. |
| M5 | `sequence.py` | Clean-up | Range checks relative to the wearer's reach (`z_ref`), Hampel outliers, speed-adaptive smoothing, infill ≤ 0.5 s inside a segment, **out-of-view estimate ≤ 1.5 s** (labelled `estimated`). Works in a head-stable frame. |
| M6 | `export.py`, `render.py` | Outputs | `hands.jsonl`, `robot_wrist_gripper.csv`, `hands_mesh.npz`, `stats.json`, `render.mp4` (x264, 3D panel following the hands with trails; rejected tracks drawn grey with their reason) |
| — | `scene.py` | Browser scene | `scene.json`: head quaternions per frame, gravity in the reference frame, hand centre (for the dashboard's 3D workspace) |
| MQ | `quality.py` | Clip rating | §4 |

**Infrastructure:**

| File | Role |
|---|---|
| `pipeline.py` | `run_clip`: runs the modules in order, with a progress callback |
| `cache.py` | Per-module results, keyed by a hash of the config section plus everything upstream |
| `config.py` | YAML config with attribute access |
| `video.py` | Batched decoding; optional RAM frame cache |
| `geometry.py` | Rays, rotations, IoU |
| `models/` | Model wrappers: `detector.py`, `reconstructor.py`, `verifier.py` (CLIP), `wilor_loader.py` (loads fine-tuned weights from `weights/` when present), `fake.py` (test stand-ins) |

### 3.3 How M3A decides

**Hand score (per frame, log-odds), averaged over the tracklet:**

| Term | Effect |
|---|---|
| Detector score | `+4 × (score − 0.5)` |
| **Far** | wrist distance > 1.9 × `z_ref` → another person (`z_ref` = 35th percentile of confident tracks, so the wearer's own hands set it) |
| **Near** | < 0.30 × `z_ref` or < 6 cm → junk filling the lens |
| **Arm away** | forearm running away from the camera → a person facing the wearer |
| **Top exit** | forearm leaving through the top 10% of the image → not the wearer's arm |
| **On a forearm** | box lying on a stronger hand's forearm or elbow |
| **Static** | still for its whole life → furniture (kept mild) |
| **Poor fit** | keypoints don't match the box (mild) |
| **CLIP veto** | penalty only when the median P(hand) < 0.3. CLIP never accepts a track by itself, because another person's hand is still a hand. A small bonus applies only when the geometry already says "the wearer's hand". |

**Side score:** the forearm's exit point (left edge or bottom-left vs right), plus the detector's handedness (counted less when the 3D fit failed), plus a weak position prior that is ignored inside crossings.

**Labelling:**
- at most one left and one right per frame;
- pieces of one cut tracklet keep the same side;
- motion links between tracklets prefer the same side but can never turn junk into a hand.

**Close-work mode** (`assign.close_work: true`, for hands near the lens or long reaches): when the forearm leaves through the **bottom** of the image, which only the wearer's own arm does, the "arm away" penalty is skipped and the far/near limits widen (3× / 0.12×). The notebook `ps4v4_closework_colab.ipynb` pairs it with `sequence.rel_range = [0.12, 3.2]`, faster re-acquisition and larger allowed boxes.

### 3.4 Versions

| Version | Change |
|---|---|
| v2 | Per-frame decisions |
| v3 | Track-first design |
| v4.0 | Per-track decisions |
| v4.1 | Cut tracks at overlaps |
| v4.2 | CLIP veto and forearm-exit side cue |
| v4.3 | Render upgrades |
| **v4.4** | CLIP veto-only, on-forearm penalty, handedness split, sibling sides. **Ran all 40 clips; checked by hand.** |
| v4.5 | MQ rating, mesh export, scene export, close-work mode (all additive) |

### 3.5 Profiles

- **`configs/default.yaml`:** full quality; used for the 40-clip batch.
- **`configs/profiles/fast.yaml`:** used by the dashboard. RAM frame cache, 3 GeoCalib frames, detector at 960 px, enhanced pass only on dark frames, sparser probe, larger batches, render deferred. The decision logic is unchanged.

---

## 4. Stage 2: MQ clip rating (usable vs unusable)

**Lanes per frame.** Each lane is merged into segments, and segments under 0.3 s are absorbed by their longer neighbour.

| Lane | Labels | Signal and threshold |
|---|---|---|
| left / right hand | visible, partial, low_confidence, gap_filled, out_of_view | Tracking state; < 70% of joints in the image = partial; 3D confidence < 0.25 = low |
| image | sharp, blurred | Laplacian variance at ¼ resolution < 40% of the clip's 75th percentile, or below an absolute floor |
| head | stable, fast_turn | Head speed from M1H > 90°/s |
| body | still, moving | Background flow not explained by head rotation > 0.4% of the image diagonal per frame |
| **usable** | usable / unusable + reasons | A hand visible or partial **and** sharp **and** head stable **and** body still |

**Score:** `100 × (0.30 visibility + 0.15 both hands + 0.20 clarity + 0.20 stability + 0.15 confidence)`. Grades: A ≥ 80, B ≥ 65, C ≥ 50, else D.

**`quality.json` contains:**
- score, grade and sub-scores;
- usable share, seconds, longest segment and the segment list;
- per-hand state shares;
- seconds lost per issue;
- per-lane segments with reasons;
- 10 Hz arrays for charts;
- the thresholds used.

---

## 5. Stage 3: PS5 SLAM world view (`code/ps5world/`)

**Why it is needed:** PS4 removes head *rotation* only. Leaning, shifting or standing still moves the hands in its output. SLAM recovers the head's full 6-DoF pose, metric scale and gravity, so hands land in a **fixed room**.

1. **Camera tracking (`backends.py`):**
   - NVIDIA **ViPE**, `wide_angle` preset (unified/Mei camera); learned dense bundle adjustment with metric depth priors.
   - Fed a 768 px copy at 15 fps; poses interpolated to every frame (Slerp + linear).
2. **Masks:**
   - ViPE's GroundingDINO + SAM masks for "person, hand, arm", plus PS4's hand boxes (grown) with a strip to the bottom edge for the forearm.
   - A **mask guard** keeps ≥ 30% static scene: it drops the forearm strips first, then falls back to ViPE's masks alone.
3. **Hands into the SLAM camera (`world.py`):**
   - each hand keeps PS4's shape;
   - the wrist is re-aimed along ViPE's ray for the same pixel, and its distance is scaled by the focal ratio `f_SLAM / f_PS4`, which fixes GeoCalib's focal error;
   - the MANO mesh gets the identical rigid move.
4. **Gravity and room frame:**
   - GeoCalib gravity, refined by RANSAC planes of the SLAM map (bench or floor); fallback = the head's average up direction.
   - Frame: **Z up**, Y = average facing, origin = first head position. Saved to `world_frame.json` (align + origin).
5. **Body:** neck 0.22 m and pelvis 0.62 m under the head; facing = heading smoothed over about 1 s; gaze = camera +z.
6. **Robustness (v1.4):**
   - retry ladder: refine intrinsics (Huber kernel, damped intrinsics) → fixed intrinsics → fixed intrinsics at 10 fps → rotation-only fallback;
   - NaN poses or intrinsics count as a failure, and every attempt is recorded in `world_stats.json → slam_attempts`;
   - `--max-seconds` trims all per-frame arrays consistently.
7. **Outputs (`export.py`, `render_world.py`):**
   - `world_render.mp4`: the frame next to the room view, with the head frustum, gaze, body axis, **shaded hand meshes** and trails;
   - `world_hands.jsonl`, `world_hands_mesh.npz`, `trajectory.csv`, `scene.rrd` (Rerun), `world_stats.json`.

**Verified:**
- On a synthetic scene: resting-hand drift 0.6 cm; hand error < 1 cm with the PS4 focal 30% off; gravity error 0°; head path exact.
- On real footage: ViPE succeeded on `MTL-SKL00089-006` and `-007`.

---

## 6. Stage 4: Room twin (`code/room/`)

**Constraint:** the seated clips come from one viewpoint with almost no parallax, so photogrammetry or plain Gaussian splatting would smear. Geometry therefore comes from **metric depth scaled to SLAM**.

1. **`views.py`:** each keyframe (2 fps, blurred frames dropped) becomes 5 virtual pinhole views (centre, left, right, up, down; 90°, 384 px), using the SLAM Mei camera. Lens-validity check by ray round trip.
2. **`depth.py`:** **MoGe-2** (metric) depth per view, batched on the GPU.
3. **`build.py`:**
   - **Scale:** per keyframe, the median of SLAM-map depth / predicted depth, clamped to [0.8, 1.25] × the clip median.
   - **Cleaning:** depth edges (flying pixels), PS4 hand boxes + forearm strip, the wearer's body column under the head, and anything < 0.2 m or > 3.5 m.
   - **Registration:** clip 006 → clip 007 room frame by yaw sweep + point-to-plane ICP + a scale-refinement ICP. Accepted if fitness ≥ 0.3, RMSE ≤ 2 cm and tilt ≤ 5°; otherwise the target clip is used alone.
   - **Fusion:** Open3D VoxelBlockGrid TSDF (1.2 cm voxels, 5 cm truncation), then a mesh, cropped to 3 m, with small clusters removed and decimated to 500k triangles for the web.
4. **`render_video.py`:**
   - CPU ray casting (Open3D/Embree); the room's vertex colours are unlit and the hands are shaded MANO meshes;
   - head frustum, gaze, trails and the original video as an inset;
   - camera path: over the shoulder → left → right → top → close-up → worker's eye → back.
5. **`web.py` + `viewer.html`:**
   - `room.glb`, `motion.json` (head, quaternions, gaze, joints, sources), `hands_*.bin` (int16 vertex offsets at 0.05 mm);
   - a standalone three.js viewer with bundled three.js, so it works offline.

**Verified on a synthetic two-clip workstation:**
- registration recovered a 20° yaw exactly;
- depth scale corrected from 1.15× back to 1.0;
- mesh error 0.35 cm median / 0.94 cm at the 90th percentile;
- bench height within 0.4 cm.

---

## 7. Stage 5: Robot replay (`code/robot/retarget.py`)

**What it is:** kinematic retargeting, with **no training**. It runs on a laptop CPU.

1. **Model:**
   - two Franka Panda arms from **MuJoCo Menagerie** (sparse git download of the Panda folder, about 36 MB);
   - assembled with `MjSpec.attach` under the prefixes `left/` and `right/`;
   - a `tcp` site between the fingertips.
2. **Targets from `world_hands.jsonl`:**
   - tool position = midway between the palm centre and the thumb–index pinch point;
   - tool z axis = wrist → middle knuckle; tool y axis = across the knuckles;
   - gripper opening = thumb–index distance;
   - smoothed within tracked runs.
3. **Bases:** beside and slightly behind the head (±0.25 m), facing from the head toward the work area, 0.15 m below the median work height.
4. **IK:**
   - **mink** differential IK: frame tasks with position cost 5 and orientation cost 0.25, a posture task, configuration limits, the DAQP solver;
   - 6 iterations per frame, warm-started from the previous frame.
5. **Grades per frame:**
   - 1 = copyable (position ≤ 3 cm, orientation ≤ 35°);
   - 2 = out of reach;
   - 3 = too fast (exceeds Panda joint speed limits);
   - 0 = no hand.
6. **Web export:**
   - `robot.glb` (one node per link, built from the MJCF visual meshes);
   - `robot_motion.bin` (float32 `[n][links][7]` world poses);
   - `robot.json` (link names, grades, summary) and `robot_traj.json` (joint angles).

**Result on `MTL-SKL00089-007` (2 min):** 79% robot-copyable (left 75%, right 83%), mostly limited by fast wrist motion; median position error ≈ 0 cm.

---

## 8. Dashboard (`dashboard/`)

**Gateway (`backend/`):**

| File | Role |
|---|---|
| `colab.py` | Discovers the Colab tunnel URL, holds the API key **server-side**, streams uploads and downloads |
| `workspace.py` | Persistent job queue per uploaded video: transcode (`media.py`, ≤ 720p, ≤ 30 fps, ≤ 60 s) → upload → poll → fetch results → background render → delete the remote job |
| `routes.py` | `/api/colab/*`, `/api/workspaces/*` (upload, status, clip, video, exports, delete), `/api/data` |
| `rooms.py` | `/api/rooms/*`: room web files, original video, render, stats |
| `dataset.py` | Rendered-video catalogue |
| `clips.py` | Adapter from `hands.jsonl` to compact clip JSON + browser video |
| `jobs.py` | Offline CPU preview |
| `app.py` | Metrics, evidence and training routes; serves the built frontend |

**Colab side:**
- `CONNECT_COLAB.ipynb` (one cell) runs `tools/colab_start.py` (cached install) → `server/app.py` + tunnel;
- `idle_guard.py` releases the runtime after 5 min without jobs, renders or transfers (`server/activity.py` counts transfers).

**Frontend (`frontend/src/`):** React + TypeScript + Tailwind + react-three-fiber.

| Page / component | Content |
|---|---|
| Landing / Showcase | `Story.tsx`, `Sections.tsx` |
| Workspace | `WorkspacePlayer.tsx`, `Workspace3D.tsx`, `Timeline.tsx` quality lanes, `VideoOverlay.tsx` |
| Dataset | `Dataset.tsx` |
| Room | `RoomPage.tsx`, `RoomViewport.tsx`, `RoomScene.tsx`, `roomData.ts`: room + hand meshes + head/gaze, orbit/pan/rotate, presets, one shared slider, original-video overlay |
| Robot replay | `RobotReplay.tsx`: second viewport, shared timeline, grade lanes, scores, downloads |

---

## 9. Coordinate frames

| Frame | Axes | Used by |
|---|---|---|
| Camera (OpenCV) | x right, y down, z forward | `kp3d_cam`, all projections |
| Head-stable (PS4) | camera at frame 0, head rotation removed | `kp3d_head`, Workspace 3D, `scene.json` |
| SLAM (ViPE) | first camera frame, metric | internal |
| **Room** | **Z up**, Y = facing, origin = first head position, metres | `world_*`, room twin, robot, Room page |

The conversion is `p_room = (p_slam − origin) · alignᵀ`, using the values in `world_frame.json`.

---

## 10. Testing and validation

All tests run without a GPU:

| Test | Checks |
|---|---|
| `test_synthetic` | End to end; recall 98.8%, side accuracy 100%, 0 false hands |
| `test_assign_overlap`, `test_side_cue` | Decision regressions |
| `test_world_synthetic` | SLAM geometry, retries, partial clips, meshes |
| `test_room_synthetic` | Geometry accuracy, registration, video, web, CLI, cache restore |
| `test_robot_synthetic` | Grades and viewer |
| `test_api`, `test_gateway`, `test_connect_notebook`, `test_empty_clip`, `test_scene_video`, `test_browser.mjs` | Server, gateway, notebook and UI |

**Manual:** v4.4 checked clip by clip on the hard cases (other people's hands, knees and feet, forearms, crossings).

**Not yet measured:** accuracy against hand-labelled real frames, and calibration of the MQ thresholds on real clips.

---

## 11. Innovations

1. Track-first, per-tracklet decisions with crossing and handedness cuts.
2. Wearer-only filtering by geometry; CLIP as a veto only; close-work bottom-exit rule.
3. Fisheye-correct 3D via virtual perspective crops.
4. Honest gaps: every guess is labelled and never counted as measured.
5. A clip-quality rating with timestamped reasons, for data curation.
6. SLAM with hands masked out, then re-lifted with their meshes into a gravity-aligned room, plus a retry/fallback ladder.
7. A room twin from single-seat video: depth scaled to SLAM, multi-clip registration, TSDF fusion.
8. Robot replay with per-frame copyability grades (an "Automation Scout" metric), with no training.
9. Engineering: hash-keyed module caching; offline re-tuning; synthetic tests for every stage; one-cell Colab connection with idle release; a credentials-safe local gateway.

## 12. Limitations

- **Room twin:** realistic only near the seat; never-filmed areas stay empty. Gaussian-splat appearance is future work.
- **Robot:** kinematic only (no contact physics); bases are placed heuristically.
- **Out of view:** hands outside the frame are estimated for at most 1.5 s.
- **Real-footage accuracy:** to be quantified with labelled frames.

## 13. Data and credits

**Data rules:**
- Footage stays private. No media, results, weights or credentials are in git.
- All copies are deleted after the event.

**Credits:**
- **Pretrained models:** WiLoR (detector + MANO reconstruction), OpenAI CLIP, GeoCalib, NVIDIA ViPE (with GroundingDINO, SAM, UniDepth), MoGe-2.
- **Simulation:** MuJoCo, MuJoCo Menagerie (Franka Panda), mink.
- **Libraries:** Open3D, trimesh, three.js, Rerun, FastAPI, React.
