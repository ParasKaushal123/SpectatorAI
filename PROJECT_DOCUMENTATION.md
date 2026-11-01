# SpectatorAI: full project documentation

Egocentric 3D hand tracking, a room-fixed world view (SLAM), a 3D workstation twin, and clip-quality rating, for robot learning from factory head-cam video.

*Code: `ps4-hand-tracking/code/`. Code versions: PS4 v4.5, PS5 world 1.4, room twin 1.0.*

---

## 1. Problem

Robots can learn manipulation by watching people work, and head-mounted cameras are the cheapest way to record that work. Raw egocentric factory footage is hard to use, for these reasons:

| Problem | Effect on existing hand models |
|---|---|
| Co-workers' hands in view | They get tracked as the wearer's hands |
| Knees, feet, gloves, cloth, wood and tools | They get detected as hands |
| Hands crossing or overlapping | Frame-by-frame trackers swap left and right |
| Fisheye lens | The distorted image edges, where hands often are, break 3D models trained on normal photos |
| Hands leaving the frame | They are dropped, or their positions are invented without saying so |
| A moving head | Hand positions relative to the head mean nothing to a robot in a fixed workspace |
| Blur, fast head turns, walking | Nobody says which seconds are usable for training |

## 2. What we built

| Part | Code | Output |
|---|---|---|
| **PS4 v4: 3D hand tracking** | `ps4v4/`, `run.py` | Both of the wearer's hands, 21 joints + MANO mesh per frame, in metres, left/right always correct, every guess labelled |
| **MQ: clip quality rating** | `ps4v4/quality.py` | Usable / unusable timeline with reasons, score 0–100, grade A–D |
| **PS5: world view (SLAM)** | `ps5world/`, `run_slam.py` | Head (camera), gaze, body axis and hands in one gravity-aligned room frame |
| **Room twin** | `room/`, `run_room.py` | 3D copy of the workstation, 2-min fly-through, browser viewer with a time slider |
| **API + dashboard** | `server/app.py`, `DASHBOARD_PLAN.md` | Upload a clip in a local dashboard; a Colab A100 processes it; results come back |

---

## 3. PS4 v4: 3D hand tracking architecture

### 3.1 Design principle: track first, decide later

Earlier versions (v2, v3) decided "is this a hand, and which side?" frame by frame, which made anything look like a hand and caused side swaps.

v4 works differently:

1. Collect weak detections.
2. Link them into short tracks without deciding anything.
3. Gather 3D and semantic evidence for each track.
4. Decide LEFT / RIGHT / REJECT once per track, using all of that evidence together.

Every decision is written down with its reasons.

### 3.2 Modules

```
video ─► M1 camera ─► M2 proposals ─► M1H head rotation ─► M3 tracklets ─► M3P probe ─► M3A assign
      ─► M4 3D hands ─► M5 sequence ─► M6 outputs ─► MQ quality
```

| Module | File | What it does |
|---|---|---|
| **M1 Camera** | `camera.py` | Focal length from GeoCalib on a few frames; equidistant fisheye model. Falls back to a calibration file, then to a 150° field-of-view guess. |
| **M2 Proposals** | `proposals.py` | WiLoR YOLO hand detector, keeping boxes down to score **0.15** (weak boxes are decided later, not dropped). A CLAHE/gamma enhanced pass runs on dark or difficult frames, and the two passes are merged by IoU. |
| **M1H Head rotation** | `camera.py` | KLT features on the background (hand boxes masked out), then RANSAC rotation (Kabsch on fisheye rays), smoothed. Gives camera_t → frame-0 rotation for every frame. |
| **M3 Tracklets** | `tracklets.py` | ByteTrack-style linking with a speed-aware gate and optical-flow coasting, compensated for head rotation. Tracks are **cut where two tracks overlap** (crossings) and **where the detector's left/right call flips and stays flipped** (15 detections, ≥85%), which catches a track that slid from one hand onto the other. No side or hand/not-hand decision is made here. |
| **M3P Probe** | `probe.py` | WiLoR 3D on every 3rd frame of each track. Evidence collected: wrist distance, forearm exit point on the image border, forearm direction toward or away from the camera, keypoint-vs-box fit, and CLIP "hand vs knee / foot / wood / head / tool". |
| **M3A Assign** | `assign.py` | Labels each track LEFT / RIGHT / REJECT (see 3.3). Writes `tracklets.csv` with every decision and its reasons. |
| **M4 3D hands** | `recon.py` | WiLoR on **perspective-corrected virtual crops** cut from the fisheye image around each hand, so a pinhole-trained model works at the lens edge. Probe frames are reused. Outputs joints + MANO vertices. |
| **M5 Sequence** | `sequence.py` | Depth sanity checks relative to the wearer's reach, Hampel outlier removal, smoothing that adapts to speed, gap infill inside one continuous segment (≤ 0.5 s), and **labelled out-of-view estimates** for gaps ≤ 1.5 s. Works in a head-stable frame. |
| **M6 Outputs** | `export.py`, `render.py` | `hands.jsonl`, `robot_wrist_gripper.csv`, `hands_mesh.npz`, `stats.json`, and `render.mp4`: 720p, x264, a 3D panel that follows the hands with wrist light trails, and rejected candidates drawn in grey with their reason. |
| **MQ Quality** | `quality.py` | Usable / unusable rating (section 4) |

### 3.3 How M3A decides

**Is this track the wearer's hand?** The log-odds are added up over the track's frames:

- Detector score.
- Distance relative to the wearer's typical hand distance (`z_ref` = the 35th percentile of confident tracks, so the wearer's own hands set the reference, not co-workers'). "Too far" means another person; "too near" means junk filling the lens.
- A forearm running away from the camera, or leaving through the top of the image, means a person facing the wearer.
- A penalty when the box lies on another, stronger hand's forearm or elbow.
- Static for its whole life means furniture or a slab. Kept mild, because a hand holding still is still a hand.
- Poor keypoint fit (mild).
- **CLIP can only veto.** It penalises a track only when its median P(hand) is below 0.3. CLIP never accepts a track on its own, because another person's hand is still a hand. A small bonus applies only when the geometry already says "the wearer's hand".

**Which side?**

- Mainly where the forearm leaves the image (left edge / bottom-left vs right).
- Plus the detector's handedness, which counts less when the 3D fit failed.
- Plus a weak position prior, which is ignored inside crossings.

**Global labelling:**

- At most one left and one right per frame.
- Pieces of one track (cut at an overlap) can't take opposite sides.
- Motion links between nearby tracks prefer the same side but can never turn junk into a hand.

### 3.4 Engineering

- **Caching:** every module is cached with a hash of its config section plus everything upstream. Changing one threshold re-runs only that module and the ones after it, and a Colab disconnect resumes where it stopped.
- **Offline tuning:** `tracklets.pkl` + `probe.pkl` let M3A be re-run in seconds without the video.
- **Settings:** every threshold lives in `configs/default.yaml`.
- **Tests (no GPU):** `test_synthetic.py` (end to end with stand-in models), `test_assign_overlap.py`, `test_side_cue.py`.

### 3.5 Version history

| Version | Change |
|---|---|
| v2 | Frame-by-frame decisions: anything could become a hand |
| v3 | Track-first design; failed on montage clips (GeoCalib focal) and on junk |
| v4.0 | Tracklets + per-track decisions |
| v4.1 | Cut tracks at overlaps |
| v4.2 | CLIP veto; forearm exit point as the side cue |
| v4.3 | Render: follow view, trails, x264 |
| v4.4 | CLIP veto-only, on-forearm penalty, handedness split, sibling sides. **Checked by hand as correct on the problem clips.** |
| v4.5 | MQ quality step + `hands_mesh.npz` export (additive; decisions unchanged) |

---

## 4. MQ: finding usable and unusable parts of a clip

**Purpose:** robot-learning datasets should only use seconds where the hands are clearly seen and the camera is steady. MQ marks every frame and explains why a stretch is unusable.

### 4.1 Lanes (per frame, merged into segments)

| Lane | Labels | Signal |
|---|---|---|
| Left hand / Right hand | `visible`, `partial`, `low_confidence`, `gap_filled`, `out_of_view` | Tracking state; share of joints inside the image (< 70% = partial); 3D confidence (< 0.25 = low) |
| Image | `sharp`, `blurred` | Laplacian variance at quarter resolution, relative to this clip's typical sharpness (< 40% of the 75th percentile) or below an absolute floor |
| Head | `stable`, `fast_turn` | Head rotation speed from M1H > 90°/s |
| Body | `still`, `moving` | Background motion not explained by head rotation > 0.4% of the image diagonal per frame (walking, leaning, standing up) |
| **Usable** | `usable`, `unusable` + **reasons** | Usable = at least one hand visible or partial **and** sharp **and** head stable **and** body still |

Segments shorter than 0.3 s are merged into their longer neighbour, so the timeline isn't cluttered with flicker.

### 4.2 Score

```
score = 100 × (0.30 hand_visibility + 0.15 both_hands + 0.20 image_clarity + 0.20 camera_stability + 0.15 tracking_confidence)
grade: A ≥ 80, B ≥ 65, C ≥ 50, else D
```

### 4.3 Output: `quality.json`

- Score, grade and sub-scores.
- Usable share, seconds and longest segment, plus the usable segments.
- Share of each hand state.
- Seconds lost per issue (`blurred_s`, `fast_head_s`, `body_moving_s`, `left_out_of_view_s`, …).
- Timestamped segments per lane, with reasons.
- 10 Hz arrays for charts, and the thresholds used.

The thresholds are in `configs/default.yaml` → `quality:`. They should be calibrated on a few real clips.

---

## 5. PS5: world view (SLAM)

### 5.1 Purpose

PS4 gives hands relative to the head. PS5 puts the head, gaze, body and hands into **one room-fixed, gravity-aligned frame** (Z up, metres), which is what a robot workspace needs.

### 5.2 Pipeline (`ps5world/`)

```
clip + PS4 results ─► ViPE SLAM (Mei fisheye camera, people/hands masked) ─► re-lift hands into the SLAM camera
                   ─► gravity + room frame ─► body axis + gaze ─► exports
```

1. **Camera tracking:** NVIDIA ViPE, Apache-2.0. It uses learned dense bundle adjustment with metric depth priors and the `wide_angle` preset (unified/Mei camera model). It runs at 15 fps on a 768-px copy, and poses are interpolated to every frame.
2. **Hands never drive the camera:** ViPE's own person/hand/arm masks, plus PS4's hand boxes (grown, extended down to the image edge for the forearm). A **mask guard** stops the masks from covering too much of a frame (≥ 30% must stay as static scene): it drops the forearm strips first, then falls back to ViPE's masks only.
3. **Hands re-lifted into the SLAM camera:**
   - Each hand keeps PS4's shape.
   - Its wrist is re-aimed along ViPE's ray for the same pixel, and its distance is rescaled by the ratio of the two focal lengths. This also corrects GeoCalib's focal error.
   - The MANO mesh gets exactly the same rigid move.
4. **Gravity:** GeoCalib gravity, refined by the dominant horizontal plane of the SLAM map (bench or floor). Fallback: the head's average up direction.
5. **Room frame:** Z up, Y = the wearer's average facing direction, origin = the first head position. Saved as `world_frame.json`.
6. **Body axis:** neck 0.22 m and pelvis 0.62 m under the head; facing = the head's heading smoothed over about 1 s. **Gaze** = the camera's forward axis.

### 5.3 Robustness (final version 1.4)

- **Retry ladder.** If bundle adjustment diverges (energy → NaN, a singular camera matrix), the next setting runs:
  1. refined intrinsics + Huber robust kernel + damped intrinsics updates;
  2. fixed intrinsics;
  3. fixed intrinsics at 10 fps;
  4. rotation-only fallback (PS4 head rotation, head position fixed).
- Every attempt is logged in `world_stats.json → slam_attempts`.
- NaN poses or intrinsics count as a failure and are never passed on.
- `--max-seconds` correctly trims every per-frame array when the PS4 result is longer than the part processed.
- The notebook selects the Drive code zip by its **code version**, not by upload date.

### 5.4 Outputs

| File | Content |
|---|---|
| `world_render.mp4` | The frame next to a 3D room view: map points, head frustum and path, gaze ray, body axis, **shaded hand meshes**, wrist trails |
| `world_hands.jsonl` | Per frame: head pose, gaze, neck/pelvis/facing, 21 world joints per hand |
| `world_hands_mesh.npz` | MANO hand meshes per frame in the room frame (for the web) |
| `trajectory.csv` | Head, gaze, pelvis, facing yaw, both wrists |
| `scene.rrd` | Rerun 3D recording |
| `world_frame.json`, `world_stats.json` | Frame definition; backend, gravity method, hand scale factor, head path, agreement with PS4's head rotation, SLAM attempts |

**Verified:**

- On a synthetic scene with known truth: resting-hand drift 0.6 cm; hand position error < 1 cm with PS4's focal length 30% off; gravity error 0°; head path length exact.
- On real clip `MTL-SKL00089-006`: ViPE succeeded on its first setting, and the room, head, gaze and hands are consistent.

---

## 6. Room twin: 3D workstation + motion replay

### 6.1 Goal

A 3D copy of the workstation that visitors can move around in, while a slider plays the worker's hand movements forward and backward. Plus a 2-min rendered fly-through. The target is 007's first 2 minutes, built from clips 006 + 007.

### 6.2 The constraint and the approach

The clips are filmed from one seat with almost no head movement, so there is very little parallax. Photogrammetry and plain Gaussian splatting need several viewpoints and would produce floaters and smears.

The geometry therefore comes from **metric depth**, not from parallax:

1. **Keyframes:** 2 fps per clip, blurred frames dropped.
2. **Virtual pinhole views:** each fisheye frame is split into 5 views (centre, left, right, up, down; 90°, 384 px) using the SLAM camera model.
3. **Masks:** PS4 hand boxes and the forearm strip; the wearer's own body and lap (a column under the head); flying pixels at depth edges.
4. **Depth:** MoGe-2 (metric) per view, **scaled to the SLAM map** per keyframe (median ratio, clamped). Each depth map then agrees with the SLAM poses.
5. **Second clip:** registered into the target clip's room frame. Both rooms are gravity-aligned, so a yaw sweep + point-to-plane ICP + a final scale refinement is enough. Accepted only if fitness ≥ 0.3, RMSE ≤ 2 cm and tilt ≤ 5°; otherwise the target clip is used alone.
6. **TSDF fusion:** Open3D VoxelBlockGrid, 1.2 cm voxels, of all views into one coloured mesh. Then crop to the workstation (3 m), remove small fragments, and decimate to 500k triangles for the web.
7. **Motion:** `world_hands.jsonl` and `world_hands_mesh.npz` are already in the same room frame, so hands land on the reconstructed bench.

### 6.3 Outputs

| File | Content |
|---|---|
| `room_mesh.ply`, `room.glb` | The 3D room (full / web) |
| `room_video.mp4` | 2-min fly-through, 1280×720. Ray cast on the CPU (Open3D/Embree), so no OpenGL is needed on Colab. Camera path: over the shoulder → left → right → top → close-up → worker's eye → back. Shows hand meshes, head frustum, gaze, wrist trails, and the original video inset. |
| `room_web.zip` | Viewer: `viewer.html` + three.js (bundled, works offline) + `room.glb` + `motion.json` + hand meshes (int16). Orbit/pan/zoom, view presets, play/speed, **time slider**, ←/→ frame step. Orbit is limited to the filmed area. Start it with `open_viewer.bat`. |
| `room_stats.json` | Registration, depth scale per clip, triangle count, bounding box |

**Tested** on a synthetic workstation with two clips, a different room frame for each, and depth with a wrong scale plus noise:

- registration recovered the 20° yaw exactly;
- depth scale corrected to 1/1.15;
- mesh within **0.35 cm median** (0.94 cm at the 90th percentile) of the true surfaces;
- bench height within 0.4 cm;
- video, web bundle and viewer page work;
- the save/restore of the SLAM cache works.

**Limits:** the room is an accurate, continuous coloured surface where the cameras saw it, not photo-real. Areas never filmed stay empty, so the viewer keeps the camera near the filmed region. A Gaussian-splat layer on top of this mesh is the next step for realism.

---

## 7. System and how to run

| Task | How |
|---|---|
| PS4 on all clips | `notebooks/ps4v4_colab_all_clips.ipynb` (A100, `PART = 1/2` over two accounts) |
| Add hand meshes to existing results | The same notebook with `MESH_FOR = [clips]` (adds only `hands_mesh.npz`) |
| World view for chosen clips | `notebooks/ps5_world_slam.ipynb` or `run_slam.py --clips-dir … --ps4-dir … --clip …` |
| Room twin + video + viewer | `notebooks/room_twin_colab.ipynb` (about 1 h on an A100) |
| Dashboard API | `notebooks/ps4_api_server_colab.ipynb` (FastAPI + Cloudflare tunnel; see `docs/API.md`) |
| Tests (no GPU) | `tests/test_synthetic.py`, `test_assign_overlap.py`, `test_side_cue.py`, `test_world_synthetic.py`, `test_room_synthetic.py`, `test_api.py` |

**Timing (A100):**

- PS4: about 25 min per 3-min clip with 4 in parallel.
- SLAM: about 7 min per 3-min clip.
- Room twin: about 15 min.
- 2-min fly-through: about 10–15 min.

### 7.1 Repository layout

```
code/
├── ps4v4/        PS4 pipeline (M1–M6, MQ), models/ (WiLoR, CLIP, stand-ins)
├── ps5world/     SLAM backends, world assembly, render, exports
├── room/         views, depth, build (fusion + registration), render_video, web + viewer.html + web_lib (three.js)
├── server/       Colab API for the dashboard
├── configs/      default.yaml
├── notebooks/    Colab notebooks
├── tests/  tools/  docs/
└── run.py  run_slam.py  run_world.py  run_room.py
```

---

## 8. What is new (USPs)

1. **Track-first decisions:** left/right and hand/not-hand are decided per track from pooled evidence. Tracks are cut at crossings and at sustained handedness flips, so left stays left.
2. **Only the wearer's hands:** reach relative to the wearer's own hands, forearm exit point and direction, an on-forearm penalty, and CLIP as a veto only. Every rejection is written down with its reason.
3. **Fisheye-correct 3D:** perspective-corrected virtual crops.
4. **Honest gaps:** infilled and estimated frames are labelled in every output.
5. **SLAM that never tracks hands:** hands are masked out of camera tracking, then re-lifted into the SLAM camera with their meshes. Everything sits in a gravity-aligned room with head, gaze and body. A retry ladder plus fallback means every clip gets a world view.
6. **Clip rating for data curation:** timestamped usable/unusable segments with reasons, so training uses only good seconds and collectors know what to fix.
7. **Room twin from single-seat video:** metric depth scaled to SLAM, multi-clip registration and TSDF fusion give an accurate 3D workstation with the worker's motion replayed inside it, viewable in a browser.

## 9. Results so far

- 40 real factory clips processed.
- v4.4 checked clip by clip on the problem cases.
- Synthetic end-to-end test: 98.8% recall, 100% side accuracy, 0 false hands, 0 swaps.
- SLAM ran on real footage (`MTL-SKL00089-006`).
- **Still to measure:**
  - accuracy against hand-labelled real frames;
  - calibration of the MQ thresholds;
  - the room twin on the real clips.

## 10. Next steps

- Run the room twin on 006 + 007 and check registration and scale.
- Add a Gaussian-splat appearance layer on top of the mesh.
- Integrate the viewer into the dashboard's Room page.
- Fine-tune WiLoR on egocentric clips and compare it with the pretrained model.
- Robot replay in MuJoCo from world wrist + gripper trajectories ("Automation Scout").

## 11. Data rules and credits

**Data rules:**

- Footage is used only for this hackathon.
- No videos, frames or outputs are in git (`.gitignore`).
- API keys come from environment variables or Colab Secrets.
- All copies are deleted after the event, and screens and faces are never shown publicly.

**Pretrained models used:** WiLoR (hand detection + MANO 3D), CLIP, GeoCalib, NVIDIA ViPE, MoGe-2. **Libraries:** Open3D, three.js, Rerun.

**Ours:** the pipeline logic, tracking, decisions, fisheye handling, quality rating, world assembly, room twin, API and viewer.
