# SpectatorAI: egocentric 3D hand tracking, world view and clip rating for robot learning

Head-mounted fisheye video of factory work goes in. Three things come out:

1. **The wearer's two hands in metric 3D**, frame by frame, with left/right always correct. Every guess is labelled as a guess.
2. **A room-fixed world view (SLAM).** The head, gaze, body axis and hands are placed in one gravity-aligned 3D room.
3. **A clip rating.** It shows which seconds are usable for robot learning and which aren't, and why.

All of it is served to a local dashboard. The user drops in a video, a Colab A100 processes it, and the result opens as a 3D workspace the user can orbit and scrub, with export.

---

## 1. Problem

Robots can learn manipulation from video of people working. Head-mounted cameras are the cheapest way to record that. But raw egocentric factory footage is hard to use:

| Problem | What goes wrong with off-the-shelf hand models |
|---|---|
| **Other people's hands** | Co-workers' hands in view are tracked as the wearer's. |
| **Hand look-alikes** | Knees, feet, gloves, wood, cloth and tools get detected as hands. |
| **Left/right swaps** | When the hands cross or overlap, frame-by-frame trackers swap the sides. |
| **Fisheye lens** | Strong distortion at the edges, where hands often are, breaks 3D models trained on normal photos. |
| **Out of view** | Hands leave the frame for a moment. Trackers either drop them or invent positions without saying so. |
| **No world frame** | Hand positions are relative to a moving head, so they mean nothing to a robot in a fixed workspace. |
| **Unusable stretches** | Motion blur, fast head turns, walking and hidden hands. Nobody tells you which parts of a clip to train on. |

## 2. Solution

A modular pipeline, **PS4 v4**. It decides about **whole tracks, not single frames**, and keeps every decision explainable. On top of it sit two more parts:

- **PS5 world view:** SLAM puts everything in one room frame.
- **MQ clip rating:** it marks the usable and unusable parts of the clip, with reasons.

Outputs per clip:

| File | Content |
|---|---|
| `hands.jsonl` | Per frame, per hand: side, state, 2D and 3D joints (21), orientation, confidence, source (observed / infilled / estimated) |
| `robot_wrist_gripper.csv` | Wrist position + orientation + gripper opening per frame, ready for a robot-learning stack |
| `quality.json` | Score 0–100, grade A–D, usable segments, timestamped issues per lane |
| `world_hands.jsonl`, `trajectory.csv`, `scene.rrd` | Head pose, gaze, body axis and hands in the room frame; a Rerun 3D recording |
| `render.mp4`, `world_render.mp4` | Side-by-side video with the 3D view |
| `tracklets.csv` | Every candidate track with its evidence and the decision taken (audit trail) |

## 3. Architecture

### 3.1 PS4 v4: 3D hand tracking (track-first)

```
video ─► M1 camera ─► M2 proposals ─► M1H head rotation ─► M3 tracklets ─► M3P probe ─► M3A assign ─► M4 3D ─► M5 sequence ─► M6 outputs ─► MQ rating
```

| Module | What it does | Built on |
|---|---|---|
| **M1 Camera** | Focal length from GeoCalib; equidistant fisheye model | GeoCalib (pretrained) |
| **M2 Proposals** | Hand boxes, kept down to score 0.15 (weak boxes are decided later, not dropped). Enhanced pass on dark frames. | WiLoR YOLO detector (pretrained) |
| **M1H Head rotation** | Background feature tracking (KLT) + RANSAC rotation, with the hands masked out | ours |
| **M3 Tracklets** | Links boxes into short tracks, with a speed-aware gate and optical-flow coasting. It cuts tracks at **crossings** and where the detector's **left/right call flips and stays flipped**. It makes no side or hand/not-hand decision. | ours |
| **M3P Probe** | 3D reconstruction on a sample of each track. Collects evidence: wrist distance, forearm exit point, forearm direction, keypoint/box fit, CLIP "hand or knee/foot/wood/tool?". | WiLoR + CLIP (pretrained) |
| **M3A Assign** | LEFT / RIGHT / REJECT per track, from log-odds evidence (see 3.2) | ours |
| **M4 3D hands** | WiLoR on **perspective-corrected virtual crops** cut out of the fisheye image; probe frames reused | ours + WiLoR |
| **M5 Sequence** | Outlier removal, adaptive smoothing, gap infill within a segment, and **labelled out-of-view estimates** for gaps ≤ 1.5 s | ours |
| **M6 Outputs** | Render, `hands.jsonl`, robot CSV, stats | ours |
| **MQ Rating** | Usable / unusable timeline and score (see 3.4) | ours |

Every module is **cached with a hash of its config**. Changing a threshold re-runs only that module and the ones after it, and a Colab disconnect resumes where it stopped.

### 3.2 How M3A decides (the core of v4)

- **Is it the wearer's hand?** Evidence is added up over the track's frames:
  - the detector score;
  - distance relative to the wearer's typical hand distance (the 35th percentile, so the wearer's own hands set the reference). "Far" means another person.
  - forearm running away from the camera, or leaving through the top of the image (someone facing the wearer);
  - the box lying on another hand's forearm or elbow;
  - static for its whole life (furniture);
  - poor keypoint fit.
- **CLIP can veto, never boost.** It vetoes only when a track's median P(hand) is low. Another person's hand is still a hand, so CLIP can't be the thing that accepts it. A small bonus applies only when the geometry already says "the wearer's hand".
- **Which side?** Mainly where the forearm leaves the image, plus the detector's handedness (which counts less when the 3D fit failed), plus a weak position prior.
- **Global labelling.** One left and one right per frame. Pieces of one track can't take opposite sides. Motion links between tracks prefer the same side but can never turn junk into a hand.

### 3.3 PS5 world view (SLAM)

```
clip + PS4 results ─► ViPE SLAM (Mei fisheye model, people/hands masked) ─► re-lift hands into SLAM camera ─► gravity + room frame ─► body axis + gaze ─► exports
```

1. **Camera trajectory:** NVIDIA ViPE (dense learned bundle adjustment with metric depth priors). It uses the `wide_angle` preset (unified/Mei camera model), runs at 15 fps, and poses are interpolated to every frame.
2. **Moving things removed from tracking:** ViPE's person/hand/arm masks plus PS4's own hand boxes, grown and extended down to the image edge for the forearm. Hands never drive the camera estimate. A guard stops the mask from covering so much that tracking breaks.
3. **Hands into the SLAM camera:** each hand keeps PS4's shape. Its wrist is re-aimed along ViPE's ray for the same pixel, and its distance is rescaled by the focal-length ratio. This also corrects GeoCalib's focal error.
4. **Gravity and room frame:** GeoCalib gravity, refined by the dominant horizontal plane of the map (bench/floor). Z up, Y = the wearer's average facing direction, origin = the first head position.
5. **Body axis:** torso under the head along gravity (neck 0.22 m, pelvis 0.62 m), facing the head's heading smoothed over about 1 s. Gaze = the camera's forward axis.
6. **Robustness:** if bundle adjustment diverges, it retries with fixed intrinsics, then at a lower frame rate. If that fails too, it falls back to rotation only (PS4's head rotation, head position fixed), so a clip always produces a world view.

### 3.4 MQ: usable vs unusable parts of a clip

Each frame gets a label on every lane. Segments shorter than 0.3 s are merged into their neighbours.

| Lane | Labels | Signal |
|---|---|---|
| Left / right hand | visible, partial, low_confidence, gap_filled, out_of_view | tracking state, joints inside the image, 3D confidence |
| Image | sharp, blurred | Laplacian variance relative to the clip's own typical sharpness |
| Head | stable, fast_turn | head rotation speed > 90°/s (from M1H) |
| Body | still, moving | background motion not explained by head rotation (walking, leaning, standing up) |
| **Usable** | usable, unusable + **reasons** | at least one hand seen **and** sharp **and** head stable **and** body still |

**Score (0–100):** 30% hand visibility + 15% both hands + 20% image clarity + 20% camera stability + 15% tracking confidence. Grades: A ≥ 80, B ≥ 65, C ≥ 50, otherwise D.

Output: the usable share and seconds, the longest usable segment, timestamped segments per lane, seconds lost per issue, and 10 Hz arrays for charts.

### 3.5 System: dashboard + Colab

```
Browser (localhost) ──► local gateway (FastAPI) ──► Cloudflare tunnel ──► Colab A100: PS4 API (FastAPI, one GPU worker, models kept loaded)
        ▲                     │  transcode 720p, upload, poll, fetch, delete remote copy
        └── 3D workspace ◄────┘  clip.json + scene.json + quality.json + video
```

- The tunnel URL is found automatically, so there are no clicks per video.
- The API key stays on the local backend, never in the browser.
- A "fast" profile targets a 30 s clip in 1–2 min. The decision logic is unchanged; the speed comes from a single frame decode, a lighter detector setting, and making the render later for export only.
- The 3D workspace shows the head frustum and gaze, hands and trails. It has orbit/pan/zoom, view presets and one slider for the video, 3D view and quality lanes. Export: the rendered MP4, a recording of the user's own view, and the data files.

## 4. What is new

1. **Track-first decisions.** Hand vs not-hand and left vs right are decided per tracklet from pooled 3D and semantic evidence, not per frame. Tracks are cut at crossings and at sustained handedness flips, so a track that slides from one hand onto the other is split.
2. **Wearer-only filtering.** Distance relative to the wearer's own reach, forearm exit and direction, an on-forearm penalty, and a CLIP that can only veto. This rejects co-workers' hands, knees, feet and objects, and every rejection is written down with its reason.
3. **Fisheye-correct 3D.** Perspective-corrected virtual crops per hand, so a pinhole-trained hand model works at the lens edge.
4. **Honest gaps.** Infilled and estimated frames are labelled in every output and are never counted as measured.
5. **SLAM with hands removed and hands re-lifted (USP).** The SLAM never tracks hands. PS4's hands are re-lifted into the SLAM camera, which fixes the focal-length error. Everything sits in a gravity-aligned room frame together with head, gaze and body axis. The divergence guards and the rotation-only fallback mean a clip always produces a world view.
6. **Clip rating for data curation (USP).** Timestamped usable/unusable segments with reasons. Collectors learn what to fix, and training uses only the good seconds.
7. **Explainable and cheap to tune.** Module caching plus `tracklets.csv` make every decision traceable. A decision can be re-tuned offline in seconds from saved evidence, without re-running video.

## 5. Results so far

| What | Result |
|---|---|
| Real footage processed | 40 factory clips (2 Colab A100 accounts) |
| Manual review | v4.4 checked clip by clip on the problem cases (other person's hands, knee/foot, forearm, crossings): passed |
| Synthetic end-to-end test | 98.8% recall, 100% side accuracy, 0 false hands, 0 left/right swaps out of 474 frames |
| SLAM step, synthetic scene with known truth | resting-hand drift 0.6 cm; hand position error < 1 cm with PS4's focal length 30% off; gravity error 0°; head path length exact |
| Speed (A100, full quality) | about 25 min per 3-min clip with 4 in parallel (about 6–7 min per clip throughput) |
| Speed target (dashboard, fast profile) | 30 s clip from upload to ready in 1–2 min |

**Not measured yet:**

- accuracy against hand-labelled real frames (the label tool exists: `tools/sample_label_frames.py` + `run.py --evaluate`);
- SLAM on real footage (first run in progress);
- MQ thresholds calibrated on real clips.

## 6. How to run

| Task | How |
|---|---|
| All clips on Colab | `notebooks/ps4v4_colab_all_clips.ipynb` (A100; `PART = 1/2` splits the work over two accounts) |
| One clip, command line | `python run.py --clip clip.mp4` (`--set key=value` to change any setting; `--from M3A` to re-decide only) |
| World view | `notebooks/ps5_world_slam.ipynb`, or `python run_slam.py --clips-dir <videos> --ps4-dir <ps4 results> --clip <name>` |
| API for the dashboard | `notebooks/ps4_api_server_colab.ipynb` (prints the URL and key); reference in `docs/API.md` |
| Dashboard | `run.bat` (first run sets up), then open `http://127.0.0.1:8000` |
| Tests (no GPU) | `python tests/test_synthetic.py`, `test_assign_overlap.py`, `test_side_cue.py`, `test_world_synthetic.py`, `test_api.py` |

## 7. Repository

```
ps4-hand-tracking/
├── code/
│   ├── ps4v4/          PS4 pipeline (M1–M6, MQ), models/ (WiLoR, CLIP, stand-ins)
│   ├── ps5world/       SLAM backends (ViPE, rotation), world assembly, render, exports
│   ├── server/         Colab API
│   ├── configs/        default.yaml (every threshold), profiles/
│   ├── notebooks/      Colab notebooks
│   ├── tests/  tools/  docs/ (API.md, DESIGN.md, SLAM_LOCAL.md)
│   ├── run.py  run_slam.py  run_world.py
├── dashboard/          local gateway + React/Three.js frontend
└── DASHBOARD_PLAN.md
```

## 8. Scaling and next steps

- **Throughput:** the pipeline runs one clip per GPU process with cached modules, so it scales out linearly over GPUs or jobs. The fast profile and pre-transcoding cut the cost per clip about 2–3×.
- **Data flywheel:** MQ ratings tell collectors what to fix (keep hands in view, slow head turns). Training sets keep only the usable segments.
- **Fine-tuning:** the dashboard is ready to compare a WiLoR model fine-tuned on egocentric factory clips against the pretrained baseline on the same clips.
- **Robot replay:** world-frame wrist and gripper trajectories go to MuJoCo (Franka arm). This shows which motions a robot can copy and why it can't copy the others ("Automation Scout").
- **Hardware:** works on Colab A100 today. A local RTX laptop GPU (WSL2 for SLAM) and CPU fallback exist for short clips.

## 9. Data and ethics

Footage is used only for this hackathon. No videos, frames or outputs are in git: `.gitignore` covers them. API keys come from environment variables or Colab Secrets. All uploaded, cloud and local copies are deleted after the event.

## 10. Credits

Pretrained models: **WiLoR** (hand detection and 3D), **CLIP** (semantic veto), **GeoCalib** (camera calibration and gravity), **NVIDIA ViPE** (SLAM, Apache-2.0), **Rerun** (3D recording). The pipeline logic, tracking, decisions, fisheye handling, world assembly, clip rating, API and dashboard are ours.
