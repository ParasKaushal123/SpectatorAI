# Code notes: quick answers for judges

## One-line answer
> "`run.py` runs the hand tracker, `run_slam.py` places the hands in the room, `run_room.py` builds the 3D workstation, and `run_robot.py` replays the motion on robot arms. Every stage writes cached files, so each step can be rerun on its own."

## Entry points (`code/`)
| File | What it does |
|---|---|
| `run.py` | PS4 hand tracking on one clip → `hands.jsonl`, `quality.json`, `render.mp4` |
| `run_slam.py` / `run_world.py` | PS5: SLAM camera path + hands in a fixed room frame |
| `run_room.py` | 3D room twin (mesh + web viewer) |
| `run_robot.py` | Two Panda arms copy the hands; grades each frame |
| `configs/default.yaml` | Every threshold lives here; `profiles/fast.yaml` = fast mode. Override with `--set key=value` |

## `ps4v4/`: hand tracking, in pipeline order
1. **`video.py`, `camera.py`:** read frames. Lens and gravity come from **GeoCalib**, with a fisheye fallback.
2. **`proposals.py`:** the **WiLoR** detector finds every possible hand per frame. We keep many on purpose (high recall).
3. **`tracklets.py`:** links detections over time into tracks. A track is cut when hands overlap or the left/right call flips.
4. **`probe.py`:** **CLIP** checks a few crops per track: "is this a hand?" (catches cloth, knees, tools).
5. **`assign.py`: the core idea.** Each track collects evidence (size and distance, where the arm enters the frame, CLIP score, handedness, motion), added up as log-odds. Then it decides: wearer left, wearer right, or reject, with a written reason. Close-work mode: a hand leaving through the bottom of the frame is still the wearer's.
6. **`recon.py`:** builds the **MANO** 3D hand mesh from virtual crops that undo the fisheye distortion.
7. **`sequence.py`:** smooths the motion and fills short gaps. Filled frames are labelled `infilled`, never `observed`.
8. **`quality.py`:** the MQ rating, per second and per clip (below).
9. **`export.py`, `render.py`, `scene.py`:** write the output files and the overlay video.
- **`cache.py`:** every stage is saved to disk, so a crash resumes where it stopped.
- **`models/fake.py`:** fake models for CPU tests.

## MQ (clip quality), `quality.py`
- **Score** = 0.30 hands visible + 0.15 both hands + 0.20 clarity (no blur) + 0.20 stability (head turning < 90°/s) + 0.15 confidence.
- **Grades:** A ≥ 80, B ≥ 65, C ≥ 50, D below. Output: which seconds are usable, and why the others aren't.

## `ps5world/`: SLAM
- **`backends.py`:** **NVIDIA ViPE** with a wide-angle lens model, hands masked out. If it fails, a retry ladder: fixed lens → lower fps → rotation only. **It always returns an answer.**
- **`world.py`:** finds "up" from gravity and the bench, re-lifts the hands into the SLAM camera, and puts them in the room frame (Z up, metres).
- **`ps4_io.py`:** reads PS4 outputs. **`render_world.py`:** renders the world view.

## `room/`: 3D twin
- **`views.py`:** removes the fisheye into normal virtual views. **`depth.py`:** **MoGe-2** depth, scaled to the SLAM map (hands and body masked out).
- **`build.py`:** registers the two clips (ICP), then **TSDF** fusion (Open3D) → mesh.
- **`web.py`, `viewer.html`:** offline three.js viewer with a time slider.

## `robot/retarget.py`
- MuJoCo Panda ×2. Arm bases sit beside the worker's head. **mink** inverse kinematics follows the wrist position.
- Each frame is graded copyable / out of reach / too fast / no hand. On 007: **79% copyable**.

## `server/`, `dashboard/`
- **`server/app.py`:** a FastAPI gateway with a job queue. Heavy work runs on a Colab GPU, connected through `notebooks/CONNECT_COLAB.ipynb`. `tools/idle_guard.py` shuts it down when idle.
- **`dashboard/`:** a React app showing the Library, the clip player with 3D hands, MQ, Room and Robot replay.

## `tests/`, `tools/`
- **`tests/`:** synthetic videos with known truth (crossings, knee, other person, hand leaving the frame, wrong lens) → 98.8% recall, 100% left/right.
- **`tools/accuracy_report.py`:** label-free checks on real clips (swaps, jumps, hand size, reach, jitter).
- **`tools/sample_label_frames.py`, `ps4v4/evaluate.py`:** labelled-benchmark tooling.

## Likely code questions
- **"What did you build vs. reuse?"** Reused: WiLoR, CLIP, GeoCalib, ViPE, MoGe, MANO, MuJoCo. Ours: the track-first decisions, the fisheye-aware 3D, honest gap labels, MQ, the world frame, the room twin and the robot replay.
- **"Why not decide per frame?"** One frame can fool any model. Collecting evidence over a whole track is far more reliable, and every decision is explainable.
- **"How do you handle failures?"** Cached stages, a SLAM retry ladder, and labelled fallbacks. No crash ever leaves a clip with no output.
- **"Where are the thresholds?"** All in `configs/default.yaml`, nothing hard-coded.
- **"API keys and data?"** Keys come from environment variables; videos are git-ignored and deleted after the event.
