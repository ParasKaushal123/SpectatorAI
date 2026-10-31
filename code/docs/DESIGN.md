# PS4 v4 design notes: tracklet-first 3D hand tracking for egocentric fisheye video

## Why v4

The v3 renders showed three root causes:

1. **Anything that looks a bit like a hand was believed.** Slabs, plastic, chair slats, heads, feet and another worker's hands became hands.
2. **Nothing knew where the arm was.** Real hands that came back mid-frame were blocked by the "enter through an edge" rule, while feet at the bottom edge passed it.
3. **Left/right came from screen position.** Hands crossing, or returning on the "wrong" half, swapped sides.

All three come from one design choice: v3 decided "is it a hand" and "which side" online, frame by frame, inside the tracker, and could never undo a decision. v4 only links boxes first. It then decides per tracklet, looking at the whole tracklet with 3D evidence.

## v4.2 changes (from hand_tracking_2 / 4 / 7 / 8)

| Symptom | Cause found | Fix |
|---|---|---|
| ht4: left hand missing for 15 s | one 16-frame overlap with another left track rejected the whole 468-frame track | tracks are cut at overlaps; only the overlap is contested (v4.1) |
| ht2: hands flipped at the start | side vote used the forearm's left/right lean; a left hand reaching forward-left leans RIGHT (toward the body). It outvoted a correct detector | side vote = where the forearm leaves the image (left edge / bottom-left of centre = left arm); detector handedness weighted up |
| ht8: knee and foot taken as hands; ht7: wood taken as the right hand | the detector has one class and nothing knew what a knee, foot or plank looks like | semantic check per probe crop with CLIP (hand vs knee, foot, leg, head, wood, tool, surface, cloth); reason shown as "looks like knee" |
| ht8: left hand inside the chair frame dropped | the hand held still (static penalty) and was partly hidden (keypoint/box penalty) | both penalties made mild; the semantic check does that job now |
| ht2: fast wiping breaks the track | association gate too tight for a fast hand | gate widens with the hand's own speed; motion links between tracks weigh more |

## v4.4 changes (hand_tracking_4 / 8 re-runs; checked offline on their tracklets.pkl + probe.pkl)

| Symptom | Cause found | Fix |
|---|---|---|
| ht4: another worker's hand used again | v4.2 let CLIP *raise* the hand score; another person's hand IS a hand, so it outvoted "arm runs away"; their hands also pulled the typical hand distance up | CLIP is a veto only (per tracklet, median P(hand) < 0.3); a small CLIP bonus only when the geometry says "the wearer's hand"; typical distance = 35th percentile |
| ht8: right hand appears on the left forearm / elbow when out of view; forearms during fast wiping | the detector fires on forearms; nothing knew a hand never sits on another hand's forearm | new "on a forearm" check: a box lying along another (wearer-like, stronger) hand's forearm, 0.8-4.5 hand lengths past its wrist, is penalised |
| ht4: a track slid from the left hand onto the right (hands close, fast head turn) | linking kept one track across two hands | tracks are cut where the detector's left/right call flips and stays flipped (15 detections each side) |
| pieces of one track labelled left AND right | overlap cuts were only softly tied | pieces of one track can never take opposite sides; motion links pick a side but never make junk a hand |
| ht8: left hand inside the chair frame (only wrist visible) | low score and the hand model can't fit the box | the CLIP/geometry bonus keeps it; detector left/right counts less when the fit failed |

## Pipeline

| Module | File | Output (`cache/<clip>/`) | What it does |
|---|---|---|---|
| M1 camera | `camera.py` | `camera.json` | Calibration JSON → GeoCalib focal → FOV guess |
| M2 proposals | `proposals.py` | `proposals.pkl` | WiLoR detector, score ≥ 0.15 (weak boxes kept for cloth/blur), enhanced pass only where needed |
| M1H head rotation | `camera.py` | `head_rot.pkl` | Background KLT → fisheye rays → RANSAC Kabsch (vectorised) |
| **M3 tracklets** | `tracklets.py` | `tracklets.pkl` | ByteTrack-style linking, head-rotation-compensated prediction, flow coasting ≤ 0.5 s. No side, no "is it a hand". Tracklets are **cut when two overlap** (crossing) and cut again when they separate |
| **M3P probe 3D** | `probe.py` | `probe.pkl` | WiLoR on every 3rd frame of each tracklet: wrist distance, forearm exit point, forearm toward/away from camera, keypoints vs box; CLIP check "hand or knee/foot/wood/...?" |
| **M3A decide** | `assign.py` | `tracks.pkl`, `tracklets.csv` | Hand score + side score per tracklet, then one labelling over all tracklets (LEFT / RIGHT / REJECT) |
| M4 3D | `recon.py` | `recon.pkl` | WiLoR on virtual pinhole crops for accepted hands only; probe frames reused |
| M5 sequence | `sequence.py` | `sequence.pkl` | Smoothing inside one segment only; depth gate relative to the wearer's typical hand distance; **out-of-view estimate** for gaps ≤ 1.5 s |
| M6 outputs | `render.py`, `export.py` | `render.mp4`, `hands.jsonl`, `robot_wrist_gripper.csv`, `stats.json` | The render shows rejected tracklets as grey boxes with the reason |

## How each root cause is fixed

### 1. Is it the wearer's hand? (M3A hand score)

Each probe frame adds or subtracts log-odds:

| Evidence | Catches | Source idea |
|---|---|---|
| Detector score | weak junk | — |
| Wrist distance > 1.9 × the wearer's typical distance (`z_ref`, the clip median) | another person's hand | EgoHands own-vs-other priors (Bambach 2015) |
| Closer than 0.3 × `z_ref` | junk filling the lens | — |
| Forearm runs away from the camera | a person facing the wearer | — |
| Forearm leaves the image through the top 10% | another person's arm | Betancourt 2016 border rule, applied to the arm |
| Hand-model keypoints don't fill the detector box | junk | — |
| Static in the scene for its whole life (≥ 1 s, head rotation removed) | table, slab, chair | — |
| Fewer than 3 detections | flicker | — |

`z_ref` is measured from the clip itself, so a wrong focal length scales it with everything else. The synthetic test passes with a focal that is 35% off.

### 2. Where is the arm? (M3A labelling + M5 estimate)

- There is no edge-entry rule any more. A hand can come back anywhere, as long as it passes the hand score.
- Consecutive tracklets whose motion links up prefer the same side. This handles loss, occlusion and crossings.
- M5 bridges short out-of-view gaps with a wrist path that stays within arm's reach. That path is labelled `estimated`, drawn faded and dashed, never counted as observed, and kept out of accuracy.

### 3. Which side? (M3A side score)

Per frame:

- **Forearm direction in the image** (strongest): the left forearm runs down-left, the right one down-right. This is Betancourt et al.'s 2016 arm-angle cue.
- Detector handedness.
- A weak position prior, which is dropped inside crossings.

The labelling is solved greedily, strongest tracklets first, then refined with eviction moves. It enforces two things:

- **At most one left and one right per frame**: two tracklets that share a frame can't get the same side.
- **Identity through a crossing** comes from arm direction and motion links once the hands separate (Rangesh 2016), not from box matching inside the overlap.

## Debugging and tuning

- `tracklets.csv` lists every tracklet with its features, `hand`, `side`, `label` and `reasons`.
- The render shows rejected tracklets in grey with the reason: `static`, `far x2.4`, `too close`, `arm away`, `kp/box`, `short`, `side taken`.
- The main knobs, in `configs/default.yaml` → `assign`:
  - `accept_bias`: raise it for stricter decisions.
  - `far_ratio`: lower it to reject distant hands more often.
  - `w_static`: the static penalty.
  - `w_arm`: the weight of the forearm side cue.
- Re-decide without recomputing anything else: `python run.py --clip X --from M3A --set assign.accept_bias=0.3`.

## Run

- **Colab:** use the notebooks in `notebooks/` (see `README.md`).
- **Locally:**

```bash
python run.py --clip "clips/*.mp4"
python run.py --clip demo.mp4 --set max_seconds=30
python run.py --clip demo.mp4 --from M3A --set assign.accept_bias=0.3
python run.py --evaluate labels.csv --systems v3=v3.jsonl v4=cache/demo/hands.jsonl
```

## Test

`python tests/test_synthetic.py` runs a 10 s synthetic clip with fake models, no GPU and no downloads. It reproduces the v3 failures:

- static junk while a hand is out
- a mid-frame occlusion and return
- another person's small, distant hand
- hands crossing and staying on the wrong halves
- blur
- 25% missed detections and 12% label flips

Current result:

| Check | Result |
|---|---|
| Junk frames tracked | 0 |
| Other person's frames tracked | 0 |
| Side swaps | 0 / 478 |
| Left hand still labelled LEFT after crossing | 94% of frames |
| Edge re-entry | 1 frame |
| Mid-frame return | 1 frame |
| Depth jumps | 0 |
| Recall | 98.8% |
| Side accuracy | 100% |
| False hands | 0% |

## Known limits

- The hand-score weights are hand-set. Check `tracklets.csv` on 2–3 clips and adjust.
- A real hand held completely still for its whole tracklet (≥ 1 s) gets the static penalty. A good detector score usually outweighs it; if not, lower `w_static`.
- Feet seen while walking can still pass if they look hand-sized and close. Watch for `walking` clips.
- The out-of-view estimate is a guess and is labelled that way. It is not a measurement.
- SAM 2 and EgoHand stay optional, as in v3. With EgoHand, the probe still uses WiLoR.

## Data rules

Keep `cache/`, `labels/` and every video out of git (see `.gitignore`). Delete all copies after the event.
