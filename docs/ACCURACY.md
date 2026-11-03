# Accuracy: what we measured, how, and what to say

## What to say (about 30 seconds)

> "We measure accuracy in two ways.
>
> First, on synthetic clips where we know the exact truth: we find **98.8% of the hands**, get **left and right 100% right**, track **zero false hands**, and place hands in the room to **within 1 cm**.
>
> Second, on real factory footage there's no ground truth. Nobody wears motion capture on a factory floor. So we measure what is physically checkable. On a full 3-minute real clip we had **zero left/right swaps**, **zero impossible jumps**, a hand size of **9.5 cm**, which matches an adult hand, and the wrist **within arm's reach 96% of the time**. We threw out **77% of more than a thousand candidates** as not the wearer's hand, each with a logged reason.
>
> The next step is a hand-labelled benchmark, and the scoring tool for it is already built."

## Table 1: synthetic test (ground truth known)

| Metric | Result |
|---|---|
| Hand recall (wearer's hands found) | **98.8%** |
| Left/right accuracy | **100%** (0 swaps in 474 two-hand frames) |
| False hands (knee, other person, junk) | **0%** |
| Left hand still left after crossing | 94% of frames |
| Hand out of view correctly reported | 100% |
| Hand position in the room (SLAM step) | **< 1 cm** error, even with the lens focal 30% wrong |
| Resting-hand drift in the room | 0.6 cm |
| Gravity / "up" error | 0° |
| 3D room geometry (room twin) | 0.35 cm median error, bench height within 0.4 cm |

## Table 2: real footage, `mytronlabs-000003-005` (3 min, polishing station)

| Metric | Result | What it means |
|---|---|---|
| Left/right swaps | **0 per minute** | No hand ever jumps into the other hand's place |
| Impossible jumps (> 25 cm in one frame) | **0 per minute** | No glitches or teleports |
| Hand size (wrist to middle knuckle) | **9.5 cm**; 100% of frames in the adult range (7–11 cm) | The 3D scale is right |
| Wrist within arm's reach (15–90 cm) | **96%** of tracked frames | Hands are placed where a real arm can be |
| Jitter | **1.6 mm** per frame² | Smooth motion |
| Hands tracked | right **92%**, left **69%** of frames | The left hand often reaches out of frame toward the wheel |
| Candidate tracks rejected | **77% of 1,039** | Main reasons: no valid 3D, too short (flicker), static object |
| Usable seconds (clip rating) | **94.5%**, grade **A** | |

Scale this up: run `accuracy_report_colab.ipynb` on all 40 clips in Drive and replace this table with the 40-clip summary.

## How we calculate it

**Synthetic tests** (`code/tests/test_synthetic.py`, `test_world_synthetic.py`, `test_room_synthetic.py`):
- We generate a video in which we know every hand, side, position and the room geometry.
- That includes a crossing, a knee, another person's hand, a hand leaving the frame, and a wrong lens focal.
- We run the full pipeline and compare against that truth. This gives true accuracy numbers.

**Real footage** (`code/tools/accuracy_report.py`, no labels needed):

| Metric | Formula |
|---|---|
| Left/right swaps | Frames where each hand is closer to where the *other* hand just was than to its own previous position: `cross < 0.5 × keep` and `keep > 10 cm`, per minute |
| Impossible jumps | Wrist moves > 25 cm between consecutive frames (> 7.5 m/s), per minute |
| Hand size | Median distance from the wrist joint to the middle-finger knuckle in 3D; share of frames within 7–11 cm |
| Arm's reach | Share of tracked frames with the wrist 15–90 cm from the camera |
| Jitter | Median wrist acceleration (second difference), in mm per frame², with head rotation removed |
| Rejection | From `tracklets.csv`: candidate tracks labelled reject, with their logged reasons |
| Usable share / grade | From the clip rating (`quality.json`) |
| SLAM check (when available) | Head rotation from SLAM vs. our own head tracking, which should agree within about 2° |

**The labelled benchmark** (next step, tooling already exists):
1. `tools/sample_label_frames.py` picks about 100 hard frames per clip.
2. A person marks the hand boxes and left/right.
3. `run.py --evaluate labels.csv` reports recall, side accuracy, false-hand rate and out-of-view correctness.

## If they push back

- **"Synthetic data is easy."** It deliberately includes the hard cases: crossings, a knee, another person's hand, a hand leaving the frame, and a wrong lens. The real-footage checks confirm the same behaviour: no swaps, no jumps, correct scale.
- **"Plausibility isn't accuracy."** Agreed. That's why we report it as a consistency check, and the labelled benchmark is our immediate next step. What it does prove is that the output has no gross errors: no swaps, no teleports, correct hand scale, and hands within reach.
