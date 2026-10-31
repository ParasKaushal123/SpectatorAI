# 2-minute demo script

**Before you go on stage:** run `run.bat` and open http://127.0.0.1:8000. Start a live upload at the very beginning
(step 0), because CPU processing takes about 3.5 min. Close every other app.

| Time | Click | Say | USP it proves |
|---|---|---|---|
| 0:00 | (Optional) **Try your video** → drop a 10 s phone clip of your hands → *Phone / normal camera* → **Run the pipeline**, then scroll to the top | "While we talk, this laptop is running our full pipeline on a clip we just shot, on CPU with no cloud." | Live, runs anywhere |
| 0:10 | Hero | "Robots learn manipulation from video of people working. Head-cams see *everyone's* hands. We output only the wearer's hands, in metres, and we label every guess." Point at the three numbers: "82% of what the detector thinks are hands get thrown out, across 7,200 real factory frames." | Wearer-only |
| 0:30 | Scroll to the **green band** | "WiLoR finds hands. These mint steps are what we added: the wearer filter, left/right labelling over whole tracks, labelled infill, and the robot output." | Honest about what's ours |
| 0:45 | USP card 1 → **See a rejection · clip 4** | "The explorer pauses on this moment. Another worker's hands are right there, top-left. Grey boxes are candidates we rejected, with the reason. Only the wearer's hand is tracked." | Rejects others' hands |
| 1:00 | Drag the **3D viewer** to orbit, then press **Play** | "This is metric 3D, joints in metres in a head-stable frame. The line is the wrist trajectory." | Metric 3D |
| 1:10 | USP card 2 → **See hands overlap · clip 5** | "The two hands come together and their boxes overlap. That is where frame-by-frame trackers swap left and right. We cut the track at the overlap and relink it by forearm direction afterwards, so left stays left." | L/R through crossings |
| 1:20 | USP card 4 → **See an estimate · clip 1**, point at the timeline strip | "When a hand leaves the frame we bridge up to 1.5 s, but it's drawn dashed and pink and labelled *estimated*. We never count it as measured." | Honest gaps |
| 1:35 | Card 5 → **↓ Trajectory CSV** | "Wrist pose plus gripper aperture per frame, ready for a robot-learning stack." | Robot-ready |
| 1:45 | **Pretrained vs fine-tuned** section | "We're fine-tuning WiLoR on our own 40–50 egocentric clips on an A100. Same pipeline, same clips, only the 3D model changes, and it lands here side by side." *(If the fine-tuned results are in, play the side-by-side and read the winning row.)* | Headline: fine-tuned |
| 1:55 | Back to the upload (if it finished, it has opened in the explorer) | "And that's the clip we shot two minutes ago." | Works on new video |

**If something breaks:** every section fails on its own. If the upload errors, press **Show the A100 demo instead**.
Clip 5 (`hand_tracking_5`) is the default demo: both hands are tracked about 90% of the time, with no depth jumps.

**Questions judges may ask:**
- *"Is this your model?"* "The pipeline is ours. The 3D hand model is pretrained WiLoR, which we're fine-tuning on our
  data. The dashboard says which one is running in the top bar."
- *"How accurate is it?"* "Against hand labels it's TO VERIFY; we haven't labelled real frames yet. On our synthetic
  test it gets 98.8% recall and 100% side accuracy. On real footage we report the automatic metrics: coverage,
  jitter, depth jumps and rejection reasons."
