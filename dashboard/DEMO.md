# Two-minute workspace demo

Before presenting: run run.bat from the repo root. Start the Colab notebook once, wait for the
connection pill to say ready, and have a permitted 30 s clip available. Use saved local results if
Colab is unavailable. The 1–2 minute target is a goal until the A100 measurements in README are filled in.

| Time | Action | Explain |
|---|---|---|
| 0:00 | Drop the clip into Workspace | Processing starts automatically on Colab; the key stays in the local gateway. |
| 0:15 | Show progress and local video | The app transcodes once, uploads, and reports actual pipeline stage timings. |
| 0:30 | Open Library and a saved workspace | Video and outputs stay usable after Colab shuts down. |
| 0:45 | Play, scrub backward, step a frame | One video clock drives the hands, head orientation and timelines. |
| 1:00 | Orbit, pan, zoom; try Top and Follow | The hands are metric 3D in a head-stable frame. Room translation is a separate SLAM job. |
| 1:15 | Click a quality lane and inspect a hand | Observed, infilled and estimated are different; estimates are guesses. Quality thresholds still need calibration. |
| 1:30 | Download rendered MP4 and robot CSV | The pipeline render is prepared in the background. Browser-view recording preserves the chosen camera. |
| 1:45 | Open Showcase | The original pipeline explanation, evidence, results, comparisons and training remain available. Read numbers from actual loaded results only. |
| 1:55 | Return to the new workspace if ready | Show its measured total time. Do not claim an unmeasured A100 latency. |

The pipeline is ours, built on pretrained WiLoR and CLIP. Fine-tuned weights are optional and labelled.
Synthetic checks reached 98.8% recall and 100% side accuracy; labelled real-footage accuracy remains
unverified. No example clips, frames or screenshots belong in public docs or Git.
