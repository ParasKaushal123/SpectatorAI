# SpectatorAI

Egocentric 3D hand tracking for robot learning. Head-mounted fisheye factory video goes in. Out come the wearer's left
and right hands as 21-joint metric 3D skeletons, plus a robot-ready wrist + gripper trajectory.

**Honest framing:** the pipeline (`code/ps4v4/`, "PS4") is ours and is **built on pretrained WiLoR** (hand detector + 3D
hand model) and CLIP. A WiLoR fine-tuned on our egocentric clips is training now. The dashboard labels everything
"pretrained baseline" until that checkpoint is in `weights/`.

## Run it

```bat
run.bat            :: Windows: first run calls setup.bat (needs internet once), then serves http://127.0.0.1:8000
```
```bash
./run.sh           # macOS / Linux / Git Bash
```

After the first setup, everything runs **offline**: model weights are cached, fonts are bundled, and the A100 results
are local files.

| Folder | What goes there | In git? |
|---|---|---|
| `results/<clip>/` | A100 runs with pretrained WiLoR (`hands.jsonl`, `stats.json`, `render.mp4`, `tracklets.csv`, …) | no (footage) |
| `results/finetuned/<clip>/` | the same clips run with our fine-tuned checkpoint, which fills the comparison section | no |
| `results/training/` | `training.json` (setup), `dataset.json`, `*.csv` loss logs (`epoch`/`step` + `*loss*` columns) | no |
| `weights/` | fine-tuned WiLoR checkpoint (`*.ckpt`/`*.pth`) and/or `detector.pt`, auto-loaded by the pipeline | no |
| `raw_clips/` | optional raw videos for a clean overlay: `<clip>.mp4` or `mapping.json` `{"hand_tracking_1": "file.mp4"}` | no |
| `data/` | generated: browser videos, compact clip JSON, upload jobs | no |

## What's in the dashboard

1. **Hero:** three headline numbers computed live from `results/*/stats.json`.
2. **How it works:** the M1–M6 pipeline, showing which steps are ours and which are pretrained models.
3. **USP cards:** each one shows a real number and a **"see it" button that jumps the explorer to a clip moment** found
   automatically in `tracklets.csv` / `hands.jsonl`.
4. **Explorer:** video + 2D keypoint overlay, synced with an orbitable 3D hand viewer (both hands, wrist trajectory), a
   per-frame confidence + source timeline (observed / infilled / estimated), and downloads (`hands.jsonl`, robot CSV,
   stats).
5. **Pretrained vs fine-tuned:** a side-by-side metrics table plus two 3D viewers on the same frames. It fills in when
   `results/finetuned/` has clips.
6. **Results:** per-clip charts and a smoothness table. Synthetic-test numbers are labelled "synthetic test".
7. **Training:** setup facts and loss curves read from `results/training/`. Unknown values show "TO VERIFY".
8. **Live upload:** at most 10 s at 8 fps, 960 px long side, run through `code/run.py` on CPU (measured about 2.6 s per
   frame on this laptop, so about 3.5 min). A phone/fisheye toggle picks the camera model. On failure it shows the error
   and offers the A100 demo.

## Adding the fine-tuned model (when training finishes)

1. Copy the checkpoint to `weights/` (for example `weights/wilor_finetuned.ckpt`; a `state_dict` or Lightning checkpoint
   both work).
2. Run the same clips on Colab with that checkpoint in `weights/` and copy each output folder to
   `results/finetuned/<clip>/`.
3. Put the loss CSV(s) and `training.json` into `results/training/`.
4. Reload the page. The banner switches to "fine-tuned", and the comparison and training sections fill in.

## API

`GET /api/model` · `/api/clips` · `/api/clips/{key}` (compact JSON) · `/api/clips/{key}/video` (H.264) ·
`/api/clips/{key}/download/{hands.jsonl|robot_wrist_gripper.csv|stats.json}` · `/api/metrics` · `/api/evidence` ·
`/api/training` · `POST /api/upload` (file, camera=phone|fisheye) · `GET /api/jobs/{id}`

## Known limits

- Raw footage for the A100 clips isn't matched to the `hand_tracking_N` names yet. The explorer shows the video panel of
  the pipeline's own render, which has its overlay baked in. Add `raw_clips/mapping.json` to get the clean video.
- Phone mode uses a pinhole camera with a 68° horizontal field of view (a typical phone main camera), not a calibration.
  Fisheye uploads use a 150° FOV guess, because GeoCalib is skipped on CPU.
- Real-footage accuracy against hand labels is **TO VERIFY** (needs a labels CSV, then `python code/run.py --evaluate`).

See [DEMO.md](DEMO.md) for the 2-minute demo script.
