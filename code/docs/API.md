# Pipeline API: reference for the frontend

The pipeline runs on Colab (A100) behind a public HTTPS tunnel. The browser calls the local gateway, which authenticates to Colab, copies results locally, then deletes the remote job.
Start the server with `notebooks/ps4_api_server_colab.ipynb`. Cell 3 prints:

- **API URL**, e.g. `https://some-words.trycloudflare.com`. It changes on every start.
- **API KEY**

Store the key only in `dashboard/.env`; Settings may override the URL. The updated notebook discovers the URL automatically via ntfy and Drive.

## Auth

Send the key on every request in one of two ways:

- **Header `X-API-Key: <key>`:** for `fetch` calls.
- **Query `?key=<key>`:** for URLs the browser loads itself (`<video src>`, download links).

CORS is open (`*`), so a frontend on `localhost` can call the server directly.

## Endpoints

| Method | Path | Body / returns |
|---|---|---|
| GET | `/health` | `{ok, gpu, queued, max_seconds, auth}` |
| POST | `/jobs` | multipart: `file` (the .mp4), optional `name` → **Job** |
| GET | `/jobs` | **Job[]**, newest first |
| GET | `/jobs/{id}` | **Job** (poll every 2–3 s while `queued` / `running`) |
| GET | `/jobs/{id}/result` | `{job, stats, quality, files}` (409 until done) |
| GET | `/jobs/{id}/files/{name}` | `input.mp4`, `render.mp4`, `hands.jsonl`, `robot_wrist_gripper.csv`, `stats.json`, `quality.json`, `tracklets.csv`. Videos support Range requests, so seeking works. |
| DELETE | `/jobs/{id}` | deletes the job and its files |

Limits:

- Clips longer than `max_seconds` (default 90) are processed up to that point (`truncated: true`).
- The tunnel accepts uploads up to about 100 MB.
- Jobs run one at a time. A 1-min clip takes about 3–5 min on an A100.

## Job

```json
{"id": "3f9a1c2b7d10", "name": "mytronlabs-000003-008", "status": "running",
 "stage": "Detecting hands…", "progress": 0.31, "queue_position": null,
 "video": {"duration_s": 58.4, "fps": 30.0, "width": 1920, "height": 1080, "frames": 1752},
 "processed_seconds": 58.4, "truncated": false, "created_at": 1761900000.0,
 "quality": {"score": 74.2, "grade": "B", "subscores": {}, "usable_share": 0.68},
 "error": null}
```

- `status`: `queued` | `running` | `done` | `failed`. On failure, `error` and `traceback` are filled.
- `stage` names, in order:
  1. Camera model
  2. Detecting hands
  3. Head motion
  4. Linking tracks
  5. Checking candidates
  6. Deciding left / right
  7. 3D hands
  8. Smoothing
  9. Rendering video
  10. Rating the clip

## quality.json: the clip rating

```json
{
  "score": 74.2, "grade": "B",                       // 0-100; A >= 80, B >= 65, C >= 50, D below
  "subscores": {"hand_visibility": 0.81, "both_hands": 0.55, "image_clarity": 0.92,
                "camera_stability": 0.77, "tracking_confidence": 0.64},       // each 0-1
  "usable": {"share": 0.68, "seconds": 39.7, "longest_segment_s": 14.2,
             "segments": [{"start_s": 0.0, "end_s": 6.3}, {"start_s": 9.1, "end_s": 23.3}]},
  "hands": {"left":  {"visible": 0.62, "partial": 0.10, "low_confidence": 0.05, "gap_filled": 0.01, "out_of_view": 0.22},
            "right": {"visible": 0.70, "partial": 0.08, "low_confidence": 0.04, "gap_filled": 0.02, "out_of_view": 0.16}},
  "issues": {"blurred_s": 3.1, "fast_head_s": 4.0, "body_moving_s": 6.2,
             "left_out_of_view_s": 12.8, "right_out_of_view_s": 9.3},
  "segments": [
    {"lane": "left_hand", "label": "visible", "start_s": 0.0, "end_s": 4.2, "start_frame": 0, "end_frame": 125, "duration_s": 4.2},
    {"lane": "usable", "label": "unusable", "start_s": 6.3, "end_s": 9.1, "reasons": ["head turning fast", "motion blur"]}
  ],
  "timeline": {"t": [0.0, 0.1], "head_speed_dps": [], "body_motion_px": [], "sharpness_rel": [],
               "left_hand": [], "right_hand": [], "left_conf": [], "right_conf": [], "usable": []},
  "thresholds": {}
}
```

### Lanes

Each lane covers the whole clip with no gaps. Draw one timeline row per lane.

| `lane` | labels |
|---|---|
| `left_hand`, `right_hand` | `visible`, `partial` (cut by the frame edge or partly hidden), `low_confidence` (blurred or hard to see), `gap_filled` (a short gap bridged), `out_of_view` |
| `image` | `sharp`, `blurred` |
| `head` | `stable`, `fast_turn` (head turning faster than 90°/s) |
| `body` | `still`, `moving` (walking, leaning, standing up: background motion not explained by head turning) |
| `usable` | `usable`, `unusable` (with `reasons`) |

A frame is **usable** when all four of these hold:

- at least one hand is visible or partial;
- the frame is sharp;
- the head isn't turning fast;
- the body isn't moving.

Segments shorter than 0.3 s are merged into their neighbours.

### `timeline`

`timeline` holds 10 Hz arrays for charts; all arrays have the same length as `t`.

## Other result files

| File | Content |
|---|---|
| `hands.jsonl` | One line per frame: `{frame, t, hands:[{side, state, source, conf, bbox, kp2d, kp3d_cam, kp3d_head, quat_head_xyzw, joint_in_image}]}`. `kp2d` gives pixel coordinates of 21 joints (wrist, then thumb, index, middle, ring and little finger, 4 joints each), for drawing skeletons over the video. |
| `render.mp4` | The video with skeletons next to the 3D hand view. |
| `robot_wrist_gripper.csv` | `t, frame, side, x, y, z, qx, qy, qz, qw, grip, source` |
| `stats.json` | Pipeline numbers + a `quality` summary. |


## Dashboard server additions (v1.1)

`POST /jobs`: multipart `profile=fast|full` (default fast), `render=none|after` (default none).
Fast merges `configs/profiles/fast.yaml` over the default config, followed by PS4_OVERRIDES;
job-owned output paths and max_seconds remain enforced. Full retains the original config.
CLI defaults and tuned decision logic are unchanged. Uploads are limited to 100 MB on the server.

`POST /jobs/{id}/render` (same auth) requires a done job and is idempotent while queued/running/done.
It queues an independent render worker reading immutable cached PS4 outputs; only rendering runs.
The next GPU job can proceed while rendering. Job.render holds status (none/queued/running/done/failed),
progress 0–1 and error. `render=after` requests it as soon as data is ready. A job cannot be deleted
while processing or rendering; queued work can be deleted safely and workers skip missing entries.

Job.stage_seconds holds measured module timings, and profile records the selected profile.
GET /health also returns warm and warm_error. Startup preloads model singletons and runs a 2 s
warm-up clip; failed warm-up is reported instead of falsely declaring readiness. Model jobs still
run one at a time. PS4_JOB_TTL_H defaults to 6; a sweep every minute removes expired inactive jobs.
PS4_API_KEY is the preferred server secret; PS4_API_TOKEN remains accepted for existing clients.

### scene.json

Available through `/jobs/{id}/files/scene.json`. Contains:

- fps, width, height, n_frames;
- camera: exact PS4 camera intrinsics including KB4 model/distortion and calibration source;
- frame: camera at frame zero, x right/y down/z forward, metres (same as kp3d_head);
- head_quat_xyzw: one camera_t → reference quaternion per processed frame;
- gravity_ref: unit **down** vector in that reference frame;
- gravity_source: geocalib or camera_y;
- hand_centre_ref: median observed wrist, fallback [0, 0.25, 0.45] when no hands are observed.

GeoCalib's gravity object represents up (its zero-roll/pitch vector is [0,-1,0]); it is negated for
this down-vector schema, rotated by each sample's head rotation, and aggregated in the reference.
See the [GeoCalib Gravity implementation](https://github.com/cvg/GeoCalib/blob/main/geocalib/gravity.py).
The browser maps reference coordinates to (x,-y,-z), then levels the scene so down becomes -Y.
This is rotation-only scene metadata, not SLAM translation or a claim of absolute room position.

### Local gateway

The browser uses `/api/workspaces/*`, `/api/colab/*` and `/api/data` on localhost. The local API,
workspace states, retries, CPU preview, exports and compact clip fields are documented completely in
[dashboard/README.md](../../dashboard/README.md). Compact clip JSON extends the previous format with
scene and quality fields; older results return null for absent metadata. No key is exposed there.


### Idle-shutdown support

`GET /health` also returns `active_transfers` and `last_activity` (a process-monotonic activity token).
Uploads to `/jobs` and streamed downloads from `/jobs/{id}/files/{name}` update these fields.
The whole response body is tracked; `/health` and job polling do not extend the GPU session.
The one-cell Colab helper combines transfer activity with queued/running job and render states,
then requests `google.colab.runtime.unassign()` after five idle minutes.
Optional `SPECTATOR_MODEL_CACHE` directs WiLoR weights into the persistent model cache; without it,
CLI/model-loading paths retain their existing behavior.
