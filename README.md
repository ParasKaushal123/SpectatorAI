# SpectatorAI

SpectatorAI turns egocentric video into an interactive workspace for inspecting the wearer's hands in 3D.
The local website sends tracking jobs to a Colab GPU, then keeps the results locally for playback and export.
Separate tools reconstruct a room and retarget hand motion to two Franka Panda robot arms.

## What the app does

- **Workspace:** upload a video, watch progress, play synchronized video and 3D hands, inspect quality lanes,
  and export tracking data or a rendered view.
- **Library:** reopen processed workspaces and locally imported results.
- **Dataset:** browse IDs and play videos from two optional output folders.
- **Room:** replay hands, head and gaze inside an imported 3D workstation. Pan, orbit, zoom, or rotate the camera.
  Open robot replay for a second interactive view with independent camera controls and a shared timeline.
- **Robot replay:** compare source-provided copyability grades, inspect timeline segments, and download joint trajectories.

Room reconstruction and robot retargeting are separate offline workflows. Opening their results does not rerun them.
A fresh clone contains source code, configuration and tests, without private footage, model weights or saved results.

## Requirements

Windows setup uses Python 3.12 (with a fallback to `python`), Node.js/npm, Git, and internet access for installation.
The existing project has also been exercised with Python 3.11. A Google account and Colab GPU runtime are required
for the dashboard's remote tracking workflow. Node.js must support the checked-in Vite 6 toolchain.

The local dashboard does not need a GPU. Its setup also installs and downloads the models used by the optional
CPU preview, so initial setup can take time and disk space. Full SLAM/room generation has separate dependencies;
see [ARCHITECTURE.md](ARCHITECTURE.md).

## First-time setup on Windows

Run these from the repository root in PowerShell:

```powershell
.\setup.bat
.venv/Scripts/python.exe tools/prepare_connection.py
.\run.bat
```

`setup.bat` creates the Python environment, installs dependencies, caches local preview models and builds the website.
`prepare_connection.py` creates a matching private `dashboard/.env` and `SpectatorAI-Colab.zip`. Running it again
preserves existing connection credentials. Keep both files private; neither belongs on GitHub.

The website opens at <http://127.0.0.1:8000>. Leave its terminal running while using it.

## Connect Colab and process a video

1. Upload `CONNECT_COLAB.ipynb` to Google Colab and select a GPU runtime. An A100 is suitable when available.
2. Press Play on the single **Start SpectatorAI** cell.
3. Allow Drive access. When prompted, upload the private `SpectatorAI-Colab.zip` generated on this computer.
4. Wait for **CONNECTED** in the notebook and the website's connection indicator.
5. Choose **Add video** on the website. Tracking starts automatically; queued clips are processed in order.
6. Open the finished workspace from Library to play, scrub, orbit the hands, inspect quality and export.

Keep the notebook cell running during processing. The one-cell workflow processes up to the first 60 seconds
of an uploaded clip. Original files can be up to 500 MB locally; tunnel limits and upload speed can impose
lower practical limits. Preparing the rendered MP4 runs in the background, so the interactive workspace may
be ready before that download completes.

The private setup ZIP and package/model caches are remembered under `MyDrive/SpectatorAI-Private`.
A fresh runtime still needs installation and model loading. Later starts reuse the saved ZIP; after an update,
generate a new bundle and upload it when the notebook requests one.

The helper requests runtime release after five minutes without active jobs, rendering or file transfers.
Status polling does not keep it alive. Stopping the cell or a startup failure also requests release.
If release fails, use Colab's **Runtime → Disconnect and delete runtime**. Saved local results remain available
when Colab is disconnected. The helper reduces idle GPU usage; it does not guarantee billing or credit usage.

## Room and robot playback

Import existing results into this layout; do not generate them merely to open the page:

```text
dashboard/
  raw_clips/
    <clip>.mp4
  results/room_twin/<clip>/
    room_stats.json
    room_video.mp4                 optional fly-through download
    web/
      motion.json
      room.glb
      hands_left.bin              when motion.json declares a left mesh
      hands_right.bin             when motion.json declares a right mesh
      robot.json                  optional robot replay metadata
      robot.glb
      robot_motion.bin
      robot_traj.json
```

Extract the contents of `room_web.zip` into `web/`; avoid an extra nested `web/web/` directory.
Put the original video in `dashboard/raw_clips/` with exactly the room's clip ID as its filename.
Then open **Room**. A missing original video does not prevent independent 3D playback.

- Drag to orbit, Shift/right-drag to pan, and scroll to zoom. **Pan** switches left-drag to panning.
- **Rotate camera**, **Follow head**, camera presets and **Reset view** control each viewport independently.
- **Show robot replay** opens the second view and starts shared playback; hiding it keeps the playhead position.
- **Original video** shows a small synchronized overlay in the SLAM/hand view's upper-right corner.
- The slider and playback controls drive the hands and robot together. Space toggles play; arrows step frames;
  Shift+arrows move about one second when keyboard focus is outside form controls.
- Robot lanes seek to their grades. The right panel shows source-provided scores and export actions.
- **Record my view** saves the current 3D canvas to WebM; the fly-through download uses the supplied MP4.

Room replay preserves the imported metre-scale, Z-up coordinate frame. The regular Workspace uses a different,
head-stable frame. Its hand surface can be an approximation fitted to saved joints when detailed mesh data
is unavailable. Room uses the supplied detailed mesh binaries when present.

## Dataset page

Dataset reads these folders next to the repository, without copying them into Git:

```text
<parent>/outputs/ps4v5_FINAL_renders/
<parent>/outputs/ps4v5_P2_renders/
```

Video filenames become IDs. If an ID has multiple versions, the player lets you select a folder/version.
These folders are optional; the page is empty until videos are supplied.

## Troubleshooting

| Symptom | What to check |
|---|---|
| Colab offline | Start the notebook, wait for CONNECTED, and check Settings. The notebook and local ZIP must use matching credentials. |
| Upload appears idle | Check its stage and queue message; an earlier clip may be processing or uploading. |
| Notebook stops during startup | Read STARTUP FAILED. A redacted log is saved to `SpectatorAI-Private/startup-error.txt` in Drive. |
| Old interface after updating | Rebuild with `npm run build --prefix dashboard/frontend`, then refresh with Ctrl+F5. |
| Port 8000 is already occupied | Stop the unrelated server, then run `run.bat` again. |
| Room missing assets | Follow the named file/path in the error and check the import layout above. |
| No robot replay | Add the matching robot files to the room's `web/` folder. Hands-only rooms remain usable. |

## Development and publication

[ARCHITECTURE.md](ARCHITECTURE.md) is the single technical reference for system flow, CLI commands, APIs,
configuration, tests and limitations. These are the only two maintained project documentation files.
Superseded notes and prompts are kept in the ignored `.local-docs/` folder on the development computer.

`.gitignore` excludes credentials, media, derived results, model downloads, generated fixtures, build output
and dependencies. Synthetic fixtures are regenerated by their test scripts. Commit source, the environment
example, configurations, notebooks with cleared outputs, and `package-lock.json`.

Before publishing, inspect `git status` and the staged diff. Do not force-add `.env`, private Colab bundles,
footage or result folders. Ignoring a file does not erase it from earlier commits.
