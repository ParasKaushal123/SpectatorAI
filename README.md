# SpectatorAI

### Egocentric hand intelligence for 3D replay, spatial understanding and robot-ready motion

> **A video is more than pixels.** SpectatorAI turns point-of-view footage into an interactive 3D record of where the wearer's hands moved, what they saw, and how a robot could replay the task.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=111111)
![Three.js](https://img.shields.io/badge/Three.js-3D%20replay-000000?logo=threedotjs&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-local%20gateway-009688?logo=fastapi&logoColor=white)

## The idea

Teaching a robot, reviewing a manual task, or understanding an operator's movement usually means replaying flat video and guessing what happened in space. SpectatorAI makes that evidence spatial.

From a head-mounted, fisheye or ordinary phone clip, it identifies the wearer's hands, reconstructs their movement in 3D, keeps a quality record for every frame, and presents the result in a browser workspace. For workstation captures, it replays detailed hand meshes inside a reconstructed 3D room, next to the synchronized source video and an optional two-arm robot replay.

The goal is not to replace human judgement. It is to make hand motion, context and uncertainty easy to inspect before that motion is used for training, review or automation.

## How it works

1. **Find every possible hand.** WiLoR proposes hand candidates in every frame, with fisheye-aware camera calibration from GeoCalib.
2. **Follow, then decide.** Candidates are linked into tracks over time. Each track collects evidence (distance, where the arm enters the frame, a CLIP "is this a hand?" check, handedness and motion) and is labelled once as the wearer's left hand, right hand or rejected, with a logged reason.
3. **Reconstruct in 3D.** Accepted hands are rebuilt as full MANO hand meshes. Short gaps are filled and clearly marked as filled, never passed off as observed.
4. **Rate the footage.** Every second gets a quality score based on hand visibility, blur, head motion and confidence, so usable training segments are separated from unusable ones.
5. **Place it in the world.** SLAM recovers the camera path and a gravity-aligned room frame, so hand motion is in fixed real-world coordinates instead of moving with the wearer's head.
6. **Replay it.** The workstation is reconstructed in 3D, and the hand motion is retargeted to two simulated Franka Panda arms in MuJoCo, with each frame graded for whether a robot could follow it.

## What you can explore

| Module | What it gives you |
|---|---|
| **Workspace** | Upload a clip, follow processing live, scrub synchronized video and 3D hands, inspect quality and export results. |
| **Library** | Reopen local workspaces and saved tracking results without reprocessing them. |
| **Dataset** | Browse output IDs, search the catalogue and play the render for each ID. |
| **Room replay** | Orbit, pan and zoom a reconstructed workstation with hand meshes, head motion and a source-video overlay. |
| **Robot replay** | Open a robot viewport with a shared timeline, per-frame copyability grades and downloadable trajectories. |

## System overview

```mermaid
flowchart LR
    A[Egocentric video] --> B[Local dashboard]
    B --> C[Persistent local queue]
    C --> D[Colab GPU tracking]
    D --> E[3D hands + quality data]
    E --> F[Interactive workspace]
    G[Room and robot assets] --> H[Room replay]
    H --> F
```

The dashboard runs locally. It sends a tracking job to a short-lived Colab GPU only when a clip is submitted, then saves playback data on your computer. Room and robot results replay locally, and opening them does **not** start a GPU runtime.

## Tech stack

| Layer | Technologies |
|---|---|
| Web experience | React 18, TypeScript, Vite, React Three Fiber / Three.js |
| Local application | FastAPI, Python, persistent job queue, local media streaming |
| Tracking | PyTorch, WiLoR, SAM 2, CLIP, GeoCalib, MANO |
| Spatial | NVIDIA ViPE (SLAM), MoGe-2 depth, Open3D TSDF fusion |
| Robotics | MuJoCo, Franka Panda, mink inverse kinematics |
| Compute | Google Colab GPU runtime with automatic idle release |
| Exports | JSONL hand data, wrist/gripper CSV, quality and scene metadata, rendered video |

## Quick start

### 1. Clone and set up

```powershell
git clone <your-repository-url>
cd ps4-hand-tracking
.\setup.bat
.\run.bat
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). `setup.bat` creates the Python environment, installs frontend dependencies and builds the dashboard. The first setup can take a few minutes.

### 2. Optional: connect a Colab GPU for new tracking jobs

```powershell
.venv\Scripts\python.exe tools\prepare_connection.py
```

Upload `CONNECT_COLAB.ipynb` to Google Colab, choose a GPU runtime and run the single **Start SpectatorAI** cell. When asked, upload the generated private `SpectatorAI-Colab.zip`. Once the dashboard shows **Colab connected**, choose **Add video**.

The notebook releases the GPU after a period of inactivity. Library, Dataset and Room results stay available while it is offline.

## Typical workflow

1. Start the local dashboard with `run.bat`.
2. Connect Colab only when you want to process a new clip.
3. Upload a short video from **Workspace**.
4. Follow the queue and processing status, then open the result when it is ready.
5. Scrub the clip, inspect the 3D motion and quality lanes, then export data or a render.
6. Use **Room** for spatial and robot playback, or **Dataset** to review final videos.

## Responsible use

Observed, filled and out-of-view hand samples stay distinguishable in the data and quality lanes. Quality grades are curation aids, not calibrated accuracy claims. Robot copyability scores are simulation estimates, **not** robot safety certification. Verify the task, workspace, hardware limits and safety controls independently before using any result on a physical system.

## Privacy

This repository contains source code, configuration, tests and notebooks only. It does not include footage, renders, room scans, model weights, API keys or local job data, and `.gitignore` keeps those out. API keys are read from environment variables, never from code.

## Repository layout

```text
dashboard/frontend/     React dashboard and 3D viewers
dashboard/backend/      Local FastAPI gateway, queue and media routes
code/ps4v4/             Hand tracking, quality rating and export pipeline
code/ps5world/          SLAM and world-frame alignment
code/room/              3D room reconstruction and web export
code/robot/             Robot retargeting and copyability scoring
code/notebooks/         Colab notebooks
code/tests/             Synthetic end-to-end tests
tools/                  Setup, connection and helper scripts
```

## More detail

- [Architecture and API reference](ARCHITECTURE.md)
- Backend API docs while running: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

Built at a hackathon as a spatial interface for understanding skilled human hand motion, from video to 3D evidence to robot-ready trajectories.