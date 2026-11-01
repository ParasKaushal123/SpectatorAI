# SpectatorAI

A local video workspace for egocentric 3D hand tracking, powered by the PS4 pipeline on a Colab GPU.
Upload a clip, inspect synchronized video and interactive 3D hands, review quality timelines, and export results.
Saved results remain available after Colab disconnects. The separate PS5 tools provide a SLAM workflow.

## Start

1. Install Python, Node.js/npm, and Git. On Windows, run `setup.bat` once.
2. Generate your private connection bundle:
   ```powershell
   .venv/Scripts/python.exe tools/prepare_connection.py
   ```
3. Run `run.bat` and open <http://127.0.0.1:8000>.
4. Open `CONNECT_COLAB.ipynb` in Colab, select a GPU, and press Play. Allow Drive access and upload the generated `SpectatorAI-Colab.zip` when prompted.
5. Wait for the connection to be ready, then upload a video on the website.

See [START_HERE.md](START_HERE.md) for notebook usage and idle shutdown, and [dashboard/README.md](dashboard/README.md) for detailed setup, API routes, and tests.
Linux/macOS users can follow the dashboard guide and use `run.sh`.

## Project layout

| Folder | Contents |
|---|---|
| `code/` | Tracking pipeline, Colab server, SLAM tools, configurations, notebooks, and tests |
| `dashboard/backend/` | Local API, processing queue, connection discovery, and dataset catalogue |
| `dashboard/frontend/` | React, TypeScript, and Three.js website |
| `tools/` | Connection bundle and notebook generators |

[code/README.md](code/README.md) documents the pipeline CLI and separate SLAM workflow.
The Room page is a placeholder; room processing is not part of the dashboard upload flow.

## Local data

Library stores processed workspaces locally. Dataset lists videos from the sibling `outputs/ps4v5_FINAL_renders/`
and `outputs/ps4v5_P2_renders/` folders when present. These folders are optional and are not included in Git.
An empty installation starts without example footage or saved results.

Credentials, private Colab bundles, datasets, videos, results, model weights, dependencies, build output,
and local planning notes are ignored by Git. Generate your own bundle after cloning; keep `dashboard/.env`
and `SpectatorAI-Colab.zip` private. `.env.example` and package lockfiles remain part of the project.
