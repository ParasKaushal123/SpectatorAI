# SpectatorAI: one Play cell

## First time

On a fresh clone, run `setup.bat`, then `.venv/Scripts/python.exe tools/prepare_connection.py`,
and start the website with `run.bat`. This creates your private `SpectatorAI-Colab.zip` locally;
the ZIP and connection credentials are intentionally excluded from Git.

1. Open CONNECT_COLAB.ipynb in Google Colab and choose a GPU runtime (A100 if available).
2. Press the single Play button, labelled Start SpectatorAI.
3. Allow Drive access when Google asks. Upload SpectatorAI-Colab.zip from this folder when prompted.
4. Wait for CONNECTED, then upload clips at http://127.0.0.1:8000.

The private ZIP and downloaded model/package caches are remembered in My Drive / SpectatorAI-Private.
Your uploaded videos are not saved to Drive by this notebook. The notebook code contains no API keys.

## Later

Open the saved notebook and press Play. It reuses the saved ZIP and caches; no code edits, keys or URLs.
Google may ask for Drive access again on a new runtime. A fresh runtime still needs packages installed
and models loaded, so startup cannot be guaranteed instant. Keep the cell running while using the site.

The GPU releases automatically after five minutes with no queued/running processing, rendering or uploads/downloads.
Status polling does not keep it alive. Stopping the cell or a setup failure also requests runtime release.
If Colab reports that release failed, use Runtime > Disconnect and delete runtime manually.
Saved workspaces continue to open locally after the GPU stops. Press Play again to process more clips.

Double-click run.bat to open the local dashboard. That local app does not consume Colab credits.
Do not leave a GPU allocated for long periods; the idle timer is a safeguard, not per-clip billing.

When an updated notebook needs a new setup ZIP, it asks you to upload the latest file automatically.
Select the newly prepared SpectatorAI-Colab.zip from this project folder.
If startup fails, copy the STARTUP FAILED error block. A redacted copy is saved in Drive /
SpectatorAI-Private / startup-error.txt before the GPU is released.
