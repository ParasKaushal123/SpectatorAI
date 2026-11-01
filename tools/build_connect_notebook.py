"""Build the single visible Play cell. No API keys are embedded in notebook code."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETUP_VERSION = "2026-11-01-startup-fix-2"

SOURCE = r'''#@title ▶ Start SpectatorAI — connect GPU, then upload clips on the website
idle_minutes = 5 #@param {type:"slider", min:1, max:30, step:1}

import io, json, zipfile
from pathlib import Path, PurePosixPath
import runpy
import traceback
import time
import torch
from google.colab import drive, files, runtime

_redactions = []
EXPECTED_SETUP_VERSION = "__SETUP_VERSION__"
try:
    if not torch.cuda.is_available():
        raise RuntimeError("Select Runtime → Change runtime type → GPU, then press Play again.")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    DRIVE_ROOT = Path("/content/drive")
    if not (DRIVE_ROOT / "MyDrive").is_dir():
        print("Connecting Drive to remember your setup and cache model downloads.", flush=True)
        drive.mount(str(DRIVE_ROOT))
    PRIVATE = DRIVE_ROOT / "MyDrive" / "SpectatorAI-Private"
    PRIVATE.mkdir(parents=True, exist_ok=True)
    BUNDLE = PRIVATE / "SpectatorAI-Colab.zip"

    _needs_setup = not BUNDLE.is_file()
    if not _needs_setup:
        try:
            with zipfile.ZipFile(BUNDLE) as _saved:
                _needs_setup = "setup-version.txt" not in _saved.namelist() or _saved.read("setup-version.txt").decode().strip() != EXPECTED_SETUP_VERSION
        except (OSError, zipfile.BadZipFile):
            _needs_setup = True
    if _needs_setup:
        print("Setup/update needed: upload the latest SpectatorAI-Colab.zip from your computer.", flush=True)
        uploaded = files.upload()
        archives = [(name, data) for name, data in uploaded.items() if name.lower().endswith(".zip")]
        if len(archives) != 1:
            raise ValueError("Choose exactly one file: SpectatorAI-Colab.zip")
        name, payload = archives[0]
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            if "connection.json" not in z.namelist() or "code/tools/colab_start.py" not in z.namelist():
                raise ValueError("Use the updated SpectatorAI-Colab.zip prepared with this notebook.")
            if "setup-version.txt" not in z.namelist() or z.read("setup-version.txt").decode().strip() != EXPECTED_SETUP_VERSION:
                raise ValueError("That ZIP is an older version. Select the updated SpectatorAI-Colab.zip from the project folder.")
        _bundle_tmp = BUNDLE.with_suffix(".tmp")
        _bundle_tmp.write_bytes(payload)
        _bundle_tmp.replace(BUNDLE)
        if Path(name).name == name:
            Path(name).unlink(missing_ok=True)
        del uploaded, archives, payload
        print("Setup remembered. Future starts reuse this file.", flush=True)

    SESSION = Path("/content/spectatorai-dashboard")
    SESSION.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BUNDLE) as z:
        if "code/tools/colab_start.py" not in z.namelist():
            raise ValueError("The saved connection ZIP is old. Replace SpectatorAI-Colab.zip in Drive → SpectatorAI-Private.")
        for member in z.infolist():
            parts = PurePosixPath(member.filename).parts
            if not parts or parts[0] not in ("code", "connection.json", "setup-version.txt") or ".." in parts or chr(92) in member.filename:
                raise ValueError("Unexpected path in the setup file")
            dest = (SESSION / member.filename).resolve()
            if not dest.is_relative_to(SESSION):
                raise ValueError("Invalid setup file path")
            if not member.is_dir():
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(z.read(member))
    connection_file = SESSION / "connection.json"
    connection = json.loads(connection_file.read_text())
    if not connection.get("api_key") or not connection.get("topic"):
        raise ValueError("The setup file is missing connection settings")
    _redactions = [connection["api_key"], connection["topic"]]
    connection["model_cache"] = str(PRIVATE / "model-cache")
    connection_file.write_text(json.dumps(connection))
    connection_file.chmod(0o600)
    del connection

    print("Starting SpectatorAI. The first start and a fresh runtime still need setup time.", flush=True)
    runpy.run_path(str(SESSION / "code/tools/colab_start.py"))["start"](SESSION, idle_seconds=idle_minutes * 60)
except Exception:
    # Print and persist BEFORE unassign(): releasing the VM can suppress Python's pending traceback.
    _diagnostic = traceback.format_exc()
    if "SESSION" in globals():
        _server_log = SESSION / "logs/server.log"
        if _server_log.is_file():
            _diagnostic += "\n\n--- GPU server log ---\n" + "\n".join(_server_log.read_text(errors="replace").splitlines()[-120:])
    for _secret in _redactions:
        _diagnostic = _diagnostic.replace(_secret, "[hidden]")
    print("\nSTARTUP FAILED — copy the error below:\n\n" + _diagnostic, flush=True)
    if "PRIVATE" in globals():
        try:
            (PRIVATE / "startup-error.txt").write_text(_diagnostic, encoding="utf-8")
            print("Error saved in Drive / SpectatorAI-Private / startup-error.txt", flush=True)
        except OSError:
            pass
    time.sleep(2)
finally:
    # Stopping this cell, idle shutdown, or a startup failure must release the allocation, not merely stop Python.
    print("Releasing the Colab runtime to stop GPU usage…", flush=True)
    try:
        runtime.unassign()
    except Exception:
        print("If Colab still shows a connected GPU, choose Runtime → Disconnect and delete runtime.", flush=True)
'''


def build():
    source = SOURCE.replace("__SETUP_VERSION__", SETUP_VERSION)
    import ast
    ast.parse(source)
    notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source":
             "# SpectatorAI: press Play, then upload on the website\n\n"
             "Select a GPU runtime, then press the Play button below.\n\n"
             "**First time:** allow Drive access and upload `SpectatorAI-Colab.zip` once. "
             "It is saved privately in Drive for future starts, along with pretrained model downloads. "
             "Your videos are not saved to Drive by this notebook.\n\n"
             "**Later:** press Play. Colab may ask for Drive access again. "
             "A fresh runtime still needs dependency setup and model loading; startup is not instant.\n\n"
             "Wait for **CONNECTED**, then upload videos in your local dashboard. "
             "After five idle minutes the cell releases the GPU automatically. "
             "Stopping the cell also releases it. Active processing, rendering and transfers keep it running.\n"},
            {"cell_type": "code", "metadata": {"cellView": "form"}, "execution_count": None,
             "outputs": [], "source": source},
        ],
        "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
                     "kernelspec": {"name": "python3", "display_name": "Python 3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 0,
    }
    path = ROOT / "CONNECT_COLAB.ipynb"
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Built CONNECT_COLAB.ipynb: one visible Play cell; no embedded credentials.")


if __name__ == "__main__":
    build()
