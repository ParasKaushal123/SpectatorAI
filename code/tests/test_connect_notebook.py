"""One-cell startup, remembered setup, dependency caching, and safe idle shutdown; no real GPU/network."""
import asyncio
from contextlib import chdir, redirect_stdout
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import types
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from server.activity import TransferMiddleware, activity


def test_idle():
    Guard = runpy.run_path(str(ROOT / "code/tools/idle_guard.py"))["IdleGuard"]
    g = Guard(300, now=0)
    assert not g.expired({"last_activity": 1}, [], now=0)
    assert not g.expired({"last_activity": 1}, [], now=299)
    assert g.expired({"last_activity": 1}, [], now=300), "Status polls must not keep a GPU alive"
    assert not g.expired({"last_activity": 1}, [{"status": "running"}], now=1000)
    assert not g.expired({"last_activity": 1}, [{"status": "done", "render": {"status": "running"}}], now=2000)
    assert not g.expired({"last_activity": 1, "active_transfers": 1}, [], now=3000)
    assert not g.expired({"last_activity": 2}, [], now=3100), "Transfer completion restarts the idle grace"
    assert not g.expired({"last_activity": 2}, [], now=3399)
    assert g.expired({"last_activity": 2}, [], now=3400)


async def test_stream_activity():
    gate = asyncio.Event()

    async def streaming_app(scope, receive, send):
        await gate.wait()

    async def unused(*args):
        return None

    app = TransferMiddleware(streaming_app)
    task = asyncio.create_task(app({"type": "http", "path": "/jobs/abc/files/render.mp4", "method": "GET"}, unused, unused))
    await asyncio.sleep(0)
    assert activity.snapshot()["active_transfers"] == 1, "The entire streamed body must count as active"
    gate.set()
    await task
    assert activity.snapshot()["active_transfers"] == 0
    token = activity.snapshot()["last_activity"]
    await app({"type": "http", "path": "/health", "method": "GET"}, unused, unused)
    await app({"type": "http", "path": "/jobs", "method": "GET"}, unused, unused)
    assert activity.snapshot()["last_activity"] == token


def test_notebook(td):
    import torch
    notebook = json.loads((ROOT / "CONNECT_COLAB.ipynb").read_text(encoding="utf8"))
    cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(cells) == 1 and cells[0]["metadata"]["cellView"] == "form"
    source = cells[0]["source"].replace('Path("/content/drive")', f"Path({str(td / 'drive')!r})")
    source = source.replace('Path("/content/spectatorai-dashboard")', f"Path({str(td / 'runtime')!r})")
    uploads, releases, starts = [], [], []

    def mount(path):
        (Path(path) / "MyDrive").mkdir(parents=True)

    def upload():
        uploads.append(True)
        return {"SpectatorAI-Colab.zip": (ROOT / "SpectatorAI-Colab.zip").read_bytes()}

    fake = types.ModuleType("google.colab")
    fake.drive = types.SimpleNamespace(mount=mount)
    fake.files = types.SimpleNamespace(upload=upload)
    fake.runtime = types.SimpleNamespace(unassign=lambda: releases.append(True))
    with chdir(td), patch.dict(sys.modules, {"google.colab": fake}), patch.object(torch.cuda, "is_available", return_value=True), \
            patch.object(torch.cuda, "get_device_name", return_value="synthetic startup check"), \
            patch.object(runpy, "run_path", return_value={"start": lambda root, idle_seconds: starts.append(idle_seconds)}):
        exec(compile(source, "CONNECT_COLAB.ipynb", "exec"), {})
        assert len(uploads) == 1 and len(releases) == 1
        exec(compile(source, "CONNECT_COLAB.ipynb", "exec"), {})
        assert len(uploads) == 1, "Later Play presses must reuse the saved setup"
        assert len(releases) == 2 and starts == [300, 300]
        secret = json.loads((td / "runtime/connection.json").read_text())["api_key"]
        output = io.StringIO()
        with patch.object(runpy, "run_path", side_effect=RuntimeError("simulated startup failure " + secret)), \
                patch("time.sleep"), redirect_stdout(output):
            exec(compile(source, "CONNECT_COLAB.ipynb", "exec"), {})
        printed = output.getvalue()
        assert printed.index("STARTUP FAILED") < printed.index("Releasing the Colab runtime")
        assert secret not in printed and "[hidden]" in printed
        report = td / "drive/MyDrive/SpectatorAI-Private/startup-error.txt"
        assert report.is_file() and secret not in report.read_text()
        assert len(releases) == 3, "A startup failure must also release the runtime"
    assert (ROOT / "SpectatorAI-Colab.zip").is_file(), "Notebook simulation must not remove the real upload bundle"


def test_install_cache(td):
    start = runpy.run_path(str(ROOT / "code/tools/colab_start.py"))["start"]
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if "--wheel-dir" in command:
            wheel_dir = Path(command[command.index("--wheel-dir") + 1])
            (wheel_dir / "synthetic-1.0-py3-none-any.whl").touch()
        return subprocess.CompletedProcess(command, 0)

    def prepare(root):
        (root / "code").mkdir(parents=True)
        (root / "code/requirements.txt").write_text("numpy\ntorch\ntorchvision\ntransformers\n")
        (root / "connection.json").write_text(json.dumps({"model_cache": str(td / "cache")}))

    first = td / "first-runtime"
    prepare(first)
    with patch.dict(os.environ), patch.object(subprocess, "run", side_effect=run), \
            patch.object(runpy, "run_path", return_value={"run_session": lambda *a, **k: "idle"}):
        assert start(first) == "idle"
        assert sum("--wheel-dir" in c for c in commands) == 3
        commands.clear()
        assert start(first) == "idle" and not commands, "Same-runtime restart must skip installs"
        second = td / "fresh-runtime"
        prepare(second)
        start(second)
        assert not any("--wheel-dir" in c for c in commands), "A fresh runtime must reuse cached Git-package wheels"


def main():
    test_idle()
    asyncio.run(test_stream_activity())
    with tempfile.TemporaryDirectory(dir=ROOT / "dashboard/data", prefix="one-cell-check-") as folder:
        test_notebook(Path(folder))
        test_install_cache(Path(folder))
    print("ONE-CELL CHECKS PASSED: remembered setup, cached installs/wheels, idle timeout, active work/transfer protection, runtime release on exit/failure")


if __name__ == "__main__":
    main()
