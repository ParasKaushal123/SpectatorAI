"""Keep an interactive Colab processing session connected to the local dashboard."""
from __future__ import annotations

import json
import os
import re
import runpy
import socket
import subprocess
import shutil
import time
import urllib.request
from pathlib import Path


def read_json(url, headers=None, timeout=10):
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def publish(url, topic, gpu):
    payload = json.dumps({"url": url, "started": time.time(), "gpu": gpu}).encode()
    request = urllib.request.Request("https://ntfy.sh/" + topic, data=payload, method="POST",
                                     headers={"Content-Type": "text/plain"})
    with urllib.request.urlopen(request, timeout=15) as response:
        response.read()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _tail(path, secret):
    return "\n".join(path.read_text(errors="replace").splitlines()[-15:]).replace(secret, "[hidden]")


def run_session(connection_file, idle_seconds=300):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Choose Runtime → Change runtime type → GPU, then Run all again.")
    connection_file = Path(connection_file).resolve()
    settings = json.loads(connection_file.read_text())
    if not settings.get("api_key") or not re.fullmatch(r"[A-Za-z0-9_-]{20,}", settings.get("topic", "")):
        raise ValueError("Upload the SpectatorAI-Colab.zip prepared on your computer.")
    root = connection_file.parent
    code = root / "code"
    port = free_port()
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    server_log, tunnel_log = logs / "server.log", logs / "tunnel.log"
    binary = root / "cloudflared"
    if not binary.exists():
        urllib.request.urlretrieve("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", binary)
        binary.chmod(0o700)
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    batch = 128 if memory_gb >= 35 else 64 if memory_gb >= 22 else 16
    overrides = {"recon": {"wilor": {"batch": batch}}, "proposals": {"batch": 32 if memory_gb >= 22 else 8}}
    env = dict(os.environ, PS4_API_KEY=settings["api_key"], PS4_JOBS_DIR=str(root / "jobs"),
               PS4_OVERRIDES=json.dumps(overrides),
               PS4_MAX_SECONDS="60", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="0")
    if settings.get("model_cache"):
        cache_root = Path(settings["model_cache"])
        env.update(SPECTATOR_MODEL_CACHE=str(cache_root), HF_HOME=str(cache_root / "huggingface"),
                   TORCH_HOME=str(cache_root / "torch"))
    # Keep credentials in runtime environment, never in command arguments or notebook outputs.
    server_file = server_log.open("w")
    tunnel_file = None
    server = subprocess.Popen([os.sys.executable, "-m", "uvicorn", "server.app:app", "--host", "127.0.0.1",
                               "--port", str(port)], cwd=code, env=env, stdout=server_file, stderr=subprocess.STDOUT)
    tunnel = None
    try:
        print("Loading WiLoR, CLIP and GeoCalib on", torch.cuda.get_device_name(0), flush=True)
        print("First connection downloads model weights. Keep this cell running.", flush=True)
        health, last_notice = None, 0
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError("Pipeline server stopped:\n" + _tail(server_log, settings["api_key"]))
            try:
                health = read_json(f"http://127.0.0.1:{port}/health", timeout=5)
            except OSError:
                health = None
            if health and health.get("warm_error"):
                raise RuntimeError("Model setup failed: " + health["warm_error"].replace(settings["api_key"], "[hidden]")
                                   + "\n" + _tail(server_log, settings["api_key"]))
            if health and health.get("warm"):
                # Verify the same key the dashboard will use, before announcing readiness.
                read_json(f"http://127.0.0.1:{port}/jobs", {"X-API-Key": settings["api_key"]})
                break
            if time.monotonic() - last_notice > 30:
                print("Preparing GPU models…", flush=True)
                last_notice = time.monotonic()
            time.sleep(2)
        else:
            raise TimeoutError("GPU warm-up timed out. Stop this cell, reconnect the Colab runtime, and Run all.")

        guard = runpy.run_path(str(code / "tools/idle_guard.py"))["IdleGuard"](idle_seconds)
        print(f"The GPU will switch off after {int(idle_seconds / 60)} minutes without processing or transfers.", flush=True)
        url, published_at = None, 0
        print("GPU ready. Connecting your dashboard…", flush=True)
        health_errors = 0
        while True:
            if server.poll() is not None:
                raise RuntimeError("Pipeline server stopped:\n" + _tail(server_log, settings["api_key"]))
            try:
                local_health = read_json(f"http://127.0.0.1:{port}/health", timeout=10)
                local_jobs = read_json(f"http://127.0.0.1:{port}/jobs", {"X-API-Key": settings["api_key"]}, timeout=10)
                health_errors = 0
            except OSError:
                health_errors += 1
                if health_errors >= 3:
                    raise RuntimeError("The GPU server is unresponsive; stopping to protect compute credits.")
                time.sleep(2)
                continue
            if guard.expired(local_health, local_jobs):
                print("IDLE — stopping the connection and releasing the GPU to save credits.", flush=True)
                return "idle"
            if tunnel is None or tunnel.poll() is not None:
                if tunnel_file:
                    tunnel_file.close()
                tunnel_file = tunnel_log.open("w")
                tunnel = subprocess.Popen([str(binary), "tunnel", "--url", f"http://127.0.0.1:{port}",
                                           "--no-autoupdate"], stdout=tunnel_file, stderr=subprocess.STDOUT)
                url, published_at = None, 0
                for _ in range(90):
                    match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", tunnel_log.read_text(errors="replace"))
                    if match:
                        url = match.group(0)
                        break
                    if tunnel.poll() is not None:
                        break
                    time.sleep(1)
                if not url:
                    print("Retrying tunnel connection…", flush=True)
                    if tunnel.poll() is None:
                        tunnel.terminate()
                        tunnel.wait(timeout=15)
                    time.sleep(3)
                    continue
            if time.monotonic() - published_at >= 600 or not published_at:
                try:
                    public_health = read_json(url + "/health", timeout=15)
                    if not public_health.get("ok") or not public_health.get("warm"):
                        raise RuntimeError("GPU is not ready")
                    publish(url, settings["topic"], public_health.get("gpu"))
                    if not published_at:
                        print("CONNECTED — return to the dashboard and upload your clips. Queued videos start automatically.", flush=True)
                        print("Keep this Colab tab and this cell running while processing.", flush=True)
                        print("Connection URL (only needed if automatic discovery is blocked):", url, flush=True)
                    published_at = time.monotonic()
                except Exception as exc:
                    print("Connection discovery retry:", type(exc).__name__, flush=True)
                    time.sleep(5)
                    continue
            time.sleep(5)
    except KeyboardInterrupt:
        print("Colab connection stopped. Uploaded videos remain on your computer.", flush=True)
    finally:
        for process in (tunnel, server):
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        server_file.close()
        if tunnel_file:
            tunnel_file.close()
        jobs = (root / "jobs").resolve()
        if jobs.is_relative_to(root) and jobs != root:
            shutil.rmtree(jobs, ignore_errors=True)
        connection_file.unlink(missing_ok=True)
