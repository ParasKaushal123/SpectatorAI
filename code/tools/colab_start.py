"""Install once per runtime and start the session using persistent model/pip caches."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys


def start(root, idle_seconds=300):
    root = Path(root)
    code = root / "code"
    connection_file = root / "connection.json"
    connection = json.loads(connection_file.read_text())
    cache = Path(connection["model_cache"])
    cache.mkdir(parents=True, exist_ok=True)
    requirements = (code / "requirements.txt").read_text()
    identity = hashlib.sha256((requirements + sys.version + "spectator-start-v1-transformers4").encode()).hexdigest()
    stamp = root / "installed.stamp"
    if not stamp.exists() or stamp.read_text() != identity:
        print("First start in this runtime: preparing dependencies. Cached downloads will be reused.", flush=True)
        packages = []
        for line in requirements.splitlines():
            item = line.split("#", 1)[0].strip()
            if item and item not in ("torch", "torchvision", "transformers"):
                packages.append(item)
        packages += ["transformers>=4.45,<5", "fastapi", "uvicorn[standard]", "python-multipart", "setuptools", "wheel"]

        def install(*args):
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--cache-dir", str(cache / "pip"), *args],
                           check=True, timeout=900)

        def install_source(name, url, no_deps=False):
            # Cache wheels for Git-only packages too; pip otherwise clones them on each new runtime.
            wheel_dir = cache / "source-wheels" / name
            wheel_dir.mkdir(parents=True, exist_ok=True)
            wheels = sorted(wheel_dir.glob("*.whl"))
            if not wheels:
                subprocess.run([sys.executable, "-m", "pip", "wheel", "-q", "--no-deps", "--no-build-isolation",
                                "--wheel-dir", str(wheel_dir), url], check=True, timeout=900)
                wheels = sorted(wheel_dir.glob("*.whl"))
            if not wheels:
                raise RuntimeError(f"Could not prepare {name}; no installable wheel was produced")
            install(*(["--no-deps"] if no_deps else []), str(wheels[0]))

        install(*packages)
        install_source("wilor", "git+https://github.com/warmshao/WiLoR-mini", no_deps=True)
        install_source("chumpy", "git+https://github.com/mattloper/chumpy", no_deps=True)
        subprocess.run([sys.executable, str(code / "tools/patch_chumpy.py")], check=True, timeout=60)
        install_source("geocalib", "git+https://github.com/cvg/GeoCalib")
        subprocess.run([sys.executable, "-c", "import chumpy; from transformers import CLIPModel; "
                        "from geocalib import GeoCalib; "
                        "from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline"],
                       check=True, timeout=120)
        stamp.write_text(identity)
    else:
        print("Dependencies already ready in this runtime.", flush=True)
    # These contain pretrained weights only; user videos are never put in Drive by this helper.
    os.environ.update(HF_HOME=str(cache / "huggingface"), TORCH_HOME=str(cache / "torch"))
    return runpy.run_path(str(code / "tools/colab_session.py"))["run_session"](connection_file, idle_seconds=idle_seconds)
