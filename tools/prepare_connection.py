"""Prepare this computer and a private one-upload Colab bundle. Never prints credentials."""
import json
import secrets
import shutil
import sys
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from build_connect_notebook import SETUP_VERSION

ROOT = Path(__file__).resolve().parents[1]


def prepare():
    from dotenv import dotenv_values
    env_file = ROOT / "dashboard/.env"
    values = dict(dotenv_values(env_file)) if env_file.exists() else {}
    values["PS4_API_KEY"] = values.get("PS4_API_KEY") or secrets.token_urlsafe(48)
    values["PS4_DISCOVERY_TOPIC"] = values.get("PS4_DISCOVERY_TOPIC") or "spectator-" + secrets.token_urlsafe(32)
    values.setdefault("PS4_API_URL", "")
    values.setdefault("PS4_DRIVE_URL_FILE", "")
    env_file.write_text("\n".join(f"{k}={v or ''}" for k, v in values.items()) + "\n", encoding="utf-8")
    destination = ROOT / "dashboard/data/colab-setup"
    destination.mkdir(parents=True, exist_ok=True)
    bundle = destination / "SpectatorAI-Colab.zip"
    allowed = {".py", ".yaml", ".yml", ".txt"}
    with ZipFile(bundle, "w", ZIP_DEFLATED) as z:
        z.writestr("setup-version.txt", SETUP_VERSION)
        for p in sorted((ROOT / "code").rglob("*")):
            if not p.is_file() or p.suffix not in allowed:
                continue
            if any(part in {"__pycache__", ".venv", "cache", "results", "data"} for part in p.relative_to(ROOT / "code").parts):
                continue
            z.write(p, p.relative_to(ROOT).as_posix())
        z.writestr("connection.json", json.dumps({
            "schema": 1, "api_key": values["PS4_API_KEY"], "topic": values["PS4_DISCOVERY_TOPIC"],
        }))
    export = ROOT / "SpectatorAI-Colab.zip"
    shutil.copyfile(bundle, export)
    print("Local connection settings configured.")
    print("Private Colab upload bundle:", export)
    print("Credentials are in ignored local files only; notebook code contains no keys.")
    return bundle


if __name__ == "__main__":
    prepare()
