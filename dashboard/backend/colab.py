"""Server-only Colab discovery and authenticated streaming client."""
import json
import os
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from .clips import DASHBOARD, DATA

load_dotenv(DASHBOARD / ".env")
_lock = threading.RLock()
_cached = {}
SETTINGS = DATA / "settings.json"


def settings():
    defaults = {"url": os.getenv("PS4_API_URL", ""), "profile": "fast", "auto_render": True,
                "retention_h": 24}
    if SETTINGS.exists():
        defaults.update(json.loads(SETTINGS.read_text()))
    return defaults


def save_settings(values):
    with _lock:
        current = settings()
        current.update(values)
        validate_url(current.get("url", ""))
        if current["profile"] not in ("fast", "full"):
            raise ValueError("profile must be fast or full")
        current["retention_h"] = max(1, min(720, int(current["retention_h"])))
        SETTINGS.parent.mkdir(parents=True, exist_ok=True)
        tmp = SETTINGS.with_suffix(".tmp")
        tmp.write_text(json.dumps(current))
        tmp.replace(SETTINGS)
        _cached.clear()
        return current


def validate_url(url):
    if not url:
        return
    p = urlparse(url)
    if p.username or p.password or p.query or p.fragment or p.path not in ("", "/"):
        raise ValueError("Use the server origin, without a key, path, or credentials")
    if not p.hostname or (p.scheme != "https" and not (p.scheme == "http" and p.hostname in ("127.0.0.1", "localhost"))):
        raise ValueError("Use an HTTPS URL (HTTP is allowed for localhost tests)")


def discover(force=False):
    with _lock:
        cache_seconds = 60 if _cached.get("connected") else 5
        if not force and _cached and time.time() - _cached.get("checked_at", 0) < cache_seconds:
            return dict(_cached)
        candidates = []
        manual = settings().get("url", "").strip().rstrip("/")
        if manual:
            candidates.append((time.time(), manual, "manual"))
        else:
            drive = os.getenv("PS4_DRIVE_URL_FILE")
            if drive:
                try:
                    data = json.loads(Path(drive).read_text())
                    candidates.append((data["started"], data["url"], "drive"))
                except (OSError, ValueError, KeyError):
                    pass
            topic = os.getenv("PS4_DISCOVERY_TOPIC")
            if topic:
                try:
                    r = httpx.get(f"https://ntfy.sh/{topic}/json", params={"poll": "1", "since": "12h"}, timeout=5)
                    r.raise_for_status()
                    for line in r.text.splitlines():
                        message = json.loads(line)
                        if message.get("event") == "message":
                            data = json.loads(message["message"])
                            candidates.append((data["started"], data["url"], "ntfy"))
                except (httpx.HTTPError, ValueError, KeyError):
                    pass
        found = {"connected": False, "url_host": None, "gpu": None, "warm": False,
                 "queued": 0, "source": "manual" if manual else None, "checked_at": time.time(),
                 "key_present": bool(os.getenv("PS4_API_KEY")), "settings": settings()}
        for _, url, source in sorted(candidates, reverse=True):
            try:
                validate_url(url)
                r = httpx.get(url.rstrip("/") + "/health", timeout=5)
                r.raise_for_status()
                health = r.json()
                if health.get("ok"):
                    if health.get("auth"):
                        authenticated = httpx.get(url.rstrip("/") + "/jobs",
                                                  headers={"X-API-Key": os.getenv("PS4_API_KEY", "")}, timeout=5)
                        authenticated.raise_for_status()
                    found.update(connected=True, url=url.rstrip("/"), url_host=urlparse(url).netloc,
                                 gpu=health.get("gpu"), warm=health.get("warm", False),
                                 queued=health.get("queued", 0), source=source)
                    break
            except (httpx.HTTPError, ValueError):
                pass
        _cached.clear()
        _cached.update(found)
        return dict(found)


def status(force=False):
    return {k: v for k, v in discover(force).items() if k != "url"}


def request(method, path, **kw):
    # Request-level timeout stays below the quick tunnel's 100-second ceiling.
    connection = discover()
    if not connection["connected"]:
        raise ConnectionError("Colab is offline; waiting for a connection")
    headers = {"X-API-Key": os.getenv("PS4_API_KEY", "")}
    r = httpx.request(method, connection["url"] + path, headers=headers, timeout=90, **kw)
    r.raise_for_status()
    return r


class ProgressFile:
    def __init__(self, fh, callback):
        self.fh, self.callback = fh, callback
        self.total = os.fstat(fh.fileno()).st_size

    def read(self, size=-1):
        data = self.fh.read(size)
        self.callback(self.fh.tell() / max(1, self.total))
        return data

    def seek(self, *args):
        return self.fh.seek(*args)

    def tell(self):
        return self.fh.tell()

    def fileno(self):
        return self.fh.fileno()


def download(jid, name, dst):
    connection = discover()
    if not connection.get("connected"):
        raise ConnectionError("Colab offline")
    tmp = dst.with_suffix(dst.suffix + ".part")
    with httpx.stream("GET", connection["url"] + f"/jobs/{jid}/files/{name}",
                      headers={"X-API-Key": os.getenv("PS4_API_KEY", "")}, timeout=90) as r:
        r.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in r.iter_bytes():
                fh.write(chunk)
    tmp.replace(dst)
