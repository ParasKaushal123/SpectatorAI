"""Configuration loading and per-module hashing."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

# which config sections each module's output depends on (a change re-runs it and everything after it)
MODULE_DEPS = {
    "M1": ["camera", "max_seconds"],        # intrinsics
    "M2": ["proposals"],                    # hand proposals
    "M1H": [],                              # head rotation (uses camera.head_rotation + proposals as masks)
    "M3": ["tracklets"],                    # side-free tracklets
    "M3P": ["probe", "recon"],              # probe 3D on sampled tracklet frames
    "M3A": ["assign"],                      # hand / side decision per tracklet
    "M4": [],                               # full 3D on accepted hands (recon already hashed at M3P)
    "M5": ["sequence"],
    "M6": ["render"],
}
MODULE_ORDER = ["M1", "M2", "M1H", "M3", "M3P", "M3A", "M4", "M5", "M6"]


class Config(dict):
    """A dict with attribute access for nested sections (cfg.tracker.min_score)."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Config(value) if isinstance(value, dict) and not isinstance(value, Config) else value


def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    if overrides:
        cfg = deep_update(cfg, overrides)
    return Config(cfg)


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def module_hash(cfg: dict, module: str) -> str:
    """Hash of this module's config sections plus every upstream module's sections."""
    idx = MODULE_ORDER.index(module)
    keys = [k for m in MODULE_ORDER[: idx + 1] for k in MODULE_DEPS[m]]
    payload = {k: cfg.get(k) for k in keys}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]
