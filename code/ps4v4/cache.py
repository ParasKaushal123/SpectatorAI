"""Per-clip cache: every module writes one output plus a stamp with its config hash.

A re-run skips a module whose output exists with a matching hash, so a Colab
disconnect costs at most the module that was running.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .config import module_hash


class ClipCache:
    def __init__(self, root: str | Path, clip_name: str, cfg: dict):
        self.dir = Path(root) / clip_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg

    def path(self, name: str) -> Path:
        return self.dir / name

    # ---- stamps -------------------------------------------------------------
    def _stamp(self, module: str) -> Path:
        return self.dir / f".{module}.stamp"

    def is_fresh(self, module: str) -> bool:
        st = self._stamp(module)
        return st.exists() and st.read_text().strip() == module_hash(self.cfg, module)

    def mark_done(self, module: str) -> None:
        self._stamp(module).write_text(module_hash(self.cfg, module))

    def invalidate_from(self, module: str, order: list[str]) -> None:
        for m in order[order.index(module):]:
            self._stamp(m).unlink(missing_ok=True)

    # ---- io helpers ---------------------------------------------------------
    def save_json(self, name: str, obj: Any) -> None:
        with open(self.path(name), "w") as fh:
            json.dump(obj, fh, indent=2, default=_json_default)

    def load_json(self, name: str) -> Any:
        with open(self.path(name)) as fh:
            return json.load(fh)

    def save_npz(self, name: str, **arrays) -> None:
        np.savez_compressed(self.path(name), **arrays)

    def load_npz(self, name: str) -> dict:
        with np.load(self.path(name), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def save_pickle(self, name: str, obj: Any) -> None:
        with open(self.path(name), "wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def load_pickle(self, name: str) -> Any:
        with open(self.path(name), "rb") as fh:
            return pickle.load(fh)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))
