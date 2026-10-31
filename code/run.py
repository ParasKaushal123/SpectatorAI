"""PS4 v4 command line.

Examples:
    python run.py --clip "/content/drive/MyDrive/mytronD/Copy of mytronlabs-000060-014.mp4"
    python run.py --clip "/data/clips/*.mp4" --set max_seconds=30
    python run.py --clip demo.mp4 --from M3A --set assign.accept_bias=0.3
    python run.py --evaluate labels.csv --systems v1=a.jsonl v3=v3.jsonl v4=cache/demo/hands.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import logging

import yaml

from ps4v4.config import MODULE_ORDER, load_config


def parse_sets(items: list[str]) -> dict:
    out: dict = {}
    for it in items or []:
        key, val = it.split("=", 1)
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--clip", nargs="*", default=[], help="video path(s) or glob(s)")
    ap.add_argument("--from", dest="start_from", choices=MODULE_ORDER, help="force re-run from this module")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. assign.accept_bias=0.3")
    ap.add_argument("--evaluate", help="labels CSV")
    ap.add_argument("--systems", nargs="*", default=[], help="name=hands.jsonl pairs to compare")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(a.config, parse_sets(a.set))

    if a.evaluate:
        from ps4v4.evaluate import compare
        systems = dict(s.split("=", 1) for s in a.systems)
        print(compare(a.evaluate, systems).to_string(index=False))
        return

    from ps4v4.pipeline import run_clip
    paths = sorted({p for pat in a.clip for p in (glob.glob(pat) or [pat])})
    for p in paths:
        stats = run_clip(p, cfg, start_from=a.start_from)
        print(json.dumps({k: stats[k] for k in ("left", "right", "seconds")}, indent=1))


if __name__ == "__main__":
    main()
