"""Test the Colab API from your computer: upload a clip, wait, download the results.

    python tools/api_client.py --url https://xxxx.trycloudflare.com --key <token> --video clip.mp4 --out results/
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--key", default=None)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="api_results")
    a = ap.parse_args()
    h = {"X-API-Key": a.key} if a.key else {}
    base = a.url.rstrip("/")
    print("health:", requests.get(f"{base}/health", timeout=30).json())
    with open(a.video, "rb") as fh:
        job = requests.post(f"{base}/jobs", headers=h, files={"file": (Path(a.video).name, fh, "video/mp4")},
                            timeout=600).json()
    print("job:", job.get("id"), job.get("video"))
    while True:
        job = requests.get(f"{base}/jobs/{job['id']}", headers=h, timeout=30).json()
        print(f"  {job['status']:8s} {job.get('stage', ''):28s} {100 * job.get('progress', 0):5.1f}%")
        if job["status"] in ("done", "failed"):
            break
        time.sleep(5)
    if job["status"] == "failed":
        print(job.get("error")); print(job.get("traceback", "")); return
    res = requests.get(f"{base}/jobs/{job['id']}/result", headers=h, timeout=60).json()
    out = Path(a.out) / job["id"]
    out.mkdir(parents=True, exist_ok=True)
    for f in res["files"]:
        r = requests.get(f"{base}/jobs/{job['id']}/files/{f}", headers=h, timeout=600)
        (out / f).write_bytes(r.content)
    q = res["quality"]
    print(json.dumps({"score": q["score"], "grade": q["grade"], "usable": q["usable"]["share"],
                      "issues": q["issues"]}, indent=1))
    print("saved to", out)


if __name__ == "__main__":
    main()
