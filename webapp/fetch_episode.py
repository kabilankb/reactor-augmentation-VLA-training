#!/usr/bin/env python3
"""Fetch one episode of a HuggingFace LeRobot dataset and convert it.

Downloads only the episode's video + parquet (plus meta), not the whole repo —
a full LeRobot dataset can be gigabytes and the pipeline works one episode at a
time. Hands off to lerobot_to_episode.py so there is one conversion path.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def get(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {url.split('/main/')[-1]}")
    req = urllib.request.Request(url, headers={"User-Agent": "reactor-augmentation/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r, dest.open("wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    print(f"    -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--video-key", default="observation.images.front")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()

    base = f"https://huggingface.co/datasets/{a.repo}/resolve/main"
    stage = ROOT / "datasets" / "_hf" / a.name
    ep = f"episode_{a.episode:06d}"

    # Some repos (the GR1 one) nest the LeRobot tree under lerobot/.
    prefix = ""
    try:
        urllib.request.urlopen(urllib.request.Request(
            f"{base}/meta/info.json", headers={"User-Agent": "reactor-augmentation/1.0"}), timeout=20)
    except Exception:
        prefix = "lerobot/"
        print("  meta/info.json not at root — trying lerobot/ prefix")

    info_p = stage / "info.json"
    get(f"{base}/{prefix}meta/info.json", info_p)
    info = json.loads(info_p.read_text())
    print(f"  {info.get('robot_type')} | {info.get('fps')}fps | "
          f"{info.get('total_episodes')} episodes")

    try:
        get(f"{base}/{prefix}meta/tasks.jsonl", stage / "tasks.jsonl")
        task = json.loads((stage / "tasks.jsonl").read_text().splitlines()[0]).get("task")
    except Exception:
        task = None
    print(f"  task: {task!r}")

    video = stage / f"{ep}.mp4"
    parquet = stage / f"{ep}.parquet"
    get(f"{base}/{prefix}videos/chunk-000/{a.video_key}/{ep}.mp4", video)
    get(f"{base}/{prefix}data/chunk-000/{ep}.parquet", parquet)

    out = ROOT / "episodes" / a.name
    cmd = [sys.executable, str(ROOT / "lerobot_to_episode.py"),
           "--video", str(video), "--parquet", str(parquet), "--out", str(out)]
    if task:
        cmd += ["--task", task]
    print(f"\n  converting -> {out}")
    rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
    if rc != 0:
        sys.exit(rc)
    print("\n  done")


if __name__ == "__main__":
    main()
