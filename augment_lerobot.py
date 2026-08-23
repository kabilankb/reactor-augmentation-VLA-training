#!/usr/bin/env python3
"""Augment a LeRobot episode through X2 and write a LeRobot v2.1 dataset.

Runs the edit in ~96-frame batches with a per-batch offset fit, because a
single long stream drifts: a 774-frame run measured coverage 0.63 with temporal
drift doubling 1.3px -> 2.7px, while 96-frame batches held 0.86 and flat.

Frames X2 does not return (~10%) are dropped from the episode along with their
actions, rather than reconstructed. Losing a tenth of a demonstration is
harmless for behaviour cloning; misaligning it is not.

    python augment_lerobot.py --episode episodes/orange_ep0 \\
        --prompt "a red apple and a yellow lemon ..." \\
        --task "Grab orange and place into plate" \\
        --out datasets/lerobot_fruit

Output layout matches LightwheelAI/leisaac-pick-orange (LeRobot v2.1).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from reactor_sdk import ReactorError

from prompt_sweep import run_prompt
from validate_x2 import load_frames, pair_frames, residual_drift, structure_coverage


def run_prompt_retrying(chunk, prompt, args, attempts: int = 3):
    """One batch, retrying transient API failures.

    Sessions occasionally fail to become ready ("not ready after 20 polls").
    Without a retry a single transient timeout aborts a build that may be an
    hour of streaming, so each batch gets its own attempts and only a
    persistent failure gives up.
    """
    for attempt in range(1, attempts + 1):
        try:
            return asyncio.run(run_prompt(chunk, prompt, args.fps, args.quiet, args.max_wait))
        except (ReactorError, asyncio.TimeoutError, OSError) as exc:
            if attempt == attempts:
                print(f"   giving up after {attempts}: {type(exc).__name__}")
                return []
            print(f"   {type(exc).__name__} — retry {attempt}/{attempts - 1}")
            time.sleep(5 * attempt)
    return []


def load_manifest(episode: Path) -> list[dict]:
    rows = [json.loads(l) for l in (episode / "manifest.jsonl").read_text().splitlines() if l.strip()]
    return sorted(rows, key=lambda r: r["seq"])


def run_batches(frames, rows, prompt, args):
    """Edit the episode in batches, returning kept (image, row) pairs."""
    kept_imgs, kept_rows, stats = [], [], []

    for start in range(0, len(frames), args.batch):
        chunk = frames[start:start + args.batch]
        chunk_rows = rows[start:start + args.batch]
        n = len(chunk)
        print(f"  batch {start:>4}-{start + n - 1:<4} ({n} frames)", end="", flush=True)

        received = run_prompt_retrying(chunk, prompt, args)
        if not received:
            print("   no frames returned — batch dropped")
            stats.append({"start": start, "kept": 0, "coverage": None})
            continue

        pairs, how = pair_frames(chunk, received)
        covs, drifts = [], []
        batch_imgs, batch_rows = [], []
        for seq, rec in pairs:
            src = chunk[seq]
            h, w = src.shape[:2]
            edit = cv2.resize(rec.frame, (w, h), interpolation=cv2.INTER_AREA)
            covs.append(structure_coverage(src, edit))
            r, nin, _, _ = residual_drift(src, edit, None)
            if r is not None and nin >= 12:
                drifts.append(r)
            batch_imgs.append(edit)
            batch_rows.append(chunk_rows[seq])

        cov = float(np.median(covs))
        drift = float(np.median(drifts)) if drifts else float("nan")
        ok = cov >= args.coverage_floor
        print(f"   coverage {cov:.3f}  drift {drift:.2f}px  {len(pairs)}/{n} kept"
              f"   {'OK' if ok else 'REJECTED'}")
        stats.append({"start": start, "kept": len(pairs) if ok else 0,
                      "coverage": cov, "drift_px": drift, "pairing": how})
        if ok:
            kept_imgs.extend(batch_imgs)
            kept_rows.extend(batch_rows)

    return kept_imgs, kept_rows, stats


def write_lerobot(out: Path, imgs, rows, task: str, src_info: dict, video_key: str, fps: int) -> None:
    """Write a single-episode LeRobot v2.1 dataset."""
    ep = 0
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out / "videos" / "chunk-000" / video_key).mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)

    # Video. LeRobot stores frames as mp4, not as loose images; encode with
    # ffmpeg since OpenCV's writer has no AV1 and yuv420p keeps it portable.
    tmp = out / "_frames"
    tmp.mkdir(exist_ok=True)
    for i, im in enumerate(imgs):
        cv2.imwrite(str(tmp / f"f{i:06d}.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    video_path = out / "videos" / "chunk-000" / video_key / f"episode_{ep:06d}.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps),
         "-i", str(tmp / "f%06d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video_path)],
        check=True,
    )
    shutil.rmtree(tmp)

    # Parquet. frame_index and index are renumbered contiguously: dropped
    # frames must not leave holes, or downstream loaders mis-slice the episode.
    n = len(rows)
    df = pd.DataFrame({
        "action": [np.asarray(r["action"], np.float32) for r in rows],
        "observation.state": [
            np.asarray(r["state"] if r.get("state") else r["action"], np.float32) for r in rows
        ],
        "timestamp": np.arange(n, dtype=np.float32) / fps,
        "frame_index": np.arange(n, dtype=np.int64),
        "episode_index": np.full(n, ep, dtype=np.int64),
        "index": np.arange(n, dtype=np.int64),
        "task_index": np.zeros(n, dtype=np.int64),
        # Provenance: which source frame each augmented frame came from.
        "source_seq": np.asarray([r["seq"] for r in rows], np.int64),
    })
    df.to_parquet(out / "data" / "chunk-000" / f"episode_{ep:06d}.parquet", index=False)

    h, w = imgs[0].shape[:2]
    dim = len(rows[0]["action"])
    info = {
        "codebase_version": "v2.1",
        "robot_type": src_info.get("robot_type", "so101_follower"),
        "total_episodes": 1,
        "total_frames": n,
        "total_tasks": 1,
        "total_videos": 1,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": "0:1"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {"dtype": "float32", "shape": [dim], "names": None},
            "observation.state": {"dtype": "float32", "shape": [dim], "names": None},
            f"observation.images.{video_key.split('.')[-1]}": {
                "dtype": "video", "shape": [h, w, 3], "names": ["height", "width", "channel"],
                "info": {"video.fps": float(fps), "video.codec": "h264",
                         "video.pix_fmt": "yuv420p", "video.is_depth_map": False},
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            # Provenance column written into the parquet. LeRobot builds its Arrow
            # schema from this dict and hard-fails the cast on any undeclared
            # column, so writing source_seq without declaring it makes the whole
            # dataset unloadable.
            "source_seq": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    (out / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": task}) + "\n")
    (out / "meta" / "episodes.jsonl").write_text(
        json.dumps({"episode_index": ep, "tasks": [task], "length": n}) + "\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=Path, required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--task", required=True, help="language instruction — must match what the pixels show")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=0, help="0 = whole episode")
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--fps", type=float, default=24.0, help="push rate")
    ap.add_argument("--out-fps", type=int, default=30, help="dataset fps (match the source)")
    ap.add_argument("--video-key", default="observation.images.front")
    ap.add_argument("--quiet", type=float, default=12.0)
    ap.add_argument("--max-wait", type=float, default=120.0)
    ap.add_argument("--coverage-floor", type=float, default=0.75)
    ap.add_argument("--src-info", type=Path, default=Path("datasets/orange/info.json"))
    args = ap.parse_args()

    if not os.environ.get("REACTOR_API_KEY"):
        sys.exit("set REACTOR_API_KEY")

    rows = load_manifest(args.episode)
    frames = load_frames(args.episode / "rgb", len(rows))
    hi = len(frames) if args.count == 0 else min(args.start + args.count, len(frames))
    frames, rows = frames[args.start:hi], rows[args.start:hi]
    print(f"  source: {len(frames)} frames, batches of {args.batch}")
    print(f"  prompt: {args.prompt[:70]}...")
    print(f"  task:   {args.task!r}\n")

    imgs, kept, stats = run_batches(frames, rows, args.prompt, args)
    if not imgs:
        sys.exit("no batch passed the coverage floor — nothing written")

    src_info = json.loads(args.src_info.read_text()) if args.src_info.exists() else {}
    if args.out.exists():
        shutil.rmtree(args.out)
    write_lerobot(args.out, imgs, kept, args.task, src_info, args.video_key, args.out_fps)

    covs = [s["coverage"] for s in stats if s["coverage"] is not None]
    print("\n" + "=" * 62)
    print(f"  wrote {args.out}")
    print(f"  {len(imgs)}/{len(frames)} frames kept ({len(imgs) / len(frames):.0%})")
    print(f"  coverage median {np.median(covs):.3f} across {len(stats)} batches")
    print(f"  task: {args.task!r}")
    (args.out / "meta" / "augmentation.json").write_text(json.dumps(
        {"source": str(args.episode), "prompt": args.prompt, "batches": stats}, indent=2))
    print(f"  provenance: {args.out}/meta/augmentation.json")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("interrupted")
