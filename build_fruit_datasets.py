#!/usr/bin/env python3
"""Build a multi-episode LeRobot dataset of fruit-augmented scenes.

Each scene in the spec becomes one episode: the same source trajectory with a
different set of distractor fruit added. The oranges stay the target and the
instruction is unchanged, because nothing about the task changed — only the
visual clutter the policy must learn to ignore.

Distractor fruit is not size-constrained the way a *substitute* would be. A
banana cannot replace an orange (the recorded grasp does not transfer), but it
is a perfectly good distractor, because the robot never touches it.

Every episode carries its own provenance: the prompt, the fruits named, the
measured coverage and drift, and the source frame each augmented frame came
from. An episode whose coverage falls below the floor is dropped rather than
written, so the dataset never contains frames whose labels may not hold.

    python build_fruit_datasets.py --episode episodes/orange_ep0 \\
        --scenes fruit_scenes.json --count 96 --out datasets/lerobot_fruit10

Output: LeRobot v2.1, mirroring LightwheelAI/leisaac-pick-orange.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from augment_lerobot import load_manifest, run_batches
from validate_x2 import load_frames


def encode_video(imgs, path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"_tmp_{path.stem}"
    tmp.mkdir(exist_ok=True)
    for i, im in enumerate(imgs):
        cv2.imwrite(str(tmp / f"f{i:06d}.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps),
         "-i", str(tmp / "f%06d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    shutil.rmtree(tmp)


def feature_stats(arr: np.ndarray) -> dict:
    """Per-dimension stats in the shape LeRobot's episodes_stats expects."""
    a = np.asarray(arr, np.float64)
    return {
        "min": a.min(0).tolist(), "max": a.max(0).tolist(),
        "mean": a.mean(0).tolist(), "std": a.std(0).tolist(),
        "count": [int(a.shape[0])],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=Path, required=True)
    ap.add_argument("--scenes", type=Path, required=True, help="JSON list of {id, fruits, prompt}")
    ap.add_argument("--task", default="Grab orange and place into plate")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=96, help="source frames per episode")
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--fps", type=float, default=24.0, help="push rate")
    ap.add_argument("--out-fps", type=int, default=30, help="dataset fps")
    ap.add_argument("--video-key", default="observation.images.front")
    ap.add_argument("--quiet", type=float, default=12.0)
    ap.add_argument("--max-wait", type=float, default=120.0)
    ap.add_argument("--coverage-floor", type=float, default=0.75)
    ap.add_argument("--preview", type=Path, default=Path("out_fruit10"))
    ap.add_argument("--src-info", type=Path, default=Path("datasets/orange/info.json"))
    args = ap.parse_args()

    if not os.environ.get("REACTOR_API_KEY"):
        sys.exit("set REACTOR_API_KEY")

    scenes = json.loads(args.scenes.read_text())
    rows_all = load_manifest(args.episode)
    frames_all = load_frames(args.episode / "rgb", len(rows_all))
    hi = min(args.start + args.count, len(frames_all))
    frames, rows = frames_all[args.start:hi], rows_all[args.start:hi]
    print(f"  source {args.episode} frames [{args.start}:{hi}] ({len(frames)})")
    print(f"  {len(scenes)} scenes, coverage floor {args.coverage_floor}\n")

    args.preview.mkdir(parents=True, exist_ok=True)
    episodes = []

    for scene in scenes:
        print(f"=== {scene['id']}  ({', '.join(scene['fruits'])})")
        imgs, kept, stats = run_batches(frames, rows, scene["prompt"], args)
        covs = [s["coverage"] for s in stats if s["coverage"] is not None]
        if not imgs:
            print("  dropped — no batch passed\n")
            continue
        cov = float(np.median(covs))
        # .get() on both sides: a batch that returned no frames has no drift_px
        # key at all, and s["drift_px"] would raise KeyError here after the
        # whole (paid) streaming run has already completed.
        drifts = [d for s in stats if (d := s.get("drift_px")) == d and d is not None]
        episodes.append({
            "scene": scene, "images": imgs, "rows": kept,
            "coverage": cov, "drift_px": float(np.median(drifts)) if drifts else None,
        })
        mid = len(imgs) // 2
        cv2.imwrite(str(args.preview / f"{scene['id']}.png"),
                    cv2.cvtColor(np.hstack([frames[kept[mid]["seq"] - args.start], imgs[mid]]),
                                 cv2.COLOR_RGB2BGR))
        print()

    if not episodes:
        sys.exit("every scene failed — nothing written")

    if args.out.exists():
        shutil.rmtree(args.out)
    (args.out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (args.out / "meta").mkdir(parents=True, exist_ok=True)

    src_info = json.loads(args.src_info.read_text()) if args.src_info.exists() else {}
    ep_lines, stat_lines, aug_lines = [], [], []
    global_index = 0

    for ei, ep in enumerate(episodes):
        n = len(ep["rows"])
        acts = np.stack([np.asarray(r["action"], np.float32) for r in ep["rows"]])
        states = np.stack([
            np.asarray(r["state"] if r.get("state") else r["action"], np.float32) for r in ep["rows"]
        ])
        df = pd.DataFrame({
            "action": list(acts),
            "observation.state": list(states),
            "timestamp": np.arange(n, dtype=np.float32) / args.out_fps,
            "frame_index": np.arange(n, dtype=np.int64),
            "episode_index": np.full(n, ei, dtype=np.int64),
            "index": np.arange(global_index, global_index + n, dtype=np.int64),
            "task_index": np.zeros(n, dtype=np.int64),
            "source_seq": np.asarray([r["seq"] for r in ep["rows"]], np.int64),
        })
        global_index += n
        df.to_parquet(args.out / "data" / "chunk-000" / f"episode_{ei:06d}.parquet", index=False)
        encode_video(ep["images"],
                     args.out / "videos" / "chunk-000" / args.video_key / f"episode_{ei:06d}.mp4",
                     args.out_fps)

        ep_lines.append({"episode_index": ei, "tasks": [args.task], "length": n})
        stat_lines.append({"episode_index": ei, "stats": {
            "action": feature_stats(acts), "observation.state": feature_stats(states),
        }})
        # Augmentation provenance — not part of the LeRobot spec, carried
        # alongside so a consumer can trace any frame back to its source.
        aug_lines.append({
            "episode_index": ei,
            "scene_id": ep["scene"]["id"],
            "fruits_added": ep["scene"]["fruits"],
            "prompt": ep["scene"]["prompt"],
            "coverage": round(ep["coverage"], 4),
            "drift_px": round(ep["drift_px"], 3) if ep["drift_px"] else None,
            "source_dataset": "LightwheelAI/leisaac-pick-orange",
            "source_episode": str(args.episode),
            "source_frames": [args.start, hi],
            "frames_kept": n,
            "augmenter": "reactor xmax/x2",
            "target_object": "orange",
            "note": "distractor fruit added; oranges unchanged so the task is unchanged",
        })

    h, w = episodes[0]["images"][0].shape[:2]
    dim = int(acts.shape[1])
    total = sum(len(e["rows"]) for e in episodes)
    info = {
        "codebase_version": "v2.1",
        "robot_type": src_info.get("robot_type", "so101_follower"),
        "total_episodes": len(episodes),
        "total_frames": total,
        "total_tasks": 1,
        "total_videos": len(episodes),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": args.out_fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {"dtype": "float32", "shape": [dim], "names": None},
            "observation.state": {"dtype": "float32", "shape": [dim], "names": None},
            args.video_key: {
                "dtype": "video", "shape": [h, w, 3],
                "names": ["height", "width", "channel"],
                "info": {"video.fps": float(args.out_fps), "video.codec": "h264",
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
    m = args.out / "meta"
    (m / "info.json").write_text(json.dumps(info, indent=4))
    (m / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": args.task}) + "\n")
    (m / "episodes.jsonl").write_text("\n".join(json.dumps(x) for x in ep_lines) + "\n")
    (m / "episodes_stats.jsonl").write_text("\n".join(json.dumps(x) for x in stat_lines) + "\n")
    (m / "augmentations.jsonl").write_text("\n".join(json.dumps(x) for x in aug_lines) + "\n")

    print("=" * 70)
    print(f"  wrote {args.out}")
    print(f"  {len(episodes)}/{len(scenes)} scenes kept, {total} frames total\n")
    print(f"  {'ep':<4} {'scene':<16} {'frames':>7} {'coverage':>9}  fruits")
    for i, e in enumerate(episodes):
        print(f"  {i:<4} {e['scene']['id']:<16} {len(e['rows']):>7} {e['coverage']:>9.3f}  "
              f"{', '.join(e['scene']['fruits'])}")
    print(f"\n  previews: {args.preview}/")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("interrupted")
