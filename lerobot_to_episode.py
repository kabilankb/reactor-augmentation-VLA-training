#!/usr/bin/env python3
"""Convert a GR00T-LeRobot episode into the frames + manifest layout the
augmentation pipeline expects.

The Arena GR1 dataset already ships rendered ego-view video alongside its
action parquet, so an episode can enter the pipeline without launching Isaac
Sim at all. Output is byte-for-byte the same shape `capture_episodes.py`
produces, so `validate_x2.py` consumes either interchangeably.

    python lerobot_to_episode.py \\
        --video datasets/lerobot/ep0.mp4 \\
        --parquet datasets/lerobot/ep0.parquet \\
        --out episodes/gr1_ep0

Dataset: https://huggingface.co/datasets/nvidia/Arena-GR1-Manipulation-Task
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def decode(video: Path) -> list[np.ndarray]:
    """Decode a video to BGR frames, falling back to ffmpeg.

    OpenCV's build has no working AV1 decoder, which several LeRobot datasets
    use. It fails by returning zero frames rather than raising, so the fallback
    triggers on an empty result, not on an exception.
    """
    cap = cv2.VideoCapture(str(video))
    frames = []
    if cap.isOpened():
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(bgr)
        cap.release()
    if frames:
        return frames

    if not shutil.which("ffmpeg"):
        sys.exit(f"cannot decode {video} and ffmpeg is not installed")
    print("  OpenCV decoded nothing (likely AV1) — falling back to ffmpeg")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
             f"{tmp}/f%06d.png"],
            check=True,
        )
        return [cv2.imread(str(p)) for p in sorted(Path(tmp).glob("*.png"))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True, help="episode mp4 (ego view)")
    ap.add_argument("--parquet", type=Path, required=True, help="matching episode parquet")
    ap.add_argument("--out", type=Path, required=True, help="output episode directory")
    ap.add_argument("--task", type=str, default=None, help="language instruction to record")
    ap.add_argument(
        "--action-col", default="action", help="parquet column holding the action vector"
    )
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    if args.action_col not in df.columns:
        sys.exit(f"no {args.action_col!r} column; have {list(df.columns)}")

    rgb_dir = args.out / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)

    frames = decode(args.video)

    if len(frames) != len(df):
        # A mismatch means the video and the action table describe different
        # timelines; pairing them anyway would silently misalign every label.
        sys.exit(f"frame/row mismatch: {len(frames)} frames vs {len(df)} rows — refusing to pair")

    episode_index = int(df["episode_index"].iloc[0]) if "episode_index" in df else 0

    manifest = []
    for i, bgr in enumerate(frames):
        name = f"f{i:06d}.png"
        cv2.imwrite(str(rgb_dir / name), bgr)
        row = df.iloc[i]
        manifest.append({
            "seq": i,
            "frame": f"rgb/{name}",
            "mask": None,
            "episode": episode_index,
            "step": int(row["frame_index"]) if "frame_index" in df else i,
            "action": [float(v) for v in np.asarray(row[args.action_col]).ravel()],
            "state": (
                [float(v) for v in np.asarray(row["observation.state"]).ravel()]
                if "observation.state" in df
                else None
            ),
            "task": args.task,
        })

    manifest_path = args.out / "manifest.jsonl"
    with manifest_path.open("w") as fh:
        for r in manifest:
            fh.write(json.dumps(r) + "\n")

    h, w = frames[0].shape[:2]
    dim = len(manifest[0]["action"])
    print(f"  episode {episode_index}: {len(frames)} frames at {w}x{h}, action dim {dim}")
    print(f"  frames:   {rgb_dir}")
    print(f"  manifest: {manifest_path}")
    print(f"\n  next: python validate_x2.py --frames-dir {rgb_dir} --count {len(frames)}")


if __name__ == "__main__":
    main()
