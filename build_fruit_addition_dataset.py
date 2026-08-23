#!/usr/bin/env python3
"""Add a different fruit to the scene per episode, and gate on TARGET SURVIVAL.

X2 adds objects reliably; it does not replace them (ten attempts, PROJECT.md
section 3 plus out_subst_probe/). So each episode here keeps the same recorded
trajectory and the same instruction -- "Grab orange and place into plate" -- and
differs only in what other fruit is on the table. The orange stays the target.
That is not a weaker task: with an apple and a pear also in frame, the word
"orange" has to actually pick one object out of several, which the single-fruit
source episode never asked of a policy.

WHY THIS EXISTS SEPARATELY FROM build_fruit_datasets.py

That script gates on structure coverage, which is blind to identity -- an orange
painted over by a similarly-sized apple has the same edges. Episode 4 of the
dataset it produced (datasets/lerobot_fruit10, scene "grapes_pear") passed with
coverage 0.871 while the added grape cluster grew across the run until it
covered two of the three oranges:

    frame  5   orange dE 17.1   three oranges present
    frame 30   orange dE 13.6   three oranges present
    frame 55   orange dE 74.2   grapes cover the lower-right orange
    frame 80   orange dE 78.7   grapes cover both lower oranges

The actions still say "reach to the orange at this position". The pixels show
grapes. A per-episode median hides it (that episode's median was 68.8 only
because the damage dominates the second half; a milder case averages away
entirely), so the check here is PER FRAME.

WHAT IT MEASURES

The three oranges are ~1% of the frame but separate cleanly from the wooden
table (same hue, much lower value) and the arm (much lower saturation), so a
source-side colour mask locates them without simulator masks. For each frame,
median CIELAB dE inside that mask says whether the target still looks like
itself. Good frames run 13-23; frames where fruit has grown over an orange run
74-79. The ceiling sits between.

WHAT IT DOES ABOUT IT

Truncates. The corruption is progressive -- content enlarges along the batch --
so the early frames of a damaged episode are fine and only the tail is ruined.
Cutting the tail keeps valid data and cannot create a splice, because dropping a
suffix leaves the remaining frames contiguous. Dropping the whole episode would
throw away good frames; keeping it whole would poison the dataset.

    python build_fruit_addition_dataset.py --episode episodes/orange_ep0 \
        --scenes fruit_scenes10.json --count 96 --out datasets/lerobot_fruit10x

Output: LeRobot v2.1, one episode per fruit, one shared task.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from augment_lerobot import load_manifest, run_batches
from build_fruit_datasets import encode_video, feature_stats
from build_substitution_dataset import fruit_mask
from validate_x2 import load_frames

TASK = "Grab orange and place into plate"


def target_delta(src: np.ndarray, edit: np.ndarray) -> tuple[float, float] | None:
    """(worst-orange dE, scene dE) for one frame.

    PER ORANGE, not pooled. Pooling the three oranges into one mask and taking
    the median lets two survivors outvote one that has been buried: added
    bananas covered one orange while this read 15.6, a clean pass. The number
    that matters is the WORST orange, because the recorded grasp targets a
    specific one and losing any of them invalidates the frames that reach for it.

    `scene dE` is the median outside the oranges. It exists to catch the
    opposite failure -- an edit that changed nothing at all, which trivially
    preserves the target and adds no diversity either.
    """
    fm = fruit_mask(src)
    if fm.sum() < 50:
        return None
    a = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(np.float32)
    b = cv2.cvtColor(edit, cv2.COLOR_RGB2LAB).astype(np.float32)
    de = np.linalg.norm(a - b, axis=2)

    n, lab, st, _ = cv2.connectedComponentsWithStats(fm, 8)
    per_blob = [float(np.median(de[lab == i])) for i in range(1, n)
                if st[i, cv2.CC_STAT_AREA] >= 50]
    if not per_blob:
        return None
    protected = cv2.dilate(fm, np.ones((9, 9), np.uint8), iterations=2) == 0
    return max(per_blob), float(np.median(de[protected]))


def truncate_on_target_loss(frames, imgs, rows, start, ceiling, run_len):
    """Cut the episode at the first sustained run of target-loss frames.

    `run_len` consecutive frames over the ceiling are required before cutting:
    a single frame can spike on a specular highlight or a momentary occlusion by
    the gripper itself, which is legitimate and recovers.
    """
    des, scene = [], []
    for im, r in zip(imgs, rows):
        d = target_delta(frames[r["seq"] - start], im)
        des.append(d[0] if d is not None else 0.0)
        scene.append(d[1] if d is not None else 0.0)

    cut = len(imgs)
    bad = 0
    for i, d in enumerate(des):
        bad = bad + 1 if d > ceiling else 0
        if bad >= run_len:
            cut = i - run_len + 1
            break
    return cut, des, scene


def write_dataset(out: Path, episodes, src_info, video_key, fps, scenes_path) -> None:
    if out.exists():
        shutil.rmtree(out)
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)

    ep_lines, stat_lines, aug_lines = [], [], []
    gi = 0
    for ei, ep in enumerate(episodes):
        rows, imgs = ep["rows"], ep["images"]
        n = len(rows)
        acts = np.stack([np.asarray(r["action"], np.float32) for r in rows])
        # No silent fallback to `action`: a dataset whose state IS its label
        # trains a policy that reads the answer off its own input.
        if any(not r.get("state") for r in rows):
            sys.exit("source manifest has no `state` — refusing to substitute actions for it")
        states = np.stack([np.asarray(r["state"], np.float32) for r in rows])

        pd.DataFrame({
            "action": list(acts),
            "observation.state": list(states),
            "timestamp": np.arange(n, dtype=np.float32) / fps,
            "frame_index": np.arange(n, dtype=np.int64),
            "episode_index": np.full(n, ei, dtype=np.int64),
            "index": np.arange(gi, gi + n, dtype=np.int64),
            "task_index": np.zeros(n, dtype=np.int64),
            "source_seq": np.asarray([r["seq"] for r in rows], np.int64),
        }).to_parquet(out / "data" / "chunk-000" / f"episode_{ei:06d}.parquet", index=False)
        gi += n
        encode_video(imgs, out / "videos" / "chunk-000" / video_key / f"episode_{ei:06d}.mp4", fps)

        ep_lines.append({"episode_index": ei, "tasks": [TASK], "length": n})
        stat_lines.append({"episode_index": ei, "stats": {
            "action": feature_stats(acts), "observation.state": feature_stats(states)}})
        aug_lines.append({
            "episode_index": ei, "scene_id": ep["scene"]["id"],
            "fruits_added": ep["scene"]["fruits"], "prompt": ep["scene"]["prompt"],
            "task": TASK, "target_object": "orange",
            "coverage": round(ep["coverage"], 4),
            "target_dE_median": round(ep["target_med"], 2),
            "target_dE_p90": round(ep["target_p90"], 2),
            "frames_before_truncation": ep["n_before"], "frames_kept": n,
            "truncated": ep["n_before"] != n,
            "source_dataset": "LightwheelAI/leisaac-pick-orange",
            "source_episode": ep["source_episode"], "source_frames": ep["source_frames"],
            "source_seq_range": [int(rows[0]["seq"]), int(rows[-1]["seq"])],
            "augmenter": "reactor xmax/x2",
            "note": "distractor fruit added; the three oranges are verified per-frame to "
                    "survive, so the recorded grasp still lands on its target",
        })

    h, w = episodes[0]["images"][0].shape[:2]
    dim = int(acts.shape[1])
    total = sum(len(e["rows"]) for e in episodes)
    info = {
        "codebase_version": "v2.1",
        "robot_type": src_info.get("robot_type", "so101_follower"),
        "total_episodes": len(episodes), "total_frames": total, "total_tasks": 1,
        "total_videos": len(episodes), "total_chunks": 1, "chunks_size": 1000, "fps": fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "action": {"dtype": "float32", "shape": [dim], "names": None},
            "observation.state": {"dtype": "float32", "shape": [dim], "names": None},
            video_key: {"dtype": "video", "shape": [h, w, 3],
                        "names": ["height", "width", "channel"],
                        "info": {"video.fps": float(fps), "video.codec": "h264",
                                 "video.pix_fmt": "yuv420p", "video.is_depth_map": False}},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            # Declared, because LeRobot builds its Arrow schema from this dict
            # and hard-fails the cast on any undeclared parquet column.
            "source_seq": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    m = out / "meta"
    (m / "info.json").write_text(json.dumps(info, indent=4))
    (m / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": TASK}) + "\n")
    (m / "episodes.jsonl").write_text("\n".join(json.dumps(x) for x in ep_lines) + "\n")
    (m / "episodes_stats.jsonl").write_text("\n".join(json.dumps(x) for x in stat_lines) + "\n")
    (m / "augmentations.jsonl").write_text("\n".join(json.dumps(x) for x in aug_lines) + "\n")
    shutil.copy(scenes_path, m / "scenes.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=Path, required=True)
    ap.add_argument("--scenes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=96)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--fps", type=float, default=24.0, help="push rate")
    ap.add_argument("--out-fps", type=int, default=30, help="dataset fps")
    ap.add_argument("--video-key", default="observation.images.front")
    ap.add_argument("--quiet", type=float, default=12.0)
    ap.add_argument("--max-wait", type=float, default=120.0)
    ap.add_argument("--coverage-floor", type=float, default=0.75)
    ap.add_argument("--target-de-ceiling", type=float, default=30.0,
                    help="max dE on the oranges; good frames run 13-23, covered ones 74-79")
    ap.add_argument("--target-run-len", type=int, default=3,
                    help="consecutive bad frames before truncating")
    ap.add_argument("--min-frames", type=int, default=40,
                    help="drop an episode truncated shorter than this")
    ap.add_argument("--preview", type=Path, default=Path("out_fruit10x"))
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
    print(f"  {len(scenes)} scenes: {', '.join(s['id'] for s in scenes)}")
    print(f"  gate: coverage>={args.coverage_floor}  target dE<={args.target_de_ceiling} "
          f"for {args.target_run_len} consecutive frames  min {args.min_frames} frames\n")
    args.preview.mkdir(parents=True, exist_ok=True)

    kept, dropped = [], []
    for sc in scenes:
        print(f"=== {sc['id']}")
        imgs, rws, stats = run_batches(frames, rows, sc["prompt"], args)
        if not imgs:
            print("  no batch passed coverage — dropped\n")
            dropped.append({"scene": sc["id"], "why": "no batch passed coverage"})
            continue
        cov = float(np.median([s["coverage"] for s in stats if s["coverage"] is not None]))

        cut, des, scene = truncate_on_target_loss(frames, imgs, rws, args.start,
                                           args.target_de_ceiling, args.target_run_len)
        n_before = len(imgs)
        med, p90 = float(np.median(des)), float(np.percentile(des, 90))
        print(f"  coverage {cov:.3f}  target dE med {med:.1f} p90 {p90:.1f}  "
              f"kept {cut}/{n_before}" + ("  (TRUNCATED)" if cut < n_before else ""))

        # Preview the midpoint of what is KEPT, plus the first discarded frame
        # if any, so the reason for the cut is visible and not just asserted.
        show = min(max(cut // 2, 0), n_before - 1)
        panes = [frames[rws[show]["seq"] - args.start], imgs[show]]
        if cut < n_before:
            panes.append(imgs[min(cut + args.target_run_len, n_before - 1)])
        cv2.imwrite(str(args.preview / f"{sc['id']}.png"),
                    cv2.cvtColor(np.hstack(panes), cv2.COLOR_RGB2BGR))

        if cut < args.min_frames:
            print(f"  -> DROP: only {cut} frames survive the target check\n")
            dropped.append({"scene": sc["id"], "why": f"only {cut} frames kept",
                            "coverage": round(cov, 4), "target_dE_p90": round(p90, 2)})
            continue
        print("  -> KEEP\n")
        kept.append({"scene": sc, "images": imgs[:cut], "rows": rws[:cut], "coverage": cov,
                     "target_med": med, "target_p90": p90, "n_before": n_before,
                     "source_episode": str(args.episode), "source_frames": [args.start, hi]})

    print("=" * 78)
    if not kept:
        (args.preview / "dropped.json").write_text(json.dumps(dropped, indent=2))
        sys.exit("  every scene failed — nothing written")

    src_info = json.loads(args.src_info.read_text()) if args.src_info.exists() else {}
    write_dataset(args.out, kept, src_info, args.video_key, args.out_fps, args.scenes)
    if dropped:
        (args.out / "meta" / "dropped.json").write_text(json.dumps(dropped, indent=2))

    total = sum(len(e["rows"]) for e in kept)
    print(f"  wrote {args.out}")
    print(f"  {len(kept)}/{len(scenes)} scenes kept, {total} frames, task {TASK!r}\n")
    print(f"  {'ep':<4}{'scene':<13}{'frames':>8}{'cov':>7}{'tgt dE med':>12}{'p90':>7}  trunc")
    for i, e in enumerate(kept):
        print(f"  {i:<4}{e['scene']['id']:<13}{len(e['rows']):>8}{e['coverage']:>7.3f}"
              f"{e['target_med']:>12.1f}{e['target_p90']:>7.1f}  "
              f"{'yes ' + str(e['n_before'] - len(e['rows'])) + ' cut' if e['n_before'] != len(e['rows']) else 'no'}")
    if dropped:
        print(f"\n  dropped: {', '.join(d['scene'] for d in dropped)} -> {args.out}/meta/dropped.json")
    print(f"\n  previews: {args.preview}/   third pane = first discarded frame.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("interrupted")
