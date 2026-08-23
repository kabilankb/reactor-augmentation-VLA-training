#!/usr/bin/env python3
"""Replace the oranges with a different fruit per episode, and gate on IDENTITY.

This is the run PROJECT.md section 3 recorded as failing. Three of four attempts
did not transform the oranges at all -- they transformed the yellow SO-101 arm,
the largest saturated object in frame, into a green apple / two lemons / a lime.
Coverage and drift passed every one of them, because edge structure survives
when one blob is repainted as another blob.

Two things are different here.

PROMPT. The earlier attempt said "replace every orange with a lemon". The word
"orange" names a colour as much as a fruit, and the arm is yellow-orange, so the
phrase binds to the arm. This binds by position, count and size instead -- three
small round fruits, on the table, in front of the plate -- and pins the arm
explicitly. Styling is dropped entirely: section 2 established that additive or
substitutive edits compete with restyling and you get one or the other.

GATE. Coverage cannot see identity: an orange repainted as an apple has the same
silhouette. So this measures WHERE the edit landed. The three oranges are ~1% of
the frame and separate cleanly from the wooden table (same hue, much lower value)
and from the arm (much lower saturation), so a source-side colour mask locates
them without any simulator masks:

    fruit_dE   median CIELAB dE INSIDE the fruit mask   -- must be HIGH
    bg_dE      median dE OUTSIDE it, dilated            -- must be LOW
    selectivity = fruit_dE / bg_dE                      -- must be >> 1

Measured on the three quarantined failures in
datasets/fruit_pick_INVALID_arm_destroyed.hdf5, selectivity is 1.37 / 1.39 /
1.38 -- the edit moved the fruit and the background by the same amount, which is
exactly what "the arm changed and the fruit did not" looks like. The floors
below reject all three.

    python build_substitution_dataset.py --episode episodes/orange_ep0 \
        --fruits apple plum peach --count 96 --out datasets/lerobot_subst

Output: LeRobot v2.1, one episode and one language instruction per fruit.
"""

from __future__ import annotations

import argparse
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
from build_fruit_datasets import encode_video, feature_stats
from validate_x2 import load_frames

# Substitutes that keep a recorded orange-grasp valid: roughly spherical and
# within about half to double an orange's diameter. A banana is a fine
# distractor but not a substitute -- the recorded grasp does not transfer.
#
# `risk` flags fruits whose colour is close to the yellow arm. Every documented
# failure was yellow or green (lemon, lime, green apple); the arm is the thing
# those words most resemble in this scene.
FRUITS: dict[str, dict] = {
    "apple":       {"descriptor": "deep red apples",                 "risk": "low"},
    "plum":        {"descriptor": "dark purple plums",               "risk": "low"},
    "peach":       {"descriptor": "pink and orange peaches",         "risk": "low"},
    "pomegranate": {"descriptor": "dark red pomegranates",           "risk": "low"},
    "kiwi":        {"descriptor": "brown fuzzy kiwifruit",           "risk": "low"},
    "mango":       {"descriptor": "red and yellow mangoes",          "risk": "med"},
    "grapefruit":  {"descriptor": "large pink grapefruit",           "risk": "med"},
    "guava":       {"descriptor": "pale green guavas",               "risk": "high"},
    "lemon":       {"descriptor": "bright yellow lemons",            "risk": "high"},
    "lime":        {"descriptor": "green limes",                     "risk": "high"},
}


def prompt_for(descriptor: str) -> str:
    """Bind the edit to the fruit by position, count and size -- not by colour.

    "replace every orange with X" is what transformed the arm. Naming the count
    and the location gives the model a specific, small, unambiguous target, and
    naming the arm as unchanged defends the thing that actually got destroyed.
    No styling terms at all: they compete for the same edit budget.
    """
    return (
        f"the three small round fruits on the wooden table in front of the white plate "
        f"are {descriptor}, same size and same position, "
        f"the yellow robot arm is unchanged, the white plate and the wooden table are unchanged"
    )


def instruction_for(name: str) -> str:
    return f"Grab {name} and place into plate"


# ---------------------------------------------------------------- identity gate


def fruit_mask(rgb: np.ndarray) -> np.ndarray:
    """Locate the oranges in a SOURCE frame.

    Measured on this scene: fruit S=183 V=172, wooden table S=124 V=118 (same
    hue, much darker), arm S=119 V=188 (much less saturated), plate S=4. So
    saturation AND value together separate the fruit from everything else that
    shares its hue. The blob-area filter drops specular flecks on the table.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = (hsv[..., i].astype(int) for i in range(3))
    m = ((s > 155) & (v > 140) & (h >= 5) & (h < 20)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    area = m.shape[0] * m.shape[1]
    for i in range(1, n):
        if 0.0004 * area < st[i, cv2.CC_STAT_AREA] < 0.05 * area:
            keep[lab == i] = 1
    return keep


def edit_selectivity(src: np.ndarray, edit: np.ndarray) -> tuple[float, float, float] | None:
    """(fruit_dE, bg_dE, selectivity) -- did the edit land on the fruit?

    The protected region is the complement of the fruit mask dilated well past
    the fruit, so a substitute slightly larger than the original does not leak
    into the background term and depress the ratio.
    """
    fm = fruit_mask(src)
    if fm.sum() < 50:
        return None
    protected = cv2.dilate(fm, np.ones((9, 9), np.uint8), iterations=2) == 0

    a = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(np.float32)
    b = cv2.cvtColor(edit, cv2.COLOR_RGB2LAB).astype(np.float32)
    de = np.linalg.norm(a - b, axis=2)

    fruit_de = float(np.median(de[fm > 0]))
    bg_de = float(np.median(de[protected]))
    return fruit_de, bg_de, fruit_de / max(bg_de, 1e-6)


def score_episode(frames, imgs, rows, start: int) -> dict:
    """Identity metrics for a whole episode, sampled every few frames."""
    fs, bs, sel = [], [], []
    for i in range(0, len(imgs), 4):
        src = frames[rows[i]["seq"] - start]
        r = edit_selectivity(src, imgs[i])
        if r:
            fs.append(r[0])
            bs.append(r[1])
            sel.append(r[2])
    if not sel:
        return {"fruit_dE": None, "bg_dE": None, "selectivity": None}
    return {"fruit_dE": float(np.median(fs)), "bg_dE": float(np.median(bs)),
            "selectivity": float(np.median(sel))}


def identity_verdict(m: dict, args) -> tuple[bool, str]:
    if m["selectivity"] is None:
        return False, "fruit mask empty — cannot judge identity"
    if m["fruit_dE"] < args.fruit_de_floor:
        return False, f"fruit barely changed (dE {m['fruit_dE']:.1f}) — substitution did not happen"
    if m["bg_dE"] > args.bg_de_ceiling:
        return False, f"background changed too much (dE {m['bg_dE']:.1f}) — the arm or scene was edited"
    if m["selectivity"] < args.selectivity_floor:
        return False, f"edit not selective (ratio {m['selectivity']:.2f}) — landed off-target"
    return True, f"selective substitution (ratio {m['selectivity']:.2f})"


# --------------------------------------------------------------------- writer


def write_multitask(out: Path, episodes: list[dict], src_info: dict, video_key: str, fps: int) -> None:
    """LeRobot v2.1 with ONE TASK PER EPISODE.

    Substitution rewrites the instruction -- that is the whole point, otherwise
    the policy learns "orange" names any object. So tasks.jsonl carries one row
    per fruit and each episode points at its own task_index. The existing
    builders hardcode a single shared task, which is right for distractors and
    wrong here.
    """
    if out.exists():
        shutil.rmtree(out)
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)

    ep_lines, stat_lines, task_lines, aug_lines = [], [], [], []
    global_index = 0

    for ei, ep in enumerate(episodes):
        rows, imgs = ep["rows"], ep["images"]
        n = len(rows)
        task = instruction_for(ep["fruit"])
        task_lines.append({"task_index": ei, "task": task})

        acts = np.stack([np.asarray(r["action"], np.float32) for r in rows])
        # No silent fallback to `action`: a dataset whose state IS its label
        # trains a policy that reads the answer off its own input.
        if any(not r.get("state") for r in rows):
            sys.exit("source manifest has no `state` — refusing to substitute actions for it")
        states = np.stack([np.asarray(r["state"], np.float32) for r in rows])

        df = pd.DataFrame({
            "action": list(acts),
            "observation.state": list(states),
            "timestamp": np.arange(n, dtype=np.float32) / fps,
            "frame_index": np.arange(n, dtype=np.int64),
            "episode_index": np.full(n, ei, dtype=np.int64),
            "index": np.arange(global_index, global_index + n, dtype=np.int64),
            "task_index": np.full(n, ei, dtype=np.int64),
            "source_seq": np.asarray([r["seq"] for r in rows], np.int64),
        })
        global_index += n
        df.to_parquet(out / "data" / "chunk-000" / f"episode_{ei:06d}.parquet", index=False)
        encode_video(imgs, out / "videos" / "chunk-000" / video_key / f"episode_{ei:06d}.mp4", fps)

        ep_lines.append({"episode_index": ei, "tasks": [task], "length": n})
        stat_lines.append({"episode_index": ei, "stats": {
            "action": feature_stats(acts), "observation.state": feature_stats(states)}})
        aug_lines.append({
            "episode_index": ei, "task": task, "fruit": ep["fruit"],
            "substituted_for": "orange", "prompt": ep["prompt"],
            "coverage": round(ep["coverage"], 4),
            "fruit_dE": round(ep["identity"]["fruit_dE"], 2),
            "bg_dE": round(ep["identity"]["bg_dE"], 2),
            "selectivity": round(ep["identity"]["selectivity"], 3),
            "source_dataset": "LightwheelAI/leisaac-pick-orange",
            "source_episode": ep["source_episode"], "source_frames": ep["source_frames"],
            "frames_kept": n, "augmenter": "reactor xmax/x2",
            "note": "target object substituted; actions reused unchanged because the "
                    "substitute keeps the position and size the recorded grasp assumed",
        })

    h, w = episodes[0]["images"][0].shape[:2]
    dim = int(acts.shape[1])
    total = sum(len(e["rows"]) for e in episodes)
    info = {
        "codebase_version": "v2.1",
        "robot_type": src_info.get("robot_type", "so101_follower"),
        "total_episodes": len(episodes), "total_frames": total,
        "total_tasks": len(task_lines), "total_videos": len(episodes),
        "total_chunks": 1, "chunks_size": 1000, "fps": fps,
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
    (m / "tasks.jsonl").write_text("\n".join(json.dumps(x) for x in task_lines) + "\n")
    (m / "episodes.jsonl").write_text("\n".join(json.dumps(x) for x in ep_lines) + "\n")
    (m / "episodes_stats.jsonl").write_text("\n".join(json.dumps(x) for x in stat_lines) + "\n")
    (m / "augmentations.jsonl").write_text("\n".join(json.dumps(x) for x in aug_lines) + "\n")


# ----------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=Path, required=True)
    ap.add_argument("--fruits", nargs="+", default=list(FRUITS),
                    help=f"from: {', '.join(FRUITS)}")
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
    ap.add_argument("--fruit-de-floor", type=float, default=25.0,
                    help="min dE inside the fruit — below this nothing was substituted")
    ap.add_argument("--bg-de-ceiling", type=float, default=8.0,
                    help="max dE outside the fruit — above this the scene/arm was edited")
    ap.add_argument("--selectivity-floor", type=float, default=2.5,
                    help="min fruit_dE/bg_dE — the 3 quarantined failures scored 1.37-1.39")
    ap.add_argument("--preview", type=Path, default=Path("out_subst"))
    ap.add_argument("--src-info", type=Path, default=Path("datasets/orange/info.json"))
    args = ap.parse_args()

    if not os.environ.get("REACTOR_API_KEY"):
        sys.exit("set REACTOR_API_KEY")
    bad = [f for f in args.fruits if f not in FRUITS]
    if bad:
        sys.exit(f"unknown fruit(s): {bad}. choose from {list(FRUITS)}")

    rows_all = load_manifest(args.episode)
    frames_all = load_frames(args.episode / "rgb", len(rows_all))
    hi = min(args.start + args.count, len(frames_all))
    frames, rows = frames_all[args.start:hi], rows_all[args.start:hi]

    print(f"  source {args.episode} frames [{args.start}:{hi}] ({len(frames)})")
    print(f"  {len(args.fruits)} fruits: {', '.join(args.fruits)}")
    print(f"  gate: coverage>={args.coverage_floor} fruit_dE>={args.fruit_de_floor} "
          f"bg_dE<={args.bg_de_ceiling} selectivity>={args.selectivity_floor}\n")

    args.preview.mkdir(parents=True, exist_ok=True)
    kept_eps, rejected = [], []

    for fruit in args.fruits:
        spec = FRUITS[fruit]
        prompt = prompt_for(spec["descriptor"])
        print(f"=== {fruit}  (colour risk {spec['risk']})")
        imgs, rws, stats = run_batches(frames, rows, prompt, args)
        if not imgs:
            print("  no batch passed coverage — dropped\n")
            rejected.append({"fruit": fruit, "why": "no batch passed coverage"})
            continue

        covs = [s["coverage"] for s in stats if s["coverage"] is not None]
        cov = float(np.median(covs))
        ident = score_episode(frames, imgs, rws, args.start)
        ok, why = identity_verdict(ident, args)

        # Preview every attempt, kept or not: the numbers have passed visibly
        # broken runs before, so the image is the check that actually works.
        mid = len(imgs) // 2
        cv2.imwrite(str(args.preview / f"{fruit.replace(' ', '_')}.png"),
                    cv2.cvtColor(np.hstack([frames[rws[mid]["seq"] - args.start], imgs[mid]]),
                                 cv2.COLOR_RGB2BGR))

        print(f"  coverage {cov:.3f}  fruit_dE {ident['fruit_dE'] or float('nan'):.1f}  "
              f"bg_dE {ident['bg_dE'] or float('nan'):.1f}  "
              f"selectivity {ident['selectivity'] or float('nan'):.2f}")
        print(f"  -> {'KEEP' if ok else 'REJECT'}: {why}\n")

        if ok:
            kept_eps.append({"fruit": fruit, "prompt": prompt, "images": imgs, "rows": rws,
                             "coverage": cov, "identity": ident,
                             "source_episode": str(args.episode),
                             "source_frames": [args.start, hi]})
        else:
            rejected.append({"fruit": fruit, "why": why, "coverage": round(cov, 4), **{
                k: (round(v, 3) if v is not None else None) for k, v in ident.items()}})

    print("=" * 74)
    if not kept_eps:
        (args.preview / "rejected.json").write_text(json.dumps(rejected, indent=2))
        print("  EVERY FRUIT REJECTED — nothing written.")
        print(f"  previews: {args.preview}/   reasons: {args.preview}/rejected.json")
        print("\n  This reproduces PROJECT.md section 3. Substitution belongs in the")
        print("  simulator: swapping a USD asset changes the right object by")
        print("  construction, and the arm is a separate prim that cannot be touched.")
        sys.exit(1)

    src_info = json.loads(args.src_info.read_text()) if args.src_info.exists() else {}
    write_multitask(args.out, kept_eps, src_info, args.video_key, args.out_fps)
    if rejected:
        (args.out / "meta" / "rejected.json").write_text(json.dumps(rejected, indent=2))

    total = sum(len(e["rows"]) for e in kept_eps)
    print(f"  wrote {args.out}")
    print(f"  {len(kept_eps)}/{len(args.fruits)} fruits kept, {total} frames, "
          f"{len(kept_eps)} tasks\n")
    print(f"  {'ep':<4} {'fruit':<13} {'frames':>7} {'cov':>6} {'fruit dE':>9} {'bg dE':>7} {'select':>7}")
    for i, e in enumerate(kept_eps):
        d = e["identity"]
        print(f"  {i:<4} {e['fruit']:<13} {len(e['rows']):>7} {e['coverage']:>6.3f} "
              f"{d['fruit_dE']:>9.1f} {d['bg_dE']:>7.1f} {d['selectivity']:>7.2f}")
    if rejected:
        print(f"\n  rejected: {', '.join(r['fruit'] for r in rejected)} "
              f"-> {args.out}/meta/rejected.json")
    print(f"\n  previews: {args.preview}/   LOOK AT THEM before training on this.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("interrupted")
