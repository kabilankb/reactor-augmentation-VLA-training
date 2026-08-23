#!/usr/bin/env python3
"""Build a multi-fruit pick dataset from a single-fruit one.

Takes an episode of "Grab orange and place into plate", swaps the orange for a
different fruit through X2, rewrites the language instruction to match, and
writes everything into one robomimic-style HDF5.

The actions are reused unchanged, which is only sound because the substitute
occupies the same position and roughly the same size as the original — a
recorded grasp for an orange-sized sphere still closes on an apple. It does not
close on a blueberry, and it is the wrong shape entirely for a banana, so the
fruit list here is filtered to grasp-compatible substitutes.

Swapping the fruit WITHOUT rewriting the instruction would teach the policy that
"orange" means "whatever object is on the table", so the task string is rewritten
per demo and stored alongside.

    python make_fruit_dataset.py --episode episodes/orange_ep0 \\
        --fruits apple lemon lime --count 96 --out datasets/fruit_pick.hdf5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

from prompt_sweep import run_prompt
from validate_x2 import load_frames, pair_frames, residual_drift, structure_coverage

# Substitutes that keep a recorded orange-grasp valid: roughly spherical, and
# within about half to double an orange's diameter.
GRASP_COMPATIBLE = {
    "apple": "a red apple",
    "green apple": "a green apple",
    "lemon": "a bright yellow lemon",
    "lime": "a green lime",
    "guava": "a pale green guava",
    "mango": "a red and yellow mango",
    "grapefruit": "a large pink grapefruit",
    "peach": "a ripe peach",
    "plum": "a dark purple plum",
    "kiwi": "a brown fuzzy kiwifruit",
}

# Rejected, with the reason kept visible rather than silently dropped.
INCOMPATIBLE = {
    "banana": "elongated — a recorded spherical grasp does not transfer",
    "pineapple": "far larger than an orange, and needs a different approach",
    "papaya": "far larger and elongated",
    "grapes": "a bunch, not a single graspable object",
    "strawberry": "much smaller — the gripper would close on air",
    "blueberry": "far too small",
    "raspberry": "far too small",
    "blackberry": "far too small",
    "cherry": "much smaller, and usually stemmed",
}


def prompt_for(descriptor: str) -> str:
    """Replacement prompt.

    Names the oranges explicitly because here we WANT them transformed — the
    same phrasing that accidentally destroyed targets when the goal was to add
    fruit alongside them. Everything else is pinned as unchanged so the plate,
    arm, and table geometry survive.
    """
    return (
        f"replace every orange on the table with {descriptor}, "
        "same size and same position, "
        "the plate, the robot arm and the table unchanged"
    )


def instruction_for(name: str) -> str:
    return f"Grab {name} and place into plate"


def load_manifest(episode: Path) -> list[dict]:
    rows = [json.loads(l) for l in (episode / "manifest.jsonl").read_text().splitlines() if l.strip()]
    return sorted(rows, key=lambda r: r["seq"])


async def build_one(frames, descriptor, fps, quiet, max_wait):
    return await run_prompt(frames, prompt_for(descriptor), fps, quiet, max_wait)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=Path, required=True, help="episode dir with rgb/ and manifest.jsonl")
    ap.add_argument("--fruits", nargs="+", required=True, help=f"from: {', '.join(GRASP_COMPATIBLE)}")
    ap.add_argument("--start", type=int, default=0, help="first source frame")
    ap.add_argument("--count", type=int, default=96, help="frames per fruit (keep near 96)")
    ap.add_argument("--resize", type=int, default=256, help="stored image size; 0 keeps native")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--quiet", type=float, default=12.0)
    ap.add_argument("--max-wait", type=float, default=120.0)
    ap.add_argument("--coverage-floor", type=float, default=0.75)
    ap.add_argument("--out", type=Path, default=Path("datasets/fruit_pick.hdf5"))
    ap.add_argument("--preview-dir", type=Path, default=Path("out_fruitset"))
    args = ap.parse_args()

    if not os.environ.get("REACTOR_API_KEY"):
        sys.exit("set REACTOR_API_KEY")

    bad = [f for f in args.fruits if f not in GRASP_COMPATIBLE]
    if bad:
        for f in bad:
            why = INCOMPATIBLE.get(f, "not in the compatible list")
            print(f"  refusing {f!r}: {why}")
        sys.exit("remove the incompatible fruits, or add them to GRASP_COMPATIBLE deliberately")

    args.preview_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args.episode)
    all_frames = load_frames(args.episode / "rgb", len(manifest))
    lo, hi = args.start, min(args.start + args.count, len(all_frames))
    frames = all_frames[lo:hi]
    rows = manifest[lo:hi]
    h, w = frames[0].shape[:2]
    diag = float(np.hypot(w, h))
    print(f"  source: {len(frames)} frames [{lo}:{hi}] at {w}x{h}\n")

    demos = []
    for name in args.fruits:
        print(f"=== {name}")
        received = asyncio.run(build_one(frames, GRASP_COMPATIBLE[name], args.fps, args.quiet, args.max_wait))
        if not received:
            print("  no frames returned — skipping\n")
            continue

        pairs, how = pair_frames(frames, received)
        covs, drifts, imgs, acts, seqs = [], [], [], [], []
        for seq, rec in pairs:
            src = frames[seq]
            edit = cv2.resize(rec.frame, (w, h), interpolation=cv2.INTER_AREA)
            covs.append(structure_coverage(src, edit))
            r, n, _, _ = residual_drift(src, edit, None)
            if r is not None and n >= 12:
                drifts.append(r)
            out = edit if args.resize == 0 else cv2.resize(edit, (args.resize, args.resize),
                                                           interpolation=cv2.INTER_AREA)
            imgs.append(out)
            acts.append(rows[seq]["action"])
            seqs.append(rows[seq]["seq"])

        cov = float(np.median(covs))
        drift = float(np.median(drifts)) if drifts else float("nan")
        keep = cov >= args.coverage_floor
        print(f"  coverage {cov:.3f}  drift {drift:.2f}px ({drift / diag * 100:.2f}%)  "
              f"{len(received)}/{len(frames)} frames  pairing {how}")
        print(f"  -> {'KEEP' if keep else 'REJECT (structure lost)'}\n")

        mid = pairs[len(pairs) // 2]
        cv2.imwrite(str(args.preview_dir / f"{name.replace(' ', '_')}.png"),
                    cv2.cvtColor(np.hstack([frames[mid[0]],
                                            cv2.resize(mid[1].frame, (w, h), interpolation=cv2.INTER_AREA)]),
                                 cv2.COLOR_RGB2BGR))
        if keep:
            demos.append({"name": name, "images": np.asarray(imgs, np.uint8),
                          "actions": np.asarray(acts, np.float32), "src_seq": np.asarray(seqs, np.int32),
                          "coverage": cov, "drift_px": drift})

    if not demos:
        sys.exit("no fruit passed the coverage floor — nothing written")

    with h5py.File(args.out, "w") as f:
        data = f.create_group("data")
        data.attrs["total"] = int(sum(len(d["actions"]) for d in demos))
        data.attrs["env_args"] = json.dumps({
            "env_name": "leisaac_pick_fruit",
            "source_dataset": "LightwheelAI/leisaac-pick-orange",
            "augmentation": "reactor xmax/x2 fruit substitution",
        })
        for i, d in enumerate(demos):
            g = data.create_group(f"demo_{i}")
            g.attrs["num_samples"] = len(d["actions"])
            g.attrs["fruit"] = d["name"]
            # The instruction is rewritten to match the pixels. Leaving it as
            # "orange" would teach the policy that the word names any object.
            g.attrs["language_instruction"] = instruction_for(d["name"])
            g.attrs["coverage"] = d["coverage"]
            g.attrs["drift_px"] = d["drift_px"]
            g.attrs["source_episode"] = str(args.episode)
            g.create_dataset("actions", data=d["actions"], compression="gzip")
            g.create_dataset("src_seq", data=d["src_seq"], compression="gzip")
            obs = g.create_group("obs")
            obs.create_dataset("front_image", data=d["images"], compression="gzip", chunks=True)

    total = sum(len(d["actions"]) for d in demos)
    print("=" * 62)
    print(f"  wrote {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")
    print(f"  {len(demos)} demos, {total} frames, images {args.resize or w}px")
    for i, d in enumerate(demos):
        print(f"    demo_{i}: {d['name']:<12} {len(d['actions']):>4} frames  "
              f"coverage {d['coverage']:.3f}  \"{instruction_for(d['name'])}\"")
    print(f"  previews: {args.preview_dir}/")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("interrupted")
