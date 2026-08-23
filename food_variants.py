#!/usr/bin/env python3
"""Find a food-adding prompt that does not erase the robot hand.

The naive attempt ("...a white plate of roast vegetables and bread with rising
steam") appended to a rung-5 style prompt dropped structure coverage to 0.155:
X2 painted plates of food over the region where the gripper was, and put them
on the counter rather than inside the microwave.

Additive content competes for canvas space. These variants attack that three
ways — pinning the food's location, naming the hand as preserved, and lowering
the styling budget so the food has less license — and score each on coverage
first, since a prompt that erases the manipulator is useless regardless of how
good it looks.

    python food_variants.py --frames-dir episodes/gr1_ep0/rgb --count 48
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from prompt_sweep import appearance_delta, run_prompt
from validate_x2 import load_frames, pair_frames, residual_drift, structure_coverage

STYLE = (
    "dim moody blue evening light, wet reflective surfaces, heavy film grain, "
    "rusted industrial metal, dark cluttered background"
)

VARIANTS: list[tuple[str, str]] = [
    # The failing baseline, kept so the comparison is honest.
    ("A-naive", f"{STYLE}, a white plate of roast vegetables and bread with rising steam"),
    # Pin the food inside the appliance instead of leaving placement free.
    ("B-pinned", f"{STYLE}, roast vegetables on the white plate inside the microwave"),
    # Name the hand as preserved. X2 carries through what you don't mention, but
    # naming a thing as unchanged may also anchor it.
    (
        "C-protected",
        f"{STYLE}, roast vegetables on the white plate inside the microwave, "
        "the black robotic hand and the microwave unchanged",
    ),
    # Drop the styling budget so the food is the only additive demand.
    (
        "D-food-only",
        "roast vegetables on the white plate inside the microwave, "
        "warm interior glow, everything else unchanged",
    ),
    # Food as a lighting/colour cue rather than an object.
    (
        "E-implied",
        f"{STYLE}, warm golden glow of hot food spilling from the microwave opening",
    ),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-dir", type=Path, required=True)
    ap.add_argument("--count", type=int, default=48)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--quiet", type=float, default=12.0)
    ap.add_argument("--max-wait", type=float, default=90.0)
    ap.add_argument("--out", type=Path, default=Path("out_food"))
    ap.add_argument("--coverage-floor", type=float, default=0.80)
    ap.add_argument("--variants", type=Path, help="JSON [[label, prompt], ...] overriding the built-ins")
    args = ap.parse_args()

    if not os.environ.get("REACTOR_API_KEY"):
        sys.exit("set REACTOR_API_KEY")

    args.out.mkdir(parents=True, exist_ok=True)
    frames = load_frames(args.frames_dir, args.count)
    h, w = frames[0].shape[:2]
    diag = float(np.hypot(w, h))
    print(f"{len(frames)} frames at {w}x{h}\n")

    variants = json.loads(args.variants.read_text()) if args.variants else VARIANTS
    rows = []
    for label, prompt in variants:
        print(f"=== {label}")
        received = asyncio.run(run_prompt(frames, prompt, args.fps, args.quiet, args.max_wait))
        if not received:
            print("  no frames returned\n")
            continue

        pairs, _ = pair_frames(frames, received)
        covs, drifts, des = [], [], []
        for seq, rec in pairs:
            src = frames[seq]
            edit = cv2.resize(rec.frame, (w, h), interpolation=cv2.INTER_AREA)
            covs.append(structure_coverage(src, edit))
            des.append(appearance_delta(src, edit))
            r, n, _, _ = residual_drift(src, edit, None)
            if r is not None and n >= 12:
                drifts.append(r)

        cov = float(np.median(covs))
        de = float(np.median(des))
        drift = float(np.median(drifts)) if drifts else None

        if pairs:
            mid = pairs[len(pairs) // 2]
            src = frames[mid[0]]
            edit = cv2.resize(mid[1].frame, (w, h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(args.out / f"{label}.png"),
                        cv2.cvtColor(np.hstack([src, edit]), cv2.COLOR_RGB2BGR))

        rows.append({"label": label, "prompt": prompt, "coverage": cov, "delta_e": de,
                     "drift_px": drift, "drift_pct": (drift / diag * 100) if drift else None,
                     "received": len(received), "sent": len(frames)})
        print(f"  coverage {cov:.3f}   dE {de:5.1f}   "
              f"drift {'n/a' if drift is None else f'{drift:.2f}px'}\n")

    print("=" * 66)
    print(f"  {'variant':<14} {'coverage':>9} {'dE':>7} {'drift px':>9}  verdict")
    ok = []
    for r in rows:
        good = r["coverage"] >= args.coverage_floor
        v = "usable" if good else ("ERASES STRUCTURE" if r["coverage"] < 0.6 else "partial loss")
        if good:
            ok.append(r)
        d = "n/a" if r["drift_px"] is None else f"{r['drift_px']:.2f}"
        print(f"  {r['label']:<14} {r['coverage']:>9.3f} {r['delta_e']:>7.1f} {d:>9}  {v}")

    print()
    if ok:
        best = max(ok, key=lambda r: r["delta_e"])
        print(f"  BEST: {best['label']} — coverage {best['coverage']:.3f}, dE {best['delta_e']:.1f}")
        print(f"    {best['prompt']}")
    else:
        print("  NO VARIANT PRESERVED THE SCENE.")
        print("  Adding objects via prompt costs structure on this data; put the")
        print("  food in the sim instead, where it occludes correctly and the")
        print("  labels stay valid.")

    (args.out / "variants.json").write_text(json.dumps(rows, indent=2))
    print(f"\n  images: {args.out}/<variant>.png   raw: {args.out}/variants.json")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("interrupted")
