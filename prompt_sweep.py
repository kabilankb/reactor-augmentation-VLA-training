#!/usr/bin/env python3
"""How hard can you push an X2 prompt before geometry breaks?

The GR1 validation showed the gate's original worry was the wrong one: geometry
survived easily, but the appearance barely moved, and an augmentation that does
not diversify appearance is worth nothing. So the real question is a curve, not
a point — push progressively stronger prompts and measure both:

  APPEARANCE  median CIELAB dE between source and edit. How much the look
              actually moved. Below ~5 the change is imperceptible and the
              augmentation buys no diversity.
  DRIFT       local geometric residual after removing global reframing. Above
              ~0.4% of the diagonal the action labels start to rot.

The usable prompt is the one with the highest dE whose drift is still low. If
no prompt clears both bars, X2 is not a useful augmentation stage for this data.

    python prompt_sweep.py --frames-dir episodes/gr1_ep0/rgb --count 48
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from reactor_sdk import Reactor
from validate_x2 import Received, load_frames, pair_frames, residual_drift

MODEL = "xmax/x2"

# Escalating edits. Every one names lighting, materials, and background only —
# never the microwave, the door, or the hand. The last is deliberately over the
# line: a full style transfer, included to show where the curve breaks.
LADDER: list[tuple[str, str]] = [
    ("1-minimal", "slightly warmer lighting"),
    ("2-mild", "warm morning light, soft shadows"),
    ("3-moderate", "warm morning kitchen light, aged worn laminate and brushed metal surfaces"),
    (
        "4-strong",
        "golden hour sunlight through window blinds, long hard shadows, scratched and worn "
        "surfaces, dust in the air, cluttered background shelves",
    ),
    (
        "5-heavy",
        "dim moody blue evening light, wet reflective surfaces, heavy film grain, rusted "
        "industrial metal, dark cluttered background",
    ),
    ("6-extreme", "watercolour painting, loose visible brushstrokes, paper texture"),
]


async def run_prompt(frames: list[np.ndarray], prompt: str, fps: float, quiet_s: float, max_wait: float):
    reactor = Reactor(model_name=MODEL, api_key=os.environ["REACTOR_API_KEY"])
    received: list[Received] = []
    arrivals: list[float] = []

    async with reactor:
        await reactor.connect()
        source = reactor.track("source")
        await source.publish()
        output = reactor.track("main_video")

        @output.on_frame
        def _collect(frame, frame_id, timestamp_us, user_data):  # noqa: ANN001
            received.append(
                Received(frame.copy(), frame_id, timestamp_us, bytes(user_data) if user_data else None, len(received))
            )
            arrivals.append(time.monotonic())

        await reactor.send_command("set_keep_backlog", {"keep_backlog": True})
        await reactor.send_command("set_prompt", {"prompt": prompt})

        interval = 1.0 / fps
        nxt = time.monotonic()
        for f in frames:
            source.push_frame(f)
            nxt += interval
            await asyncio.sleep(max(0.0, nxt - time.monotonic()))

        # Quiet-based drain: stop as soon as output stops arriving, rather than
        # burning a fixed wait on every rung of the ladder.
        deadline = time.monotonic() + max_wait
        while time.monotonic() < deadline:
            await asyncio.sleep(0.5)
            if arrivals and (time.monotonic() - arrivals[-1]) > quiet_s:
                break
        source.unpublish()

    return received


def appearance_delta(src: np.ndarray, edited: np.ndarray) -> float:
    """Median perceptual colour distance (CIELAB dE) between frame and edit."""
    a = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(np.float32)
    b = cv2.cvtColor(edited, cv2.COLOR_RGB2LAB).astype(np.float32)
    return float(np.median(np.linalg.norm(a - b, axis=2)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-dir", type=Path, required=True)
    ap.add_argument("--count", type=int, default=48)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--quiet", type=float, default=12.0)
    ap.add_argument("--max-wait", type=float, default=90.0)
    ap.add_argument("--out", type=Path, default=Path("out_sweep"))
    ap.add_argument("--drift-budget", type=float, default=0.4, help="%% of diagonal")
    ap.add_argument("--de-floor", type=float, default=5.0, help="minimum useful dE")
    args = ap.parse_args()

    if not os.environ.get("REACTOR_API_KEY"):
        sys.exit("set REACTOR_API_KEY")

    args.out.mkdir(parents=True, exist_ok=True)
    frames = load_frames(args.frames_dir, args.count)
    h, w = frames[0].shape[:2]
    diag = float(np.hypot(w, h))
    print(f"{len(frames)} frames at {w}x{h}\n")

    rows = []
    for label, prompt in LADDER:
        print(f"=== {label}: {prompt[:64]}{'...' if len(prompt) > 64 else ''}")
        received = asyncio.run(run_prompt(frames, prompt, args.fps, args.quiet, args.max_wait))
        if not received:
            print("  no frames returned\n")
            continue

        pairs, how = pair_frames(frames, received)
        des, drifts, matches = [], [], []
        for seq, rec in pairs:
            src = frames[seq]
            edit = cv2.resize(rec.frame, (w, h), interpolation=cv2.INTER_AREA)
            des.append(appearance_delta(src, edit))
            res, n_in, _, _ = residual_drift(src, edit, None)
            if res is not None and n_in >= 12:
                drifts.append(res)
                matches.append(n_in)

        de = float(np.median(des)) if des else 0.0
        drift = float(np.median(drifts)) if drifts else None
        pct = (drift / diag * 100) if drift is not None else None

        # Save a look at the midpoint of each rung.
        if pairs:
            mid = pairs[len(pairs) // 2]
            src = frames[mid[0]]
            edit = cv2.resize(mid[1].frame, (w, h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(args.out / f"{label}.png"),
                        cv2.cvtColor(np.hstack([src, edit]), cv2.COLOR_RGB2BGR))

        rows.append({
            "label": label, "prompt": prompt, "delta_e": de,
            "drift_px": drift, "drift_pct": pct,
            "matches": float(np.median(matches)) if matches else 0.0,
            "received": len(received), "sent": len(frames), "pairing": how,
        })
        print(f"  dE {de:5.1f}   drift {drift if drift is None else round(drift, 2)}px "
              f"({'n/a' if pct is None else f'{pct:.2f}%'})   "
              f"{len(received)}/{len(frames)} frames\n")

    print("=" * 72)
    print(f"  {'rung':<12} {'dE':>6} {'drift px':>9} {'drift %':>8} {'matches':>8}  verdict")
    usable = []
    for r in rows:
        ok_de = r["delta_e"] >= args.de_floor
        ok_dr = r["drift_pct"] is not None and r["drift_pct"] <= args.drift_budget
        v = "USABLE" if (ok_de and ok_dr) else ("no change" if not ok_de else "drift too high")
        if ok_de and ok_dr:
            usable.append(r)
        d_px = "n/a" if r["drift_px"] is None else f"{r['drift_px']:.2f}"
        d_pct = "n/a" if r["drift_pct"] is None else f"{r['drift_pct']:.2f}"
        print(f"  {r['label']:<12} {r['delta_e']:>6.1f} {d_px:>9} {d_pct:>8} "
              f"{r['matches']:>8.0f}  {v}")

    print()
    if usable:
        best = max(usable, key=lambda r: r["delta_e"])
        print(f"  BEST: {best['label']} — dE {best['delta_e']:.1f} at "
              f"{best['drift_pct']:.2f}% drift")
        print(f"    {best['prompt']}")
    else:
        print("  NO USABLE RUNG — every prompt either changed nothing or moved geometry.")
        print("  X2 is not a useful augmentation stage for this data.")

    (args.out / "sweep.json").write_text(json.dumps(rows, indent=2))
    print(f"\n  comparisons: {args.out}/<rung>.png   raw: {args.out}/sweep.json")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("interrupted")
