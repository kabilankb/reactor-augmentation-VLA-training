#!/usr/bin/env python3
"""Why does X2 return fewer frames than it consumes?

Three explanations, with different consequences for the pipeline:

  A. TAIL TRUNCATION   Frames were still in flight when we stopped draining.
                       Harmless — drain longer.
  B. CONSTANT LOSS     A fixed cost per session (warmup in, tail out). Yield
                       improves with longer runs; amortise it.
  C. PROPORTIONAL LOSS The model emits fewer frames than it eats, throughout.
                       Yield never improves and a fixed pairing offset breaks.

Distinguishing them: run several stream lengths and see whether the *count* of
missing frames stays flat (B) or grows with N (C), and whether output was still
arriving when we stopped (A).

    python probe_yield.py --counts 24 48 96 --drain 150
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from reactor_sdk import Reactor

MODEL = "xmax/x2"
PROMPT = "warm late-afternoon sunlight, worn scuffed tabletop, cluttered background shelves"


async def run_once(frames: list[np.ndarray], fps: float, drain: float, quiet_s: float) -> dict:
    """One session. Returns counts plus the arrival timeline."""
    reactor = Reactor(model_name=MODEL, api_key=os.environ["REACTOR_API_KEY"])
    arrivals: list[float] = []
    t_last_push = None

    async with reactor:
        await reactor.connect()
        source = reactor.track("source")
        await source.publish()
        output = reactor.track("main_video")

        @output.on_frame
        def _collect(frame, frame_id, timestamp_us, user_data):  # noqa: ANN001
            arrivals.append(time.monotonic())

        await reactor.send_command("set_keep_backlog", {"keep_backlog": True})
        await reactor.send_command("set_prompt", {"prompt": PROMPT})

        interval = 1.0 / fps
        nxt = time.monotonic()
        for f in frames:
            source.push_frame(f)
            nxt += interval
            await asyncio.sleep(max(0.0, nxt - time.monotonic()))
        t_last_push = time.monotonic()

        # Drain until output goes quiet for `quiet_s`, or the deadline passes.
        # Stopping on quiet rather than a fixed wait is what separates
        # truncation from real loss: if we never go quiet, we cut it short.
        deadline = t_last_push + drain
        while time.monotonic() < deadline:
            await asyncio.sleep(0.5)
            if arrivals and (time.monotonic() - arrivals[-1]) > quiet_s:
                break

        went_quiet = bool(arrivals) and (time.monotonic() - arrivals[-1]) > quiet_s
        source.unpublish()

    gaps = [b - a for a, b in zip(arrivals, arrivals[1:])] if len(arrivals) > 1 else []
    return {
        "sent": len(frames),
        "received": len(arrivals),
        "missing": len(frames) - len(arrivals),
        "went_quiet": went_quiet,
        "tail_s": (arrivals[-1] - t_last_push) if arrivals else 0.0,
        "first_s": (arrivals[0] - t_last_push) if arrivals else 0.0,
        "median_gap_ms": (statistics.median(gaps) * 1000) if gaps else 0.0,
        "max_gap_ms": (max(gaps) * 1000) if gaps else 0.0,
        "out_fps": (len(arrivals) / (arrivals[-1] - arrivals[0])) if len(arrivals) > 1 else 0.0,
    }


def synth(n: int, w: int = 640, h: int = 480) -> list[np.ndarray]:
    """Frames with moving structure — a static feed would let the encoder
    collapse the stream and confound the count."""
    rng = np.random.default_rng(11)
    base = np.full((h, w, 3), (195, 175, 150), np.uint8)
    cv2.rectangle(base, (0, 250), (w, h), (165, 130, 95), -1)
    base = np.clip(base.astype(np.int16) + (rng.random((h, w, 1)) * 24).astype(np.uint8), 0, 255).astype(np.uint8)
    out = []
    for i in range(n):
        f = base.copy()
        x = 120 + (i * 4) % 380
        cv2.rectangle(f, (x, 300), (x + 70, 370), (200, 52, 48), -1)
        cv2.rectangle(f, (x + 20, 120), (x + 74, 250), (70, 70, 70), -1)
        cv2.circle(f, (520, 180), 34, (90, 150, 60), -1)
        out.append(f)
    return out


def diagnose(rows: list[dict]) -> list[str]:
    """Constant vs proportional loss.

    The discriminator is not how much the missing count varies — it is whether
    it tracks N. Compare the largest run against what a proportional loss rate
    measured on the smallest run would predict: constant loss undershoots that
    prediction badly, proportional loss lands on it.
    """
    if not all(r["went_quiet"] for r in rows):
        return [
            "A. TAIL TRUNCATION — output had not stopped when we gave up.",
            "   Increase --drain; the yield numbers are not yet trustworthy.",
        ]
    if len(rows) < 2:
        return ["inconclusive — need at least two stream lengths"]

    rows = sorted(rows, key=lambda r: r["sent"])
    small, large = rows[0], rows[-1]
    predicted = small["missing"] * (large["sent"] / small["sent"])
    actual = large["missing"]

    yields = ", ".join(f"{r['received'] / r['sent']:.0%}@{r['sent']}" for r in rows)
    out = [f"yield by length: {yields}"]

    # Proportional loss would put the large run near `predicted`; constant loss
    # leaves it near the small run's absolute count.
    if actual <= 0.5 * predicted:
        out += [
            f"B. CONSTANT LOSS — {large['sent']} frames lost {actual}, not the "
            f"{predicted:.0f} a proportional rate predicts.",
            f"   A fixed per-session cost of ~{actual} frames. Yield improves with",
            "   length, and a fitted pairing offset stays valid across the run.",
            f"   Projected: {1 - actual / 480:.0%} at 480 frames, "
            f"{1 - actual / 960:.0%} at 960.",
        ]
    else:
        rate = actual / large["sent"]
        out += [
            f"C. PROPORTIONAL LOSS — {large['sent']} frames lost {actual}, close to "
            f"the {predicted:.0f} predicted.",
            f"   Loss rate ~{rate:.0%}. Longer runs will not help and a fixed",
            "   pairing offset will drift as the stream goes on.",
        ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts", type=int, nargs="+", default=[24, 48, 96])
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--drain", type=float, default=150.0, help="max seconds to wait after the last push")
    ap.add_argument("--quiet", type=float, default=20.0, help="seconds of silence that count as done")
    ap.add_argument("--out", type=Path, default=Path("out/yield.json"))
    ap.add_argument("--from-json", type=Path, help="re-diagnose a saved run, no streaming")
    args = ap.parse_args()

    if args.from_json:
        rows = json.loads(args.from_json.read_text())
        print("  diagnosis:")
        for line in diagnose(rows):
            print(f"    {line}")
        return

    if not os.environ.get("REACTOR_API_KEY"):
        sys.exit("set REACTOR_API_KEY")

    rows = []
    for n in args.counts:
        print(f"\n=== {n} frames ===")
        r = asyncio.run(run_once(synth(n), args.fps, args.drain, args.quiet))
        r["count"] = n
        rows.append(r)
        print(
            f"  sent {r['sent']}  received {r['received']}  missing {r['missing']}"
            f"  ({r['received'] / r['sent']:.0%})"
        )
        print(
            f"  quiet-before-deadline: {r['went_quiet']}   "
            f"first out {r['first_s']:+.1f}s / last out {r['tail_s']:+.1f}s after last push"
        )
        print(
            f"  output {r['out_fps']:.1f} fps   gap median {r['median_gap_ms']:.0f}ms "
            f"max {r['max_gap_ms']:.0f}ms"
        )

    print("\n" + "=" * 62)
    print(f"  {'sent':>6} {'recv':>6} {'missing':>8} {'yield':>7} {'quiet':>7} {'out fps':>8}")
    for r in rows:
        print(
            f"  {r['sent']:>6} {r['received']:>6} {r['missing']:>8} "
            f"{r['received'] / r['sent']:>6.0%} {str(r['went_quiet']):>7} {r['out_fps']:>8.1f}"
        )

    print("\n  diagnosis:")
    for line in diagnose(rows):
        print(f"    {line}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))
    print(f"\n  raw: {args.out}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("interrupted")
