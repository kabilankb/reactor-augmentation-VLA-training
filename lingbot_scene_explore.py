#!/usr/bin/env python3
"""Preview multi-fruit table scenes with LingBot World 2, before spending X2
budget on the gated dataset build.

WHY THIS SCRIPT EXISTS SEPARATELY FROM build_fruit_addition_dataset.py

LingBot World 2 cannot build the dataset itself. It takes one seed image plus
a prompt and then *generates* every following frame from scratch, steered by
camera movement -- there is no way to hand it an existing episode and have it
preserve that episode's actual arm motion. So it cannot produce frames that
carry valid `action`/`state` labels the way X2's per-frame edits can (X2 edits
real recorded frames in place; the arm motion in the pixels is still the arm
motion that produced the logged actions).

What it IS good for: iterating on multi-fruit *prompt wording* fast. Seed it
once with a real frame from the source episode, hold the camera idle (no
movement/look commands are sent, so nothing walks or turns), and hot-swap
`set_prompt` across candidate combos -- each swap steers the next chunk
without a full reconnect. Skim the previews, keep the wording that reads as
a clean multi-fruit arrangement, drop it into fruit_scenes_multi.json, then
run the real, gated pipeline:

    python build_fruit_addition_dataset.py --episode episodes/orange_ep0 \
        --scenes fruit_scenes_multi.json --count 96 --out datasets/lerobot_fruit_multi

Usage:
    python lingbot_scene_explore.py --seed-frame episodes/orange_ep0/rgb/000000.png \
        --scenes fruit_scenes_multi.json --out out_lingbot_multi

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

MODEL = "reactor/lingbot-world-2"
OUTPUT_TRACK = "main_video"


async def explore(seed_frame: Path, scenes: list[dict], out: Path,
                   settle_s: float, seed: int) -> None:
    api_key = os.environ.get("REACTOR_API_KEY")
    if not api_key:
        sys.exit("set REACTOR_API_KEY")

    reactor = Reactor(model_name=MODEL, api_key=api_key)
    latest: dict[str, np.ndarray] = {}
    ready = asyncio.Event()

    @reactor.on_status
    def _status(new: str) -> None:
        print(f"  status: {new}")

    @reactor.on_error
    def _error(exc: Exception) -> None:
        print(f"  error: {exc}")

    @reactor.on_message
    def _message(msg: dict) -> None:
        kind = msg.get("type") or msg.get("event") or "message"
        if kind == "conditions_ready":
            ready.set()
        if kind in {"conditions_ready", "generation_started", "command_error"}:
            print(f"  event: {json.dumps(msg)[:300]}")

    async with reactor:
        await reactor.connect()
        print(f"  session: {reactor.session_id}")

        output = reactor.track(OUTPUT_TRACK)

        @output.on_frame
        def _collect(frame, frame_id, timestamp_us, user_data):  # noqa: ANN001
            latest["frame"] = frame.copy()

        file_ref = await reactor.upload_file(seed_frame)
        await reactor.send_command("set_seed", {"seed": seed})
        await reactor.send_command("set_image", {"image": file_ref})
        await reactor.send_command("set_prompt", {"prompt": scenes[0]["prompt"]})

        await asyncio.wait_for(ready.wait(), timeout=30.0)
        await reactor.send_command("start", {})

        out.mkdir(parents=True, exist_ok=True)
        for i, sc in enumerate(scenes):
            if i > 0:
                # Hot-swap: applies at the next chunk boundary, no reset needed.
                # Camera axes are never touched, so the viewpoint stays fixed.
                await reactor.send_command("set_prompt", {"prompt": sc["prompt"]})
            print(f"=== {sc['id']}  (settling {settle_s:.0f}s)")
            await asyncio.sleep(settle_s)
            if "frame" not in latest:
                print("  no frame received yet — skipping")
                continue
            cv2.imwrite(str(out / f"{sc['id']}.png"),
                        cv2.cvtColor(latest["frame"], cv2.COLOR_RGB2BGR))
            print(f"  saved {out}/{sc['id']}.png")

        await reactor.send_command("reset", {})

    print(f"\n  previews: {out}/<scene-id>.png")
    print("  pick the wording that reads clean, keep it in fruit_scenes_multi.json,")
    print("  then run the real gated build via build_fruit_addition_dataset.py.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed-frame", type=Path, required=True,
                    help="one real frame from the source episode, e.g. episodes/orange_ep0/rgb/000000.png")
    ap.add_argument("--scenes", type=Path, default=Path("fruit_scenes_multi.json"))
    ap.add_argument("--out", type=Path, default=Path("out_lingbot_multi"))
    ap.add_argument("--settle", type=float, default=8.0,
                    help="seconds to let each prompt swap settle before capturing")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    scenes = json.loads(args.scenes.read_text())
    asyncio.run(explore(args.seed_frame, scenes, args.out, args.settle, args.seed))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("interrupted")
