#!/usr/bin/env python3
"""Merge LeRobot fruit datasets, dropping episodes by scene id.

Building ten scenes in one run means one bad scene either poisons the set or
forces a full (paid) rebuild of the nine good ones. This lets a scene be rerun
on its own and spliced in, so a rebuild costs one API session instead of ten.

Episode files are copied, never re-encoded -- re-encoding h264 that was already
encoded from the model's output would add a second generation of loss to frames
that are the dataset. Only `episode_index` and the global `index` are rewritten,
because both must stay contiguous or downstream loaders mis-slice the episodes.

    python merge_fruit_datasets.py --base datasets/lerobot_fruit10x \
        --add datasets/_fruit_extra --drop lemon --out datasets/lerobot_fruit10x
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def read_ds(root: Path) -> dict:
    m = root / "meta"
    aug = [json.loads(l) for l in (m / "augmentations.jsonl").read_text().splitlines() if l.strip()]
    stats = [json.loads(l) for l in (m / "episodes_stats.jsonl").read_text().splitlines() if l.strip()]
    eps = [json.loads(l) for l in (m / "episodes.jsonl").read_text().splitlines() if l.strip()]
    return {"root": root, "info": json.loads((m / "info.json").read_text()),
            "aug": {a["episode_index"]: a for a in aug},
            "stats": {s["episode_index"]: s for s in stats},
            "eps": {e["episode_index"]: e for e in eps}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--add", type=Path, nargs="*", default=[])
    ap.add_argument("--drop", nargs="*", default=[],
                    help="scene ids to exclude; 'lemon' drops it from every source, "
                         "'lerobot_fruit10x:lemon' drops it from that source only")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sources = [read_ds(args.base)] + [read_ds(p) for p in args.add]
    video_key = next(k for k in sources[0]["info"]["features"] if k.startswith("observation.images"))

    picked = []
    for ds in sources:
        for ei in sorted(ds["aug"]):
            sid = ds["aug"][ei]["scene_id"]
            # Unqualified id drops the scene everywhere; "dataset:id" drops it
            # from one source, which is what rerunning a single bad scene needs.
            if sid in args.drop or f"{ds['root'].name}:{sid}" in args.drop:
                print(f"  drop {sid} (from {ds['root'].name})")
                continue
            if any(p["scene"] == sid for p in picked):
                print(f"  skip duplicate {sid} (from {ds['root'].name})")
                continue
            picked.append({"scene": sid, "ds": ds, "ei": ei})

    if not picked:
        sys.exit("nothing to write")

    # Stage in a temp dir so --out may safely equal --base.
    tmp = Path(tempfile.mkdtemp(dir=args.out.parent))
    (tmp / "data" / "chunk-000").mkdir(parents=True)
    (tmp / "videos" / "chunk-000" / video_key).mkdir(parents=True)
    (tmp / "meta").mkdir(parents=True)

    ep_lines, stat_lines, aug_lines = [], [], []
    gi = 0
    for new_ei, p in enumerate(picked):
        ds, old = p["ds"], p["ei"]
        df = pd.read_parquet(ds["root"] / "data" / "chunk-000" / f"episode_{old:06d}.parquet")
        n = len(df)
        df["episode_index"] = np.full(n, new_ei, dtype=np.int64)
        df["index"] = np.arange(gi, gi + n, dtype=np.int64)
        gi += n
        df.to_parquet(tmp / "data" / "chunk-000" / f"episode_{new_ei:06d}.parquet", index=False)
        shutil.copy(ds["root"] / "videos" / "chunk-000" / video_key / f"episode_{old:06d}.mp4",
                    tmp / "videos" / "chunk-000" / video_key / f"episode_{new_ei:06d}.mp4")

        e = dict(ds["eps"][old]); e["episode_index"] = new_ei; ep_lines.append(e)
        s = dict(ds["stats"][old]); s["episode_index"] = new_ei; stat_lines.append(s)
        a = dict(ds["aug"][old]); a["episode_index"] = new_ei
        a["merged_from"] = str(ds["root"]); aug_lines.append(a)
        print(f"  ep {new_ei:<3} {p['scene']:<12} {n:>4} frames   <- {ds['root'].name} ep{old}")

    info = dict(sources[0]["info"])
    info.update({"total_episodes": len(picked), "total_frames": gi, "total_videos": len(picked),
                 "splits": {"train": f"0:{len(picked)}"}})
    m = tmp / "meta"
    (m / "info.json").write_text(json.dumps(info, indent=4))
    shutil.copy(args.base / "meta" / "tasks.jsonl", m / "tasks.jsonl")
    (m / "episodes.jsonl").write_text("\n".join(json.dumps(x) for x in ep_lines) + "\n")
    (m / "episodes_stats.jsonl").write_text("\n".join(json.dumps(x) for x in stat_lines) + "\n")
    (m / "augmentations.jsonl").write_text("\n".join(json.dumps(x) for x in aug_lines) + "\n")
    for extra in ("scenes.json", "dropped.json"):
        if (args.base / "meta" / extra).exists():
            shutil.copy(args.base / "meta" / extra, m / extra)

    if args.out.exists():
        shutil.rmtree(args.out)
    tmp.rename(args.out)
    print(f"\n  wrote {args.out}: {len(picked)} episodes, {gi} frames")


if __name__ == "__main__":
    main()
