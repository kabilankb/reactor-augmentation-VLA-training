#!/usr/bin/env python3
"""Push a local LeRobot dataset directory to a Hugging Face dataset repo.

Uploads at the repo ROOT, not nested under a subfolder -- LeRobot tooling
(LeRobotDataset, the Hub's built-in dataset viewer) expects meta/info.json at
the repo root. Nesting two datasets under one repo is exactly what broke the
viewer earlier in this project; this script always publishes one dataset to
its own repo root.

    python webapp/push_hf.py --dataset datasets/lerobot_fruit95 \
        --repo-id kabilanKB/reactor_x2_lerobot_fruit95
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True, help="local dataset dir")
    ap.add_argument("--repo-id", required=True, help="namespace/name")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    if not (args.dataset / "meta" / "info.json").exists():
        sys.exit(f"not a LeRobot dataset dir (no meta/info.json): {args.dataset}")

    api = HfApi()
    print(f"  creating/confirming repo {args.repo_id} (private={args.private})")
    api.create_repo(repo_id=args.repo_id, repo_type="dataset",
                     private=args.private, exist_ok=True)

    print(f"  uploading {args.dataset} -> {args.repo_id} (root)")
    url = api.upload_folder(
        folder_path=str(args.dataset),
        path_in_repo=".",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=f"Add {args.dataset.name} (reactor X2 augmented dataset)",
    )
    print(f"  done -> https://huggingface.co/datasets/{args.repo_id}")
    print(f"  {url}")


if __name__ == "__main__":
    main()
