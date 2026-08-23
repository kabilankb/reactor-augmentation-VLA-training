#!/usr/bin/env python3
"""Replay an action-labeled Isaac Lab dataset and capture aligned frames.

Produces the input side of the X2 augmentation pipeline: RGB frames on disk,
optional robot masks, and a manifest pairing every frame with the action that
was taken from it. Feed the frames to `validate_x2.py`, then re-pair the edited
frames with the actions here — the actions themselves never leave the sim.

The alignment convention is the one behavioural cloning expects: the frame is
captured BEFORE the action is applied, so `manifest[i]` is the (obs, action)
pair the policy is trained on. Getting this off by one silently teaches the
policy to act on the consequence of its own action.

Usage:
    # from an existing recorded / Mimic-generated dataset
    ./isaaclab.sh -p capture_episodes.py \
        --dataset_file datasets/dataset.hdf5 \
        --output_dir renders/ --headless

    # with robot masks, so the geometry check ignores the background
    ./isaaclab.sh -p capture_episodes.py \
        --dataset_file datasets/dataset.hdf5 \
        --output_dir renders/ --mask_pattern Robot --headless

Datasets come from `scripts/tools/record_demos.py` or
`scripts/imitation_learning/isaaclab_mimic/generate_dataset.py`.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Capture aligned frames + actions from an Isaac Lab dataset.")
parser.add_argument("--dataset_file", type=str, required=True, help="HDF5 dataset to replay.")
parser.add_argument("--output_dir", type=str, default="renders", help="Where to write frames and manifest.")
parser.add_argument("--task", type=str, default=None, help="Override the task name stored in the dataset.")
parser.add_argument("--select_episodes", type=int, nargs="+", default=[], help="Episode indices (default: all).")
parser.add_argument("--max_episodes", type=int, default=None, help="Stop after this many episodes.")
parser.add_argument("--width", type=int, default=640, help="Capture width.")
parser.add_argument("--height", type=int, default=480, help="Capture height.")
parser.add_argument(
    "--camera_pos", type=float, nargs=3, default=[1.4, 0.0, 0.9], help="Camera position in env-local frame."
)
parser.add_argument(
    "--camera_look", type=float, nargs=3, default=[0.0, 0.0, 0.3], help="Point the camera looks at."
)
parser.add_argument(
    "--mask_pattern",
    type=str,
    default=None,
    help="Substring of the prim path to mask (e.g. 'Robot'). Enables instance segmentation.",
)
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Use USD I/O instead of fabric.")
parser.add_argument("--enable_pinocchio", action="store_true", default=False, help="Needed by PINK IK / GR1T2 tasks.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Pinocchio must be imported before AppLauncher so Isaac Lab's build wins.
if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import gymnasium as gym
import json
import numpy as np
import os
import torch
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab.utils.math import quat_from_matrix

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

CAMERA_KEY = "capture_cam"


def look_at_quat(eye: np.ndarray, target: np.ndarray) -> tuple[float, float, float, float]:
    """World-convention orientation quaternion (w, x, y, z) for a look-at camera.

    USD cameras look down -Z with +Y up, which is what the "world" offset
    convention expects.
    """
    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-6:
        return (1.0, 0.0, 0.0, 0.0)
    forward = forward / norm

    world_up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(forward, world_up))) > 0.999:
        # Degenerate straight-down / straight-up view; pick any stable right.
        world_up = np.array([0.0, 1.0, 0.0])

    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    # Columns are the camera's x, y, z axes; -forward because USD looks down -Z.
    rot = np.stack([right, up, -forward], axis=1)
    quat = quat_from_matrix(torch.from_numpy(rot).float().unsqueeze(0))[0]
    return tuple(float(v) for v in quat)


def add_capture_camera(env_cfg) -> None:
    """Inject a capture camera into the task's scene config."""
    eye = np.asarray(args_cli.camera_pos, dtype=float)
    target = np.asarray(args_cli.camera_look, dtype=float)

    data_types = ["rgb"]
    if args_cli.mask_pattern:
        # instance_segmentation_fast gives per-prim ids without requiring the
        # assets to carry semantic labels, which most Isaac Lab tasks do not.
        data_types.append("instance_segmentation_fast")

    setattr(
        env_cfg.scene,
        CAMERA_KEY,
        TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/" + CAMERA_KEY,
            offset=TiledCameraCfg.OffsetCfg(
                pos=tuple(float(v) for v in eye),
                rot=look_at_quat(eye, target),
                convention="world",
            ),
            data_types=data_types,
            colorize_instance_segmentation=False,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 20.0),
            ),
            width=args_cli.width,
            height=args_cli.height,
            update_period=0.0,  # refresh every step
        ),
    )


def to_uint8_rgb(tensor: torch.Tensor) -> np.ndarray:
    """(H, W, C) tensor from the camera → (H, W, 3) uint8 RGB.

    Tolerates the float-normalised and RGBA variants different Isaac Sim
    versions hand back, so the pipeline does not silently write black frames.
    """
    arr = tensor.detach().cpu().numpy()
    if arr.dtype != np.uint8:
        hi = float(arr.max()) if arr.size else 0.0
        arr = (arr * 255.0 if hi <= 1.0 else arr).clip(0, 255).astype(np.uint8)
    return arr[..., :3]


def robot_mask(seg: torch.Tensor, info: dict, pattern: str) -> np.ndarray | None:
    """Binary mask of every prim whose path contains `pattern`."""
    id_to_labels = (info or {}).get("idToLabels")
    if not id_to_labels:
        return None

    wanted = set()
    for key, label in id_to_labels.items():
        path = label if isinstance(label, str) else str(label)
        if pattern.lower() in path.lower():
            with contextlib.suppress(ValueError, TypeError):
                wanted.add(int(key))
    if not wanted:
        return None

    ids = seg.detach().cpu().numpy()
    if ids.ndim == 3:
        ids = ids[..., 0]
    return (np.isin(ids, list(wanted)).astype(np.uint8) * 255)


def main() -> None:
    if not os.path.exists(args_cli.dataset_file):
        raise FileNotFoundError(f"dataset not found: {args_cli.dataset_file}")

    handler = HDF5DatasetFileHandler()
    handler.open(args_cli.dataset_file)
    env_name = args_cli.task.split(":")[-1] if args_cli.task else handler.get_env_name()
    if env_name is None:
        raise ValueError("task name not given and not stored in the dataset")

    episode_names = list(handler.get_episode_names())
    episode_count = handler.get_num_episodes()
    if episode_count == 0:
        raise SystemExit("dataset contains no episodes")

    indices = args_cli.select_episodes or list(range(episode_count))
    if args_cli.max_episodes:
        indices = indices[: args_cli.max_episodes]

    out = Path(args_cli.output_dir)
    rgb_dir = out / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = out / "seg"
    if args_cli.mask_pattern:
        mask_dir.mkdir(parents=True, exist_ok=True)

    # One env: replay is deterministic and the manifest stays unambiguous.
    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    env_cfg.recorders = {}
    env_cfg.terminations = {}
    add_capture_camera(env_cfg)

    env = gym.make(env_name, cfg=env_cfg).unwrapped
    camera = env.scene[CAMERA_KEY]
    dt = env.step_dt if hasattr(env, "step_dt") else env.physics_dt

    # Imported late so a missing OpenCV fails after the config work, not before.
    import cv2

    env.reset()
    manifest: list[dict] = []
    seq = 0

    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        for ep_pos, ep_index in enumerate(indices):
            if ep_index >= episode_count:
                print(f"skipping out-of-range episode {ep_index}")
                continue

            episode = handler.load_episode(episode_names[ep_index], env.device)
            initial_state = episode.get_initial_state()
            env.reset_to(initial_state, torch.tensor([0], device=env.device), is_relative=True)

            step = 0
            ep_start = seq
            while True:
                action = episode.get_next_action()
                if action is None:
                    break

                # Capture BEFORE stepping: this frame is the observation the
                # action was chosen from.
                camera.update(dt, force_recompute=True)
                rgb = to_uint8_rgb(camera.data.output["rgb"][0])
                name = f"f{seq:06d}.png"
                cv2.imwrite(str(rgb_dir / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

                mask_name = None
                if args_cli.mask_pattern:
                    mask = robot_mask(
                        camera.data.output["instance_segmentation_fast"][0],
                        camera.data.info[0].get("instance_segmentation_fast", {}),
                        args_cli.mask_pattern,
                    )
                    if mask is not None:
                        mask_name = name
                        cv2.imwrite(str(mask_dir / mask_name), mask)

                manifest.append({
                    "seq": seq,
                    "frame": f"rgb/{name}",
                    "mask": f"seg/{mask_name}" if mask_name else None,
                    "episode": int(ep_index),
                    "step": step,
                    "action": [float(v) for v in action.flatten().tolist()],
                })

                env.step(action.unsqueeze(0) if action.ndim == 1 else action)
                seq += 1
                step += 1

            print(f"[{ep_pos + 1}/{len(indices)}] episode {ep_index}: {step} steps  (seq {ep_start}..{seq - 1})")

    manifest_path = out / "manifest.jsonl"
    with manifest_path.open("w") as fh:
        for row in manifest:
            fh.write(json.dumps(row) + "\n")

    masked = sum(1 for r in manifest if r["mask"])
    print()
    print(f"  frames:   {seq} -> {rgb_dir}")
    if args_cli.mask_pattern:
        print(f"  masks:    {masked} -> {mask_dir}")
        if masked == 0:
            print(f"    no prim path matched {args_cli.mask_pattern!r} — check the scene's asset names")
    print(f"  manifest: {manifest_path}")
    print()
    print("  next:")
    print(f"    python validate_x2.py --frames-dir {rgb_dir}", end="")
    print(f" --mask-dir {mask_dir}" if args_cli.mask_pattern else "")
    print("  `seq` in the manifest matches the tag validate_x2.py sends, so an")
    print("  edited frame re-pairs to its action by that field alone.")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
