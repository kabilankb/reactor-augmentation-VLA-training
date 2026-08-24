# Reactor X2 → robot-policy data augmentation

https://github.com/user-attachments/assets/a4d6d8b6-27c7-4a9a-8e50-55c0dd22a3aa

Using Reactor's XMAX X2 video-editing model to augment robot demonstration data
for VLA / diffusion-policy post-training.

**→ [PROJECT.md](PROJECT.md) — findings, measurements, and prompt cookbook.**

## Status

| Use | Verdict |
|---|---|
| Restyle appearance | ✅ works — 0.9px drift at ΔE 34.9 |
| Add distractor objects | ✅ works — place in an *empty* region, gate per-frame |
| Replace the target object | ❌ fails — 10/10 attempts |
| Produce a training corpus | ✅ `datasets/lerobot_fruit10x/` — 10 fruits, 899 frames |
| Improve a policy | ❓ unproven — no A/B run yet |

## Datasets on Hugging Face

Each dataset is its own repo, with `meta/`, `data/`, `videos/` at the repo
root — the standard LeRobot layout, so `LeRobotDataset(repo_id=...)` and any
LeRobot viewer/visualizer works without extra path config:

- **[kabilanKB/reactor_x2_lerobot_fruit10x](https://huggingface.co/datasets/kabilanKB/reactor_x2_lerobot_fruit10x)** — 10 episodes, one distractor fruit per scene, 899 frames.
- **[kabilanKB/reactor_x2_lerobot_env50](https://huggingface.co/datasets/kabilanKB/reactor_x2_lerobot_env50)** — 40 episodes (of 50 attempted), fruit + lighting-style combos, 3506 frames. 10 scenes dropped on the per-frame target-survival gate — see `meta/dropped.json` in that repo.
- **[kabilanKB/reactor_x2_100](https://huggingface.co/datasets/kabilanKB/reactor_x2_100)** — 79 episodes (of 95 attempted), 9 distractor fruits x 5 lighting styles, 6818 frames. 16 scenes dropped on the per-frame target-survival gate — see `meta/dropped.json` in that repo.
- **[kabilanKB/reactor-x2-GR1-Manipulation-Task-v3](https://huggingface.co/datasets/kabilanKB/reactor-x2-GR1-Manipulation-Task-v3)** — 31 episodes (of 32 food items attempted), 2656 frames, built from [nvidia/Arena-GR1-Manipulation-Task-v3](https://huggingface.co/datasets/nvidia/Arena-GR1-Manipulation-Task-v3) (GR1 humanoid, "reach out to the microwave and open it") with one food item added to the plate/turntable per episode. 1 scene (`dhokla`) dropped on the coverage gate — see `meta/augmentations.jsonl` in that repo.

## Quickstart

```bash
pip install reactor-sdk numpy opencv-python pandas h5py
set -a && . ./.env && set +a          # REACTOR_API_KEY=rk_...

python lerobot_to_episode.py --video datasets/orange/ep0_front.mp4 \
    --parquet datasets/orange/ep0.parquet --out episodes/orange_ep0

python build_fruit_addition_dataset.py --episode episodes/orange_ep0 \
    --scenes fruit_scenes10.json --count 96 --out datasets/lerobot_fruit10x
```

## The four rules

1. **Name lighting and materials freely; name objects only with a distinct
   empty location.** "Beside the oranges" makes X2 reinterpret the oranges.
   The location must actually be *empty*: "at the back" is where the arm is,
   and fruit sent there lands on the oranges. The foreground works.
2. **Never name the robot or the manipulated object.** X2 carries through what
   you don't mention — that carry-through is what keeps the labels valid.
3. **Batch at ~96 frames.** A single long stream drifts (coverage 0.63 vs 0.84).
4. **Gate the target per frame, and per object.** Pooling the three oranges into
   one mask lets two survivors outvote one that has been painted over.

## Always look at the images

Every metric here has passed a run that was visibly broken. Coverage cannot see
identity changes; drift cannot see deletions; a per-episode median cannot see a
target that is buried only in the second half. `out_*/pair_*.png` and
`out_*/<variant>.png` are the check that actually works.
