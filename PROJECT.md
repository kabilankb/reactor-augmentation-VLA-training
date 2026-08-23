# Reactor X2 as an augmentation stage for robot-policy training data

Can NVIDIA-style sim-to-real augmentation be done through Reactor's XMAX X2
video-editing model, to post-train a VLA or diffusion policy on Isaac Lab data?

**Answer: yes for appearance, conditionally for adding objects, no for
replacing them.** Validated on two real robot datasets across 40+ live API
sessions. What remains unproven is whether the augmented data improves a
policy — that needs an A/B training run.

| Use | Verdict | Evidence |
|---|---|---|
| Restyle appearance (light, materials, mood) | **works** | 0.9px drift, ΔE 34.9 |
| Add distractor objects | **works** | 10 fruits, gated per-frame on target survival |
| Replace the target object | **fails** | 10 of 10 attempts (§3, §3a) |
| Produce a training corpus | **works** | `datasets/lerobot_fruit10x`, loads in LeRobot |
| Improve a policy | **unproven** | no A/B run yet |

---

## The constraint that shapes everything

A VLA trains on `(observation, instruction, action)` triplets. X2 returns
pixels and nothing else — no actions, no proprioception, no camera pose. So it
**cannot be a data source**, only an augmentation layer over data that already
carries actions.

```
Isaac Lab / LeRobot          Reactor X2                Post-training
───────────────────          ──────────                ─────────────
 episode frames      ──►   restyle / augment   ──►   augmented frames
 + ACTIONS                 (pixels only)             + ORIGINAL actions
      │                                                     │
      └────────── actions never leave the machine ──────────┘
```

Every measurement below answers one question: **do the original action labels
still describe the augmented frames?** They do only if the manipulator and the
target land on the same pixels.

---

## 1. Appearance restyling — works

Local drift on real GR1 renders: **0.9px (0.12% of diagonal)** with 237 ORB
matches per frame. The edge overlay is almost entirely white across the
microwave, door, control panel, display digits, and robot hand.

The metric had to be split in two to see this. Raw displacement conflates
harmless global reframing (X2 picks its own output resolution) with local drift
that actually corrupts labels. On synthetic frames raw read 7.2px — apparent
failure — while the local residual was 1.0px.

### Prompt strength is a threshold, not a trade-off

Six escalating prompts, 48 real GR1 frames:

| rung | ΔE | drift % | ORB matches | verdict |
|---|---:|---:|---:|---|
| 1-minimal | 3.2 | 0.13% | 224 | no change |
| 2-mild | 3.0 | 0.16% | 164 | no change |
| 3-moderate | 3.4 | 0.09% | 225 | no change |
| 4-strong | 7.0 | 0.17% | 146 | usable |
| **5-heavy** | **34.9** | **0.21%** | 38 | **best** |
| 6-extreme | 73.2 | n/a | 0 | unmeasurable |

Appearance scales ~10x while drift barely doubles, never approaching the 0.4%
budget. An intermediate conclusion that X2 "preserves geometry but barely
changes anything" was **wrong** — that test used a rung-3 prompt, inside a dead
zone where ΔE ≈ 3 is imperceptible.

ΔE is driven by **distance from the source scene**, not verbosity: rung 4 is the
longest prompt and reaches only 7.0.

The cliff at rung 6 is semantic, not gradual — "watercolour painting" turned
the microwave into a pencil-drawn human face.

---

## 2. Adding objects — conditional on placement wording

Adding fruit to the pick-orange scene. All three variants passed the coverage
gate; only one was correct:

| variant | coverage | ΔE | what actually happened |
|---|---:|---:|---|
| "beside the oranges" | 0.851 | 11.0 | **oranges replaced by apples** |
| "among the oranges" | 0.873 | 15.1 | **grapes occluded an orange** |
| **"at the back"** | 0.861 | 9.4 | **all 3 oranges intact, fruit added** |

**Name a distinct empty region.** Phrasing that relates new objects to existing
ones ("beside", "among") invites the model to reinterpret those objects instead
of adding alongside them.

Adding also competes with restyling. Asked for food at rung-5 styling, X2
painted plates of food **over the robot's hand**; coverage fell to 0.155. Five
variants, only the one that dropped the styling budget worked:

| variant | coverage | result |
|---|---:|---|
| rung-5 + food | 0.357 | erases the hand |
| + "inside the microwave" | 0.278 | worse |
| + "hand unchanged" | 0.547 | helps, insufficient |
| **food only, minimal styling** | **0.89** | **works** |

**You get heavy restyling or new objects, not both.**

---

## 3. Replacing the target object — fails

Goal: swap the orange for other fruits and rewrite the instruction, building a
multi-fruit pick dataset. Sound in principle; the recorded grasp stays valid for
a substitute of similar position and size.

| fruit | coverage | what actually happened |
|---|---:|---|
| apple | 0.856 | **nothing changed** |
| lemon | 0.687 | **gripper became two yellow lemons** |
| lime | 0.768 | **gripper turned green** |
| green apple | 0.756 | **arm became a green apple** |

X2 applied the substitution to the **yellow robot arm** — the largest salient
object — rather than three small oranges. Every metric passed it.

Output preserved as `datasets/fruit_pick_INVALID_arm_destroyed.hdf5`,
deliberately named so it cannot be mistaken for training data.

**Substitution belongs in the simulator.** Swapping a USD asset guarantees the
right object changes, the arm cannot be transformed (separate prim), occlusion
and shading are correct, masks come free, and actions stay exactly valid.
Generative editing suits *appearance*; it has no notion of which pixels are
which object.

---

## 3a. Substitution, retried with a better prompt and a real gate — still fails

Section 3 blamed the prompt: "replace every orange with a lemon" binds to the
yellow arm because *orange* names a colour and the arm is the largest saturated
object. That diagnosis was right, and fixing it fixed the destruction — but not
the task.

Six further runs (`out_subst_pilot/`, `out_subst_probe/`), binding the edit by
**count, position and size** and pinning the arm explicitly:

> the three small round fruits on the wooden table in front of the white plate
> are {descriptor}, same size and same position, the yellow robot arm is
> unchanged, the white plate and the wooden table are unchanged

| prompt family | fruit ΔE | bg ΔE | selectivity | outcome |
|---|---:|---:|---:|---|
| positional (apple / plum / peach) | 13.2 / 13.0 / 11.7 | ~9.6 | 1.43 / 1.41 / 1.44 | fruit **added**, oranges remain |
| "oranges removed, none remain" | 13.3 | 9.3 | 1.59 | fruit added, oranges remain |
| "there are no oranges anywhere" | 11.4 | 9.4 | 1.21 | nothing changed |
| "purple instead of orange, only colour differs" | 15.0 | 9.9 | 1.50 | fruit added, oranges remain |

**The arm now survives every run** — that part is solved, and the prompt is
worth keeping. But X2 *adds*; it does not delete-and-replace, and it ignores
negation outright: "no orange fruit remains anywhere" returns oranges plus
plums. Ten attempts, zero substitutions. The conclusion of §3 stands, for a
different reason than §3 gave.

### The identity metric this needed

Coverage is blind to identity, so the gate measures **where the edit landed**.
The three oranges separate from the wooden table (same hue, V 118 vs 172) and
from the arm (S 119 vs 183) on saturation and value together, so a source-side
colour mask finds them with no simulator masks:

    fruit_dE / bg_dE  — high for a real substitution, ~1.0 for an edit that
                        moved the fruit and the background equally

The three quarantined failures in `fruit_pick_INVALID_arm_destroyed.hdf5` score
1.37 / 1.39 / 1.38. Every new attempt scores 1.21-1.59. Nothing has ever scored
above 2.

---

## 3b. The addition gate — and what it found in the shipped dataset

Adding works, but the failure it *can* have is the one every metric here was
blind to: added fruit growing until it covers the pick target. Re-scored with a
per-orange check, `datasets/lerobot_fruit10` (§the deliverable, as shipped)
does not hold up:

| ep | scene | clean until | what happens |
|---|---|---:|---|
| 2 | berries | 89/89 | fine |
| 0 | apple_lemon | 63/90 | an orange becomes a peach-like fruit |
| 4 | grapes_pear | 34/90 | grapes grow over two of the three oranges |
| 1 | tropical | 3/90 | giant mangoes dominate the scene |

Episode 4 measured, frame by frame: orange ΔE 17.1 → 13.6 → **74.2** → **78.7**.
The actions still say "reach to the orange at this position"; by frame 55 the
pixels there are grapes. Coverage passed it at 0.871.

Two properties are required to see this:

- **Per frame, not per episode.** The corruption is progressive, so an episode
  median averages the ruined tail against a clean head.
- **Per orange, not pooled.** Pooling all three into one mask lets two
  survivors outvote one that is buried. An early build read 15.6 — a clean pass
  — on a frame where bananas covered an orange.

The response is to **truncate**, not to drop: cutting the tail keeps the good
head and cannot create a splice, because dropping a suffix leaves the remaining
frames contiguous.

### Placement is the whole game

"At the back" is where the *arm* is, so fruit sent there lands on the oranges.
Moving the placement to the empty foreground fixed almost everything:

| placement | scenes kept | scenes needing truncation |
|---|---:|---:|
| "on the table at the back" | 9/10 | 5 (one dropped outright) |
| "empty table in the foreground, near the front edge" | 9/10 | 1 |

**Name a region that is actually empty, not merely distant.**

---

## The deliverable

`datasets/lerobot_fruit10x/` — 10 episodes, one fruit each (apple, banana,
pear, plum, strawberry, grape, pineapple, peach, kiwi, lemon), 899 frames,
LeRobot v2.1. Converts and loads with `lerobot` 0.4.3.

The instruction stays `"Grab orange and place into plate"` in every episode,
because the orange remains the target and the recorded grasp goes to it.
Substitution would have justified rewriting it; addition does not. The task does
get harder: with a pear and a pineapple also in frame, "orange" has to pick one
object out of several, which the single-fruit source never asked.

Every episode carries its prompt, coverage, per-frame target ΔE, truncation
count and source frame range in `meta/augmentations.jsonl`.

---

## 4. Frame yield saturates near 90%

| sent | received | missing | yield |
|---:|---:|---:|---:|
| 24 | 18 | 6 | 75% |
| 48 | 38 | 10 | 79% |
| 96 | 86 | 10 | 90% |
| 240 | 215 | 25 | 90% |

Least-squares: `missing = 4.1 + 0.085 * N` (R² = 0.96) — a fixed ~4-frame
warmup plus a sustained ~8.5% shortfall. **Yield does not climb toward 100%.**
An early reading of the first three points suggested constant loss; the
240-frame run showed that was coincidence.

Push rate does not help (24fps→90%, 20fps→94%, 16fps→85%, non-monotonic).
**Run-to-run variance is ~±9 percentage points** — no single measurement here
should be trusted tightly.

---

## 5. Pairing needs a fitted offset, scored on coverage

X2 **does not echo `user_data`**, the SDK's designed correspondence channel, and
`frame_id`/`timestamp_us` are explicitly not preserved. Pairing is therefore
arrival order plus a fitted offset.

The fit must score on **coverage, not residual drift**. Once an edit adds
content, residual goes flat across every candidate offset — added pixels match
nothing at any `k`. Measured on a fruit run: residual spanned 0.90-1.02px across
k=0..8 (useless) while coverage ran 0.64-0.84 and peaked sharply at the true
offset. This bug made one run report coverage 0.68 instead of its true 0.839.

Probe frames also skip the opening of the episode — a near-black first frame
makes every offset score alike and the fit lands on 0 by default.

---

## 6. Batch at ~96 frames

| | single 774-frame run | 4 x 96-frame batches |
|---|---:|---:|
| coverage | 0.63 | **0.833-0.851, flat** |
| temporal drift | 1.3px → 2.7px | flat |

One offset cannot hold across a long stream when ~8.5% of frames drop
throughout.

Within a batch, quality still varies: coverage by position ran 0.76/0.85/0.85
for batch 0 but 0.83/0.84/**0.67** for batch 2 — fruit enlarging and multiplying
in later frames. 48-frame batches would likely tighten this.

Per-frame stability is fine: frame-to-frame pixel delta is 0.6x the source at
the median and 1.4x at p95. No flicker; the degradation is slow drift.

---

## The gate, and its blind spots

Two metrics, checked in order:

**`structure_coverage`** — for each source edge pixel, is there an edited edge
within 4px? Deleted structure drives it down. Hard NO-GO below 0.60. This exists
because drift measures how far *matched* features moved and is blind to features
that **vanish**: on the run that erased the robot hand, drift read 0.21% (a
clean pass) while coverage read 0.155.

**`residual_drift`** — local displacement after fitting out global reframing.
Budget 0.4% of the diagonal.

| run | coverage | drift | correct verdict |
|---|---:|---:|---|
| clean restyle | 0.912 | 0.89px | pass |
| food added | 0.89 | 0.90px | pass |
| rung-5 + food | 0.155 | 1.51px | **NO-GO** |

### Known blind spots

- **Identity** — an orange swapped for a similarly-sized apple preserves edge
  structure and scores fine. This is what let the arm-destroying substitutions
  through. Needs a per-object check; sim masks would supply it.
- **Occlusion** — nothing detects a new object covering a target.
- **Slow drift** — per-frame metrics see no flicker while content gradually
  enlarges across a batch.
- **Confidence collapses where it matters** — ORB matches fall 225 → 146 → 38 →
  0 as prompts strengthen. The number you would act on is the least certain one.
  **Always inspect the comparison images.**

---

## Prompt cookbook

**Restyle (rung 5)** — coverage 0.91, ΔE 34.9

> dim moody blue evening light, wet reflective surfaces, heavy film grain,
> rusted industrial metal, dark cluttered background

**Add objects** — coverage 0.86

> a red apple and a yellow lemon on the wooden table beside the oranges, a small
> wooden bowl of mixed fruit at the back, everything else unchanged

**Never**

```
"a robot arm opening a microwave"       redraws the arm
"...beside / among the oranges"         reinterprets the oranges
"replace every orange with a lemon"     transforms the arm instead
"watercolour painting"                  semantic replacement
```

Rule: name **lighting, materials, atmosphere** freely; name **objects** only
with a distinct empty location; never name the robot or the manipulated object.

---

## Files

| Path | Purpose |
|---|---|
| `validate_x2.py` | The gate: coverage, drift, yield, offset fitting |
| `build_fruit_addition_dataset.py` | **10-fruit dataset; per-frame, per-orange target gate + truncation** |
| `build_substitution_dataset.py` | Substitution + selectivity gate (**model fails the task**; gate is reusable) |
| `merge_fruit_datasets.py` | Splice a rerun scene into an existing set without rebuilding all ten |
| `prompt_sweep.py` | ΔE vs drift across escalating prompt strength |
| `probe_yield.py` | Frame-yield model across stream lengths |
| `food_variants.py` | Score candidate prompts on coverage (`--variants` JSON) |
| `lerobot_to_episode.py` | LeRobot mp4+parquet → frames + manifest (ffmpeg AV1 fallback) |
| `augment_lerobot.py` | Batched augmentation → LeRobot v2.1 dataset |
| `make_fruit_dataset.py` | Fruit substitution → robomimic HDF5 (**model fails the task**) |
| `capture_episodes.py` | Isaac Lab replay → frames + masks + manifest (**unrun**) |
| `.env` | API key, mode 600, gitignored |

## Outputs

| Path | Contents |
|---|---|
| **`datasets/lerobot_fruit10x/`** | **10 fruits, 899 frames — the deliverable** |
| `datasets/lerobot_fruit10/` | Superseded: eps 0/1/4 lose the target mid-episode (§3b) |
| `datasets/lerobot_fruit/` | Superseded: 359 frames, 3 batch-boundary splices, no `episodes_stats.jsonl` |
| `datasets/fruit_pick_INVALID_arm_destroyed.hdf5` | Failed substitution, quarantined |
| `episodes/orange_ep0/`, `episodes/gr1_ep0/` | Converted source episodes |
| `out_gr1_food/edited/` | 86 GR1 frames, food added |
| `out_gr1_fruit/edited/` | 86 GR1 frames, fruit added |
| `out_sweep/`, `out_food/`, `out_fruit*/`, `out_orange/` | Variant comparisons |
| `out_orange_full/` | 774-frame run — evidence of drift, not usable data |
| `out/` | Synthetic validation + yield probe data |

`out_orange_full/` is 746 MB and can be deleted; it exists only as evidence.

---

## Running it

```bash
pip install reactor-sdk numpy opencv-python pandas h5py
set -a && . ./.env && set +a          # REACTOR_API_KEY=rk_...

# 1. convert a LeRobot episode (no Isaac Sim needed)
python lerobot_to_episode.py --video datasets/orange/ep0_front.mp4 \
    --parquet datasets/orange/ep0.parquet --out episodes/orange_ep0

# 2. find a prompt that survives the gate
python food_variants.py --frames-dir episodes/orange_ep0/rgb \
    --count 96 --variants my_prompts.json --out out_try

# 3. generate the dataset in batches
python augment_lerobot.py --episode episodes/orange_ep0 \
    --prompt "<the winner>" --task "Grab orange and place into plate" \
    --out datasets/lerobot_fruit --batch 96
```

X2 bills **$0.0017/sec of session wall-time**, not per frame — drain windows
dominate. Roughly $0.50 for a 50-episode pass once parameters are known.

---

## Datasets

**`nvidia/Arena-GR1-Manipulation-Task`** — GR1 humanoid, IsaacLab-Arena,
*"Reach out to the microwave and open it."* 50Hz, 512x512 ego-view, 10 teleop +
50 MimicGen demos, CC-BY-4.0. Episode 0 has **26-dim** actions; the dataset
README says 36. Scene contains exactly two objects — `articulation/microwave`
and `articulation/robot`. **There is no fruit in it**; the pale disc is the
turntable. `gr1_open_microwave` needs IsaacLab-Arena, present at
`~/IsaacLab/IsaacLab-Arena/` but not pip-installed (imports only from its own
directory, wants Python 3.10 while the venv is 3.11).

**`LightwheelAI/leisaac-pick-orange`** — SO-101 arm (6-DoF), 60 episodes, 36,293
frames @30fps, two camera views, Apache-2.0. *"Grab orange and place into
plate."* Scene: white plate, three oranges, wooden table. Videos are **AV1**,
which OpenCV cannot decode — it returns zero frames rather than raising, so
`lerobot_to_episode.py` falls back to ffmpeg/libdav1d on an empty decode.

---

## What is not done

- **No policy has been trained.** The mechanics are proven; the value is not.
  The decisive experiment is an A/B — train on raw vs raw+augmented, evaluate
  under held-out appearance with `robomimic/robust_eval.py`.
- **`capture_episodes.py` has never run.** APIs verified against Isaac Lab
  2.3.0 and pure helpers unit-tested, but Isaac Sim was never launched.
- **One episode per dataset**, one camera, 384 of 774 frames augmented.
- **Substitution unsolved** by this route; the sim path is the recommendation.
- **No identity or occlusion metric** in the gate.

---

## Security

The API key used here was pasted in plaintext into a chat transcript and should
be rotated at https://www.reactor.inc/account/api-keys.
