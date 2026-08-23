#!/usr/bin/env python3
"""Go/no-go validation for using XMAX X2 as an appearance-augmentation stage
for VLA / diffusion-policy training data.

Streams rendered frames (Isaac Lab / Replicator output) into X2's `source`
track, collects the edited frames off `main_video`, and answers the three
questions the pipeline depends on:

  1. TAG ECHO      Does X2 mirror `user_data` back? That tag is the only thing
                   that survives the round trip (`frame_id` / `timestamp_us`
                   explicitly do not), so it is the only sound way to re-pair an
                   edited frame with the action label of the frame it came from.
  2. FRAME COUNT   Under `keep_backlog=true` the model consumes every source
                   frame in order. Does the count actually come back 1:1?
  3. GEOMETRY      How far does structure move between source and edit? Action
                   labels stay valid only while the manipulator lands on the
                   same pixels. Drift here silently corrupts training data.

Usage:
    export REACTOR_API_KEY=rk_...
    python validate_x2.py --frames-dir renders/episode_000 --count 96

    # Restrict the geometry metric to the robot, using Replicator masks:
    python validate_x2.py --frames-dir renders/rgb --mask-dir renders/seg

Deps: reactor-sdk, numpy, opencv-python
Docs: https://docs.reactor.inc/model-api-reference/x2/schema
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from reactor_sdk import Reactor

MODEL = "xmax/x2"
SOURCE_TRACK = "source"
OUTPUT_TRACK = "main_video"

# X2's native source pacing. Pushing faster only grows the backlog.
NATIVE_FPS = 24.0

# An edit that names only lighting and materials. It must never mention the
# robot, the gripper, or the manipulated object: X2 carries through what you
# don't mention, and that carry-through is what keeps the action labels valid.
DEFAULT_PROMPT = (
    "warm late-afternoon sunlight, worn scuffed tabletop, "
    "cluttered background shelves, dust in the air"
)


@dataclass
class Received:
    """One frame off `main_video`, with whatever tag survived the round trip."""

    frame: np.ndarray
    frame_id: int
    timestamp_us: int
    user_data: bytes | None
    arrival_index: int


@dataclass
class Report:
    sent: int = 0
    received: int = 0
    tagged: int = 0
    pairing: str = "none"
    output_resolution: tuple[int, int] | None = None
    orb_median_px: float | None = None
    orb_p95_px: float | None = None
    orb_matches_median: float | None = None
    chamfer_median_px: float | None = None
    chamfer_p90_px: float | None = None
    residual_median_px: float | None = None
    residual_p95_px: float | None = None
    global_scale: float | None = None
    global_shift_px: float | None = None
    drift_early_px: float | None = None
    drift_late_px: float | None = None
    coverage: float | None = None
    pairs_measured: int = 0
    diagonal_px: float | None = None
    notes: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------
# Frame loading
# ----------------------------------------------------------------------------


def load_frames(frames_dir: Path, count: int) -> list[np.ndarray]:
    """Load up to `count` RGB frames, sorted by filename."""
    paths = sorted(
        p
        for p in frames_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )
    if not paths:
        sys.exit(f"no images found in {frames_dir}")

    frames = []
    for path in paths[:count]:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"  skipping unreadable {path.name}")
            continue
        # push_frame takes (H, W, 3) uint8 RGB — the same layout Replicator's
        # rgb annotator hands back, so nothing else is needed here.
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return frames


def load_masks(mask_dir: Path | None, count: int) -> list[np.ndarray] | None:
    """Load segmentation masks used to restrict the geometry metric.

    Any non-zero pixel counts as robot. A Replicator instance-segmentation PNG
    works directly; so does a hand-painted binary mask.
    """
    if mask_dir is None:
        return None
    paths = sorted(
        p for p in mask_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    masks = []
    for path in paths[:count]:
        m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            masks.append((m > 0).astype(np.uint8) * 255)
    return masks or None


# ----------------------------------------------------------------------------
# Geometry metrics
# ----------------------------------------------------------------------------


def orb_displacement(
    src: np.ndarray, edited: np.ndarray, mask: np.ndarray | None
) -> tuple[float | None, int]:
    """Median keypoint displacement in pixels between a frame and its edit.

    Directly interpretable: "structure moved N pixels". Returns (None, n) when
    too few features survive the restyle to trust the number — a heavy edit can
    destroy the texture ORB depends on, and a confident number from six matches
    would be worse than no number.
    """
    src_g = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    edit_g = cv2.cvtColor(edited, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(src_g, mask)
    kp2, des2 = orb.detectAndCompute(edit_g, mask)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None, 0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = matcher.knnMatch(des1, des2, k=2)

    # Lowe's ratio test — restyled frames produce a lot of confident nonsense
    # without it.
    good = [m for m, n in (p for p in raw if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 12:
        return None, len(good)

    deltas = [
        float(np.hypot(*(np.subtract(kp2[m.trainIdx].pt, kp1[m.queryIdx].pt))))
        for m in good
    ]
    return float(np.median(deltas)), len(good)


def residual_drift(
    src: np.ndarray, edited: np.ndarray, mask: np.ndarray | None
) -> tuple[float | None, int, float, float]:
    """Local drift after factoring out the model's global reframing.

    X2 picks its own resolution and re-frames slightly, which shows up as a
    uniform scale + shift across the whole image. That part is harmless and
    correctable — you can resize the edit back. What corrupts action labels is
    what moves *relative to the rest of the scene*: the gripper sliding while
    the table stays put.

    So fit a similarity transform to the matches and report the residual.
    Returns (median residual px, inliers, fitted scale, fitted shift px).
    """
    src_g = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    edit_g = cv2.cvtColor(edited, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(nfeatures=3000)
    kp1, des1 = orb.detectAndCompute(src_g, mask)
    kp2, des2 = orb.detectAndCompute(edit_g, mask)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None, 0, 1.0, 0.0

    raw = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des1, des2, k=2)
    good = [m for m, n in (p for p in raw if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 12:
        return None, len(good), 1.0, 0.0

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])

    M, inliers = cv2.estimateAffinePartial2D(
        src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    if M is None:
        return None, len(good), 1.0, 0.0

    scale = float(np.hypot(M[0, 0], M[0, 1]))
    shift = float(np.hypot(M[0, 2], M[1, 2]))

    warped = (src_pts @ M[:, :2].T) + M[:, 2]
    resid = np.linalg.norm(dst_pts - warped, axis=1)
    n_in = int(inliers.sum()) if inliers is not None else len(good)
    return float(np.median(resid)), n_in, scale, shift


def chamfer_edge_distance(
    src: np.ndarray, edited: np.ndarray, mask: np.ndarray | None
) -> tuple[float, float] | None:
    """Distance from each edited-frame edge pixel to the nearest source edge.

    Robust where ORB fails: a restyle changes colour and texture wholesale but
    leaves silhouettes where they were, so edge geometry is the part that must
    not move. Returns (median, p90) in pixels.
    """
    src_e = cv2.Canny(cv2.cvtColor(src, cv2.COLOR_RGB2GRAY), 80, 200)
    edit_e = cv2.Canny(cv2.cvtColor(edited, cv2.COLOR_RGB2GRAY), 80, 200)
    if mask is not None:
        src_e = cv2.bitwise_and(src_e, mask)
        edit_e = cv2.bitwise_and(edit_e, mask)

    if src_e.sum() == 0 or edit_e.sum() == 0:
        return None

    # Distance to the nearest source edge, sampled at the edited edges.
    dist = cv2.distanceTransform(cv2.bitwise_not(src_e), cv2.DIST_L2, 3)
    d = dist[edit_e > 0]
    if d.size == 0:
        return None
    return float(np.median(d)), float(np.percentile(d, 90))


def structure_coverage(
    src: np.ndarray, edited: np.ndarray, tol: int = 4
) -> float:
    """Fraction of source structure that still has a counterpart in the edit.

    The residual-drift metric answers "how far did matched features move" and
    is therefore blind to features that simply VANISH — an object the model
    paints over scores no displacement at all, because it contributes no match.
    This measures the opposite direction: for every source edge pixel, is there
    an edited edge within `tol` pixels? Deleted structure drives it down.

    1.0 = everything survived. Below ~0.7 something in the scene was erased.
    """
    src_e = cv2.Canny(cv2.cvtColor(src, cv2.COLOR_RGB2GRAY), 80, 200)
    edit_e = cv2.Canny(cv2.cvtColor(edited, cv2.COLOR_RGB2GRAY), 80, 200)
    if src_e.sum() == 0:
        return 1.0
    dist = cv2.distanceTransform(cv2.bitwise_not(edit_e), cv2.DIST_L2, 3)
    d = dist[src_e > 0]
    return float((d <= tol).mean())


def montage(src: np.ndarray, edited: np.ndarray) -> np.ndarray:
    """Source | edit | edge overlay, for eyeballing what the numbers mean."""
    src_e = cv2.Canny(cv2.cvtColor(src, cv2.COLOR_RGB2GRAY), 80, 200)
    edit_e = cv2.Canny(cv2.cvtColor(edited, cv2.COLOR_RGB2GRAY), 80, 200)
    # Source edges green, edited edges magenta. Where geometry held they sit on
    # top of each other and read as white.
    overlay = np.zeros_like(src)
    overlay[..., 1] = src_e
    overlay[..., 0] = edit_e
    overlay[..., 2] = edit_e
    return np.hstack([src, edited, overlay])


# ----------------------------------------------------------------------------
# Streaming
# ----------------------------------------------------------------------------


async def stream(
    frames: list[np.ndarray], prompt: str, fps: float, drain_s: float
) -> tuple[list[Received], list[str]]:
    api_key = os.environ.get("REACTOR_API_KEY")
    if not api_key:
        sys.exit("set REACTOR_API_KEY — https://www.reactor.inc/account/api-keys")

    reactor = Reactor(model_name=MODEL, api_key=api_key)
    received: list[Received] = []
    events: list[str] = []

    @reactor.on_status
    def _status(new: str) -> None:
        print(f"  status: {new}")

    @reactor.on_error
    def _error(exc: Exception) -> None:
        events.append(f"error: {exc}")
        print(f"  error: {exc}")

    @reactor.on_message
    def _message(msg: dict) -> None:
        kind = msg.get("type") or msg.get("event") or "message"
        # generation_started carries the resolution X2 picked for the session.
        if kind in {"generation_started", "state_update"}:
            events.append(json.dumps(msg)[:400])

    async with reactor:
        await reactor.connect()
        print(f"  session: {reactor.session_id}")

        source = reactor.track(SOURCE_TRACK)
        await source.publish()

        output = reactor.track(OUTPUT_TRACK)

        @output.on_frame
        def _collect(frame, frame_id, timestamp_us, user_data):  # noqa: ANN001
            received.append(
                Received(
                    frame=frame.copy(),
                    frame_id=frame_id,
                    timestamp_us=timestamp_us,
                    user_data=bytes(user_data) if user_data else None,
                    arrival_index=len(received),
                )
            )

        # Every frame in order, no dropping. Without this the model reads only
        # the newest frames and the 1:1 correspondence the labels rely on is
        # gone — silently.
        await reactor.send_command("set_keep_backlog", {"keep_backlog": True})

        # Publish before prompting: X2 has no `start`, it begins once it has
        # both a prompt and frames, and a prompt with no sender behind the slot
        # buys nothing.
        await reactor.send_command("set_prompt", {"prompt": prompt})

        interval = 1.0 / fps
        next_push = time.monotonic()
        for i, frame in enumerate(frames):
            # The tag is the whole experiment: if it comes back, labels re-pair
            # deterministically and nothing else about ordering matters.
            tag = json.dumps({"seq": i}).encode()
            source.push_frame(frame, user_data=tag)
            next_push += interval
            await asyncio.sleep(max(0.0, next_push - time.monotonic()))

        print(f"  sent {len(frames)}; draining {drain_s:.0f}s for the backlog")
        # keep_backlog=true trades latency for completeness, so the tail of the
        # stream is still in flight when the last push returns.
        deadline = time.monotonic() + drain_s
        last_count = -1
        while time.monotonic() < deadline:
            await asyncio.sleep(1.0)
            if len(received) == last_count and len(received) >= len(frames):
                break
            last_count = len(received)

        source.unpublish()

    return received, events


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------


def pair_frames(
    frames: list[np.ndarray], received: list[Received]
) -> tuple[list[tuple[int, Received]], str]:
    """Match edited frames back to their source frames.

    Prefers the echoed tag. Falls back to arrival order, which `keep_backlog`
    makes plausible but does not guarantee — a fallback pairing makes every
    geometry number below advisory rather than conclusive.
    """
    tagged = [r for r in received if r.user_data]
    if tagged and len(tagged) >= 0.5 * len(received):
        pairs = []
        for r in tagged:
            try:
                seq = json.loads(r.user_data)["seq"]
            except (ValueError, KeyError, TypeError):
                continue
            if 0 <= seq < len(frames):
                pairs.append((seq, r))
        if pairs:
            return pairs, "user_data"

    # No echoed tags. Arrival order is still monotonic under keep_backlog, but
    # the model consumes some leading frames warming up, so output i generally
    # corresponds to input i + k. Fit k by trying each candidate and keeping
    # whichever minimises drift on a few probe frames — a wrong k inflates every
    # geometry number afterwards and looks exactly like model infidelity.
    best_k, best_score = 0, float("inf")
    # Skip the opening frames: a near-black or featureless first frame makes
    # every offset score alike and the fit lands on 0 by default.
    lo = min(len(received) - 1, max(0, len(received) // 6))
    probes = [int(p) for p in np.linspace(lo, len(received) - 1, min(6, len(received) - lo))]
    max_k = max(0, len(frames) - len(received))

    for k in range(0, max_k + 1):
        scores = []
        for j in probes:
            if j + k >= len(frames):
                continue
            src = frames[j + k]
            edit = cv2.resize(
                received[j].frame, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_AREA
            )
            # Scored on coverage, not residual drift. Once an edit ADDS
            # content, residual goes flat across every candidate offset — the
            # added pixels match nothing at any k — so it cannot discriminate.
            # Coverage still peaks sharply at the true offset. Measured on a
            # fruit-adding run: residual spanned 0.90-1.02px across k=0..8
            # (useless) while coverage ran 0.64-0.84 and peaked at the right k.
            scores.append(1.0 - structure_coverage(src, edit))
        if scores:
            score = float(np.median(scores))
            if score < best_score:
                best_k, best_score = k, score

    n = min(len(frames) - best_k, len(received))
    label = f"arrival-order +{best_k} (fitted, coverage {1.0 - best_score:.2f})"
    if best_score == float("inf"):
        label = "arrival-order (fallback, offset unfittable)"
    return [(i + best_k, received[i]) for i in range(n)], label


def analyse(
    frames: list[np.ndarray],
    masks: list[np.ndarray] | None,
    received: list[Received],
    out_dir: Path,
    sample: int,
    dump_edited: bool = False,
) -> Report:
    rep = Report(sent=len(frames), received=len(received))
    rep.tagged = sum(1 for r in received if r.user_data)

    if not received:
        rep.notes.append("no frames came back — nothing to measure")
        return rep

    h, w = received[0].frame.shape[:2]
    rep.output_resolution = (w, h)

    pairs, how = pair_frames(frames, received)
    rep.pairing = how
    if not pairs:
        rep.notes.append("could not pair any frames")
        return rep

    src_h, src_w = frames[0].shape[:2]
    rep.diagonal_px = float(np.hypot(src_w, src_h))

    orb_meds, orb_counts, cham_meds, cham_p90s = [], [], [], []
    resids, scales, shifts = [], [], []
    coverages = []
    step = max(1, len(pairs) // max(1, sample))

    edited_dir = out_dir / "edited"
    if dump_edited:
        edited_dir.mkdir(parents=True, exist_ok=True)

    for idx, (seq, rec) in enumerate(pairs):
        src = frames[seq]
        # X2 picks its own resolution from the source aspect ratio, so the edit
        # comes back a different size; compare in source pixels.
        edited = cv2.resize(rec.frame, (src_w, src_h), interpolation=cv2.INTER_AREA)
        mask = masks[seq] if masks and seq < len(masks) else None
        if mask is not None and mask.shape[:2] != (src_h, src_w):
            mask = cv2.resize(mask, (src_w, src_h), interpolation=cv2.INTER_NEAREST)

        med, n_matches = orb_displacement(src, edited, mask)
        if med is not None:
            orb_meds.append(med)
        orb_counts.append(n_matches)

        cham = chamfer_edge_distance(src, edited, mask)
        if cham is not None:
            cham_meds.append(cham[0])
            cham_p90s.append(cham[1])

        coverages.append(structure_coverage(src, edited))

        res, n_in, scale, shift = residual_drift(src, edited, mask)
        if res is not None and n_in >= 12:
            resids.append(res)
            scales.append(scale)
            shifts.append(shift)

        if dump_edited:
            # These are the augmented dataset — keep them, don't regenerate.
            cv2.imwrite(
                str(edited_dir / f"f{seq:06d}.png"),
                cv2.cvtColor(rec.frame, cv2.COLOR_RGB2BGR),
            )

        rep.pairs_measured += 1

        if idx % step == 0 and idx // step < sample:
            cv2.imwrite(
                str(out_dir / f"pair_{seq:04d}.png"),
                cv2.cvtColor(montage(src, edited), cv2.COLOR_RGB2BGR),
            )

    if orb_meds:
        rep.orb_median_px = float(np.median(orb_meds))
        rep.orb_p95_px = float(np.percentile(orb_meds, 95))
    if orb_counts:
        rep.orb_matches_median = float(np.median(orb_counts))
    if cham_meds:
        rep.chamfer_median_px = float(np.median(cham_meds))
        rep.chamfer_p90_px = float(np.median(cham_p90s))
    if coverages:
        rep.coverage = float(np.median(coverages))
    if resids:
        rep.residual_median_px = float(np.median(resids))
        rep.residual_p95_px = float(np.percentile(resids, 95))
        rep.global_scale = float(np.median(scales))
        rep.global_shift_px = float(np.median(shifts))
        # Fidelity is not constant: the model's own output feeds back, so late
        # frames drift further than early ones. A big gap here means short
        # segments with periodic resets, not one long stream.
        third = max(1, len(resids) // 3)
        rep.drift_early_px = float(np.median(resids[:third]))
        rep.drift_late_px = float(np.median(resids[-third:]))

    return rep


def verdict(rep: Report) -> tuple[str, list[str]]:
    """Turn the measurements into a go / no-go, with the reasoning shown."""
    reasons = []
    fatal = False
    caution = False

    if rep.received == 0:
        return "NO-GO", ["no frames returned"]

    # 1. Tag echo.
    if rep.tagged == 0:
        caution = True
        reasons.append(
            "X2 does not echo user_data — no reliable frame correspondence. "
            "Labels can only be re-paired by arrival order, which keep_backlog "
            "makes likely but does not guarantee."
        )
    elif rep.tagged < rep.received:
        caution = True
        reasons.append(
            f"only {rep.tagged}/{rep.received} frames carried a tag — partial correspondence"
        )
    else:
        reasons.append("tag echo works: every edited frame re-pairs to its source")

    # 2. Frame count.
    ratio = rep.received / rep.sent if rep.sent else 0
    if ratio < 0.9:
        caution = True
        reasons.append(
            f"got {rep.received} frames for {rep.sent} sent ({ratio:.0%}) — "
            "frames were dropped or the drain window was too short"
        )
    else:
        reasons.append(f"frame count {rep.received}/{rep.sent} ({ratio:.0%})")

    # 3. Geometry. Thresholds are relative to the source diagonal so they hold
    # across render resolutions; ~0.5% of the diagonal is roughly 2px at 256².
    diag = rep.diagonal_px or 1.0

    # Checked before drift: drift only sees features that still exist, so a
    # scene the model painted over scores a clean displacement while the thing
    # the labels describe is gone.
    if rep.coverage is not None:
        reasons.append(f"structure coverage {rep.coverage:.2f} (1.0 = nothing erased)")
        if rep.coverage < 0.60:
            fatal = True
            reasons.append(
                "  → source structure was ERASED, not merely restyled; the drift "
                "number below is measured on whatever survived and is meaningless"
            )
        elif rep.coverage < 0.80:
            caution = True
            reasons.append("  → some source structure lost; inspect the montages")

    if rep.global_scale is not None:
        reasons.append(
            f"global reframing: scale {rep.global_scale:.3f}, shift "
            f"{rep.global_shift_px:.1f}px — uniform, correctable by resize"
        )
    if rep.residual_median_px is not None:
        pct = rep.residual_median_px / diag * 100
        reasons.append(
            f"LOCAL drift {rep.residual_median_px:.1f}px after removing global "
            f"reframing ({pct:.2f}% of diagonal) — this is what breaks labels"
        )
        if pct > 1.0:
            fatal = True
            reasons.append("  → structure moves relative to the scene; labels corrupted")
        elif pct > 0.4:
            caution = True
            reasons.append("  → borderline; safe only for coarse action chunks")
        if rep.drift_early_px is not None and rep.drift_late_px is not None:
            reasons.append(
                f"temporal drift: {rep.drift_early_px:.1f}px early → "
                f"{rep.drift_late_px:.1f}px late"
            )
            if rep.drift_late_px > 2.0 * max(rep.drift_early_px, 0.5):
                caution = True
                reasons.append(
                    "  → fidelity decays along the stream; use short segments "
                    "with periodic reset rather than one long run"
                )
    if rep.orb_median_px is not None:
        reasons.append(
            f"raw ORB displacement {rep.orb_median_px:.1f}px "
            f"(includes global reframing), median {rep.orb_matches_median:.0f} matches/frame"
        )
    if rep.residual_median_px is None:
        caution = True
        reasons.append(
            "ORB found too few stable matches to measure — the edit is heavy "
            "enough to destroy texture. INCONCLUSIVE, not a pass: soften the "
            "prompt and re-run, or check the montages by eye"
        )

    if rep.chamfer_median_px is not None and rep.chamfer_p90_px is not None:
        pct = rep.chamfer_median_px / diag * 100
        # Advisory only. Measured against known shifts this metric barely
        # separates a 0px recolour (0.95px) from an 8px translation (1.37px):
        # on textured frames there is always a source edge nearby, so it
        # under-reports badly. It is a sanity backstop for when ORB cannot
        # measure at all, never grounds for a pass.
        reasons.append(
            f"edge displacement {rep.chamfer_median_px:.1f}px median / "
            f"{rep.chamfer_p90_px:.1f}px p90 ({pct:.2f}% of diagonal) [advisory]"
        )
        if pct > 2.0:
            caution = True
            reasons.append("  → silhouettes appear to have moved; inspect the montages")

    if fatal:
        return "NO-GO", reasons
    if caution:
        return "CONDITIONAL", reasons
    return "GO", reasons


# ----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-dir", type=Path, required=True, help="directory of rendered RGB frames")
    ap.add_argument("--mask-dir", type=Path, help="optional masks; restricts geometry to the robot")
    ap.add_argument("--count", type=int, default=96, help="frames to send (default 96 = 4s at 24fps)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="edit instruction — must not mention the robot")
    ap.add_argument("--fps", type=float, default=NATIVE_FPS, help=f"push rate (default {NATIVE_FPS:g})")
    ap.add_argument("--drain", type=float, default=60.0, help="seconds to wait for the backlog")
    ap.add_argument("--sample", type=int, default=6, help="comparison images to write")
    ap.add_argument("--out", type=Path, default=Path("out"), help="output directory")
    ap.add_argument(
        "--dump-edited", action="store_true", help="save every edited frame (the augmented dataset)"
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"loading frames from {args.frames_dir}")
    frames = load_frames(args.frames_dir, args.count)
    masks = load_masks(args.mask_dir, args.count)
    print(f"  {len(frames)} frames at {frames[0].shape[1]}x{frames[0].shape[0]}")
    if masks:
        print(f"  {len(masks)} masks — geometry restricted to masked region")
    print(f"prompt: {args.prompt!r}")

    print(f"streaming to {MODEL}")
    received, events = asyncio.run(stream(frames, args.prompt, args.fps, args.drain))

    print("analysing")
    rep = analyse(frames, masks, received, args.out, args.sample, args.dump_edited)
    call, reasons = verdict(rep)

    print()
    print("=" * 68)
    print(f"  VERDICT: {call}")
    print("=" * 68)
    for r in reasons:
        print(f"  {r}")
    print()
    print(f"  pairing:           {rep.pairing}")
    print(f"  output resolution: {rep.output_resolution}")
    print(f"  pairs measured:    {rep.pairs_measured}")
    print(f"  comparisons:       {args.out}/pair_*.png")
    print("     green = source edges, magenta = edited edges, white = aligned")

    (args.out / "report.json").write_text(
        json.dumps(
            {"verdict": call, "reasons": reasons, "metrics": rep.__dict__, "events": events[:20]},
            indent=2,
            default=str,
        )
    )
    print(f"  report:            {args.out}/report.json")

    sys.exit(0 if call != "NO-GO" else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("interrupted")
