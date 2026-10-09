"""Run MediaPipe BlazePose over the Multi-View Fitness Dataset and cache each
clip's raw per-frame world landmarks, leg visibility and body scale. The rule-based
squat classifier is evaluated on this cache (scripts/eval_rule_classifier.py).

The body scale cached here is a PER-CLIP estimate (femur, hip width), which is
unreliable whenever a clip's camera framing cuts the legs out - see
process_clip()'s docstring. Run scripts/rebuild_multiview_scale.py afterwards
to replace it with a per-subject scale pooled across all of that subject's clips;
this script caches raw landmarks so that correction never needs to re-run
MediaPipe.

Each clip is decoded once and its landmarks cached as a .npz under
--out; re-running the script skips any clip whose cache already has raw
landmarks (an older cache missing them is reprocessed automatically), so an
interrupted run resumes for free and a 1000+ clip pass only ever pays the
MediaPipe cost once per clip.

    python scripts/preprocess_multiview.py --data "data/A Multi-View Raw Video Dataset of Seven Fitness Ex/Dataset Exercise Quality-wise.zip"
    python scripts/preprocess_multiview.py --data ... --limit 20        # smoke test
    python scripts/preprocess_multiview.py --data ... --exercises squat  # one exercise only
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stancesense.datasets.multiview_fitness import discover_clips, extract_clip  # noqa: E402
from stancesense.datasets.mmfit import MP, estimate_scale                        # noqa: E402


# The six lower-body landmarks that actually determine femur/hip-width scale
# (see estimate_scale in mmfit.py). A clip's own per-landmark visibility on
# THESE points — not whether MediaPipe reported "a pose" at all — is what
# tells you whether its scale estimate can be trusted (see module docstring:
# hip visibility stays ~0.99 even in badly-cropped clips; knee/ankle
# visibility is what actually drops).
_SCALE_LANDMARKS = (MP.L_HIP, MP.R_HIP, MP.L_KNEE, MP.R_KNEE, MP.L_ANKLE, MP.R_ANKLE)


def landmarks_from_mediapipe(world_landmarks) -> tuple:
    """(33, 3) coords and (33,) per-landmark visibility from a MediaPipe
    ``pose_world_landmarks`` result; zeros if no pose was found this frame."""
    lm = np.zeros((MP.N, 3), dtype=float)
    vis = np.zeros(MP.N, dtype=float)
    if world_landmarks is not None:
        for i, p in enumerate(world_landmarks.landmark):
            if i >= MP.N:
                break
            lm[i] = (p.x, p.y, p.z)
            vis[i] = p.visibility
    return lm, vis


def process_clip(clip, pose, tmp_dir: str) -> dict:
    """Decode one clip, run MediaPipe on every frame, return its landmark stream,
    leg visibility and body scale.

    Frames where MediaPipe finds no pose are forward-filled from the last
    successful detection (back-filled from the first, for a leading gap) —
    a tracker holds the last value, it does not see the future.

    The per-clip (femur, hip_w) computed here is a fallback only. It is
    unreliable whenever a clip's camera framing cuts off the legs — MediaPipe
    keeps reporting the HIPS as confidently detected (visibility ~0.99) even
    then, so filtering on "was a pose detected" cannot catch it; only the
    knee/ankle visibility does (see _SCALE_LANDMARKS). The authoritative scale
    is computed later, per SUBJECT, by scripts/rebuild_multiview_scale.py
    pooling well-tracked-leg frames across all of a subject's clips — a
    physical constant should not be re-estimated from whichever single clip
    happens to have a bad camera angle. Raw landmarks + per-frame leg
    visibility are cached here so that rebuild never needs to re-run
    MediaPipe.
    """
    import cv2

    local_path = extract_clip(clip, tmp_dir)
    cleanup = clip.zip_member is not None
    try:
        cap = cv2.VideoCapture(local_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        raw_lm, raw_vis, detected = [], [], []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            wl = result.pose_world_landmarks
            lm, vis = landmarks_from_mediapipe(wl)
            raw_lm.append(lm)
            raw_vis.append(vis)
            detected.append(wl is not None)
        cap.release()
    finally:
        if cleanup and os.path.exists(local_path):
            os.remove(local_path)

    n = len(raw_lm)
    if n == 0:
        raise RuntimeError("no frames decoded")
    detected = np.array(detected, dtype=bool)
    if not detected.any():
        raise RuntimeError("MediaPipe detected no pose in any frame")

    lm_all = np.stack(raw_lm)                        # (N, 33, 3)
    vis_all = np.stack(raw_vis)                       # (N, 33)
    leg_vis = vis_all[:, _SCALE_LANDMARKS].mean(axis=1)   # (N,) — NOT forward-filled;
                                                            # a filled frame is not a real detection
    idx = np.where(detected, np.arange(n), 0)
    idx = np.maximum.accumulate(idx)                 # forward-fill
    first_ok = int(np.argmax(detected))
    idx[:first_ok] = first_ok                        # back-fill a leading gap
    lm_filled = lm_all[idx]

    femur, hip_w = estimate_scale(lm_filled)
    return dict(landmarks=lm_filled.astype(np.float32),
               leg_vis=leg_vis.astype(np.float32), n_frames=n, fps=fps,
               detected_frac=float(detected.mean()), femur=femur, hip_w=hip_w)


def _open_append_csv(path: str, header: list):
    write_header = not os.path.exists(path)
    fh = open(path, "a", newline="")
    w = csv.writer(fh)
    if write_header:
        w.writerow(header)
    return fh, w


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True,
                    help="dataset .zip archive or an already-extracted directory")
    ap.add_argument("--out", default="data/processed_multiview")
    ap.add_argument("--config", default="config/default.yaml",
                    help="reuse the project's own MediaPipe pose settings")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N clips")
    ap.add_argument("--exercises", nargs="*", default=None)
    ap.add_argument("--force", action="store_true", help="reprocess even if cached")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard-index", type=int, default=0,
                    help="0-based shard id for running several workers in parallel")
    ap.add_argument("--shard-count", type=int, default=1,
                    help="total worker count; clips are split shard_index::shard_count "
                         "over a fixed, sorted clip order so shards never overlap")
    args = ap.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)

    clips = discover_clips(args.data)
    if args.exercises:
        wanted = set(args.exercises)
        clips = [c for c in clips if c.exercise in wanted]
    if args.limit:
        clips = clips[:args.limit]
    if args.shard_count > 1:
        if not (0 <= args.shard_index < args.shard_count):
            raise SystemExit("--shard-index must be in [0, --shard-count)")
        clips = clips[args.shard_index::args.shard_count]
    print(f"{len(clips)} clips to process from {args.data!r}"
         + (f"  [shard {args.shard_index}/{args.shard_count}]" if args.shard_count > 1 else ""))

    with open(args.config) as f:
        pose_cfg = yaml.safe_load(f)["pose"]
    import mediapipe as mp
    pose = mp.solutions.pose.Pose(
        model_complexity=pose_cfg["model_complexity"],
        smooth_landmarks=pose_cfg["smooth_landmarks"],
        min_detection_confidence=pose_cfg["min_detection_confidence"],
        min_tracking_confidence=pose_cfg["min_tracking_confidence"],
    )

    os.makedirs(args.out, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix=f"multiview_extract_{args.shard_index}_")
    # Sharded runs write to per-shard manifest/failures files (concurrent writers
    # to one CSV would corrupt it); multiview_fitness.load_manifest merges every
    # manifest*.csv it finds, so a single-shard run (the default) is unaffected.
    suffix = f".shard{args.shard_index}" if args.shard_count > 1 else ""
    manifest_f, manifest_w = _open_append_csv(
        os.path.join(args.out, f"manifest{suffix}.csv"),
        ["clip_id", "subject_id", "exercise", "quality", "view", "label",
         "n_frames", "fps", "detected_frac", "femur", "hip_w", "npz_path"])
    failures_f, failures_w = _open_append_csv(
        os.path.join(args.out, f"failures{suffix}.csv"), ["clip_id", "error"])

    t_start = time.time()
    n_done = n_skip = n_fail = 0
    try:
        for i, clip in enumerate(clips):
            out_dir = os.path.join(args.out, clip.exercise, clip.quality)
            os.makedirs(out_dir, exist_ok=True)
            npz_path = os.path.join(out_dir, clip.clip_id + ".npz")
            if os.path.exists(npz_path) and not args.force:
                # Self-heal a stale cache from before raw landmarks were saved
                # (see process_clip's docstring) instead of silently skipping it.
                with np.load(npz_path) as cached:
                    has_landmarks = "landmarks" in cached
                if has_landmarks:
                    n_skip += 1
                    continue
            try:
                r = process_clip(clip, pose, tmp_dir)
            except Exception as e:  # a bad clip must not kill a multi-hour run
                n_fail += 1
                failures_w.writerow([clip.clip_id, repr(e)])
                failures_f.flush()
                print(f"  [FAIL] {clip.clip_id}: {e}")
                continue

            np.savez_compressed(
                npz_path, landmarks=r["landmarks"], leg_vis=r["leg_vis"],
                n_frames=r["n_frames"], fps=r["fps"],
                detected_frac=r["detected_frac"], femur=r["femur"], hip_w=r["hip_w"],
                subject_id=clip.subject_id,
                exercise=clip.exercise, quality=clip.quality, view=clip.view,
                label=clip.label)
            manifest_w.writerow([
                clip.clip_id, clip.subject_id, clip.exercise, clip.quality, clip.view,
                clip.label, r["n_frames"], round(r["fps"], 2),
                round(r["detected_frac"], 4), round(r["femur"], 4), round(r["hip_w"], 4),
                npz_path])
            manifest_f.flush()
            n_done += 1

            if (i + 1) % 25 == 0 or (i + 1) == len(clips):
                elapsed = time.time() - t_start
                rate = n_done / elapsed if elapsed > 0 else 0
                remaining = len(clips) - (i + 1)
                eta_min = remaining / rate / 60 if rate > 0 else float("nan")
                print(f"[{i+1}/{len(clips)}] done={n_done} skipped={n_skip} "
                     f"failed={n_fail}  ({rate:.2f} clips/s, ETA {eta_min:.1f} min)")
    finally:
        manifest_f.close()
        failures_f.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\nfinished: {n_done} processed, {n_skip} already cached, {n_fail} failed")
    print(f"manifest -> {os.path.join(args.out, 'manifest.csv')}")
    if n_fail:
        print(f"failures -> {os.path.join(args.out, 'failures.csv')}")


if __name__ == "__main__":
    main()
