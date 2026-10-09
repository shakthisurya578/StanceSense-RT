"""Replace each cached clip's PER-CLIP body-scale estimate with a PER-SUBJECT one.

WHY THIS EXISTS
----------------
scripts/preprocess_multiview.py estimates (femur, hip_w) independently for
every clip, the way MM-Fit's adapter estimates it independently per WORKOUT.
That assumption breaks here: MM-Fit's "workout" is one continuous recording
of a whole session, but this dataset's "clip" is a single tightly-cropped
exercise repetition, and some exercises (bicep_curl, tricep) are filmed close
enough that the camera cuts the legs out of frame.

Diagnosing subject_023's bicep_curl clips (2026-09 investigation) showed why
that is not caught by "was a pose detected": MediaPipe kept reporting the HIP
landmarks as ~99% confident throughout, while knee/ankle visibility dropped
to ~0.3-0.6 — and even restricting the SAME clip to only its highest-
visibility frames left the estimate unchanged (0.1044 either way). The bias
is not a few corrupted frames; the whole clip's scale estimate is
systematically low because the legs are barely in shot. Meanwhile that same
subject's squat/abs clips — which by exercise definition show the whole body
— give hip_w ~0.20-0.24, consistent with a real person and with every other
subject.

Femur length and hip width are physical constants for a person. This script
therefore pools well-tracked-leg frames (mean visibility of hip/knee/ankle
landmarks >= --vis-threshold) across ALL of a subject's clips, computes ONE
(femur, hip_w) per subject from that pool via mmfit.py's own estimate_scale
(reused unmodified), and stores that constant in every one of the subject's
clips - instead of whatever the clip's own camera framing happened to allow.

    python scripts/rebuild_multiview_scale.py --processed data/processed_multiview
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stancesense.datasets.mmfit import estimate_scale                   # noqa: E402


def load_all_manifest_rows(processed_dir: str) -> list:
    by_id = {}
    for path in sorted(glob.glob(os.path.join(processed_dir, "manifest*.csv"))):
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                by_id[row["clip_id"]] = row
    return list(by_id.values())


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", default="data/processed_multiview")
    ap.add_argument("--vis-threshold", type=float, default=0.5,
                    help="min mean hip/knee/ankle visibility for a frame to "
                         "count toward a subject's pooled scale estimate")
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    rows = load_all_manifest_rows(args.processed)
    if not rows:
        raise SystemExit(f"no manifest*.csv under {args.processed!r}")
    missing_landmarks = [r for r in rows if "npz_path" not in r]
    by_subject = defaultdict(list)
    for r in rows:
        by_subject[int(r["subject_id"])].append(r)

    print(f"{len(rows)} clips across {len(by_subject)} subjects")

    report = {}
    n_rebuilt_clips = 0
    for subject_id in sorted(by_subject):
        clip_rows = by_subject[subject_id]
        pooled_lm = []
        old_scales = []
        for r in clip_rows:
            with np.load(r["npz_path"]) as d:
                if "landmarks" not in d:
                    raise SystemExit(
                        f"{r['npz_path']!r} has no cached raw landmarks — "
                        f"re-run scripts/preprocess_multiview.py first (it "
                        f"self-heals caches from before this fix)")
                lm = d["landmarks"]
                vis = d["leg_vis"]
                old_scales.append((float(d["femur"]), float(d["hip_w"])))
            good = vis >= args.vis_threshold
            if good.any():
                pooled_lm.append(lm[good])

        n_pool_frames = sum(len(p) for p in pooled_lm)
        if pooled_lm:
            pooled = np.concatenate(pooled_lm, axis=0)
            femur, hip_w = estimate_scale(pooled)
            fallback = False
        else:
            # No clip for this subject ever had a well-tracked lower body —
            # fall back to the per-clip estimates' own median rather than
            # silently guessing 1.0/1.0 (estimate_scale's own degenerate case).
            femur = float(np.median([f for f, _ in old_scales]))
            hip_w = float(np.median([h for _, h in old_scales]))
            fallback = True
            print(f"  [warn] subject_{subject_id:03d}: no frame anywhere met "
                 f"the visibility threshold — using median of per-clip estimates")

        per_clip_before = []
        for r in clip_rows:
            with np.load(r["npz_path"]) as d:
                lm = d["landmarks"]
                old_femur, old_hip_w = float(d["femur"]), float(d["hip_w"])
                n_frames = int(d["n_frames"])
                fps = float(d["fps"])
                detected_frac = float(d["detected_frac"])
                leg_vis = d["leg_vis"]
                subj = int(d["subject_id"])
                exercise, quality, view = str(d["exercise"]), str(d["quality"]), str(d["view"])
                label = int(d["label"])
            per_clip_before.append(dict(clip_id=r["clip_id"], old_femur=old_femur,
                                        old_hip_w=old_hip_w,
                                        mean_leg_vis=float(leg_vis.mean())))
            np.savez_compressed(
                r["npz_path"], landmarks=lm, leg_vis=leg_vis,
                n_frames=n_frames, fps=fps, detected_frac=detected_frac,
                femur=femur, hip_w=hip_w,
                subject_id=subj, exercise=exercise, quality=quality, view=view,
                label=label)
            n_rebuilt_clips += 1

        biggest_shift = max(
            (abs(b["old_hip_w"] - hip_w) for b in per_clip_before), default=0.0)
        report[f"subject_{subject_id:03d}"] = dict(
            n_clips=len(clip_rows), n_pooled_frames=n_pool_frames,
            fallback_used=fallback, subject_femur=femur, subject_hip_w=hip_w,
            biggest_per_clip_hip_w_shift=biggest_shift,
            per_clip=per_clip_before)
        flag = "  <-- corrected a clip whose per-clip hip_w was off by >0.03" \
            if biggest_shift > 0.03 else ""
        print(f"subject_{subject_id:03d}: femur={femur:.4f} hip_w={hip_w:.4f}  "
             f"(pooled from {n_pool_frames} well-tracked frames across "
             f"{len(clip_rows)} clips){flag}")

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "multiview_scale_correction.json")
    with open(out_path, "w") as fh:
        json.dump(dict(vis_threshold=args.vis_threshold, n_clips_rebuilt=n_rebuilt_clips,
                       n_subjects=len(by_subject), by_subject=report), fh, indent=2)
    print(f"\nrewrote {n_rebuilt_clips} clips with their subject's body scale")
    print(f"correction report -> {out_path}")


if __name__ == "__main__":
    main()
