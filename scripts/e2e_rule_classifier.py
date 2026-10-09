"""Real-time path test for the rule-based classifier, on recorded video.

    video -> OpenCV -> the app's PoseEstimator (MediaPipe) -> world landmarks +
    leg visibility -> RuleSquatRunner (what live_assessment.py now constructs)
    with frame timestamps -> reps counted / confirmed / rejected -> stance

The camera itself is the only piece not exercised. ``--simulate-fps 30`` shows
each 15 fps frame twice, 1/30 s apart, as a webcam stream would arrive.
Reference rep count: knee-angle valleys (reference_count below).

    python scripts/e2e_rule_classifier.py --clip subject_002_squat_good_front subject_025_tricep_good_front
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from stancesense.datasets.mmfit import estimate_scale  # noqa: E402
from stancesense.datasets.multiview_fitness import discover_clips, extract_clip  # noqa: E402
from stancesense.squat.rule_classifier import RuleSquatRunner  # noqa: E402
from stancesense.kinematics.pose_estimator import PoseEstimator  # noqa: E402
import cv2  # noqa: E402

LEG = (23, 24, 25, 26, 27, 28)


def video_landmarks(path):
    """World landmarks, MEAN leg visibility (the cache's definition), times, fps, frames."""
    pose = PoseEstimator()
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    lms, vis, times, i = [], [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        _, world = pose.infer(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if world is not None:
            pts = world.landmark
            lms.append([[q.x, q.y, q.z] for q in pts])
            vis.append(float(np.mean([pts[j].visibility for j in LEG])))
            times.append(i / fps)
        i += 1
    cap.release()
    return np.asarray(lms, np.float32), np.asarray(vis), np.asarray(times), fps, i
from scipy.signal import find_peaks  # noqa: E402
from stancesense.squat.kinematics import knee_flexion  # noqa: E402
from stancesense.common.types import LEFT, RIGHT  # noqa: E402


def reference_count(landmarks):
    """Rep count from knee-angle valleys: an independent reference for the counter."""
    k = np.array([(knee_flexion(f, LEFT) + knee_flexion(f, RIGHT)) / 2 for f in landmarks])
    ks = np.convolve(k, np.ones(9) / 9, mode="same")
    rng = np.percentile(ks, 95) - np.percentile(ks, 5)
    peaks, _ = find_peaks(-ks, prominence=0.5 * rng, distance=15)
    return len(peaks)


def run(path, simulate_fps=0.0):
    lm, vis, times, fps, n_frames = video_landmarks(path)
    good = lm[vis >= 0.5] if (vis >= 0.5).sum() >= 10 else lm
    femur, hip_w = estimate_scale(good)
    runner = RuleSquatRunner(femur, hip_w)
    trace = []
    rep = max(1, int(round(simulate_fps / fps))) if simulate_fps else 1
    for f, v, t0 in zip(lm, vis, times):
        for r in range(rep):
            st = runner.update(f, now=t0 + r / (simulate_fps or fps), leg_visibility=float(v))
        trace.append((st.knee_angle, runner.clf.last.get("torso", np.nan), runner.clf.last.get("stance_width", np.nan),
                      st.depth))
    s = runner.summary()
    tr = np.array(trace)
    name = os.path.splitext(os.path.basename(path))[0]
    truth = "squat" if "_squat_" in name else "non-squat"
    pred = "squat" if s["confirmed"] >= runner.clf.cfg.min_valid_reps else "non-squat"
    return dict(video=name, truth=truth, prediction=pred, correct=pred == truth, stream_fps=simulate_fps or fps,
                frames_with_pose=int(len(lm)), reps_counted=s["counted"], reps_confirmed=s["confirmed"],
                reps_rejected=s["rejected"], reference_reps=int(reference_count(lm)) if truth == "squat" else None,
                rep_scores=s["scores"], rejected_reasons=[r.rule_failed for r in runner.reps if not r.confirmed][:5],
                mean_depth_confirmed=round(s["mean_depth"], 2), knee_angle_min_max=[round(float(np.nanmin(tr[:, 0])), 1),
                                                                                    round(float(np.nanmax(tr[:, 0])), 1)],
                torso_median=round(float(np.nanmedian(tr[:, 1])), 1), stance_width_median=round(float(np.nanmedian(tr[:, 2])), 2),
                stance=s["stance"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", nargs="*", default=[])
    ap.add_argument("--video", nargs="*", default=[])
    ap.add_argument("--simulate-fps", type=float, default=0.0)
    ap.add_argument("--out", default="models/rule_classifier/e2e_realtime.json")
    args = ap.parse_args()
    paths = list(args.video)
    if args.clip:
        zp = glob.glob("data/A Multi-View Raw Video Dataset*/*.zip")[0]
        by_id = {os.path.splitext(c.filename)[0]: c for c in discover_clips(zp)}
        tmp = tempfile.mkdtemp(prefix="e2e_rules_")
        paths += [extract_clip(by_id[c], tmp) for c in args.clip]
    res = []
    for p in paths:
        r = run(p, args.simulate_fps)
        res.append(r)
        print(f"{r['video']:36s} {r['stream_fps']:.0f} fps | {r['truth']:9s} -> {r['prediction']:9s} "
              f"({'OK' if r['correct'] else 'WRONG'}) | cycles {r['reps_counted']}, squat reps {r['reps_confirmed']}, "
              f"rejected {r['reps_rejected']}, reference {r['reference_reps']} | knee {r['knee_angle_min_max']} "
              f"torso {r['torso_median']} stance {r['stance_width_median']} | "
              f"{(r['stance'] or {}).get('current_stance', '-')} -> {(r['stance'] or {}).get('recommendation', '-')}",
              flush=True)
    old = json.load(open(args.out)) if os.path.exists(args.out) else []
    merged = {(r["video"], r["stream_fps"]): r for r in old}
    merged.update({(r["video"], r["stream_fps"]): r for r in res})
    json.dump(list(merged.values()), open(args.out, "w"), indent=1)
    print(f"{sum(r['correct'] for r in res)}/{len(res)} correct -> {args.out}")


if __name__ == "__main__":
    main()
