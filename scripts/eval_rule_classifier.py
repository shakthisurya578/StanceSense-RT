"""Run the rule-based squat classifier on every cached Multi-View video.

No training. Labels are used only to score the predictions; the classifier sees
landmarks, leg visibility and the clip's body scale, nothing else.

People are split ONCE (numpy default_rng(--split-seed) over all 26 person IDs):
the first 6 are DEVELOPMENT people (rules may be adjusted while looking at
them), the other 20 are EVALUATION people (scored with frozen rules only).
Duplicate recordings listed in duplicates.json are skipped (identical files,
some with conflicting labels).

    python scripts/eval_rule_classifier.py --split dev
    python scripts/eval_rule_classifier.py --split eval      # frozen rules, run once
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from stancesense.squat.rule_classifier import RuleConfig, RuleSquatClassifier, stance_recommendation  # noqa: E402
from stancesense.datasets.multiview_fitness import load_manifest  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def people_split(ids, seed, n_dev=6):
    perm = [int(i) for i in np.random.default_rng(seed).permutation(sorted(ids))]
    return sorted(perm[:n_dev]), sorted(perm[n_dev:])


def run_clip(path, cfg):
    d = np.load(path)
    lm, vis, fps = d["landmarks"], d["leg_vis"], float(d["fps"])
    clf = RuleSquatClassifier(float(d["femur"]), float(d["hip_w"]), cfg)
    for i in range(len(lm)):
        clf.push(lm[i], i / fps, float(vis[i]))
    return clf, int(d["label"])


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    tp, tn = int(((p == 1) & (y == 1)).sum()), int(((p == 0) & (y == 0)).sum())
    fp, fn = int(((p == 1) & (y == 0)).sum()), int(((p == 0) & (y == 1)).sum())
    div = lambda a, b: a / b if b else float("nan")
    prec, rec, spec = div(tp, tp + fp), div(tp, tp + fn), div(tn, tn + fp)
    return dict(videos=len(y), squat=int(y.sum()), non_squat=int((1 - y).sum()), TP=tp, FN=fn, FP=fp, TN=tn,
                accuracy=div(tp + tn, len(y)), precision=prec, recall=rec,
                f1=div(2 * prec * rec, prec + rec) if tp else 0.0, specificity=spec,
                balanced_accuracy=(rec + spec) / 2, false_positive_rate=div(fp, fp + tn),
                false_negative_rate=div(fn, fn + tp))


def reason(clf, label):
    v = clf.verdict()
    if label == 1:
        if clf.base_hip is None:
            return "never detected an upright standing start"
        if not clf.cycles:
            return f"no complete stand-descend-bottom-ascend-stand cycle ({clf.aborted} aborted)"
        fails = Counter(f for cy in clf.cycles for f in cy["failed"])
        return (f"{v['valid_reps']} of {v['cycles']} cycles passed; most failed checks: "
                + ", ".join(f"{k} x{n}" for k, n in fails.most_common(3)))
    sq = [cy for cy in clf.cycles if cy["squat"]]
    return (f"{len(sq)} cycles scored >= threshold (scores {[cy['score'] for cy in sq]}); "
            f"failed checks in them: {dict(Counter(f for cy in sq for f in cy['failed']))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["dev", "eval", "all"], required=True)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--processed", default="data/processed_multiview")
    ap.add_argument("--out", default="models/rule_classifier")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cfg = RuleConfig()
    rows = load_manifest(args.processed)
    excluded = set(json.load(open(os.path.join(args.processed, "duplicates.json")))["excluded"])
    rows = [r for r in rows if r["clip_id"] not in excluded]
    dev, ev = people_split({int(r["subject_id"]) for r in rows}, args.split_seed)
    keep = {"dev": dev, "eval": ev, "all": dev + ev}[args.split]
    rows = sorted([r for r in rows if int(r["subject_id"]) in keep], key=lambda r: r["clip_id"])
    print(f"development people {dev}\nevaluation people {ev}\nsplit '{args.split}': {len(rows)} videos")

    os.makedirs(args.out, exist_ok=True)
    out_rows, y, p, fails, stances = [], [], [], [], []
    t0 = time.time()
    for r in rows:
        path = r["npz_path"] if os.path.isabs(r["npz_path"]) else os.path.join(ROOT, r["npz_path"])
        clf, label = run_clip(path, cfg)
        v = clf.verdict()
        pred = int(v["squat"])
        y.append(label); p.append(pred)
        st = stance_recommendation([c for c in clf.cycles if c["squat"]]) if label == 1 and pred else None
        if st:
            stances.append((r["clip_id"], st))
        out_rows.append(dict(video=r["clip_id"], person=r["subject_id"], view=r["view"], exercise=r["exercise"],
                             ground_truth="squat" if label else "non-squat",
                             prediction="squat" if pred else "non-squat",
                             biomechanical_score=v["median_score"], best_score=v["best_score"],
                             rep_count=v["valid_reps"], cycles=v["cycles"], aborted=v["aborted"],
                             correct=int(pred == label),
                             stance=st["current_stance"] if st else "", recommendation=st["recommendation"] if st else ""))
        if pred != label:
            fails.append((r["clip_id"], label, reason(clf, label)))
    m = metrics(y, p)
    with open(os.path.join(args.out, f"results_{args.split}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)
    by_ex = {}
    for r in out_rows:
        e = by_ex.setdefault(r["exercise"], [0, 0])
        e[0] += r["correct"]; e[1] += 1
    json.dump(dict(split=args.split, people=keep, development_people=dev, evaluation_people=ev,
                   config=asdict(cfg), metrics=m, correct_by_exercise=by_ex,
                   failures=[dict(video=f[0], truth=f[1], reason=f[2]) for f in fails],
                   stance_examples=[dict(video=a, **b) for a, b in stances[:10]],
                   stance_recommendation_counts=dict(Counter(b["recommendation"] for _, b in stances)),
                   seconds=round(time.time() - t0, 1)),
              open(os.path.join(args.out, f"metrics_{args.split}.json"), "w"), indent=1)

    print(f"\n{len(rows)} videos in {time.time() - t0:.0f} s")
    print(f"squat {m['squat']}  non-squat {m['non_squat']}")
    print(f"                 Pred squat  Pred non-squat\nActual squat     {m['TP']:10d}  {m['FN']:14d}\n"
          f"Actual non-squat {m['FP']:10d}  {m['TN']:14d}")
    for k in ("accuracy", "precision", "recall", "f1", "specificity", "balanced_accuracy",
              "false_positive_rate", "false_negative_rate"):
        print(f"{k:20s} {m[k] * 100:6.1f}%")
    print("correct by exercise:", {k: f"{a}/{b}" for k, (a, b) in sorted(by_ex.items())})
    print(f"\nFAILURES ({len(fails)}):")
    for vid, lab, why in fails:
        print(f"  {vid} [{'squat' if lab else 'non-squat'}] {why}")


if __name__ == "__main__":
    main()
