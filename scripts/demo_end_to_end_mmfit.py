"""Offline end-to-end demo - the whole StanceSense-RT chain on one MM-Fit workout.

No webcam required.  This is the proof that every phase is wired together:

    MM-Fit 3-D pose
      -> functional hip-rotation readings   (Phase 3, kinematics/rotation.py)
      -> hip profile                        (Phase 5, profiling/profiler.py)
      -> stance recommendation              (Phase 6, recommendation/stance.py)
      -> squat set: the rule-based biomechanical classifier counts and checks
         every rep (squat/rule_classifier.py, the same code the app uses)
      -> squat-based stance advice: NARROW / MODERATE / WIDE
      -> persisted to SQLite                (Phase 11, storage/store.py)
      -> summary + figures in demo_output/  (Phase 12)

Usage
-----
    python scripts/demo_end_to_end_mmfit.py                    # first workout found
    python scripts/demo_end_to_end_mmfit.py --workout w04
    python scripts/demo_end_to_end_mmfit.py --workout w04 --no-store

HONEST SCOPE OF THE ROTATION PART
---------------------------------
MM-Fit contains no seated hip-rotation clips, so this demo cannot run the guided
seated protocol.  It instead applies the *same* shank-based rotation geometry to
the rotation excursion the participant actually shows during the workout, and
summarises it with the same profiler.  That exercises the real code path, but the
resulting profile is a **functional movement-screen signal only** - it is not a
radiographic femoral/acetabular version measurement, and no angular accuracy is
claimed for it.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stancesense.datasets.mmfit import (                       # noqa: E402
    discover_workouts, load_workout, load_flat_workout, squat_sets,
    estimate_scale, landmarks_from_pose3d, SQUAT_LABEL,
)
from stancesense.kinematics.rotation import hip_rotation_angle_from_points, IDX  # noqa: E402
from stancesense.profiling.profiler import build_hip_profile   # noqa: E402
from stancesense.recommendation.stance import recommend_stance  # noqa: E402
from stancesense.squat.kinematics import detect_stance         # noqa: E402
from stancesense.squat.rule_classifier import RuleSquatRunner, stance_recommendation  # noqa: E402
from stancesense.storage import Store, DEFAULT_DB               # noqa: E402
from stancesense.ui import (                                    # noqa: E402
    format_profile, format_stance, format_rep_summary, profile_note, depth_verdict,
    arc_warning, format_squat_stance,
)
from stancesense.common.types import LEFT, RIGHT                # noqa: E402

OUT_DIR = "demo_output"
RULE = "=" * 74


# --------------------------------------------------------------------- loading

def pick_workout(data_root: str, wanted: str = ""):
    """Load the requested workout (or the first one found)."""
    found = discover_workouts(data_root)
    if not found:
        raise SystemExit(f"No MM-Fit workouts found under {data_root!r}. "
                         f"Extract mm-fit.zip so the layout is {data_root}/w00/ ...")
    items = []
    for item in found:
        if isinstance(item, tuple) and item[0] == "__flat__":
            items.append((item[2], lambda i=item: load_flat_workout(i[1], i[2])))
        else:
            items.append((os.path.basename(item), lambda p=item: load_workout(p)))
    if wanted:
        for wid, loader in items:
            if wid == wanted:
                return loader()
        raise SystemExit(f"workout {wanted!r} not found; available: "
                         f"{[w for w, _ in items]}")
    return items[0][1]()


# ------------------------------------------------------- 1. rotation profiling

def quiet_standing_mask(landmarks, workout, motion_pct: float = 40.0):
    """Frames where the participant is between sets and roughly still.

    The seated protocol reads the shank angle while the thigh is FIXED.  Nothing
    in MM-Fit is seated, but the closest analogue is a frame outside any labelled
    exercise (so no lunge or jumping-jack leg swing) where the ankles are barely
    moving.  Reading rotation anywhere else measures gait, not hip rotation.
    """
    fids = workout.frame_ids
    in_exercise = np.zeros(len(fids), dtype=bool)
    for start, end, _reps, _name in workout.labels:
        in_exercise |= (fids >= start) & (fids <= end)
    ankle_mid = (landmarks[:, IDX[LEFT]["ankle"]] + landmarks[:, IDX[RIGHT]["ankle"]]) / 2
    speed = np.concatenate([[0.0], np.linalg.norm(np.diff(ankle_mid, axis=0), axis=1)])
    still = speed < np.percentile(speed, motion_pct)
    mask = (~in_exercise) & still
    return mask if mask.sum() > 100 else np.ones(len(fids), dtype=bool)


def rotation_profile(workout, stride: int = 5):
    """Functional hip-rotation profile from the workout's own quiet-standing frames.

    Runs the Phase-3 shank geometry on every ``stride``-th quiet frame (see
    :func:`quiet_standing_mask`), removes the per-leg neutral (the median reading,
    i.e. the participant's resting shank alignment), and takes the same robust
    extremes the live profiler uses.  Returns ``(profile, per_leg_series)``.

    This is a demonstration of the code path on real pose data, NOT a clinical
    range-of-motion measurement: MM-Fit has no guided seated protocol, so the
    participant never rotates to a true end range.  See the module docstring.
    """
    lm_all = landmarks_from_pose3d(workout.pose3d)
    lm = lm_all[quiet_standing_mask(lm_all, workout)][::stride]
    series = {}
    for side in (LEFT, RIGHT):
        k, a = IDX[side]["knee"], IDX[side]["ankle"]
        raw = np.array([hip_rotation_angle_from_points(f[k], f[a], side) for f in lm])
        series[side] = raw - float(np.median(raw))       # neutral-corrected
    out = {}
    for side in (LEFT, RIGHT):
        v = series[side]
        out[side] = (max(0.0, float(np.percentile(v, 95))),    # internal rotation
                     max(0.0, float(-np.percentile(v, 5))))    # external rotation
    profile = build_hip_profile(out[LEFT][0], out[LEFT][1],
                                out[RIGHT][0], out[RIGHT][1])
    return profile, series


# ------------------------------------------------------------ 2. the squat run

def run_squat_sets(workout):
    """Run every labelled squat set through the rule-based classifier.

    Returns (reps, per_set, confirmed squat cycles).
    """
    all_reps, per_set, squat_cycles = [], [], []
    for s in squat_sets(workout):
        fem, hip_w = estimate_scale(s.landmarks)
        runner = RuleSquatRunner(fem, hip_w)
        for i, f in enumerate(s.landmarks):
            runner.update(f, now=i / 30.0)            # MM-Fit pose is 30 fps
        summary = runner.summary()
        squat_cycles += [c for c in runner.clf.cycles if c["squat"]]
        width_factor, toe_out = detect_stance(s.landmarks[0], hip_w)
        summary.update(start=s.start_frame, end=s.end_frame, true_reps=s.true_reps,
                       stance_width=width_factor, stance_toe=toe_out)
        per_set.append(summary)
        # renumber so reps are unique across the whole session
        for r in runner.reps:
            r.rep_index = len(all_reps) + 1
            all_reps.append(r)
    return all_reps, per_set, squat_cycles


# ----------------------------------------------------------------- 3. figures

def write_figures(workout, series, profile, rec, reps, per_set, out_dir,
                  judge="rule-based classifier"):
    """Save a one-page figure summarising the run. Returns the path, or None."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[figures] matplotlib not installed - skipping the figure")
        return None

    fig, ax = plt.subplots(3, 1, figsize=(9.5, 10.5))
    fig.suptitle(f"StanceSense-RT end-to-end demo - MM-Fit {workout.workout_id}",
                 fontsize=13, fontweight="bold")

    # (a) rotation excursion
    for side, colour in ((LEFT, "tab:blue"), (RIGHT, "tab:orange")):
        ax[0].plot(series[side], lw=0.6, alpha=0.8, color=colour, label=side)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_title("Functional hip rotation, neutral-corrected  "
                    "(+ internal / - external).  Qualitative signal - no accuracy claim.",
                    fontsize=9)
    ax[0].set_ylabel("degrees")
    ax[0].set_xlabel("sample (every 5th frame)")
    ax[0].legend(fontsize=8)

    # (b) per-rep depth, coloured by the classifier's verdict
    import matplotlib.patches as mpatches
    verdict_colour = {True: "tab:green", False: "tab:red", None: "tab:gray"}
    idx = [r.rep_index for r in reps]
    depths = [r.max_depth_ratio for r in reps]
    ax[1].bar(idx, depths, color=[verdict_colour[r.confirmed] for r in reps])
    ax[1].axhline(1.0, color="k", ls="--", lw=0.9)
    ax[1].set_title(f"Per-rep depth and the {judge} verdict", fontsize=9)
    ax[1].set_xlabel("rep")
    ax[1].set_ylabel("depth ratio")
    handles = [mpatches.Patch(color="tab:green", label="confirmed squat"),
               mpatches.Patch(color="tab:red", label="rejected - not a squat"),
               plt.Line2D([], [], color="k", ls="--", lw=0.9,
                          label="thighs parallel (1.0)")]
    ax[1].legend(handles=handles, fontsize=7.5, loc="upper right", ncol=2)

    # (c) the recommendation, as text
    ax[2].axis("off")
    lines = [
        f"HIP PROFILE   {format_profile(_profile_dict(profile))}",
        "",
    ]
    if (warn := arc_warning(_profile_dict(profile))):
        lines += ["WARNING       " + _wrap(warn, 78)[0]]
        lines += ["              " + w for w in _wrap(warn, 78)[1:]]
        lines += [""]
    lines += [
        f"STANCE        {rec.width_factor:.2f}x hip width, {rec.toe_out_deg:.0f} deg toe-out",
        "",
    ]
    lines += _wrap(rec.rationale, 92)
    lines += ["", f"SQUAT SETS    {len(per_set)} set(s):"]
    for p in per_set:
        lines.append(f"   frames {p['start']}-{p['end']}: "
                     f"true {p['true_reps']} | counted {p['counted']} | "
                     f"confirmed {p['confirmed']} | rejected {p['rejected']} | "
                     f"best depth {p['max_depth']:.2f}")
    ax[2].text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=8.5,
               family="monospace", transform=ax[2].transAxes)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = os.path.join(out_dir, f"end_to_end_{workout.workout_id}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _wrap(text, width):
    import textwrap
    return textwrap.wrap(text, width) or [""]


def _profile_dict(p):
    """HipProfile -> the plain dict the ui.report helpers expect."""
    return dict(ir_max=p.ir_max, er_max=p.er_max, total_arc=p.total_arc,
                rotation_bias=p.rotation_bias, pattern=p.pattern,
                ir_left=p.ir_left, er_left=p.er_left,
                ir_right=p.ir_right, er_right=p.er_right)


# --------------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/mm-fit", help="MM-Fit root or flat folder")
    ap.add_argument("--workout", default="", help="e.g. w04 (default: the first found)")
    ap.add_argument("--db", default=DEFAULT_DB, help="SQLite store to append to")
    ap.add_argument("--no-store", action="store_true", help="skip writing to SQLite")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    workout = pick_workout(args.data, args.workout)
    n_squat = sum(1 for r in workout.labels if r[3] == SQUAT_LABEL)
    print(RULE)
    print(f"StanceSense-RT - end-to-end demo on MM-Fit {workout.workout_id} "
          f"({len(workout.frame_ids)} pose frames, {n_squat} labelled squat set(s))")
    print(RULE)

    # -- 1. rotation -> profile -------------------------------------------------
    profile, series = rotation_profile(workout)
    print("\n[1] FUNCTIONAL HIP-ROTATION PROFILE")
    print("    " + format_profile(_profile_dict(profile)))
    print("    per leg: "
          f"L IR {profile.ir_left:.1f} / ER {profile.er_left:.1f}   "
          f"R IR {profile.ir_right:.1f} / ER {profile.er_right:.1f}   "
          f"symmetry {profile.symmetry:.1f}")
    for line in _wrap(profile_note(_profile_dict(profile)), 88):
        print("    " + line)
    if (warn := arc_warning(_profile_dict(profile))):
        for line in _wrap("WARNING: " + warn, 84):
            print("    " + line)

    # -- 2. profile -> stance ---------------------------------------------------
    hip_w = estimate_scale(landmarks_from_pose3d(workout.pose3d))[1]
    rec = recommend_stance(profile, hip_w)
    print("\n[2] RECOMMENDED STANCE")
    print("    " + format_stance(dict(width_factor=rec.width_factor,
                                      toe_out_deg=rec.toe_out_deg,
                                      asymmetric=rec.asymmetric)))
    for line in _wrap(rec.rationale, 88):
        print("    " + line)

    # -- 3. squat run -------------------------------------------------------------
    # The app's rule-based classifier (no training; its rules were set on
    # Multi-View people, never on MM-Fit).
    print("\n[3] SQUAT RUN  (rule-based biomechanical classifier - the app's default)")
    reps, per_set, squat_cycles = run_squat_sets(workout)
    if not reps:
        raise SystemExit(f"{workout.workout_id} has no labelled squat sets to run.")

    true_total = sum(p["true_reps"] for p in per_set)
    counted = len(reps)
    confirmed = sum(1 for r in reps if r.confirmed is True)
    rejected = sum(1 for r in reps if r.confirmed is False)
    print(f"    {'set':<14}{'true':>5}{'counted':>9}{'confirmed':>11}"
          f"{'best depth':>12}{'stance':>16}")
    for p in per_set:
        # With no completed rep there is no per-rep depth; show the deepest frame
        # actually seen (starred) rather than printing a misleading 0.00.
        depth = (f"{p['max_depth']:>12.2f}" if p["counted"]
                 else f"{p['max_depth_seen']:>11.2f}*")
        print(f"    {str(p['start']) + '-' + str(p['end']):<14}{p['true_reps']:>5}"
              f"{p['counted']:>9}{p['confirmed']:>11}{depth}"
              f"{p['stance_width']:>11.2f}x{p['stance_toe']:>4.0f}d")
    if any(not p["counted"] for p in per_set):
        print("    * no rep completed in that set - deepest frame observed, "
              "not a per-rep depth")
    print(f"    {'TOTAL':<14}{true_total:>5}{counted:>9}{confirmed:>11}")
    scores = [r.squat_score for r in reps]
    print(f"    rules: {confirmed} confirmed, {rejected} rejected; rep scores (of 9): {scores}")
    ok = [r for r in reps if r.confirmed]
    depths = [r.max_depth_ratio for r in ok]
    if depths:
        print(f"    depth{' (confirmed squats)' if ok else ''}: mean {np.mean(depths):.2f}, "
              f"best {max(depths):.2f} ({depth_verdict(max(depths))}; 1.0 ~ thighs parallel)")
    else:
        print("    depth: no confirmed squat")
    squat_stance = stance_recommendation(squat_cycles)
    print("\n[3b] SQUAT-BASED STANCE ADVICE")
    for line in _wrap(format_squat_stance(squat_stance), 88):
        print("    " + line)

    # -- 4. persist -------------------------------------------------------------
    session_id = None
    if not args.no_store:
        with Store(args.db) as store:
            session_id = store.save_assessment(
                profile=profile, stance=rec, reps=reps,
                subject_id=workout.workout_id, source="mmfit",
                notes=f"offline demo on MM-Fit {workout.workout_id} (classifier=rules)",
                classifier="rules", squat_stance=squat_stance,
            )
            stored = store.rep_summary(session_id)
        print(f"\n[4] STORED   session #{session_id} -> {args.db}")
        print("    " + format_rep_summary(stored))
    else:
        print("\n[4] STORED   skipped (--no-store)")

    # -- 5. figures + text summary ---------------------------------------------
    fig_path = write_figures(workout, series, profile, rec, reps, per_set, args.out)
    txt_path = os.path.join(args.out, f"end_to_end_{workout.workout_id}.txt")
    with open(txt_path, "w") as fh:
        fh.write(f"StanceSense-RT end-to-end demo - MM-Fit {workout.workout_id}\n")
        fh.write(RULE + "\n\n")
        fh.write("1. FUNCTIONAL HIP-ROTATION PROFILE\n   "
                 + format_profile(_profile_dict(profile)) + "\n   "
                 + profile_note(_profile_dict(profile)) + "\n")
        if warn:
            fh.write("   WARNING: " + warn + "\n")
        fh.write("\n")
        fh.write(f"2. RECOMMENDED STANCE\n   {rec.width_factor:.2f}x hip width, "
                 f"{rec.toe_out_deg:.0f} deg toe-out\n   {rec.rationale}\n\n")
        fh.write("3. SQUAT RUN (rule-based classifier)\n")
        for p in per_set:
            fh.write(f"   frames {p['start']}-{p['end']}: true {p['true_reps']}, "
                     f"counted {p['counted']}, confirmed {p['confirmed']}, "
                     f"best depth {p['max_depth']:.2f}\n")
        fh.write(f"   TOTAL true {true_total}, counted {counted}, "
                 f"confirmed {confirmed}, rejected {rejected}\n")
        fh.write(f"\n3b. SQUAT-BASED STANCE ADVICE\n   {format_squat_stance(squat_stance)}\n")
        fh.write(f"\n4. STORED session #{session_id} in {args.db}\n"
                 if session_id else "\n4. STORED skipped\n")
    print(f"\n[5] WROTE    {txt_path}")
    if fig_path:
        print(f"             {fig_path}")
    print("\nView it:     streamlit run scripts/app.py")
    print(RULE)


if __name__ == "__main__":
    main()
