"""Figures for the TARP report (docs/tarp_report/figures/).

Every chart is computed from saved results or cached pose data at run time:
  models/rule_classifier/{metrics_eval.json, results_eval.csv}   rule classifier
  data/processed_multiview/...                                     cached MediaPipe landmarks
  the Multi-View video zip                                         frames for the pose overlay
Diagrams (architecture, rotation geometry, state machine) use the thresholds in
the code. Run with the project venv (MediaPipe + OpenCV are needed for the overlay):

    .venv/Scripts/python.exe scripts/make_tarp_figures.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys
import tempfile
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)
OUT = os.path.join("docs", "tarp_report", "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 11})
BLUE, RED, GREEN, GREY, ORANGE, PURPLE = "#1f4e9a", "#c0392b", "#2e8b57", "#7f7f7f", "#e67e22", "#7d3c98"


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("->", name)


def box(ax, x, y, w, h, text, fc="#eef3fb", ec=BLUE, fs=9.5, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, text=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.2))
    if text:
        ax.text((x1 + x2) / 2 + 0.05, (y1 + y2) / 2, text, fontsize=8, color="#333333", va="center")


# ------------------------------------------------------------------ 3.1 architecture
def architecture():
    fig, ax = plt.subplots(figsize=(10, 6.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.6); ax.axis("off")
    box(ax, 1.4, 6.0, 2.3, 0.7, "Webcam / video\n(front view)", fc="#f4f4f4", ec=GREY)
    box(ax, 4.6, 6.0, 3.2, 0.7, "MediaPipe BlazePose\n33 world landmarks + visibility", fc="#fdf2e9", ec=ORANGE)
    box(ax, 8.2, 6.0, 2.9, 0.7, "Calibration\nhip width, femur, neutral shank")
    arrow(ax, 2.55, 6.0, 3.0, 6.0); arrow(ax, 6.2, 6.0, 6.75, 6.0)
    ax.text(0.15, 5.0, "Stage 1 - seated", fontsize=10, fontweight="bold", color=BLUE)
    box(ax, 2.0, 4.35, 3.4, 0.8, "Hip-rotation test (per leg)\nneutral hold, then two held extremes")
    box(ax, 5.6, 4.35, 2.9, 0.8, "Hip profile\nIR, ER, arc, bias, symmetry, pattern")
    box(ax, 8.6, 4.35, 2.4, 0.8, "Starting stance\nwidth x hip width, toe-out")
    arrow(ax, 8.2, 5.65, 8.2, 5.55); arrow(ax, 8.2, 5.55, 2.0, 4.75)
    arrow(ax, 3.7, 4.35, 4.15, 4.35); arrow(ax, 7.05, 4.35, 7.4, 4.35)
    ax.text(0.15, 3.35, "Stage 2 - standing", fontsize=10, fontweight="bold", color=BLUE)
    box(ax, 1.55, 2.65, 2.5, 0.8, "Stand-up gate\nknee >= 145 deg for 1 s")
    box(ax, 4.6, 2.65, 3.1, 0.8, "Rule-based squat classifier\nsmoothing, state machine,\n9-check squat score", fc="#eaf6ee", ec=GREEN)
    box(ax, 8.2, 2.65, 2.9, 0.8, "Squat analysis\ndepth, trunk lean, stance,\nknee tracking, symmetry")
    arrow(ax, 2.8, 2.65, 3.05, 2.65); arrow(ax, 6.15, 2.65, 6.75, 2.65)
    # starting stance -> stand up (stage 2 starts)
    ax.plot([8.6, 8.6, 1.55], [3.95, 3.3, 3.3], color="#333333", lw=1.2)
    arrow(ax, 1.55, 3.3, 1.55, 3.05)
    box(ax, 4.6, 1.05, 3.1, 0.75, "Rep count + verdict per rep\n(confirmed / rejected, score of 9)")
    box(ax, 8.2, 1.05, 2.9, 0.75, "Squat-based stance advice\nNARROW / MODERATE / WIDE")
    arrow(ax, 4.6, 2.25, 4.6, 1.43); arrow(ax, 8.2, 2.25, 8.2, 1.43)
    box(ax, 1.55, 1.05, 2.5, 0.75, "SQLite store\nStreamlit dashboard", fc="#f4f4f4", ec=GREY)
    arrow(ax, 3.05, 1.05, 2.8, 1.05)
    ax.plot([8.2, 8.2, 1.55], [0.67, 0.45, 0.45], color="#333333", lw=1.2)
    arrow(ax, 1.55, 0.45, 1.55, 0.67)
    ax.text(5.0, 0.12, "Orange: pre-trained pose network.  Green: the squat decision (rules, no training).  "
            "Blue: measurement and rule layers.", ha="center", fontsize=8.5, color="#333333")
    save(fig, "fig_architecture.png")


# ------------------------------------------------------------------ 3.2 pose overlay
def pose_overlay(clip_id="subject_002_squat_good_front"):
    import cv2
    import mediapipe as mp
    from stancesense.datasets.multiview_fitness import discover_clips, extract_clip
    from stancesense.squat.kinematics import knee_flexion
    from stancesense.common.types import LEFT, RIGHT
    zp = glob.glob("data/A Multi-View Raw Video Dataset*/*.zip")[0]
    clip = {os.path.splitext(c.filename)[0]: c for c in discover_clips(zp)}[clip_id]
    path = extract_clip(clip, tempfile.mkdtemp(prefix="tarpfig_"))
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    pose = mp.solutions.pose.Pose(model_complexity=1, smooth_landmarks=True)
    res = [pose.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames[:120]]
    knees = [((knee_flexion([[p.x, p.y, p.z] for p in r.pose_world_landmarks.landmark], LEFT)
               + knee_flexion([[p.x, p.y, p.z] for p in r.pose_world_landmarks.landmark], RIGHT)) / 2)
             if r.pose_world_landmarks else np.nan for r in res]
    k = np.array(knees)
    i_top = int(np.nanargmax(k[:40])); i_bot = int(np.nanargmin(k[i_top:i_top + 40])) + i_top
    i_mid = i_top + int(np.nanargmin(np.abs(k[i_top:i_bot + 1] - (k[i_top] + k[i_bot]) / 2)))
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 4.4))
    draw, style = mp.solutions.drawing_utils, mp.solutions.drawing_styles
    for ax, i, lab in zip(axs, (i_top, i_mid, i_bot), ("Standing", "Descending", "Bottom")):
        img = frames[i].copy()
        draw.draw_landmarks(img, res[i].pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=style.get_default_pose_landmarks_style())
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax.axis("off")
        ax.set_title(f"{lab}: knee angle {k[i]:.0f} deg", fontsize=10.5)
    save(fig, "fig_pose_overlay.png")


# ------------------------------------------------------------------ 3.3 rotation geometry
def shank_geometry():
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, deg, lab, col in zip(axs, (0, 30, -20), ("Neutral (0 deg)", "Internal rotation (+30 deg)",
                                                     "External rotation (-20 deg)"), (BLUE, GREEN, RED)):
        t = np.radians(deg)
        kx, ky = 0.0, 1.0
        ax_, ay_ = kx + np.sin(t), ky - np.cos(t)
        ax.plot([kx, kx], [ky, 0], ls="--", color="#b0b0b0", lw=1.2)
        ax.plot([kx, ax_], [ky, ay_], color=col, lw=4)
        ax.plot([kx - 0.25, kx], [ky + 0.35, ky], color="#555555", lw=6, solid_capstyle="round")
        ax.scatter([kx, ax_], [ky, ay_], s=70, color="#222222", zorder=3)
        ax.text(kx + 0.06, ky + 0.04, "knee", fontsize=12); ax.text(ax_ + 0.06, ay_ - 0.05, "ankle", fontsize=12)
        if deg:
            a = np.linspace(-np.pi / 2, -np.pi / 2 + t, 30)
            ax.plot(kx + 0.3 * np.cos(a), ky + 0.3 * np.sin(a), color=col, lw=1.5)
        ax.set_xlim(-0.9, 0.9); ax.set_ylim(-0.15, 1.45); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(lab, fontsize=13)
    save(fig, "fig_shank_geometry.png")


# ------------------------------------------------------------------ 3.4 state machine
def state_machine():
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.2); ax.axis("off")
    pos = {"UNREADY": (1.0, 3.2), "STANDING": (3.5, 3.2), "DESCENDING": (6.5, 3.2),
           "BOTTOM": (6.5, 1.0), "ASCENDING": (3.5, 1.0)}
    for s, (x, y) in pos.items():
        box(ax, x, y, 1.9, 0.6, s, fc="#eaf6ee" if s != "UNREADY" else "#f4f4f4",
            ec=GREEN if s != "UNREADY" else GREY, fs=9.5, bold=True)
    arrow(ax, 1.95, 3.2, 2.55, 3.2); ax.text(1.55, 3.75, "upright 0.15 s:\ntrunk <= 40, knee >= 135,\nhip >= 1.3 femur", fontsize=7.5)
    arrow(ax, 4.45, 3.2, 5.55, 3.2); ax.text(4.3, 3.62, "hip drop >= 0.12 femur and\nknee bend >= 10 deg, moving down", fontsize=7.5)
    arrow(ax, 6.5, 2.9, 6.5, 1.3); ax.text(6.6, 2.1, "downward speed\n< 0.15 femur/s", fontsize=7.5)
    arrow(ax, 5.55, 1.0, 4.45, 1.0); ax.text(4.5, 1.12, "rise >= 0.08 femur\nabove lowest point", fontsize=7.5, va="bottom")
    arrow(ax, 3.5, 1.3, 3.5, 2.9); ax.text(1.4, 2.0, "75% of the drop recovered and\nknee within 20 deg of standing:\ncycle closes -> scored (9 checks)", fontsize=7.5)
    ax.text(5.0, 0.05, "Any phase longer than 4 s aborts the cycle (back to UNREADY). Frames with mean leg visibility "
            "< 0.5 are skipped.", ha="center", fontsize=8)
    save(fig, "fig_state_machine.png")


# ------------------------------------------------------------------ 3.5 signal trace
def signal_trace(clip="data/processed_multiview/squat/good/subject_002_squat_good_front.npz"):
    from stancesense.squat.rule_classifier import RuleSquatClassifier
    d = np.load(clip)
    lm, vis = d["landmarks"], d["leg_vis"]
    clf = RuleSquatClassifier(float(d["femur"]), float(d["hip_w"]))
    T, K, Hh, Ph = [], [], [], []
    for i in range(len(lm)):
        clf.push(lm[i], i / 15.0, float(vis[i]))
        if clf.last:
            T.append(i / 15.0); K.append(clf.last["knee"]); Hh.append(clf.last["hip_height"]); Ph.append(clf.phase)
    clf.flush()
    T, K, Hh = np.array(T), np.array(K), np.array(Hh)
    sel = T <= 14.0
    fig, ax = plt.subplots(figsize=(10, 4.2))
    cols = {"DESCENDING": "#fdebd0", "BOTTOM": "#f5b7b1", "ASCENDING": "#d5f5e3"}
    for j in np.where(sel)[0][:-1]:
        if Ph[j] in cols:
            ax.axvspan(T[j], T[j + 1], color=cols[Ph[j]], lw=0)
    ax.plot(T[sel], K[sel], color=BLUE, lw=1.8, label="Knee angle, smoothed (blue, left axis)")
    ax.set_ylabel("Knee angle (deg)"); ax.set_xlabel("Time (s)")
    ax2 = ax.twinx()
    ax2.plot(T[sel], Hh[sel], color=RED, lw=1.5, label="Hip height above ankles (red, right axis)")
    ax2.set_ylabel("Hip height (femur lengths)")
    for c in clf.cycles:
        if c["t_end"] <= 14.0:
            ax.text((c["t_start"] + c["t_end"]) / 2, K[sel].max() + 4, f"{c['score']}/9", ha="center", fontsize=8.5,
                    color=GREEN if c["squat"] else RED)
    from matplotlib.patches import Patch
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ph = [Patch(color=v, label=f"{k.capitalize()} phase") for k, v in cols.items()]
    ax.legend(h1 + h2 + ph, l1 + l2 + [p.get_label() for p in ph], fontsize=8, loc="lower left", ncol=2, framealpha=0.95)
    ax.set_ylim(K[sel].min() - 45, K[sel].max() + 12)
    ax.set_title("Rule engine on a real front-view squat (Multi-View, person 2): phases and each rep's score",
                 fontsize=10.5)
    save(fig, "fig_signal_trace.png")


# ------------------------------------------------------------------ 5.x rule classifier results
def rule_results():
    m = json.load(open("models/rule_classifier/metrics_eval.json"))["metrics"]
    M = np.array([[m["TP"], m["FN"]], [m["FP"], m["TN"]]])
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.imshow(M / M.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            f = M[i, j] / M[i].sum()
            ax.text(j, i, f"{M[i, j]}\n({f * 100:.1f}% of row)", ha="center", va="center", fontsize=11,
                    color="white" if f > 0.5 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Squat", "Non-squat"]); ax.set_xlabel("Predicted")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Squat", "Non-squat"]); ax.set_ylabel("Actual")
    ax.set_title(f"Evaluation people, {m['videos']} videos\n(colour = share of the actual-class row)", fontsize=10)
    save(fig, "fig_confusion_rules.png")

    per_exercise(m)

    front = [r for r in rows_eval() if r["view"] == "front" and r["stance"]]
    order = ["NARROW", "MODERATE", "WIDE"]
    cur = Counter(r["stance"] for r in front); rec = Counter(r["recommendation"] for r in front)
    x = np.arange(3); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(x - w / 2, [cur[o] for o in order], w, color=GREY, label="Measured stance (grey)")
    ax.bar(x + w / 2, [rec[o] for o in order], w, color=BLUE, label="Recommended stance (blue)")
    for i, o in enumerate(order):
        ax.text(i - w / 2, cur[o] + 0.3, str(cur[o]), ha="center", fontsize=9)
        ax.text(i + w / 2, rec[o] + 0.3, str(rec[o]), ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(order); ax.set_ylabel("Squat videos")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"Front-view evaluation squats with confirmed reps (n={len(front)})", fontsize=10.5)
    save(fig, "fig_stance_front.png")


def rows_eval():
    return list(csv.DictReader(open("models/rule_classifier/results_eval.csv")))


def per_exercise(m):
    """Correct decisions per exercise on the evaluation participants, with the errors spelled out.

    Squat bar = share of squat videos found (recall); the others = share of that exercise's
    videos correctly rejected (specificity per exercise).
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    rows = rows_eval()
    names = {"abs": "Side bend\n(abs)", "back": "Bent-over row\n(back)", "bicep_curl": "Bicep curl",
             "push_up": "Push-up", "shoulder": "Shoulder\npress", "squat": "Squat", "tricep": "Triceps\nextension"}
    ex = ["squat"] + sorted({r["exercise"] for r in rows} - {"squat"})
    n = {e: sum(r["exercise"] == e for r in rows) for e in ex}
    ok = {e: sum(int(r["correct"]) for r in rows if r["exercise"] == e) for e in ex}
    acc = [ok[e] / n[e] * 100 for e in ex]
    overall = m["accuracy"] * 100
    x = np.array([0.0] + [1.0 + i for i in range(1, len(ex))])          # a gap after the squat bar

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(x, acc, width=0.7, color=[GREEN] + [BLUE] * (len(ex) - 1), zorder=2)
    for xi, e, a in zip(x, ex, acc):
        ax.text(xi, a + 0.5, f"{a:.1f}%\n{ok[e]} of {n[e]}", ha="center", va="bottom", fontsize=9, zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
        err = n[e] - ok[e]
        if e == "squat":
            lab = f"{err} missed"
        else:
            lab = "no errors" if err == 0 else f"{err} false\nalarm{'s' if err > 1 else ''}"
        ax.text(xi, 80.6, lab, ha="center", va="bottom", fontsize=8.5, color="white", fontweight="bold", zorder=3)
    ax.hlines(overall, x[0] - 0.6, x[-1] + 0.45, linestyles="--", lw=1.1, color="#555555", zorder=1)
    ax.text(x[-1] + 0.55, overall, f"overall\naccuracy\n{overall:.1f}%", va="center", fontsize=8.5, color="#555555")
    ax.axvline((x[0] + x[1]) / 2, color="#bbbbbb", lw=1, zorder=1)
    ax.set_xticks(x); ax.set_xticklabels([names[e] for e in ex], fontsize=9)
    ax.set_xlim(x[0] - 0.6, x[-1] + 1.3)
    ax.set_ylim(80, 105.5); ax.set_yticks([80, 85, 90, 95, 100]); ax.set_ylabel("Videos classified correctly (%)")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    # the axis starts at 80 %: mark the break so small differences are not read as large ones
    kw = dict(transform=ax.transAxes, color="black", clip_on=False, lw=1)
    for yy in (0.015, 0.04):
        ax.plot((-0.012, 0.012), (yy - 0.012, yy + 0.012), **kw)
    ax.legend(handles=[Patch(color=GREEN, label="Squat videos: share found (recall)"),
                       Patch(color=BLUE, label="Non-squat videos: share correctly rejected"),
                       Line2D([], [], ls="--", color="#555555", label="Overall accuracy")],
              loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=9, frameon=False)
    save(fig, "fig_per_exercise.png")


if __name__ == "__main__" and "--diagrams" in sys.argv:
    architecture(); state_machine(); raise SystemExit
if __name__ == "__main__" and "--results" in sys.argv:          # result charts only (no MediaPipe needed)
    rule_results(); raise SystemExit
if __name__ == "__main__":
    architecture()
    shank_geometry()
    state_machine()
    signal_trace()
    rule_results()
    pose_overlay()
    print("done")
