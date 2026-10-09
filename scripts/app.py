"""StanceSense-RT — the whole thing as one app.

    streamlit run scripts/app.py

Four tabs:

  Assess       launches the live assessment and shows its result
  History      every stored assessment, with the profile trend across sessions
  Verification the rule-based squat classifier's evaluation (models/rule_classifier/)
  Offline demo run the whole chain on an MM-Fit recording when there is no camera

WHERE THE WEBCAM LIVES
----------------------
NOT in the browser.  Streaming frames through Streamlit means re-sending a JPEG
over a websocket per frame and re-laying-out the page around it, which flickers
and drops to a few frames a second.  Pressing Start therefore launches
`scripts/live_assessment.py` as a separate process, which owns the camera and
draws a native OpenCV window at full frame rate.  This page stays the control
panel: it polls a small JSON heartbeat for progress and reads the result out of
the SQLite store when the window closes.
"""
from __future__ import annotations

import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import streamlit as st

# Only the store and the pure formatting helpers are imported here. Pose
# (MediaPipe) and OpenCV belong to the assessment process; pulling either in
# would slow the first paint for no benefit.
from stancesense.storage import Store, DEFAULT_DB
from stancesense.ui import (
    format_profile, format_stance, format_rep_summary, format_squat_stance, rep_table,
    profile_note, depth_verdict, arc_warning,
)

DB = os.environ.get("STANCESENSE_DB", DEFAULT_DB)
MODELS = "models"

ROTATION_CAVEAT = (
    "StanceSense-RT measures **functional hip rotation** from video — a contactless "
    "analogue of the clinical Craig's test. It does **not** image the acetabulum and "
    "does **not** measure true femoral or acetabular version, which needs X-ray/CT/MRI. "
    "Rotation is reported **qualitatively**; no angular-accuracy claim is made."
)

_ICON = os.path.join(os.path.dirname(__file__), "..", "assets", "stancesense.png")
# The desktop app's window and taskbar take this icon (scripts/desktop_app.py).
st.set_page_config(page_title="StanceSense-RT",
                   page_icon=_ICON if os.path.exists(_ICON) else "\N{RUNNER}", layout="wide")


# --------------------------------------------------------------------- helpers

def verifier_status() -> str:
    """What judges the squat reps: the rule-based classifier in the camera window."""
    return ("**rule-based biomechanical classifier** — each rep is checked against 9 movement "
            "rules (no trained model). Squat facing the camera.")


STATUS_FILE = os.path.join(".streamlit", "live_status.json")


def launch_assessment(do_squat: bool, subject: str, target_reps: int = 5):
    """Start the native assessment window as a separate process."""
    import subprocess
    os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
    for path in (STATUS_FILE, STATUS_FILE + ".tmp"):
        try:
            os.remove(path)
        except OSError:
            pass
    cmd = [sys.executable, "scripts/live_assessment.py",
           "--db", DB, "--subject", subject,
           "--status-file", STATUS_FILE]
    cmd += ["--reps", str(int(target_reps))]
    if not do_squat:
        cmd.append("--no-squat")
    return subprocess.Popen(cmd, env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"})


def read_status():
    """The heartbeat the assessment process writes. None until it appears."""
    try:
        with open(STATUS_FILE) as fh:
            return json.load(fh)
    except Exception:
        return None


def assessment_alive() -> bool:
    proc = st.session_state.get("proc")
    return proc is not None and proc.poll() is None


def load_json(name):
    path = os.path.join(MODELS, name)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def reset_assessment():
    """Forget the last run. Does not touch the stored session."""
    proc = st.session_state.pop("proc", None)
    if proc is not None and proc.poll() is None:
        proc.terminate()
    for key in ("last_session", "last_video", "launch_error"):
        st.session_state.pop(key, None)


def _on_start(do_squat: bool, subject: str, target_reps: int):
    """Launch the native window. Runs as an on_click callback, so the button
    label reflects the new state on the same interaction."""
    if assessment_alive():
        return
    try:
        st.session_state.proc = launch_assessment(do_squat, subject, target_reps)
        st.session_state.pop("launch_error", None)
        st.session_state.pop("last_session", None)
    except Exception as exc:                       # missing script, no python, ...
        st.session_state.launch_error = str(exc)


def _on_stop():
    proc = st.session_state.pop("proc", None)
    if proc is not None and proc.poll() is None:
        proc.terminate()


# ------------------------------------------------------------------ ASSESS TAB

def render_assess():
    st.subheader("Live assessment")
    left, right = st.columns([3, 2])

    with right:
        st.markdown("**Setup**")
        subject = st.text_input("Subject id", value="local")
        do_squat = st.checkbox("Include the squat check", value=True)
        target_reps = st.number_input(
            "Squat reps to collect", 0, 30, 5, disabled=not do_squat,
            help="The set ends by itself after this many CONFIRMED reps. "
                 "0 = keep going until you press SPACE in the camera window.")
        st.caption(f"Verifier: {verifier_status()}")
        in_docker = os.environ.get("STANCESENSE_IN_DOCKER") == "1"
        if in_docker:
            # Docker Desktop on Windows cannot pass a webcam into a Linux container, so
            # Start would only fail with "could not open camera source 0".
            st.warning("Running in Docker: a container cannot reach your webcam, so Start is "
                       "turned off here. Run the live assessment on Windows instead "
                       "(`.venv/Scripts/python.exe scripts/live_assessment.py`); it saves to "
                       "`data/stancesense.db`, which this dashboard shows under **History**.")

        alive = assessment_alive()
        c1, c2 = st.columns(2)
        if alive:
            c1.button("Stop", type="secondary", use_container_width=True,
                      on_click=_on_stop)
        else:
            c1.button("Start", type="primary", use_container_width=True,
                      on_click=_on_start, disabled=in_docker,
                      args=(do_squat, subject, int(target_reps)))
        c2.button("Reset", use_container_width=True, on_click=reset_assessment)

        st.markdown("---")
        st.markdown(
            "**How it goes**\n\n"
            "1. Sit with your lower legs hanging free, fully in frame.\n"
            "2. Before every step a **4-second GET READY countdown** tells you what to do "
            "next. Nothing is measured while it runs.\n"
            "3. Point the foot straight ahead and hold still. That sets your neutral.\n"
            "4. Swing the lower leg as far as it goes one way and **hold still at your "
            "limit** for 2 s. It only locks at the furthest point you reached, so take "
            "your time getting there.\n"
            "5. Countdown again, then the other way. Repeat both steps for the second leg.\n"
            "6. **Stand up.** Counting is held back until you are upright, so "
            "standing up is never mistaken for a rep.\n"
            "7. Squat. The set ends by itself once the target is confirmed.\n\n"
            "In the window: **ESC** quit · **S** skip a leg · **SPACE** finish the "
            "squat set early."
        )

    with left:
        if st.session_state.get("launch_error"):
            st.error(f"Could not start the assessment: "
                     f"{st.session_state['launch_error']}")
        _live_panel()


@st.fragment(run_every=0.5)
def _live_panel():
    """Progress mirror for the native window.

    Only a small JSON file is read here - no frames cross the websocket, which
    is the whole point of running the camera in its own process.
    """
    alive = assessment_alive()
    status = read_status()

    if not alive and status is None:
        st.info(
            "Press **Start**. A separate camera window opens — follow the cues "
            "there.\n\nThe webcam runs locally at full frame rate rather than "
            "streaming into this page, so it does not flicker."
        )
        _show_last_result()
        return

    if alive:
        st.success("Assessment window is open — follow the on-screen cues there.")
    if status:
        if status.get("error"):
            st.error(status.get("cue", "The assessment ended early."))
        elif not status.get("done"):
            if status.get("tracking") is False:
                st.warning("Not measuring — the leg is not fully in frame.")
            st.markdown(f"### {status.get('cue', '')}")
            if status.get("detail"):
                st.caption(status["detail"])
            st.progress(float(np.clip(status.get("progress") or 0.0, 0.0, 1.0)))
            cols = st.columns(3)
            cols[0].metric("Stage", status.get("stage", "—"))
            cols[1].metric("Leg", status.get("side") or "—")
            if status.get("stage") == "SQUAT":
                cols[2].metric("Reps", f"{status.get('confirmed', 0)}"
                                       f" / {status.get('reps', 0)}")
            elif status.get("angle") is not None:
                cols[2].metric("Live angle", f"{status['angle']:+.0f} deg")
        else:
            sid = status.get("session_id")
            if sid is not None:
                st.session_state["last_session"] = sid
            if status.get("video"):
                st.session_state["last_video"] = status["video"]
            st.success("Assessment complete.")

    if not alive:
        _show_last_result()


def _show_last_result():
    """Read the finished run back out of the store."""
    sid = st.session_state.get("last_session")
    if sid is None:
        return
    if not os.path.exists(DB):
        return
    with Store(DB) as store:
        profile = store.profile_for(sid)
        stance = store.stance_for(sid)
        summary = store.rep_summary(sid)
        reps = store.reps_for(sid)
        squat_run = store.squat_run_for(sid)
    if profile is None:
        return

    st.markdown("---")
    a, b, c, d = st.columns(4)
    a.metric("Pattern", profile["pattern"])
    b.metric("Stance width", f"{stance['width_factor']:.2f}x" if stance else "—")
    c.metric("Toe-out", f"{stance['toe_out_deg']:.0f} deg" if stance else "—")
    d.metric("Reps confirmed", f"{summary['confirmed']} / {summary['counted']}")

    st.markdown("#### Your hip profile")
    st.write(format_profile(profile))
    st.caption(profile_note(profile))
    if (warn := arc_warning(profile)):
        st.error(warn)
    cols = st.columns(4)
    for col, label, key in zip(cols, ("IR left", "ER left", "IR right", "ER right"),
                               ("ir_left", "er_left", "ir_right", "er_right")):
        v = profile.get(key)
        col.metric(label, "—" if v is None else f"{v:.0f} deg")
    st.warning(ROTATION_CAVEAT)

    if stance:
        st.markdown("#### Recommended stance")
        st.write(format_stance(stance))
        st.caption(stance["rationale"])

    if reps:
        st.markdown("#### Squat check")
        st.write(format_rep_summary(summary))
        st.dataframe(rep_table(reps), use_container_width=True, hide_index=True)
    if squat_run and squat_run.get("recommendation"):
        st.markdown("#### Squat-based stance advice")
        m1, m2, m3 = st.columns(3)
        m1.metric("Measured stance", f"{squat_run['current_stance']} ({squat_run['stance_width']:.2f}x)")
        m2.metric("Recommended", squat_run["recommendation"])
        m3.metric("Depth ratio", f"{squat_run['depth_ratio']:.2f}")
        st.write(format_squat_stance(squat_run))
        st.caption("Measured from the confirmed squats: stance width, depth, trunk lean and knee "
                   "tracking. Advice is only given from a front-facing camera.")
    elif squat_run is not None:
        st.caption("No squat-based stance advice: no squat was confirmed.")

    video = st.session_state.get("last_video")
    if video and os.path.exists(video):
        st.caption(f"Recording: `{os.path.abspath(video)}`")
    st.caption(f"Saved as session #{sid} — also in the **History** tab.")


# ----------------------------------------------------------------- HISTORY TAB

def render_history():
    st.subheader("Stored assessments")
    if not os.path.exists(DB):
        st.info(f"No assessments stored yet (`{DB}`). Run one in **Assess**, or "
                f"generate one from a recording in **Offline demo**.")
        return
    with Store(DB) as store:
        sessions = store.sessions()
        if not sessions:
            st.info("The store is empty.")
            return
        labels = {f"#{s['id']}  {s['created_at']}  [{s['source']}]  {s['subject_id']}": s["id"]
                  for s in sessions}
        chosen = st.selectbox(f"{len(sessions)} session(s)", list(labels))
        sid = labels[chosen]

        profile = store.profile_for(sid)
        stance = store.stance_for(sid)
        summary = store.rep_summary(sid)
        reps = store.reps_for(sid)
        squat_run = store.squat_run_for(sid)
        hist = store.profile_history()

    a, b, c, d = st.columns(4)
    a.metric("Pattern", (profile or {}).get("pattern", "—"))
    b.metric("Stance width", f"{stance['width_factor']:.2f}x" if stance else "—")
    c.metric("Toe-out", f"{stance['toe_out_deg']:.0f}°" if stance else "—")
    d.metric("Reps confirmed", f"{summary['confirmed']} / {summary['counted']}")

    if profile:
        st.write(format_profile(profile))
        st.caption(profile_note(profile))
        if (warn := arc_warning(profile)):
            st.error(warn)
    st.warning(ROTATION_CAVEAT)

    if stance:
        st.markdown("#### Recommended stance")
        st.write(format_stance(stance))
        st.caption(stance["rationale"])

    st.markdown("#### Squat run")
    st.write(format_rep_summary(summary))
    if reps:
        rows = rep_table(reps)
        st.dataframe(rows, use_container_width=True, hide_index=True)
        depths = [r["depth"] for r in rows]
        st.bar_chart({"max depth ratio": depths})
        best = summary["max_depth"]
        st.caption(f"1.0 ≈ thighs parallel. Best rep ({summary.get('depth_from', 'all reps')}): "
                   f"{best:.2f} ({depth_verdict(best)}).")
    if squat_run and squat_run.get("recommendation"):
        st.markdown("#### Squat-based stance advice")
        m1, m2, m3 = st.columns(3)
        m1.metric("Measured stance", f"{squat_run['current_stance']} ({squat_run['stance_width']:.2f}x)")
        m2.metric("Recommended", squat_run["recommendation"])
        m3.metric("Depth ratio", f"{squat_run['depth_ratio']:.2f}")
        st.write(format_squat_stance(squat_run))
        st.caption("Measured from the confirmed squats: stance width, depth, trunk lean and knee "
                   "tracking. Advice is only given from a front-facing camera.")
    elif squat_run is not None:
        st.caption("No squat-based stance advice: no squat was confirmed.")

    st.markdown("#### Profile history")
    if len(hist) > 1:
        oldest_first = list(reversed(hist))
        st.line_chart({
            "IR max (deg)": [h["ir_max"] for h in oldest_first],
            "ER max (deg)": [h["er_max"] for h in oldest_first],
            "total arc (deg)": [h["total_arc"] for h in oldest_first],
        })
    else:
        st.info("Only one profile stored — run another to see a trend.")


# ------------------------------------------------------------ VERIFICATION TAB

def _front_only(csv_path):
    """Confusion counts for the front-camera videos of an evaluation CSV."""
    try:
        rows = [r for r in csv.DictReader(open(csv_path)) if r["view"] == "front"]
    except OSError:
        return None
    if not rows:
        return None
    sq = lambda r: r["ground_truth"] == "squat"
    pr = lambda r: r["prediction"] == "squat"
    tp = sum(sq(r) and pr(r) for r in rows); fn = sum(sq(r) and not pr(r) for r in rows)
    fp = sum(not sq(r) and pr(r) for r in rows); tn = sum(not sq(r) and not pr(r) for r in rows)
    return dict(videos=len(rows), TP=tp, FN=fn, FP=fp, TN=tn,
                accuracy=(tp + tn) / len(rows), recall=tp / max(tp + fn, 1),
                specificity=tn / max(tn + fp, 1), precision=tp / max(tp + fp, 1))


def render_verification():
    st.subheader("How well does it work?")
    rc = load_json(os.path.join("rule_classifier", "metrics_eval.json"))
    st.markdown("#### Active squat classifier: rule-based biomechanics (no training)")
    if not rc:
        st.info("No evaluation yet — run `python scripts/eval_rule_classifier.py --split eval`.")
    else:
        m = rc["metrics"]
        st.caption(f"Rules were set on {len(rc['development_people'])} development people and frozen; "
                   f"these numbers are one run on the other {len(rc['evaluation_people'])} people "
                   f"({m['videos']} videos: {m['squat']} squat, {m['non_squat']} non-squat). "
                   f"Read from `models/rule_classifier/metrics_eval.json`.")
        cols = st.columns(5)
        for col, (lab, key) in zip(cols, (("Accuracy", "accuracy"), ("Precision", "precision"),
                                          ("Recall", "recall"), ("F1", "f1"),
                                          ("Balanced acc.", "balanced_accuracy"))):
            col.metric(lab, f"{m[key]*100:.1f}%")
        st.dataframe([{"": "Actual squat", "Predicted squat": m["TP"], "Predicted non-squat": m["FN"]},
                      {"": "Actual non-squat", "Predicted squat": m["FP"], "Predicted non-squat": m["TN"]}],
                     use_container_width=True, hide_index=True)
        st.caption(f"Specificity {m['specificity']*100:.1f}%, false positive rate "
                   f"{m['false_positive_rate']*100:.1f}%, false negative rate {m['false_negative_rate']*100:.1f}%.")
        fr = _front_only(os.path.join(MODELS, "rule_classifier", "results_eval.csv"))
        if fr:
            st.caption(f"Front camera only (how the app is used): {fr['videos']} videos, accuracy "
                       f"{fr['accuracy']*100:.1f}%, recall {fr['recall']*100:.1f}%, specificity "
                       f"{fr['specificity']*100:.1f}% (TP {fr['TP']}, FN {fr['FN']}, FP {fr['FP']}, TN {fr['TN']}).")
        if rc.get("failures"):
            with st.expander(f"The {len(rc['failures'])} wrongly classified videos and why"):
                st.dataframe([{"video": f["video"], "truth": "squat" if f["truth"] else "non-squat",
                               "reason": f["reason"]} for f in rc["failures"]],
                             use_container_width=True, hide_index=True)

    dup = load_json(os.path.join("..", "data", "processed_multiview", "duplicates.json"))
    if dup:
        with st.expander("Data integrity checks"):
            st.markdown(f"The video archive holds {dup['n_excluded']} redundant or conflicting "
                        f"copies across {dup['n_duplicate_groups']} duplicate recordings "
                        f"(some filed under two different people). They are excluded before "
                        f"the development / evaluation split: {dup['reasons']}")

    st.warning(ROTATION_CAVEAT)


# ------------------------------------------------------------ OFFLINE DEMO TAB

def render_offline():
    st.subheader("Run the whole chain on a recording")
    st.caption("No camera needed. Replays a public MM-Fit recording in `data/mm-fit/` through "
               "the chain with the app's rule-based squat classifier. The rules were set on "
               "Multi-View people, never on MM-Fit, so this is a cross-dataset demonstration, "
               "not a benchmark. The demo is not saved to History.")

    from stancesense.datasets.mmfit import discover_workouts
    found = discover_workouts("data/mm-fit")
    ids = sorted(os.path.basename(f) if not isinstance(f, tuple) else f[2] for f in found)
    if not ids:
        st.error("No MM-Fit workouts found under `data/mm-fit/`.")
        return

    default = ids.index("w19") if "w19" in ids else 0
    wid = st.selectbox("Workout", ids, index=default)

    if st.button("Run it", type="primary"):
        import subprocess
        with st.spinner(f"Running the full pipeline on {wid}…"):
            proc = subprocess.run(
                [sys.executable, "scripts/demo_end_to_end_mmfit.py",
                 "--workout", wid, "--no-store"],
                capture_output=True, text=True,
                env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"},
            )
        if proc.returncode != 0:
            st.error("The demo failed:")
            st.code(proc.stderr[-3000:])
            return
        st.code("\n".join(l for l in proc.stdout.splitlines()
                          if "oneDNN" not in l and "cpu_feature" not in l))
        fig = os.path.join("demo_output", f"end_to_end_{wid}.png")
        if os.path.exists(fig):
            st.image(fig, use_column_width=True)
        st.caption("Demo run — not stored in History.")


# ---------------------------------------------------------------------- layout

st.title("StanceSense-RT")
st.caption("Functional hip-rotation profiling → personalised squat stance → verified squat run")

tab_assess, tab_history, tab_verif, tab_offline = st.tabs(
    ["Assess", "History", "Verification", "Offline demo"])
with tab_assess:
    render_assess()
with tab_history:
    render_history()
with tab_verif:
    render_verification()
with tab_offline:
    render_offline()
