"""Smoke tests for the Streamlit app, driven through Streamlit's own AppTest.

These exist because the app's failure modes all look identical in a browser - a
blank or hanging page - and none of them raise anything you could read:

  * a `while True` capture loop pins the script thread, so no widget is clickable;
  * `st.rerun()` after every frame batch restarts the whole script forever;
  * importing TensorFlow during a render blocks the first paint until the
    websocket gives up ("Connection error").

The first two make `AppTest.run()` time out; the third is caught by the
subprocess test at the bottom.
"""
import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = "scripts/app.py"
TIMEOUT = 120


def _fresh():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    return at


def _labels(at):
    return [b.label for b in at.button]


def _kill(at):
    proc = at.session_state["proc"] if "proc" in at.session_state else None
    if proc is not None and proc.poll() is None:
        proc.terminate()
        proc.wait(timeout=30)


def test_the_app_loads_without_error():
    at = _fresh()
    assert not at.exception
    assert "Start" in _labels(at)


def test_every_tab_renders():
    """All four tabs are built on load; a broken one raises here."""
    at = _fresh()
    text = " ".join(m.value for m in at.markdown) + " ".join(
        s.value for s in at.subheader)
    for expected in ("Live assessment", "Stored assessments",
                     "How well does it work?", "Run the whole chain"):
        assert expected in text, f"missing tab content: {expected}"


def test_start_launches_the_assessment_process_and_flips_the_button():
    """The camera lives in its own process; Start spawns it, Stop kills it."""
    at = _fresh()
    [b for b in at.button if b.label == "Start"][0].click().run()
    try:
        assert not at.exception
        assert "Stop" in _labels(at), "the button must flip on the same interaction"
        proc = at.session_state["proc"]
        assert proc is not None and proc.poll() is None, "no live child process"
    finally:
        _kill(at)


def test_stop_terminates_the_process_and_start_comes_back():
    at = _fresh()
    [b for b in at.button if b.label == "Start"][0].click().run()
    proc = at.session_state["proc"]
    [b for b in at.button if b.label == "Stop"][0].click().run()
    assert not at.exception
    assert "proc" not in at.session_state
    proc.wait(timeout=30)                       # terminated, not orphaned
    assert "Start" in _labels(at)


def test_reset_terminates_the_process_too():
    at = _fresh()
    [b for b in at.button if b.label == "Start"][0].click().run()
    proc = at.session_state["proc"]
    [b for b in at.button if b.label == "Reset"][0].click().run()
    assert not at.exception
    assert "proc" not in at.session_state
    proc.wait(timeout=30)


def test_the_app_never_opens_the_camera_itself():
    """The camera belongs to the child process. If this page grabbed it too, the
    two would fight over the device and one would silently get no frames."""
    src = open("scripts/app.py", encoding="utf-8").read()
    assert "VideoCapture" not in src
    assert "import cv2" not in src


def test_the_rotation_caveat_is_always_on_screen():
    """The project's hard constraint must survive any UI refactor."""
    at = _fresh()
    warnings = " ".join(w.value for w in at.warning).lower()
    assert "functional hip rotation" in warnings
    assert "femoral or acetabular version" in warnings
    assert "qualitatively" in warnings
    assert "no angular-accuracy claim" in warnings


def test_the_page_renders_without_importing_tensorflow():
    """Rendering must never pull TensorFlow in.

    The import costs ~5 s. Doing it while the page renders blocks the FIRST
    paint, and Streamlit's websocket can drop before anything appears - the user
    sees "Connection error" over an empty page. The verifier is therefore warmed
    on a background thread and only resolved at the squat hand-off.

    Run in a subprocess because another test in the same session may already
    have imported TensorFlow.
    """
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        import sys, time
        from streamlit.testing.v1 import AppTest
        t0 = time.time()
        at = AppTest.from_file("scripts/app.py", default_timeout=120)
        at.run()
        print("TF_IMPORTED", "tensorflow" in sys.modules)
        print("ELAPSED", round(time.time() - t0, 2))
        print("EXC", bool(at.exception))
        """
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, timeout=300,
                          env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"})
    out = dict(line.split(maxsplit=1) for line in proc.stdout.splitlines()
               if line.startswith(("TF_IMPORTED", "ELAPSED", "EXC")))
    assert out.get("EXC") == "False", proc.stdout + proc.stderr[-2000:]
    assert out.get("TF_IMPORTED") == "False", (
        "the first render imported TensorFlow — the page will hang before it paints")
    assert float(out["ELAPSED"]) < 15.0, f"first render took {out['ELAPSED']}s"
