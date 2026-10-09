"""Guided live assessment in a native OpenCV window.

Run directly, or let `scripts/app.py` launch it when you press Start.

    python scripts/live_assessment.py
    python scripts/live_assessment.py --no-squat --db data/stancesense.db

WHY A NATIVE WINDOW
-------------------
Streaming frames into a browser through Streamlit means re-sending a JPEG over a
websocket for every frame and letting the page re-layout around it, which flickers
and throttles to a few frames a second.  Capture and preview therefore happen
HERE, in a plain OpenCV window at full frame rate; the dashboard keeps the
controls, the results and the history.  The two talk through a small JSON status
file and the SQLite store.

FLOW  (mirrors AssessmentController)
    CALIBRATE -> per leg (NEUTRAL -> EXTREME_A -> EXTREME_B) -> SQUAT -> saved

Keys:  ESC quit    S skip the current leg    SPACE finish the squat set early
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2

from stancesense.app.hud import (  # noqa: F401  (re-exported: tests import them from here)
    GREEN, AMBER, WHITE, RED, shade, text, draw_shank, draw_hud, infer_and_annotate,
)
from stancesense.app.live_session import LiveSession
from stancesense.app.recorder import CODECS, Recorder  # noqa: F401
from stancesense.ingestion.video_stream import VideoStream
from stancesense.kinematics.pose_estimator import PoseEstimator
from stancesense.storage import DEFAULT_DB

WINDOW = "StanceSense-RT  -  ESC quit   S skip leg   SPACE finish squat set"


# --------------------------------------------------------------- status handoff

class Status:
    """Tiny JSON heartbeat the dashboard polls. Never raises."""

    def __init__(self, path: str | None):
        self.path = path
        self._last = 0.0

    def write(self, **fields):
        if not self.path:
            return
        now = time.monotonic()
        if now - self._last < 0.2 and not fields.get("done"):
            return                                  # at most ~5 writes a second
        self._last = now
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(fields, fh, default=float)
            os.replace(tmp, self.path)              # atomic, never half-read
        except Exception:
            pass


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--subject", default="local")
    ap.add_argument("--no-squat", action="store_true")
    ap.add_argument("--reps", type=int, default=5,
                    help="finish the squat set after this many CONFIRMED reps "
                         "(0 = run until you press SPACE)")
    ap.add_argument("--countdown", type=float, default=4.0,
                    help="seconds of GET READY instructions before each hip-rotation step "
                         "(nothing is measured meanwhile; 0 turns it off)")
    ap.add_argument("--hold", type=float, default=2.0,
                    help="seconds to hold still at your limit before an extreme locks")
    ap.add_argument("--no-store", action="store_true")
    ap.add_argument("--no-record", action="store_true",
                    help="do not save an annotated video of the assessment")
    ap.add_argument("--codec", choices=sorted(CODECS), default="mjpg",
                    help="mjpg (.avi, opens in any player - default) or "
                         "mp4v (.mp4, ~2x smaller but writes FMP4, which Windows "
                         "Photos and Media Player usually refuse)")
    ap.add_argument("--status-file", default="", help="JSON heartbeat for the dashboard")
    args = ap.parse_args()

    status = Status(args.status_file or None)
    status.write(stage="STARTING", cue="Opening the camera…", progress=0.0, done=False)

    pose = PoseEstimator("config/default.yaml")
    session = LiveSession(pose, do_squat=not args.no_squat, target_reps=args.reps,
                          countdown_seconds=args.countdown, hold_seconds=args.hold)

    try:
        stream = VideoStream("config/default.yaml").open()
    except RuntimeError as exc:
        print(f"[camera] {exc}")
        status.write(stage="ERROR", cue=str(exc), progress=0.0, done=True, error=True)
        return 2

    recorder = Recorder(not args.no_record, args.codec)
    if recorder.path:
        print(f"recording -> {recorder.path}")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 960, 720)
    print("Assessment window open. Follow the on-screen cues. ESC quits.")

    try:
        while True:
            ok, frame, _ = stream.read()
            if not ok:
                break
            frame, state = session.step(frame, time.monotonic())
            if state.just_locked:
                print(f"  [{state.stage}] {state.just_locked}")

            recorder.write(frame)
            cv2.imshow(WINDOW, frame)

            status.write(stage=session.stage, side=state.side, cue=state.cue,
                         detail=state.detail, progress=round(state.progress, 3),
                         angle=state.angle, reps=state.rep_count,
                         confirmed=state.confirmed, tracking=state.tracking,
                         countdown=state.countdown, done=False)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:                                   # ESC
                break
            if key in (ord("s"), ord("S")):
                session.skip_leg()
            if key == 32:                                   # SPACE ends the squat set
                session.finish_set()
            if session.done:
                break
    finally:
        stream.release()
        cv2.destroyAllWindows()
        video_path = recorder.close()

    return _finish(session, args, status, video_path)


def _finish(session, args, status, video_path=None) -> int:
    if not session.complete:
        print("\nAssessment incomplete - nothing was saved.")
        status.write(stage="ABORTED", cue="Assessment incomplete", progress=0.0,
                     done=True, error=True)
        return 1

    res = session.result()
    print("\n=== HIP PROFILE ===")
    print(res["profile_text"])
    print(res["profile_note"])
    print("\n=== RECOMMENDED STANCE ===")
    print(res["stance_text"])
    if res["squat_text"]:
        print("\n=== SQUAT CHECK ===")
        print(res["squat_text"])
    if res["squat_stance"]:
        s = res["squat_stance"]
        print(f"Measured stance {s['current_stance']} ({s['stance_width_hip_widths']}x hip width) -> "
              f"recommend {s['recommendation']}: {s['reason']}")

    session_id = None
    if not args.no_store:
        session_id = session.save(args.db, subject=args.subject, source="live")
        print(f"\nstored session #{session_id} -> {args.db}")

    status.write(stage="DONE", cue="Assessment complete", progress=1.0, done=True,
                 session_id=session_id, video=video_path, pattern=res["pattern"],
                 width_factor=res["width_factor"], toe_out_deg=res["toe_out_deg"],
                 reps=res["reps"], confirmed=res["confirmed"],
                 squat_stance=res["squat_stance"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
