"""Drawing the assessment onto camera frames.

Used by the camera window (``scripts/live_assessment.py``) through
:class:`~stancesense.app.live_session.LiveSession`. OpenCV's Hershey font is ASCII-only: an em dash or a degree sign comes out as '?'.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..kinematics.rotation import IDX
from .controller import SQUAT

GREEN, AMBER, WHITE, RED = (80, 220, 80), (0, 200, 255), (255, 255, 255), (60, 60, 235)


def shade(frame, y0, y1):
    """Darken a band so text stays readable over any background."""
    band = frame[y0:y1]
    if band.size:
        frame[y0:y1] = (band * 0.35).astype(np.uint8)


def text(frame, s, org, scale=0.7, colour=WHITE, thick=2):
    cv2.putText(frame, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3,
                cv2.LINE_AA)
    cv2.putText(frame, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick,
                cv2.LINE_AA)


def draw_shank(frame, img_lm, side):
    """Highlight the segment being measured, with a vertical reference."""
    if img_lm is None or side is None:
        return
    h, w = frame.shape[:2]
    lm = img_lm.landmark
    ki, ai = IDX[side]["knee"], IDX[side]["ankle"]
    if lm[ki].visibility < 0.3 or lm[ai].visibility < 0.3:
        return
    knee = (int(lm[ki].x * w), int(lm[ki].y * h))
    ankle = (int(lm[ai].x * w), int(lm[ai].y * h))
    for yy in range(knee[1], min(h - 1, knee[1] + 220), 16):      # dashed vertical
        cv2.line(frame, (knee[0], yy), (knee[0], yy + 8), (190, 190, 190), 2, cv2.LINE_AA)
    cv2.line(frame, knee, ankle, GREEN, 5, cv2.LINE_AA)
    for pt in (knee, ankle):
        cv2.circle(frame, pt, 8, WHITE, -1)
        cv2.circle(frame, pt, 8, GREEN, 2)


def draw_hud(frame, state):
    h, w = frame.shape[:2]
    shade(frame, 0, 96)
    colour = GREEN if state.tracking else AMBER
    if not state.tracking:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), AMBER, 6)
        text(frame, "NOT MEASURING", (w - 250, h - 20), 0.7, AMBER, 2)
    text(frame, state.title or state.stage, (18, 34), 0.62, colour, 2)
    text(frame, state.cue, (18, 68), 0.72, WHITE, 2)
    if state.detail:
        shade(frame, 96, 124)
        text(frame, state.detail, (18, 116), 0.5, (200, 200, 200), 1)

    if state.progress > 0:                                        # hold-to-lock bar
        x0, y0, bw, bh = 18, h - 58, w - 36, 26
        cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + bh), WHITE, 2)
        cv2.rectangle(frame, (x0 + 3, y0 + 3),
                      (x0 + 3 + int((bw - 6) * state.progress), y0 + bh - 3), GREEN, -1)
        text(frame, state.progress_label, (x0, y0 - 10), 0.55, GREEN, 2)

    # Readouts sit above the bar's label when the bar is showing, so the two never overlap.
    base = h - 100 if state.progress > 0 else h - 76
    if state.stage == SQUAT:
        text(frame, f"reps {state.rep_count}   confirmed {state.confirmed}",
             (18, base), 0.7, GREEN, 2)
        text(frame, f"{state.phase}   depth {state.depth:.2f}", (18, base - 28), 0.55,
             (200, 200, 200), 1)
    elif state.angle is not None:
        peak = "" if state.peak is None else f"    peak {state.peak:+.0f}"
        text(frame, f"{state.angle:+.0f} deg{peak}", (18, base), 0.85, GREEN, 2)

    if state.countdown is not None:                               # get-ready countdown
        n = str(state.countdown)
        (nw, nh), _ = cv2.getTextSize(n, cv2.FONT_HERSHEY_SIMPLEX, 4.0, 8)
        (gw, _), _ = cv2.getTextSize("GET READY", cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        text(frame, "GET READY", ((w - gw) // 2, h // 2 - nh // 2 - 24), 0.9, AMBER, 2)
        text(frame, n, ((w - nw) // 2, h // 2 + nh // 2), 4.0, AMBER, 8)

    if state.just_locked:                                         # centred, shrunk to fit the frame
        msg = f"LOCKED  {state.just_locked}"
        scale = fit_scale(msg, w - 36, 0.9, 3)
        (mw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)
        text(frame, msg, ((w - mw) // 2, h // 2), scale, GREEN, 3)


def fit_scale(s, max_width, scale, thick):
    """Largest font scale <= ``scale`` at which ``s`` is at most ``max_width`` pixels wide."""
    while scale > 0.3 and cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0] > max_width:
        scale *= 0.95
    return scale


def infer_and_annotate(frame, pose, ctrl, now):
    """Run pose, advance the controller, and return a frame ready to show.

    ORDER MATTERS. Pose runs on the RAW, UNMIRRORED frame: mirroring the image
    before inference swaps MediaPipe's anatomical left/right labels (verified
    8/8 on a real recording), so the app would guide you through your "right"
    leg while measuring your left. The mirror is applied AFTER the skeleton and
    shank are drawn, so the preview still reads like a mirror, and the HUD text
    is drawn last so it is not written backwards.
    """
    img_lm, world = pose.infer(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    state = ctrl.update(world, now)
    pose.draw(frame, img_lm)
    # Only trace the shank when the leg is genuinely tracked - drawing it from
    # guessed joints would suggest a measurement that is not happening.
    show_side = state.side if (ctrl.stage != SQUAT and state.tracking) else None
    draw_shank(frame, img_lm, show_side)

    display = cv2.flip(frame, 1)          # mirror for a natural self-view
    draw_hud(display, state)
    return display, state
