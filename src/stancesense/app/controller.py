"""Headless driver for the whole guided assessment.

This owns the *logic* of the app - what stage we are in, what the user should be
told to do right now, when a reading is locked in - with no Streamlit, no OpenCV
and no MediaPipe imports.  The UI layer only has to push landmarks in and render
the :class:`UiState` that comes back, which keeps the flow unit-testable and
means the same controller can drive a Streamlit page, an OpenCV window, or a
test.

STAGES
------
    CALIBRATE -> per leg (NEUTRAL -> EXTREME_A -> EXTREME_B) -> REVIEW
              -> SQUAT (optional) -> DONE

GET-READY COUNTDOWN AND LOCKING AT THE REAL EXTREME
---------------------------------------------------
Before each measuring step (each leg's neutral and both extremes) the user gets
``countdown_seconds`` (default 4 s) of instructions with a countdown; nothing is
measured during it. An extreme then locks only while the reading stays within
``peak_tol_deg`` of the furthest point reached in that step and is held still for
``hold_seconds`` (default 2 s). Before this, a slow swing towards the limit could
lock part-way, and the next step started measuring the instant a reading locked.

HOLD-TO-LOCK IS TIME-BASED, NOT FRAME-BASED
-------------------------------------------
Counting frames would silently assume a frame rate. The camera runs at whatever
rate it manages, so every hold here is measured in SECONDS of wall clock.  The caller passes the timestamp, so
tests can drive it deterministically without sleeping.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import math

import numpy as np

from ..common.types import LEFT, RIGHT, HipProfile, StanceRec
from ..kinematics.rotation import hip_rotation_angle, IDX
from ..squat.kinematics import knee_flexion
from ..squat.rule_classifier import facing_angle, FRONT_FACING_MAX_DEG
from ..profiling.profiler import build_hip_profile
from ..recommendation.stance import recommend_stance

# --- stages -----------------------------------------------------------------
CALIBRATE = "CALIBRATE"
NEUTRAL = "NEUTRAL"
EXTREME_A = "EXTREME_A"
EXTREME_B = "EXTREME_B"
REVIEW = "REVIEW"
SQUAT = "SQUAT"
DONE = "DONE"


@dataclass
class UiState:
    """Everything the UI needs to draw one frame. Pure data."""
    stage: str
    side: Optional[str] = None
    title: str = ""
    cue: str = ""                     # the one instruction to follow right now
    detail: str = ""                  # secondary line
    progress: float = 0.0             # 0..1, fills the hold bar
    progress_label: str = ""
    angle: Optional[float] = None     # live neutral-corrected rotation, degrees
    peak: Optional[float] = None      # best reading this stage
    tracking: bool = True             # False when the person is out of frame
    just_locked: str = ""             # set on the frame a reading is captured
    countdown: Optional[int] = None   # whole seconds left of a get-ready countdown
    # squat stage
    rep_count: int = 0
    confirmed: int = 0
    phase: str = ""
    depth: float = 0.0
    p_squat: Optional[float] = None


@dataclass
class _Hold:
    """Time-based steadiness accumulator."""
    seconds_required: float
    tol_deg: float
    anchor: Optional[float] = None
    started: Optional[float] = None
    samples: list = field(default_factory=list)

    def reset(self):
        self.anchor = None
        self.started = None
        self.samples = []

    def update(self, value: float, now: float, eligible: bool) -> float:
        """Feed one reading; return progress 0..1."""
        if not eligible:
            self.reset()
            return 0.0
        if self.anchor is None or abs(value - self.anchor) > self.tol_deg:
            # first eligible sample, or the user moved somewhere new: restart here
            self.anchor, self.started, self.samples = value, now, [value]
            return 0.0
        self.anchor = 0.85 * self.anchor + 0.15 * value
        self.samples.append(value)
        return min(1.0, (now - self.started) / self.seconds_required)

    def locked_value(self) -> float:
        """Robust centre of the held samples (median rejects jitter/overshoot)."""
        return float(np.median(self.samples)) if self.samples else 0.0


def landmark_visibility(lm, index: int) -> float:
    """MediaPipe's confidence that a joint is really where it says.

    Plain arrays (tests, MM-Fit) carry no visibility, so they count as visible.
    """
    o = lm[index]
    return float(getattr(o, "visibility", 1.0))


def leg_visible(lm, side: str, min_vis: float) -> bool:
    """Are the hip, knee and ankle of one leg actually being seen?

    MediaPipe returns all 33 landmarks whenever it finds a person, INCLUDING
    invented positions for joints that are out of frame - only the visibility
    score gives that away. Without this check the rotation geometry happily
    measures a hallucinated shank and the hold-to-lock can capture a reading
    before the leg is even on camera.
    """
    return all(landmark_visibility(lm, IDX[side][joint]) >= min_vis
               for joint in ("hip", "knee", "ankle"))


def _as_points(lm):
    """Accept MediaPipe landmark objects or a plain (33, 3) array."""
    if hasattr(lm, "landmark"):
        lm = lm.landmark
    if hasattr(lm[0], "x"):
        return [[o.x, o.y, o.z] for o in lm]
    return lm


class AssessmentController:
    """Drives one full assessment. Feed it landmarks; render what it returns.

    Parameters
    ----------
    calibrator : a :class:`~stancesense.kinematics.calibrator.Calibrator`.
    hold_seconds : how long a reading must be held steady before it locks.
    min_move_deg : a reading must be this far from neutral to count as an extreme.
    tol_deg : "steady" means staying within this many degrees.
    do_squat : include the squat-verification stage after the recommendation.
    min_visibility : a joint below this MediaPipe visibility is treated as not
        seen; readings are refused until the whole leg is in frame.
    target_reps : finish the squat set automatically after this many CONFIRMED
        reps. 0 means run until the user ends it.
    stand_deg : knee angle above which the lifter counts as upright. 145 deg, not
        155: MediaPipe reads some people's straight standing knee as low as
        151-154 deg (3 of 25 Multi-View people), and a 155 gate never let them start.
    """

    def __init__(self, calibrator, hold_seconds: float = 2.0,
                 min_move_deg: float = 5.0, tol_deg: float = 4.0,
                 neutral_seconds: float = 1.0, neutral_tol_deg: float = 4.0,
                 do_squat: bool = True, target_reps: int = 5,
                 stand_deg: float = 145.0, stand_seconds: float = 1.0,
                 min_visibility: float = 0.6, countdown_seconds: float = 4.0,
                 peak_tol_deg: float = 3.0):
        self.calib = calibrator
        self.countdown_seconds = float(countdown_seconds)
        self.peak_tol_deg = float(peak_tol_deg)
        self._countdown_until: Optional[float] = None   # end of the running get-ready countdown
        self._countdown_done = False                     # countdown finished for the current step
        self.min_move_deg = min_move_deg
        self.do_squat = do_squat
        self.target_reps = int(target_reps)
        self.min_visibility = float(min_visibility)
        self.stand_deg = stand_deg
        self._stand_hold = _Hold(stand_seconds, tol_deg=180.0)
        self._facing: list = []            # recent hip facing angles (deg), for the face-the-camera check
        self._stand_ready = False
        self.finish_reason = ""

        self.stage = CALIBRATE
        self.side = LEFT
        self._hold = _Hold(hold_seconds, tol_deg)
        self._neutral_hold = _Hold(neutral_seconds, neutral_tol_deg)
        self._neutral = {LEFT: None, RIGHT: None}
        self._extremes = {LEFT: [], RIGHT: []}
        self._peak: Optional[float] = None
        self._smooth: list = []

        self.profile: Optional[HipProfile] = None
        self.stance: Optional[StanceRec] = None
        self.runner = None                 # SquatRunner, set when the squat stage opens

    # ------------------------------------------------------------------ helpers
    def _angle(self, lm) -> float:
        """Neutral-corrected rotation for the current leg, lightly smoothed."""
        raw = hip_rotation_angle(lm, self.side)
        self._smooth.append(raw)
        del self._smooth[:-5]
        value = float(np.median(self._smooth))
        neutral = self._neutral[self.side]
        return value - neutral if neutral is not None else value

    def _new_step(self):
        """A measuring step begins: the user gets a countdown before anything is read."""
        self._countdown_until = None
        self._countdown_done = False

    def _countdown(self, now: float) -> Optional[UiState]:
        """Get-ready screen while the countdown runs; None once it has finished."""
        if self._countdown_done or self.countdown_seconds <= 0:
            return None
        if self._countdown_until is None:
            self._countdown_until = now + self.countdown_seconds
        left = self._countdown_until - now
        if left <= 0:
            self._countdown_done = True
            self._hold.reset(); self._neutral_hold.reset(); self._smooth.clear(); self._peak = None
            return None
        n = math.ceil(left)
        leg = self.side.upper()
        # Cues stay short: the camera window draws them on a 640-px frame.
        if self.stage == NEUTRAL:
            cue = (f"LEFT done. {leg} leg: foot STRAIGHT ahead" if self.side == RIGHT
                   else f"{leg} leg: point the foot STRAIGHT ahead")
            detail = f"Starts in {n} s. Sit tall, lower leg hanging, hold still."
        elif self.stage == EXTREME_A:
            cue = f"Next: {leg} leg as FAR as it goes, HOLD"
            detail = f"Starts in {n} s. Swing from the knee, thigh still."
        else:
            first = self._extremes[self.side][0] if self._extremes[self.side] else 0.0
            cue = f"Locked {first:+.0f}. Next: the OTHER way, HOLD"
            detail = f"Starts in {n} s. Same leg, opposite direction, to your limit."
        return UiState(stage=self.stage, side=self.side, title=self._title(), cue=cue, detail=detail,
                       progress=1.0 - left / self.countdown_seconds, angle=None,
                       progress_label=f"Get ready  {n}", countdown=n)

    def _next_leg_or_review(self):
        if self.side == LEFT:
            self.side = RIGHT
            self.stage = NEUTRAL
            self._smooth.clear()
            self._neutral_hold.reset()
            self._hold.reset()
            self._peak = None
            self._new_step()
        else:
            self._finish_profile()

    def _finish_profile(self):
        """Turn the four locked extremes into a profile + stance recommendation."""
        def ir_er(vals):
            pos = max([v for v in vals if v > 0], default=0.0)
            neg = min([v for v in vals if v < 0], default=0.0)
            return abs(pos), abs(neg)

        ir_l, er_l = ir_er(self._extremes[LEFT])
        ir_r, er_r = ir_er(self._extremes[RIGHT])
        self.profile = build_hip_profile(ir_l, er_l, ir_r, er_r)
        self.stance = recommend_stance(self.profile, self.calib.hip_width)
        self.stage = SQUAT if self.do_squat else DONE

    # ------------------------------------------------------------------- update
    def update(self, world_landmarks, now: float) -> UiState:
        """Push one pose frame. ``now`` is a monotonic timestamp in seconds."""
        if world_landmarks is None:
            return UiState(stage=self.stage, side=self.side, tracking=False,
                           title=self._title(), cue="Step back until you are fully in frame",
                           detail="No person detected")
        lm = getattr(world_landmarks, "landmark", world_landmarks)

        if self.stage == CALIBRATE:
            if not (leg_visible(lm, LEFT, self.min_visibility)
                    and leg_visible(lm, RIGHT, self.min_visibility)):
                return self._not_visible("both legs")
            return self._do_calibrate(world_landmarks)
        if self.stage in (NEUTRAL, EXTREME_A, EXTREME_B):
            if not leg_visible(lm, self.side, self.min_visibility):
                return self._not_visible(f"your {self.side.lower()} leg")
            ready = self._countdown(now)
            if ready is not None:
                return ready
            if self.stage == NEUTRAL:
                return self._do_neutral(lm, now)
            return self._do_extreme(lm, now)
        if self.stage == SQUAT:
            return self._do_squat(lm, now)
        return UiState(stage=self.stage, title=self._title(), cue="Assessment complete")

    def _not_visible(self, what: str) -> UiState:
        """The leg is not on camera: discard any hold rather than measure a guess."""
        self._hold.reset()
        self._neutral_hold.reset()
        self._smooth.clear()
        self._peak = None
        return UiState(
            stage=self.stage, side=self.side, title=self._title(), tracking=False,
            cue=f"Move back until {what} {'are' if what == 'both legs' else 'is'} fully in frame",
            detail="Nothing is measured until hip, knee and ankle are visible")

    # --------------------------------------------------------------- stage impls
    def _do_calibrate(self, world_landmarks) -> UiState:
        self.calib.update(world_landmarks)
        done = len(self.calib._hip_w)
        need = self.calib.needed
        if done >= need:
            self.calib.finalize()
            self.stage = NEUTRAL
            self._new_step()
            return UiState(stage=NEUTRAL, side=self.side, title=self._title(),
                           cue="Calibrated", just_locked="calibration",
                           detail="Body scale captured")
        return UiState(stage=CALIBRATE, title="Calibrating",
                       cue="Sit still, fully in frame",
                       detail="Lower legs hanging free, feet off the floor",
                       progress=done / need,
                       progress_label=f"Hold still  {int(100*done/need)}%")

    def _do_neutral(self, lm, now: float) -> UiState:
        raw = hip_rotation_angle(lm, self.side)
        self._smooth.append(raw)
        del self._smooth[:-5]
        value = float(np.median(self._smooth))
        progress = self._neutral_hold.update(value, now, eligible=True)
        if progress >= 1.0:
            self._neutral[self.side] = self._neutral_hold.locked_value()
            self._neutral_hold.reset()
            self._hold.reset()
            self._peak = None
            # Drop the neutral readings: leaving them in the median window would
            # hold the reported angle near zero for the first few frames of the
            # next stage, delaying the moment the user's rotation registers.
            self._smooth.clear()
            self.stage = EXTREME_A
            self._new_step()
            return UiState(stage=EXTREME_A, side=self.side, title=self._title(),
                           cue="Neutral set", just_locked="neutral", angle=0.0,
                           detail="Now rotate to one extreme")
        return UiState(stage=NEUTRAL, side=self.side, title=self._title(),
                       cue=f"{self.side} leg: foot STRAIGHT ahead, hold still",
                       detail="This zeroes your neutral, so hold the same pose each time",
                       progress=progress, angle=value,
                       progress_label=f"Hold still  {int(progress*100)}%")

    def _do_extreme(self, lm, now: float) -> UiState:
        angle = self._angle(lm)
        first = self.stage == EXTREME_A
        if first:
            eligible = abs(angle) >= self.min_move_deg
            want = "either direction"
        else:
            # the second extreme must be the OTHER way from the first
            sign = -1.0 if self._extremes[self.side][0] > 0 else 1.0
            eligible = abs(angle) >= self.min_move_deg and angle * sign > 0
            want = "the other way" if sign < 0 else "the other way"

        if eligible and (self._peak is None or abs(angle) > abs(self._peak)):
            self._peak = angle
        # Lock only AT the extreme: within peak_tol_deg of the furthest point reached in
        # this step. Pausing part-way, or drifting back from the limit, does not count.
        at_peak = eligible and self._peak is not None and abs(angle) >= abs(self._peak) - self.peak_tol_deg
        progress = self._hold.update(angle, now, at_peak)

        if progress >= 1.0:
            locked = self._hold.locked_value()
            self._extremes[self.side].append(locked)
            self._hold.reset()
            self._peak = None
            self._smooth.clear()          # same reason as the neutral -> extreme hop
            if first:
                self.stage = EXTREME_B
                self._new_step()
                return UiState(stage=EXTREME_B, side=self.side, title=self._title(),
                               cue="Locked - now rotate the OTHER way",
                               just_locked=f"{locked:+.0f} deg", angle=angle)
            self._next_leg_or_review()
            return UiState(stage=self.stage, side=self.side, title=self._title(),
                           cue="Locked", just_locked=f"{locked:+.0f} deg", angle=angle)

        direction = ("Rotate as FAR as it goes, then HOLD STILL" if first
                     else "OTHER way, as FAR as it goes, HOLD STILL")
        if eligible and not at_peak:
            direction = f"Back to your furthest point ({self._peak:+.0f}), HOLD"
        return UiState(
            stage=self.stage, side=self.side, title=self._title(), cue=direction,
            detail=("Swing the lower leg sideways from the knee, thigh still"
                    if first else f"Opposite direction to your first reading ({want})"),
            progress=progress, angle=angle, peak=self._peak,
            progress_label=(f"Hold still at your limit  {int(progress*100)}%"
                            if progress > 0 else "Rotate further to begin"),
        )

    def _do_squat(self, lm, now: float) -> UiState:
        if self.runner is None:
            return self._do_stand_up(lm, now)
        st = self.runner.update(lm, now)
        locked = ""
        if st.new_rep is not None:
            r = st.new_rep
            if r.confirmed is None:
                locked = f"Rep {r.rep_index} counted"
            elif r.confirmed:
                locked = f"Rep {r.rep_index} confirmed  (score {r.squat_score}/9)"
            else:
                why = getattr(r, "rule_failed", None)
                locked = f"Rep {r.rep_index} rejected - not a squat" + (f" ({', '.join(why[:2])})" if why else "")
        done = self.runner.confirmed_count
        if self.target_reps and done >= self.target_reps:
            self.finish_reason = f"{done} confirmed squats"
            self.finish()

        if self.target_reps:
            cue = f"Squat - {done} of {self.target_reps} confirmed"
            detail = "Each rep is checked as a squat. SPACE finishes early."
        else:
            cue = "Squat as you normally would"
            detail = "Press SPACE when you are done."
        if getattr(self.runner, "facing_ok", True) is False:
            detail = "Turn to face the camera - stance needs a front view."
        return UiState(stage=self.stage, title="Squat check", cue=cue, detail=detail,
                       progress=(min(1.0, done / self.target_reps)
                                 if self.target_reps else 0.0),
                       progress_label=(f"{done}/{self.target_reps} confirmed"
                                       if self.target_reps else ""),
                       rep_count=st.rep_count, confirmed=done,
                       phase=st.phase, depth=st.depth, p_squat=st.p_squat,
                       just_locked=locked)

    def _do_stand_up(self, lm, now: float) -> UiState:
        """Wait until the lifter is actually standing before arming the counter.

        Straight out of the seated rotation stage the knees are bent ~50 deg and
        the hips sit level with them, which the FSM reads as the BOTTOM of a
        squat and the depth measure reads as ~1.0-2.4.  Standing up then walks
        BOTTOM -> ASCENT -> STANDING and books a rep that never happened, with an
        impossible depth attached.  Gate on being upright first.
        """
        arr = np.asarray(_as_points(lm), dtype=float)
        knee = (knee_flexion(arr, LEFT) + knee_flexion(arr, RIGHT)) / 2.0
        # Facing is a WARNING here, not a gate: MediaPipe's hip-direction reading
        # jitters on some front-camera footage (up to 70% of frames flagged in the
        # worst Multi-View front clip), and blocking on it locked people out. The
        # stance advice has its own front-view check over the confirmed squats.
        self._facing = (self._facing + [facing_angle(arr)])[-9:]
        facing = float(np.median(self._facing))
        turned = facing > FRONT_FACING_MAX_DEG
        progress = self._stand_hold.update(knee, now, eligible=knee >= self.stand_deg)
        if progress >= 1.0:
            self._stand_ready = True
            self._stand_hold.reset()
            return UiState(stage=SQUAT, title="Squat check", cue="Ready - start squatting",
                           just_locked="standing")
        return UiState(
            stage=SQUAT, title="Squat check",
            cue=("Turn to face the camera" if turned else "Stand up for the squat check"),
            detail=(f"Stance advice needs a front view (turned {facing:.0f} deg)"
                    if turned else "Face the camera. Counting starts once you are upright."),
            progress=progress, angle=None,
            progress_label=f"Stand tall  {int(progress * 100)}%")

    # ------------------------------------------------------------------ control
    @property
    def needs_runner(self) -> bool:
        """True once the lifter is upright and the counter should be armed."""
        return self.stage == SQUAT and self.runner is None and self._stand_ready

    def start_squat(self, runner):
        """Attach a SquatRunner and begin the squat stage."""
        self.runner = runner
        self.stage = SQUAT

    def finish(self):
        self.stage = DONE

    def skip_leg(self):
        """Abandon the current leg (e.g. it will not track) and move on."""
        self._next_leg_or_review()

    @property
    def reps(self) -> list:
        return list(self.runner.reps) if self.runner is not None else []

    def _title(self) -> str:
        if self.stage == CALIBRATE:
            return "Calibrating"
        if self.stage in (NEUTRAL, EXTREME_A, EXTREME_B):
            step = {NEUTRAL: "set neutral", EXTREME_A: "first extreme",
                    EXTREME_B: "second extreme"}[self.stage]
            return f"{self.side} leg - {step}"
        return {REVIEW: "Your result", SQUAT: "Squat check",
                DONE: "Done"}.get(self.stage, self.stage)
