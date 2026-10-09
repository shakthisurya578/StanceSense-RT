"""One guided assessment from camera frames to the SQLite store.

``scripts/live_assessment.py`` (the camera window) feeds it frames with their
capture time; the session runs pose, the controller and the rule-based squat
classifier, and saves the run when it is complete.
"""
from __future__ import annotations

from typing import Optional

from ..kinematics.calibrator import Calibrator
from ..squat.rule_classifier import RuleSquatRunner
from ..storage import Store
from ..ui import format_profile, format_rep_summary, profile_note
from .controller import AssessmentController, DONE, SQUAT
from .hud import infer_and_annotate

class LiveSession:
    """Pose -> AssessmentController -> rule-based squat runner -> store, for one person.

    ``pose`` needs ``infer(rgb) -> (image_landmarks, world_landmarks)`` and
    ``draw(bgr, image_landmarks)``; :class:`PoseEstimator` provides both.
    """

    classifier = "rules"

    def __init__(self, pose, *, do_squat: bool = True, target_reps: int = 5,
                 countdown_seconds: float = 4.0, hold_seconds: float = 2.0):
        self.pose = pose
        self.ctrl = AssessmentController(
            Calibrator(), do_squat=do_squat, target_reps=target_reps,
            countdown_seconds=countdown_seconds, hold_seconds=hold_seconds)

    # ------------------------------------------------------------------ frames
    def step(self, frame_bgr, now: float):
        """Process one BGR frame captured at ``now`` (seconds, monotonic).

        Returns ``(display, state)``: the mirrored, annotated frame and the UiState.
        """
        display, state = infer_and_annotate(frame_bgr, self.pose, self.ctrl, now)
        if self.ctrl.needs_runner:           # the person is standing: arm the squat check
            calib = self.ctrl.calib
            self.ctrl.start_squat(RuleSquatRunner(calib.femur_length, calib.hip_width))
        return display, state

    # ----------------------------------------------------------------- control
    def skip_leg(self):
        if self.ctrl.stage != SQUAT and not self.done:
            self.ctrl.skip_leg()

    def finish_set(self):
        """End the squat set early (the SPACE key)."""
        if self.ctrl.stage == SQUAT:
            self.ctrl.finish_reason = "you ended the set"
            self.ctrl.finish()

    @property
    def stage(self) -> str:
        return self.ctrl.stage

    @property
    def done(self) -> bool:
        return self.ctrl.stage == DONE

    @property
    def complete(self) -> bool:
        """True once the hip profile exists, which is what makes a run worth saving."""
        return self.ctrl.profile is not None

    # ----------------------------------------------------------------- results
    def squat_summary(self) -> Optional[dict]:
        runner = self.ctrl.runner
        return runner.summary() if runner is not None else None

    def save(self, db: str, subject: str = "local", source: str = "live") -> Optional[int]:
        """Store a complete run; returns the session id, or None if incomplete."""
        if not self.complete:
            return None
        runner = self.ctrl.runner
        summ = self.squat_summary()
        notes = f"live assessment (classifier={self.classifier})"
        with Store(db) as store:
            return store.save_assessment(
                profile=self.ctrl.profile, stance=self.ctrl.stance, reps=self.ctrl.reps,
                subject_id=subject, source=source, notes=notes,
                classifier=self.classifier if runner is not None else None,
                squat_stance=(summ.get("stance") if summ else None))

    def result(self) -> dict:
        """Plain-data summary of the run (JSON-safe), for printing or a web page."""
        if not self.complete:
            return {"complete": False}
        p, rec = self.ctrl.profile, self.ctrl.stance
        prof = dict(ir_max=p.ir_max, er_max=p.er_max, total_arc=p.total_arc,
                    rotation_bias=p.rotation_bias, pattern=p.pattern)
        out = {
            "complete": True,
            "pattern": p.pattern,
            "ir_left": p.ir_left, "er_left": p.er_left,
            "ir_right": p.ir_right, "er_right": p.er_right,
            "profile_text": format_profile(prof),
            "profile_note": profile_note(dict(pattern=p.pattern)),
            "width_factor": rec.width_factor,
            "toe_out_deg": rec.toe_out_deg,
            "stance_text": rec.rationale,
            "reps": len(self.ctrl.reps),
            "confirmed": self.ctrl.runner.confirmed_count if self.ctrl.runner else 0,
            "squat_text": None,
            "squat_stance": None,
            "finish_reason": self.ctrl.finish_reason,
        }
        summ = self.squat_summary()
        if summ is not None and self.ctrl.reps:
            out["squat_text"] = format_rep_summary(summ)
        if summ is not None and summ.get("stance"):
            out["squat_stance"] = summ["stance"]
        return out
