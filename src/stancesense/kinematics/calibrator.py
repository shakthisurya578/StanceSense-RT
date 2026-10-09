"""Phase 4 — neutral calibration + body scale.

From a couple of seconds of relaxed standing we record:
* hip_width      -> distance between the two hip landmarks (our length unit)
* femur_length   -> hip->knee distance (used to normalise squat depth later)
* neutral shank angle per leg -> the "zero" so rotation reads ~0 at rest.

Everything downstream is expressed in multiples of hip_width, which makes the
system scale-invariant across people of different sizes.
"""
from __future__ import annotations
import numpy as np
from .rotation import IDX, signed_shank_angle_deg
from ..common.types import LEFT, RIGHT


def _p(lm, i):
    o = lm[i]
    return np.array([o.x, o.y, o.z]) if hasattr(o, "x") else np.asarray(o, float)


class Calibrator:
    def __init__(self, needed_frames: int = 45):   # ~1.5 s @ 30 FPS
        self.needed = needed_frames
        self._hip_w, self._femur, self._shank0 = [], [], {LEFT: [], RIGHT: []}
        self.hip_width = None
        self.femur_length = None
        self.neutral_shank = {LEFT: 0.0, RIGHT: 0.0}

    def update(self, world_landmarks):
        """Accumulate one neutral-standing frame."""
        lm = world_landmarks.landmark if hasattr(world_landmarks, "landmark") else world_landmarks
        l_hip, r_hip = _p(lm, IDX[LEFT]["hip"]), _p(lm, IDX[RIGHT]["hip"])
        self._hip_w.append(float(np.linalg.norm(l_hip - r_hip)))
        # Average both femurs, matching datasets.mmfit.estimate_scale: a single
        # badly-tracked knee should not skew the constant every depth reading is
        # divided by, and the live and offline paths must normalise identically.
        l_knee, r_knee = _p(lm, IDX[LEFT]["knee"]), _p(lm, IDX[RIGHT]["knee"])
        self._femur.append(0.5 * (float(np.linalg.norm(l_hip - l_knee))
                                  + float(np.linalg.norm(r_hip - r_knee))))
        for side in (LEFT, RIGHT):
            knee, ankle = _p(lm, IDX[side]["knee"]), _p(lm, IDX[side]["ankle"])
            self._shank0[side].append(signed_shank_angle_deg(knee, ankle))

    def is_ready(self) -> bool:
        return len(self._hip_w) >= self.needed

    def finalize(self):
        """Freeze the calibration constants (call once is_ready())."""
        self.hip_width = float(np.median(self._hip_w))
        self.femur_length = float(np.median(self._femur))
        for side in (LEFT, RIGHT):
            self.neutral_shank[side] = float(np.median(self._shank0[side]))
        return self

    def correct(self, raw_rotation_deg: float, side: str) -> float:
        """Subtract the neutral shank offset so rest reads ~0 degrees."""
        sign = 1.0 if side == RIGHT else -1.0
        return raw_rotation_deg - sign * self.neutral_shank[side]
