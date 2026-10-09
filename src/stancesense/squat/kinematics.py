"""Phase 7a — squat kinematics (pure geometry, no model needed).

All measurements are built from MediaPipe world landmarks and normalised by
femur length / hip width so they are comparable across body sizes.
"""
from __future__ import annotations
import numpy as np
from ..common.geometry import angle_between
from ..kinematics.rotation import IDX
from ..common.types import LEFT, RIGHT


def _p(lm, i):
    o = lm[i]
    return np.array([o.x, o.y, o.z]) if hasattr(o, "x") else np.asarray(o, float)


def joint_angle(a, b, c) -> float:
    """Interior angle at b formed by segments b->a and b->c (degrees)."""
    return angle_between(np.asarray(a) - np.asarray(b), np.asarray(c) - np.asarray(b))


def knee_flexion(lm, side: str) -> float:
    """Knee angle: ~180 standing, decreases as the person squats."""
    hip, knee, ankle = (_p(lm, IDX[side][k]) for k in ("hip", "knee", "ankle"))
    return joint_angle(hip, knee, ankle)


def trunk_lean(lm) -> float:
    """Trunk lean of the torso away from vertical (degrees).

    Measured in full 3D against the up axis, so it captures forward lean (the
    sagittal plane, where squat lean actually happens) as well as sideways lean,
    and does not depend on which way the lifter is facing the camera.
    """
    l_sh, r_sh = _p(lm, 11), _p(lm, 12)
    l_hip, r_hip = _p(lm, IDX[LEFT]["hip"]), _p(lm, IDX[RIGHT]["hip"])
    torso = ((l_sh + r_sh) / 2.0) - ((l_hip + r_hip) / 2.0)
    return angle_between(torso, (0.0, -1.0, 0.0))


def depth_ratio(lm, femur_length: float) -> float:
    """Femur-normalised squat depth: how far the hip has dropped toward the knee.

    0 ~ standing tall; ~1 hip level with knee (parallel); >1 below parallel.
    """
    hip = (_p(lm, IDX[LEFT]["hip"]) + _p(lm, IDX[RIGHT]["hip"])) / 2.0
    knee = (_p(lm, IDX[LEFT]["knee"]) + _p(lm, IDX[RIGHT]["knee"])) / 2.0
    if femur_length <= 0:
        return 0.0
    # world y increases downward: hip below knee => hip.y approaches/exceeds knee.y
    return float((hip[1] - knee[1]) / femur_length + 1.0)


def lateral_axis(lm):
    """Unit vector along the lifter's own left-right axis, in the ground plane.

    y points down, so the horizontal plane is (x, z) and the hip line gives the
    lateral direction.  Measuring sideways offsets against THIS rather than the
    camera's x axis makes them independent of which way the lifter faces - the
    same squat scores the same whether they stand square to the camera or at 40
    degrees to it.  Returns ``None`` when the hips are degenerate/untracked.
    """
    l_hip, r_hip = _p(lm, IDX[LEFT]["hip"]), _p(lm, IDX[RIGHT]["hip"])
    v = np.array([l_hip[0] - r_hip[0], l_hip[2] - r_hip[2]], dtype=float)
    n = float(np.linalg.norm(v))
    return None if n < 1e-9 else v / n


def _lateral_offset(a, b, axis) -> float:
    """Signed ground-plane separation of a and b along ``axis``."""
    d = np.array([a[0] - b[0], a[2] - b[2]], dtype=float)
    return float(np.dot(d, axis))


def knee_over_foot_dev(lm, side: str, hip_width: float) -> float:
    """Sideways knee-vs-ankle offset (valgus/varus), normalised by hip width.

    Measured along the lifter's own lateral axis (see :func:`lateral_axis`), so
    it reports knee travel across the foot rather than a facing-dependent mix of
    sideways and fore/aft travel.
    """
    if hip_width <= 0:
        return 0.0
    axis = lateral_axis(lm)
    if axis is None:
        return 0.0
    knee, ankle = _p(lm, IDX[side]["knee"]), _p(lm, IDX[side]["ankle"])
    return float(abs(_lateral_offset(knee, ankle, axis)) / hip_width)


def detect_stance(lm, hip_width: float):
    """Return (width_used_factor, toe_out_used_deg) actually adopted by the lifter.

    Both are measured in the GROUND plane (x, z) - y points down, so the vertical
    axis must be excluded or standing height leaks into the stance numbers, and
    the depth axis must be included or the reading collapses as the lifter turns.

    ``width_used_factor`` is the ankle separation in hip widths.  ``toe_out_deg``
    is how far the feet point away from straight ahead: a foot pointing straight
    forward lies perpendicular to the hip line, so the toe-out angle is 90 deg
    minus the angle between the foot vector and the lifter's own lateral axis.
    Taking it against the lifter's axis keeps it independent of their facing.
    Magnitude only, as before - toe-in reads the same as toe-out.
    """
    l_ank, r_ank = _p(lm, IDX[LEFT]["ankle"]), _p(lm, IDX[RIGHT]["ankle"])
    l_foot, r_foot = _p(lm, 31), _p(lm, 32)   # foot index landmarks

    def ground(v):
        return np.array([v[0], v[2]], dtype=float)

    sep = float(np.linalg.norm(ground(l_ank) - ground(r_ank)))
    width_factor = sep / hip_width if hip_width > 0 else 0.0

    axis = lateral_axis(lm)

    def toe(ankle, foot):
        v = ground(foot) - ground(ankle)
        n = float(np.linalg.norm(v))
        if n < 1e-9 or axis is None:
            return 0.0                       # no foot landmark (e.g. MM-Fit)
        cos_to_lateral = abs(float(np.dot(v / n, axis)))
        return 90.0 - float(np.degrees(np.arccos(np.clip(cos_to_lateral, 0.0, 1.0))))

    toe_out = (toe(l_ank, l_foot) + toe(r_ank, r_foot)) / 2.0
    return round(width_factor, 3), round(toe_out, 1)
