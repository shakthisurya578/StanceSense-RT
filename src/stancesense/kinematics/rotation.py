"""Phase 3 — hip axial-rotation angle from the shank (the "digital Craig's test").

THE ONE IDEA
------------
With the thigh fixed (person seated, knee bent ~90, facing the camera), rotating
the hip about the femur's long axis makes the SHANK (knee->ankle) swing sideways.
The shank's deviation from the downward vertical, seen from the front, IS the hip
rotation angle.  We measure the long, high-visibility shank instead of the deep,
noisy hip joint centre (the least reliable landmark in markerless pose).

SIGN CONVENTION (documented + unit-tested)
------------------------------------------
theta_rot > 0  => INTERNAL rotation (ankle swings laterally, away from midline)
theta_rot < 0  => EXTERNAL rotation (ankle swings medially, toward midline)
The neutral (standing) offset is removed by the Calibrator so rest reads ~0.
"""
from __future__ import annotations
from ..common.geometry import signed_angle_2d
from ..common.types import LEFT, RIGHT

# MediaPipe BlazePose landmark indices for the lower limb.
IDX = {
    LEFT:  {"hip": 23, "knee": 25, "ankle": 27},
    RIGHT: {"hip": 24, "knee": 26, "ankle": 28},
}

# "Down" in the frontal plane. MediaPipe world y increases downward, so the
# floor direction is (0, +1). A hanging shank aligns with this => 0 degrees.
_DOWN = (0.0, 1.0)


def _xy(landmark):
    """Return (x, y) from either a MediaPipe landmark object or a sequence."""
    if hasattr(landmark, "x"):
        return (landmark.x, landmark.y)
    return (landmark[0], landmark[1])


def signed_shank_angle_deg(p_knee, p_ankle) -> float:
    """Signed deviation (deg) of the shank from the downward vertical.

    Ankle swung toward image -x  => positive; toward +x => negative.
    This is the raw, side-agnostic geometry (unit-tested with known angles).
    """
    kx, ky = _xy(p_knee)
    ax, ay = _xy(p_ankle)
    shank_frontal = (ax - kx, ay - ky)     # knee -> ankle, projected to (x, y)
    return signed_angle_2d(_DOWN, shank_frontal)


def hip_rotation_angle_from_points(p_knee, p_ankle, side: str) -> float:
    """Hip rotation angle for one leg, given its knee & ankle points.

    Applies the per-side sign so that INTERNAL rotation is POSITIVE for BOTH legs
    (a right leg's lateral swing is image -x; a left leg's lateral swing is +x).
    """
    theta = signed_shank_angle_deg(p_knee, p_ankle)
    return theta if side == RIGHT else -theta


def hip_rotation_angle(world_landmarks, side: str) -> float:
    """Convenience wrapper: pull knee & ankle from MediaPipe world landmarks."""
    lm = world_landmarks.landmark if hasattr(world_landmarks, "landmark") else world_landmarks
    idx = IDX[side]
    return hip_rotation_angle_from_points(lm[idx["knee"]], lm[idx["ankle"]], side)
