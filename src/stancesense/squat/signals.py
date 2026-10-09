"""Per-frame biomechanical signals the rule-based squat classifier reads.

All signals come from MediaPipe WORLD landmarks (metres, hip-centred, y DOWN).
Lengths are divided by the person's femur length or hip width, so body size and
distance from the camera cancel out.

  knee_L/R       hip-knee-ankle angle (deg, ~175 standing)
  hip_L/R        shoulder-hip-knee angle (deg)
  ankle_L/R      knee-ankle-foot_index angle (deg)
  knee/hip/ankle the left/right means of the three angles above
  torso          shoulder-mid -> hip-mid line vs vertical (deg)
  stance_width   ankle separation on the ground plane / hip width
  knee_width     knee separation on the ground plane / hip width
  hip_height     hip-mid height above ankle-mid / femur (drops in a squat)
  knee_height    knee-mid height above ankle-mid / femur
  depth          (hip.y - knee.y) / femur + 1  (~0 standing, ~1 thighs parallel)
  foot_stagger   fore/aft ankle separation in the body frame / hip width (lunges)
  knee_travel    fore/aft knee-over-ankle distance in the body frame / femur
  leg_drop_asym  |left - right hip-above-ankle| / femur

Also holds :class:`FrameState`, what a squat runner reports for each frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..common.types import RepResult

L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE = 11, 12, 23, 24, 25, 26
L_ANK, R_ANK, L_FOOT, R_FOOT = 27, 28, 31, 32


def as_array(lm):
    """Accept MediaPipe landmark lists as well as plain (33, 3) arrays."""
    if hasattr(lm, "landmark"):
        lm = lm.landmark
    if hasattr(lm[0], "x"):
        return np.array([[o.x, o.y, o.z] for o in lm], dtype=float)
    return np.asarray(lm, dtype=float)


@dataclass
class FrameState:
    """What the squat runner computed for the frame just pushed."""
    phase: str
    knee_angle: float
    depth: float
    rep_count: int
    p_squat: Optional[float] = None       # unused by the rule-based runner (always None)
    new_rep: Optional[RepResult] = None   # set on the frame a rep completes


def _angle(a, b, c):
    """Angle at b (deg) for (T, 3) point arrays."""
    u, v = a - b, c - b
    cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _ground(p):
    return p[..., [0, 2]]          # y is vertical, so the ground plane is (x, z)


def frame_signals(lm: np.ndarray, femur: float, hip_w: float) -> dict:
    """Per-frame signals for a (T, 33, 3) landmark array (see module docstring)."""
    lm = np.asarray(lm, dtype=np.float64)
    femur = femur if femur > 0 else 1.0
    hip_w = hip_w if hip_w > 0 else 1.0
    P = lambda i: lm[:, i, :]
    hip_mid, knee_mid = (P(L_HIP) + P(R_HIP)) / 2, (P(L_KNEE) + P(R_KNEE)) / 2
    ank_mid, sh_mid = (P(L_ANK) + P(R_ANK)) / 2, (P(L_SH) + P(R_SH)) / 2
    torso_v = sh_mid - hip_mid
    up = np.array([0.0, -1.0, 0.0])
    torso = np.degrees(np.arccos(np.clip(torso_v @ up / (np.linalg.norm(torso_v, axis=1) + 1e-9), -1, 1)))
    # body-frame fore/aft axis: horizontal normal of the hip line
    lat = _ground(P(L_HIP) - P(R_HIP))
    lat = lat / (np.linalg.norm(lat, axis=1, keepdims=True) + 1e-9)
    fwd = np.stack([-lat[:, 1], lat[:, 0]], axis=1)
    s = dict(
        knee_L=_angle(P(L_HIP), P(L_KNEE), P(L_ANK)), knee_R=_angle(P(R_HIP), P(R_KNEE), P(R_ANK)),
        hip_L=_angle(P(L_SH), P(L_HIP), P(L_KNEE)), hip_R=_angle(P(R_SH), P(R_HIP), P(R_KNEE)),
        ankle_L=_angle(P(L_KNEE), P(L_ANK), P(L_FOOT)), ankle_R=_angle(P(R_KNEE), P(R_ANK), P(R_FOOT)),
        torso=torso,
        stance_width=np.linalg.norm(_ground(P(L_ANK) - P(R_ANK)), axis=1) / hip_w,
        knee_width=np.linalg.norm(_ground(P(L_KNEE) - P(R_KNEE)), axis=1) / hip_w,
        hip_height=(ank_mid[:, 1] - hip_mid[:, 1]) / femur,          # +y is down
        knee_height=(ank_mid[:, 1] - knee_mid[:, 1]) / femur,
        depth=(hip_mid[:, 1] - knee_mid[:, 1]) / femur + 1.0,
        foot_stagger=np.abs((_ground(P(L_ANK) - P(R_ANK)) * fwd).sum(1)) / hip_w,
        knee_travel=np.abs((_ground(knee_mid - ank_mid) * fwd).sum(1)) / femur,
        leg_drop_asym=np.abs((P(L_ANK)[:, 1] - P(L_HIP)[:, 1]) - (P(R_ANK)[:, 1] - P(R_HIP)[:, 1])) / femur,
    )
    s["knee"] = (s["knee_L"] + s["knee_R"]) / 2
    s["hip"] = (s["hip_L"] + s["hip_R"]) / 2
    s["ankle"] = (s["ankle_L"] + s["ankle_R"]) / 2
    return s
