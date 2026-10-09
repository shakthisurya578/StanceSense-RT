"""Phase 5 - the hip profile.

The camera window's controller (app/controller.py) locks four extremes, internal
and external rotation for each leg. This module turns them into one
:class:`HipProfile`: average IR / ER, total arc, rotation bias, left/right symmetry
and a pattern (IR-dominant, balanced or ER-dominant).
"""
from __future__ import annotations

from ..common.types import HipProfile


def classify_pattern(rotation_bias: float, dead_zone: float = 5.0) -> str:
    if rotation_bias > dead_zone:
        return "IR-dominant"
    if rotation_bias < -dead_zone:
        return "ER-dominant"
    return "balanced"


def build_hip_profile(ir_left, er_left, ir_right, er_right) -> HipProfile:
    """Combine both legs' IR/ER extrema into the final profile (pure + testable)."""
    ir = (ir_left + ir_right) / 2.0
    er = (er_left + er_right) / 2.0
    bias_l = (ir_left - er_left) / 2.0
    bias_r = (ir_right - er_right) / 2.0
    bias = (bias_l + bias_r) / 2.0
    return HipProfile(
        ir_max=ir, er_max=er,
        total_arc=ir + er,
        rotation_bias=bias,
        symmetry=abs(bias_l - bias_r),
        pattern=classify_pattern(bias),
        ir_left=ir_left, er_left=er_left, ir_right=ir_right, er_right=er_right,
    )
