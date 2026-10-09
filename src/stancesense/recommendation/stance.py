"""Phase 6 — stance recommendation engine (Algorithm 3).

Transparent, tunable rule layer that turns a HipProfile into a STARTING stance
(width factor + toe-out).  Every constant lives in config/stance_rules.yaml, so
the mapping is fully explainable -- no black box.
"""
from __future__ import annotations
import yaml
from ..common.types import HipProfile, StanceRec


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _arc_scale(total_arc: float, arc_ref: float) -> float:
    """Small total arc (limited mobility) => scale width gain down toward 0.5."""
    return _clamp(total_arc / arc_ref, 0.5, 1.0)


def recommend_stance(profile: HipProfile, hip_width: float,
                     rules_path: str = "config/stance_rules.yaml") -> StanceRec:
    with open(rules_path, "r") as f:
        r = yaml.safe_load(f)
    t, w = r["toe_out"], r["width"]
    bias = profile.rotation_bias                 # + = IR dominant, - = ER dominant
    er_dominance = -bias                          # positive when ER-dominant

    # 1) Toe-out grows with external-rotation dominance (need room to open the hip)
    toe_out = _clamp(t["base_deg"] + t["k"] * er_dominance, t["min_deg"], t["max_deg"])

    # 2) Width also grows with ER dominance, but is damped by a small total arc
    width_factor = _clamp(
        w["base_factor"] + w["k"] * er_dominance * _arc_scale(profile.total_arc,
                                                              r["arc_reference_deg"]),
        w["min_factor"], w["max_factor"],
    )

    # 3) Asymmetry flag
    asymmetric = profile.symmetry > r["symmetry"]["tol_deg"]

    rationale = _rationale(profile, toe_out, width_factor, asymmetric)
    return StanceRec(width_factor=round(width_factor, 3),
                     toe_out_deg=round(toe_out, 1),
                     rationale=rationale, asymmetric=asymmetric)


def _rationale(p: HipProfile, toe_out, width_factor, asymmetric) -> str:
    if p.pattern == "ER-dominant":
        why = ("your hips externally rotate more than they internally rotate, so a "
               "wider stance with more toe-out gives the joint room to reach depth "
               "without pinching")
    elif p.pattern == "IR-dominant":
        why = ("your hips internally rotate more than they externally rotate, so a "
               "narrower, straighter stance keeps you in your comfortable arc")
    else:
        why = ("your rotation is fairly balanced, so a near-generic stance should "
               "already suit you")
    msg = (f"Recommended start: {width_factor:.2f}x hip width, {toe_out:.0f} toe-out - {why}. "
           f"(functional rotation IR {p.ir_max:.0f} / ER {p.er_max:.0f}, "
           f"arc {p.total_arc:.0f}, pattern {p.pattern}.)")
    if asymmetric:
        msg += " Left and right rotation differ noticeably - expect a slight stance offset between feet."
    msg += " This is a STARTING stance; the live-squat check will confirm or nudge it."
    return msg
