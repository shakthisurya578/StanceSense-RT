"""Phase 12 - presentation helpers shared by the CLI summaries and the dashboard.

Pure formatting: takes the rows the :mod:`stancesense.storage` layer returns and
turns them into the strings/tables a human reads.  No Streamlit, no matplotlib —
so the CLI demo and the dashboard cannot drift apart, and this module stays
testable without a UI installed.
"""
from __future__ import annotations

from typing import Optional, Sequence

PATTERN_BLURB = {
    "IR-dominant": "internally-rotation dominant - a narrower, straighter stance "
                   "keeps you inside your comfortable arc",
    "ER-dominant": "externally-rotation dominant - a wider stance with more toe-out "
                   "gives the joint room to reach depth",
    "balanced": "fairly balanced - a near-generic stance should already suit you",
}


def format_profile(p: dict) -> str:
    """One-line summary of a stored hip profile row."""
    if not p:
        return "no hip profile recorded"
    return (f"IR {p['ir_max']:.1f} deg / ER {p['er_max']:.1f} deg   "
            f"arc {p['total_arc']:.1f} deg   bias {p['rotation_bias']:+.1f} deg   "
            f"pattern {p['pattern']}")


def profile_note(p: dict) -> str:
    """The qualitative caveat that must accompany every rotation number.

    StanceSense-RT measures FUNCTIONAL hip rotation from video.  It is not a
    radiographic measurement of femoral or acetabular version, and no angular
    accuracy is claimed (the goniometer validation study was dropped).
    """
    pattern = (p or {}).get("pattern", "balanced")
    return (f"Functional hip rotation, reported qualitatively: {pattern} "
            f"({PATTERN_BLURB.get(pattern, '')}). This is a movement-screen "
            f"signal from video, not a radiographic femoral/acetabular version "
            f"measurement, and it carries no validated angular accuracy.")


# A hip's combined internal + external rotation arc is generously ~90 deg.  A
# reading past that is not hip rotation - it is tracking noise or the person
# moving about - so every surface says so rather than reporting it flat.
IMPLAUSIBLE_ARC_DEG = 90.0


def arc_warning(p: dict, limit: float = IMPLAUSIBLE_ARC_DEG):
    """Return a warning string when the measured arc is not physiologically real.

    Returns ``None`` when the arc is plausible, so callers can just do
    ``if (w := arc_warning(profile)): ...``.
    """
    arc = (p or {}).get("total_arc")
    if arc is None or arc <= limit:
        return None
    return (f"A {arc:.0f} deg total rotation arc exceeds a plausible hip range "
            f"(~{limit:.0f} deg). For this recording the shank-angle signal is "
            f"dominated by the person moving rather than by hip rotation, so treat "
            f"this profile - and the stance derived from it - as a pipeline "
            f"demonstration, not a measurement.")


def format_stance(s: dict) -> str:
    """One-line summary of a stored stance recommendation row."""
    if not s:
        return "no stance recommendation recorded"
    flag = "  (asymmetric)" if s.get("asymmetric") else ""
    return (f"{s['width_factor']:.2f}x hip width, {s['toe_out_deg']:.0f} deg "
            f"toe-out{flag}")


def format_squat_stance(s: dict) -> str:
    """One line for the squat-based stance advice (stored squat_runs row or the
    runner's ``summary()["stance"]`` dict)."""
    if not s or not s.get("recommendation"):
        return "no squat-based stance advice (no confirmed squat)"
    width = s.get("stance_width", s.get("stance_width_hip_widths"))
    w = "" if width is None else f" ({width:.2f}x hip width)"
    return f"Measured stance {s.get('current_stance')}{w} -> recommended {s['recommendation']}: {s.get('reason', '')}"


def format_rep_summary(summary: dict) -> str:
    """One-line summary of a run's reps (see Store.rep_summary)."""
    if not summary or not summary.get("counted"):
        return "no squat reps recorded"
    judge = {"rules": "the biomechanical rules", "gru": "the GRU"}.get(summary.get("classifier"), "the verifier")
    txt = (f"{summary['counted']} reps counted, "
           f"{summary['confirmed']} confirmed as squats by {judge}")
    if summary.get("rejected"):
        txt += f", {summary['rejected']} rejected"
    if summary.get("ungated"):
        # the verifier's window had not filled yet - counted, but not judged
        txt += f", {summary['ungated']} not yet scored"
    txt += (f"   |   mean depth {summary['mean_depth']:.2f}, "
            f"best {summary['max_depth']:.2f} (1.0 ~ thighs parallel)")
    if summary.get("mean_gru_prob") is not None:
        txt += f"   |   mean P(squat) {summary['mean_gru_prob']:.2f}"
    return txt


def rep_table(reps: Sequence[dict]) -> list:
    """Rows ready for tabular display: one dict per rep, display-friendly keys.

    The P(squat) column only appears for sessions stored by the earlier GRU
    verifier; the rule-based classifier has no probability to show.
    """
    out = []
    has_prob = any(r.get("gru_prob") is not None for r in reps)
    for r in reps:
        if r.get("confirmed") is None:
            verdict = "not gated"
        else:
            verdict = "squat" if r["confirmed"] else "rejected"
        row = {
            "rep": r["rep_index"],
            "depth": round(r["max_depth_ratio"] or 0.0, 3),
            "knee-over-foot": round(r["knee_over_foot_dev"] or 0.0, 3),
            "symmetry ok": bool(r["symmetry_ok"]),
        }
        if has_prob:
            row["P(squat)"] = None if r["gru_prob"] is None else round(r["gru_prob"], 3)
        row["verdict"] = verdict
        out.append(row)
    return out


def depth_verdict(depth: Optional[float]) -> str:
    """Plain-language reading of the femur-normalised depth ratio."""
    if depth is None:
        return "unknown"
    if depth >= 1.0:
        return "at or below parallel"
    if depth >= 0.8:
        return "close to parallel"
    if depth >= 0.5:
        return "above parallel"
    return "shallow"


def session_headline(profile: dict, stance: dict, summary: dict) -> str:
    """The three-line block the demo prints and the dashboard shows at the top."""
    return "\n".join([
        f"HIP PROFILE     {format_profile(profile)}",
        f"STANCE          {format_stance(stance)}",
        f"SQUAT RUN       {format_rep_summary(summary)}",
    ])
