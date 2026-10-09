"""Phase 12 - presentation layer (formatting helpers + the Streamlit dashboard).

Only the pure formatting helpers live here; Streamlit itself is imported by
``scripts/dashboard.py`` so importing this package never requires it.
"""
from .report import (
    format_profile, format_stance, format_rep_summary, format_squat_stance, rep_table,
    profile_note, depth_verdict, session_headline,
    arc_warning, IMPLAUSIBLE_ARC_DEG,
)

__all__ = [
    "format_profile", "format_stance", "format_rep_summary", "format_squat_stance", "rep_table",
    "profile_note", "depth_verdict", "session_headline",
    "arc_warning", "IMPLAUSIBLE_ARC_DEG",
]
