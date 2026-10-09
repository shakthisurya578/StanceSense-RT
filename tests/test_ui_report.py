"""Tests for the Phase-12 formatting helpers.

The dashboard and the CLI summaries both render through these, so a bug here
shows the user two different stories about the same run.  The rotation caveat is
a project constraint, not a nicety, so it is asserted too.
"""
import pytest

from stancesense.ui import (
    format_profile, format_stance, format_rep_summary, rep_table,
    profile_note, depth_verdict, session_headline,
)

PROFILE = dict(ir_max=30.0, er_max=40.0, total_arc=70.0, rotation_bias=-5.0,
               pattern="ER-dominant", ir_left=28.0, er_left=41.0,
               ir_right=32.0, er_right=39.0)
STANCE = dict(width_factor=1.15, toe_out_deg=19.0, asymmetric=1,
              rationale="wider stance suits ER dominance")


def test_format_profile_shows_every_headline_number():
    out = format_profile(PROFILE)
    for token in ("30.0", "40.0", "70.0", "-5.0", "ER-dominant"):
        assert token in out


def test_format_stance_flags_asymmetry():
    assert "asymmetric" in format_stance(STANCE)
    assert "asymmetric" not in format_stance({**STANCE, "asymmetric": 0})


def test_empty_inputs_render_a_message_not_a_crash():
    assert "no hip profile" in format_profile({})
    assert "no stance" in format_stance(None)
    assert "no squat reps" in format_rep_summary({})
    assert "no squat reps" in format_rep_summary({"counted": 0})


def test_profile_note_always_disclaims_version_measurement():
    """The project must never imply radiographic femoral/acetabular version."""
    for pattern in ("IR-dominant", "ER-dominant", "balanced"):
        note = profile_note({"pattern": pattern}).lower()
        assert "not a radiographic" in note
        assert "version" in note
        assert "no validated angular accuracy" in note


def test_profile_note_survives_a_missing_pattern():
    assert "not a radiographic" in profile_note({}).lower()


def test_rep_summary_distinguishes_confirmed_rejected_and_ungated():
    out = format_rep_summary(dict(counted=10, confirmed=7, rejected=2, ungated=1,
                                  mean_depth=0.75, max_depth=0.92,
                                  mean_gru_prob=0.88))
    assert "10 reps counted" in out
    assert "7 confirmed" in out
    assert "2 rejected" in out
    assert "1 not yet scored" in out
    assert "0.88" in out


def test_rep_summary_omits_the_gru_line_when_nothing_was_scored():
    out = format_rep_summary(dict(counted=3, confirmed=0, rejected=0, ungated=3,
                                  mean_depth=0.5, max_depth=0.6,
                                  mean_gru_prob=None))
    assert "P(squat)" not in out
    assert "3 not yet scored" in out


def test_rep_table_labels_each_verdict():
    rows = rep_table([
        dict(rep_index=1, max_depth_ratio=0.9, knee_over_foot_dev=0.1,
             symmetry_ok=1, gru_prob=0.97, confirmed=1),
        dict(rep_index=2, max_depth_ratio=0.8, knee_over_foot_dev=0.2,
             symmetry_ok=0, gru_prob=0.10, confirmed=0),
        dict(rep_index=3, max_depth_ratio=0.7, knee_over_foot_dev=0.3,
             symmetry_ok=1, gru_prob=None, confirmed=None),
    ])
    assert [r["verdict"] for r in rows] == ["squat", "rejected", "not gated"]
    assert rows[2]["P(squat)"] is None
    assert rows[1]["symmetry ok"] is False


def test_rep_table_has_no_probability_column_for_rule_based_reps():
    rows = rep_table([dict(rep_index=1, max_depth_ratio=0.6, knee_over_foot_dev=0.9,
                           symmetry_ok=1, gru_prob=None, confirmed=1)])
    assert "P(squat)" not in rows[0] and rows[0]["verdict"] == "squat"


@pytest.mark.parametrize("depth,expected", [
    (1.2, "at or below parallel"), (1.0, "at or below parallel"),
    (0.85, "close to parallel"), (0.6, "above parallel"),
    (0.3, "shallow"), (None, "unknown"),
])
def test_depth_verdict_bands(depth, expected):
    assert depth_verdict(depth) == expected


def test_session_headline_covers_all_three_stages():
    out = session_headline(PROFILE, STANCE,
                           dict(counted=5, confirmed=5, rejected=0, ungated=0,
                                mean_depth=0.8, max_depth=0.9, mean_gru_prob=0.99))
    assert out.count("\n") == 2
    assert "HIP PROFILE" in out and "STANCE" in out and "SQUAT RUN" in out


def test_arc_warning_fires_only_on_an_implausible_arc():
    from stancesense.ui import arc_warning, IMPLAUSIBLE_ARC_DEG
    assert arc_warning({"total_arc": 70.0}) is None
    assert arc_warning({"total_arc": IMPLAUSIBLE_ARC_DEG}) is None
    assert arc_warning({}) is None
    warn = arc_warning({"total_arc": 130.0})
    assert warn is not None
    assert "130" in warn and "not a measurement" in warn


def test_squat_stance_and_classifier_wording():
    from stancesense.ui import format_squat_stance
    out = format_squat_stance(dict(current_stance="NARROW", stance_width=1.31,
                                   recommendation="MODERATE", reason="room for the hips"))
    assert "NARROW" in out and "1.31x" in out and "MODERATE" in out
    assert "no squat-based stance advice" in format_squat_stance(None)
    rules = format_rep_summary(dict(counted=5, confirmed=3, rejected=2, ungated=0, mean_depth=0.5,
                                    max_depth=0.6, mean_gru_prob=None, classifier="rules"))
    assert "biomechanical rules" in rules and "GRU" not in rules
