"""Tests for the Phase-11 SQLite store."""
import pytest

from stancesense.common.types import HipProfile, StanceRec, RepResult
from stancesense.storage import Store


def _profile(**kw):
    base = dict(ir_max=30.0, er_max=40.0, total_arc=70.0, rotation_bias=-5.0,
                symmetry=3.0, pattern="ER-dominant",
                ir_left=28.0, er_left=41.0, ir_right=32.0, er_right=39.0)
    base.update(kw)
    return HipProfile(**base)


def _stance():
    return StanceRec(width_factor=1.15, toe_out_deg=19.0,
                     rationale="wider stance suits ER dominance", asymmetric=True)


def _reps():
    return [
        RepResult(1, 0.92, 0.11, True, ["STANDING", "DESCENT"], gru_prob=0.97, confirmed=True),
        RepResult(2, 0.81, 0.14, True, ["STANDING", "DESCENT"], gru_prob=0.12, confirmed=False),
        RepResult(3, 0.88, 0.09, False, ["STANDING"]),          # un-gated
    ]


@pytest.fixture()
def store(tmp_path):
    s = Store(str(tmp_path / "test.db"))
    yield s
    s.close()


def test_creates_the_db_file_on_first_use(tmp_path):
    path = tmp_path / "nested" / "fresh.db"
    assert not path.exists()
    s = Store(str(path))
    assert path.exists()
    assert s.sessions() == []
    s.close()


def test_round_trips_a_whole_assessment(store):
    sid = store.save_assessment(profile=_profile(), stance=_stance(), reps=_reps(),
                                subject_id="w04", source="mmfit", notes="demo")
    p = store.profile_for(sid)
    assert p["pattern"] == "ER-dominant"
    assert p["ir_max"] == pytest.approx(30.0)
    assert p["er_left"] == pytest.approx(41.0)

    st = store.stance_for(sid)
    assert st["width_factor"] == pytest.approx(1.15)
    assert st["asymmetric"] == 1

    reps = store.reps_for(sid)
    assert [r["rep_index"] for r in reps] == [1, 2, 3]
    assert reps[0]["confirmed"] == 1 and reps[1]["confirmed"] == 0
    assert reps[2]["confirmed"] is None            # un-gated stays NULL, not 0


def test_rep_summary_separates_confirmed_rejected_and_ungated(store):
    sid = store.save_assessment(reps=_reps())
    s = store.rep_summary(sid)
    assert (s["counted"], s["confirmed"], s["rejected"], s["ungated"]) == (3, 1, 1, 1)
    assert s["max_depth"] == pytest.approx(0.92)
    assert s["mean_gru_prob"] == pytest.approx((0.97 + 0.12) / 2)


def test_rep_summary_of_an_empty_session_is_zeroed(store):
    sid = store.save_assessment(profile=_profile())
    assert store.rep_summary(sid)["counted"] == 0


def test_profile_history_is_newest_first_and_filterable(store):
    store.save_assessment(profile=_profile(ir_max=10.0), subject_id="a")
    store.save_assessment(profile=_profile(ir_max=20.0), subject_id="b")
    store.save_assessment(profile=_profile(ir_max=30.0), subject_id="a")

    all_rows = store.profile_history()
    assert [r["ir_max"] for r in all_rows] == [30.0, 20.0, 10.0]

    only_a = store.profile_history(subject_id="a")
    assert [r["ir_max"] for r in only_a] == [30.0, 10.0]
    assert all(r["subject_id"] == "a" for r in only_a)


def test_last_session_with_reps_ignores_rep_less_sessions(store):
    with_reps = store.save_assessment(reps=_reps())
    store.save_assessment(profile=_profile())          # newer, but no reps
    assert store.last_session_with_reps() == with_reps


def test_reopening_the_same_file_keeps_the_data(tmp_path):
    path = str(tmp_path / "persist.db")
    s1 = Store(path)
    sid = s1.save_assessment(profile=_profile(), reps=_reps())
    s1.close()

    s2 = Store(path)                                   # schema is CREATE IF NOT EXISTS
    assert s2.profile_for(sid)["pattern"] == "ER-dominant"
    assert len(s2.reps_for(sid)) == 3
    s2.close()


def test_context_manager_closes_the_connection(tmp_path):
    with Store(str(tmp_path / "ctx.db")) as s:
        s.save_assessment(profile=_profile())
    with pytest.raises(Exception):
        s.sessions()                                   # connection is closed


def test_squat_run_round_trip_and_classifier_in_summary(store):
    adv = dict(current_stance="NARROW", stance_width_hip_widths=1.31, depth_ratio=0.43,
               max_trunk_lean_deg=59.3, knee_to_ankle_width_at_bottom=1.5, facing_deg=4.2,
               front_facing=True, recommendation="MODERATE", reason="depth stays above parallel")
    sid = store.save_assessment(profile=_profile(), stance=_stance(), reps=_reps(),
                                classifier="rules", squat_stance=adv)
    run = store.squat_run_for(sid)
    assert run["classifier"] == "rules" and run["recommendation"] == "MODERATE"
    assert run["stance_width"] == pytest.approx(1.31) and run["front_facing"] == 1
    assert store.rep_summary(sid)["classifier"] == "rules"
    old = store.save_assessment(reps=_reps())                 # stored before squat_runs existed
    assert store.squat_run_for(old) is None and store.rep_summary(old)["classifier"] is None


def test_depth_stats_ignore_rejected_reps(store):
    """A rejected rep (noise, other exercise) must not set the best depth."""
    reps = [RepResult(1, 0.55, 0.1, True, [], confirmed=True),
            RepResult(2, 1.67, 0.1, False, [], confirmed=False),   # impossible depth, rejected
            RepResult(3, 0.45, 0.1, True, [], confirmed=True)]
    s = store.rep_summary(store.save_assessment(reps=reps))
    assert s["max_depth"] == pytest.approx(0.55) and s["mean_depth"] == pytest.approx(0.50)
    assert s["depth_from"] == "confirmed squats"
    fsm_only = [RepResult(1, 0.6, 0.1, True, []), RepResult(2, 0.8, 0.1, True, [])]
    s = store.rep_summary(store.save_assessment(reps=fsm_only))
    assert s["max_depth"] == pytest.approx(0.8) and "none were judged" in s["depth_from"]
    none_ok = [RepResult(1, 1.7, 0.1, True, [], confirmed=False)]
    s = store.rep_summary(store.save_assessment(reps=none_ok))
    assert s["max_depth"] == 0.0 and s["depth_from"] == "no confirmed squat"
