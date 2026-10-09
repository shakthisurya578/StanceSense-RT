from stancesense.profiling.profiler import build_hip_profile
from stancesense.recommendation.stance import recommend_stance

RULES = "config/stance_rules.yaml"


def test_er_dominant_is_wider_and_more_toeout_than_ir():
    er = build_hip_profile(20, 45, 22, 44)   # ER dominant
    ir = build_hip_profile(45, 20, 44, 22)   # IR dominant
    rec_er = recommend_stance(er, 0.30, RULES)
    rec_ir = recommend_stance(ir, 0.30, RULES)
    assert rec_er.toe_out_deg > rec_ir.toe_out_deg
    assert rec_er.width_factor > rec_ir.width_factor


def test_outputs_within_clamped_bounds():
    p = build_hip_profile(5, 80, 5, 80)      # extreme ER
    rec = recommend_stance(p, 0.30, RULES)
    assert 5.0 <= rec.toe_out_deg <= 30.0
    assert 0.9 <= rec.width_factor <= 1.6


def test_asymmetry_flag():
    p = build_hip_profile(ir_left=50, er_left=10, ir_right=10, er_right=50)
    rec = recommend_stance(p, 0.30, RULES)
    assert rec.asymmetric is True
