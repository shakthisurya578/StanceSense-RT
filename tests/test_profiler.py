from stancesense.profiling.profiler import build_hip_profile, classify_pattern


def test_bias_arc_symmetry():
    p = build_hip_profile(ir_left=45, er_left=20, ir_right=43, er_right=22)
    assert p.total_arc == (44 + 21)          # averaged ir + er
    assert p.rotation_bias > 0               # IR dominant
    assert p.pattern == "IR-dominant"
    assert p.symmetry >= 0


def test_pattern_thresholds():
    assert classify_pattern(12) == "IR-dominant"
    assert classify_pattern(-12) == "ER-dominant"
    assert classify_pattern(2) == "balanced"
