"""Rule-based squat classifier on synthetic landmark streams (y down, metres)."""
import numpy as np

from stancesense.squat.rule_classifier import (RuleConfig, RuleSquatClassifier, RuleSquatRunner,
                                               stance_class, stance_recommendation)


def _person(bend=0.0, feet=0.13):
    """(33, 3) landmarks; bend 0 = standing, 1.2 = below parallel."""
    lm = np.zeros((33, 3))
    hip_y = 0.30 * bend
    for side, x in ((0, 0.1), (1, -0.1)):
        sh, hip, knee, ank, foot = (11, 23, 25, 27, 31) if side == 0 else (12, 24, 26, 28, 32)
        fx = feet if side == 0 else -feet
        lm[sh] = [x * 1.6, hip_y - 0.5, 0.0]
        lm[hip] = [x, hip_y, 0.0]
        lm[knee] = [x * 1.2 + (fx - x) * 0.3, 0.42 + 0.1 * bend, -0.25 * bend]
        lm[ank] = [fx, 0.85, 0.0]
        lm[foot] = [fx, 0.88, -0.15]
    return lm


def _stream(amp=1.2, reps=5, fps=15.0, rep_s=2.0):
    frames = [_person(0)] * int(3 * fps)
    for _ in range(reps):
        for a in np.linspace(0, 2 * np.pi, int(rep_s * fps), endpoint=False):
            frames.append(_person(amp * (1 - np.cos(a)) / 2))
    frames += [_person(0)] * int(2 * fps)
    return frames


def _run(frames, fps=15.0):
    clf = RuleSquatClassifier(0.4, 0.2)
    for i, f in enumerate(frames):
        clf.push(f, i / fps, 1.0)
    return clf.verdict(), clf


def test_full_squats_are_confirmed():
    v, clf = _run(_stream())
    assert v["squat"] and v["valid_reps"] >= 4
    assert all(cy["plausible"] for cy in clf.cycles)


def test_shallow_dips_are_counted_but_not_squats():
    v, clf = _run(_stream(amp=0.35))
    assert not v["squat"] and v["valid_reps"] == 0
    assert all("depth" in cy["failed"] for cy in clf.cycles)


def test_standing_still_gives_no_cycles():
    v, clf = _run([_person(0)] * 150)
    assert v["cycles"] == 0 and not v["squat"] and clf.phase == "STANDING"


def test_horizontal_body_never_starts():
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)      # lie the person down
    v, clf = _run([f @ R.T for f in _stream()])
    assert v["cycles"] == 0 and clf.phase == "UNREADY"


def test_frame_rate_does_not_change_the_decision():
    v15, _ = _run(_stream(fps=15.0), fps=15.0)
    v30, _ = _run(_stream(fps=30.0), fps=30.0)
    assert v15["squat"] and v30["squat"] and abs(v15["valid_reps"] - v30["valid_reps"]) <= 1


def test_low_visibility_frames_are_ignored():
    clf = RuleSquatClassifier(0.4, 0.2)
    for i, f in enumerate(_stream()):
        clf.push(f, i / 15.0, 0.1)
    assert clf.verdict()["cycles"] == 0 and clf.phase == "UNREADY"


class _P:
    def __init__(self, xyz, vis):
        self.x, self.y, self.z = xyz
        self.visibility = vis


def test_runner_uses_mean_leg_visibility_like_the_cache():
    """Side view: the far leg is hidden. The MEAN over hips/knees/ankles (the cached
    leg_vis definition) keeps the frame; a MIN would drop every frame."""
    r = RuleSquatRunner(0.4, 0.2)
    for i, f in enumerate(_stream()):
        pts = [_P(f[j], 0.1 if j in (24, 26, 28) else 0.95) for j in range(33)]   # right leg hidden
        r.update(pts, now=i / 15.0)
    assert r.summary()["confirmed"] >= 4


def test_runner_interface_and_stance():
    r = RuleSquatRunner(0.4, 0.2)
    for i, f in enumerate(_stream()):
        st = r.update(f, now=i / 15.0)
    s = r.summary()
    assert s["confirmed"] >= 4 and s["classifier"] == "rules" and r.confirmed_count == s["confirmed"]
    assert all(rep.squat_score >= RuleConfig().score_threshold for rep in r.reps if rep.confirmed)
    assert s["stance"]["recommendation"] in ("NARROW", "MODERATE", "WIDE")
    assert st.rep_count == len(r.reps)


def _turn(frames, deg):
    a = np.radians(deg)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    return [f @ R.T for f in frames]


def test_facing_angle_and_stance_needs_a_front_view():
    from stancesense.squat.rule_classifier import facing_angle
    assert facing_angle(_person(0)) < 1 and abs(facing_angle(_turn([_person(0)], 80)[0]) - 80) < 1
    front = RuleSquatRunner(0.4, 0.2)
    side = RuleSquatRunner(0.4, 0.2)
    for i, (f, g) in enumerate(zip(_stream(), _turn(_stream(), 75))):
        front.update(f, now=i / 15.0)
        side.update(g, now=i / 15.0)
    assert front.facing_ok and front.summary()["stance"]["front_facing"]
    s = side.summary()
    assert not side.facing_ok and s["confirmed"] >= 4                   # still counted as squats
    assert not s["stance"]["front_facing"] and "facing the camera" in s["stance"]["reason"]


def test_stance_bands_and_rules():
    assert (stance_class(1.0), stance_class(1.8), stance_class(2.5)) == ("NARROW", "MODERATE", "WIDE")
    base = dict(stance_width=1.0, max_depth_ratio=0.4, max_torso_deg=50.0, knee_width_bottom=1.0, stance_bottom=1.0,
                leg_drop_asym=0.05, knee_over_foot_max=0.1)
    assert stance_recommendation([base])["recommendation"] == "MODERATE"     # shallow AND lean -> wider
    lean_only = dict(base, max_depth_ratio=0.9)
    assert stance_recommendation([lean_only])["recommendation"] == "NARROW"  # deep squat with lean: keep
    shallow_only = dict(base, max_torso_deg=20.0)
    assert stance_recommendation([shallow_only])["recommendation"] == "NARROW"   # upright half squat: keep
    ok = dict(base, stance_width=1.8, max_depth_ratio=1.0, max_torso_deg=30.0)
    assert stance_recommendation([ok])["recommendation"] == "MODERATE"       # fine -> keep
    cave = dict(ok, stance_width=2.5, stance_bottom=2.5, knee_width_bottom=1.5)
    assert stance_recommendation([cave])["recommendation"] == "MODERATE"     # wide + knees in -> narrower
    assert stance_recommendation([]) is None
