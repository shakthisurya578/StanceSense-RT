"""Per-frame signals the rule-based classifier reads (squat/signals.py)."""
import numpy as np

from stancesense.squat.signals import FrameState, as_array, frame_signals


def _person(knee_bend=0.0):
    """(33, 3) world landmarks, y down, facing -z; knee_bend 0 = standing, 1 = deep."""
    lm = np.zeros((33, 3))
    hip_y = -0.0 + 0.30 * knee_bend               # hips drop as the knees bend
    for side, x in ((0, 0.1), (1, -0.1)):
        sh, hip, knee, ank, foot = (11, 23, 25, 27, 31) if side == 0 else (12, 24, 26, 28, 32)
        lm[sh] = [x * 1.6, hip_y - 0.5, 0.0]
        lm[hip] = [x, hip_y, 0.0]
        lm[ank] = [x * 1.3, 0.85, 0.0]
        lm[knee] = [x * 1.2, 0.42 + 0.1 * knee_bend, -0.25 * knee_bend]
        lm[foot] = [x * 1.3, 0.88, -0.15]
    return lm


def test_signals_use_y_down():
    s = frame_signals(np.stack([_person(0), _person(1)]), 0.4, 0.2)
    assert s["hip_height"][0] > s["hip_height"][1]              # hips drop in the squat
    assert s["knee_L"][0] > s["knee_L"][1]                      # knee closes
    assert s["depth"][1] > s["depth"][0]                        # deeper at the bottom
    assert s["torso"][0] < 5


def test_signals_ignore_turning_and_body_size():
    lm = np.stack([_person(0), _person(0.6), _person(1)])
    ref = frame_signals(lm, 0.4, 0.2)
    a = np.radians(40)                                          # turn about the vertical (y) axis
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    for other in (frame_signals(lm @ R.T, 0.4, 0.2),
                  frame_signals(lm * 1.3, 0.4 * 1.3, 0.2 * 1.3)):
        for key in ("knee", "hip", "depth", "hip_height", "stance_width", "knee_width", "torso"):
            np.testing.assert_allclose(other[key], ref[key], rtol=1e-4, atol=1e-2, err_msg=key)


def test_landmark_objects_and_arrays_are_equivalent():
    class P:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z
    lm = _person(0.5)
    np.testing.assert_array_equal(as_array([P(*p) for p in lm]), lm)
    np.testing.assert_array_equal(as_array(lm), lm)


def test_frame_state_defaults():
    st = FrameState(phase="STANDING", knee_angle=175.0, depth=0.0, rep_count=0)
    assert st.new_rep is None and st.p_squat is None
