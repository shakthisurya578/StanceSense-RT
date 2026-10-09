import numpy as np
import pytest

from stancesense.squat.kinematics import detect_stance


# --------------------------------------------------- stance detection geometry

def _stance_frame(yaw_deg=0.0, toe_deg=20.0, half_stance=0.15, hip_half=0.1):
    """Standing frame with a known stance width and toe-out, rotated by yaw.

    y points DOWN, so the ground plane is (x, z) and the feet point along +z.
    """
    lm = np.zeros((33, 3))
    pts = {
        23: [-hip_half, -0.9, 0.0], 24: [hip_half, -0.9, 0.0],
        25: [-hip_half, -0.5, 0.05], 26: [hip_half, -0.5, 0.05],
        27: [-half_stance, 0.0, 0.0], 28: [half_stance, 0.0, 0.0],
        11: [-0.15, -1.3, 0.0], 12: [0.15, -1.3, 0.0],
    }
    t, L = np.radians(toe_deg), 0.13
    pts[31] = [pts[27][0] - L * np.sin(t), 0.0, L * np.cos(t)]   # toes point out
    pts[32] = [pts[28][0] + L * np.sin(t), 0.0, L * np.cos(t)]
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    for i, (x, y, z) in pts.items():
        lm[i] = [c * x - s * z, y, s * x + c * z]
    return lm


def test_detect_stance_recovers_the_true_width_and_toe_out():
    width, toe = detect_stance(_stance_frame(half_stance=0.15, toe_deg=20.0), 0.2)
    assert width == pytest.approx(1.5, abs=1e-3)     # 0.30 sep / 0.20 hip width
    assert toe == pytest.approx(20.0, abs=0.2)


def test_detect_stance_is_invariant_to_which_way_the_lifter_faces():
    """Both readings live in the ground plane, so yaw must not change them."""
    ref = detect_stance(_stance_frame(yaw_deg=0), 0.2)
    for yaw in (30, 60, 90, 135, -45):
        assert detect_stance(_stance_frame(yaw_deg=yaw), 0.2) == pytest.approx(ref, abs=0.2)


@pytest.mark.parametrize("toe_deg", [0.0, 10.0, 25.0, 40.0])
def test_toe_out_tracks_the_real_foot_angle(toe_deg):
    """A flat foot pointing straight ahead must read 0, not 90."""
    assert detect_stance(_stance_frame(toe_deg=toe_deg), 0.2)[1] == pytest.approx(
        toe_deg, abs=0.2)


def test_toe_out_is_zero_when_there_is_no_foot_landmark():
    """MM-Fit has no foot-index joint; the adapter sets foot == ankle."""
    lm = _stance_frame()
    lm[31], lm[32] = lm[27], lm[28]
    assert detect_stance(lm, 0.2)[1] == 0.0


def test_stance_width_ignores_standing_height():
    """Width must not pick up the vertical axis."""
    lm = _stance_frame()
    raised = lm.copy()
    raised[:, 1] -= 0.4                              # whole body lifted
    assert detect_stance(raised, 0.2)[0] == pytest.approx(detect_stance(lm, 0.2)[0])


def test_calibrator_femur_averages_both_legs():
    """One badly-tracked knee must not skew the depth normalisation constant."""
    from stancesense.kinematics.calibrator import Calibrator
    lm = _stance_frame()
    lm[25] = [-0.1, -0.5, 0.0]      # left knee  -> femur 0.4
    lm[26] = [0.1, -0.3, 0.0]       # right knee -> femur 0.6 (mis-tracked)
    c = Calibrator(needed_frames=1)
    c.update(lm)
    c.finalize()
    assert c.femur_length == pytest.approx(0.5, abs=1e-6)   # mean of 0.4 and 0.6
