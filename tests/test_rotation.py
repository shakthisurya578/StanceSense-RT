"""Verify the core measurement on synthetic shank vectors of KNOWN angle."""
import math
import pytest
from stancesense.kinematics.rotation import (
    hip_rotation_angle_from_points, signed_shank_angle_deg)
from stancesense.common.types import LEFT, RIGHT


def shank(deg_from_vertical, to_plus_x=True):
    """Build (knee, ankle) so the shank tilts `deg` from straight-down."""
    s = math.sin(math.radians(deg_from_vertical))
    c = math.cos(math.radians(deg_from_vertical))
    ankle = ((s if to_plus_x else -s), c)     # knee at origin, ankle below+side
    return (0.0, 0.0), ankle


@pytest.mark.parametrize("deg", [0, 10, 25, 40])
def test_vertical_and_magnitude(deg):
    knee, ankle = shank(deg, to_plus_x=True)
    assert abs(abs(signed_shank_angle_deg(knee, ankle)) - deg) < 1e-6


def test_zero_at_vertical():
    knee, ankle = shank(0)
    assert abs(hip_rotation_angle_from_points(knee, ankle, RIGHT)) < 1e-9
    assert abs(hip_rotation_angle_from_points(knee, ankle, LEFT)) < 1e-9


def test_internal_positive_both_legs():
    # RIGHT leg internal = ankle swings to -x ; LEFT leg internal = ankle to +x
    kR, aR = shank(30, to_plus_x=False)
    kL, aL = shank(30, to_plus_x=True)
    assert hip_rotation_angle_from_points(kR, aR, RIGHT) == pytest.approx(30, abs=1e-6)
    assert hip_rotation_angle_from_points(kL, aL, LEFT) == pytest.approx(30, abs=1e-6)


def test_external_negative():
    kR, aR = shank(30, to_plus_x=True)   # RIGHT leg ankle to +x = external
    assert hip_rotation_angle_from_points(kR, aR, RIGHT) == pytest.approx(-30, abs=1e-6)
