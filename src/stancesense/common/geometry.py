"""Tiny, dependency-light geometry helpers.

Keeping these in one place means the *core measurement* (hip rotation angle)
is built from functions that are individually unit-tested with known inputs.
"""
from __future__ import annotations
import numpy as np


def dot2(u, v) -> float:
    """2-D dot product of two (x, y) vectors."""
    return float(u[0] * v[0] + u[1] * v[1])


def cross2(u, v) -> float:
    """2-D cross product (scalar z-component). Sign tells us rotation direction."""
    return float(u[0] * v[1] - u[1] * v[0])


def angle_between(u, v) -> float:
    """Unsigned angle (degrees) between two vectors of any dimension."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    cos_t = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_t)))


def signed_angle_2d(u, v) -> float:
    """Signed angle (degrees) FROM vector u TO vector v in the 2-D plane.

    Uses atan2(cross, dot) so the result is in (-180, 180].  Positive means v is
    counter-clockwise from u.  This is the workhorse behind the shank-angle
    measurement: u = straight-down reference, v = the shank vector.
    """
    return float(np.degrees(np.arctan2(cross2(u, v), dot2(u, v))))
