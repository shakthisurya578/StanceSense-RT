"""Dataset adapters for StanceSense-RT verification.

Public datasets used to VERIFY the squat rep/depth engine (Phases 8-10) without
recording our own clips.  Currently: MM-Fit (Stromback et al., 2020).
"""
from .mmfit import (
    MMFitWorkout,
    discover_workouts,
    load_workout,
    landmarks_from_pose3d,
    squat_sets,
    estimate_scale,
    H36M, MP,
)

__all__ = [
    "MMFitWorkout",
    "discover_workouts",
    "load_workout",
    "landmarks_from_pose3d",
    "squat_sets",
    "estimate_scale",
    "H36M", "MP",
]
