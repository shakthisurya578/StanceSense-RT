"""Tests for the MM-Fit adapter (used by the offline demo) and the squat kinematics."""
import numpy as np

from stancesense.datasets.mmfit import (
    MMFitWorkout, landmarks_from_pose3d, squat_sets, estimate_scale, H36M, MP,
)
from stancesense.squat.kinematics import knee_flexion, depth_ratio, trunk_lean
from stancesense.common.types import LEFT


def _standing_joints():
    """A crude H36M skeleton (17 joints, +z up), standing, in mm."""
    j = np.zeros((17, 3))
    j[H36M.PELVIS]   = [0,   0, 900]
    j[H36M.R_HIP]    = [-100, 0, 900]; j[H36M.L_HIP] = [100, 0, 900]
    j[H36M.R_KNEE]   = [-100, 0, 500]; j[H36M.L_KNEE] = [100, 0, 500]
    j[H36M.R_ANKLE]  = [-100, 0, 100]; j[H36M.L_ANKLE] = [100, 0, 100]
    j[H36M.R_SHOULDER] = [-150, 0, 1300]; j[H36M.L_SHOULDER] = [150, 0, 1300]
    j[H36M.HEAD]     = [0,   0, 1500]
    return j


def _make_workout(n_frames=200, squat_reps=5):
    """Synthetic MM-Fit workout: frames 0..n with one labelled squat set."""
    base = _standing_joints()
    frames = []
    for t in range(n_frames):
        j = base.copy()
        # squat oscillation only inside [50, 150]
        if 50 <= t < 150:
            phase = np.sin((t - 50) / 100 * squat_reps * 2 * np.pi)
            drop = max(0.0, (1 - phase) / 2) * 350   # hips/knees dip
            j[[H36M.PELVIS, H36M.R_HIP, H36M.L_HIP], 2] -= drop
            j[[H36M.R_KNEE, H36M.L_KNEE], 2] -= drop * 0.3
            j[[H36M.R_KNEE, H36M.L_KNEE], 1] += drop * 0.4  # knees travel forward
        frames.append(j)
    arr = np.stack(frames, axis=1)                 # (17, N, 3)
    pose = np.zeros((3, n_frames, 18))
    pose[:, :, 0] = np.arange(n_frames)            # frame-id column
    pose[:, :, 1:] = np.transpose(arr, (2, 1, 0))  # (3, N, 17)
    labels = [(50, 149, squat_reps, "squats"), (150, 199, 3, "lunges")]
    return MMFitWorkout("wSY", pose, labels)


def test_adapter_maps_joints_and_flips_vertical():
    w = _make_workout()
    lm = landmarks_from_pose3d(w.pose3d)           # (N,33,3)
    assert lm.shape == (200, MP.N, 3)
    # +z up in MM-Fit becomes -y (down) in MediaPipe convention: hip above ankle
    hip_y = lm[0, MP.L_HIP, 1]
    ankle_y = lm[0, MP.L_ANKLE, 1]
    assert hip_y < ankle_y   # hip higher => smaller (more negative) y-down value


def test_standing_knee_is_extended():
    w = _make_workout()
    lm = landmarks_from_pose3d(w.pose3d)
    # frame 0 is standing -> knee close to 180 deg
    assert knee_flexion(lm[0], LEFT) > 165


def test_squat_sets_and_depth_changes():
    w = _make_workout()
    sets = list(squat_sets(w))
    assert len(sets) == 1                    # only the 'squats' row, not 'lunges'
    s = sets[0]
    assert s.true_reps == 5
    fem, hw = estimate_scale(s.landmarks)
    assert fem > 0 and hw > 0
    depths = np.array([depth_ratio(f, fem) for f in s.landmarks])
    assert depths.max() - depths.min() > 0.2  # the squat actually dips


# ---------------------------------------------- squat kinematics on a leg frame

def _leg_frame(l_ankle_z=0.0, r_ankle_z=0.0, l_hip_y=-0.9, r_hip_y=-0.9,
               yaw_deg=0.0):
    """A (33,3) frame with controllable foot stagger, hip drop, and facing."""
    lm = np.zeros((MP.N, 3))
    pts = {
        MP.L_HIP: [-0.1, l_hip_y, 0.0], MP.R_HIP: [0.1, r_hip_y, 0.0],
        MP.L_KNEE: [-0.1, -0.5, 0.05], MP.R_KNEE: [0.1, -0.5, 0.05],
        MP.L_ANKLE: [-0.1, 0.0, l_ankle_z], MP.R_ANKLE: [0.1, 0.0, r_ankle_z],
        MP.L_SHOULDER: [-0.15, -1.3, 0.0], MP.R_SHOULDER: [0.15, -1.3, 0.0],
    }
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    for i, (x, y, z) in pts.items():                 # rotate about the vertical
        lm[i] = [c * x - s * z, y, s * x + c * z]
    lm[MP.L_FOOT], lm[MP.R_FOOT] = lm[MP.L_ANKLE], lm[MP.R_ANKLE]
    return lm


def test_trunk_lean_measures_forward_lean_not_just_sideways():
    """Forward lean happens in the sagittal (z) plane; it must register there."""
    upright = _leg_frame()
    forward = _leg_frame()
    forward[MP.L_SHOULDER, 2] = forward[MP.R_SHOULDER, 2] = 0.5   # torso pitches forward
    assert trunk_lean(upright) < 2.0
    assert trunk_lean(forward) > 20.0


def test_knee_over_foot_dev_still_measures_lateral_knee_travel():
    from stancesense.squat.kinematics import knee_over_foot_dev
    square = _leg_frame()
    valgus = _leg_frame()
    valgus[MP.L_KNEE, 0] += 0.06        # left knee caves inward over the foot
    assert (knee_over_foot_dev(valgus, LEFT, 0.2)
            > knee_over_foot_dev(square, LEFT, 0.2) + 0.25)
