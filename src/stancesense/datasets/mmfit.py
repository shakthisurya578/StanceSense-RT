"""MM-Fit dataset adapter (Phase 8 input layer).

WHY THIS EXISTS
---------------
Recording our own squat clips is not currently possible, so the squat rep/depth
engine (Phases 7b-10) is verified on the public **MM-Fit** dataset
(Stromback, Huang & Radu, IMWUT 2020, https://mmfit.github.io/).  MM-Fit ships
time-synchronised 3D pose for 10 gym exercises with per-set repetition counts, so
it is a real benchmark for a squat counter/analyser - but NOT a person-
independent one by workout: its 21 workouts come from only 10 people (two did six
sessions each, one did two; Stromback et al. 2020, s3.1). Holding out a workout
does not hold out a person. The dataset authors' unseen-person test split is
w00, w05, w12, w13, w20 (github.com/KDMStromback/mm-fit).

WHAT MM-FIT GIVES US (verified against the real files, workouts w00-w20)
-----------------------------------------------------------------------
Layout:   <root>/wNN/wNN_pose_3d.npy   and   <root>/wNN/wNN_labels.csv
pose_3d:  float array, shape (3, N_frames, 18).
          axis 0 = (x, y, z);  axis 1 = time;  axis 2 = 18 columns where
          column 0 is the integer FRAME ID and columns 1..17 are 17 joints.
labels:   CSV rows  ``start_frame, end_frame, repetitions, exercise_name``.
          The squat rows carry exercise_name == "squats".

JOINT ORDER (Human3.6M 17-joint layout, confirmed empirically from w00 by the
vertical (+z up) ordering ankle < knee < hip < ... < head):

    0 pelvis   1 RHip 2 RKnee 3 RAnkle   4 LHip 5 LKnee 6 LAnkle
    7 spine    8 thorax 9 neck/nose 10 head
    11 LShoulder 12 LElbow 13 LWrist    14 RShoulder 15 RElbow 16 RWrist

COORDINATE CONVENTION
---------------------
MM-Fit world units are millimetre-scale; **+z is up**.  The StanceSense squat
kinematics were written for MediaPipe world landmarks where **+y is DOWN**.  The
adapter therefore remaps each joint to a MediaPipe-style (x, y, z) with
``y = -z_mmfit`` so ``depth_ratio`` and ``trunk_lean`` keep their documented
meaning.  Joint angles (knee flexion) are full-3D and convention-independent.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from glob import glob
from typing import Iterator

import numpy as np


# --- Human3.6M 17-joint indices in the MM-Fit pose_3d array (after the frame col) --
class H36M:
    PELVIS = 0
    R_HIP, R_KNEE, R_ANKLE = 1, 2, 3
    L_HIP, L_KNEE, L_ANKLE = 4, 5, 6
    SPINE, THORAX, NECK, HEAD = 7, 8, 9, 10
    L_SHOULDER, L_ELBOW, L_WRIST = 11, 12, 13
    R_SHOULDER, R_ELBOW, R_WRIST = 14, 15, 16


# --- MediaPipe BlazePose indices the squat kinematics read (see kinematics/rotation.py) --
class MP:
    L_SHOULDER, R_SHOULDER = 11, 12
    L_HIP, R_HIP = 23, 24
    L_KNEE, R_KNEE = 25, 26
    L_ANKLE, R_ANKLE = 27, 28
    L_FOOT, R_FOOT = 31, 32
    N = 33


# MM-Fit(H36M joint) -> MediaPipe landmark index.  Only the lower-body + shoulder
# joints the squat engine needs are mapped; the rest are left as zeros.
_H36M_TO_MP = {
    H36M.L_SHOULDER: MP.L_SHOULDER,
    H36M.R_SHOULDER: MP.R_SHOULDER,
    H36M.L_HIP: MP.L_HIP,
    H36M.R_HIP: MP.R_HIP,
    H36M.L_KNEE: MP.L_KNEE,
    H36M.R_KNEE: MP.R_KNEE,
    H36M.L_ANKLE: MP.L_ANKLE,
    H36M.R_ANKLE: MP.R_ANKLE,
}

SQUAT_LABEL = "squats"


@dataclass
class SquatSet:
    """One labelled squat set inside a workout."""
    workout_id: str
    start_frame: int
    end_frame: int
    true_reps: int
    landmarks: np.ndarray          # (n_frames, 33, 3), MediaPipe-style coords
    frame_ids: np.ndarray          # (n_frames,) MM-Fit frame ids


@dataclass
class MMFitWorkout:
    """A single MM-Fit workout: its 3D pose and its label rows."""
    workout_id: str
    pose3d: np.ndarray             # (3, N, 18) raw MM-Fit array
    labels: list                   # list[ (start, end, reps, exercise) ]
    path: str = ""

    @property
    def frame_ids(self) -> np.ndarray:
        return self.pose3d[0, :, 0].astype(int)


# --------------------------------------------------------------------------- IO

def load_labels(path: str) -> list:
    """Read a MM-Fit label CSV -> list of (start, end, reps, exercise)."""
    rows = []
    with open(path, "r", newline="") as fh:
        for line in csv.reader(fh):
            if not line or len(line) < 4:
                continue
            rows.append((int(line[0]), int(line[1]), int(line[2]), line[3].strip()))
    return rows


def load_workout(workout_dir: str) -> MMFitWorkout:
    """Load one ``wNN`` folder into an :class:`MMFitWorkout`."""
    wid = os.path.basename(os.path.normpath(workout_dir))
    pose_path = os.path.join(workout_dir, f"{wid}_pose_3d.npy")
    label_path = os.path.join(workout_dir, f"{wid}_labels.csv")
    pose = np.load(pose_path)
    if pose.ndim != 3 or pose.shape[0] != 3:
        raise ValueError(f"{pose_path}: expected (3, N, 18), got {pose.shape}")
    return MMFitWorkout(wid, pose, load_labels(label_path), workout_dir)


def discover_workouts(root: str) -> list:
    """Return sorted workout directories under ``root`` that have pose+labels.

    Accepts either the MM-Fit root (containing ``wNN/``) or a flat folder that
    holds the ``wNN_pose_3d.npy`` / ``wNN_labels.csv`` files directly.
    """
    dirs = []
    for d in sorted(glob(os.path.join(root, "w[0-9]*"))):
        if os.path.isdir(d):
            wid = os.path.basename(d)
            if os.path.exists(os.path.join(d, f"{wid}_pose_3d.npy")):
                dirs.append(d)
    if dirs:
        return dirs
    # flat layout: synthesise per-workout access from files in `root`
    flat = []
    for p in sorted(glob(os.path.join(root, "w[0-9]*_pose_3d.npy"))):
        wid = os.path.basename(p).replace("_pose_3d.npy", "")
        if os.path.exists(os.path.join(root, f"{wid}_labels.csv")):
            flat.append(("__flat__", root, wid))
    return flat  # sentinel handled by load_flat_workout


def load_flat_workout(root: str, wid: str) -> MMFitWorkout:
    """Load a workout whose files sit directly in ``root`` (no wNN subfolder)."""
    pose = np.load(os.path.join(root, f"{wid}_pose_3d.npy"))
    labels = load_labels(os.path.join(root, f"{wid}_labels.csv"))
    return MMFitWorkout(wid, pose, labels, root)


# ----------------------------------------------------------------- adaptation

def landmarks_from_pose3d(pose3d: np.ndarray) -> np.ndarray:
    """Map a MM-Fit ``pose_3d`` array to MediaPipe-style landmarks.

    Parameters
    ----------
    pose3d : (3, N, 18) MM-Fit array (column 0 is the frame id).

    Returns
    -------
    (N, 33, 3) float array.  Index it with the ``MP`` / MediaPipe indices; the
    y axis points DOWN (``y = -z_mmfit``) to match the squat kinematics.
    Unmapped landmarks are zero.
    """
    joints = pose3d[:, :, 1:]                 # (3, N, 17)  drop frame-id column
    x, y, z = joints[0], joints[1], joints[2]  # each (N, 17)
    n = x.shape[0]
    lm = np.zeros((n, MP.N, 3), dtype=float)
    for h, m in _H36M_TO_MP.items():
        lm[:, m, 0] = x[:, h]                  # right/left
        lm[:, m, 1] = -z[:, h]                 # up(+z) -> down(+y)
        lm[:, m, 2] = y[:, h]                  # depth (unused by squat metrics)
    # MM-Fit has no foot-index joint; approximate it with the ankle so that
    # detect_stance() runs (toe-out then reads ~0, width from ankles is valid).
    lm[:, MP.L_FOOT] = lm[:, MP.L_ANKLE]
    lm[:, MP.R_FOOT] = lm[:, MP.R_ANKLE]
    return lm


def estimate_scale(landmarks: np.ndarray) -> tuple:
    """Robust (femur_length, hip_width) in MM-Fit units from a landmark clip.

    Uses the median over frames so a few bad poses do not distort the
    normalisation constants that depth_ratio / knee_over_foot_dev rely on.
    """
    lh, rh = landmarks[:, MP.L_HIP], landmarks[:, MP.R_HIP]
    lk, rk = landmarks[:, MP.L_KNEE], landmarks[:, MP.R_KNEE]
    femur = np.concatenate([
        np.linalg.norm(lh - lk, axis=1),
        np.linalg.norm(rh - rk, axis=1),
    ])
    hip_w = np.linalg.norm(lh - rh, axis=1)
    fem = float(np.median(femur[femur > 0])) if np.any(femur > 0) else 1.0
    hw = float(np.median(hip_w[hip_w > 0])) if np.any(hip_w > 0) else 1.0
    return fem, hw


def _slice_by_frames(workout: MMFitWorkout, start: int, end: int) -> np.ndarray:
    """Row indices of the workout whose frame id falls in [start, end]."""
    fids = workout.frame_ids
    lo = int(np.searchsorted(fids, start, "left"))
    hi = int(np.searchsorted(fids, end, "right"))
    return np.arange(lo, hi)


def squat_sets(workout: MMFitWorkout) -> Iterator[SquatSet]:
    """Yield every labelled squat set in a workout as ready-to-use landmarks."""
    all_lm = landmarks_from_pose3d(workout.pose3d)
    fids = workout.frame_ids
    for start, end, reps, name in workout.labels:
        if name != SQUAT_LABEL:
            continue
        idx = _slice_by_frames(workout, start, end)
        if idx.size == 0:
            continue
        yield SquatSet(
            workout_id=workout.workout_id,
            start_frame=start,
            end_frame=end,
            true_reps=reps,
            landmarks=all_lm[idx],
            frame_ids=fids[idx],
        )
