"""Phase 2 — pose estimation.

Wraps MediaPipe Pose (BlazePose GHUM 3D).  We expose BOTH landmark sets:

* image landmarks  -> normalised (0..1) pixel positions, good for DRAWING.
* world landmarks  -> metres, hip-centred, good for MEASURING angles.

All geometry in this project uses the WORLD landmarks.
"""
from __future__ import annotations
import yaml

try:
    import mediapipe as mp
except ImportError:
    mp = None


class PoseEstimator:
    def __init__(self, config_path: str = "config/default.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)["pose"]
        if mp is None:
            raise RuntimeError("mediapipe is not installed.")
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            model_complexity=cfg["model_complexity"],
            smooth_landmarks=cfg["smooth_landmarks"],
            min_detection_confidence=cfg["min_detection_confidence"],
            min_tracking_confidence=cfg["min_tracking_confidence"],
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def infer(self, frame_rgb):
        """Run pose on an RGB frame. Returns (image_landmarks, world_landmarks)."""
        result = self._pose.process(frame_rgb)
        return result.pose_landmarks, result.pose_world_landmarks

    def draw(self, image_bgr, image_landmarks):
        """Draw the skeleton onto a BGR image in place."""
        if image_landmarks is not None:
            self.mp_drawing.draw_landmarks(
                image_bgr, image_landmarks, self._mp_pose.POSE_CONNECTIONS
            )
        return image_bgr
