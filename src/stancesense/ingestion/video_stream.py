"""Phase 1 — webcam ingestion.

A thin, friendly wrapper around cv2.VideoCapture that reads its settings from
config/default.yaml.  Keeping capture in one class means every script opens the
camera the same way.
"""
from __future__ import annotations
import time
import yaml

try:
    import cv2
except ImportError:  # allows the module to be imported on machines without OpenCV
    cv2 = None


class VideoStream:
    def __init__(self, config_path: str = "config/default.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        cam = cfg["camera"]
        self.source = cam["source"]
        self.width = cam["width"]
        self.height = cam["height"]
        self.fps = cam["fps"]
        self._cap = None
        self._t0 = None
        self._count = 0

    def open(self):
        """Open the camera and apply the requested resolution / FPS."""
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is not installed.")
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            # Without this the capture object exists but every read() returns
            # False, so the caller's frame loop exits instantly and the user is
            # left staring at a silent, immediate exit.
            self._cap.release()
            self._cap = None
            raise RuntimeError(
                f"could not open camera source {self.source!r}. Check that a "
                f"webcam is connected and not in use by another application, or "
                f"set a different camera.source in config/default.yaml. To run "
                f"without a camera use scripts/demo_end_to_end_mmfit.py."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._t0 = time.monotonic()
        return self

    def read(self):
        """Return (ok, frame_bgr, timestamp). ok is False when the stream ends."""
        if self._cap is None:
            raise RuntimeError("VideoStream.read() before open(); use it as a "
                               "context manager: `with VideoStream() as vs:`")
        ok, frame = self._cap.read()
        ts = time.monotonic() - self._t0
        self._count += 1
        return ok, frame, ts

    def release(self):
        if self._cap is not None:
            self._cap.release()

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.release()
