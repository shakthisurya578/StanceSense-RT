"""Optional recording of the annotated assessment video."""
from __future__ import annotations

import os
import time

import cv2

# Windows Photos / Media Player refuse MPEG-4 Part 2 ("FMP4") in an .mp4
# container, which is what cv2's "mp4v" writes - the file looks fine to OpenCV
# and VLC but will not open in the default player. Motion-JPEG in .avi plays
# essentially everywhere with no extra codec, so it is the default here; the
# smaller mp4v is still available for anyone who plays files in VLC.
# ("xvid" is deliberately absent: OpenCV silently writes FMP4 for it here, so it
# would be an alias for mp4v with a misleading name.)
CODECS = {
    "mjpg": ("MJPG", ".avi"),      # larger, but opens in any player
    "mp4v": ("mp4v", ".mp4"),      # smaller; writes FMP4, often unplayable on Windows
}


class Recorder:
    """Writes the annotated frames to a file that actually opens. Optional."""

    def __init__(self, enabled: bool, codec: str, out_dir: str = "recordings",
                 fps: float = 20.0):
        self.path = None
        self._writer = None
        self._frames = 0
        self._fps = float(fps)
        if not enabled:
            return
        fourcc, ext = CODECS[codec]
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(
            out_dir, time.strftime("assessment_%Y%m%d_%H%M%S") + ext)
        self._fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self._codec = codec

    def write(self, frame):
        if self.path is None:
            return
        if self._writer is None:                   # size is known at first frame
            h, w = frame.shape[:2]
            self._writer = cv2.VideoWriter(self.path, self._fourcc, self._fps, (w, h))
            if not self._writer.isOpened():
                print(f"[record] {self._codec} unavailable - not recording")
                self._writer, self.path = None, None
                return
        self._writer.write(frame)
        self._frames += 1

    def close(self):
        """Release, then prove the file can actually be read back."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self.path is None or not os.path.exists(self.path):
            return None
        cap = cv2.VideoCapture(self.path)
        readable = cap.isOpened() and cap.read()[0]
        cap.release()
        if not readable:
            print(f"[record] {self.path} was written but cannot be read back")
            return None
        mb = os.path.getsize(self.path) / 1e6
        print(f"\n[record] {self._frames} frames -> {os.path.abspath(self.path)} "
              f"({mb:.1f} MB)")
        return self.path
