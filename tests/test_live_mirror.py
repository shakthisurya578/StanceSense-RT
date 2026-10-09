"""The mirror must never reach the pose model.

`cv2.flip(frame, 1)` before inference swaps MediaPipe's anatomical left/right
labels — verified 8/8 on a real recording by comparing sign(left.x - right.x)
before and after mirroring. The consequence in the app was that the LEFT-leg
prompts measured the lifter's RIGHT leg, so one leg's extremes could never be
captured and its profile values came from the other leg.

Nothing in a unit test can detect a swapped label from MediaPipe itself, so
these pin the ORDER instead: pose sees the raw frame, the mirror is applied
afterwards for display.
"""
import sys

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
sys.path.insert(0, "scripts")
from live_assessment import infer_and_annotate  # noqa: E402


class _FakePose:
    """Records the frame it was asked to run on."""

    def __init__(self):
        self.seen = None

    def infer(self, frame_rgb):
        self.seen = frame_rgb.copy()
        return None, None            # no landmarks -> controller reports untracked

    def draw(self, image_bgr, image_landmarks):
        return image_bgr


class _FakeCtrl:
    stage = "NEUTRAL"
    runner = None

    def update(self, world, now):
        from stancesense.app import UiState
        return UiState(stage="NEUTRAL", side="LEFT", title="t", cue="c",
                       tracking=False)


def _sided_frame(h=120, w=160):
    """Left half white, right half black — so a mirror is unmistakable."""
    f = np.zeros((h, w, 3), dtype=np.uint8)
    f[:, : w // 2] = 255
    return f


def test_pose_runs_on_the_unmirrored_frame():
    pose = _FakePose()
    infer_and_annotate(_sided_frame(), pose, _FakeCtrl(), 0.0)
    seen = pose.seen
    assert seen is not None
    left_mean = seen[:, : seen.shape[1] // 2].mean()
    right_mean = seen[:, seen.shape[1] // 2:].mean()
    assert left_mean > right_mean, (
        "the pose model was given a mirrored frame — MediaPipe's left/right "
        "labels would be swapped and the wrong leg measured")


def test_the_displayed_frame_is_mirrored():
    """The preview must still read like a mirror for the user."""
    out, _state = infer_and_annotate(_sided_frame(), _FakePose(), _FakeCtrl(), 0.0)
    left_mean = out[:, : out.shape[1] // 2].mean()
    right_mean = out[:, out.shape[1] // 2:].mean()
    assert right_mean > left_mean
