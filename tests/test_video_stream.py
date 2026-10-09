"""Tests for webcam ingestion - specifically its failure modes.

Most users hitting these paths have no camera attached, so the errors have to
say what to do instead of failing silently.
"""
import pytest

from stancesense.ingestion.video_stream import VideoStream


class _FakeCap:
    def __init__(self, opened):
        self._opened = opened
        self.released = False

    def isOpened(self):
        return self._opened

    def set(self, *_a):
        return True

    def read(self):
        return True, "frame"

    def release(self):
        self.released = True


@pytest.fixture()
def fake_cv2(monkeypatch):
    import stancesense.ingestion.video_stream as vs

    class _CV2:
        CAP_PROP_FRAME_WIDTH = CAP_PROP_FRAME_HEIGHT = CAP_PROP_FPS = 0
        made = []

        @staticmethod
        def VideoCapture(src):
            cap = _FakeCap(_CV2.opened)
            _CV2.made.append(cap)
            return cap

    _CV2.made = []
    monkeypatch.setattr(vs, "cv2", _CV2)
    return _CV2


def test_a_missing_camera_raises_an_actionable_error(fake_cv2):
    fake_cv2.opened = False
    with pytest.raises(RuntimeError) as e:
        VideoStream("config/default.yaml").open()
    msg = str(e.value)
    assert "could not open camera" in msg
    assert "config/default.yaml" in msg          # how to point at another camera
    assert "demo_end_to_end_mmfit" in msg        # how to proceed with no camera


def test_a_failed_open_releases_the_capture(fake_cv2):
    fake_cv2.opened = False
    with pytest.raises(RuntimeError):
        VideoStream("config/default.yaml").open()
    assert fake_cv2.made[0].released is True     # no leaked device handle


def test_reading_before_open_explains_the_context_manager():
    with pytest.raises(RuntimeError, match="context manager"):
        VideoStream("config/default.yaml").read()


def test_a_working_camera_opens_and_reads(fake_cv2):
    fake_cv2.opened = True
    with VideoStream("config/default.yaml") as stream:
        ok, frame, ts = stream.read()
    assert ok is True and frame == "frame" and ts >= 0.0
    assert fake_cv2.made[0].released is True     # context manager cleans up
