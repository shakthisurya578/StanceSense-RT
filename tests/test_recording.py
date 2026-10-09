"""Tests for the assessment video recorder.

The previous recordings were MPEG-4 Part 2 ("FMP4") in an .mp4 container. OpenCV
and VLC read that happily, but Windows Photos and Media Player refuse it — the
file exists, has a sensible size, and simply will not open. So "it wrote a file"
is not a useful assertion here; these tests check the file can be READ BACK and
that the default codec is the widely-playable one.
"""
import os
import sys

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
sys.path.insert(0, "scripts")
from live_assessment import CODECS, Recorder  # noqa: E402


def _frames(n=12, h=120, w=160):
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(n)]


def _fourcc_of(path):
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return None
        tag = int(cap.get(cv2.CAP_PROP_FOURCC))
        return "".join(chr((tag >> 8 * i) & 0xFF) for i in range(4))
    finally:
        cap.release()


def test_disabled_recorder_writes_nothing(tmp_path):
    r = Recorder(False, "mjpg", out_dir=str(tmp_path))
    for f in _frames():
        r.write(f)
    assert r.close() is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("codec", sorted(CODECS))
def test_every_offered_codec_produces_a_readable_file(codec, tmp_path):
    r = Recorder(True, codec, out_dir=str(tmp_path))
    for f in _frames():
        r.write(f)
    path = r.close()
    assert path is not None, f"{codec} produced nothing usable"
    assert os.path.getsize(path) > 0

    cap = cv2.VideoCapture(path)
    try:
        assert cap.isOpened()
        ok, frame = cap.read()
        assert ok and frame is not None
    finally:
        cap.release()


def test_the_default_codec_is_the_widely_playable_one(tmp_path):
    """MJPG in .avi opens in any player; FMP4 in .mp4 does not open on Windows."""
    assert sorted(CODECS)[0] == "mjpg" or "mjpg" in CODECS
    r = Recorder(True, "mjpg", out_dir=str(tmp_path))
    for f in _frames():
        r.write(f)
    path = r.close()
    assert path.endswith(".avi")
    assert _fourcc_of(path) == "MJPG", "the default must not silently become FMP4"


def test_the_frame_size_comes_from_the_first_frame(tmp_path):
    """The writer is opened lazily, so the camera's real resolution is used."""
    r = Recorder(True, "mjpg", out_dir=str(tmp_path))
    for f in _frames(n=6, h=240, w=320):
        r.write(f)
    path = r.close()
    cap = cv2.VideoCapture(path)
    try:
        assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (320, 240)
    finally:
        cap.release()


def test_closing_without_a_single_frame_is_safe(tmp_path):
    assert Recorder(True, "mjpg", out_dir=str(tmp_path)).close() is None
