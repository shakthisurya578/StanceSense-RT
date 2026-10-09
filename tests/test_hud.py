"""Camera-window drawing (app/hud.py): long banners fit, readouts clear the progress bar."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from stancesense.app.controller import SQUAT, UiState  # noqa: E402
from stancesense.app.hud import draw_hud, fit_scale  # noqa: E402


def test_a_long_locked_banner_is_shrunk_to_fit_the_frame():
    msg = "LOCKED  Rep 5 confirmed  (score 9/9)"          # 581 px at full size
    scale = fit_scale(msg, 400, 0.9, 3)
    assert scale < 0.9
    assert cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)[0][0] <= 400
    assert fit_scale("LOCKED  +21 deg", 604, 0.9, 3) == 0.9          # short text keeps its size


def _green_rows(frame):
    g = (frame[..., 1] > 200) & (frame[..., 0] < 120) & (frame[..., 2] < 120)
    return set(np.where(g.any(axis=1))[0])


def test_squat_readout_does_not_overlap_the_progress_label():
    frame = np.zeros((480, 640, 3), np.uint8)
    st = UiState(stage=SQUAT, title="Squat check", cue="Squat - 3 of 5 confirmed", progress=0.6,
                 progress_label="3/5 confirmed", rep_count=3, confirmed=3, phase="STANDING", depth=0.1,
                 just_locked="Rep 3 confirmed  (score 9/9)")
    draw_hud(frame, st)
    label_rows = set(range(480 - 58 - 10 - 16, 480 - 58 - 10 + 4))      # progress label band
    readout_rows = set(range(480 - 100 - 18, 480 - 100 + 6))            # reps / confirmed line
    assert not (label_rows & readout_rows)
    assert _green_rows(frame) & readout_rows                             # the readout was drawn there
    banner = frame[480 // 2 - 30: 480 // 2 + 10]
    green_cols = np.where(((banner[..., 1] > 200) & (banner[..., 0] < 120)).any(axis=0))[0]
    assert green_cols.min() > 0 and green_cols.max() < 639                # banner inside the frame
