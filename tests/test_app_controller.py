"""Tests for the guided-assessment controller.

The controller is the app's brain, so the whole flow is driven here with
synthetic landmarks and a fake clock - no camera, no Streamlit, no sleeping.
"""
import numpy as np
import pytest

from stancesense.app import (
    AssessmentController, CALIBRATE, NEUTRAL, EXTREME_A, EXTREME_B, SQUAT, DONE,
)
from stancesense.common.types import LEFT, RIGHT
from stancesense.kinematics.calibrator import Calibrator


def seated_frame(rot_deg_left=0.0, rot_deg_right=0.0, shank=0.4):
    """A seated pose whose shanks hang at the requested rotation angles.

    y points DOWN. A hanging shank points straight down; rotating the hip swings
    the ankle sideways, which is exactly what the rotation geometry reads.
    """
    lm = np.zeros((33, 3))
    lm[11], lm[12] = [-0.15, -0.45, 0.0], [0.15, -0.45, 0.0]      # shoulders
    lm[23], lm[24] = [-0.10, 0.0, 0.0], [0.10, 0.0, 0.0]          # hips
    lm[25], lm[26] = [-0.10, 0.10, 0.35], [0.10, 0.10, 0.35]      # knees (forward)
    for knee_i, ankle_i, deg, sign in ((25, 27, rot_deg_left, +1.0),
                                       (26, 28, rot_deg_right, -1.0)):
        # +ve angle = INTERNAL rotation; the ankle swings laterally, which is
        # image -x for the right leg and +x for the left (see rotation.py).
        t = np.radians(deg)
        knee = lm[knee_i]
        lm[ankle_i] = [knee[0] + sign * shank * np.sin(t),
                       knee[1] + shank * np.cos(t),
                       knee[2]]
    lm[31], lm[32] = lm[27], lm[28]
    return lm


def _drive(ctrl, frames, angle_l=0.0, angle_r=0.0, t0=0.0, dt=0.05):
    """Push N frames at a fixed angle; returns the last UiState."""
    state = None
    for i in range(frames):
        state = ctrl.update(seated_frame(angle_l, angle_r), t0 + i * dt)
    return state


@pytest.fixture()
def ctrl():
    calib = Calibrator(needed_frames=5)
    return AssessmentController(calib, hold_seconds=0.5, countdown_seconds=0.0, neutral_seconds=0.5,
                                min_move_deg=3.0, do_squat=False)


# ------------------------------------------------------------------ calibration

def test_starts_by_calibrating_and_says_so(ctrl):
    s = ctrl.update(seated_frame(), 0.0)
    assert s.stage == CALIBRATE
    assert "still" in s.cue.lower()
    assert 0.0 < s.progress < 1.0


def test_calibration_completes_and_advances_to_neutral(ctrl):
    s = _drive(ctrl, 5)
    assert s.stage == NEUTRAL
    assert ctrl.calib.hip_width is not None and ctrl.calib.hip_width > 0


def test_no_person_in_frame_is_reported_not_crashed(ctrl):
    s = ctrl.update(None, 0.0)
    assert s.tracking is False
    assert "frame" in s.cue.lower()


# ----------------------------------------------------------------- hold to lock

def test_neutral_locks_only_after_the_hold_time_elapses(ctrl):
    _drive(ctrl, 5)                                   # calibrate
    s = ctrl.update(seated_frame(), 1.0)
    assert s.stage == NEUTRAL and s.progress == 0.0   # first sample starts the hold
    s = ctrl.update(seated_frame(), 1.2)
    assert 0.0 < s.progress < 1.0                     # part way through
    s = ctrl.update(seated_frame(), 1.6)              # 0.6s > 0.5s required
    assert s.stage == EXTREME_A
    assert s.just_locked == "neutral"


def test_progress_is_time_based_not_frame_based(ctrl):
    """Two frames far apart must lock; many frames close together must not."""
    _drive(ctrl, 5)
    ctrl.update(seated_frame(), 1.0)
    s = ctrl.update(seated_frame(), 9.0)              # only 2 frames, but 8s apart
    assert s.stage == EXTREME_A                       # locked on elapsed time

    c2 = AssessmentController(Calibrator(needed_frames=5), hold_seconds=0.5, countdown_seconds=0.0,
                              neutral_seconds=0.5, do_squat=False)
    _drive(c2, 5)
    s = _drive(c2, 50, t0=1.0, dt=0.001)              # 50 frames in 0.05s
    assert s.stage == NEUTRAL                         # not enough WALL CLOCK


def test_moving_during_the_hold_restarts_it(ctrl):
    """A sustained move discards the elapsed hold and restarts where you settled."""
    _drive(ctrl, 5)
    ctrl.update(seated_frame(), 1.0)
    before = ctrl.update(seated_frame(), 1.4).progress
    assert before > 0.5                          # most of the way to locking

    for i in range(4):
        s = ctrl.update(seated_frame(rot_deg_left=40.0), 1.45 + 0.01 * i)
    assert s.stage == NEUTRAL                    # did not lock the wrong value
    assert s.progress < before / 2               # the elapsed hold was thrown away


def test_a_single_jittery_frame_does_not_restart_the_hold(ctrl):
    """Median-of-5 smoothing must absorb one bad landmark frame."""
    _drive(ctrl, 5)
    ctrl.update(seated_frame(), 1.0)
    ctrl.update(seated_frame(), 1.1)
    s = ctrl.update(seated_frame(rot_deg_left=40.0), 1.2)   # one outlier
    assert s.progress > 0.0                                  # hold survived


# --------------------------------------------------------------- extreme capture

def _reach_extreme_a(ctrl):
    _drive(ctrl, 5)                                   # calibrate
    ctrl.update(seated_frame(), 1.0)
    ctrl.update(seated_frame(), 1.6)                  # neutral locked
    assert ctrl.stage == EXTREME_A


def test_a_reading_near_neutral_does_not_start_the_hold(ctrl):
    _reach_extreme_a(ctrl)
    s = ctrl.update(seated_frame(rot_deg_left=1.0), 2.0)   # below min_move_deg
    assert s.progress == 0.0
    assert "Rotate" in s.cue


def test_holding_an_extreme_locks_it_and_asks_for_the_other_way(ctrl):
    _reach_extreme_a(ctrl)
    ctrl.update(seated_frame(rot_deg_left=25.0), 2.0)
    s = ctrl.update(seated_frame(rot_deg_left=25.0), 2.7)
    assert s.stage == EXTREME_B
    assert "OTHER" in s.cue
    assert len(ctrl._extremes[LEFT]) == 1


def test_the_second_extreme_must_be_the_opposite_direction(ctrl):
    _reach_extreme_a(ctrl)
    ctrl.update(seated_frame(rot_deg_left=25.0), 2.0)
    ctrl.update(seated_frame(rot_deg_left=25.0), 2.7)     # first locked (positive)
    # rotating the SAME way again must not count
    s = ctrl.update(seated_frame(rot_deg_left=30.0), 3.0)
    s = ctrl.update(seated_frame(rot_deg_left=30.0), 3.9)
    assert s.stage == EXTREME_B                          # still waiting
    assert len(ctrl._extremes[LEFT]) == 1


def test_both_legs_are_measured_then_a_profile_appears(ctrl):
    _reach_extreme_a(ctrl)
    t = 2.0
    for side_angle in (+25.0, -20.0):                    # left leg, both extremes
        ctrl.update(seated_frame(rot_deg_left=side_angle), t)
        ctrl.update(seated_frame(rot_deg_left=side_angle), t + 0.7)
        t += 1.5
    assert ctrl.side == RIGHT and ctrl.stage == NEUTRAL   # moved to the other leg

    ctrl.update(seated_frame(), t)
    ctrl.update(seated_frame(), t + 0.7)                  # right neutral
    t += 1.5
    for side_angle in (+30.0, -10.0):
        ctrl.update(seated_frame(rot_deg_right=side_angle), t)
        ctrl.update(seated_frame(rot_deg_right=side_angle), t + 0.7)
        t += 1.5

    assert ctrl.stage == DONE                             # do_squat=False
    assert ctrl.profile is not None and ctrl.stance is not None
    assert ctrl.profile.ir_max > 0 and ctrl.profile.er_max > 0
    assert ctrl.stance.width_factor > 0
    assert ctrl.profile.pattern in ("IR-dominant", "balanced", "ER-dominant")


def test_ir_and_er_land_on_the_right_side_of_zero(ctrl):
    """A positive locked reading must become IR, a negative one ER."""
    _reach_extreme_a(ctrl)
    t = 2.0
    for angle in (+25.0, -15.0):
        ctrl.update(seated_frame(rot_deg_left=angle), t)
        ctrl.update(seated_frame(rot_deg_left=angle), t + 0.7)
        t += 1.5
    ctrl.skip_leg()                                       # reuse the left readings
    assert ctrl.profile is not None
    assert ctrl.profile.ir_left == pytest.approx(25.0, abs=3.0)
    assert ctrl.profile.er_left == pytest.approx(15.0, abs=3.0)


def test_skip_leg_moves_on_without_crashing(ctrl):
    _drive(ctrl, 5)
    assert ctrl.side == LEFT
    ctrl.skip_leg()
    assert ctrl.side == RIGHT


# ------------------------------------------------------------------ squat stage

def test_squat_stage_is_offered_when_enabled():
    c = AssessmentController(Calibrator(needed_frames=5), hold_seconds=0.5, countdown_seconds=0.0,
                             neutral_seconds=0.5, do_squat=True)
    _drive(c, 5)
    c._extremes[LEFT] = [20.0, -20.0]
    c._extremes[RIGHT] = [20.0, -20.0]
    c._finish_profile()
    assert c.stage == SQUAT
    s = c.update(seated_frame(), 10.0)
    assert "squat" in s.cue.lower()          # prompts before a runner is attached
    assert c.reps == []


# ------------------------------------------------- squat stage: stand-up gate

def standing_frame(knee_deg=178.0, L=0.4, hip_w=0.2):
    """Upright pose with a controllable knee angle (y points DOWN)."""
    h = 2 * L * np.cos(np.radians((180.0 - knee_deg) / 2.0))
    y_knee, z_knee = h / 2.0, float(np.sqrt(max(L ** 2 - (h / 2) ** 2, 0.0)))
    lm = np.zeros((33, 3))
    for sign, hip_i, knee_i, ank_i, sh_i, foot_i in (
            (-1, 23, 25, 27, 11, 31), (+1, 24, 26, 28, 12, 32)):
        x = sign * hip_w / 2.0
        lm[ank_i] = [x, 0.0, 0.0]
        lm[knee_i] = [x, -y_knee, z_knee]
        lm[hip_i] = [x, -h, 0.0]
        lm[sh_i] = [x * 1.4, -h - 0.45, 0.0]
        lm[foot_i] = [x, 0.0, 0.0]
    return lm


def seated_on_chair(L=0.4, hip_w=0.2):
    """Hips level with the knees, shins vertical — the pose the rotation stage
    leaves you in. Knee angle here is ~50 deg, i.e. below the FSM's bottom gate."""
    lm = np.zeros((33, 3))
    for sign, hip_i, knee_i, ank_i, sh_i, foot_i in (
            (-1, 23, 25, 27, 11, 31), (+1, 24, 26, 28, 12, 32)):
        x = sign * hip_w / 2.0
        lm[ank_i] = [x, 0.0, 0.0]
        lm[knee_i] = [x, -L, L]
        lm[hip_i] = [x, -L, 0.0]
        lm[sh_i] = [x * 1.4, -L - 0.45, 0.0]
        lm[foot_i] = [x, 0.0, 0.0]
    return lm


def _at_squat_stage(target_reps=5):
    c = AssessmentController(Calibrator(needed_frames=5), hold_seconds=0.5, countdown_seconds=0.0,
                             neutral_seconds=0.5, do_squat=True,
                             target_reps=target_reps, stand_seconds=0.5)
    _drive(c, 5)
    c._extremes[LEFT] = [20.0, -20.0]
    c._extremes[RIGHT] = [20.0, -20.0]
    c._finish_profile()
    assert c.stage == SQUAT
    return c


def test_the_counter_is_not_armed_while_still_seated():
    """Seated, the knee reads ~50 deg — the FSM's BOTTOM. Arming there books a
    phantom rep the moment you stand up, with an impossible depth attached."""
    c = _at_squat_stage()
    for i in range(10):
        s = c.update(seated_on_chair(), 10.0 + 0.1 * i)
    assert c.needs_runner is False
    assert c.runner is None
    assert "stand up" in s.cue.lower()


def test_standing_up_arms_the_counter():
    c = _at_squat_stage()
    c.update(standing_frame(), 10.0)
    s = c.update(standing_frame(), 10.8)          # held upright past stand_seconds
    assert c.needs_runner is True
    assert s.just_locked == "standing"


def test_standing_side_on_is_told_to_face_the_camera_but_not_locked_out():
    """Facing is a warning, not a gate: MediaPipe's facing reading jitters on some
    front footage, and a hard gate locked people out. Stance advice checks it itself."""
    a = np.radians(70)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    c = _at_squat_stage()
    s = c.update(standing_frame() @ R.T, 10.0)
    assert "face the camera" in s.cue.lower()
    for i in range(1, 12):
        c.update(standing_frame() @ R.T, 10.0 + 0.1 * i)
    assert c.needs_runner is True


def test_a_standing_knee_read_below_155_still_arms_it():
    """MediaPipe reads some people's straight knee as 151-154 deg; a 155 gate locked them out."""
    c = _at_squat_stage()
    c.update(standing_frame(knee_deg=151.0), 10.0)
    c.update(standing_frame(knee_deg=151.0), 10.8)
    assert c.needs_runner is True


def test_briefly_straightening_does_not_arm_it():
    """The gate is a sustained hold, not a single frame."""
    c = _at_squat_stage()
    c.update(standing_frame(), 10.0)
    c.update(seated_on_chair(), 10.1)             # dropped back down
    s = c.update(standing_frame(), 10.3)
    assert c.needs_runner is False
    assert s.progress < 1.0


def test_the_set_ends_once_the_target_is_reached():
    c = _at_squat_stage(target_reps=3)
    c.update(standing_frame(), 10.0)
    c.update(standing_frame(), 10.8)
    c.start_squat(_FakeRunner(confirmed=2))
    s = c.update(standing_frame(), 11.0)
    assert c.stage == SQUAT                        # 2 of 3, keep going
    assert "2 of 3" in s.cue

    c.runner.confirmed = 3
    c.update(standing_frame(), 11.2)
    assert c.stage == DONE
    assert "3 confirmed" in c.finish_reason


def test_target_zero_means_run_until_the_user_stops():
    c = _at_squat_stage(target_reps=0)
    c.update(standing_frame(), 10.0)
    c.update(standing_frame(), 10.8)
    c.start_squat(_FakeRunner(confirmed=99))
    s = c.update(standing_frame(), 11.0)
    assert c.stage == SQUAT
    assert "SPACE" in s.detail


class _FakeRunner:
    """Minimal SquatRunner stand-in: the gate logic is what is under test."""

    def __init__(self, confirmed=0):
        self.confirmed = confirmed
        self.reps = []

    @property
    def confirmed_count(self):
        return self.confirmed

    def update(self, lm, now=None):
        from stancesense.squat.signals import FrameState
        return FrameState(phase="STANDING", knee_angle=178.0, depth=0.0,
                          rep_count=self.confirmed)


# ------------------------------------------------- visibility gating

class _LM:
    """A MediaPipe-style landmark: coordinates plus a visibility score."""

    def __init__(self, x, y, z, visibility=1.0):
        self.x, self.y, self.z = x, y, z
        self.visibility = visibility


def mp_frame(rot_deg_left=0.0, rot_deg_right=0.0, leg_visibility=1.0):
    """The synthetic seated pose, wrapped as MediaPipe landmark objects.

    MediaPipe emits all 33 landmarks whenever it sees a person — including
    invented ones for joints that are off camera. Only ``visibility`` reveals
    that, which is exactly what this exercises.
    """
    arr = seated_frame(rot_deg_left, rot_deg_right)
    leg_joints = {23, 24, 25, 26, 27, 28}
    return [_LM(x, y, z, leg_visibility if i in leg_joints else 1.0)
            for i, (x, y, z) in enumerate(arr)]


def test_an_invisible_leg_is_never_measured():
    """The original bug: rotation was read off hallucinated joints, so the app
    'measured' IR/ER before the legs were even on camera."""
    c = AssessmentController(Calibrator(needed_frames=5), hold_seconds=0.5, countdown_seconds=0.0,
                             neutral_seconds=0.5, do_squat=False)
    for i in range(30):
        s = c.update(mp_frame(leg_visibility=0.2), 1.0 + 0.1 * i)
    assert s.tracking is False
    assert "in frame" in s.cue.lower()
    assert c.stage == CALIBRATE            # never even got past calibration
    assert c.profile is None


def test_calibration_needs_both_legs_visible():
    c = AssessmentController(Calibrator(needed_frames=5), do_squat=False)
    s = c.update(mp_frame(leg_visibility=0.1), 0.0)
    assert s.tracking is False
    assert len(c.calib._hip_w) == 0        # nothing accumulated from a guess


def test_a_visible_leg_measures_normally():
    c = AssessmentController(Calibrator(needed_frames=5), hold_seconds=0.5, countdown_seconds=0.0,
                             neutral_seconds=0.5, do_squat=False)
    for i in range(5):
        s = c.update(mp_frame(leg_visibility=0.95), 0.05 * i)
    assert s.stage == NEUTRAL
    assert s.tracking is True


def test_losing_sight_of_the_leg_throws_the_hold_away():
    """A reading half-held must not survive the leg leaving the frame."""
    c = AssessmentController(Calibrator(needed_frames=5), hold_seconds=0.5, countdown_seconds=0.0,
                             neutral_seconds=0.5, do_squat=False)
    for i in range(5):
        c.update(mp_frame(leg_visibility=0.95), 0.05 * i)
    c.update(mp_frame(leg_visibility=0.95), 1.0)
    before = c.update(mp_frame(leg_visibility=0.95), 1.4).progress
    assert before > 0.5

    s = c.update(mp_frame(leg_visibility=0.1), 1.45)     # leg leaves the frame
    assert s.progress == 0.0
    assert s.tracking is False
    assert c.stage == NEUTRAL                            # did not lock a guess


def test_partial_visibility_still_blocks_it():
    """A visible hip and knee with an off-screen ankle is still unmeasurable."""
    from stancesense.app.controller import leg_visible
    lm = mp_frame(leg_visibility=0.95)
    lm[27].visibility = 0.1                              # left ankle out of frame
    assert leg_visible(lm, LEFT, 0.6) is False
    assert leg_visible(lm, RIGHT, 0.6) is True


def test_plain_arrays_have_no_visibility_and_count_as_visible():
    """MM-Fit and the offline path pass bare arrays; they must not be blocked."""
    from stancesense.app.controller import leg_visible
    assert leg_visible(seated_frame(), LEFT, 0.6) is True


# ------------------------------------------------- get-ready countdown + lock at the extreme

def _hold_angle(c, t0, seconds, left=0.0, right=0.0, dt=0.1):
    s = None
    for i in range(int(round(seconds / dt)) + 1):
        s = c.update(seated_frame(left, right), t0 + i * dt)
    return s, t0 + seconds


def _counted_controller():
    c = AssessmentController(Calibrator(needed_frames=5), hold_seconds=1.0, neutral_seconds=0.5,
                             min_move_deg=3.0, do_squat=False, countdown_seconds=4.0)
    _drive(c, 5)                                       # calibrated -> NEUTRAL (countdown pending)
    return c


def test_nothing_is_measured_during_the_countdown():
    c = _counted_controller()
    s, t = _hold_angle(c, 1.0, 3.5)                    # holding neutral perfectly, but still counting down
    assert c.stage == NEUTRAL and s.countdown == 1 and "Starts in" in s.detail and s.angle is None
    s, t = _hold_angle(c, t + 0.6, 0.8)                # countdown over, then 0.5 s neutral hold
    assert c.stage == EXTREME_A


def test_every_step_gets_its_own_countdown():
    c = _counted_controller()
    _, t = _hold_angle(c, 1.0, 5.0)                    # countdown + neutral
    assert c.stage == EXTREME_A
    s, t = _hold_angle(c, t + 0.1, 3.0, left=25.0)     # already at the extreme, but counting down
    assert c.stage == EXTREME_A and "FAR as it goes" in s.cue and s.countdown is not None
    s, t = _hold_angle(c, t + 0.1, 2.5, left=25.0)     # countdown ends, 1 s hold at 25 deg
    assert c.stage == EXTREME_B
    s = c.update(seated_frame(25.0), t + 0.1)
    assert "Locked +25" in s.cue and "OTHER" in s.cue and s.countdown in (3, 4)   # starts on the lock frame


def test_an_extreme_locks_only_at_the_furthest_point_reached():
    """Pausing part-way back from the limit must not lock; holding at the limit must."""
    c = AssessmentController(Calibrator(needed_frames=5), hold_seconds=1.0, neutral_seconds=0.5,
                             min_move_deg=3.0, do_squat=False, countdown_seconds=0.0)
    _drive(c, 5)
    _, t = _hold_angle(c, 1.0, 1.0)                    # neutral
    assert c.stage == EXTREME_A
    _, t = _hold_angle(c, t + 0.1, 0.5, left=30.0)     # reach 30 deg briefly
    s, t = _hold_angle(c, t + 0.1, 2.0, left=15.0)     # back off to 15 and hold still
    assert c.stage == EXTREME_A                        # NOT locked at 15
    assert "furthest point" in s.cue and s.countdown is None
    s, t = _hold_angle(c, t + 0.1, 1.5, left=30.0)     # return to the limit and hold
    assert c.stage == EXTREME_B
    assert c._extremes[LEFT][0] == pytest.approx(30.0, abs=2.0)


def test_the_second_leg_is_announced_before_it_is_measured():
    c = AssessmentController(Calibrator(needed_frames=5), hold_seconds=0.5, neutral_seconds=0.5,
                             min_move_deg=3.0, do_squat=False, countdown_seconds=4.0)
    _drive(c, 5)
    c.skip_leg()                                       # move to the right leg
    s = c.update(seated_frame(), 10.0)
    assert c.side == RIGHT and "LEFT done" in s.cue and "RIGHT leg" in s.cue and s.countdown == 4


def _fits(states):
    """Every title / cue / detail line must fit the 640-px camera frame in the
    window's ASCII-only Hershey font (it draws em dashes and degree signs as '?')."""
    cv2 = pytest.importorskip("cv2")
    for st in states:
        for txt, scale, thick in ((st.title, 0.62, 2), (st.cue, 0.72, 2), (st.detail, 0.5, 1)):
            if not txt:
                continue
            assert txt.isascii(), txt
            width = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0]
            assert width <= 620, (txt, width)


def test_cues_fit_the_camera_window():
    c = AssessmentController(Calibrator(needed_frames=5), hold_seconds=1.0, neutral_seconds=0.5,
                             min_move_deg=3.0, do_squat=True, countdown_seconds=4.0)
    states = [c.update(mp_frame(leg_visibility=0.1), 0.0)]          # not visible
    for i in range(6):
        states.append(c.update(seated_frame(), 0.1 * i))            # calibration
    t = 1.0
    for _ in range(400):                                            # both legs: countdowns + holds
        if c.stage not in (NEUTRAL, EXTREME_A, EXTREME_B):
            break
        ang = 0.0 if c.stage == NEUTRAL else (28.0 if c.stage == EXTREME_A else -22.0)
        kw = {"rot_deg_left": ang} if c.side == LEFT else {"rot_deg_right": ang}
        states.append(c.update(seated_frame(**kw), t)); t += 0.25
    assert c.stage == SQUAT
    states.append(c.update(seated_on_chair(), t))                   # stand-up gate
    a = np.radians(70)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    for i in range(12):
        states.append(c.update(standing_frame() @ R.T, t + 1 + 0.1 * i))
    c.start_squat(_FakeRunner())
    states.append(c.update(standing_frame(), t + 5))
    # cues that depend on the reading rather than the stage
    c2 = AssessmentController(Calibrator(needed_frames=5), countdown_seconds=4.0)
    for side in (LEFT, RIGHT):
        c2.side = side
        c2._extremes[side] = [28.0]
        for st in (NEUTRAL, EXTREME_A, EXTREME_B):
            c2.stage = st
            c2._new_step()
            states.append(c2._countdown(0.0))
    states = [st for st in states if st is not None]
    extra = ["Back to your furthest point (+28), HOLD", "OTHER way, as FAR as it goes, HOLD STILL",
             "Rotate as FAR as it goes, then HOLD STILL"]
    states += [type(states[0])(stage=EXTREME_A, side=LEFT, title="RIGHT leg - other extreme",
                               cue=e, detail="Swing the lower leg sideways from the knee, thigh still")
               for e in extra]
    _fits(states)
