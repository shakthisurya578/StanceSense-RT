"""Rule-based biomechanical squat classifier (no training).

    landmarks -> per-frame biomechanics -> causal smoothing -> state machine
    STANDING -> DESCENDING -> BOTTOM -> ASCENDING -> STANDING -> per-cycle
    biomechanical squat score (9 checks) -> squat rep / rejected movement

Works frame by frame with timestamps, so the live camera path and the offline
evaluation run the same code. Every threshold is in :class:`RuleConfig`.

:class:`RuleSquatRunner` is what the app's controller drives during the squat
check (``update(lm, now)`` -> ``FrameState``, ``reps``, ``confirmed_count``,
``summary()``). A completed cycle is always counted (``rep_count``); it is a squat
(``RepResult.confirmed = True``) only when its score reaches the threshold.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np

from ..common.types import RepResult
from .signals import FrameState, as_array as _as_array, frame_signals

UNREADY, STANDING, DESCENDING, BOTTOM, ASCENDING = "UNREADY", "STANDING", "DESCENDING", "BOTTOM", "ASCENDING"
CHECKS = ("depth", "knee_flexion", "hip_motion", "descent", "bottom", "ascent",
          "standing_return", "bilateral", "upright_trunk")
LEG = (23, 24, 25, 26, 27, 28)     # hips, knees, ankles: leg visibility = their MEAN (as in the cache)


@dataclass(frozen=True)
class RuleConfig:
    # ---- smoothing
    median_window_s: float = 0.30        # moving median over the last 0.3 s (kills single-frame spikes)
    ema_tau_s: float = 0.12              # then exponential smoothing, time constant 0.12 s
    velocity_span_s: float = 0.20        # hip velocity measured over the last 0.2 s
    min_leg_visibility: float = 0.5      # frames whose leg landmarks are less visible are skipped
    # ---- standing (start of every cycle)
    stand_torso_max_deg: float = 40.0
    stand_knee_min_deg: float = 135.0
    stand_hip_height_min: float = 1.3    # hip above ankles, femur lengths
    stand_hold_s: float = 0.15           # v2: fast continuous squatters barely pause at the top
    baseline_tau_s: float = 1.0          # standing reference adapts while standing
    # ---- transitions (hysteresis)
    descent_start_drop: float = 0.12     # hip drop below standing (femur) to start a descent
    descent_start_flex_deg: float = 10.0 # and knee flexion below standing
    bottom_speed: float = 0.15           # descending slower than this (femur/s) = turning point (v2: speed only)
    ascent_start_rise: float = 0.08      # rise above the lowest point (femur) to start ascending
    rebottom_rise: float = 0.04          # falling back within this of the lowest point = bottom again
    return_frac: float = 0.75            # share of the drop recovered to be standing again
    return_knee_tol_deg: float = 20.0
    max_phase_s: float = 4.0             # any phase longer than this aborts the cycle
    return_window_s: float = 0.5         # v3: standing return judged over 0.5 s after the cycle closes
    # ---- the 9 checks of the biomechanical squat score
    min_depth_ratio: float = 0.45        # (hip.y - knee.y)/femur + 1: 0 standing, 1 thighs parallel
    min_knee_flex_deg: float = 40.0      # knee flexion from standing
    max_bottom_knee_deg: float = 130.0
    min_hip_drop: float = 0.35           # femur lengths
    min_knee_hip_coupling: float = 0.6   # correlation of knee angle and hip height in the cycle
    phase_min_s: float = 0.12            # v2: 2 frames at 15 fps (phases are measured between gates)
    phase_max_s: float = 3.5
    min_monotone: float = 0.6            # share of descent (ascent) frames moving the right way
    max_bottom_s: float = 2.5
    end_knee_tol_deg: float = 15.0
    min_bilateral_ratio: float = 0.6     # smaller / larger knee flexion
    max_leg_drop_asym: float = 0.35      # femur lengths, mean over the cycle
    max_cycle_torso_deg: float = 70.0
    score_threshold: int = 7             # of 9
    # ---- v3 plausibility gate: geometry no squatting body can produce = tracking noise, never a squat
    max_plausible_torso_deg: float = 100.0
    max_plausible_hip_drop: float = 1.5  # femur lengths
    max_plausible_depth: float = 1.6
    # ---- video-level decision
    min_valid_reps: int = 2


@dataclass
class Cycle:
    t_start: float
    base_knee: float
    base_hip: float
    start_torso: float
    knee: list = field(default_factory=list)
    knee_L: list = field(default_factory=list)
    knee_R: list = field(default_factory=list)
    hip: list = field(default_factory=list)
    depth: list = field(default_factory=list)
    torso: list = field(default_factory=list)
    asym: list = field(default_factory=list)
    kof: list = field(default_factory=list)
    stance: list = field(default_factory=list)
    knee_w: list = field(default_factory=list)
    vel: list = field(default_factory=list)
    facing: list = field(default_factory=list)
    phase: list = field(default_factory=list)
    t: list = field(default_factory=list)
    t_bottom: Optional[float] = None
    t_ascent: Optional[float] = None


FRONT_FACING_MAX_DEG = 20.0     # hip line within 20 deg of the camera's x axis = facing the camera
                                # (Multi-View dev people: front clips 1-11 deg, diagonal 25-56, side 61-85)


def facing_angle(lm) -> float:
    """Degrees the hip line is turned from square-on to the camera: 0 = facing it,
    90 = side-on. MediaPipe world landmarks are camera-aligned (x right, z depth)."""
    lm = np.asarray(lm, float)
    dx, dz = lm[23, 0] - lm[24, 0], lm[23, 2] - lm[24, 2]
    return float(np.degrees(np.arctan2(abs(dz), abs(dx)))) if abs(dx) + abs(dz) > 1e-9 else 0.0


def _kof(lm, hip_w):
    """Knee-over-foot: |knee - ankle| along the hip line, in hip widths (max of both sides)."""
    lat = np.array([lm[23, 0] - lm[24, 0], lm[23, 2] - lm[24, 2]])
    n = np.linalg.norm(lat)
    if n < 1e-9 or hip_w <= 0:
        return 0.0
    lat /= n
    d = [abs(np.dot([lm[k, 0] - lm[a, 0], lm[k, 2] - lm[a, 2]], lat)) / hip_w for k, a in ((25, 27), (26, 28))]
    return float(max(d))


class RuleSquatClassifier:
    """Streaming classifier: push frames, collect scored movement cycles."""

    def __init__(self, femur: float, hip_w: float, config: RuleConfig = RuleConfig()):
        self.femur, self.hip_w, self.cfg = float(femur), float(hip_w), config
        self.phase = UNREADY
        self.cycles: list = []            # finished cycles (dict with score and checks)
        self.aborted = 0
        self._raw: deque = deque()
        self._ema = None
        self._t_prev = None
        self._dt = 1.0 / 15.0
        self._hist: deque = deque()       # (t, hip) for velocity
        self._stand_since = None
        self.base_knee = self.base_hip = self.base_torso = None
        self._cur: Optional[Cycle] = None
        self._pending = None
        self.last = {}

    # ------------------------------------------------------------ signals
    def _signals(self, lm):
        s = frame_signals(lm[None], self.femur, self.hip_w)
        return np.array([s["knee_L"][0], s["knee_R"][0], s["hip_height"][0], s["depth"][0], s["torso"][0],
                         s["stance_width"][0], s["knee_width"][0], s["leg_drop_asym"][0], _kof(lm, self.hip_w),
                         facing_angle(lm)])

    def _smooth(self, t, x):
        c = self.cfg
        self._raw.append((t, x))
        while self._raw and self._raw[0][0] < t - c.median_window_s:
            self._raw.popleft()
        med = np.median(np.stack([v for _, v in self._raw]), axis=0)
        if self._ema is None:
            self._ema = med
        else:
            self._dt = max(t - self._t_prev, 1e-3)
            a = 1.0 - np.exp(-self._dt / c.ema_tau_s)
            self._ema = self._ema + a * (med - self._ema)
        self._t_prev = t
        return self._ema

    def _velocity(self, t, hip):
        self._hist.append((t, hip))
        while len(self._hist) > 2 and self._hist[0][0] < t - self.cfg.velocity_span_s:
            self._hist.popleft()
        t0, h0 = self._hist[0]
        return (hip - h0) / (t - t0) if t > t0 else 0.0

    # ------------------------------------------------------------ update
    def push(self, lm, t: float, leg_visibility: Optional[float] = None):
        """One frame of (33, 3) world landmarks at time ``t`` (s). Returns a finished
        cycle dict on the frame a cycle completes, else None."""
        c = self.cfg
        if leg_visibility is not None and leg_visibility < c.min_leg_visibility:
            return None                                    # unreliable frame: hold everything
        kL, kR, hip, depth, torso, stance, kw, asym, kof, facing = self._smooth(t, self._signals(np.asarray(lm, float)))
        knee = (kL + kR) / 2
        v = self._velocity(t, hip)
        self.last = dict(knee=knee, knee_L=kL, knee_R=kR, hip_height=hip, depth=depth, torso=torso,
                         stance_width=stance, knee_width=kw, hip_velocity=v, knee_over_foot=kof, facing_deg=facing)
        upright = torso <= c.stand_torso_max_deg and knee >= c.stand_knee_min_deg and hip >= c.stand_hip_height_min
        fin = None
        if self._pending is not None:
            self._pending["max_knee"] = max(self._pending["max_knee"], knee)
            self._pending["min_torso"] = min(self._pending["min_torso"], torso)
            if t - self._pending["t_close"] >= c.return_window_s:
                fin = self._finalize()

        if self.phase == UNREADY:
            if upright:
                self._stand_since = t if self._stand_since is None else self._stand_since
                if t - self._stand_since >= c.stand_hold_s:
                    self.phase, self.base_knee, self.base_hip, self.base_torso = STANDING, knee, hip, torso
            else:
                self._stand_since = None
            return fin

        if self.phase == STANDING:
            if upright:                                     # standing reference follows the person
                a = 1.0 - np.exp(-self._dt / c.baseline_tau_s)
                self.base_knee += a * (knee - self.base_knee)
                self.base_hip += a * (hip - self.base_hip)
                self.base_torso += a * (torso - self.base_torso)
            if (self.base_hip - hip >= c.descent_start_drop and self.base_knee - knee >= c.descent_start_flex_deg
                    and v < 0):
                if self._pending is not None:
                    fin = self._finalize()
                self._cur = Cycle(t, self.base_knee, self.base_hip, self.base_torso)
                self.phase = DESCENDING
            elif torso > c.max_cycle_torso_deg:            # lay down / bent over: wait for standing again
                self._reset()
            if self.phase != DESCENDING:
                return fin

        cur = self._cur
        for k, val in (("knee", knee), ("knee_L", kL), ("knee_R", kR), ("hip", hip), ("depth", depth),
                       ("torso", torso), ("asym", asym), ("kof", kof), ("stance", stance), ("knee_w", kw),
                       ("vel", v), ("t", t), ("facing", facing)):
            getattr(cur, k).append(val)
        cur.phase.append(self.phase)
        low = min(cur.hip)
        drop = cur.base_hip - low

        if t - (cur.t_ascent or cur.t_bottom or cur.t_start) > c.max_phase_s:
            self.aborted += 1
            self._reset()
            return fin
        if self.phase == DESCENDING:
            if v > -c.bottom_speed:                                  # stopped going down: turning point
                self.phase, cur.t_bottom = BOTTOM, t
        elif self.phase == BOTTOM:
            if hip >= low + c.ascent_start_rise:
                self.phase, cur.t_ascent = ASCENDING, t
        elif self.phase == ASCENDING:
            if hip <= low + c.rebottom_rise:
                self.phase, cur.t_ascent = BOTTOM, None
            elif hip >= low + c.return_frac * drop and knee >= cur.base_knee - c.return_knee_tol_deg:
                self._pending = dict(cycle=self._score(cur, t), t_close=t, base_knee=cur.base_knee,
                                     max_knee=knee, min_torso=torso)
                self.phase, self._cur = STANDING, None
        return fin

    def _finalize(self):
        """Close the pending cycle: judge the standing return, then the score."""
        c, p = self.cfg, self._pending
        self._pending = None
        cy = p["cycle"]
        cy["checks"]["standing_return"] = bool(p["max_knee"] >= p["base_knee"] - c.end_knee_tol_deg
                                               and p["min_torso"] <= c.stand_torso_max_deg)
        cy["score"] = int(sum(cy["checks"].values()))
        cy["failed"] = [k for k, ok in cy["checks"].items() if not ok]
        cy["squat"] = bool(cy["plausible"] and cy["score"] >= c.score_threshold)
        self.cycles.append(cy)
        return cy

    def flush(self):
        """End of stream: close a cycle still waiting for its standing-return window."""
        return self._finalize() if self._pending is not None else None

    def _reset(self):
        self.phase, self._cur, self._stand_since = UNREADY, None, None

    # ------------------------------------------------------------ scoring
    def _score(self, cur: Cycle, t_end: float) -> dict:
        c = self.cfg
        knee, hip, ph = np.array(cur.knee), np.array(cur.hip), np.array(cur.phase)
        vel = np.array(cur.vel)
        flex_L = cur.base_knee - min(cur.knee_L) if cur.knee_L else 0.0
        flex_R = cur.base_knee - min(cur.knee_R) if cur.knee_R else 0.0
        flex = cur.base_knee - knee.min()
        drop = cur.base_hip - hip.min()
        coup = (float(np.corrcoef(knee, hip)[0, 1]) if len(knee) > 2 and knee.std() > 1e-6 and hip.std() > 1e-6
                else 0.0)
        d_desc = (cur.t_bottom or t_end) - cur.t_start
        d_bot = (cur.t_ascent or t_end) - (cur.t_bottom or t_end)
        d_asc = t_end - (cur.t_ascent or t_end)
        desc_mono = float(np.mean(vel[ph == DESCENDING] <= 0.05)) if np.any(ph == DESCENDING) else 0.0
        asc_mono = float(np.mean(vel[ph == ASCENDING] >= -0.05)) if np.any(ph == ASCENDING) else 0.0
        big, small = max(flex_L, flex_R), min(flex_L, flex_R)
        checks = dict(
            depth=max(cur.depth) >= c.min_depth_ratio,
            knee_flexion=flex >= c.min_knee_flex_deg and knee.min() <= c.max_bottom_knee_deg,
            hip_motion=drop >= c.min_hip_drop and coup >= c.min_knee_hip_coupling,
            descent=c.phase_min_s <= d_desc <= c.phase_max_s and desc_mono >= c.min_monotone,
            bottom=cur.t_bottom is not None and d_bot <= c.max_bottom_s,
            ascent=c.phase_min_s <= d_asc <= c.phase_max_s and asc_mono >= c.min_monotone,
            standing_return=False,                    # judged in _finalize over the return window
            bilateral=(big > 0 and small / big >= c.min_bilateral_ratio) and np.mean(cur.asym) <= c.max_leg_drop_asym,
            upright_trunk=cur.start_torso <= c.stand_torso_max_deg and max(cur.torso) <= c.max_cycle_torso_deg,
        )
        score = int(sum(checks.values()))
        bottom = int(np.argmin(hip))
        plausible = (max(cur.torso) <= c.max_plausible_torso_deg and drop <= c.max_plausible_hip_drop
                     and max(cur.depth) <= c.max_plausible_depth)
        return dict(t_start=cur.t_start, t_end=t_end, score=score, squat=False, plausible=bool(plausible),
                    checks=checks, failed=[k for k, ok in checks.items() if not ok],
                    max_depth_ratio=float(max(cur.depth)), knee_flexion_deg=float(flex), min_knee_deg=float(knee.min()),
                    hip_drop=float(drop), coupling=coup, descent_s=d_desc, bottom_s=d_bot, ascent_s=d_asc,
                    max_torso_deg=float(max(cur.torso)), flex_L=float(flex_L), flex_R=float(flex_R),
                    stance_width=float(np.median(cur.stance)), knee_width_bottom=float(cur.knee_w[bottom]),
                    stance_bottom=float(cur.stance[bottom]), knee_over_foot_max=float(max(cur.kof)),
                    leg_drop_asym=float(np.mean(cur.asym)), facing_deg=float(np.median(cur.facing)))

    # ------------------------------------------------------------ video verdict
    def verdict(self) -> dict:
        self.flush()
        valid = [cy for cy in self.cycles if cy["squat"]]
        scores = [cy["score"] for cy in self.cycles]
        return dict(squat=len(valid) >= self.cfg.min_valid_reps, valid_reps=len(valid), cycles=len(self.cycles),
                    aborted=self.aborted, median_score=float(np.median(scores)) if scores else 0.0,
                    best_score=max(scores) if scores else 0)


# ---------------------------------------------------------------- stance
STANCE_BANDS = (1.4, 2.1)      # ankle separation in hip-joint widths: < 1.4 narrow, > 2.1 wide
ORDER = ("NARROW", "MODERATE", "WIDE")
SHALLOW_DEPTH = 0.5            # depth ratio < 0.5: thigh still > 30 deg above horizontal at the bottom
FORWARD_LEAN_DEG = 45.0        # trunk lean from vertical at the bottom
VALGUS_RATIO = 0.75            # knee separation / ankle separation at the bottom


def stance_class(width: float) -> str:
    return ORDER[0] if width < STANCE_BANDS[0] else ORDER[2] if width > STANCE_BANDS[1] else ORDER[1]


def stance_recommendation(squat_cycles: list) -> Optional[dict]:
    """NARROW / MODERATE / WIDE from the squat reps actually measured.

    Current stance = median ankle separation. The stance is changed only when a
    measured problem points at the width itself; otherwise it is kept.

    * One step WIDER only when depth is limited AND the trunk folds forward
      together: the hips have no room between the feet, so the lifter stops
      short and leans. Either sign alone is not a width problem (a deep squat
      needs some lean; a shallow upright squat is a depth problem).
    * One step NARROWER from WIDE when the knees fall inside the feet at the
      bottom: the stance is wider than the hips control.
    * Knee cave in a narrow or moderate stance keeps the width and adds a cue.
    """
    if not squat_cycles:
        return None
    m = lambda k: float(np.median([c[k] for c in squat_cycles]))
    width, depth, torso = m("stance_width"), m("max_depth_ratio"), m("max_torso_deg")
    cave = m("knee_width_bottom") / max(m("stance_bottom"), 1e-6)
    asym = m("leg_drop_asym")
    cur = stance_class(width)
    i = ORDER.index(cur)
    shallow, lean, valgus = depth < SHALLOW_DEPTH, torso > FORWARD_LEAN_DEG, cave < VALGUS_RATIO
    facing = float(np.median([c.get("facing_deg", 0.0) for c in squat_cycles]))
    front = facing <= FRONT_FACING_MAX_DEG
    if not front:
        # side-on, stance and knee spacing lie along the camera's depth axis, which
        # MediaPipe measures poorly: report, but do not advise
        rec, why = cur, (f"body turned {facing:.0f} deg from the camera - squat facing the camera "
                         f"for stance advice")
    elif valgus and cur == "WIDE":
        rec, why = "MODERATE", "knees move inside the feet at the bottom of a wide stance"
    elif shallow and lean and not valgus:
        rec = ORDER[min(i + 1, 2)]
        why = ("depth stays well above parallel while the trunk folds forward; "
               + ("a wider stance gives the hips room to sit down" if rec != cur
                  else "already wide - work on hip and ankle mobility"))
    else:
        rec, why = cur, ("knees cave in - keep this width and push the knees out over the toes" if valgus
                         else "no width-related problem measured - keep this stance")
    return dict(current_stance=cur, stance_width_hip_widths=round(width, 2), depth_ratio=round(depth, 2),
                max_trunk_lean_deg=round(torso, 1), knee_to_ankle_width_at_bottom=round(cave, 2),
                left_right_drop_asym=round(asym, 2), knee_over_foot=round(m("knee_over_foot_max"), 2),
                facing_deg=round(facing, 1), front_facing=front, recommendation=rec, reason=why)


# ---------------------------------------------------------------- runner for the app
class RuleSquatRunner:
    """Drop-in replacement for SquatRunner: rule engine decides AND counts."""

    def __init__(self, femur: float, hip_w: float, config: RuleConfig = RuleConfig()):
        self.femur, self.hip_w = float(femur), float(hip_w)
        self.clf = RuleSquatClassifier(femur, hip_w, config)
        self.reps: list = []
        self.max_depth = 0.0
        self.gate_info = dict(source="rule-based biomechanical classifier", config=asdict(config))
        self._t0 = None
        self._n = 0

    def update(self, lm, now=None, leg_visibility: Optional[float] = None) -> FrameState:
        vis = leg_visibility
        src = lm.landmark if hasattr(lm, "landmark") else lm
        if vis is None and hasattr(src[0], "visibility"):
            vis = float(np.mean([src[j].visibility for j in LEG]))   # same definition as the cached leg_vis
        arr = _as_array(lm)
        t = now if now is not None else self._n / 15.0
        self._n += 1
        done = self.clf.push(arr, t, vis)
        st = self.clf.last
        self.max_depth = max(self.max_depth, st.get("depth", 0.0))
        new = None
        if done is not None:
            new = self._add(done)
        return FrameState(phase=self.clf.phase, knee_angle=float(st.get("knee", 0.0)), depth=float(st.get("depth", 0.0)),
                          rep_count=len(self.reps), p_squat=None, new_rep=new)

    @property
    def facing_deg(self) -> Optional[float]:
        return self.clf.last.get("facing_deg")

    @property
    def facing_ok(self) -> bool:
        f = self.facing_deg
        return f is None or f <= FRONT_FACING_MAX_DEG

    @property
    def rep_count(self):
        return len(self.reps)

    @property
    def confirmed_count(self):
        return sum(1 for r in self.reps if r.confirmed)

    @property
    def rejected_count(self):
        return sum(1 for r in self.reps if r.confirmed is False)

    @property
    def ungated_count(self):
        return 0

    def stance(self):
        return stance_recommendation([c for c in self.clf.cycles if c["squat"]])

    def _add(self, done):
        r = RepResult(rep_index=len(self.reps) + 1, max_depth_ratio=round(done["max_depth_ratio"], 3),
                      knee_over_foot_dev=round(done["knee_over_foot_max"], 3),
                      symmetry_ok=bool(done["checks"]["bilateral"]),
                      phase_sequence=[DESCENDING, BOTTOM, ASCENDING, STANDING],
                      confirmed=bool(done["squat"]), squat_score=done["score"],
                      rule_failed=list(done["failed"]) + ([] if done["plausible"] else ["implausible_landmarks"]))
        self.reps.append(r)
        return r

    def summary(self) -> dict:
        done = self.clf.flush()
        if done is not None:
            self._add(done)
        depths = [r.max_depth_ratio for r in self.reps if r.confirmed]
        return dict(counted=self.rep_count, confirmed=self.confirmed_count, rejected=self.rejected_count,
                    ungated=0, mean_depth=float(np.mean(depths)) if depths else 0.0,
                    max_depth=float(np.max(depths)) if depths else 0.0, max_depth_seen=float(self.max_depth),
                    mean_gru_prob=None, gated=True, gates=dict(source=self.gate_info["source"]),
                    classifier="rules", scores=[r.squat_score for r in self.reps], stance=self.stance())
