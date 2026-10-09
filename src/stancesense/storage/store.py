"""Phase 11 - persistence.

A tiny SQLite store for the three things an assessment produces:

    HipProfile   the functional hip-rotation profile (Phase 5)
    StanceRec    the recommended starting stance     (Phase 6)
    RepResult    one row per verified squat rep      (Phases 7b + 9)
    squat run    which classifier judged the reps, and the stance advice
                 measured from the confirmed squats (NARROW / MODERATE / WIDE)

Everything hangs off a ``session`` row so a single run of
``scripts/run_assessment.py`` (or ``scripts/demo_end_to_end_mmfit.py``) is one
retrievable unit.  Tables are created on first use; the file is plain SQLite so
the Streamlit dashboard (Phase 12) can read it with no extra machinery.

Deliberately dependency-free (stdlib ``sqlite3`` only) and safe to import even
when the rest of the runtime stack is missing.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Iterable, Optional

from ..common.types import HipProfile, StanceRec, RepResult

DEFAULT_DB = os.path.join("data", "stancesense.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT    NOT NULL,
    subject_id   TEXT    NOT NULL DEFAULT 'local',
    source       TEXT    NOT NULL DEFAULT 'live',
    notes        TEXT    DEFAULT ''
);

CREATE TABLE IF NOT EXISTS hip_profiles (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER NOT NULL REFERENCES sessions(id),
    ir_max        REAL, er_max        REAL,
    total_arc     REAL, rotation_bias REAL,
    symmetry      REAL, pattern       TEXT,
    ir_left       REAL, er_left       REAL,
    ir_right      REAL, er_right      REAL
);

CREATE TABLE IF NOT EXISTS stance_recs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER NOT NULL REFERENCES sessions(id),
    width_factor  REAL, toe_out_deg REAL,
    asymmetric    INTEGER, rationale TEXT
);

CREATE TABLE IF NOT EXISTS rep_results (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id         INTEGER NOT NULL REFERENCES sessions(id),
    rep_index          INTEGER,
    max_depth_ratio    REAL,
    knee_over_foot_dev REAL,
    symmetry_ok        INTEGER,
    phase_sequence     TEXT,
    gru_prob           REAL,
    confirmed          INTEGER
);

CREATE INDEX IF NOT EXISTS ix_rep_session ON rep_results(session_id);

CREATE TABLE IF NOT EXISTS squat_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          INTEGER NOT NULL REFERENCES sessions(id),
    classifier          TEXT,
    current_stance      TEXT,  stance_width        REAL,
    depth_ratio         REAL,  max_trunk_lean_deg  REAL,
    knee_to_ankle_width REAL,  facing_deg          REAL,
    front_facing        INTEGER,
    recommendation      TEXT,  reason              TEXT
);
"""
# rep_results.gru_prob  = P(squat) from the project's earlier GRU verifier; NULL for
#                         the rule-based classifier (column kept for old sessions).
# rep_results.confirmed = 1 accepted / 0 rejected / NULL not gated.


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    """SQLite-backed store. Usable as a context manager."""

    def __init__(self, path: str = DEFAULT_DB):
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # ----------------------------------------------------------------- writes
    def start_session(self, subject_id: str = "local", source: str = "live",
                      notes: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO sessions (created_at, subject_id, source, notes) VALUES (?,?,?,?)",
            (_utcnow(), subject_id, source, notes),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def save_profile(self, session_id: int, p: HipProfile) -> int:
        cur = self.conn.execute(
            "INSERT INTO hip_profiles (session_id, ir_max, er_max, total_arc, "
            "rotation_bias, symmetry, pattern, ir_left, er_left, ir_right, er_right) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (session_id, p.ir_max, p.er_max, p.total_arc, p.rotation_bias,
             p.symmetry, p.pattern, p.ir_left, p.er_left, p.ir_right, p.er_right),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def save_stance(self, session_id: int, r: StanceRec) -> int:
        cur = self.conn.execute(
            "INSERT INTO stance_recs (session_id, width_factor, toe_out_deg, "
            "asymmetric, rationale) VALUES (?,?,?,?,?)",
            (session_id, r.width_factor, r.toe_out_deg, int(r.asymmetric), r.rationale),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def save_reps(self, session_id: int, reps: Iterable[RepResult]) -> int:
        rows = [
            (session_id, r.rep_index, r.max_depth_ratio, r.knee_over_foot_dev,
             int(r.symmetry_ok), json.dumps(list(r.phase_sequence)),
             r.gru_prob, None if r.confirmed is None else int(r.confirmed))
            for r in reps
        ]
        self.conn.executemany(
            "INSERT INTO rep_results (session_id, rep_index, max_depth_ratio, "
            "knee_over_foot_dev, symmetry_ok, phase_sequence, gru_prob, confirmed) "
            "VALUES (?,?,?,?,?,?,?,?)", rows,
        )
        self.conn.commit()
        return len(rows)

    def save_squat_run(self, session_id: int, classifier: Optional[str] = None,
                       squat_stance: Optional[dict] = None) -> int:
        """The classifier that judged the reps and the squat-based stance advice
        (``RuleSquatRunner.summary()["stance"]``; None when no squat was confirmed)."""
        s = squat_stance or {}
        cur = self.conn.execute(
            "INSERT INTO squat_runs (session_id, classifier, current_stance, stance_width, "
            "depth_ratio, max_trunk_lean_deg, knee_to_ankle_width, facing_deg, front_facing, "
            "recommendation, reason) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (session_id, classifier, s.get("current_stance"), s.get("stance_width_hip_widths"),
             s.get("depth_ratio"), s.get("max_trunk_lean_deg"), s.get("knee_to_ankle_width_at_bottom"),
             s.get("facing_deg"), None if s.get("front_facing") is None else int(s["front_facing"]),
             s.get("recommendation"), s.get("reason")),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def save_assessment(self, profile: Optional[HipProfile] = None,
                        stance: Optional[StanceRec] = None,
                        reps: Iterable[RepResult] = (),
                        subject_id: str = "local", source: str = "live",
                        notes: str = "", classifier: Optional[str] = None,
                        squat_stance: Optional[dict] = None) -> int:
        """Write a whole run in one call; returns the new session id."""
        sid = self.start_session(subject_id, source, notes)
        if profile is not None:
            self.save_profile(sid, profile)
        if stance is not None:
            self.save_stance(sid, stance)
        reps = list(reps)
        if reps:
            self.save_reps(sid, reps)
        if classifier is not None or squat_stance is not None:
            self.save_squat_run(sid, classifier, squat_stance)
        return sid

    # ------------------------------------------------------------------ reads
    def sessions(self, limit: int = 100) -> list:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM sessions ORDER BY id DESC LIMIT ?", (limit,))]

    def profile_history(self, subject_id: Optional[str] = None) -> list:
        """All stored profiles, newest first, joined to their session."""
        sql = ("SELECT s.id AS session_id, s.created_at, s.subject_id, s.source, h.* "
               "FROM hip_profiles h JOIN sessions s ON s.id = h.session_id")
        args: tuple = ()
        if subject_id:
            sql += " WHERE s.subject_id = ?"
            args = (subject_id,)
        sql += " ORDER BY s.id DESC"
        return [dict(r) for r in self.conn.execute(sql, args)]

    def stance_for(self, session_id: int) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM stance_recs WHERE session_id = ? ORDER BY id DESC LIMIT 1",
            (session_id,)).fetchone()
        return dict(row) if row else None

    def profile_for(self, session_id: int) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM hip_profiles WHERE session_id = ? ORDER BY id DESC LIMIT 1",
            (session_id,)).fetchone()
        return dict(row) if row else None

    def squat_run_for(self, session_id: int) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM squat_runs WHERE session_id = ? ORDER BY id DESC LIMIT 1",
            (session_id,)).fetchone()
        return dict(row) if row else None

    def reps_for(self, session_id: int) -> list:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM rep_results WHERE session_id = ? ORDER BY rep_index",
            (session_id,))]

    def last_session_with_reps(self) -> Optional[int]:
        row = self.conn.execute(
            "SELECT session_id FROM rep_results ORDER BY session_id DESC LIMIT 1"
        ).fetchone()
        return int(row["session_id"]) if row else None

    def rep_summary(self, session_id: int) -> dict:
        """Aggregate one run's reps: counted, confirmed, rejected, depth stats.

        Depth describes the CONFIRMED squats when any rep was confirmed: a rejected
        rep (another exercise, or tracking noise with an impossible depth) must not
        set the "best depth". Only when nothing was judged at all (FSM-only) do the
        depth stats fall back to every counted rep. ``classifier`` comes from the
        squat_runs row (None for sessions stored before it existed).
        """
        reps = self.reps_for(session_id)
        run = self.squat_run_for(session_id)
        classifier = run["classifier"] if run else None
        if not reps:
            return dict(counted=0, confirmed=0, rejected=0, ungated=0,
                        mean_depth=0.0, max_depth=0.0, mean_gru_prob=None,
                        classifier=classifier, depth_from="no reps")
        probs = [r["gru_prob"] for r in reps if r["gru_prob"] is not None]
        confirmed = sum(1 for r in reps if r["confirmed"] == 1)
        rejected = sum(1 for r in reps if r["confirmed"] == 0)
        ungated = sum(1 for r in reps if r["confirmed"] is None)
        judged = [r for r in reps if r["confirmed"] == 1]
        if judged:
            basis, depth_from = judged, "confirmed squats"
        elif confirmed == 0 and rejected == 0:
            basis, depth_from = reps, "all counted reps (none were judged)"
        else:
            basis, depth_from = [], "no confirmed squat"
        depths = [r["max_depth_ratio"] or 0.0 for r in basis]
        return dict(
            counted=len(reps), confirmed=confirmed, rejected=rejected, ungated=ungated,
            mean_depth=sum(depths) / len(depths) if depths else 0.0,
            max_depth=max(depths) if depths else 0.0,
            mean_gru_prob=(sum(probs) / len(probs)) if probs else None,
            classifier=classifier, depth_from=depth_from,
        )

    # --------------------------------------------------------------- lifecycle
    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
