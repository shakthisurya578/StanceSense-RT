"""Core data structures passed between StanceSense-RT modules.

Everything the pipeline produces is a small, explicit dataclass so that each
module has a clear contract.  These types are deliberately plain (no heavy
dependencies) so they are easy to serialise to SQLite / CSV.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# A single side label used throughout the codebase.
LEFT = "LEFT"
RIGHT = "RIGHT"


@dataclass
class HipProfile:
    """The output of the assessment for one person (both legs summarised).

    All angles are in degrees.  IR/ER are stored as POSITIVE magnitudes.
    rotation_bias > 0 => internal-rotation dominant.
    rotation_bias < 0 => external-rotation dominant.

    These describe FUNCTIONAL rotation measured from video.  They are not, and
    must not be presented as, radiographic femoral or acetabular version: the
    project makes no validated angular-accuracy claim about any of these numbers
    (the goniometer validation study was dropped).
    """
    ir_max: float                  # max internal rotation (deg, +)
    er_max: float                  # max external rotation (deg, +)
    total_arc: float               # ir_max + er_max
    rotation_bias: float           # (ir_max - er_max) / 2
    symmetry: float                # |bias_left - bias_right| across legs
    pattern: str = "balanced"      # "IR-dominant" | "balanced" | "ER-dominant"
    ir_left: Optional[float] = None
    er_left: Optional[float] = None
    ir_right: Optional[float] = None
    er_right: Optional[float] = None


@dataclass
class StanceRec:
    """A recommended STARTING stance, to be verified on a live squat."""
    width_factor: float            # multiply by hip width to get foot separation
    toe_out_deg: float             # how far to point the toes out
    rationale: str                 # plain-language explanation for the user
    asymmetric: bool = False       # True if left/right differ enough to matter


@dataclass
class RepResult:
    """Per-repetition summary from the squat verification engine.

    ``confirmed`` is the rule-based classifier's verdict (True = a squat).
    ``gru_prob`` is always ``None`` now; it is kept so sessions stored by the
    project's earlier GRU verifier still load from the database.
    """
    rep_index: int
    max_depth_ratio: float         # femur-normalised depth (higher = deeper)
    knee_over_foot_dev: float      # worst knee-vs-foot horizontal deviation (norm.)
    symmetry_ok: bool
    phase_sequence: list = field(default_factory=list)
    gru_prob: Optional[float] = None    # legacy (earlier GRU verifier); None for the rules
    confirmed: Optional[bool] = None    # True = the classifier judged this a squat
    squat_score: Optional[int] = None   # rule-based classifier: checks passed (of 9)
    rule_failed: list = field(default_factory=list)   # rule-based classifier: failed checks
