"""Multi-View Fitness Exercise Dataset adapter (evaluation data for the rule classifier).

WHY THIS EXISTS
----------------
The rule-based squat classifier is evaluated on this dataset because it is raw
RGB video: running MediaPipe BlazePose over it gives exactly the kind of
landmarks the live camera produces, from three real viewpoints. (MM-Fit's
"pose" is a precomputed, non-MediaPipe skeleton, so it is used only for the
offline demo.)

WHAT THE DATASET GIVES US (verified against the real archive, audited 2026-09-15)
-----------------------------------------------------------------------------
Source: "A Multi-View Raw Video Dataset of Seven Fitness Exercises" — 26
nominal subjects performing 7 exercises (abs, back, bicep_curl, push_up,
shoulder, squat, tricep) in both good and bad form, filmed from front, side
and diagonal cameras. It ships as three zip archives (subject-wise,
exercise-wise, exercise-quality-wise); these were verified to contain the
IDENTICAL 1042 video clips just reorganised into different folder layouts
(same basenames, same count, same total size across all three) — this adapter
therefore only ever needs to be pointed at one of them.

Two subjects break the "26 uniform subjects" assumption the dataset's own
README implies:
  subject_025  only 4 clips exist for this subject in the whole archive
               (tricep good/bad, front+diagonal only — no side view). ZERO
               squat videos.
  subject_026  missing `back` and `push_up` entirely (0 clips each), but has
               complete squat data. Kept, but its negative-class exposure is
               narrower than every other subject's — worth stating per-fold.

Unlike MM-Fit, this dataset carries only a WHOLE-CLIP label (subject,
exercise, quality, view) — no rep count, no segment boundaries, no pose. It
therefore cannot grade a rep counter; it is used for clip-level
squat / not-squat evaluation.

MediaPipe has never been run on this footage before, so extracting features
from it is a separate, one-time preprocessing pass
(``scripts/preprocess_multiview.py``); this module's only job is enumerating
clips and parsing their labels from filenames — no video decoding, no
MediaPipe import, so it stays usable (and testable) without those heavy
dependencies installed, matching how ``mmfit.py`` keeps its own heavy
imports scoped to where they are actually needed.

FILE NAMING
-----------
    subject_<NNN>_<exercise>_<good|bad>_<front|side|diagonal>.mp4

One filename typo was found and is silently normalised: a single clip
(subject_003, bicep_curl, good, side) is named "..._bicep_good_side.mp4"
instead of "..._bicep_curl_good_side.mp4" — left alone it would parse as its
own one-video "bicep" exercise class instead of joining bicep_curl.
"""
from __future__ import annotations

import csv
import os
import re
import zipfile
from dataclasses import dataclass
from glob import glob
from typing import Optional

POSITIVE_EXERCISE = "squat"

EXERCISES = ["abs", "back", "bicep_curl", "push_up", "shoulder", "squat", "tricep"]
VIEWS = ["front", "side", "diagonal"]
QUALITIES = ["good", "bad"]

# Verified against the archive (see module docstring): this subject has no
# squat clips at all. Any subject-independent squat/not-squat evaluation must
# exclude it, or the fold that holds it out tests on zero positive examples.

_NAME_RE = re.compile(
    r"subject_(\d+)_(.+)_(good|bad)_(front|side|diagonal)\.mp4$", re.IGNORECASE)

# The one filename typo verified in the archive (see module docstring).
_EXERCISE_ALIASES = {"bicep": "bicep_curl"}


@dataclass
class ClipMeta:
    """One labelled video clip — metadata only, no pixels/pose loaded yet."""
    subject_id: int
    exercise: str
    quality: str                        # "good" | "bad"
    view: str                           # "front" | "side" | "diagonal"
    filename: str                       # e.g. subject_003_squat_good_front.mp4
    source: str                         # zip path or directory this was found under
    zip_member: Optional[str] = None    # in-archive path, if source is a zip

    @property
    def clip_id(self) -> str:
        return f"subject_{self.subject_id:03d}_{self.exercise}_{self.quality}_{self.view}"

    @property
    def label(self) -> int:
        """1 = squat (good OR bad form — form quality is not this task's label), 0 = anything else."""
        return int(self.exercise == POSITIVE_EXERCISE)


def _parse_filename(name: str) -> Optional[tuple]:
    m = _NAME_RE.match(name)
    if not m:
        return None
    subj, exercise, quality, view = m.groups()
    exercise = _EXERCISE_ALIASES.get(exercise.lower(), exercise.lower())
    return int(subj), exercise, quality.lower(), view.lower()


def discover_clips(root: str) -> list:
    """Enumerate every labelled clip under ``root``.

    ``root`` is either a path to one of the dataset's zip archives (any of the
    three layouts — verified to hold identical clips) or a directory of
    already-extracted ``.mp4`` files in any subfolder structure. No dataset
    path is hardcoded here; callers supply ``root`` explicitly (e.g. a script's
    ``--data`` flag).
    """
    if os.path.isfile(root) and root.lower().endswith(".zip"):
        return _discover_from_zip(root)
    if os.path.isdir(root):
        return _discover_from_dir(root)
    raise FileNotFoundError(f"{root!r} is neither a .zip archive nor a directory")


def _discover_from_zip(zip_path: str) -> list:
    clips = []
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if not info.filename.lower().endswith(".mp4"):
                continue
            base = info.filename.rsplit("/", 1)[-1]
            parsed = _parse_filename(base)
            if parsed is None:
                continue
            subj, exercise, quality, view = parsed
            clips.append(ClipMeta(subj, exercise, quality, view, base,
                                  source=zip_path, zip_member=info.filename))
    return sorted(clips, key=lambda c: c.clip_id)


def _discover_from_dir(root: str) -> list:
    clips = []
    for path in glob(os.path.join(root, "**", "*.mp4"), recursive=True):
        base = os.path.basename(path)
        parsed = _parse_filename(base)
        if parsed is None:
            continue
        subj, exercise, quality, view = parsed
        clips.append(ClipMeta(subj, exercise, quality, view, base,
                              source=path, zip_member=None))
    return sorted(clips, key=lambda c: c.clip_id)


def extract_clip(clip: ClipMeta, dest_dir: str) -> str:
    """Materialise ``clip`` as a real file on disk under ``dest_dir``.

    A directory-sourced clip already lives on disk and is returned as-is. A
    zip-sourced clip is extracted once into ``dest_dir``; the caller owns
    cleanup of that temp file (``clip.zip_member is not None`` tells you
    whether one was created).
    """
    if clip.zip_member is None:
        return clip.source
    os.makedirs(dest_dir, exist_ok=True)
    out_path = os.path.join(dest_dir, clip.filename)
    if not os.path.exists(out_path):
        with zipfile.ZipFile(clip.source) as z, z.open(clip.zip_member) as src, \
                open(out_path, "wb") as dst:
            dst.write(src.read())
    return out_path


# ---------------------------------------------------------------------------
# Duplicate recordings (audited 2026-10-05)
# ---------------------------------------------------------------------------
# The archive holds 1042 video files but only 1008 distinct recordings: 33
# recordings are stored under 2-3 names (byte-identical files, same CRC and
# size). Some copies are filed under two DIFFERENT people (e.g. subject_005 and
# subject_006 share their side-view squat videos), which leaks a test person's
# video into another person's training data under leave-one-person-out; one is
# filed as both "squat" and "back" (subject_007, diagonal), a direct label
# conflict. Resolution, applied before any split:
#   1. filed under more than one person          -> drop every copy
#   2. filed as squat AND as a non-squat exercise -> drop every copy
#   3. filed under more than one camera view      -> drop every copy
#   4. otherwise (only good/bad or the non-squat exercise name differs, which the
#      squat/not-squat label ignores) -> keep the alphabetically first copy
DUPLICATES_FILE = "duplicates.json"


def scan_duplicate_recordings(zip_path: str) -> list:
    """Groups of clip_ids whose video files are byte-identical (CRC + size)."""
    by = {}
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename.rsplit("/", 1)[-1]
            if not name.lower().endswith(".mp4") or _parse_filename(name) is None:
                continue
            by.setdefault((info.CRC, info.file_size), set()).add(name[:-4])
    return sorted(sorted(g) for g in by.values() if len(g) > 1)


def resolve_duplicates(groups: list) -> dict:
    """{clip_id: reason} for every clip to EXCLUDE, per the rules above."""
    excluded = {}
    for group in groups:
        parsed = {cid: _parse_filename(cid + ".mp4") for cid in group}
        subjects = {p[0] for p in parsed.values()}
        is_squat = {p[1] == POSITIVE_EXERCISE for p in parsed.values()}
        views = {p[3] for p in parsed.values()}
        if len(subjects) > 1:
            reason = "same recording filed under different people"
        elif len(is_squat) > 1:
            reason = "same recording filed as squat and as non-squat"
        elif len(views) > 1:
            reason = "same recording filed under different camera views"
        else:
            for cid in sorted(group)[1:]:
                excluded[cid] = f"duplicate of {sorted(group)[0]}"
            continue
        for cid in group:
            excluded[cid] = reason
    return excluded


def load_manifest(processed_dir: str, exclude_subjects=frozenset()) -> list:
    """Rows of the preprocessing manifest(s): one per cached clip.

    Loads every manifest*.csv under ``processed_dir`` (plain or sharded run).

    A parallel preprocessing run (``--shard-count > 1``) writes one
    ``manifest.shardN.csv`` per worker to avoid concurrent-write corruption;
    this merges all of them, deduplicating on clip_id in case a clip was
    reprocessed (last write wins).
    """
    paths = sorted(glob(os.path.join(processed_dir, "manifest*.csv")))
    if not paths:
        raise SystemExit(
            f"no manifest*.csv under {processed_dir!r} — "
            "run scripts/preprocess_multiview.py first")
    by_id = {}
    for path in paths:
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                by_id[row["clip_id"]] = row
    rows = [row for row in by_id.values()
           if int(row["subject_id"]) not in exclude_subjects]
    return rows
