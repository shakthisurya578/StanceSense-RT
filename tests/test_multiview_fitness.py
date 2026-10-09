"""Tests for the Multi-View Fitness Dataset adapter (filename parsing only —
no video decoding, no MediaPipe, so these run without those dependencies).
"""
import os
import zipfile

from stancesense.datasets.multiview_fitness import (
    ClipMeta, POSITIVE_EXERCISE, discover_clips, extract_clip,
)


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "wb").close()


def test_discover_from_directory_parses_every_field(tmp_path):
    root = tmp_path / "dataset"
    _touch(str(root / "squat" / "good" / "front" / "subject_003_squat_good_front.mp4"))
    _touch(str(root / "abs" / "bad" / "side" / "subject_012_abs_bad_side.mp4"))

    clips = discover_clips(str(root))
    assert len(clips) == 2
    squat = next(c for c in clips if c.exercise == "squat")
    assert (squat.subject_id, squat.quality, squat.view) == (3, "good", "front")
    assert squat.label == 1
    abs_clip = next(c for c in clips if c.exercise == "abs")
    assert abs_clip.label == 0


def test_bicep_typo_is_normalised_to_bicep_curl(tmp_path):
    """One real clip in the archive is named '..._bicep_good_side.mp4', not
    '..._bicep_curl_good_side.mp4'. Left alone it becomes its own one-video
    exercise class; the adapter folds it back into bicep_curl."""
    root = tmp_path / "dataset"
    _touch(str(root / "subject_003_bicep_good_side.mp4"))

    clips = discover_clips(str(root))
    assert len(clips) == 1
    assert clips[0].exercise == "bicep_curl"


def test_malformed_filenames_are_skipped_not_crashed_on(tmp_path):
    root = tmp_path / "dataset"
    _touch(str(root / "readme.mp4"))                       # doesn't match the pattern
    _touch(str(root / "subject_x_squat_good_front.mp4"))    # non-numeric subject
    _touch(str(root / "subject_001_squat_ok_front.mp4"))    # quality not good/bad
    _touch(str(root / "subject_001_squat_good_front.mp4"))  # the one valid file

    clips = discover_clips(str(root))
    assert len(clips) == 1
    assert clips[0].filename == "subject_001_squat_good_front.mp4"


def test_discover_from_zip_matches_directory_layout(tmp_path):
    zpath = tmp_path / "dataset.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("Dataset/squat/good/front/subject_005_squat_good_front.mp4", b"fake-mp4")
        z.writestr("Dataset/tricep/bad/diagonal/subject_005_tricep_bad_diagonal.mp4", b"fake-mp4")
        z.writestr("Dataset/squat/good/", b"")  # a directory entry, must be ignored

    clips = discover_clips(str(zpath))
    assert len(clips) == 2
    assert {c.exercise for c in clips} == {"squat", "tricep"}
    assert all(c.zip_member is not None for c in clips)


def test_extract_clip_from_zip_writes_the_bytes(tmp_path):
    zpath = tmp_path / "dataset.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("squat/good/front/subject_001_squat_good_front.mp4", b"payload-bytes")
    clip = discover_clips(str(zpath))[0]

    dest = tmp_path / "extracted"
    out_path = extract_clip(clip, str(dest))
    assert os.path.exists(out_path)
    with open(out_path, "rb") as fh:
        assert fh.read() == b"payload-bytes"


def test_extract_clip_from_directory_is_a_no_op(tmp_path):
    root = tmp_path / "dataset"
    path = root / "subject_001_squat_good_front.mp4"
    _touch(str(path))
    clip = discover_clips(str(root))[0]
    assert extract_clip(clip, str(tmp_path / "unused")) == clip.source


def test_clip_id_is_stable_and_human_readable():
    c = ClipMeta(7, "push_up", "bad", "diagonal", "x.mp4", source="dummy")
    assert c.clip_id == "subject_007_push_up_bad_diagonal"


def test_positive_exercise_constant_is_squat():
    assert POSITIVE_EXERCISE == "squat"


# ------------------------------------------------- duplicate recordings

from stancesense.datasets.multiview_fitness import resolve_duplicates  # noqa: E402


def test_cross_person_and_label_conflict_duplicates_drop_every_copy():
    ex = resolve_duplicates([
        ["subject_005_squat_bad_side", "subject_006_squat_bad_side"],          # two people
        ["subject_007_back_bad_diagonal", "subject_007_squat_good_diagonal"],  # squat vs not
        ["subject_023_back_bad_diagonal", "subject_023_back_bad_side"],        # two views
    ])
    assert set(ex) == {"subject_005_squat_bad_side", "subject_006_squat_bad_side",
                       "subject_007_back_bad_diagonal", "subject_007_squat_good_diagonal",
                       "subject_023_back_bad_diagonal", "subject_023_back_bad_side"}


def test_same_person_label_and_view_keeps_exactly_one_copy():
    ex = resolve_duplicates([["subject_016_shoulder_bad_side", "subject_016_shoulder_good_side"],
                             ["subject_015_shoulder_bad_side", "subject_015_tricep_bad_side"]])
    assert ex == {"subject_016_shoulder_good_side": "duplicate of subject_016_shoulder_bad_side",
                  "subject_015_tricep_bad_side": "duplicate of subject_015_shoulder_bad_side"}
