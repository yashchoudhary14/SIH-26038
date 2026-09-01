"""Loaders for the added corpora, and the training-pool curation policy.

Each test here guards a failure that produces a plausible number rather than a
crash: a sixth grade class, a fellow eye across the train/val boundary, a
curated evaluation split, or a grade supplied entirely by one camera estate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drscreen.data.registry import (Sample, curate_training_pool, load_ddr,
                                    load_eyepacs, grade_source_matrix)


def _touch_images(directory, names):
    directory.mkdir(parents=True, exist_ok=True)
    for n in names:
        (directory / n).write_bytes(b"")


def test_ddr_ungradable_is_not_loaded_as_a_sixth_class(tmp_path):
    """DDR encodes ungradable as grade 5, not as a missing label.

    Loading that verbatim yields a sixth class that corrupts the confusion
    matrix, QWK and every effective-number class weight -- silently, because
    nothing raises. The loader must hand it back as grade=None/gradable=False
    so the cohort builder can drop it on purpose.
    """
    grading = tmp_path / "DR_grading"
    _touch_images(grading / "train", ["a.jpg", "b.jpg", "c.jpg"])
    (grading / "train.txt").write_text(
        "a.jpg 0\nb.jpg 4\nc.jpg 5\n", encoding="utf-8")

    got = load_ddr(tmp_path)
    by_stem = {s.image_path.stem: s for s in got}
    assert len(got) == 3

    assert by_stem["a"].grade == 0
    assert by_stem["b"].grade == 4
    assert by_stem["c"].grade is None, (
        "DDR grade 5 means ungradable; loading it as an ordinal grade adds a "
        "sixth class to a five-class problem.")
    assert by_stem["c"].gradable is False

    grades = {s.grade for s in got if s.grade is not None}
    assert max(grades) <= 4


def test_eyepacs_fellow_eyes_share_a_subject_id(tmp_path):
    """Both eyes of one patient must land in the same split.

    EyePACS ships <patient>_left and <patient>_right. Splitting on the image
    id puts one eye in train and the other in val, which inflates every metric
    -- the most common silent bug in published DR pipelines.
    """
    _touch_images(tmp_path / "train",
                  ["10_left.jpeg", "10_right.jpeg", "11_left.jpeg"])
    pd.DataFrame({"image": ["10_left", "10_right", "11_left"],
                  "level": [2, 3, 0]}).to_csv(tmp_path / "trainLabels.csv",
                                              index=False)

    got = load_eyepacs(tmp_path)
    assert len(got) == 3
    by_stem = {s.image_path.stem: s for s in got}
    assert by_stem["10_left"].subject_id == by_stem["10_right"].subject_id
    assert by_stem["10_left"].subject_id != by_stem["11_left"].subject_id
    assert by_stem["10_right"].grade == 3


def test_eyepacs_ignores_an_all_zero_submission_stub(tmp_path):
    """A sampleSubmission CSV has the same columns and all-zero grades.

    Mirrors of the competition ship one next to the real labels. Picking it up
    would relabel the entire corpus as grade 0 -- every image present, every
    grade wrong, and no error anywhere.
    """
    _touch_images(tmp_path / "train", ["1_left.jpeg", "1_right.jpeg"])
    pd.DataFrame({"image": ["1_left", "1_right"], "level": [0, 0]}).to_csv(
        tmp_path / "sampleSubmission.csv", index=False)
    assert load_eyepacs(tmp_path) == []

    pd.DataFrame({"image": ["1_left", "1_right"], "level": [0, 4]}).to_csv(
        tmp_path / "trainLabels.csv", index=False)
    got = load_eyepacs(tmp_path)
    assert {s.grade for s in got} == {0, 4}


def _pool(spec):
    """spec: {(grade, dataset): n} -> samples with unique subjects."""
    from pathlib import Path
    out = []
    for (grade, ds), n in spec.items():
        for i in range(n):
            uid = f"{ds}{grade}_{i}"
            out.append(Sample(Path(uid + ".png"), grade=grade, dataset=ds,
                              gradable=True, subject_id=uid))
    return out


def test_curation_never_drops_sight_threatening_grades():
    """Grades 3 and 4 are the binding constraint, so they are never capped.

    CORN trains task 3 only on grades 3-4; discarding any of them shrinks the
    only supervision the deepest conditional ever sees.
    """
    pool = _pool({(0, "eyepacs"): 4000, (1, "aptos2019"): 300,
                  (2, "eyepacs"): 2000, (3, "aptos2019"): 150,
                  (4, "ddr"): 120})
    kept = curate_training_pool(pool, cap_per_grade=200, seed=0)
    counts = {g: sum(1 for s in kept if s.grade == g) for g in range(5)}

    assert counts[3] == 150, "grade 3 must survive curation intact"
    assert counts[4] == 120, "grade 4 must survive curation intact"
    assert counts[0] <= 200 and counts[2] <= 200


def test_curation_mixes_sources_within_a_capped_grade():
    """A capped grade must not be handed to a single dataset.

    Truncating a concatenated pool would fill grade 0 entirely from whichever
    corpus was loaded first, so grade would correlate with imaging chain and
    the model could score well by recognising the camera instead of the
    disease.
    """
    pool = _pool({(0, "aptos2019"): 500, (0, "eyepacs"): 5000,
                  (0, "ddr"): 400, (3, "aptos2019"): 50})
    kept = curate_training_pool(pool, cap_per_grade=300, seed=0)
    row = grade_source_matrix(kept)[0]

    assert set(row) == {"aptos2019", "ddr", "eyepacs"}, (
        f"every available source must contribute to a capped grade, got {row}")
    # Round-robin by subject: with all three sources able to supply 100 each,
    # no source should dominate the cap.
    assert max(row.values()) <= 0.5 * sum(row.values()), (
        f"one source dominates the capped grade: {row}")


def test_curation_auto_cap_tracks_the_scarce_grades():
    """cap=0 resolves to twice the largest of grades 3 and 4.

    Tying the cap to what is actually scarce keeps the ratio stable as corpora
    are added, instead of hard-coding a number that silently stops making
    sense once EyePACS multiplies the pool.
    """
    pool = _pool({(0, "eyepacs"): 9000, (3, "aptos2019"): 100,
                  (4, "ddr"): 250})
    kept = curate_training_pool(pool, cap_per_grade=0, seed=0)
    counts = {g: sum(1 for s in kept if s.grade == g) for g in range(5)}
    assert counts[0] == 500, f"expected 2 x max(100, 250) = 500, got {counts[0]}"


def test_curation_is_deterministic_for_a_seed():
    pool = _pool({(0, "eyepacs"): 1000, (0, "aptos2019"): 1000,
                  (3, "ddr"): 40})
    a = [s.subject_id for s in curate_training_pool(pool, 100, seed=7)]
    b = [s.subject_id for s in curate_training_pool(pool, 100, seed=7)]
    c = [s.subject_id for s in curate_training_pool(pool, 100, seed=8)]
    assert a == b
    assert a != c
