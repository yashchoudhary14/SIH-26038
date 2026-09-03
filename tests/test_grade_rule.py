"""The grade decision rule: one rule everywhere, and its cut-points fitted.

Two defects are guarded here.

**Two rules for one decision.** The pipeline assigned the grade with
``argmax(class_probs)`` while training and validation used the ordinal
``corn_predict``. Those disagreed on 3.65% of the internal test split, with
argmax the worse of the pair (QWK 0.8855 vs 0.8939) -- so the served grade was
one that no reported metric described.

**An un-fitted boundary.** The referral threshold has always been fitted on
validation data; the grade boundaries sat at a hard-coded 0.5. Exact grade-3
recall fell across three checkpoints (0.426 -> 0.383 -> 0.362) while referral
sensitivity rose to 1.000: the model was ordering severity correctly and the
rule reading that ordering was mis-set.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from drscreen.models.calibration import (GRADE_OBJECTIVES, fit_grade_thresholds,
                                         grades_from_cumulative)
from drscreen.models.grader import (corn_class_probs, corn_cumulative_probs,
                                    corn_predict, cumulative_from_class_probs,
                                    grade_from_cumulative)


def _logits(n=2000, k=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, k, generator=g) * 2


def test_scalar_threshold_reproduces_the_previous_counting_rule():
    """Backward compatibility: an un-fitted run must be comparable to old ones.

    Cumulative probabilities are monotone non-increasing, so for a scalar
    threshold the passes form a prefix and counting equals first-failure.
    """
    z = _logits()
    previous = (corn_cumulative_probs(z) > 0.5).sum(dim=1).long()
    assert torch.equal(corn_predict(z, 0.5), previous)


def test_first_failure_differs_from_counting_once_thresholds_vary():
    """Per-boundary cut-points are exactly when the two rules come apart.

    A low cut-point on a later boundary can pass after an earlier one has
    failed. Counting would return a grade that no chain of conditionals
    supports; first-failure stays inside the ordinal model.
    """
    z = _logits(20000)
    thr = torch.tensor([0.30, 0.39, 0.39, 0.36])
    counted = (corn_cumulative_probs(z) > thr).sum(dim=1).long()
    stopped = corn_predict(z, thr)
    assert not torch.equal(counted, stopped), (
        "expected the two rules to differ somewhere on 20k samples")
    # Where they differ, first-failure must be the shorter (prefix) answer.
    d = counted != stopped
    assert bool((stopped[d] <= counted[d]).all())


def test_cumulative_round_trips_through_class_probabilities():
    """Needed so the rule can be applied to an averaged MC-dropout posterior."""
    z = _logits()
    direct = corn_cumulative_probs(z)
    recovered = cumulative_from_class_probs(corn_class_probs(z))
    assert torch.allclose(direct, recovered, atol=1e-6)


def test_numpy_and_torch_rules_agree():
    """The fit runs in numpy for speed; inference runs in torch. They must match."""
    z = _logits()
    cum = corn_cumulative_probs(z).numpy().astype(np.float64)
    thr = np.array([0.30, 0.39, 0.39, 0.36])
    a = grades_from_cumulative(cum, thr)
    b = grade_from_cumulative(torch.from_numpy(cum), torch.from_numpy(thr)).numpy()
    assert (a == b).all()


def test_wrong_number_of_thresholds_is_rejected():
    with pytest.raises(ValueError, match="expected 4 grade thresholds"):
        corn_predict(_logits(10), [0.5, 0.5])


def _separable(n=1500, seed=1):
    """Labels with a genuine ordinal signal, so fitting has something to find."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 5, n)
    # Cumulative probabilities that order correctly but sit systematically low,
    # which is the situation a 0.5 cut-point handles badly.
    cum = np.zeros((n, 4))
    for k in range(4):
        cum[:, k] = np.clip(0.42 * (y > k) + 0.18 + rng.normal(0, 0.08, n), 0, 1)
    cum = np.sort(cum, axis=1)[:, ::-1]      # enforce monotone non-increasing
    return cum, y


def test_fitting_beats_the_unfitted_default_on_its_objective():
    cum, y = _separable()
    base = np.full(4, 0.5)

    def macro(p):
        return float(np.mean([(p[y == g] == g).mean() for g in range(5)
                              if (y == g).any()]))

    thr = fit_grade_thresholds(cum, y, objective="macro_recall")
    assert macro(grades_from_cumulative(cum, thr)) > macro(
        grades_from_cumulative(cum, base))


def test_every_objective_is_supported_and_returns_one_cutpoint_per_boundary():
    cum, y = _separable()
    for obj in GRADE_OBJECTIVES:
        thr = fit_grade_thresholds(cum, y, objective=obj)
        assert thr.shape == (4,)
        assert np.all((thr >= 0.0) & (thr <= 1.0))


def test_unknown_objective_is_rejected():
    cum, y = _separable()
    with pytest.raises(ValueError, match="objective must be one of"):
        fit_grade_thresholds(cum, y, objective="accuracy")


def test_unlabelled_cases_are_excluded_from_the_fit():
    """Messidor-2 ships ungradable images as -1; they must not steer the rule."""
    cum, y = _separable()
    y_mixed = y.copy()
    y_mixed[::7] = -1
    a = fit_grade_thresholds(cum, y_mixed)
    b = fit_grade_thresholds(cum[y_mixed >= 0], y_mixed[y_mixed >= 0])
    assert np.allclose(a, b)


def test_degenerate_input_falls_back_to_the_neutral_rule():
    assert np.allclose(fit_grade_thresholds(np.zeros((0, 4)), np.zeros(0)), 0.5)
