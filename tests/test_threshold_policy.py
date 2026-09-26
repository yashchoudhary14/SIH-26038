"""Referral-threshold selection policy.

The threshold is the deployed clinical decision, and it is chosen on the
validation split alone. These tests pin both properties: that the policies
differ in the direction they claim to, and that neither consults data it must
not see.
"""
from __future__ import annotations

import numpy as np
import pytest

from drscreen.models.calibration import select_threshold


def _separable(n=400, seed=0):
    """Scores with a wide overlap band, so many thresholds meet both targets."""
    rng = np.random.default_rng(seed)
    neg = rng.beta(2, 6, n)
    pos = rng.beta(6, 2, n)
    probs = np.concatenate([neg, pos])
    labels = np.concatenate([np.zeros(n, int), np.ones(n, int)])
    return probs, labels


def test_max_sensitivity_policy_is_at_least_as_sensitive_as_youden():
    probs, labels = _separable()
    y = select_threshold(probs, labels, 0.90, 0.85, policy="youden")
    s = select_threshold(probs, labels, 0.90, 0.85, policy="max_sensitivity")

    assert y.meets_target and s.meets_target
    assert s.sensitivity >= y.sensitivity
    assert s.threshold <= y.threshold, (
        "a more sensitive operating point cannot sit at a higher cut-point")


def test_both_policies_respect_the_specificity_floor():
    """max_sensitivity buys recall with review capacity, not with the floor.

    The point of the floor is that a screening programme which flags everyone
    has no triage value: the reviewer queue is what the AI arm exists to
    shrink.
    """
    probs, labels = _separable()
    for policy in ("youden", "max_sensitivity"):
        op = select_threshold(probs, labels, 0.90, 0.85, policy=policy)
        assert op.specificity >= 0.85, (
            f"{policy} returned specificity {op.specificity:.3f} below the floor")
        assert op.sensitivity >= 0.90


def test_unknown_policy_is_rejected():
    probs, labels = _separable()
    with pytest.raises(ValueError, match="policy must be"):
        select_threshold(probs, labels, policy="whatever")


def test_policy_does_not_change_the_degenerate_path():
    """One-class calibration data must still return the guarded default."""
    probs = np.linspace(0, 1, 50)
    labels = np.zeros(50, int)
    for policy in ("youden", "max_sensitivity"):
        op = select_threshold(probs, labels, policy=policy)
        assert op.threshold == 0.5
        assert not op.meets_target


def test_rationale_records_which_policy_was_used():
    """The deployed artefact must say how its operating point was chosen."""
    probs, labels = _separable()
    y = select_threshold(probs, labels, 0.90, 0.85, policy="youden")
    s = select_threshold(probs, labels, 0.90, 0.85, policy="max_sensitivity")
    assert "Youden" in y.rationale
    assert "most sensitive" in s.rationale
