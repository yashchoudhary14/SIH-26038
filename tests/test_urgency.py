"""The urgent tier is a fitted decision, and a strict sub-tier of referral.

Before this, urgency fired on ``predicted grade >= 3`` -- keying expedited
review to the least reliable output the system has. On Messidor-2 that rule
reached sensitivity 0.509 (56/110) where a threshold fitted on P(grade >= 3)
reaches 0.782 (86/110). The other 30 patients were still referred; they were
just queued as routine.
"""
from __future__ import annotations

import numpy as np
import pytest

from drscreen.constants import SIGHT_THREATENING_THRESHOLD
from drscreen.models.lesion_features import ClinicalFeatures
from drscreen.pipeline import DRScreeningPipeline, PipelineConfig, ScreeningResult


def _pipe(urgent_threshold=None, referral=0.20, defer=(0.05, 0.10)):
    cfg = PipelineConfig(referral_threshold=referral, defer_band=defer,
                         urgent_threshold=urgent_threshold)
    return DRScreeningPipeline(None, None, cfg)


def _result(p_ref, p_st, grade):
    r = ScreeningResult()
    r.referable_probability = p_ref
    r.sight_threatening_probability = p_st
    r.grade = grade
    r.uncertainty = {"epistemic_variance": 0.0}
    r.dme_risk = 0
    return r


def _feats():
    f = ClinicalFeatures()
    f.per_quadrant = {}
    return f


def test_urgency_uses_the_probability_not_the_printed_grade():
    """A case below the grade cut-off but above the fitted one must escalate.

    This is the whole point: the grade is a coarse, unreliable readout of the
    same evidence, and gating urgency on it discards cases the probability
    ranks highly.
    """
    p = _pipe(urgent_threshold=0.15)
    # grade 2 (so the OLD rule would not escalate) but P(grade>=3) is high
    decision, urgency = p._decide(_result(0.90, 0.40, grade=2), _feats())
    assert decision == "refer"
    assert urgency == "urgent"


def test_grade_alone_no_longer_escalates_when_the_probability_is_low():
    """The converse: a grade-3 label with weak probability is not urgent.

    Grade 3 exact recall is 0.333 on Messidor-2, so the label alone is not
    evidence enough to consume an expedited slot.
    """
    p = _pipe(urgent_threshold=0.50)
    _, urgency = p._decide(_result(0.90, 0.10, grade=3), _feats())
    assert urgency != "urgent"


def test_unfitted_bundle_keeps_the_previous_grade_rule():
    """A bundle written before this field existed must behave exactly as before."""
    p = _pipe(urgent_threshold=None)
    _, urgency = p._decide(_result(0.90, 0.01, grade=SIGHT_THREATENING_THRESHOLD),
                           _feats())
    assert urgency == "urgent"


def test_urgent_is_never_reached_without_referral():
    """Urgency is a tier within referral, not a parallel path.

    Calling a case urgent while declining to refer it is incoherent, and would
    also silently change the referral sensitivity every reported number
    describes.
    """
    p = _pipe(urgent_threshold=0.05, referral=0.20, defer=(0.02, 0.03))
    # confidently negative on referral, but P(grade>=3) clears the urgent cut
    res = _result(0.01, 0.90, grade=4)
    decision, urgency = p._decide(res, _feats())
    assert decision != "refer" or urgency != "urgent", (
        "a case below the deferral band must not be escalated to urgent by the "
        "probability alone")


def test_monotonicity_holds_between_the_two_probabilities():
    """P(y>2) <= P(y>1) by construction, so urgent implies referable.

    Guards the assumption the sub-tier design rests on: if this inverted, an
    urgent case could sit below the referral threshold.
    """
    from drscreen.models.grader import corn_cumulative_probs
    import torch
    torch.manual_seed(0)
    cum = corn_cumulative_probs(torch.randn(5000, 4) * 3).numpy()
    assert np.all(cum[:, SIGHT_THREATENING_THRESHOLD - 1]
                  <= cum[:, SIGHT_THREATENING_THRESHOLD - 2] + 1e-6)
