"""An unassessed lesion channel must not be allowed to escalate a case.

IDRiD ships no neovascularisation masks, so on the real cohort the NV channel
of the segmenter trains against all-zero targets and is never supervised. At
the default 0.5 threshold it still returns a blob on almost every retina --
and ``_decide`` escalated on that blob before any other rule ran, ahead of the
guard the rest of the method is built around.

The measured consequence: **631 of 631** held-out real photographs came back
``refer / urgent``, healthy eyes included. A screener that escalates every
patient has destroyed the triage that is its entire reason to exist, and it
does it while looking confident -- P(referable) was 0.000004 on the grade-0
case that got flagged for emergency review.

``rule_grade`` already refuses to read this channel when it is unassessed
(``constants.PIXEL_ANNOTATED_LESION_CLASSES`` documents why). These tests pin
the decision rule to the same contract, in both directions: silent when the
channel is unsupervised, still escalating on sight when it is not.
"""
from __future__ import annotations

from drscreen.models.lesion_features import ClinicalFeatures
from drscreen.pipeline import DRScreeningPipeline, PipelineConfig, ScreeningResult


def _pipe():
    return DRScreeningPipeline(None, None, PipelineConfig(
        referral_threshold=0.20, defer_band=(0.05, 0.35), urgent_threshold=0.12))


def _confident_negative():
    r = ScreeningResult()
    r.referable_probability = 4.8e-06      # the real grade-0 case, verbatim
    r.sight_threatening_probability = 0.0
    r.grade = 0
    r.uncertainty = {"epistemic_variance": 0.0}
    r.dme_risk = 0
    return r


def _feats(nv_at_disc=False, nv_elsewhere=False, unassessed=()):
    f = ClinicalFeatures()
    f.per_quadrant = {}
    f.nv_at_disc = nv_at_disc
    f.nv_elsewhere = nv_elsewhere
    f.unassessed = tuple(unassessed)
    return f


def test_unsupervised_nv_channel_cannot_escalate_a_confident_negative():
    """The regression: this returned ("refer", "urgent") for every real image."""
    decision, urgency = _pipe()._decide(
        _confident_negative(),
        _feats(nv_elsewhere=True, unassessed=("neovascularization",)))
    assert (decision, urgency) == ("auto_report", "routine")


def test_unsupervised_nv_at_the_disc_cannot_escalate_either():
    """NVD is the more severe finding, which is exactly why an unsupervised
    channel claiming it must not be believed."""
    decision, urgency = _pipe()._decide(
        _confident_negative(),
        _feats(nv_at_disc=True, unassessed=("neovascularization",)))
    assert (decision, urgency) == ("auto_report", "routine")


def test_supervised_nv_still_escalates_on_sight():
    """The guard is about provenance, not about softening the rule.

    A cohort that does annotate NV -- the synthetic phantoms do -- must keep
    the original behaviour: neovascularisation defines proliferative DR and
    outranks a confident negative.
    """
    for feats in (_feats(nv_at_disc=True), _feats(nv_elsewhere=True)):
        assert _pipe()._decide(_confident_negative(), feats) == ("refer", "urgent")


def test_a_screener_that_escalates_everything_is_the_failure_being_guarded():
    """Triage has to survive an unsupervised channel that fires on every image.

    Stated as the property rather than the mechanism: feed the decision rule a
    spread of confident negatives with the NV flag stuck on, and the referral
    rate must track the probabilities, not the stuck flag.
    """
    pipe = _pipe()
    feats = _feats(nv_elsewhere=True, unassessed=("neovascularization",))
    referred = 0
    for p in (0.0, 1e-05, 0.001, 0.01, 0.04):
        r = _confident_negative()
        r.referable_probability = p
        if pipe._decide(r, feats)[0] == "refer":
            referred += 1
    assert referred == 0
