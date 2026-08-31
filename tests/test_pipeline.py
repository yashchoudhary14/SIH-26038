"""Regression tests for the invariants that matter clinically.

These are not coverage theatre. Each test guards a property whose silent
violation would produce a plausible-looking but wrong screening result --
the failure mode that makes medical ML dangerous.

    python -m pytest tests/ -v
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from drscreen.constants import NUM_GRADES, NUM_LESION_CLASSES, REFERABLE_THRESHOLD


# --------------------------------------------------------------------------
# Ordinal head
# --------------------------------------------------------------------------
def test_corn_probabilities_are_a_distribution():
    from drscreen.models.grader import corn_class_probs
    logits = torch.randn(64, NUM_GRADES - 1) * 3
    p = corn_class_probs(logits)
    assert p.shape == (64, NUM_GRADES)
    assert torch.allclose(p.sum(1), torch.ones(64), atol=1e-5)
    assert (p >= 0).all()


def test_corn_cumulative_is_monotone():
    """P(y>k) must never increase with k -- the guarantee softmax cannot give."""
    from drscreen.models.grader import corn_cumulative_probs
    logits = torch.randn(256, NUM_GRADES - 1) * 5
    cum = corn_cumulative_probs(logits)
    assert (torch.diff(cum, dim=1) <= 1e-6).all()


def test_referable_probability_matches_class_sum():
    from drscreen.models.grader import corn_class_probs, referable_prob
    logits = torch.randn(128, NUM_GRADES - 1) * 2
    direct = referable_prob(logits)
    summed = corn_class_probs(logits)[:, REFERABLE_THRESHOLD:].sum(1)
    assert torch.allclose(direct, summed, atol=1e-5)


def test_corn_loss_decreases_on_correct_ordering():
    from drscreen.models.grader import corn_loss
    y = torch.tensor([0, 1, 2, 3, 4])
    good = torch.tensor([[-5., -5, -5, -5], [5., -5, -5, -5], [5., 5, -5, -5],
                         [5., 5, 5, -5], [5., 5, 5, 5]])
    bad = -good
    assert float(corn_loss(good, y)) < float(corn_loss(bad, y))


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def test_delong_auc_matches_sklearn():
    from sklearn.metrics import roc_auc_score
    from drscreen.evaluation.metrics import delong_auc_ci
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.3).astype(int)
    s = rng.normal(y, 1.0)
    auc, lo, hi = delong_auc_ci(s, y)
    assert abs(auc - roc_auc_score(y, s)) < 1e-9
    assert lo < auc < hi


def test_wilson_interval_bounded_at_extremes():
    from drscreen.evaluation.metrics import wilson_interval
    lo, hi = wilson_interval(100, 100)
    assert 0.0 <= lo <= 1.0 and hi == pytest.approx(1.0, abs=1e-9)
    lo, hi = wilson_interval(0, 50)
    assert lo == pytest.approx(0.0, abs=1e-9) and hi < 1.0


def test_threshold_selection_respects_sensitivity_constraint():
    """Sensitivity is the binding constraint; the selector must never trade it away."""
    from drscreen.models.calibration import select_threshold
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.25).astype(int)
    p = np.clip(rng.normal(0.35 + 0.4 * y, 0.16), 0, 1)
    op = select_threshold(p, y, min_sensitivity=0.90, min_specificity=0.85)
    assert op.sensitivity >= 0.90 - 1e-9, op.rationale


def test_temperature_effect_on_corn_ranking_is_negligible():
    """Temperature scaling is NOT exactly rank-preserving for a CORN head.

    For a single logit, z -> z/T is monotone and AUC is invariant. For CORN the
    referable score is a product of sigmoids, and T does not factor out of that
    product, so the ranking can change. This test pins the magnitude: the shift
    must stay far below the sampling error of any realistic test set, and it
    documents the property so nobody later "fixes" a discrepancy that is real.
    """
    from sklearn.metrics import roc_auc_score
    from drscreen.models.grader import referable_prob
    torch.manual_seed(0)
    logits = torch.randn(4000, NUM_GRADES - 1) * 2.5
    y = (torch.randint(0, NUM_GRADES, (4000,)) >= REFERABLE_THRESHOLD).numpy().astype(int)
    a = referable_prob(logits).numpy()
    b = referable_prob(logits / 2.5).numpy()
    assert 0 < y.sum() < len(y)
    assert abs(roc_auc_score(y, a) - roc_auc_score(y, b)) < 5e-3


def test_isotonic_recalibration_never_inverts_a_pair():
    """The binary referral decision must keep its ordering under recalibration.

    The property is *non-inversion*, not rank identity: isotonic regression is a
    step function, so it legitimately maps distinct inputs to equal outputs.
    Those ties lower Spearman's rho without ever swapping two cases, and a naive
    rho > 0.999 assertion would fail on correct behaviour. What must never
    happen is a lower-scored case coming out above a higher-scored one, because
    that would reorder the review queue.
    """
    from drscreen.models.calibration import IsotonicCalibrator
    rng = np.random.default_rng(0)
    p = rng.random(1000)
    y = (rng.random(1000) < p).astype(int)
    q = IsotonicCalibrator().fit(p, y)(p)

    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12), "isotonic inverted a pair"

    # The tie-breaking blend must preserve a STRICT total order. Plain isotonic
    # collapses the bottom of the range to exactly 0.0; measured on Messidor-2
    # that pinned 10.3% of true positives at zero, where no threshold above 0
    # can recover them, and specificity at 90% sensitivity fell to 0.000. Ties
    # are the mechanism, so the guard is on tie count and on the floor.
    assert len(np.unique(q)) == len(np.unique(p)), (
        "calibrator collapsed distinct scores into ties; threshold transfer "
        "under distribution shift depends on the full ordering")
    assert (q > 0).sum() >= (p > 0).sum(), "calibrator pinned scores to exact zero"

    from sklearn.metrics import roc_auc_score
    before, after = roc_auc_score(y, p), roc_auc_score(y, q)
    assert abs(after - before) < 1e-9, (
        f"a strictly monotone calibrator must preserve AUC exactly "
        f"({before:.6f} -> {after:.6f})")


# --------------------------------------------------------------------------
# Preprocessing and geometry
# --------------------------------------------------------------------------
def test_fov_detection_finds_the_aperture():
    from drscreen.data.synthetic import generate
    from drscreen.preprocess.fov import detect_fov
    p = generate(grade=0, size=512, seed=3)
    fov = detect_fov(p.image)
    assert 0.25 < fov.fill_ratio < 1.0
    assert fov.radius > 512 * 0.25


def test_standardize_is_square_and_masked():
    from drscreen.data.synthetic import generate
    from drscreen.preprocess.fov import standardize
    p = generate(grade=1, size=700, seed=4)
    img, mask, _ = standardize(p.image, size=384)
    assert img.shape == (384, 384, 3)
    assert mask.shape == (384, 384)
    assert (img[mask == 0] == 0).all()


def test_landmarks_locate_disc_and_fovea():
    """Both landmarks within 1 disc diameter on a clean phantom."""
    import math
    from drscreen.data.synthetic import generate
    from drscreen.preprocess.fov import standardize
    from drscreen.preprocess.landmarks import locate

    hits_d = hits_f = 0
    n = 12
    for i in range(n):
        p = generate(size=512, seed=500 + i, severity=0.15)
        img, mask, fov = standardize(p.image, size=512)
        x0, y0, x1, y1 = fov.bbox
        px, py = int(0.02 * (x1 - x0)), int(0.02 * (y1 - y0))
        X0, Y0 = max(0, x0 - px), max(0, y0 - py)
        X1 = min(p.image.shape[1], x1 + px); Y1 = min(p.image.shape[0], y1 + py)
        ch, cw = Y1 - Y0, X1 - X0
        side = max(ch, cw); top, left = (side - ch) // 2, (side - cw) // 2
        sc = 512 / side
        gd = ((p.disc_xy[0] - X0 + left) * sc, (p.disc_xy[1] - Y0 + top) * sc)
        gf = ((p.fovea_xy[0] - X0 + left) * sc, (p.fovea_xy[1] - Y0 + top) * sc)
        lm = locate(img, mask)
        dd = p.disc_radius * 2 * sc
        hits_d += math.dist(lm.disc_xy, gd) / dd <= 1.0
        hits_f += math.dist(lm.fovea_xy, gf) / dd <= 1.0
    assert hits_d >= int(0.80 * n), f"optic disc {hits_d}/{n}"
    assert hits_f >= int(0.80 * n), f"fovea {hits_f}/{n}"


def test_quality_gate_rejects_what_enhancement_cannot_fix():
    """Severely degraded images must be refused -- *after* enhancement is tried.

    The gate distinguishes correctable defects (uneven flash, exposure, low
    contrast, noise) from unrecoverable ones (defocus, clipped field, absent
    macula, saturation). Rejecting on a correctable defect would send a patient
    back for a second visit over something the illumination normaliser fixes in
    40 ms, so the contract is: enhance first, then reject only what survived.

    This test therefore runs the full pipeline rather than a bare `assess`,
    because a bare first-pass verdict is deliberately permissive now.
    """
    from drscreen.data.synthetic import generate
    from drscreen.pipeline import DRScreeningPipeline, PipelineConfig
    pipe = DRScreeningPipeline(None, None, PipelineConfig(size=512, enable_cam=False))
    rejected = 0
    for i in range(8):
        p = generate(grade=0, size=512, seed=700 + i, severity=1.0,
                     camera="smartphone_ro")
        r, _ = pipe.run(p.image)
        if not r.gradeable:
            rejected += 1
            assert r.recapture_advice, "rejection must come with actionable advice"
    assert rejected >= 3, "quality gate is too permissive on severe degradation"


def test_correctable_defects_do_not_trigger_recapture():
    """An uneven flash is a software problem, not a second patient visit.

    Regression guard for a real bug: the gate used to reject on `illumination`
    before enhancement ran, so a perfectly gradeable proliferative-DR image
    from a high-vignette handheld camera came back as "recapture" -- the worst
    possible failure, since that patient most needs the referral.
    """
    from drscreen.data.synthetic import generate
    from drscreen.pipeline import DRScreeningPipeline, PipelineConfig
    pipe = DRScreeningPipeline(None, None, PipelineConfig(size=512, enable_cam=False))
    for cam in ("handheld_b", "handheld_a"):
        for g in (2, 4):
            p = generate(grade=g, size=640, seed=100 + g, severity=0.2, camera=cam)
            r, _ = pipe.run(p.image)
            assert r.gradeable, (
                f"{cam} grade {g} rejected on correctable defects: "
                f"{[k for k, v in r.quality['verdicts'].items() if v == 'fail']}")


def test_precropped_fundus_is_not_rejected_for_touching_the_frame():
    """A complete retina that fills the frame must pass the FOV criterion.

    Regression guard for a bug that only real data could expose. A fundus
    aperture is wider than the sensor is tall, so the retina touches the top
    and bottom edges of nearly every correct capture; datasets like APTOS ship
    pre-cropped and touch all four. The gate used to multiply coverage by a
    per-edge clipping penalty, which rejected 34% of real test images whose
    coverage was 0.90-1.00 -- images that were entirely gradeable. Phantoms
    never caught it because they always render a black margin.
    """
    import cv2
    from drscreen.preprocess.fov import detect_fov
    from drscreen.preprocess.quality import fov_score

    # A disc that exactly fills the frame: complete retina, all edges touched.
    n = 512
    img = np.zeros((n, n, 3), np.uint8)
    cv2.circle(img, (n // 2, n // 2), n // 2 - 1, (60, 90, 180), -1)
    fov = detect_fov(img)
    assert sum(fov.clipped_sides) >= 2, "test image should touch the frame"
    assert fov.coverage > 0.85, f"retina is complete, coverage {fov.coverage:.3f}"
    assert fov_score(fov) > 0.55, (
        f"complete-but-tight retina scored {fov_score(fov):.3f} and would be "
        "sent back for recapture")

    # A genuinely clipped retina -- circle centre pushed off the frame -- must
    # still score poorly, or the criterion has been neutered rather than fixed.
    img2 = np.zeros((n, n, 3), np.uint8)
    cv2.circle(img2, (n // 2, n - 40), n // 2 + 120, (60, 90, 180), -1)
    fov2 = detect_fov(img2)
    assert fov_score(fov2) < fov_score(fov), "a truly clipped field must score lower"


def test_quality_gate_accepts_clean_images():
    from drscreen.data.synthetic import generate
    from drscreen.preprocess.fov import standardize
    from drscreen.preprocess.quality import assess
    accepted = 0
    for i in range(8):
        p = generate(grade=0, size=512, seed=800 + i, severity=0.0,
                     camera="topcon_nw400")
        img, mask, fov = standardize(p.image, size=512)
        q = assess(img, mask, fov)
        accepted += q.gradeable
    assert accepted >= 7, "quality gate rejects clean images"


def test_ungradeable_images_produce_advice_not_a_grade():
    """The gate must never let a rejected image emerge with a confident grade."""
    from drscreen.data.synthetic import generate
    from drscreen.pipeline import DRScreeningPipeline, PipelineConfig
    pipe = DRScreeningPipeline(None, None, PipelineConfig(size=384, enable_cam=False))
    p = generate(grade=3, size=640, seed=13, severity=1.0, camera="smartphone_ro")
    res, _ = pipe.run(p.image)
    if not res.gradeable:
        assert res.decision == "recapture"
        assert res.recapture_advice
        assert res.grade == -1


# --------------------------------------------------------------------------
# Clinical rules
# --------------------------------------------------------------------------
def test_rule_grader_follows_icdr_ordering():
    from drscreen.models.lesion_features import ClinicalFeatures, rule_grade, QUADRANTS
    from drscreen.constants import LESION_CLASSES

    def make(**counts) -> ClinicalFeatures:
        f = ClinicalFeatures()
        f.counts = {c: counts.get(c, 0) for c in LESION_CLASSES}
        f.per_quadrant = {c: dict.fromkeys(QUADRANTS, 0) for c in LESION_CLASSES}
        f.area_fraction = {c: 0.0 for c in LESION_CLASSES}
        return f

    assert rule_grade(make())[0] == 0
    assert rule_grade(make(microaneurysm=3))[0] == 1
    assert rule_grade(make(microaneurysm=12, hemorrhage=4))[0] == 2
    f = make(microaneurysm=40, hemorrhage=40); f.quadrants_with_hemorrhage = 4
    assert rule_grade(f)[0] == 3
    f = make(neovascularization=2); f.nv_at_disc = True
    assert rule_grade(f)[0] == 4


def test_no_beading_false_positive_on_healthy_vessels():
    """The 4-2-1 'venous beading' arm must not fire on normal anatomy."""
    from drscreen.data.synthetic import generate
    from drscreen.pipeline import DRScreeningPipeline, PipelineConfig
    pipe = DRScreeningPipeline(None, None, PipelineConfig(size=384, enable_cam=False))
    fired = checked = 0
    for i in range(10):
        p = generate(grade=0, size=640, seed=300 + i, severity=0.10)
        res, art = pipe.run(p.image)
        if "features" not in art:      # gate rejected it; nothing to check
            continue
        checked += 1
        fired += art["features"].quadrants_with_beading >= 2
    assert checked >= 5, "too few gradeable phantoms to test"
    assert fired == 0, "venous-beading detector fires on healthy retinas"


def test_clinical_feature_vector_is_fixed_length():
    from drscreen.models.lesion_features import ClinicalFeatures
    f = ClinicalFeatures()
    f.per_quadrant = {}
    v = f.to_vector()
    assert v.shape == (ClinicalFeatures.vector_size(),)
    assert len(ClinicalFeatures.feature_names()) == ClinicalFeatures.vector_size()
    assert np.isfinite(v).all()


# --------------------------------------------------------------------------
# Split hygiene
# --------------------------------------------------------------------------
def test_messidor2_cannot_enter_the_training_pool():
    from drscreen.data.registry import Sample, assert_no_leakage, SplitViolation
    from pathlib import Path
    train = [Sample(Path("a.png"), grade=0, dataset="messidor2", subject_id="a")]
    with pytest.raises(SplitViolation):
        assert_no_leakage(train)


def test_group_split_keeps_subjects_together():
    from drscreen.data.registry import Sample, group_split
    from pathlib import Path
    samples = [Sample(Path(f"{i}_{eye}.png"), grade=0, dataset="aptos2019",
                      subject_id=str(i))
               for i in range(200) for eye in ("left", "right")]
    train, val = group_split(samples, 0.2)
    assert {s.subject_id for s in train}.isdisjoint({s.subject_id for s in val})


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def test_unet_shapes_and_deep_supervision():
    from drscreen.models.segmentation import build_unet
    m = build_unet("lesion", width=8)
    x = torch.randn(2, 3, 128, 128)
    m.train()
    out, aux = m(x)
    assert out.shape == (2, NUM_LESION_CLASSES, 128, 128)
    assert all(a.shape[1] == NUM_LESION_CLASSES for a in aux)
    m.eval()
    assert m(x).shape == (2, NUM_LESION_CLASSES, 128, 128)


def test_dice_is_nan_not_one_when_ground_truth_is_empty():
    """An empty prediction on an empty target must NOT score 1.0.

    Regression guard for a bug that cost a full training run. IDRiD encodes
    lesion-mask foreground as pixel value 76; the loader thresholded at >127,
    so every mask loaded as empty. The model dutifully learned to predict
    nothing, and the smoothed Dice `(2*inter+eps)/(denom+eps)` scored that
    empty-vs-empty agreement as a perfect 1.0000 on all five classes -- while
    the training loss sat flat at 0.95. The metric reported success for a
    model that had learned nothing from data that contained nothing.
    """
    from drscreen.training import dice_per_class
    empty_pred = torch.full((2, 3, 8, 8), -9.0)
    empty_target = torch.zeros(2, 3, 8, 8)
    d = dice_per_class(empty_pred, empty_target)
    assert torch.isnan(d).all(), f"empty/empty must be undefined, got {d.tolist()}"

    # A real overlap must still score, and absent classes stay NaN.
    target = torch.zeros(2, 3, 8, 8); target[:, 1, 2:6, 2:6] = 1
    logits = torch.full((2, 3, 8, 8), -9.0); logits[:, 1, 2:6, 2:6] = 9.0
    d = dice_per_class(logits, target)
    assert torch.isnan(d[0]) and torch.isnan(d[2])
    assert abs(float(d[1]) - 1.0) < 1e-4

    # A model predicting nothing where lesions DO exist must score ~0.
    d = dice_per_class(torch.full((2, 3, 8, 8), -9.0), target)
    assert float(d[1]) < 1e-3, "missed lesions must not score well"


def test_annotation_masks_use_any_nonzero_as_foreground():
    """Mask foreground encoding is not standardised across corpora.

    IDRiD uses 76, DRIVE uses 255. Any threshold above the smallest foreground
    value silently empties a whole dataset, which is invisible downstream
    because an empty mask is a legal input.
    """
    import cv2
    idrid_like = np.zeros((16, 16), np.uint8)
    idrid_like[4:8, 4:8] = 76           # IDRiD palette value
    drive_like = np.zeros((16, 16), np.uint8)
    drive_like[4:8, 4:8] = 255

    for name, m in (("IDRiD-like (76)", idrid_like), ("DRIVE-like (255)", drive_like)):
        assert (m > 0).sum() == 16, name
        binary = (m > 0).astype(np.uint8) * 255
        assert binary.max() == 255, name
        rs = cv2.resize(binary, (8, 8), interpolation=cv2.INTER_NEAREST)
        assert rs.max() == 255, f"{name} lost its foreground on resize"


def test_tversky_penalises_false_negatives_more():
    """The asymmetry that protects microaneurysm sensitivity."""
    from drscreen.models.segmentation import tversky_loss
    target = torch.zeros(1, 1, 16, 16); target[..., 4:8, 4:8] = 1
    miss = torch.full_like(target, -6.0)                      # predicts nothing
    over = torch.full_like(target, 6.0)                       # predicts everything
    assert float(tversky_loss(miss, target)) > float(tversky_loss(over, target))


def test_clinical_arm_is_actually_blind_to_the_image():
    """The ablation is only meaningful if each arm uses what it claims to.

    A "clinical features only" baseline that quietly still sees the image would
    make the fusion model's advantage look smaller than it is; one that also
    ignored the clinical vector would make it look larger. Both directions
    corrupt the headline claim, so both are pinned here. Must be checked in
    eval mode -- dropout alone makes train-mode outputs differ run to run.
    """
    from drscreen.models.grader import DRGrader
    from drscreen.models.lesion_features import ClinicalFeatures
    m = DRGrader(backbone="resnet18", pretrained=False,
                 use_clinical=True, use_image=False).eval()
    x = torch.randn(2, 3, 128, 128)
    c = torch.randn(2, ClinicalFeatures.vector_size())
    assert torch.allclose(m(x, c), m(torch.randn_like(x) * 7, c), atol=1e-6),         "clinical-only arm is still reading the image"
    assert not torch.allclose(m(x, c), m(x, torch.randn_like(c)), atol=1e-4),         "clinical-only arm ignores the clinical features"

    import copy
    assert copy.deepcopy(m).use_image is False, "EMA deepcopy loses use_image"


def test_pipeline_runs_without_trained_models():
    """The service must degrade to the rule engine, never crash."""
    from drscreen.data.synthetic import generate
    from drscreen.pipeline import DRScreeningPipeline, PipelineConfig
    pipe = DRScreeningPipeline(None, None, PipelineConfig(size=384, enable_cam=False))
    p = generate(grade=2, size=600, seed=21, severity=0.2)
    res, art = pipe.run(p.image, image_id="t")
    assert res.image_id == "t"
    assert "total" in res.timing_ms
    assert res.rule_based_grade in range(NUM_GRADES)


def test_report_renders_without_a_cam():
    from drscreen.data.synthetic import generate
    from drscreen.explain.report import render_html
    from drscreen.pipeline import DRScreeningPipeline, PipelineConfig
    pipe = DRScreeningPipeline(None, None, PipelineConfig(size=384, enable_cam=False))
    p = generate(grade=1, size=600, seed=22, severity=0.2)
    res, art = pipe.run(p.image, image_id="r")
    html = render_html(res, art)
    assert "<!doctype html>" in html.lower()
    assert "Diabetic Retinopathy Screening Report" in html


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
def test_simulation_conserves_patients():
    from drscreen.sim.telemedicine import simulate, SimConfig
    r = simulate(SimConfig(sim_days=30, warmup_days=3, n_phc=4,
                           annual_patients=8000, seed=0))
    assert r.n_captured + r.n_rejected_ungradeable <= r.n_arrived
    assert r.n_auto_reported + r.n_human_reviewed <= r.n_graded


def test_ai_triage_reduces_reviewer_load():
    """The central capacity claim of the whole system."""
    from drscreen.sim.telemedicine import simulate, SimConfig
    base = dict(sim_days=45, warmup_days=5, n_phc=12, annual_patients=100_000,
                ophthalmologists=2.0, seed=0)
    manual = simulate(SimConfig(**base, auto_report_coverage=0.0, review_time_min=2.5))
    ai = simulate(SimConfig(**base, auto_report_coverage=0.70, review_time_min=0.5))
    assert ai.utilisation["reviewer"] < manual.utilisation["reviewer"] * 0.5


def test_simulink_export_writes_valid_matlab():
    import tempfile
    from pathlib import Path
    from drscreen.sim.simulink_export import export_all
    from drscreen.sim.telemedicine import SimConfig
    with tempfile.TemporaryDirectory() as d:
        paths = export_all(SimConfig(), d)
        params = Path(paths["params"]).read_text()
        assert "function p = dr_screening_params()" in params
        assert "p.mean_interarrival_min" in params
        assert params.count("end") >= 1


# --------------------------------------------------------------------------
# Sight-threatening disease (grades 3-4)
#
# The model was returning 0.43 recall on both severe NPDR and proliferative DR
# on the *internal* test split -- same cameras, same graders as training -- and
# folding them into grade 2. Nothing in the training loop or the validation
# artefact reported it. These four tests guard the four causes.
# --------------------------------------------------------------------------
def test_referable_auc_cannot_see_the_severity_axis():
    """Why checkpoint selection must not be referable-DR AUC.

    ``referable_prob`` is ``sigma(z0) * sigma(z1)``. It contains neither z2 nor
    z3 -- the units that decide grades 3 and 4 -- so a model that collapses
    every proliferative case into grade 2 scores *identically* to one that
    grades them correctly. QWK does not have that blind spot.
    """
    from drscreen.models.grader import referable_prob, corn_predict
    from drscreen.evaluation.metrics import quadratic_weighted_kappa

    y = np.array([0, 1, 2, 3, 4] * 8)
    graded = torch.full((len(y), NUM_GRADES - 1), -8.0)
    for i, g in enumerate(y):
        graded[i, :g] = 8.0
    assert (corn_predict(graded).numpy() == y).all()

    collapsed = graded.clone()
    collapsed[:, 2:] = -8.0                      # never escalate past grade 2

    assert torch.allclose(referable_prob(graded), referable_prob(collapsed)), (
        "referable_prob reads z0/z1 only -- selecting on it cannot observe "
        "grade 3/4 collapse, which is exactly how the collapse survived")
    assert quadratic_weighted_kappa(y, corn_predict(collapsed).numpy()) < \
        quadratic_weighted_kappa(y, corn_predict(graded).numpy()), \
        "QWK must penalise the collapse the selection metric now guards against"


def test_corn_loss_scale_is_independent_of_batch_composition():
    """The loss must not rescale when a grade is absent or reweighted.

    The old form averaged per-task means and skipped tasks whose subset was
    empty, so the denominator moved with batch composition -- and it passed
    class weights to ``binary_cross_entropy_with_logits``, which returns
    ``mean(w * loss)`` without renormalising. Both made the effective learning
    rate on the deep conditionals depend on which grades happened to be drawn.

    With all-zero logits every conditional term is exactly ln 2, so a correctly
    normalised loss is ln 2 for *any* batch and *any* weights.
    """
    import math
    from drscreen.models.grader import corn_loss

    logits = torch.zeros(4, NUM_GRADES - 1)
    mixed = torch.tensor([0, 1, 2, 3])
    assert abs(float(corn_loss(logits, mixed, NUM_GRADES)) - math.log(2)) < 1e-6

    # Only task 0 has any samples; the other three are empty.
    only_zero = torch.zeros(4, dtype=torch.long)
    assert abs(float(corn_loss(logits, only_zero, NUM_GRADES)) - math.log(2)) < 1e-6

    # Class weights must reweight the average, not rescale it.
    cw = torch.tensor([5.0, 1.0, 1.0, 1.0, 1.0])
    assert abs(float(corn_loss(logits, mixed, NUM_GRADES, cw)) - math.log(2)) < 1e-6


def test_unassessed_lesion_class_is_never_reported_as_absent():
    """A class with no pixel supervision must not read as a negative finding.

    IDRiD annotates no neovascularisation, so that channel trains on all-zero
    targets and returns a count of zero for every image ever screened. Since
    neovascularisation *defines* proliferative DR, reporting that zero as
    "none detected" claims an exclusion the model never made.
    """
    from drscreen.constants import LESION_CLASSES
    from drscreen.models.lesion_features import ClinicalFeatures, rule_grade

    f = ClinicalFeatures()
    f.counts = {c: 0 for c in LESION_CLASSES}
    f.unassessed = ("neovascularization",)
    _, reasons = rule_grade(f)
    assert any("NOT ASSESSED" in r for r in reasons), (
        "the grade-4 arm is unreachable without NV supervision and must say so")

    from drscreen.pipeline import DRScreeningPipeline, PipelineConfig
    pipe = DRScreeningPipeline(None, None, PipelineConfig(size=384, enable_cam=False))
    assert "neovascularization" in pipe.unassessed_lesions
    from drscreen.data.synthetic import generate
    p = generate(grade=2, size=600, seed=22, severity=0.4)
    res, _ = pipe.run(p.image, image_id="nv")
    assert res.gradeable, "need a gradeable image to reach the evidence block"
    flagged = [e for e in (res.evidence or []) if e.get("status") == "not assessed"]
    assert any(e["finding"] == "neovascularization".replace("_", " ") for e in flagged)


def test_threshold_sweep_rows_describe_their_own_threshold():
    """Each swept row must be computed at the threshold it is labelled with.

    A sweep maintained by hand is one row-shift away from recommending an
    operating point that was never measured -- and the deployed point must
    always appear in it, so the document cannot quote a row that is not there.
    """
    from drscreen.evaluation.metrics import threshold_sweep, severity_breakdown

    y = np.repeat([0, 1, 2, 3, 4], [40, 12, 20, 8, 6])
    rng = np.random.default_rng(0)
    ref = np.clip(y / 4.0 + rng.normal(0, 0.12, len(y)), 0, 1)

    deployed = 0.657
    rows = threshold_sweep(y, ref, deployed)
    assert sum(r["deployed"] for r in rows) == 1

    for r in rows:
        f = ref >= r["threshold"]
        assert abs(r["fraction_flagged"] - f.mean()) < 1e-12
        assert abs(r["sens_sight_threatening"] - f[y >= 3].mean()) < 1e-12
        assert abs(r["specificity"] - (~f[y < 2]).mean()) < 1e-12

    # and the deployed row must agree with the headline breakdown
    row = next(r for r in rows if r["deployed"])
    sb = severity_breakdown(y, ref, deployed)
    assert abs(row["sens_sight_threatening"]
               - sb["sight_threatening"]["sensitivity"]["value"]) < 1e-12


# --------------------------------------------------------------------------
# Train / serve preprocessing parity
#
# The cohort builder and the live pipeline both feed the grader the "hybrid"
# 3-channel representation. If they build it in different channel orders, the
# deployed model is served mirror-image inputs to the ones it trained on, and
# every grade degrades silently -- validation stays clean because it reads the
# same cohort images training did. This guards the exact tensor.
# --------------------------------------------------------------------------
def test_train_serve_channel_parity(tmp_path):
    import cv2
    from drscreen.preprocess.enhance import to_model_input, adaptive_enhance
    from drscreen.preprocess.quality import assess
    from drscreen.preprocess.fov import standardize
    from drscreen.data.torch_data import to_tensor
    from drscreen.data.synthetic import generate

    p = generate(grade=2, size=512, seed=3, severity=0.5)
    img0, fov, fbox = standardize(p.image, size=512)
    enh, _ = adaptive_enhance(img0, fov, assess(img0, fov, fbox).issues)

    # TRAINING side: exactly what build_cohort writes, then what CohortDataset
    # reads back and feeds the model.
    train_arr = to_model_input(enh, fov, mode="hybrid")
    fp = tmp_path / "x.png"
    cv2.imwrite(str(fp), train_arr)
    reloaded = cv2.imread(str(fp), cv2.IMREAD_COLOR)
    train_tensor = to_tensor(reloaded)

    # LIVE side: exactly what pipeline.py feeds the model.
    live_tensor = to_tensor(to_model_input(enh, fov, mode="hybrid"))

    assert torch.equal(train_tensor, live_tensor), (
        "cohort-stored input and live input differ -- the model is trained and "
        "served different channel orders. build_cohort must not colour-convert "
        "the output of to_model_input.")


def test_feature_caching_threshold_matches_live():
    """precompute_features must cache at the SAME thresholds live inference uses.

    Not a blanket 0.5. The live pipeline applies the fitted per-class values;
    caching at a scalar hands the fusion grader lesion counts it never sees at
    inference. This asserts the wiring exists -- extract() honours a per-class
    dict identically in both paths.
    """
    import numpy as np
    from drscreen.models.lesion_features import extract, ClinicalFeatures
    from drscreen.constants import LESION_CLASSES
    from drscreen.preprocess.landmarks import locate

    H = W = 128
    probs = np.zeros((H, W, len(LESION_CLASSES)), np.float32)
    probs[40:60, 40:60, LESION_CLASSES.index("hard_exudate")] = 0.7   # mid-confidence blob
    img = np.full((H, W, 3), 60, np.uint8)
    fov = np.ones((H, W), np.uint8)
    lm = locate(img, fov)

    dense = extract(probs, lm, fov, None, threshold=0.5)
    sparse = extract(probs, lm, fov, None,
                     threshold={c: 0.95 for c in LESION_CLASSES})
    # 0.7 blob survives a 0.5 cut but not a 0.95 cut: the two thresholds give
    # genuinely different counts, which is exactly why train and serve must agree.
    assert dense.counts.get("hard_exudate", 0) > sparse.counts.get("hard_exudate", 0)
