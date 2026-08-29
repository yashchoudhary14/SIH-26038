"""Interpretable image-quality assessment (IQA) gatekeeper.

Design decision that differs from the usual "train a MobileNet on synthetic
blur" recipe: the gate is **physics-first and interpretable**, with an
optional learned head layered on top.

Reasons:

1. The problem statement requires *recapture feedback*, not just a reject
   flag.  A CNN logit cannot tell a rural technician "the macula is out of
   frame"; a per-criterion metric can.
2. The gate runs on an edge device before any heavy model, so it must be
   milliseconds and CPU-only.
3. A learned quality model trained on synthetic degradations is exactly the
   kind of unvalidated component that fails in the field, because real
   failures (media opacity, pupil miosis, lens flare, uncleaned optics) do
   not look like Gaussian blur.

Each criterion produces a score in [0, 1] plus a pass/borderline/fail verdict
against thresholds that are declared here and can be re-fitted on a labelled
set with :func:`fit_thresholds`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np

from ..constants import RECAPTURE_ADVICE
from .fov import FOVInfo, detect_fov
from .landmarks import Landmarks, locate, macula_score as landmark_macula_score


# --------------------------------------------------------------------------
# Default operating thresholds
# --------------------------------------------------------------------------
#: (fail_below, borderline_below) for each criterion; scores are in [0, 1] and
#: higher is always better.
DEFAULT_THRESHOLDS: dict[str, tuple[float, float]] = {
    "focus":          (0.25, 0.50),
    "illumination":   (0.30, 0.55),
    "contrast":       (0.25, 0.50),
    "fov":            (0.55, 0.80),
    "macula":         (0.20, 0.45),
    "artifact":       (0.30, 0.60),
    "under_exposure": (0.25, 0.50),
    "over_exposure":  (0.25, 0.50),
    "noise":          (0.20, 0.45),
}

#: Defects software can undo. A failure here means "enhance and re-check",
#: never "send the patient away".
#:
#: The distinction is clinical, not cosmetic. Asking a patient in a rural PHC
#: to return because the flash was uneven -- when
#: :func:`drscreen.preprocess.enhance.illumination_normalize` exists precisely
#: to remove that gradient -- wastes a visit that may cost the patient a day's
#: wages and a bus fare, and inflates the recapture rate that the capacity
#: model is sized against.
CORRECTABLE = {"illumination", "under_exposure", "over_exposure", "contrast", "noise"}

#: Defects no amount of processing can undo, because the information is not in
#: the file: out-of-focus detail is gone, retina outside the aperture was never
#: captured, a saturated region carries no signal, and an absent macula cannot
#: be invented. These are the only grounds for demanding a recapture.
NON_CORRECTABLE = {"focus", "fov", "macula", "artifact"}


@dataclass
class QualityReport:
    scores: dict[str, float] = field(default_factory=dict)
    verdicts: dict[str, str] = field(default_factory=dict)   # pass|borderline|fail
    overall: str = "good"                                    # good|borderline|ungradeable
    gradeable: bool = True
    needs_enhancement: bool = False
    issues: list[str] = field(default_factory=list)
    advice: list[str] = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Individual criteria
# --------------------------------------------------------------------------
def _retina_pixels(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    sel = gray[mask > 0]
    return sel if sel.size else gray.ravel()


def focus_score(image: np.ndarray, mask: np.ndarray) -> float:
    """Scale-normalised focus measure.

    Plain variance-of-Laplacian is not comparable across resolutions or across
    images with different pathology load (a retina full of exudates has huge
    Laplacian energy while being out of focus).  We therefore use the ratio of
    high-frequency to mid-frequency energy in the green channel, which is
    largely content-independent, and calibrate it with a soft curve.
    """
    green = image[..., 1] if image.ndim == 3 else image
    green = cv2.bitwise_and(green, green, mask=(mask > 0).astype(np.uint8) * 255)
    g = green.astype(np.float32)

    # Band-pass energies via difference of Gaussians at resolution-relative scales.
    s = max(image.shape[:2]) / 512.0
    hi = g - cv2.GaussianBlur(g, (0, 0), 1.0 * s)
    mid = cv2.GaussianBlur(g, (0, 0), 1.0 * s) - cv2.GaussianBlur(g, (0, 0), 4.0 * s)

    m = mask > 0
    if m.sum() < 100:
        return 0.0
    e_hi = float(np.mean(np.abs(hi[m])))
    e_mid = float(np.mean(np.abs(mid[m])))
    ratio = e_hi / max(e_mid, 1e-3)

    # ratio ~0.25 = badly blurred, ~0.9 = crisp. Map through a logistic.
    return float(1.0 / (1.0 + np.exp(-(ratio - 0.45) / 0.12)))


def illumination_score(image: np.ndarray, mask: np.ndarray,
                       blocks: int = 8) -> float:
    """Uniformity of the illumination field.

    Block-wise median luminance inside the FOV; the score is 1 minus the
    robust coefficient of variation, so a strong centre-to-periphery gradient
    or a one-sided flash both score low.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape[:2]
    bh, bw = h // blocks, w // blocks
    meds = []
    for i in range(blocks):
        for j in range(blocks):
            sub_m = mask[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]
            if (sub_m > 0).mean() < 0.6:          # mostly outside the aperture
                continue
            sub = gray[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]
            meds.append(float(np.median(sub[sub_m > 0])))
    if len(meds) < 4:
        return 0.0
    meds = np.asarray(meds)
    center = float(np.median(meds))
    if center < 1:
        return 0.0
    # Robust spread (IQR-based) normalised by the level.
    spread = float(np.percentile(meds, 90) - np.percentile(meds, 10))
    cv_robust = spread / max(center, 1.0)
    return float(np.clip(1.0 - cv_robust / 0.9, 0.0, 1.0))


def contrast_score(image: np.ndarray, mask: np.ndarray) -> float:
    """Vessel-visibility proxy.

    A retina you cannot grade is usually one where the vasculature has
    vanished into the background (media opacity, cataract, miosis).  We
    measure the response of a multi-scale vesselness-like filter relative to
    background variation, which correlates with a grader's ability to see
    microaneurysms far better than global histogram spread does.
    """
    green = image[..., 1] if image.ndim == 3 else image
    m = mask > 0
    if m.sum() < 100:
        return 0.0
    g = green.astype(np.float32)
    s = max(image.shape[:2]) / 512.0

    # Dark-structure (vessel) response: background estimate minus image.
    resp = np.zeros_like(g)
    for sigma in (1.0, 2.0, 3.5):
        bg = cv2.morphologyEx(
            g, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (int(2 * sigma * 3 * s) | 1,) * 2))
        resp = np.maximum(resp, bg - g)

    vessel_signal = float(np.percentile(resp[m], 99))
    noise = float(np.std(resp[m]))
    snr = vessel_signal / max(noise, 1e-3)
    return float(np.clip((snr - 1.5) / 4.0, 0.0, 1.0))


def fov_score(fov: FOVInfo) -> float:
    """Fraction of the 45-degree field actually captured.

    ``coverage`` is the fraction of the fitted aperture circle that landed on
    the sensor, which is precisely the quantity of interest.

    An earlier version multiplied it by a per-edge clipping penalty. That was
    wrong twice over: it double-counts what coverage already measures, and it
    misreads normal fundus geometry as a defect. A fundus aperture is wider
    than the sensor is tall, so the retina touches the top and bottom edges of
    almost every correctly captured image -- and datasets like APTOS ship
    pre-cropped, touching all four. On real data that penalty rejected 34% of
    images whose coverage was 0.90-1.00, i.e. images that were completely fine.
    The phantoms never caught it because they always render black margin.

    Edge contact is retained only as a weak corroborating signal when coverage
    is *already* poor, which is the case where the retina genuinely does run
    off the sensor.
    """
    score = float(fov.coverage)
    if score < 0.80:
        n_clipped = sum(fov.clipped_sides)
        score *= (1.0 - 0.05 * n_clipped)
    if fov.fill_ratio < 0.15:
        score *= 0.4          # barely any retina in frame at all
    return float(np.clip(score, 0.0, 1.0))


def exposure_scores(image: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Return (under_exposure_score, over_exposure_score); higher = better."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    sel = _retina_pixels(gray, mask).astype(np.float32)
    if sel.size < 100:
        return 0.0, 0.0
    med = float(np.median(sel))
    frac_dark = float((sel < 25).mean())
    frac_blown = float((sel > 245).mean())

    under = np.clip((med - 25.0) / 55.0, 0.0, 1.0) * np.clip(1.0 - frac_dark / 0.45, 0.0, 1.0)
    over = np.clip((215.0 - med) / 55.0, 0.0, 1.0) * np.clip(1.0 - frac_blown / 0.12, 0.0, 1.0)
    return float(under), float(over)


def artifact_score(image: np.ndarray, mask: np.ndarray) -> float:
    """Detect large specular reflections, dust arcs and lens flare.

    Looks for saturated blobs that are too large and too round to be exudates
    and that sit away from the optic disc.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    m = (mask > 0).astype(np.uint8)
    area = max(int(m.sum()), 1)
    blown = ((gray > 250) & (m > 0)).astype(np.uint8) * 255
    if blown.sum() == 0:
        return 1.0
    blown = cv2.morphologyEx(
        blown, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(blown, connectivity=8)
    bad = 0
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a > 0.004 * area:          # bigger than a plausible exudate cluster
            bad += a
    return float(np.clip(1.0 - (bad / area) / 0.05, 0.0, 1.0))


def noise_score(image: np.ndarray, mask: np.ndarray) -> float:
    """Estimate sensor noise in flat retinal regions (Immerkaer estimator)."""
    green = image[..., 1] if image.ndim == 3 else image
    g = green.astype(np.float32)
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    lap = cv2.filter2D(g, -1, kernel)
    m = mask > 0
    if m.sum() < 100:
        return 0.0
    sigma = float(np.sqrt(np.pi / 2) / (6 * max(m.sum(), 1)) * np.abs(lap[m]).sum())
    # sigma < 1.5 is clean, > 6 is unusable
    return float(np.clip(1.0 - (sigma - 1.5) / 5.0, 0.0, 1.0))


def macula_visibility_score(image: np.ndarray, mask: np.ndarray,
                            fov: FOVInfo,
                            lm: "Landmarks | None" = None) -> float:
    """Is the macula inside the frame with enough surrounding retina to grade?

    Delegates to the analytic landmark detector so the quality gate and the
    clinical coordinate frame agree on where the fovea is -- a screening
    report that says "adequate macular view" while the grader used a
    different fovea would be indefensible.
    """
    if mask.sum() < 100:
        return 0.0
    if lm is None:
        lm = locate(image, mask)
    return landmark_macula_score(lm, image.shape[:2])


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def assess(image: np.ndarray, mask: np.ndarray | None = None,
           fov: FOVInfo | None = None,
           thresholds: dict[str, tuple[float, float]] | None = None,
           landmarks: Landmarks | None = None) -> QualityReport:
    """Run every criterion and produce a gradeability verdict with advice."""
    th = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        th.update(thresholds)

    if fov is None:
        fov = detect_fov(image)
    if mask is None:
        mask = fov.mask
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

    under, over = exposure_scores(image, mask)
    if landmarks is None:
        landmarks = locate(image, mask)
    scores = {
        "focus":          focus_score(image, mask),
        "illumination":   illumination_score(image, mask),
        "contrast":       contrast_score(image, mask),
        "fov":            fov_score(fov),
        "macula":         macula_visibility_score(image, mask, fov, landmarks),
        "artifact":       artifact_score(image, mask),
        "under_exposure": under,
        "over_exposure":  over,
        "noise":          noise_score(image, mask),
    }

    verdicts, issues, advice = {}, [], []
    hard_fail, soft_fail = [], []
    for name, val in scores.items():
        fail_below, borderline_below = th[name]
        if val < fail_below:
            verdicts[name] = "fail"
            issues.append(name)
            if name in NON_CORRECTABLE:
                hard_fail.append(name)
                advice.append(RECAPTURE_ADVICE.get(name, f"Recapture: {name} inadequate."))
            else:
                soft_fail.append(name)
        elif val < borderline_below:
            verdicts[name] = "borderline"
            issues.append(name)
        else:
            verdicts[name] = "pass"

    n_border = sum(v == "borderline" for v in verdicts.values())

    # A correctable defect is a reason to enhance, not to reject. The caller
    # runs `adaptive_enhance` and then re-assesses (see
    # `DRScreeningPipeline.run`); only if a correctable criterion still fails
    # *after* enhancement does the image become ungradeable, and only then is
    # recapture advice issued for it.
    if hard_fail:
        overall, gradeable = "ungradeable", False
    elif soft_fail:
        overall, gradeable = "borderline", True
    elif n_border > 0:
        overall, gradeable = "borderline", True
    else:
        overall, gradeable = "good", True

    # A scalar quality confidence: the soft-min of the criteria, so one bad
    # axis dominates (quality is a conjunction, not an average).
    vals = np.asarray(list(scores.values()), np.float64)
    beta = 8.0
    conf = float(-np.log(np.exp(-beta * vals).sum() / vals.size) / beta)

    return QualityReport(
        scores={k: round(float(v), 4) for k, v in scores.items()},
        verdicts=verdicts,
        overall=overall,
        gradeable=gradeable,
        needs_enhancement=bool(n_border > 0 or soft_fail or hard_fail),
        issues=issues,
        advice=advice,
        confidence=round(float(np.clip(conf, 0.0, 1.0)), 4),
    )


def fit_thresholds(scores_by_image: list[dict[str, float]],
                   labels: list[int],
                   target_specificity: float = 0.95,
                   out_path: str | Path | None = None
                   ) -> dict[str, tuple[float, float]]:
    """Re-fit fail/borderline cut-points from a labelled quality set.

    `labels`: 0 = ungradeable, 1 = borderline, 2 = good (see
    :data:`drscreen.constants.QUALITY_LABELS`).

    The fail cut-point for each criterion is set at the value that retains
    `target_specificity` of the good images (i.e. we accept rejecting a few
    gradeable images to avoid passing an ungradeable one -- the asymmetry a
    screening programme wants).  The borderline cut-point is the median of the
    borderline class.
    """
    labels_arr = np.asarray(labels)
    keys = sorted({k for d in scores_by_image for k in d})
    out: dict[str, tuple[float, float]] = {}
    for k in keys:
        vals = np.asarray([d.get(k, 0.0) for d in scores_by_image], np.float64)
        good = vals[labels_arr == 2]
        border = vals[labels_arr == 1]
        fail_cut = float(np.percentile(good, 100 * (1 - target_specificity))) if good.size else DEFAULT_THRESHOLDS.get(k, (0.25, 0.5))[0]
        border_cut = float(np.median(border)) if border.size else fail_cut * 2
        out[k] = (round(fail_cut, 4), round(max(border_cut, fail_cut + 1e-3), 4))
    if out_path:
        Path(out_path).write_text(json.dumps(out, indent=2))
    return out
