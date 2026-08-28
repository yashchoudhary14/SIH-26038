"""Adaptive enhancement for fundus images.

The problem statement asks for enhancement *of borderline images* -- not of
everything.  Unconditional enhancement is actively harmful: CLAHE on an
already well-exposed image amplifies sensor noise into structures that the
microaneurysm detector then reports, and Ben Graham normalisation on a clean
image destroys the absolute colour cues that separate hard exudates (yellow,
sharp) from cotton-wool spots (pale, fuzzy).

So every routine here is individually callable, and `adaptive_enhance`
applies only the corrections that the quality report says are required,
recording exactly what it did for the audit trail.
"""
from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


# --------------------------------------------------------------------------
# Individual operators
# --------------------------------------------------------------------------
def illumination_normalize(image: np.ndarray, mask: np.ndarray | None = None,
                           sigma_frac: float = 0.05) -> np.ndarray:
    """Flat-field correction: divide out the low-frequency illumination field.

    The retina is a concave surface lit by a single coaxial flash, producing a
    bright centre and a vignetted periphery.  Estimating that field with a
    wide Gaussian and dividing it out is multiplicative (physically correct
    for a reflectance model), unlike Ben Graham's subtractive variant which
    also removes true low-frequency pathology such as large blot hemorrhages.
    """
    h, w = image.shape[:2]
    sigma = max(3.0, sigma_frac * max(h, w))
    img = image.astype(np.float32)

    if mask is not None:
        m = (mask > 0).astype(np.float32)
        # Normalised convolution so the black surround does not drag the
        # background estimate down near the rim.
        num = cv2.GaussianBlur(img * m[..., None], (0, 0), sigma)
        den = cv2.GaussianBlur(m, (0, 0), sigma)[..., None]
        background = num / np.maximum(den, 1e-3)
    else:
        background = cv2.GaussianBlur(img, (0, 0), sigma)

    if mask is not None:
        target = np.array([float(np.median(img[..., c][mask > 0]))
                           for c in range(img.shape[2])], np.float32)
    else:
        target = np.array([float(np.median(img[..., c]))
                           for c in range(img.shape[2])], np.float32)

    out = img / np.maximum(background, 1.0) * target
    out = np.clip(out, 0, 255).astype(np.uint8)
    if mask is not None:
        out = cv2.bitwise_and(out, out, mask=(mask > 0).astype(np.uint8) * 255)
    return out


def ben_graham(image: np.ndarray, mask: np.ndarray | None = None,
               sigma_frac: float = 0.033, alpha: float = 4.0,
               beta: float = -4.0, gamma: float = 128.0) -> np.ndarray:
    """Ben Graham's subtractive local-contrast normalisation (Kaggle DR 2015).

    `sigma` is expressed as a fraction of image width so the operator is
    resolution-invariant -- the fixed `sigma=10` seen in most reimplementations
    silently changes meaning between a 512px and a 1024px input.
    """
    sigma = max(1.0, sigma_frac * image.shape[1])
    blur = cv2.GaussianBlur(image, (0, 0), sigma)
    out = cv2.addWeighted(image, alpha, blur, beta, gamma)
    if mask is not None:
        out = cv2.bitwise_and(out, out, mask=(mask > 0).astype(np.uint8) * 255)
    return out


def clahe_lab(image: np.ndarray, clip_limit: float = 2.0,
              tiles: int = 8) -> np.ndarray:
    """CLAHE on L* only, so hue/chroma (and therefore lesion colour) survive."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tiles, tiles))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def clahe_green(image: np.ndarray, clip_limit: float = 3.0,
                tiles: int = 8) -> np.ndarray:
    """CLAHE applied to the green channel, returned as single-channel uint8.

    Green carries the best vessel/lesion contrast: red saturates against the
    choroid and blue is dominated by scatter through the ocular media.  This
    is the input the vessel U-Net and the microaneurysm branch consume.
    """
    green = image[..., 1] if image.ndim == 3 else image
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tiles, tiles))
    return clahe.apply(green)


def denoise(image: np.ndarray, strength: float = 3.0) -> np.ndarray:
    """Edge-preserving denoise.

    Non-local means is used rather than a Gaussian because microaneurysms are
    3-8 px blobs; any isotropic smoothing wide enough to suppress read-noise
    also erases them.
    """
    return cv2.fastNlMeansDenoisingColored(
        image, None, h=float(strength), hColor=float(strength),
        templateWindowSize=7, searchWindowSize=21)


def gamma_correct(image: np.ndarray, gamma: float) -> np.ndarray:
    inv = 1.0 / max(gamma, 1e-3)
    lut = np.clip((np.arange(256) / 255.0) ** inv * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(image, lut)


def auto_exposure(image: np.ndarray, mask: np.ndarray | None = None,
                  target_median: float = 110.0) -> np.ndarray:
    """Push the retinal median luminance to a canonical value via gamma.

    Gain (linear scaling) would clip the optic disc; gamma preserves the
    highlight roll-off that distinguishes a bright disc from a hard exudate.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sel = gray[mask > 0] if mask is not None else gray.ravel()
    sel = sel[sel > 5]
    if sel.size < 100:
        return image
    med = float(np.median(sel))
    if med < 1:
        return image
    # solve (med/255) ** (1/g) == target/255
    g = np.log(max(med, 1.0) / 255.0) / np.log(max(target_median, 1.0) / 255.0)
    g = float(np.clip(g, 0.4, 2.5))
    return gamma_correct(image, g)


def grey_world(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Colour-constancy correction across cameras (shades-of-grey, p=6).

    Different fundus cameras have very different white balance; without this a
    classifier can learn "this hue implies this hospital implies this
    prevalence", which is exactly the shortcut that collapses on Messidor-2.
    """
    img = image.astype(np.float32)
    sel = img[mask > 0] if mask is not None else img.reshape(-1, 3)
    if sel.size == 0:
        return image
    p = 6.0
    norms = np.power(np.mean(np.power(np.maximum(sel, 1e-6), p), axis=0), 1.0 / p)
    scale = float(norms.mean()) / np.maximum(norms, 1e-6)
    out = np.clip(img * scale, 0, 255).astype(np.uint8)
    if mask is not None:
        out = cv2.bitwise_and(out, out, mask=(mask > 0).astype(np.uint8) * 255)
    return out


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def adaptive_enhance(image: np.ndarray, mask: np.ndarray | None = None,
                     issues: Iterable[str] = (),
                     always: Iterable[str] = ("grey_world",),
                     ) -> tuple[np.ndarray, list[str]]:
    """Apply only the corrections implied by `issues`.

    Parameters
    ----------
    issues
        Quality-metric names flagged by :mod:`drscreen.preprocess.quality`
        (e.g. ``{"illumination", "contrast"}``).
    always
        Operators applied to every image for cross-camera standardisation.

    Returns
    -------
    (enhanced_bgr, applied_operator_names)
    """
    issues = set(issues)
    always = set(always)
    applied: list[str] = []
    out = image

    if "grey_world" in always:
        out = grey_world(out, mask)
        applied.append("grey_world")

    if issues & {"illumination", "under_exposure", "over_exposure"}:
        out = illumination_normalize(out, mask)
        applied.append("illumination_normalize")

    if issues & {"under_exposure", "over_exposure"}:
        out = auto_exposure(out, mask)
        applied.append("auto_exposure")

    if issues & {"contrast", "focus"}:
        out = clahe_lab(out, clip_limit=2.5)
        applied.append("clahe_lab")

    if "noise" in issues:
        out = denoise(out, strength=3.0)
        applied.append("denoise")

    if mask is not None:
        out = cv2.bitwise_and(out, out, mask=(mask > 0).astype(np.uint8) * 255)
    return out, applied


def to_model_input(image: np.ndarray, mask: np.ndarray | None = None,
                   mode: str = "hybrid") -> np.ndarray:
    """Build the tensor-facing 3-channel representation.

    ``hybrid`` stacks complementary evidence instead of a plain RGB copy:

    ==========  ========================================================
    channel     content
    ==========  ========================================================
    0           CLAHE(green)      - lesion and vessel contrast
    1           Ben-Graham green  - local contrast, illumination-free
    2           L* of LAB         - absolute luminance, keeps exudate cue
    ==========  ========================================================

    This is a cheap, fully reproducible way to give the grader both the
    absolute photometry it needs for exudates and the high-pass detail it
    needs for microaneurysms, and it is measured against raw RGB in the
    ablation study (see :mod:`drscreen.evaluation.ablation`).
    """
    if mode == "rgb":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if mode == "green":
        g = clahe_green(image)
        return np.stack([g, g, g], axis=-1)

    g_clahe = clahe_green(image, clip_limit=3.0)
    bg = ben_graham(image, mask)[..., 1]
    lab_l = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[..., 0]
    out = np.stack([g_clahe, bg, lab_l], axis=-1)
    if mask is not None:
        out = cv2.bitwise_and(out, out, mask=(mask > 0).astype(np.uint8) * 255)
    return out
