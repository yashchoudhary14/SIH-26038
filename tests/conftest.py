"""Shared fixtures: real fundus photographs, and real optical degradations.

Every image in this suite is a photograph of an actual retina. The tests used to
run on generated phantoms, which is a bad way to test a screening pipeline for
the same reason it is a bad way to validate one: the phantom is drawn from the
pipeline's own assumptions, so it agrees with them, and a test that only ever
sees agreement cannot fail on a mistaken assumption. Two production defects lived
behind a green suite for exactly that reason -- the FOV criterion rejected a
third of genuinely gradeable real images because phantoms always render a black
margin, and the triage rule escalated every real photograph to urgent because
phantoms never exercised an unsupervised lesion channel.

The photographs come from ``drscreen.data.samples``: twelve APTOS-2019 and IDRiD
held-out test images committed to the repository, covering all five ICDR grades.

Degradation is applied *to those photographs* rather than simulated from scratch.
The distinction matters: blurring a real retina tests whether the focus criterion
can tell a defocused real image from a sharp one, which is the question. Drawing
a blurry picture tests whether the criterion can recognise the drawing.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from drscreen.data import samples

# Tests that need a photograph are pointless without one, and silently passing
# would be worse than either failing or skipping.
requires_samples = pytest.mark.skipif(
    not samples.available(),
    reason="committed sample photographs missing from outputs/verification_set/images")


@pytest.fixture(scope="session")
def real_images() -> list[tuple[np.ndarray, str, int | None]]:
    """All twelve committed photographs, as ``(bgr, name, reference_grade)``."""
    if not samples.available():
        pytest.skip("no sample photographs available")
    return samples.load_all()


@pytest.fixture(scope="session")
def real_image(real_images) -> np.ndarray:
    return real_images[0][0]


@pytest.fixture(scope="session")
def healthy_images(real_images) -> list[np.ndarray]:
    """Grade-0 photographs, for false-positive checks on normal anatomy."""
    out = [img for img, _, g in real_images if g == 0]
    if not out:
        pytest.skip("no grade-0 photographs available")
    return out


# --------------------------------------------------------------------------
# Degradations
# --------------------------------------------------------------------------
# Each models a real capture failure a rural PHC actually produces, and each is
# labelled with whether software can undo it -- which is the distinction the
# quality gate is built around (drscreen.preprocess.quality.CORRECTABLE).
def defocus(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Out of focus. NOT correctable: the high-frequency detail is gone."""
    sigma = max(1.0, 0.012 * strength * max(image.shape[:2]))
    return cv2.GaussianBlur(image, (0, 0), sigma)


def blow_out(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Lens flare / specular sheet. NOT correctable: saturated pixels carry no
    signal, so there is nothing for an enhancer to recover."""
    h, w = image.shape[:2]
    out = image.astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h * 0.45, w * 0.5
    r = np.hypot(yy - cy, xx - cx) / (0.42 * max(h, w))
    flare = np.clip(1.0 - r, 0, 1) ** 1.4
    out += (255.0 * flare * strength)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def clip_field(image: np.ndarray, keep: float = 0.55) -> np.ndarray:
    """Retina runs off the sensor. NOT correctable: it was never captured."""
    h, w = image.shape[:2]
    return image[:, : max(8, int(w * keep))].copy()


def vignette(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Uneven flash / pupil miosis. CORRECTABLE by illumination normalisation --
    this one must NOT cost the patient a second visit."""
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - h / 2, xx - w / 2) / (0.5 * max(h, w))
    gain = np.clip(1.0 - strength * 0.85 * r ** 2, 0.12, 1.0).astype(np.float32)
    return np.clip(image.astype(np.float32) * gain[..., None], 0, 255).astype(np.uint8)


def underexpose(image: np.ndarray, factor: float = 0.35) -> np.ndarray:
    """Under-driven flash. CORRECTABLE by the exposure/gamma correction."""
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


#: Defects the gate is required to refuse, after enhancement has been tried.
UNRECOVERABLE = {"defocus": defocus, "blow_out": blow_out, "clip_field": clip_field}

#: Defects the gate is required to fix and pass.
RECOVERABLE = {"vignette": vignette, "underexpose": underexpose}
