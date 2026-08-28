"""Circular field-of-view (FOV) detection, tight cropping and geometry checks.

Fundus cameras image a circular aperture onto a rectangular sensor, so every
raw frame carries a black surround whose thickness depends on the camera
model.  Feeding that surround to a CNN wastes capacity and -- worse -- lets
the network identify the *camera* rather than the *pathology*, which is the
single most common cause of the train/test collapse seen when models move
from APTOS to Messidor-2.

This module recovers the retinal disc analytically (no learning involved) so
the same code path works on any camera, and reports how much of the circle
was clipped by the sensor edge, which is a graded quality criterion in the
problem statement ("field of view").
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Tuple

import cv2
import numpy as np


@dataclass
class FOVInfo:
    """Geometry of the retinal aperture within the raw frame."""
    cx: float                  # circle centre, x (px, raw-frame coords)
    cy: float                  # circle centre, y
    radius: float              # fitted radius (px)
    mask: np.ndarray           # uint8 {0,255}, raw-frame sized
    bbox: Tuple[int, int, int, int]   # x0, y0, x1, y1 of the retina
    coverage: float            # fraction of the fitted circle inside the frame
    fill_ratio: float          # retina px / frame px -- catches "no retina at all"
    clipped_sides: Tuple[bool, bool, bool, bool]  # left, top, right, bottom

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("mask")
        return d


def _largest_component(binary: np.ndarray) -> np.ndarray:
    """Keep only the biggest connected blob; kills specular dust and text overlays."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return binary
    # index 0 is background
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == biggest, 255, 0).astype(np.uint8)


def detect_fov(image: np.ndarray, threshold: int | None = None) -> FOVInfo:
    """Locate the circular retina in a BGR frame.

    Strategy: threshold the per-pixel channel maximum (the retina is bright in
    at least the red channel even in badly under-exposed captures), clean up
    with morphology, keep the largest blob, fill it, then fit a circle from
    the blob's area and centroid.  Area-based radius is far more stable than
    a Hough circle when part of the aperture is cut off by the sensor.
    """
    if image.ndim == 2:
        gray = image
    else:
        gray = image.max(axis=2)

    h, w = gray.shape[:2]

    if threshold is None:
        # Otsu on a blurred copy, but floored: a nearly-black frame would
        # otherwise get an Otsu threshold of ~2 and "detect" sensor noise.
        blur = cv2.GaussianBlur(gray, (0, 0), max(1.0, min(h, w) / 200.0))
        otsu, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = int(np.clip(otsu * 0.45, 8, 60))

    binary = (gray > threshold).astype(np.uint8) * 255

    k = max(3, (min(h, w) // 100) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = _largest_component(binary)

    # Fill interior holes (dark choroid / large hemorrhages must not punch
    # holes in the FOV mask).
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        binary = np.zeros_like(binary)
        cv2.drawContours(binary, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)

    area = float((binary > 0).sum())
    if area < 0.01 * h * w:
        # Degenerate: treat the whole frame as FOV rather than crash the pipeline.
        mask = np.full((h, w), 255, np.uint8)
        return FOVInfo(w / 2, h / 2, min(h, w) / 2, mask, (0, 0, w, h),
                       coverage=1.0, fill_ratio=1.0,
                       clipped_sides=(True, True, True, True))

    ys, xs = np.nonzero(binary)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    cx, cy = float(xs.mean()), float(ys.mean())

    # Radius from area (robust to clipping) blended with the half-width of the
    # widest scanline (robust to a partially imaged pupil).
    r_area = float(np.sqrt(area / np.pi))
    r_span = max(x1 - x0, y1 - y0) / 2.0
    radius = float(max(r_area, 0.85 * r_span))

    # How much of the ideal circle actually landed on the sensor?
    yy, xx = np.mgrid[0:h, 0:w]
    ideal_in_frame = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius ** 2
    ideal_area = np.pi * radius ** 2
    coverage = float(ideal_in_frame.sum() / max(ideal_area, 1.0))
    coverage = float(np.clip(coverage, 0.0, 1.0))

    margin = max(2, int(0.005 * min(h, w)))
    clipped = (x0 <= margin, y0 <= margin, x1 >= w - margin, y1 >= h - margin)

    return FOVInfo(cx, cy, radius, binary, (x0, y0, x1, y1),
                   coverage=coverage,
                   fill_ratio=float(area / (h * w)),
                   clipped_sides=clipped)


def crop_to_fov(image: np.ndarray, fov: FOVInfo | None = None,
                pad: float = 0.02) -> tuple[np.ndarray, np.ndarray, FOVInfo]:
    """Crop tight around the retina. Returns (cropped_bgr, cropped_mask, fov)."""
    if fov is None:
        fov = detect_fov(image)
    h, w = image.shape[:2]
    x0, y0, x1, y1 = fov.bbox
    px = int(pad * (x1 - x0))
    py = int(pad * (y1 - y0))
    x0, y0 = max(0, x0 - px), max(0, y0 - py)
    x1, y1 = min(w, x1 + px), min(h, y1 + py)
    return image[y0:y1, x0:x1].copy(), fov.mask[y0:y1, x0:x1].copy(), fov


def square_pad(image: np.ndarray, mask: np.ndarray | None = None,
               value: int = 0) -> tuple[np.ndarray, np.ndarray | None]:
    """Zero-pad to a square so `imresize` cannot change the aspect ratio.

    Aspect distortion matters clinically: optic-disc diameter (DD) is the unit
    used for 'severe NPDR' rules, so a stretched image corrupts the geometry
    the grading criteria are defined in.
    """
    h, w = image.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    bottom, right = side - h - top, side - w - left
    out = cv2.copyMakeBorder(image, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=value)
    out_mask = None
    if mask is not None:
        out_mask = cv2.copyMakeBorder(mask, top, bottom, left, right,
                                      cv2.BORDER_CONSTANT, value=0)
    return out, out_mask


def standardize(image: np.ndarray, size: int = 512,
                fov: FOVInfo | None = None,
                interpolation: int = cv2.INTER_AREA
                ) -> tuple[np.ndarray, np.ndarray, FOVInfo]:
    """Full geometric normalisation: detect -> crop -> square-pad -> resize."""
    cropped, mask, fov = crop_to_fov(image, fov)
    cropped, mask = square_pad(cropped, mask)
    interp = interpolation if cropped.shape[0] > size else cv2.INTER_CUBIC
    img = cv2.resize(cropped, (size, size), interpolation=interp)
    msk = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    # Erode 1% to drop the aliased rim, a notorious source of false exudates.
    er = max(3, (size // 100) | 1)
    msk = cv2.erode(msk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (er, er)))
    img = cv2.bitwise_and(img, img, mask=msk)
    return img, msk, fov
