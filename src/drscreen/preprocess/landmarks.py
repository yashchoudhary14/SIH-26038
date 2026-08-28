"""Analytic optic-disc and fovea localisation.

Why this is not a neural network
--------------------------------
Two of the ICDR grading criteria are *geometric*: severe NPDR is defined by
lesion counts **per quadrant** (the 4-2-1 rule), and clinically significant
macular oedema is defined by lesion distance from the fovea **in disc
diameters**.  Both need a coordinate frame, and a coordinate frame that fails
silently is worse than none.

A closed-form detector is auditable, needs no training data, runs in ~20 ms on
a CPU, and -- crucially -- returns a confidence we can propagate into the
report instead of a hallucinated point.  The learned segmentation head in
:mod:`drscreen.models.segmentation` refines this when it is available; this
module is the always-present fallback and the sanity check on that head.

Optic disc cue
    Brightest region at disc scale, weighted by local vessel convergence --
    the disc is where every arcade meets, which distinguishes it from a large
    confluent exudate (bright but avascular).

Fovea cue
    Darkest region on a ring 2-3 disc diameters temporal to the disc, along
    the axis of the vascular arcades.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import cv2
import numpy as np


@dataclass
class Landmarks:
    disc_xy: tuple[int, int]
    disc_radius: float
    disc_confidence: float
    fovea_xy: tuple[int, int]
    fovea_confidence: float
    laterality: str            # "OD" (right eye) | "OS" (left eye) | "unknown"
    disc_diameter_px: float    # the clinical unit of length

    def to_dict(self) -> dict:
        return asdict(self)


_GRID_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _norm_grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Cached normalised coordinate grids.

    ``np.mgrid`` alone was 12% of the landmark stage's runtime, and the grids
    depend only on the (fixed) working resolution.
    """
    key = (h, w)
    g = _GRID_CACHE.get(key)
    if g is None:
        yy, xx = np.mgrid[0:h, 0:w]
        g = ((xx / w - 0.5).astype(np.float32), (yy / h - 0.5).astype(np.float32))
        if len(_GRID_CACHE) > 8:
            _GRID_CACHE.clear()
        _GRID_CACHE[key] = g
    return g


def flat_field(gray: np.ndarray, mask: np.ndarray, order: int = 2,
               iterations: int = 3) -> np.ndarray:
    """Divide out the illumination field, modelled as a low-order polynomial.

    A wide Gaussian is the usual choice, but its kernel is necessarily of the
    same spatial scale as the macula (both are a sizeable fraction of the
    frame), so it absorbs the macular depression into the "illumination" and
    the fovea detector then has nothing to find.  A degree-2 surface has only
    six free parameters: it captures vignetting and a tilted flash -- which is
    what the physics produces -- and is structurally incapable of representing
    a localised dip.

    The fit is re-weighted a few times with the brightest and darkest
    residuals trimmed, so the optic disc and large haemorrhages do not drag
    the surface.
    """
    g = gray.astype(np.float32)
    h, w = g.shape[:2]
    m = mask > 0
    if m.sum() < 50:
        return (g / max(float(g.mean()), 1.0)).astype(np.float32)

    Xn, Yn = _norm_grid(h, w)
    xs_all = Xn[m]; ys_all = Yn[m]; z_all = g[m]

    # Six coefficients do not need a quarter of a million equations. Subsampling
    # to a few thousand points is statistically identical and takes the fit from
    # ~700 ms to a few ms, which is what makes the gate viable on an edge device.
    max_pts = 20000
    if z_all.size > max_pts:
        step = int(np.ceil(z_all.size / max_pts))
        xs_all, ys_all, z_all = xs_all[::step], ys_all[::step], z_all[::step]

    xn = xs_all.astype(np.float64)
    yn = ys_all.astype(np.float64)
    z = z_all.astype(np.float64)

    terms = [(d - i, i) for d in range(1, order + 1) for i in range(d + 1)]
    cols = [np.ones_like(xn)] + [(xn ** a) * (yn ** b) for a, b in terms]
    A = np.stack(cols, axis=1)

    keep = np.ones(z.shape, bool)
    coef = np.zeros(A.shape[1])
    for _ in range(max(1, iterations)):
        coef, *_ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
        resid = z - A @ coef
        lo, hi = np.percentile(resid[keep], [10, 90])
        keep = (resid >= lo) & (resid <= hi)
        if keep.sum() < A.shape[1] * 10:
            break

    # Evaluate the surface over the whole frame in float32.
    coef = coef.astype(np.float32)
    field = np.full((h, w), coef[0], np.float32)
    for k, (a, b) in enumerate(terms, start=1):
        contrib = coef[k]
        if a == 1:
            contrib = contrib * Xn
        elif a == 2:
            contrib = contrib * Xn * Xn
        elif a > 2:
            contrib = contrib * np.power(Xn, a)
        if b == 1:
            contrib = contrib * Yn
        elif b == 2:
            contrib = contrib * Yn * Yn
        elif b > 2:
            contrib = contrib * np.power(Yn, b)
        field += contrib

    out = g / np.maximum(field, 1.0)
    out[~m] = 1.0
    return out


def vessel_density(image: np.ndarray, mask: np.ndarray,
                   scale: float = 1.0) -> np.ndarray:
    """Cheap vesselness: bottom-hat on green, smoothed to a density field."""
    green = image[..., 1] if image.ndim == 3 else image
    g = green.astype(np.float32)
    k = max(3, int(9 * scale) | 1)
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    resp = np.clip(bg - g, 0, None)
    resp[mask == 0] = 0
    return cv2.GaussianBlur(resp, (0, 0), max(3.0, 0.04 * max(image.shape[:2])))


def locate(image: np.ndarray, mask: np.ndarray | None = None) -> Landmarks:
    """Locate the optic disc and fovea in a standardised (square) fundus image."""
    h, w = image.shape[:2]
    if mask is None:
        mask = np.full((h, w), 255, np.uint8)
    m = (mask > 0).astype(np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    flat = flat_field(gray, m)

    # Exclude a rim so the aperture edge cannot win either arg-extremum.
    er = max(5, int(0.06 * min(h, w)) | 1)
    inner = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (er, er)))

    # ---------------- optic disc ----------------------------------------
    # Typical disc diameter is ~0.13 of the 45-degree field width.
    disc_r0 = 0.065 * min(h, w)
    bright = cv2.GaussianBlur(flat, (0, 0), disc_r0 * 0.7)
    vd = vessel_density(image, m)
    vd_n = vd / max(float(vd.max()), 1e-6)
    lo = float(np.percentile(flat[inner > 0], 5))
    hi = float(np.percentile(flat[inner > 0], 99.5))
    bright_n = np.clip((bright - lo) / max(hi - lo, 1e-6), 0.0, 1.5)

    # Weight on vessel convergence. Tuned on phantoms to 0.70; re-fit on the
    # IDRiD optic-disc masks with scripts/eval_landmarks.py --real once the
    # dataset is present, since a bright confluent exudate is the failure mode
    # this term exists to defeat and phantoms under-represent those.
    score = bright_n * (0.30 + 0.70 * vd_n)
    score[inner == 0] = -1.0
    idx = int(np.argmax(score))
    dy, dx = divmod(idx, w)
    disc_peak = float(score[dy, dx])

    # Refine the radius: grow a bright region around the peak.
    thr = float(np.percentile(flat[inner > 0], 97))
    bw = ((flat > thr) & (inner > 0)).astype(np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(disc_r0 * 0.5) | 1,) * 2))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(bw, connectivity=8)
    disc_r = disc_r0
    if n > 1 and labels[dy, dx] > 0:
        lab = labels[dy, dx]
        area = stats[lab, cv2.CC_STAT_AREA]
        disc_r = float(np.clip(np.sqrt(area / np.pi), 0.5 * disc_r0, 1.8 * disc_r0))
        dx, dy = int(cents[lab][0]), int(cents[lab][1])

    # Confidence: how much the peak stands out above the field.
    bg_level = float(np.percentile(score[inner > 0], 95))
    disc_conf = float(np.clip((disc_peak - bg_level) / max(bg_level, 1e-3) * 3.0, 0.0, 1.0))

    laterality = "OS" if dx < w / 2 else "OD"

    # ---------------- fovea ---------------------------------------------
    # The temporal direction is taken as disc -> FOV centroid rather than a
    # left/right half-plane test: when the disc sits near the vertical
    # midline the half-plane test flips sign on noise, which sends the search
    # nasally and lands on an arcade instead of the macula.
    ys_in, xs_in = np.nonzero(inner)
    cx_in, cy_in = float(xs_in.mean()), float(ys_in.mean())
    vx, vy = cx_in - dx, cy_in - dy
    norm = math.hypot(vx, vy)
    if norm < 1e-3:
        vx, vy, norm = (-1.0 if dx > w / 2 else 1.0), 0.0, 1.0
    axis = math.atan2(vy / norm, vx / norm)

    dd = 2.0 * disc_r                            # one disc diameter
    smooth = cv2.GaussianBlur(flat, (0, 0), disc_r * 0.55)

    # Score = darkness + a weak prior on the textbook geometry (2.5 DD from
    # the disc, on the disc-to-centre axis). The prior only breaks ties; a
    # genuinely displaced fovea still wins on darkness.
    best, best_val, best_dark = None, 1e9, 0.0
    for r_mult in np.linspace(1.6, 3.4, 19):
        for ang in np.linspace(-0.62, 0.62, 25):    # +-35 deg about the axis
            fx = dx + dd * r_mult * math.cos(axis + ang)
            fy = dy + dd * r_mult * math.sin(axis + ang)
            xi, yi = int(round(fx)), int(round(fy))
            if not (0 <= xi < w and 0 <= yi < h) or inner[yi, xi] == 0:
                continue
            penalty = 0.010 * abs(r_mult - 2.5) + 0.020 * abs(ang)
            val = float(smooth[yi, xi]) + penalty
            if val < best_val:
                best_val, best, best_dark = val, (xi, yi), float(smooth[yi, xi])

    if best is None:                              # disc near the rim: fall back
        s = smooth.copy()
        s[inner == 0] = 1e9
        cv2.circle(s, (dx, dy), int(disc_r * 2.0), 1e9, -1)
        i2 = int(np.argmin(s))
        fy2, fx2 = divmod(i2, w)
        best, best_dark = (fx2, fy2), float(smooth[fy2, fx2])

    surround = float(np.median(flat[inner > 0]))
    fovea_conf = float(np.clip((surround - best_dark) / 0.18, 0.0, 1.0))

    return Landmarks(disc_xy=(int(dx), int(dy)), disc_radius=float(disc_r),
                     disc_confidence=round(disc_conf, 4),
                     fovea_xy=(int(best[0]), int(best[1])),
                     fovea_confidence=round(fovea_conf, 4),
                     laterality=laterality,
                     disc_diameter_px=float(2.0 * disc_r))


# --------------------------------------------------------------------------
# Clinical coordinate frame
# --------------------------------------------------------------------------
def quadrant_of(x: float, y: float, lm: Landmarks) -> str:
    """Assign a point to a retinal quadrant, for the 4-2-1 severe-NPDR rule.

    Quadrants are defined about the fovea with the horizontal raphe as the
    superior/inferior divider and the fovea-disc axis as the nasal/temporal
    divider, matching the ETDRS convention.
    """
    fx, fy = lm.fovea_xy
    dxx, dyy = lm.disc_xy
    axis = math.atan2(dyy - fy, dxx - fx)          # fovea -> disc  == nasal
    a = math.atan2(y - fy, x - fx) - axis
    a = (a + math.pi) % (2 * math.pi) - math.pi
    if -math.pi / 4 <= a < math.pi / 4:
        return "nasal"
    if math.pi / 4 <= a < 3 * math.pi / 4:
        return "inferior"
    if -3 * math.pi / 4 <= a < -math.pi / 4:
        return "superior"
    return "temporal"


def disc_diameters_from_fovea(x: float, y: float, lm: Landmarks) -> float:
    """Distance from the fovea in disc diameters -- the CSME yardstick."""
    fx, fy = lm.fovea_xy
    return float(math.hypot(x - fx, y - fy) / max(lm.disc_diameter_px, 1e-6))


def macula_score(lm: Landmarks, image_shape: tuple[int, int]) -> float:
    """Quality criterion: is the macula inside the frame with usable margin?"""
    h, w = image_shape[:2]
    fx, fy = lm.fovea_xy
    margin_px = min(fx, fy, w - fx, h - fy)
    # We want at least one disc diameter of retina around the fovea.
    margin = float(np.clip(margin_px / max(lm.disc_diameter_px, 1e-6), 0.0, 1.5)) / 1.5
    return float(np.clip(0.55 * margin + 0.45 * lm.fovea_confidence, 0.0, 1.0))
