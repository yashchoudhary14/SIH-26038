"""Procedural fundus phantom generator.

Purpose
-------
The real corpora (APTOS, IDRiD, DRIVE, Messidor-2) all require manual licence
acceptance and multi-GB downloads.  This module produces anatomically
structured phantoms so that **the entire pipeline -- training, calibration,
explainability, validation and the telemedicine simulation -- runs end to end
on a fresh clone with zero downloads.**

It is deliberately *not* a claim of clinical realism.  Its jobs are:

1. Integration testing: every tensor shape, loss, metric and report path is
   exercised for real, not mocked.
2. Demo: judges can watch the full flow in a minute.
3. Sanity: a model that cannot learn a grade from explicit lesion counts has
   a bug, so this doubles as a regression test for the training code.

What is modelled
----------------
* Circular aperture with camera-specific radius, tint and vignetting.
* Optic disc (bright, with a cup) and a macula/fovea depression at a
  physiologically plausible 2.5 disc-diameter separation.
* A recursive, tapering vascular tree with four arcades.
* Lesions placed according to the International Clinical DR severity scale,
  including the 4-2-1 rule for severe NPDR and neovascularisation for PDR.
* Field degradations: defocus, illumination gradient, sensor noise,
  over/under exposure, lens flare and aperture clipping.

Ground truth returned alongside every image: the ICDR grade, per-class lesion
masks, the vessel mask, the optic-disc mask, the fovea location and the
applied degradations.  That is enough to train and validate every head in the
system.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..constants import LESION_CLASSES, NUM_LESION_CLASSES


# --------------------------------------------------------------------------
# Camera profiles -- deliberately heterogeneous, to mimic the domain shift
# between a Topcon in a district hospital and a handheld in a rural PHC.
# --------------------------------------------------------------------------
@dataclass
class CameraProfile:
    name: str
    tint: tuple[float, float, float]   # BGR multiplicative tint
    fov_frac: float                    # aperture radius / half-min-dimension
    vignette: float                    # 0 = none, 1 = severe
    noise_sigma: float
    base_blur: float

CAMERAS = [
    CameraProfile("topcon_nw400",  (0.62, 0.78, 1.00), 0.94, 0.25, 1.2, 0.4),
    CameraProfile("canon_cr2",     (0.55, 0.72, 1.00), 0.90, 0.35, 1.8, 0.6),
    CameraProfile("handheld_a",    (0.68, 0.82, 0.98), 0.82, 0.55, 3.4, 1.1),
    CameraProfile("handheld_b",    (0.50, 0.70, 1.00), 0.78, 0.62, 4.2, 1.4),
    CameraProfile("smartphone_ro", (0.72, 0.85, 0.95), 0.70, 0.70, 5.0, 1.7),
]


@dataclass
class Phantom:
    image: np.ndarray                              # BGR uint8
    grade: int
    fov_mask: np.ndarray                           # uint8 {0,255}
    vessel_mask: np.ndarray                        # uint8 {0,255}
    disc_mask: np.ndarray                          # uint8 {0,255}
    lesion_masks: np.ndarray                       # (H, W, NUM_LESION_CLASSES) uint8
    fovea_xy: tuple[int, int]
    disc_xy: tuple[int, int]
    disc_radius: float
    camera: str
    quality_label: int                             # 0 ungradeable, 1 borderline, 2 good
    degradations: dict = field(default_factory=dict)
    lesion_counts: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Anatomy
# --------------------------------------------------------------------------
def _draw_vessel_tree(canvas: np.ndarray, rng: np.random.Generator,
                      start: tuple[float, float], angle: float, width: float,
                      length: float, depth: int, tortuosity: float,
                      beading: float = 0.0) -> None:
    """Recursively draw a tapering, tortuous vessel branch onto `canvas`."""
    if depth <= 0 or width < 0.6 or length < 4:
        return
    x, y = start
    steps = max(3, int(length / 4))
    pts = [(x, y)]
    a = angle
    for _ in range(steps):
        a += rng.normal(0, tortuosity)
        x += math.cos(a) * (length / steps)
        y += math.sin(a) * (length / steps)
        pts.append((x, y))

    for i in range(len(pts) - 1):
        t = i / max(len(pts) - 1, 1)
        w = width * (1.0 - 0.35 * t)
        if beading > 0 and (i % 3 == 0):
            w *= 1.0 + beading * rng.uniform(0.3, 0.9)   # venous beading (grade 3)
        cv2.line(canvas, tuple(np.int32(pts[i])), tuple(np.int32(pts[i + 1])),
                 255, max(1, int(round(w))), cv2.LINE_AA)

    end = pts[-1]
    n_children = 2 if depth > 2 else rng.integers(1, 3)
    for _ in range(int(n_children)):
        da = rng.normal(0, 0.45) + rng.choice([-1, 1]) * rng.uniform(0.18, 0.55)
        _draw_vessel_tree(canvas, rng, end, a + da, width * rng.uniform(0.62, 0.80),
                          length * rng.uniform(0.62, 0.82), depth - 1,
                          tortuosity * 1.1, beading)


def _build_vessels(size: int, rng: np.random.Generator, disc: tuple[float, float],
                   disc_r: float, beading: float = 0.0) -> np.ndarray:
    """Four arcades leaving the optic disc, plus nasal radial branches."""
    canvas = np.zeros((size, size), np.uint8)
    dx, dy = disc
    # Temporal direction points toward the macula.
    temporal = 1.0 if dx < size / 2 else -1.0
    base_w = max(2.0, size / 190.0)

    arcades = [
        (math.radians(35) if temporal > 0 else math.radians(145), 1.00),   # sup. temporal
        (math.radians(-35) if temporal > 0 else math.radians(-145), 1.00), # inf. temporal
        (math.radians(150) if temporal > 0 else math.radians(30), 0.72),   # sup. nasal
        (math.radians(-150) if temporal > 0 else math.radians(-30), 0.72), # inf. nasal
    ]
    for ang, scale in arcades:
        _draw_vessel_tree(canvas, rng, (dx, dy), ang, base_w * scale,
                          size * 0.34 * scale, 6, 0.10, beading)
    for _ in range(4):
        ang = rng.uniform(0, 2 * math.pi)
        _draw_vessel_tree(canvas, rng, (dx, dy), ang, base_w * 0.45,
                          size * 0.16, 4, 0.16, beading)
    return canvas


def _radial_falloff(size: int, cx: float, cy: float, radius: float,
                    strength: float) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(radius, 1.0)
    return np.clip(1.0 - strength * np.clip(d, 0, 1.4) ** 2, 0.15, 1.0)


# --------------------------------------------------------------------------
# Lesions
# --------------------------------------------------------------------------
def _lesion_budget(grade: int, rng: np.random.Generator) -> dict[str, int]:
    """Lesion counts consistent with the ICDR definitions.

    Grade 1  : microaneurysms only.
    Grade 2  : more than just MAs, but less than severe.
    Grade 3  : the 4-2-1 rule -- >20 haemorrhages in each of 4 quadrants, or
               venous beading in 2 quadrants, or prominent IRMA in 1.
    Grade 4  : neovascularisation and/or vitreous/preretinal haemorrhage.
    """
    if grade == 0:
        return dict.fromkeys(LESION_CLASSES, 0)
    if grade == 1:
        return {"microaneurysm": int(rng.integers(2, 8)), "hemorrhage": 0,
                "hard_exudate": 0, "soft_exudate": 0, "neovascularization": 0}
    if grade == 2:
        return {"microaneurysm": int(rng.integers(8, 26)),
                "hemorrhage": int(rng.integers(3, 12)),
                "hard_exudate": int(rng.integers(2, 14)),
                "soft_exudate": int(rng.integers(0, 3)),
                "neovascularization": 0}
    if grade == 3:
        # The 4-2-1 rule is a disjunction, so a severe-NPDR phantom must
        # actually satisfy one of its three arms -- otherwise the label says
        # "severe" while the pixels do not, and any grader that correctly
        # implements ICDR is penalised for being right.
        #
        # Arm '4' needs >=20 haemorrhages/microaneurysms in EACH of four
        # quadrants, i.e. >=80 lesions placed quadrant-uniformly, not the
        # ~50 scattered at random that an earlier version produced (which
        # reached the threshold in only ~1.6 quadrants on average).
        arm = rng.choice(["four", "beading", "irma"], p=[0.55, 0.30, 0.15])
        if arm == "four":
            per_q = int(rng.integers(22, 34))       # per quadrant, H + MA
            total = per_q * 4
            ma = int(total * rng.uniform(0.45, 0.6))
            return {"microaneurysm": ma, "hemorrhage": total - ma,
                    "hard_exudate": int(rng.integers(8, 30)),
                    "soft_exudate": int(rng.integers(3, 10)),
                    "neovascularization": 0,
                    "_severe_arm": arm, "_per_quadrant": per_q}
        # The beading and IRMA arms qualify on vascular signs, so the lesion
        # load is genuinely lower -- which is exactly why grading on counts
        # alone misses these patients.
        return {"microaneurysm": int(rng.integers(15, 40)),
                "hemorrhage": int(rng.integers(12, 35)),
                "hard_exudate": int(rng.integers(6, 25)),
                "soft_exudate": int(rng.integers(3, 10)),
                "neovascularization": 0,
                "_severe_arm": arm}
    return {"microaneurysm": int(rng.integers(20, 55)),
            "hemorrhage": int(rng.integers(20, 60)),
            "hard_exudate": int(rng.integers(5, 28)),
            "soft_exudate": int(rng.integers(2, 9)),
            "neovascularization": int(rng.integers(2, 6))}


def _sample_site(rng, size, fov_c, fov_r, disc, disc_r, avoid_disc=True):
    for _ in range(60):
        a = rng.uniform(0, 2 * math.pi)
        r = fov_r * math.sqrt(rng.uniform(0.0, 0.86))
        x = fov_c[0] + r * math.cos(a)
        y = fov_c[1] + r * math.sin(a)
        if avoid_disc and math.hypot(x - disc[0], y - disc[1]) < disc_r * 1.5:
            continue
        return float(x), float(y)
    return float(fov_c[0]), float(fov_c[1])


#: Sector index -> the quadrant name used by
#: :func:`drscreen.preprocess.landmarks.quadrant_of`, which measures the angle
#: from the fovea with the fovea->disc direction as nasal.
_QUADRANT_ORDER = ["nasal", "inferior", "temporal", "superior"]


def _sample_site_in_quadrant(rng, quadrant: str, fov_c, fov_r,
                             disc, disc_r, fovea):
    """Place a lesion inside a named ETDRS quadrant.

    Mirrors the angular convention of ``landmarks.quadrant_of`` exactly: angles
    are measured from the fovea, with the fovea->disc direction defining nasal.
    If the two ever drift apart, the phantom's ground truth stops matching what
    the feature extractor reports, and the 4-2-1 evaluation becomes meaningless.
    """
    axis = math.atan2(disc[1] - fovea[1], disc[0] - fovea[0])
    base = {"nasal": 0.0, "inferior": math.pi / 2,
            "temporal": math.pi, "superior": -math.pi / 2}[quadrant]
    for _ in range(80):
        ang = axis + base + rng.uniform(-math.pi / 4 + 0.08, math.pi / 4 - 0.08)
        r = fov_r * math.sqrt(rng.uniform(0.02, 0.80))
        x = fovea[0] + r * math.cos(ang)
        y = fovea[1] + r * math.sin(ang)
        if math.hypot(x - fov_c[0], y - fov_c[1]) > fov_r * 0.93:
            continue
        if math.hypot(x - disc[0], y - disc[1]) < disc_r * 1.5:
            continue
        return float(x), float(y)
    return _sample_site(rng, 0, fov_c, fov_r, disc, disc_r)


def _paint_lesions(retina: np.ndarray, masks: list[np.ndarray],
                   rng: np.random.Generator,
                   budget: dict[str, int], size: int, fov_c, fov_r,
                   disc, disc_r, vessels: np.ndarray,
                   fovea=None) -> dict[str, int]:
    """Paint lesions into the BGR retina and record per-class masks.

    `masks` is a list of contiguous 2-D uint8 arrays (one per lesion class);
    OpenCV cannot draw into a strided slice of a 3-D array, so the channels are
    kept separate here and stacked by the caller.
    """
    counts = dict.fromkeys(LESION_CLASSES, 0)
    px = size / 512.0     # scale factor: all radii below are quoted at 512 px

    # Severe NPDR via the '4' arm requires >=20 haemorrhages/microaneurysms in
    # each of four quadrants, so those two classes are dealt round-robin across
    # quadrants rather than scattered uniformly over the retina.
    per_q = budget.get("_per_quadrant")
    quad_cycle = None
    if per_q and fovea is not None:
        quad_cycle = [_QUADRANT_ORDER[i % 4] for i in range(per_q * 4 + 8)]
        rng.shuffle(quad_cycle)
    _qi = 0

    def _site(is_hem_or_ma: bool = False):
        nonlocal _qi
        if quad_cycle is not None and is_hem_or_ma and _qi < len(quad_cycle):
            q = quad_cycle[_qi]; _qi += 1
            return _sample_site_in_quadrant(rng, q, fov_c, fov_r, disc, disc_r, fovea)
        return _sample_site(rng, size, fov_c, fov_r, disc, disc_r)

    for _ in range(budget.get("microaneurysm", 0)):
        x, y = _site(True)
        r = max(1, int(round(rng.uniform(1.2, 3.0) * px)))
        col = (int(rng.uniform(18, 45)), int(rng.uniform(12, 32)), int(rng.uniform(110, 165)))
        cv2.circle(retina, (int(x), int(y)), r, col, -1, cv2.LINE_AA)
        cv2.circle(masks[LESION_CLASSES.index("microaneurysm")],
                   (int(x), int(y)), max(r, 2), 255, -1)
        counts["microaneurysm"] += 1

    for _ in range(budget.get("hemorrhage", 0)):
        x, y = _site(True)
        kind = rng.random()
        col = (int(rng.uniform(14, 38)), int(rng.uniform(8, 26)), int(rng.uniform(78, 130)))
        idx = LESION_CLASSES.index("hemorrhage")
        if kind < 0.55:                                   # dot/blot
            r = max(2, int(round(rng.uniform(3.0, 9.0) * px)))
            axes = (r, max(2, int(r * rng.uniform(0.7, 1.0))))
            ang = rng.uniform(0, 180)
            cv2.ellipse(retina, (int(x), int(y)), axes, ang, 0, 360, col, -1, cv2.LINE_AA)
            cv2.ellipse(masks[idx], (int(x), int(y)), axes, ang, 0, 360, 255, -1)
        else:                                             # flame-shaped
            L = rng.uniform(8, 22) * px
            ang = rng.uniform(0, 2 * math.pi)
            pts = np.int32([[x, y],
                            [x + L * math.cos(ang - 0.25), y + L * math.sin(ang - 0.25)],
                            [x + L * 1.15 * math.cos(ang), y + L * 1.15 * math.sin(ang)],
                            [x + L * math.cos(ang + 0.25), y + L * math.sin(ang + 0.25)]])
            cv2.fillPoly(retina, [pts], col, cv2.LINE_AA)
            cv2.fillPoly(masks[idx], [pts], 255)
        counts["hemorrhage"] += 1

    for _ in range(budget.get("hard_exudate", 0)):
        x, y = _sample_site(rng, size, fov_c, fov_r, disc, disc_r)
        idx = LESION_CLASSES.index("hard_exudate")
        n_blobs = int(rng.integers(1, 5))                 # exudates cluster
        for _ in range(n_blobs):
            ox, oy = rng.normal(0, 7 * px, 2)
            r = max(1, int(round(rng.uniform(1.8, 5.5) * px)))
            col = (int(rng.uniform(120, 175)), int(rng.uniform(215, 250)), int(rng.uniform(225, 255)))
            cv2.circle(retina, (int(x + ox), int(y + oy)), r, col, -1, cv2.LINE_AA)
            cv2.circle(masks[idx], (int(x + ox), int(y + oy)), r, 255, -1)
        counts["hard_exudate"] += 1

    for _ in range(budget.get("soft_exudate", 0)):
        x, y = _sample_site(rng, size, fov_c, fov_r, disc, disc_r)
        idx = LESION_CLASSES.index("soft_exudate")
        r = max(3, int(round(rng.uniform(6, 16) * px)))
        patch = np.zeros((size, size), np.uint8)
        cv2.circle(patch, (int(x), int(y)), r, 255, -1)
        patch = cv2.GaussianBlur(patch, (0, 0), r * 0.35)   # fuzzy border
        col = np.array([225, 240, 240], np.float32)
        alpha = (patch.astype(np.float32) / 255.0)[..., None] * 0.85
        retina[:] = np.clip(retina * (1 - alpha) + col * alpha, 0, 255).astype(np.uint8)
        masks[idx][:] = np.maximum(masks[idx], (patch > 90).astype(np.uint8) * 255)
        counts["soft_exudate"] += 1

    for _ in range(budget.get("neovascularization", 0)):
        idx = LESION_CLASSES.index("neovascularization")
        # NV grows at the disc (NVD) or elsewhere (NVE) -- fine, chaotic tufts.
        at_disc = rng.random() < 0.5
        if at_disc:
            cx, cy = disc[0] + rng.normal(0, disc_r * 0.4), disc[1] + rng.normal(0, disc_r * 0.4)
        else:
            cx, cy = _sample_site(rng, size, fov_c, fov_r, disc, disc_r)
        tuft = np.zeros((size, size), np.uint8)
        for _ in range(int(rng.integers(6, 14))):
            _draw_vessel_tree(tuft, rng, (cx, cy), rng.uniform(0, 2 * math.pi),
                              max(1.0, 1.2 * px), rng.uniform(12, 30) * px, 3, 0.55)
        retina[tuft > 0] = (np.array([30, 20, 150]) * rng.uniform(0.85, 1.1)).astype(np.uint8)
        masks[idx][:] = np.maximum(masks[idx], tuft)
        counts["neovascularization"] += 1

    return counts


# --------------------------------------------------------------------------
# Degradations
# --------------------------------------------------------------------------
def _degrade(img: np.ndarray, fov_mask: np.ndarray, rng: np.random.Generator,
             cam: CameraProfile, severity: float) -> tuple[np.ndarray, dict, int]:
    """Apply field-condition degradations. `severity` in [0, 1]. Returns quality label."""
    deg: dict[str, float] = {}
    size = img.shape[0]
    out = img.astype(np.float32)

    blur = cam.base_blur + severity * rng.uniform(0.0, 5.0) * (size / 512.0)
    if blur > 0.3:
        out = cv2.GaussianBlur(out, (0, 0), blur)
        deg["defocus_sigma"] = round(float(blur), 3)

    if rng.random() < 0.75:
        ang = rng.uniform(0, 2 * math.pi)
        amp = severity * rng.uniform(0.0, 0.65)
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        grad = 1.0 + amp * ((xx * math.cos(ang) + yy * math.sin(ang)) / size - 0.5) * 2.0
        out *= grad[..., None]
        deg["illum_gradient"] = round(float(amp), 3)

    gain = 1.0 + rng.normal(0, 0.10 + 0.35 * severity)
    gain = float(np.clip(gain, 0.35, 2.0))
    out *= gain
    deg["exposure_gain"] = round(gain, 3)

    sigma = cam.noise_sigma + severity * rng.uniform(0, 9.0)
    out += rng.normal(0, sigma, out.shape)
    deg["noise_sigma"] = round(float(sigma), 3)

    if rng.random() < 0.35 * severity + 0.05:              # lens flare / reflection
        fx, fy = rng.uniform(0.2, 0.8, 2) * size
        fr = rng.uniform(0.05, 0.16) * size * (0.5 + severity)
        flare = np.zeros((size, size), np.float32)
        cv2.circle(flare, (int(fx), int(fy)), int(fr), 1.0, -1)
        flare = cv2.GaussianBlur(flare, (0, 0), fr * 0.5)
        out += flare[..., None] * rng.uniform(80, 210)
        deg["flare"] = round(float(fr / size), 3)

    out = np.clip(out, 0, 255).astype(np.uint8)
    out = cv2.bitwise_and(out, out, mask=fov_mask)

    # Quality label derived from the degradation magnitudes actually applied.
    bad = 0.0
    bad += max(0.0, (blur / (size / 512.0) - 1.6)) / 3.0
    bad += max(0.0, deg.get("illum_gradient", 0.0) - 0.32) / 0.35
    bad += max(0.0, abs(math.log(gain)) - 0.30) / 0.45
    bad += max(0.0, deg.get("noise_sigma", 0.0) - 4.5) / 6.0
    bad += deg.get("flare", 0.0) / 0.14
    quality = 0 if bad > 1.35 else (1 if bad > 0.55 else 2)
    deg["degradation_index"] = round(float(bad), 3)
    return out, deg, quality


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def generate(grade: int | None = None, size: int = 512,
             seed: int | None = None, severity: float | None = None,
             camera: str | None = None) -> Phantom:
    """Generate one fundus phantom with full ground truth."""
    rng = np.random.default_rng(seed)
    if grade is None:
        # Prevalence roughly matching a rural Indian screening cohort.
        grade = int(rng.choice([0, 1, 2, 3, 4], p=[0.50, 0.20, 0.18, 0.07, 0.05]))
    cam = next((c for c in CAMERAS if c.name == camera), None) or CAMERAS[int(rng.integers(len(CAMERAS)))]
    if severity is None:
        severity = float(np.clip(rng.beta(1.6, 3.2), 0, 1))

    # --- aperture -------------------------------------------------------
    fov_r = size / 2 * cam.fov_frac
    jitter = size * 0.035
    fov_c = (size / 2 + rng.normal(0, jitter), size / 2 + rng.normal(0, jitter))
    fov_mask = np.zeros((size, size), np.uint8)
    cv2.circle(fov_mask, (int(fov_c[0]), int(fov_c[1])), int(fov_r), 255, -1)

    # --- base retina ----------------------------------------------------
    base = np.zeros((size, size, 3), np.float32)
    base[..., 0] = rng.uniform(28, 52)     # B
    base[..., 1] = rng.uniform(58, 92)     # G
    base[..., 2] = rng.uniform(150, 205)   # R
    texture = cv2.GaussianBlur(rng.normal(0, 9, (size, size)).astype(np.float32),
                               (0, 0), size / 90.0)
    base += texture[..., None] * np.array([0.35, 0.7, 1.0], np.float32)
    retina = np.clip(base, 0, 255).astype(np.uint8)

    # --- optic disc & macula -------------------------------------------
    left_eye = rng.random() < 0.5
    disc_r = size * rng.uniform(0.055, 0.075)
    side = -1.0 if left_eye else 1.0
    disc = (fov_c[0] + side * fov_r * rng.uniform(0.38, 0.50),
            fov_c[1] + rng.normal(0, size * 0.025))
    # Fovea sits ~2.5 disc diameters temporal to the disc.
    fovea = (disc[0] - side * disc_r * 5.0, disc[1] + rng.normal(0, size * 0.012))

    disc_mask = np.zeros((size, size), np.uint8)
    cv2.ellipse(disc_mask, (int(disc[0]), int(disc[1])),
                (int(disc_r), int(disc_r * rng.uniform(1.0, 1.18))),
                0, 0, 360, 255, -1)
    disc_soft = cv2.GaussianBlur(disc_mask.astype(np.float32) / 255.0, (0, 0), disc_r * 0.18)
    disc_col = np.array([190, 225, 250], np.float32) * rng.uniform(0.92, 1.05)
    retina = np.clip(retina * (1 - disc_soft[..., None] * 0.92)
                     + disc_col * disc_soft[..., None] * 0.92, 0, 255).astype(np.uint8)
    cup = np.zeros((size, size), np.float32)
    cv2.circle(cup, (int(disc[0]), int(disc[1])), int(disc_r * rng.uniform(0.3, 0.5)), 1.0, -1)
    cup = cv2.GaussianBlur(cup, (0, 0), disc_r * 0.15)
    retina = np.clip(retina.astype(np.float32) + cup[..., None] * 25, 0, 255).astype(np.uint8)

    mac = np.zeros((size, size), np.float32)
    cv2.circle(mac, (int(fovea[0]), int(fovea[1])), int(disc_r * 1.6), 1.0, -1)
    mac = cv2.GaussianBlur(mac, (0, 0), disc_r * 0.75)
    retina = np.clip(retina.astype(np.float32) * (1.0 - mac[..., None] * 0.38), 0, 255).astype(np.uint8)

    # --- vessels --------------------------------------------------------
    beading = 0.6 if grade >= 3 else 0.0
    vessels = _build_vessels(size, rng, disc, disc_r, beading)
    vessels = cv2.bitwise_and(vessels, vessels, mask=fov_mask)
    v_soft = cv2.GaussianBlur(vessels.astype(np.float32) / 255.0, (0, 0), 0.8)
    v_col = np.array([25, 20, 105], np.float32)
    retina = np.clip(retina * (1 - v_soft[..., None] * 0.88)
                     + v_col * v_soft[..., None] * 0.88, 0, 255).astype(np.uint8)

    # --- lesions --------------------------------------------------------
    lesion_planes = [np.zeros((size, size), np.uint8)
                     for _ in range(NUM_LESION_CLASSES)]
    budget = _lesion_budget(grade, rng)
    counts = _paint_lesions(retina, lesion_planes, rng, budget, size,
                            fov_c, fov_r, disc, disc_r, vessels, fovea)

    # --- optics: vignetting, tint, aperture ------------------------------
    fall = _radial_falloff(size, fov_c[0], fov_c[1], fov_r, cam.vignette)
    retina = np.clip(retina.astype(np.float32) * fall[..., None], 0, 255).astype(np.uint8)
    retina = np.clip(retina.astype(np.float32) * np.array(cam.tint, np.float32),
                     0, 255).astype(np.uint8)
    retina = cv2.bitwise_and(retina, retina, mask=fov_mask)

    # --- field degradations ---------------------------------------------
    retina, deg, qlabel = _degrade(retina, fov_mask, rng, cam, severity)

    for arr in (vessels, disc_mask):
        arr[:] = cv2.bitwise_and(arr, arr, mask=fov_mask)
    lesion_planes = [cv2.bitwise_and(m, m, mask=fov_mask) for m in lesion_planes]
    lesion_masks = np.stack(lesion_planes, axis=-1)

    return Phantom(image=retina, grade=grade, fov_mask=fov_mask,
                   vessel_mask=vessels, disc_mask=disc_mask,
                   lesion_masks=lesion_masks,
                   fovea_xy=(int(fovea[0]), int(fovea[1])),
                   disc_xy=(int(disc[0]), int(disc[1])), disc_radius=float(disc_r),
                   camera=cam.name, quality_label=qlabel,
                   degradations=deg, lesion_counts=counts)


def generate_cohort(n: int, size: int = 512, seed: int = 0,
                    domain_shift: bool = False,
                    prevalence: tuple[float, ...] = (0.50, 0.20, 0.18, 0.07, 0.05)
                    ) -> list[Phantom]:
    """Generate `n` phantoms.

    `domain_shift=True` restricts to the low-end handheld cameras and raises
    the degradation severity -- this is the stand-in for Messidor-2 while the
    real download is in flight, and it is what the external-validation script
    falls back to so the generalisation claim is never silently skipped.
    """
    rng = np.random.default_rng(seed)
    cams = [c.name for c in CAMERAS[2:]] if domain_shift else [c.name for c in CAMERAS]
    out = []
    p = np.asarray(prevalence, np.float64)
    p = p / p.sum()
    for i in range(n):
        g = int(rng.choice(len(p), p=p))
        sev = float(np.clip(rng.beta(2.2, 2.0) if domain_shift else rng.beta(1.6, 3.2), 0, 1))
        out.append(generate(grade=g, size=size, seed=int(rng.integers(1 << 31)),
                            severity=sev,
                            camera=cams[int(rng.integers(len(cams)))]))
    return out
