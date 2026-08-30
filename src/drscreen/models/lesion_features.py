"""Clinical feature extraction from lesion masks, and a rule-based ICDR grader.

This module is the bridge between "pixels the network lit up" and "the reason
a human grader would give".  It converts lesion segmentation masks into the
quantities the International Clinical DR severity scale is actually *defined*
in -- lesion counts per retinal quadrant, neovascularisation location,
distance from the fovea in disc diameters -- and then applies the published
criteria directly.

Two things fall out of that:

1. **Explainability with clinical vocabulary.**  The report can say
   "27 haemorrhages across 4 quadrants; venous beading in 2 -- meets the 4-2-1
   criterion for severe NPDR", which an ophthalmologist can verify in seconds.
   A Grad-CAM blob cannot be verified, only eyeballed.

2. **A genuine single-technique baseline.**  The problem statement requires
   showing that the *integrated* pipeline beats any single technique.  This
   rule grader is one of those single techniques (the classical CV arm), the
   CNN is another, and the fusion model must beat both -- measured, not
   asserted, in :mod:`drscreen.evaluation.ablation`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

from ..constants import LESION_CLASSES, REFERABLE_THRESHOLD
from ..preprocess.landmarks import Landmarks, quadrant_of, disc_diameters_from_fovea

QUADRANTS = ["superior", "inferior", "nasal", "temporal"]


@dataclass
class LesionInstance:
    lesion: str
    x: float
    y: float
    area_px: float
    quadrant: str
    dd_from_fovea: float
    confidence: float


@dataclass
class ClinicalFeatures:
    counts: dict[str, int] = field(default_factory=dict)
    area_fraction: dict[str, float] = field(default_factory=dict)
    per_quadrant: dict[str, dict[str, int]] = field(default_factory=dict)
    quadrants_with_hemorrhage: int = 0
    quadrants_with_beading: int = 0
    nv_at_disc: bool = False
    nv_elsewhere: bool = False
    #: Lesion classes the segmentation model was never trained to detect, so a
    #: count of zero for them means "not assessed", not "absent". Populated
    #: from the segmentation checkpoint's ``supervised_lesion_classes``.
    unassessed: tuple[str, ...] = ()
    lesions_within_1dd_of_fovea: int = 0
    exudates_within_1dd_of_fovea: int = 0
    nearest_lesion_dd: float = 99.0
    vessel_density: float = 0.0
    vessel_caliber_cv: float = 0.0
    instances: list[LesionInstance] = field(default_factory=list)

    def to_vector(self) -> np.ndarray:
        """Fixed-length numeric vector for the fusion head.

        Counts are log1p-compressed: the difference between 0 and 5
        microaneurysms is clinically decisive, the difference between 60 and 65
        is not, and a linear encoding would let the high end dominate the
        gradient.
        """
        v: list[float] = []
        for c in LESION_CLASSES:
            v.append(np.log1p(self.counts.get(c, 0)))
            v.append(self.area_fraction.get(c, 0.0) * 100.0)
        for c in LESION_CLASSES:
            for q in QUADRANTS:
                v.append(np.log1p(self.per_quadrant.get(c, {}).get(q, 0)))
        v += [
            float(self.quadrants_with_hemorrhage),
            float(self.quadrants_with_beading),
            float(self.nv_at_disc),
            float(self.nv_elsewhere),
            np.log1p(self.lesions_within_1dd_of_fovea),
            np.log1p(self.exudates_within_1dd_of_fovea),
            float(np.clip(self.nearest_lesion_dd, 0, 10)) / 10.0,
            self.vessel_density,
            self.vessel_caliber_cv,
        ]
        return np.asarray(v, np.float32)

    @staticmethod
    def vector_size() -> int:
        return len(LESION_CLASSES) * 2 + len(LESION_CLASSES) * len(QUADRANTS) + 9

    @staticmethod
    def feature_names() -> list[str]:
        names = []
        for c in LESION_CLASSES:
            names += [f"log_count_{c}", f"area_pct_{c}"]
        for c in LESION_CLASSES:
            for q in QUADRANTS:
                names.append(f"log_count_{c}_{q}")
        names += ["quadrants_with_hemorrhage", "quadrants_with_beading",
                  "nv_at_disc", "nv_elsewhere", "log_lesions_1dd_fovea",
                  "log_exudates_1dd_fovea", "nearest_lesion_dd_norm",
                  "vessel_density", "vessel_caliber_cv"]
        return names

    def to_dict(self) -> dict:
        d = asdict(self)
        d["instances"] = [asdict(i) for i in self.instances]
        return d


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------
#: Minimum blob area in px at 512x512, per lesion class. Below this a blob is
#: read as noise. Microaneurysms are genuinely 2-3 px across, so their floor is
#: at the resolution limit and the value scales with image size.
MIN_AREA_512 = {
    "microaneurysm": 3,
    "hemorrhage": 12,
    "hard_exudate": 4,
    "soft_exudate": 25,
    "neovascularization": 20,
}


def extract(lesion_probs: np.ndarray, lm: Landmarks,
            fov_mask: np.ndarray | None = None,
            vessel_mask: np.ndarray | None = None,
            threshold: float | dict[str, float] = 0.5,
            unassessed: tuple[str, ...] = ()) -> ClinicalFeatures:
    """Turn per-class lesion probability maps into clinical features.

    Parameters
    ----------
    lesion_probs
        ``(H, W, NUM_LESION_CLASSES)`` float array in [0, 1].
    lm
        Landmarks providing the clinical coordinate frame.
    threshold
        Scalar, or per-class mapping. A single 0.5 is the wrong default: fitted
        on held-out IDRiD the F1-optimal cut-points are 0.85-0.95, and at 0.5
        the exudate channel produced enough false positives on healthy retinas
        to trip the macular-oedema rule and mark them urgent.
    """
    h, w = lesion_probs.shape[:2]
    scale = (h * w) / (512.0 * 512.0)
    retina_area = float((fov_mask > 0).sum()) if fov_mask is not None else float(h * w)
    retina_area = max(retina_area, 1.0)

    feats = ClinicalFeatures()
    feats.unassessed = tuple(unassessed)
    feats.per_quadrant = {c: dict.fromkeys(QUADRANTS, 0) for c in LESION_CLASSES}
    nearest = 99.0

    for ci, cname in enumerate(LESION_CLASSES):
        thr = threshold.get(cname, 0.5) if isinstance(threshold, dict) else threshold
        prob = lesion_probs[..., ci]
        binary = (prob >= thr).astype(np.uint8)
        if fov_mask is not None:
            binary &= (fov_mask > 0).astype(np.uint8)
        feats.area_fraction[cname] = float(binary.sum()) / retina_area

        min_area = max(1, int(MIN_AREA_512[cname] * scale))
        n, labels, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
        count = 0
        for i in range(1, n):
            area = float(stats[i, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            cx, cy = float(cents[i][0]), float(cents[i][1])
            q = quadrant_of(cx, cy, lm)
            dd = disc_diameters_from_fovea(cx, cy, lm)
            conf = float(prob[labels == i].mean())
            feats.instances.append(LesionInstance(cname, cx, cy, area, q, dd, conf))
            feats.per_quadrant[cname][q] += 1
            count += 1
            nearest = min(nearest, dd)
            if dd <= 1.0:
                feats.lesions_within_1dd_of_fovea += 1
                if cname == "hard_exudate":
                    feats.exudates_within_1dd_of_fovea += 1
            if cname == "neovascularization":
                d_disc = np.hypot(cx - lm.disc_xy[0], cy - lm.disc_xy[1])
                if d_disc <= 1.5 * lm.disc_radius:
                    feats.nv_at_disc = True
                else:
                    feats.nv_elsewhere = True
        feats.counts[cname] = count

    feats.nearest_lesion_dd = float(nearest)

    # 4-2-1: quadrants carrying >= 20 haemorrhages/microaneurysms.
    heavy = 0
    for q in QUADRANTS:
        n_q = feats.per_quadrant["hemorrhage"].get(q, 0) + \
              feats.per_quadrant["microaneurysm"].get(q, 0)
        if n_q >= 20:
            heavy += 1
    feats.quadrants_with_hemorrhage = heavy

    if vessel_mask is not None:
        feats.vessel_density = float((vessel_mask > 0).sum()) / retina_area
        feats.vessel_caliber_cv, feats.quadrants_with_beading = _caliber_stats(vessel_mask, lm)

    return feats


def _caliber_stats(vessel_mask: np.ndarray, lm: Landmarks,
                   min_segment_px: int = 40,
                   min_caliber_px: float = 1.8,
                   beading_ratio: float = 1.9) -> tuple[float, int]:
    """Venous-beading proxy, measured *within* individual vessel segments.

    Beading is a focal dilatation along one venule.  The tempting shortcut --
    take the calibre of every skeleton pixel in a quadrant and look at its
    spread -- is wrong and produces a near-constant false positive, because
    that spread is dominated by the difference between a first-order arcade and
    a terminal capillary, which is normal anatomy rather than pathology.

    So calibre is compared only against the same segment's own median, and only
    for segments thick enough to be venules (capillaries cannot bead).  This is
    a screening cue; the report labels it as such and never grades on it alone.
    """
    binary = (vessel_mask > 0).astype(np.uint8)
    if binary.sum() < 50:
        return 0.0, 0
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dil = cv2.dilate(dist, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    skel = ((dist >= dil - 1e-3) & (dist > 1.0)).astype(np.uint8)
    if skel.sum() < 30:
        return 0.0, 0

    cv_all = float(np.std(dist[skel > 0]) / max(np.mean(dist[skel > 0]), 1e-6))

    # Split the skeleton into segments; each is analysed against itself.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(skel, connectivity=8)
    beaded = set()
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_segment_px:
            continue
        ys, xs = np.nonzero(labels == i)
        c = dist[ys, xs]
        med = float(np.median(c))
        if med < min_caliber_px:          # too fine to be a venule
            continue
        focal = float(np.percentile(c, 95)) / max(med, 1e-6)
        if focal < beading_ratio:
            continue
        # Attribute the segment to the quadrant holding most of its length.
        counts: dict[str, int] = {}
        step = max(1, len(xs) // 60)
        for j in range(0, len(xs), step):
            q = quadrant_of(float(xs[j]), float(ys[j]), lm)
            counts[q] = counts.get(q, 0) + 1
        if counts:
            beaded.add(max(counts, key=counts.get))
    return cv_all, len(beaded)


# --------------------------------------------------------------------------
# Rule-based ICDR grader (the classical-CV arm of the ablation)
# --------------------------------------------------------------------------
def rule_grade(f: ClinicalFeatures) -> tuple[int, list[str]]:
    """Apply the International Clinical DR Severity Scale directly.

    Returns ``(grade, list_of_criteria_met)``.  The criteria strings are what
    the clinical report prints as its justification.
    """
    reasons: list[str] = []
    ma = f.counts.get("microaneurysm", 0)
    he = f.counts.get("hemorrhage", 0)
    ex = f.counts.get("hard_exudate", 0)
    se = f.counts.get("soft_exudate", 0)
    nv = f.counts.get("neovascularization", 0)

    # --- Grade 4: proliferative DR ------------------------------------
    #
    # If neovascularisation was never assessed, say so. This arm is the only
    # route to grade 4, and on a model trained against IDRiD the NV channel has
    # no annotated pixel anywhere in the corpus -- so it returns zero for every
    # image and this branch silently never fires. A rule engine that cannot
    # reach its own top grade must declare that, or a "grade 3" it returns
    # reads as "assessed and not proliferative" when it means no such thing.
    if "neovascularization" in f.unassessed:
        reasons.append("Neovascularisation NOT ASSESSED - no pixel supervision "
                       "for this class in the training corpus, so proliferative "
                       "DR cannot be excluded by lesion evidence.")
    elif nv > 0 or f.nv_at_disc or f.nv_elsewhere:
        where = []
        if f.nv_at_disc:
            where.append("at the disc (NVD)")
        if f.nv_elsewhere:
            where.append("elsewhere (NVE)")
        reasons.append(f"Neovascularisation detected {' and '.join(where) or ''}".strip()
                       + " - defines proliferative DR.")
        return 4, reasons

    # --- Grade 3: severe NPDR, the 4-2-1 rule --------------------------
    severe = False
    if f.quadrants_with_hemorrhage >= 4:
        reasons.append(f"20 or more haemorrhages/microaneurysms in each of "
                       f"{f.quadrants_with_hemorrhage} quadrants (4-2-1 rule, '4').")
        severe = True
    if f.quadrants_with_beading >= 2:
        reasons.append(f"Definite venous beading in {f.quadrants_with_beading} "
                       f"quadrants (4-2-1 rule, '2').")
        severe = True
    if severe:
        return 3, reasons

    # --- Grade 2: moderate NPDR ---------------------------------------
    if he > 0 or ex > 0 or se > 0 or ma >= 8:
        bits = []
        if ma:
            bits.append(f"{ma} microaneurysm{'s' if ma != 1 else ''}")
        if he:
            bits.append(f"{he} haemorrhage{'s' if he != 1 else ''}")
        if ex:
            bits.append(f"{ex} hard-exudate focus/foci")
        if se:
            bits.append(f"{se} cotton-wool spot{'s' if se != 1 else ''}")
        reasons.append("More than microaneurysms alone but less than severe NPDR: "
                       + ", ".join(bits) + ".")
        return 2, reasons

    # --- Grade 1: mild NPDR -------------------------------------------
    if ma > 0:
        reasons.append(f"{ma} microaneurysm{'s' if ma != 1 else ''} only - mild NPDR.")
        return 1, reasons

    reasons.append("No microaneurysms, haemorrhages, exudates or "
                   "neovascularisation detected.")
    return 0, reasons


def dme_risk(f: ClinicalFeatures) -> tuple[int, str]:
    """Macular-oedema risk from exudate proximity to the fovea (IDRiD scale).

    0 = no apparent oedema, 1 = exudates present but > 1 DD from the fovea,
    2 = exudates within 1 DD of the fovea (clinically significant).
    """
    if f.exudates_within_1dd_of_fovea > 0:
        return 2, (f"{f.exudates_within_1dd_of_fovea} hard-exudate focus/foci within "
                   f"1 disc diameter of the fovea - clinically significant macular "
                   f"oedema likely; urgent referral.")
    if f.counts.get("hard_exudate", 0) > 0:
        return 1, (f"Hard exudates present but the nearest is "
                   f"{f.nearest_lesion_dd:.1f} DD from the fovea.")
    return 0, "No hard exudates detected in the macular region."


def is_referable(grade: int) -> bool:
    return grade >= REFERABLE_THRESHOLD
