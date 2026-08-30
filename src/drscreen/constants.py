"""Clinical constants shared across the pipeline.

Everything that a clinician or an evaluator might want to audit lives here,
so the grading scale, lesion taxonomy and target operating point are stated
once and never re-hardcoded inside model code.
"""
from __future__ import annotations

# --- International Clinical Diabetic Retinopathy (ICDR) severity scale ------
ICDR_GRADES = {
    0: "No apparent DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}
NUM_GRADES = len(ICDR_GRADES)

#: Grades >= this threshold require referral to an ophthalmologist.
REFERABLE_THRESHOLD = 2

#: Grades >= this threshold are sight-threatening (severe NPDR / PDR / CSME).
SIGHT_THREATENING_THRESHOLD = 3

# --- Problem-statement targets ---------------------------------------------
TARGET_SENSITIVITY = 0.90   # referable DR (grade >= 2)
TARGET_SPECIFICITY = 0.85   # referable DR (grade >= 2)

# --- Lesion taxonomy (channel order is fixed across the whole codebase) -----
LESION_CLASSES = [
    "microaneurysm",       # MA  - earliest sign, 15-60 um, sub-pixel at low res
    "hemorrhage",          # HE  - dot/blot/flame
    "hard_exudate",        # EX  - lipid deposits, sharp yellow
    "soft_exudate",        # SE  - cotton wool spots, fuzzy pale
    "neovascularization",  # NV  - defines proliferative DR
]
NUM_LESION_CLASSES = len(LESION_CLASSES)

#: Lesion classes the public corpora used here actually annotate at pixel level.
#: IDRiD -- the only lesion-annotated corpus in this project -- ships masks for
#: MA, HE, EX and SE, and none for neovascularisation. On real data the NV
#: channel therefore trains against all-zero targets, learns to predict nothing,
#: and returns a count of zero for every image ever screened. That is not a "no
#: neovascularisation" finding, it is an *unassessed* one -- and proliferative
#: DR is defined by neovascularisation, so reporting the two as the same thing
#: is the difference between "we looked and it is clear" and "we never looked".
#:
#: This tuple is the documented default for the real cohort. The authoritative
#: value is detected from the training masks by ``scripts/train_seg.py`` and
#: stored in the segmentation checkpoint as ``supervised_lesion_classes``; the
#: pipeline reads it from there, so a cohort that *does* annotate NV -- the
#: synthetic phantoms do -- lights the channel up on its own with no code change.
PIXEL_ANNOTATED_LESION_CLASSES = (
    "microaneurysm", "hemorrhage", "hard_exudate", "soft_exudate",
)

LESION_COLORS = {           # BGR, used for annotated overlays
    "microaneurysm":      (0, 0, 255),
    "hemorrhage":         (0, 96, 255),
    "hard_exudate":       (0, 255, 255),
    "soft_exudate":       (255, 255, 255),
    "neovascularization": (255, 0, 255),
}

# --- Structural landmarks ---------------------------------------------------
STRUCTURE_CLASSES = ["vessel", "optic_disc", "fovea"]

# --- Image quality --------------------------------------------------------
QUALITY_LABELS = {0: "ungradeable", 1: "borderline", 2: "good"}

#: Human-readable recapture instructions keyed by the failing quality metric.
RECAPTURE_ADVICE = {
    "focus":        "Image is out of focus. Re-focus on the optic disc and hold steady before capture.",
    "illumination": "Uneven or insufficient illumination. Re-centre the flash and check the working distance.",
    "over_exposure":"Image is over-exposed / washed out. Reduce flash intensity and recapture.",
    "under_exposure":"Image is too dark. Increase flash intensity or dilate the pupil before recapture.",
    "fov":          "Field of view is clipped. Centre the retina in the frame and recapture the full 45-degree field.",
    "artifact":     "Large artifact / reflection detected. Clean the lens, ask the patient to blink, then recapture.",
    "macula":       "Macula is not visible or is obscured. Direct the patient's gaze so the macula is centred.",
    "contrast":     "Vessel contrast is too low to grade. Consider pupil dilation before recapture.",
}
