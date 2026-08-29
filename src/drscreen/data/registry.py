"""Dataset registry, discovery and the split policy.

Split policy (enforced in code, not just documented)
----------------------------------------------------
================  ==========================  ==============================
Dataset           Role                        Used for
================  ==========================  ==============================
APTOS 2019        train + internal val        Grader training, class balance
IDRiD (grading)   train + internal val        Grader fine-tuning
IDRiD (segment.)  train + internal val        Lesion / optic-disc heads
DRIVE             train + internal val        Vessel U-Net
Messidor-2        **held-out external test**  Zero-shot generalisation only
================  ==========================  ==============================

Messidor-2 is a *different country, different cameras, different graders*
than APTOS/IDRiD.  Reporting on it is the only honest way to claim the
>90% sensitivity / >85% specificity target will survive deployment, so the
loader refuses to hand it out for training and
:func:`assert_no_leakage` is called by the training scripts.

Download locations (all require manual acceptance of their licence terms):

* APTOS 2019 - https://www.kaggle.com/c/aptos2019-blindness-detection
* IDRiD      - https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
* DRIVE      - https://drive.grand-challenge.org/
* Messidor-2 - https://www.adcis.net/en/third-party/messidor2/
  (adjudicated grades: https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades)
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

#: ``.gif`` matters: DRIVE ships its vessel ground truth as ``21_manual1.gif``.
#: Omitting it makes the vessel masks invisible while the images still load,
#: so DRIVE appears present but silently contributes no supervision.
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".gif", ".ppm", ".bmp",
              ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF", ".GIF", ".PPM", ".BMP")


class SplitViolation(RuntimeError):
    """Raised when a held-out dataset is requested for training."""


@dataclass
class Sample:
    image_path: Path
    grade: int | None = None            # ICDR 0-4
    dme: int | None = None              # macular oedema risk 0-2 (IDRiD/Messidor)
    gradable: bool | None = None
    dataset: str = ""
    masks: dict[str, Path] | None = None  # lesion/vessel ground truth, if any
    subject_id: str = ""                  # for group-aware splitting


HELD_OUT = {"messidor2"}


def _find_dir(root: Path, *patterns: str) -> Path | None:
    """Case-insensitive recursive search for the first directory matching any pattern."""
    for pat in patterns:
        rx = re.compile(pat, re.IGNORECASE)
        for p in root.rglob("*"):
            if p.is_dir() and rx.search(p.name):
                return p
    return None


def _find_file(root: Path, *patterns: str) -> Path | None:
    for pat in patterns:
        rx = re.compile(pat, re.IGNORECASE)
        for p in root.rglob("*"):
            if p.is_file() and rx.search(p.name):
                return p
    return None


def _images_in(d: Path) -> list[Path]:
    return sorted(p for p in d.rglob("*") if p.suffix in IMAGE_EXTS)


def _subject_of(name: str) -> str:
    """Best-effort patient id so both eyes of one patient stay in one split.

    Messidor-2 names encode the exam (``20051020_43808_0100_PP``); APTOS uses
    opaque hashes with one image per patient.  Leaking a fellow eye across the
    train/val boundary inflates every metric, and it is the single most common
    silent bug in published DR pipelines.
    """
    stem = Path(name).stem
    m = re.match(r"^(\d{8}_\d+)", stem)          # Messidor-2 exam id
    if m:
        return m.group(1)
    m = re.match(r"^(IDRiD_\d+)", stem, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.match(r"^(\d+)_(left|right)$", stem, re.IGNORECASE)  # EyePACS style
    if m:
        return m.group(1)
    return stem


# --------------------------------------------------------------------------
# Individual dataset loaders
# --------------------------------------------------------------------------
def load_aptos(root: str | Path) -> list[Sample]:
    root = Path(root)
    csv = _find_file(root, r"^train.*\.csv$")
    img_dir = _find_dir(root, r"train.*images?$", r"^images?$")
    if csv is None or img_dir is None:
        return []
    df = pd.read_csv(csv)
    id_col = "id_code" if "id_code" in df.columns else df.columns[0]
    y_col = "diagnosis" if "diagnosis" in df.columns else df.columns[1]
    by_stem = {p.stem: p for p in _images_in(img_dir)}
    out = []
    for _, r in df.iterrows():
        p = by_stem.get(str(r[id_col]))
        if p is None:
            continue
        out.append(Sample(p, grade=int(r[y_col]), dataset="aptos2019",
                          gradable=True, subject_id=_subject_of(p.name)))
    return out


def load_idrid_grading(root: str | Path) -> list[Sample]:
    root = Path(root)
    # Scope to the grading subtree. IDRiD ships three topic folders
    # (A. Segmentation, B. Disease Grading, C. Localization) and *each* of
    # them contains a directory literally named "a. Training Set". Searching
    # the whole tree returns whichever one rglob reaches first, and the
    # segmentation set uses IDRiD_01 while grading uses IDRiD_001, so the
    # filename join silently produces zero samples rather than an error.
    grading_root = _find_dir(root, r"^B\.?\s*Disease.?Grading$",
                             r"Disease.?Grading$") or root

    out: list[Sample] = []
    for split in ("Training", "Testing"):
        csv = _find_file(grading_root, rf"Disease.?Grading.*{split}.*Labels.*\.csv$",
                         rf"{split}.*Labels.*\.csv$")
        # The "a."/"b." prefix indexes the split, it does not spell it, so
        # match on the split word and require an Original Images ancestor.
        img_dir = None
        for d in grading_root.rglob("*"):
            if (d.is_dir() and re.search(rf"{split}\s*Set$", d.name, re.I)
                    and re.search(r"Original", str(d.parent), re.I)):
                img_dir = d
                break
        if img_dir is None:
            img_dir = _find_dir(grading_root, rf"{split}\s*Set$")
        if csv is None or img_dir is None:
            continue
        df = pd.read_csv(csv)
        df.columns = [c.strip() for c in df.columns]
        name_col = next((c for c in df.columns if "image" in c.lower()), df.columns[0])
        grade_col = next((c for c in df.columns if "retinopathy" in c.lower()), df.columns[1])
        dme_col = next((c for c in df.columns if "macular" in c.lower() or "edema" in c.lower()), None)
        by_stem = {p.stem.lower(): p for p in _images_in(img_dir)}
        for _, r in df.iterrows():
            key = str(r[name_col]).strip().lower()
            p = by_stem.get(key)
            if p is None:
                continue
            try:
                grade = int(r[grade_col])
            except (TypeError, ValueError):
                continue
            out.append(Sample(p, grade=grade,
                              dme=int(r[dme_col]) if dme_col and not pd.isna(r[dme_col]) else None,
                              dataset="idrid_grading", gradable=True,
                              subject_id=_subject_of(p.name)))
    return out


IDRID_MASK_DIRS = {
    "microaneurysm":      r"Microaneurysm",
    "hemorrhage":         r"H(a)?emorrhage",
    "hard_exudate":       r"Hard.?Exudate",
    "soft_exudate":       r"Soft.?Exudate",
    "optic_disc":         r"Optic.?Disc",
}


def load_idrid_segmentation(root: str | Path) -> list[Sample]:
    """IDRiD pixel-level lesion ground truth (the only public sub-pixel MA set)."""
    root = Path(root)
    seg_root = _find_dir(root, r"^A\.?\s*Segmentation$", r"^Segmentation$") or root
    out: list[Sample] = []
    for split in ("Training", "Testing"):
        img_dir = None
        for d in seg_root.rglob("*"):
            if d.is_dir() and re.search(rf"{split}\s*Set$", d.name, re.I) \
               and re.search(r"Original", str(d.parent), re.I):
                img_dir = d
                break
        if img_dir is None:
            continue
        gt_root = None
        for d in seg_root.rglob("*"):
            if d.is_dir() and re.search(rf"{split}\s*Set$", d.name, re.I) \
               and re.search(r"Groundtruth", str(d.parent), re.I):
                gt_root = d
                break
        for p in _images_in(img_dir):
            masks: dict[str, Path] = {}
            if gt_root is not None:
                for lesion, pat in IDRID_MASK_DIRS.items():
                    sub = next((d for d in gt_root.iterdir()
                                if d.is_dir() and re.search(pat, d.name, re.I)), None)
                    if sub is None:
                        continue
                    hit = next((q for q in _images_in(sub)
                                if q.stem.lower().startswith(p.stem.lower())), None)
                    if hit is not None:
                        masks[lesion] = hit
            out.append(Sample(p, dataset="idrid_segmentation", gradable=True,
                              masks=masks or None, subject_id=_subject_of(p.name)))
    return out


def load_drive(root: str | Path) -> list[Sample]:
    root = Path(root)
    out: list[Sample] = []
    for split in ("training", "test"):
        img_dir = _find_dir(root, rf"^{split}$")
        if img_dir is None:
            continue
        imgs = _find_dir(img_dir, r"^images$")
        man = _find_dir(img_dir, r"^1st_manual$", r"^manual1?$")
        fov = _find_dir(img_dir, r"^mask$")
        if imgs is None:
            continue
        for p in _images_in(imgs):
            num = re.match(r"^(\d+)", p.stem)
            key = num.group(1) if num else p.stem
            masks: dict[str, Path] = {}
            if man is not None:
                hit = next((q for q in _images_in(man) if q.stem.startswith(key)), None)
                if hit:
                    masks["vessel"] = hit
            if fov is not None:
                hit = next((q for q in _images_in(fov) if q.stem.startswith(key)), None)
                if hit:
                    masks["fov"] = hit
            out.append(Sample(p, dataset="drive", masks=masks or None,
                              subject_id=key))
    return out


def load_messidor2(root: str | Path) -> list[Sample]:
    """Held-out external test set. Grades come from the adjudicated CSV if present.

    The ADCIS distribution ships images only; the adjudicated reference
    standard (Krause et al. 2018) is a separate CSV with columns
    ``image_id, adjudicated_dr_grade, adjudicated_dme, adjudicated_gradable``.
    Without it the loader still returns images so you can run inference, but
    metrics will be unavailable.
    """
    root = Path(root)
    img_dir = _find_dir(root, r"^IMAGES$", r"^images$") or root

    # Pick the grades CSV by what it CONTAINS, not by what it is called.
    #
    # A Messidor-2 directory typically holds two CSVs whose names both look
    # right: `messidor-2.csv` from ADCIS, which is a left;right eye-pairing
    # table with no grades at all, and `messidor_data.csv` from the adjudicated
    # release (Krause et al. 2018), which has the labels. Matching on filename
    # picked the pairing file and silently produced 1,748 images with no
    # grades -- present, loadable, and unscoreable.
    grades: dict[str, tuple[int | None, int | None, bool | None]] = {}
    csv = None
    for cand in sorted(root.rglob("*.csv")):
        try:
            head = pd.read_csv(cand, nrows=5)
        except Exception:
            continue
        cols = [c.strip().lower() for c in head.columns]
        if any("dr_grade" in c or "retinopathy" in c or "grade" == c for c in cols):
            csv = cand
            break

    if csv is not None:
        df = pd.read_csv(csv)
        df.columns = [c.strip().lower() for c in df.columns]
        id_col = next((c for c in df.columns if "image" in c or "id" in c), df.columns[0])
        g_col = next((c for c in df.columns if "dr_grade" in c or "retinopathy" in c
                      or c == "grade"), None)
        d_col = next((c for c in df.columns if "dme" in c or "edema" in c), None)
        q_col = next((c for c in df.columns if "gradab" in c), None)
        for _, r in df.iterrows():
            key = Path(str(r[id_col])).stem.lower()
            g = None if g_col is None or pd.isna(r[g_col]) else int(r[g_col])
            d = None if d_col is None or pd.isna(r[d_col]) else int(r[d_col])
            q = None if q_col is None or pd.isna(r[q_col]) else bool(int(r[q_col]))
            grades[key] = (g, d, q)

    out = []
    for p in _images_in(img_dir):
        g, d, q = grades.get(p.stem.lower(), (None, None, None))
        out.append(Sample(p, grade=g, dme=d, gradable=q, dataset="messidor2",
                          subject_id=_subject_of(p.name)))
    return out


LOADERS = {
    "aptos2019": load_aptos,
    "idrid_grading": load_idrid_grading,
    "idrid_segmentation": load_idrid_segmentation,
    "drive": load_drive,
    "messidor2": load_messidor2,
}


def discover(data_root: str | Path, strict: bool = True
             ) -> dict[str, list[Sample]]:
    """Auto-detect whichever datasets are present under ``data_root``.

    Each loader is confined to its own named subdirectory
    (``aptos2019/``, ``idrid/``, ``drive/``, ``messidor2/``).

    An earlier version also tried ``data_root`` itself as a last resort, which
    is actively dangerous: with ``messidor2/`` absent, the Messidor-2 loader
    fell back to the whole tree, matched ``drive/test/images`` on its ``^images$``
    pattern, and returned 20 DRIVE images labelled ``dataset="messidor2"``.
    Held-out data silently becoming a different dataset is the exact failure
    the split policy exists to prevent, so the fallback is gone.

    Set ``strict=False`` only for a directory known to contain one dataset.
    """
    data_root = Path(data_root)
    found: dict[str, list[Sample]] = {}
    for name, loader in LOADERS.items():
        base = data_root / name.split("_")[0]
        candidates = [data_root / name, base]
        if not strict:
            candidates.append(data_root)
        for c in candidates:
            if not c.exists():
                continue
            try:
                samples = loader(c)
            except Exception:
                samples = []
            if samples:
                # A loader must only return files from inside its own subtree.
                stray = [s for s in samples if c not in s.image_path.parents]
                if stray:
                    raise SplitViolation(
                        f"loader '{name}' returned {len(stray)} file(s) from "
                        f"outside {c}, e.g. {stray[0].image_path}")
                found[name] = samples
                break
    return found


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
def _hash_frac(key: str, salt: str = "drscreen") -> float:
    h = hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


def group_split(samples: list[Sample], val_frac: float = 0.15,
                salt: str = "drscreen") -> tuple[list[Sample], list[Sample]]:
    """Deterministic, subject-grouped train/val split.

    Hash-based rather than shuffle-based so the split is stable across machines
    and across reruns without carrying an index file around, and grouped by
    subject so fellow eyes never straddle the boundary.
    """
    train, val = [], []
    for s in samples:
        (val if _hash_frac(s.subject_id or s.image_path.stem, salt) < val_frac else train).append(s)
    return train, val


def assert_no_leakage(train: list[Sample], *held_out: list[Sample]) -> None:
    """Fail loudly if a held-out image or subject reached the training pool."""
    bad_ds = sorted({s.dataset for s in train} & HELD_OUT)
    if bad_ds:
        raise SplitViolation(
            f"Held-out dataset(s) {bad_ds} appear in the training pool. "
            "Messidor-2 is reserved for blind external validation.")
    train_keys = {s.image_path.name for s in train} | {s.subject_id for s in train if s.subject_id}
    for pool in held_out:
        overlap = ({s.image_path.name for s in pool} | {s.subject_id for s in pool if s.subject_id}) & train_keys
        if overlap:
            raise SplitViolation(
                f"{len(overlap)} image/subject id(s) appear in both train and a "
                f"held-out set, e.g. {sorted(overlap)[:5]}")


def grade_distribution(samples: list[Sample]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for s in samples:
        if s.grade is None:
            continue
        counts[s.grade] = counts.get(s.grade, 0) + 1
    return dict(sorted(counts.items()))


def class_weights(samples: list[Sample], num_classes: int = 5,
                  scheme: str = "effective") -> np.ndarray:
    """Class weights for the severe imbalance in DR data (grade 0 is ~50%).

    ``effective`` uses the effective-number reweighting of Cui et al. (2019),
    which is markedly better behaved than plain inverse frequency when grade 3
    has only a few dozen examples.
    """
    counts = np.zeros(num_classes, np.float64)
    for s in samples:
        if s.grade is not None and 0 <= s.grade < num_classes:
            counts[s.grade] += 1
    counts = np.maximum(counts, 1.0)
    if scheme == "inverse":
        w = counts.sum() / (num_classes * counts)
    else:
        beta = 0.9999
        eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)
        w = 1.0 / eff
    return (w / w.mean()).astype(np.float32)
