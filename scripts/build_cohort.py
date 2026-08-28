"""Materialise a cohort to disk.

Two sources, one output format:

* ``--source synthetic`` generates phantoms (no downloads required).
* ``--source real`` discovers APTOS / IDRiD / DRIVE / Messidor-2 under
  ``--data-root`` and standardises them.

Split policy is enforced here, not left to convention: Messidor-2 always lands
in the ``external`` split and never in ``train`` or ``val``.

Examples
--------
    python scripts/build_cohort.py --source synthetic --n 4000 --out data/cohort_synth
    python scripts/build_cohort.py --source real --data-root data/raw --out data/cohort_real
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from drscreen.constants import NUM_LESION_CLASSES
from drscreen.data.cohort import CohortRecord, CohortWriter
from drscreen.data.synthetic import generate, CAMERAS
from drscreen.preprocess.enhance import adaptive_enhance, to_model_input
from drscreen.preprocess.fov import standardize
from drscreen.preprocess.quality import assess


def _warp_masks(masks: np.ndarray, raw_shape, fov, size: int) -> np.ndarray:
    from drscreen.data.torch_data import _apply_same_geometry
    return _apply_same_geometry(masks, raw_shape, fov, size)


def _make_one(job: tuple) -> tuple:
    """Worker: generate + standardise + assess one phantom.

    Returns picklable arrays; the parent process does all the disk writing so
    the manifest stays a single append-ordered file.
    """
    split, i, size, case_seed, shift, enhance, label_noise = job
    rng = np.random.default_rng(case_seed)
    cams = [c.name for c in (CAMERAS[2:] if shift else CAMERAS)]
    sev = float(np.clip(rng.beta(2.2, 2.0) if shift else rng.beta(1.6, 3.2), 0, 1))
    p = generate(size=size, seed=int(rng.integers(1 << 31)), severity=sev,
                 camera=cams[int(rng.integers(len(cams)))])
    img, fov_mask, fov = standardize(p.image, size=size)

    stack = np.stack(
        [p.vessel_mask, p.disc_mask] +
        [p.lesion_masks[..., c] for c in range(NUM_LESION_CLASSES)], axis=-1)
    stack = _warp_masks(stack, p.image.shape[:2], fov, size)

    q = assess(img, fov_mask, fov)
    if enhance:
        img, applied = adaptive_enhance(img, fov_mask, q.issues)
        img = to_model_input(img, fov_mask, mode="hybrid")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        applied = []

    # Optional reference-standard noise. Real ICDR reference standards are not
    # ground truth: human graders agree exactly only ~60-75% of the time, and
    # almost all disagreement is by one grade. A pipeline validated against
    # noiseless labels looks better than it will ever be in the field -- on the
    # noiseless phantoms the fusion grader reaches AUC 1.00, which tells you
    # nothing except that the task was too easy. Disagreement is modelled as
    # +/-1 grade, matching the observed shape.
    label = p.grade
    if label_noise > 0:
        nrng = np.random.default_rng(case_seed ^ 0x5EED)
        if nrng.random() < label_noise:
            label = int(np.clip(p.grade + nrng.choice([-1, 1]), 0, 4))

    rec = CohortRecord(uid=f"{split}_{i:06d}", grade=label, split=split,
                       source="synthetic", quality_label=p.quality_label,
                       camera=p.camera,
                       meta={"applied": applied, "quality_overall": q.overall,
                             "true_grade": p.grade,
                             "label_noise_applied": bool(label != p.grade),
                             "lesion_counts": p.lesion_counts,
                             "degradations": p.degradations,
                             "domain_shift": shift})
    return rec, img, fov_mask, stack


def build_synthetic(out: Path, n_train: int, n_val: int, n_test: int,
                    n_external: int, size: int, seed: int, enhance: bool,
                    workers: int = 0, label_noise: float = 0.0):
    plan = [("train", n_train, False), ("val", n_val, False),
            ("test", n_test, False), ("external", n_external, True)]
    rng = np.random.default_rng(seed)
    jobs = [(split, i, size, int(rng.integers(1 << 31)), shift, enhance, label_noise)
            for split, n, shift in plan for i in range(n)]
    total = len(jobs)
    t0 = time.time()
    done = 0

    def _write(w, payload):
        rec, img, fov_mask, stack = payload
        w.add(rec, img, fov_mask,
              masks={"vessel": stack[..., 0], "disc": stack[..., 1],
                     "lesions": stack[..., 2:]})

    with CohortWriter(out) as w:
        if workers and workers > 1:
            import multiprocessing as mp
            with mp.Pool(workers) as pool:
                for payload in pool.imap(_make_one, jobs, chunksize=8):
                    _write(w, payload)
                    done += 1
                    if done % 250 == 0:
                        el = time.time() - t0
                        print(f"  {done}/{total}  ({el:.0f}s, {done/el:.1f}/s)", flush=True)
        else:
            for job in jobs:
                _write(w, _make_one(job))
                done += 1
                if done % 250 == 0:
                    el = time.time() - t0
                    print(f"  {done}/{total}  ({el:.0f}s, {done/el:.1f}/s)", flush=True)
    print(f"Wrote {done} cases to {out} in {time.time()-t0:.0f}s")


def build_real(out: Path, data_root: Path, size: int, val_frac: float, enhance: bool):
    from drscreen.data.registry import (discover, group_split, assert_no_leakage,
                                        grade_distribution)
    found = discover(data_root)
    if not found:
        print(f"No datasets discovered under {data_root}.", file=sys.stderr)
        print("Expected subdirectories such as aptos2019/, idrid/, drive/, messidor2/.",
              file=sys.stderr)
        return 1

    print("Discovered:")
    for k, v in found.items():
        dist = grade_distribution(v)
        print(f"  {k:22s} {len(v):6d} images   grades {dist or '(none)'}")

    train_pool: list = []
    for name in ("aptos2019", "idrid_grading"):
        train_pool += found.get(name, [])
    external = found.get("messidor2", [])
    seg_pool = found.get("idrid_segmentation", []) + found.get("drive", [])

    train, val = group_split(train_pool, val_frac)
    assert_no_leakage(train, val, external)
    print(f"\nSplit: train {len(train)}  val {len(val)}  external(messidor2) {len(external)}")
    print("Messidor-2 is held out; it is never used for training or threshold selection.")

    t0 = time.time()
    written = 0
    with CohortWriter(out) as w:
        for split, samples in (("train", train), ("val", val),
                               ("external", external), ("seg", seg_pool)):
            for i, s in enumerate(samples):
                raw = cv2.imread(str(s.image_path), cv2.IMREAD_COLOR)
                if raw is None:
                    continue
                img, fov_mask, fov = standardize(raw, size=size)
                q = assess(img, fov_mask, fov)
                if enhance:
                    img, applied = adaptive_enhance(img, fov_mask, q.issues)
                    img = to_model_input(img, fov_mask, mode="hybrid")
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                else:
                    applied = []

                masks = None
                if s.masks:
                    planes = []
                    for key in ("microaneurysm", "hemorrhage", "hard_exudate",
                                "soft_exudate", "neovascularization"):
                        p = s.masks.get(key)
                        if p is None:
                            planes.append(np.zeros((size, size), np.uint8))
                        else:
                            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                            m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
                            planes.append((m > 127).astype(np.uint8) * 255)
                    lesions = np.stack(planes, axis=-1)
                    vessel = np.zeros((size, size), np.uint8)
                    if "vessel" in s.masks:
                        m = cv2.imread(str(s.masks["vessel"]), cv2.IMREAD_GRAYSCALE)
                        vessel = (cv2.resize(m, (size, size),
                                             interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8) * 255
                    disc = np.zeros((size, size), np.uint8)
                    if "optic_disc" in s.masks:
                        m = cv2.imread(str(s.masks["optic_disc"]), cv2.IMREAD_GRAYSCALE)
                        disc = (cv2.resize(m, (size, size),
                                           interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8) * 255
                    masks = {"vessel": vessel, "disc": disc, "lesions": lesions}

                uid = f"{s.dataset}_{split}_{i:06d}"
                rec = CohortRecord(uid=uid, grade=s.grade if s.grade is not None else -1,
                                   split=split, source=s.dataset,
                                   quality_label=2 if s.gradable is not False else 0,
                                   meta={"applied": applied, "quality_overall": q.overall,
                                         "original": str(s.image_path),
                                         "subject": s.subject_id})
                w.add(rec, img, fov_mask, masks)
                written += 1
                if written % 200 == 0:
                    print(f"  {written} written ({time.time()-t0:.0f}s)", flush=True)
    print(f"Wrote {written} cases to {out} in {time.time()-t0:.0f}s")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("data/raw"))
    ap.add_argument("--n", type=int, default=4000, help="total synthetic cases")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--label-noise", type=float, default=0.0,
                    help="probability a case gets a +/-1 grade label error, "
                         "modelling real inter-grader disagreement (try 0.25)")
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel generation workers (0/1 = serial)")
    ap.add_argument("--no-enhance", action="store_true",
                    help="store standardised BGR instead of the hybrid model input")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    if a.source == "synthetic":
        n = a.n
        build_synthetic(a.out, int(n * 0.62), int(n * 0.13), int(n * 0.13),
                        int(n * 0.12), a.size, a.seed, not a.no_enhance,
                        workers=a.workers, label_noise=a.label_noise)
        return 0
    return build_real(a.out, a.data_root, a.size, a.val_frac, not a.no_enhance)


if __name__ == "__main__":
    raise SystemExit(main())
