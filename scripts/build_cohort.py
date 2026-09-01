"""Materialise a cohort to disk.

Two sources, one output format:

* ``--source synthetic`` generates phantoms (no downloads required).
* ``--source real`` discovers APTOS / EyePACS / DDR / IDRiD / DRIVE /
  Messidor-2 under ``--data-root`` and standardises them.

``--curate`` down-samples over-represented grades in the **training split
only**, mixing sources within each grade so the model cannot shortcut on the
imaging chain. Val and test keep the natural distribution.

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
        # NO colour conversion here. to_model_input already returns the exact
        # 3-channel representation the model consumes -- [CLAHE-green,
        # Ben-Graham, L*], feature planes, not an RGB image. The previous
        # cv2.COLOR_RGB2BGR reversed channels 0 and 2 before cv2.imwrite, and
        # since imwrite/imread round-trips an array unchanged, the cohort stored
        # [L*, Ben-Graham, CLAHE-green] -- the reverse of what the live pipeline
        # (pipeline.py, no conversion) feeds the same model. Training and
        # deployment saw mirror-image inputs. See test_train_serve_channel_parity.
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


def _process_real(job: tuple):
    """Worker: read one real image, standardise, assess, enhance, load masks.

    Returns picklable arrays; the parent does all the disk writing so the
    manifest stays a single append-ordered file.
    """
    s_path, s_dataset, s_grade, s_gradable, s_subject, s_masks, split, i, size, enhance = job
    raw = cv2.imread(str(s_path), cv2.IMREAD_COLOR)
    if raw is None:
        return None
    img, fov_mask, fov = standardize(raw, size=size)
    q = assess(img, fov_mask, fov)
    if enhance:
        img, applied = adaptive_enhance(img, fov_mask, q.issues)
        img = to_model_input(img, fov_mask, mode="hybrid")
        # No colour conversion: store exactly what the live pipeline feeds the
        # model. See the note in the synthetic path and the parity test.
    else:
        applied = []

    # Masks must go through the SAME geometry as the image.
    #
    # `standardize` crops to the FOV bounding box, square-pads, then resizes.
    # Resizing a mask straight from source to (size, size) skips the crop and
    # the pad, so every annotation lands in the wrong place -- off by the crop
    # offset and stretched by the aspect-ratio correction. The model then
    # learns to associate lesions with whatever happens to sit at the shifted
    # coordinates, which is worse than no supervision because it trains
    # confidently on noise.
    #
    # Foreground encoding is also not standardised across corpora: IDRiD ships
    # palette TIFFs whose foreground value is 76, DRIVE uses 255. Any non-zero
    # pixel is foreground; thresholding at >127 silently empties IDRiD.
    masks = None
    if s_masks:
        def _load(path) -> np.ndarray | None:
            if path is None:
                return None
            m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            return None if m is None else (m > 0).astype(np.uint8) * 255

        raw_h, raw_w = raw.shape[:2]
        keys = ("microaneurysm", "hemorrhage", "hard_exudate",
                "soft_exudate", "neovascularization")
        planes = [_load(s_masks.get(k)) for k in keys]
        vessel_raw = _load(s_masks.get("vessel"))
        disc_raw = _load(s_masks.get("optic_disc"))

        blank = np.zeros((raw_h, raw_w), np.uint8)
        stack_raw = np.stack(
            [vessel_raw if vessel_raw is not None else blank,
             disc_raw if disc_raw is not None else blank] +
            [p if p is not None else blank for p in planes], axis=-1)

        # Some corpora annotate at a different resolution than the image.
        if stack_raw.shape[:2] != (raw_h, raw_w):
            stack_raw = cv2.resize(stack_raw, (raw_w, raw_h),
                                   interpolation=cv2.INTER_NEAREST)

        warped = _warp_masks(stack_raw, (raw_h, raw_w), fov, size)
        masks = {"vessel": warped[..., 0], "disc": warped[..., 1],
                 "lesions": warped[..., 2:]}

    rec = CohortRecord(uid=f"{s_dataset}_{split}_{i:06d}",
                       grade=s_grade if s_grade is not None else -1,
                       split=split, source=s_dataset,
                       quality_label=2 if s_gradable is not False else 0,
                       meta={"applied": applied, "quality_overall": q.overall,
                             "original": str(s_path), "subject": s_subject})
    return rec, img, fov_mask, masks


def build_real(out: Path, data_root: Path, size: int, val_frac: float,
               enhance: bool, workers: int = 0,
               only_splits: set[str] | None = None,
               curate: bool = False, curate_cap: int = 0):
    from drscreen.data.registry import (discover, group_split, assert_no_leakage,
                                        grade_distribution, curate_training_pool,
                                        format_grade_source_matrix)
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

    # --- grading pool: APTOS + IDRiD disease grading -----------------------
    train_pool: list = []
    for name in ("aptos2019", "eyepacs", "ddr", "idrid_grading"):
        train_pool += found.get(name, [])
    # DDR marks ungradable images as grade 5, which its loader returns as
    # grade=None rather than as a sixth class. Anything without a grade is
    # unusable for ordinal training and would break the class weights, so it
    # is dropped here explicitly rather than silently surviving into a split.
    n_ungradable = sum(1 for x in train_pool if x.grade is None)
    if n_ungradable:
        train_pool = [x for x in train_pool if x.grade is not None]
        print(f"  dropped {n_ungradable} ungradable images from the grading pool")
    external = found.get("messidor2", [])

    # --- segmentation pool, kept on its own splits -------------------------
    # These are different tasks over different corpora and they must not be
    # mixed. IDRiD segmentation carries pixel lesion masks; DRIVE carries
    # vessel masks and no lesion annotation at all. Pooling them would feed
    # the lesion head 40 images labelled "no lesions anywhere", teaching it
    # that a whole imaging domain is disease-free. They also cannot live on
    # the grading splits, because APTOS has no masks and the segmentation
    # trainer would fail on the missing key.
    seg_pool = found.get("idrid_segmentation", [])
    vessel_pool = found.get("drive", [])
    seg_train, seg_val = group_split(seg_pool, max(val_frac, 0.2))
    ves_train, ves_val = group_split(vessel_pool, max(val_frac, 0.2))

    # Three-way, not two. `val` fits the temperature and selects the referral
    # threshold; `test` is the internal held-out estimate. Sharing one split
    # between those two jobs is the single most common way DR results get
    # inflated, so the cohort makes it structurally impossible rather than
    # relying on the trainer to remember.
    rest, test = group_split(train_pool, val_frac, salt="drscreen-test")
    train, val = group_split(rest, val_frac / max(1.0 - val_frac, 1e-6))

    # Curation applies to TRAIN ONLY, and only after the split. Val selects the
    # referral threshold against a target sensitivity and test is the reported
    # estimate, so both must keep the population's real prevalence: a curated
    # val fits a threshold for a distribution that does not exist, and a
    # curated test reports PPV for one. The training distribution is a
    # modelling choice; the evaluation distribution is a measurement.
    if curate:
        before = len(train)
        print('\nTraining pool before curation:')
        print(format_grade_source_matrix(train))
        train = curate_training_pool(train, cap_per_grade=curate_cap, seed=0)
        print(f'\nTraining pool after curation ({before} -> {len(train)}):')
        print(format_grade_source_matrix(train))
    elif len({x.dataset for x in train}) > 2:
        print('\nTraining pool by grade and source:')
        print(format_grade_source_matrix(train))

    assert_no_leakage(train, val, test, external)
    print(f"\nGrading split : train {len(train)}  val {len(val)}  "
          f"test {len(test)}  external/messidor2 {len(external)}")
    print(f"Lesion  split : train {len(seg_train)}  val {len(seg_val)}   (IDRiD segmentation)")
    print(f"Vessel  split : train {len(ves_train)}  val {len(ves_val)}   (DRIVE)")
    print("Messidor-2 is held out; it is never used for training or threshold selection.")

    t0 = time.time()
    written = 0
    with CohortWriter(out) as w:
        plan = (("train", train), ("val", val), ("test", test),
                ("external", external),
                ("seg_train", seg_train), ("seg_val", seg_val),
                ("vessel_train", ves_train), ("vessel_val", ves_val))
        if only_splits:
            plan = tuple(x for x in plan if x[0] in only_splits)
            print(f"Restricted to splits: {sorted(only_splits)}")

        jobs = []
        for split, samples in plan:
            for i, sm in enumerate(samples):
                jobs.append((sm.image_path, sm.dataset, sm.grade, sm.gradable,
                             sm.subject_id, sm.masks, split, i, size, enhance))
        total = len(jobs)
        print(f"\nProcessing {total} images with {max(1, workers)} worker(s)...")

        def _write(payload):
            if payload is None:
                return 0
            rec, img, fov_mask, masks = payload
            w.add(rec, img, fov_mask, masks)
            return 1

        if workers and workers > 1:
            import multiprocessing as mp
            with mp.Pool(workers) as pool:
                for payload in pool.imap(_process_real, jobs, chunksize=4):
                    written += _write(payload)
                    if written % 200 == 0:
                        el = time.time() - t0
                        print(f"  {written}/{total}  ({el:.0f}s, {written/max(el,1):.1f}/s)",
                              flush=True)
        else:
            for job in jobs:
                written += _write(_process_real(job))
                if written % 200 == 0:
                    el = time.time() - t0
                    print(f"  {written}/{total}  ({el:.0f}s, {written/max(el,1):.1f}/s)",
                          flush=True)

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
    ap.add_argument("--only-splits", nargs="*", default=None,
                    help="build only these splits, e.g. seg_train seg_val. Lets the "
                         "lesion model use a higher resolution than the grading "
                         "cohort without materialising every image twice.")
    ap.add_argument("--label-noise", type=float, default=0.0,
                    help="probability a case gets a +/-1 grade label error, "
                         "modelling real inter-grader disagreement (try 0.25)")
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel generation workers (0/1 = serial)")
    ap.add_argument("--no-enhance", action="store_true",
                    help="store standardised BGR instead of the hybrid model input")
    ap.add_argument("--curate", action="store_true",
                    help="down-sample over-represented grades in the TRAINING "
                         "split only, mixing sources within each grade. Val and "
                         "test keep the natural distribution.")
    ap.add_argument("--curate-cap", type=int, default=0,
                    help="per-grade cap for --curate. 0 (default) resolves to "
                         "twice the largest of grades 3 and 4. Grades 3 and 4 "
                         "are never down-sampled.")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    if a.source == "synthetic":
        n = a.n
        build_synthetic(a.out, int(n * 0.62), int(n * 0.13), int(n * 0.13),
                        int(n * 0.12), a.size, a.seed, not a.no_enhance,
                        workers=a.workers, label_noise=a.label_noise)
        return 0
    return build_real(a.out, a.data_root, a.size, a.val_frac,
                      not a.no_enhance, workers=a.workers,
                      only_splits=set(a.only_splits) if a.only_splits else None,
                      curate=a.curate, curate_cap=a.curate_cap)


if __name__ == "__main__":
    raise SystemExit(main())
