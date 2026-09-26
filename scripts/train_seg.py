"""Stage 1: train the lesion segmentation network.

    python scripts/train_seg.py --cohort data/cohort_synth --epochs 12
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from drscreen.constants import LESION_CLASSES
from drscreen.data.cohort import CohortDataset
from drscreen.models.segmentation import build_unet
from drscreen.training import train_segmentation


def supervised_channels(cohort: Path, ds, num_classes: int) -> np.ndarray:
    """Which lesion channels carry at least one annotated pixel in this split.

    Read straight from the stored masks rather than by iterating the dataset,
    so it costs no decoding or augmentation and reflects the ground truth as
    written to disc.

    This exists because the failure it detects is silent. IDRiD ships no
    neovascularisation masks, so ``build_cohort`` fills that plane with zeros
    and the channel trains against an all-zero target: it converges to "never
    fire", scores an undefined Dice, and then reports zero NV on every image
    for the life of the model. Nothing crashes and no metric goes red.
    """
    present = np.zeros(num_classes, bool)
    for r in ds.records:
        if not getattr(r, "has_masks", False):
            continue
        z = np.load(cohort / "masks" / f"{r.uid}.npz")
        les = z["lesions"]
        present |= les.reshape(-1, les.shape[-1]).max(axis=0) > 0
        if present.all():
            break
    return present


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/segmentation"))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--pos-weight", type=float, default=8.0,
                    help="BCE positive weight; lesions occupy <1% of pixels")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--train-split", default=None,
                    help="cohort split to train on (default: auto-detect)")
    ap.add_argument("--val-split", default=None)
    a = ap.parse_args()

    # Real cohorts keep lesion segmentation on its own splits, because the
    # grading corpora (APTOS) carry no masks and DRIVE carries vessel masks
    # but no lesion annotation. Synthetic cohorts have masks on every split.
    train_split, val_split = a.train_split, a.val_split
    if train_split is None:
        from drscreen.data.cohort import read_manifest
        available = {r.split for r in read_manifest(a.cohort)}
        if "seg_train" in available:
            train_split, val_split = "seg_train", "seg_val"
            print("Using the lesion-segmentation splits (seg_train/seg_val).")
        else:
            train_split, val_split = "train", "val"

    train_ds = CohortDataset(a.cohort, train_split, size=a.size, train=True, with_masks=True)
    val_ds = CohortDataset(a.cohort, val_split, size=a.size, train=False,
                           augment=False, with_masks=True)
    print(f"train {len(train_ds)}  val {len(val_ds)}  size {a.size}")

    model = build_unet("lesion", in_ch=3, width=a.width)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"U-Net: {n_par/1e6:.2f} M parameters, {len(LESION_CLASSES)} lesion channels")

    sup = supervised_channels(a.cohort, train_ds, len(LESION_CLASSES))
    supervised = [c for c, s in zip(LESION_CLASSES, sup) if s]
    missing = [c for c, s in zip(LESION_CLASSES, sup) if not s]
    print(f"pixel-annotated channels: {', '.join(supervised) or '(none)'}")
    if missing:
        print(f"  !! NO annotation in this cohort for: {', '.join(missing)}")
        print("     excluded from the loss; the pipeline will report them as")
        print("     'not assessed' rather than as a negative finding.")
    if not sup.any():
        raise SystemExit("No lesion channel has any annotated pixel -- check the cohort.")

    log = train_segmentation(
        model, train_ds, val_ds, mask_key="lesion_mask",
        epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
        num_workers=a.workers, device=a.device, out_dir=a.out,
        pos_weight=a.pos_weight, channel_mask=sup)

    print(f"\nBest mean Dice {log.best_metric:.4f} at epoch {log.best_epoch} "
          f"({log.elapsed_s/60:.1f} min)")

    ck = torch.load(a.out / "best.pt", map_location="cpu", weights_only=False)
    ck["width"] = a.width
    ck["lesion_classes"] = LESION_CLASSES
    # Which of those channels were actually trained. The pipeline reads this to
    # separate "looked and found nothing" from "never looked".
    ck["supervised_lesion_classes"] = supervised
    # The resolution matters at inference: a model trained at 1024 run at 512
    # produces different lesion counts, so the clinical features drift away
    # from the ones the grader was fitted on.
    ck["size"] = a.size
    ck["train_split"] = train_split
    torch.save(ck, a.out / "best.pt")
    print(f"Checkpoint: {a.out/'best.pt'}")


if __name__ == "__main__":
    main()
