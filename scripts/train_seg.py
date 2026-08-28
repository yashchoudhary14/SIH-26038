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
    a = ap.parse_args()

    train_ds = CohortDataset(a.cohort, "train", size=a.size, train=True, with_masks=True)
    val_ds = CohortDataset(a.cohort, "val", size=a.size, train=False,
                           augment=False, with_masks=True)
    print(f"train {len(train_ds)}  val {len(val_ds)}  size {a.size}")

    model = build_unet("lesion", in_ch=3, width=a.width)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"U-Net: {n_par/1e6:.2f} M parameters, {len(LESION_CLASSES)} lesion channels")

    log = train_segmentation(
        model, train_ds, val_ds, mask_key="lesion_mask",
        epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
        num_workers=a.workers, device=a.device, out_dir=a.out,
        pos_weight=a.pos_weight)

    print(f"\nBest mean Dice {log.best_metric:.4f} at epoch {log.best_epoch} "
          f"({log.elapsed_s/60:.1f} min)")

    ck = torch.load(a.out / "best.pt", map_location="cpu", weights_only=False)
    ck["width"] = a.width
    ck["lesion_classes"] = LESION_CLASSES
    torch.save(ck, a.out / "best.pt")
    print(f"Checkpoint: {a.out/'best.pt'}")


if __name__ == "__main__":
    main()
