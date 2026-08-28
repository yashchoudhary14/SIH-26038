"""Stage 3: train the ordinal DR severity grader.

Supports the ablation arms the problem statement requires, selected with
``--arm``:

============  =========================================================
arm           what it uses
============  =========================================================
``fusion``    CNN image features + clinical lesion features (the system)
``cnn``       CNN image features only (single technique: deep learning)
``clinical``  clinical lesion features only, no CNN (single technique:
              classical CV + segmentation)
============  =========================================================

    python scripts/train_grader.py --cohort data/cohort_synth --arm fusion
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from drscreen.data.cohort import CohortDataset, clinical_from_batch
from drscreen.models.grader import DRGrader
from drscreen.models.lesion_features import ClinicalFeatures
from drscreen.training import train_grader


def class_weights_from(ds: CohortDataset, num_classes: int = 5) -> np.ndarray:
    """Effective-number reweighting (Cui et al. 2019) over the observed grades."""
    counts = np.bincount(ds.grades(), minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    beta = 0.9999
    eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    w = 1.0 / eff
    return (w / w.mean()).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--arm", choices=["fusion", "cnn", "clinical"], default="fusion")
    ap.add_argument("--backbone", default="tf_efficientnet_b0")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--feature-mode", choices=["predicted", "gt"], default="predicted")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    out = a.out or Path(f"outputs/grader_{a.arm}")
    use_clinical = a.arm in ("fusion", "clinical")

    train_ds = CohortDataset(a.cohort, "train", size=a.size, train=True,
                             with_features=use_clinical, feature_mode=a.feature_mode)
    val_ds = CohortDataset(a.cohort, "val", size=a.size, train=False, augment=False,
                           with_features=use_clinical, feature_mode=a.feature_mode)
    print(f"arm={a.arm}  train {len(train_ds)}  val {len(val_ds)}")
    print(f"grade distribution (train): {np.bincount(train_ds.grades(), minlength=5).tolist()}")

    if a.arm == "clinical":
        # No image pathway: a tiny backbone is kept only so the tensor plumbing
        # is identical across arms, and its embedding is zeroed, so the model
        # must grade from the clinical vector alone -- an honest
        # single-technique baseline rather than a crippled fusion model.
        model = DRGrader(backbone="resnet18", pretrained=False,
                         use_clinical=True, use_image=False)
        for prm in model.backbone.parameters():
            prm.requires_grad_(False)
    else:
        model = DRGrader(backbone=a.backbone, pretrained=not a.no_pretrained,
                         use_clinical=use_clinical)

    print(f"parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
    print(f"clinical feature vector: {ClinicalFeatures.vector_size()} dims")

    log = train_grader(
        model, train_ds, val_ds, epochs=a.epochs, batch_size=a.batch_size,
        lr=a.lr, num_workers=a.workers, device=a.device, out_dir=out,
        class_weights=class_weights_from(train_ds),
        clinical_fn=clinical_from_batch if use_clinical else None)

    print(f"\nBest referable AUC {log.best_metric:.4f} at epoch {log.best_epoch} "
          f"({log.elapsed_s/60:.1f} min)")

    ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    ck.update({"use_clinical": use_clinical,
               "backbone": a.backbone if a.arm != "clinical" else "resnet18",
               "use_image": a.arm != "clinical",
               "arm": a.arm, "size": a.size})
    torch.save(ck, out / "best.pt")
    (out / "arm.json").write_text(json.dumps(
        {"arm": a.arm, "use_clinical": use_clinical,
         "best_metric": log.best_metric, "best_epoch": log.best_epoch}, indent=2))
    print(f"Checkpoint: {out/'best.pt'}")


if __name__ == "__main__":
    main()
