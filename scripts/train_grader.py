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


#: Rebalancing is split evenly between the sampler and the loss weights: each
#: applies the square root of the inverse class frequency, so the two compose
#: to full balance rather than compounding into a ~30x over-weighting of the
#: rarest grade. With only a couple of hundred grade-3 images, full balance in
#: both places would trade the current under-fitting of grades 3-4 for
#: straightforward overfitting on repeated samples.
REBALANCE_TEMPERATURE = 0.5


def class_weights_from(ds: CohortDataset, num_classes: int = 5,
                       temperature: float = REBALANCE_TEMPERATURE) -> np.ndarray:
    """Effective-number reweighting (Cui et al. 2019) over the observed grades.

    ``temperature`` tempers the weights: 1.0 is the full inverse-effective-
    number weight, 0.5 its square root. See ``REBALANCE_TEMPERATURE``.
    """
    counts = np.bincount(ds.grades(), minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    beta = 0.9999
    eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    w = np.power(1.0 / eff, temperature)
    return (w / w.mean()).astype(np.float32)


def balanced_sampler(ds: CohortDataset, num_classes: int = 5,
                     temperature: float = REBALANCE_TEMPERATURE):
    """Grade-stratified sampler that keeps the deep CORN conditionals fed.

    CORN trains task *k* only on the subset with ``y > k-1``. Under natural
    sampling, task 3 (grade 4 vs grade 3) sees grades 3-4 only -- about 17% of
    the cohort, i.e. two or three images in a batch of 16 -- so the gradient on
    z3 is dominated by sampling noise, and z3 gates the highest grade through
    a product of four sigmoids. Square-root stratification lifts grades 3-4 to
    roughly 28% of a batch without the heavy repetition of full balancing.
    """
    from torch.utils.data import WeightedRandomSampler
    g = np.asarray(ds.grades())
    counts = np.maximum(np.bincount(g, minlength=num_classes).astype(np.float64), 1.0)
    w = np.power(1.0 / counts, temperature)[g]
    share = np.bincount(g, weights=w, minlength=num_classes) / w.sum()
    print("expected batch composition by grade: "
          + "  ".join(f"{i}:{s:.1%}" for i, s in enumerate(share)))
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                 num_samples=len(g), replacement=True)


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
    ap.add_argument("--select-on", choices=["qwk", "referable_auc", "composite"],
                    default="qwk",
                    help="checkpoint-selection metric. Referable AUC reads only "
                         "z0 and z1 and cannot see grade 3/4 collapse; QWK can.")
    ap.add_argument("--clinical-dropout", type=float, default=0.5,
                    help="probability of zeroing the whole clinical branch per "
                         "sample during training (fusion arm only). Stops the "
                         "image backbone under-training behind an easier "
                         "signal; 0.0 restores the old co-adapting behaviour.")
    ap.add_argument("--no-balanced-sampler", action="store_true",
                    help="sample grades at their natural frequency (the old behaviour)")
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
        # No modality dropout here: with the image pathway zeroed there is no
        # second modality to fall back on, so dropping the clinical branch would
        # leave the model nothing at all to grade from.
        model = DRGrader(backbone="resnet18", pretrained=False,
                         use_clinical=True, use_image=False,
                         clinical_dropout=0.0)
        for prm in model.backbone.parameters():
            prm.requires_grad_(False)
    else:
        model = DRGrader(backbone=a.backbone, pretrained=not a.no_pretrained,
                         use_clinical=use_clinical,
                         clinical_dropout=a.clinical_dropout)

    print(f"parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
    print(f"clinical feature vector: {ClinicalFeatures.vector_size()} dims")

    sampler = None if a.no_balanced_sampler else balanced_sampler(train_ds)
    log = train_grader(
        model, train_ds, val_ds, epochs=a.epochs, batch_size=a.batch_size,
        lr=a.lr, num_workers=a.workers, device=a.device, out_dir=out,
        class_weights=class_weights_from(train_ds),
        clinical_fn=clinical_from_batch if use_clinical else None,
        sampler=sampler, select_on=a.select_on)

    best = next((e for e in log.epochs if e["epoch"] == log.best_epoch), {})
    print(f"\nBest {a.select_on} {log.best_metric:.4f} at epoch {log.best_epoch} "
          f"({log.elapsed_s/60:.1f} min)")
    print(f"  at that epoch: AUC(ref) {best.get('val_auc_referable', float('nan')):.4f}  "
          f"QWK {best.get('val_qwk', float('nan')):.4f}  "
          f"sens(g>=3) {best.get('val_sens_sight_threatening', float('nan')):.4f}")
    print(f"  per-grade recall: "
          + "  ".join(f"{i}:{v:.2f}" for i, v in
                      enumerate(best.get("val_recall_per_grade", []))))

    ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    ck.update({"use_clinical": use_clinical,
               "backbone": a.backbone if a.arm != "clinical" else "resnet18",
               "use_image": a.arm != "clinical",
               "arm": a.arm, "size": a.size})
    torch.save(ck, out / "best.pt")
    (out / "arm.json").write_text(json.dumps(
        {"arm": a.arm, "use_clinical": use_clinical,
         "clinical_dropout": (0.0 if a.arm == "clinical" else a.clinical_dropout),
         "best_metric": log.best_metric, "best_epoch": log.best_epoch,
         "select_on": a.select_on,
         "balanced_sampler": not a.no_balanced_sampler}, indent=2))
    print(f"Checkpoint: {out/'best.pt'}")


if __name__ == "__main__":
    main()
