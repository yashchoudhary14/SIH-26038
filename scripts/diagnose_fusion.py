"""Is the fusion arm's image backbone under-trained behind its clinical branch?

Multimodal co-adaptation is invisible in a headline metric. A fusion model whose
backbone has learned far less than it should still scores well, because the
easier modality carries it -- and it only shows up as a *relative* loss against
the single-modality arm, which reads like a modelling trade-off rather than a
defect.

The measurement that separates them is to run the trained fusion model with its
clinical branch zeroed and compare that against the image-only arm trained on
the same images:

* image-pathway-alone close to the ``cnn`` arm  -> the backbone learned properly
  and the clinical branch is additive;
* image-pathway-alone far below it -> the backbone under-trained behind the
  clinical shortcut, and the fused score is propped up by the crutch.

This is what found the defect fixed by ``--clinical-dropout``: on the DDR cohort
the fusion backbone scored 0.8638 against the image-only arm's 0.9562, and
modality dropout restored it to 0.9515.

Usage::

    python scripts/diagnose_fusion.py --cohort data/cohort_real \\
        --fusion outputs/grader_fusion/best.pt --cnn outputs/grader_cnn/best.pt

The features cached in the cohort must be the ones the fusion arm was trained
on, or the comparison measures a feature change rather than the backbone.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "src")

from drscreen.constants import REFERABLE_THRESHOLD
from drscreen.data.cohort import CohortDataset, clinical_from_batch
from drscreen.models.grader import DRGrader, referable_prob


def _auc(y_true: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, score))


def _load(ckpt: Path, use_clinical: bool) -> DRGrader:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = DRGrader(backbone=ck.get("backbone", "tf_efficientnet_b0"),
                 pretrained=False, use_clinical=use_clinical,
                 clinical_dim=ck.get("clinical_dim"))
    m.load_state_dict(ck["model"])
    return m.eval(), ck


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--fusion", type=Path, required=True)
    ap.add_argument("--cnn", type=Path, default=None,
                    help="image-only arm to compare against; the reference for "
                         "what this backbone should have reached")
    ap.add_argument("--split", default="val",
                    help="val by default: test is for reporting, not diagnosis")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    dev = torch.device(a.device)
    ds = CohortDataset(a.cohort, a.split, size=a.size, train=False,
                       augment=False, with_features=True)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                    num_workers=a.workers)
    print(f"{a.cohort.name}/{a.split}: {len(ds)} images")

    fusion, fck = _load(a.fusion, use_clinical=True)
    fusion.to(dev)
    # Read the rate the checkpoint was TRAINED with, never the live attribute:
    # the constructor default is now 0.5, so a rebuilt model reports 0.5 for a
    # checkpoint trained long before the flag existed.
    trained_rate = fck.get("clinical_dropout")
    rate_str = ("not recorded (predates --clinical-dropout, so trained at 0.0)"
                if trained_rate is None else f"{trained_rate}")
    print(f"fusion: {a.fusion}  epoch {fck.get('epoch')}")
    print(f"  trained with clinical_dropout = {rate_str}")

    full, imgonly, gates, labels = [], [], [], []
    with torch.no_grad():
        for b in dl:
            img = b["image"].to(dev)
            clin = clinical_from_batch(b).to(dev)
            z = fusion.features(img)
            g = fusion.gate(z)
            c = fusion.clinical_encoder(clin) * g
            full.append(fusion.head(torch.cat([z, c], 1)).cpu())
            imgonly.append(fusion.head(torch.cat([z, torch.zeros_like(c)], 1)).cpu())
            gates.append(g.squeeze(1).cpu())
            labels.append(b["grade"])

    y = torch.cat(labels).numpy()
    keep = y >= 0
    yb = (y[keep] >= REFERABLE_THRESHOLD).astype(int)
    auc_full = _auc(yb, referable_prob(torch.cat(full)).numpy()[keep])
    auc_img = _auc(yb, referable_prob(torch.cat(imgonly)).numpy()[keep])
    g = torch.cat(gates).numpy()

    auc_cnn = None
    if a.cnn is not None:
        cnn, _ = _load(a.cnn, use_clinical=False)
        cnn.to(dev)
        out = []
        with torch.no_grad():
            for b in dl:
                out.append(cnn(b["image"].to(dev)).cpu())
        auc_cnn = _auc(yb, referable_prob(torch.cat(out)).numpy()[keep])

    print()
    print("referable AUC")
    print(f"  fusion, as trained             {auc_full:.4f}")
    print(f"  fusion, clinical branch zeroed {auc_img:.4f}   "
          f"(the backbone on its own)")
    if auc_cnn is not None:
        print(f"  image-only arm                 {auc_cnn:.4f}   (reference)")
    print()
    print(f"  clinical branch contributes    {auc_full - auc_img:+.4f}")
    if auc_cnn is not None:
        deficit = auc_cnn - auc_img
        print(f"  backbone deficit vs reference  {deficit:+.4f}")
        print()
        if deficit > 0.02:
            print("  VERDICT: co-adapted. The backbone has learned materially less")
            print("  than the same architecture trained without a clinical branch to")
            print("  lean on. Retrain with --clinical-dropout > 0.")
        else:
            print("  VERDICT: healthy. The backbone matches the image-only arm, so")
            print("  the clinical branch is additive rather than a substitute.")
    print()
    print(f"gate: min {g.min():.3f}  median {np.median(g):.3f}  "
          f"max {g.max():.3f}  std {g.std():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
