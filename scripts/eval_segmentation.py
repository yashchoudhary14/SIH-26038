"""Score segmentation checkpoints on a cohort split, broken down by source.

Two things this exists to get right.

**Source breakdown.** The IDRiD-only baseline was measured on 17 IDRiD images;
the combined cohort's val is 17 IDRiD + 108 DDR. A mixed-val mean compared
against that baseline would credit the model for a different eval set rather
than for better segmentation. The IDRiD subset is the only comparable number,
so it is reported separately.

**Aggregation convention.** ``training.py`` sums over dims (0, 2, 3), pooling
pixels across the batch before dividing -- a micro-Dice, where images with many
lesion pixels dominate. Averaging per-image Dice instead (macro) weights every
image equally and gives materially different numbers on the same model: 0.484
vs 0.539 for haemorrhage on the baseline checkpoint. Neither is wrong, but
comparing one against the other is. Both are reported, and every checkpoint is
scored under both, so a comparison is never accidentally cross-convention.

Dice is undefined where the ground truth has no positives, so those cases are
NaN and excluded rather than scored as 1.0 -- the convention that once reported
"mean Dice 1.0000" for a model that had learned nothing from masks that were
entirely empty.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")

from drscreen.constants import LESION_CLASSES
from drscreen.data.cohort import CohortDataset
from drscreen.models.segmentation import build_unet

EPS = 1e-6


def _fmt_row(label: str, vals: np.ndarray, width: int = 24) -> str:
    line = "    " + label.ljust(width)
    for v in vals:
        line += ("      n/a" if np.isnan(v) else f"{v:.4f}").rjust(11)
    with np.errstate(invalid="ignore"):
        m = np.nanmean(vals)
    line += ("n/a" if np.isnan(m) else f"{m:.4f}").rjust(9)
    return line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, nargs="+", required=True)
    ap.add_argument("--split", default="seg_val")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    ds = CohortDataset(a.cohort, a.split, size=a.size, train=False,
                       augment=False, with_masks=True)
    sources = [r.source for r in ds.records]
    counts = {s: sources.count(s) for s in sorted(set(sources))}
    print(f"{a.cohort.name}/{a.split}: {len(ds)} images  {counts}")

    dev = torch.device(a.device)
    n_cls = len(LESION_CLASSES)

    for ckpt_path in a.ckpt:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = build_unet("lesion", width=int(ck.get("width", 24)))
        model.load_state_dict(ck["model"])
        model.eval().to(dev)

        # macro: per-image Dice, averaged.  micro: pooled intersection/denominator.
        macro = defaultdict(list)
        inter = defaultdict(lambda: np.zeros(n_cls))
        denom = defaultdict(lambda: np.zeros(n_cls))
        haspos = defaultdict(lambda: np.zeros(n_cls, bool))

        with torch.no_grad():
            for i in range(len(ds)):
                item = ds[i]
                src = sources[i]
                logits = model(item["image"][None].to(dev))
                if isinstance(logits, tuple):
                    logits = logits[0]
                p = (torch.sigmoid(logits.float())[0].cpu().numpy()
                     > a.threshold).astype(np.float64)
                t = item["lesion_mask"].numpy().astype(np.float64)

                per_img = np.full(n_cls, np.nan)
                for c in range(n_cls):
                    tc, pc = t[c], p[c]
                    inter[src][c] += (pc * tc).sum()
                    denom[src][c] += pc.sum() + tc.sum()
                    if tc.sum() > 0:
                        haspos[src][c] = True
                        per_img[c] = (2 * (pc * tc).sum() + EPS) / (pc.sum() + tc.sum() + EPS)
                macro[src].append(per_img)

        print()
        print(f"=== {ckpt_path}  (epoch {ck.get('epoch')}, trained-at {ck.get('size')}) ===")
        print(f"    supervised: {ck.get('supervised_lesion_classes')}")
        print("    " + "".ljust(24) + "".join(c[:9].rjust(11) for c in LESION_CLASSES)
              + "mean".rjust(9))

        for tag, srcs in (("", sorted(macro)), ("ALL", [None])):
            for src in srcs:
                if src is None:
                    if len(macro) < 2:
                        continue
                    rows = np.concatenate([np.array(macro[s]) for s in macro], axis=0)
                    I = sum(inter[s] for s in inter)
                    D = sum(denom[s] for s in denom)
                    H = np.any([haspos[s] for s in haspos], axis=0)
                    label = f"ALL (n={len(rows)})"
                else:
                    rows = np.array(macro[src])
                    I, D, H = inter[src], denom[src], haspos[src]
                    label = f"{src} (n={len(rows)})"

                with np.errstate(invalid="ignore"):
                    mac = np.nanmean(rows, axis=0)
                mic = np.where(H, (2 * I + EPS) / (D + EPS), np.nan)
                print(_fmt_row(label + "  macro", mac))
                print(_fmt_row(" " * len(label) + "  micro", mic))
                print("    " + "  images with positives".ljust(24)
                      + "".join(str(int(k)).rjust(11)
                                for k in (~np.isnan(rows)).sum(axis=0)))


if __name__ == "__main__":
    main()
