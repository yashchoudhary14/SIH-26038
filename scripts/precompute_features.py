"""Stage 2: run segmentation over a cohort and cache clinical feature vectors.

Writes ``<cohort>/features/<uid>.npy`` (from *predicted* masks -- what runs at
inference) and optionally ``<cohort>/features_gt/<uid>.npy`` (from ground
truth), which exists solely to quantify how much the fusion grader loses to
segmentation error rather than to its own limitations.

    python scripts/precompute_features.py --cohort data/cohort_synth \
        --seg outputs/segmentation/best.pt
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from drscreen.constants import NUM_LESION_CLASSES
from drscreen.data.cohort import read_manifest
from drscreen.data.torch_data import to_tensor
from drscreen.models.lesion_features import extract
from drscreen.models.segmentation import build_unet
from drscreen.pipeline import DRScreeningPipeline
from drscreen.preprocess.landmarks import locate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--seg", type=Path, required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--also-gt", action="store_true",
                    help="also cache features computed from ground-truth masks")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    dev = torch.device(a.device)
    ck = torch.load(a.seg, map_location="cpu", weights_only=False)
    model = build_unet("lesion", width=int(ck.get("width", 24)))
    model.load_state_dict(ck["model"])
    model = model.to(dev).eval()
    print(f"Loaded segmentation (epoch {ck.get('epoch')}, dice {ck.get('dice'):.4f})")

    recs = read_manifest(a.cohort)
    feat_dir = a.cohort / "features"; feat_dir.mkdir(exist_ok=True)
    pred_dir = a.cohort / "preds"; pred_dir.mkdir(exist_ok=True)
    gt_dir = a.cohort / "features_gt"
    if a.also_gt:
        gt_dir.mkdir(exist_ok=True)

    t0 = time.time()
    batch_imgs, batch_meta = [], []

    def flush():
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(dev)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=dev.type == "cuda"):
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.sigmoid(logits.float()).permute(0, 2, 3, 1).cpu().numpy()
        for p, (rec, img_bgr, fov) in zip(probs, batch_meta):
            lm = locate(img_bgr, fov)
            vessel = DRScreeningPipeline._vessel_proxy(img_bgr, fov)
            f = extract(p, lm, fov, vessel, threshold=a.threshold)
            np.save(feat_dir / f"{rec.uid}.npy", f.to_vector())
            # Store a compact uint8 version of the maps for the report path.
            np.savez_compressed(pred_dir / f"{rec.uid}.npz",
                                lesions=(p * 255).astype(np.uint8))
            if a.also_gt and rec.has_masks:
                z = np.load(a.cohort / "masks" / f"{rec.uid}.npz")
                gt = (z["lesions"] > 127).astype(np.float32)
                if gt.shape[0] != p.shape[0]:
                    gt = cv2.resize(gt, p.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
                fg = extract(gt, lm, fov, vessel, threshold=0.5)
                np.save(gt_dir / f"{rec.uid}.npy", fg.to_vector())
        batch_imgs.clear(); batch_meta.clear()

    for i, rec in enumerate(recs):
        img = cv2.imread(str(a.cohort / "images" / f"{rec.uid}.png"), cv2.IMREAD_COLOR)
        fov = cv2.imread(str(a.cohort / "fov" / f"{rec.uid}.png"), cv2.IMREAD_GRAYSCALE)
        if img is None or fov is None:
            continue
        if img.shape[0] != a.size:
            img = cv2.resize(img, (a.size, a.size), interpolation=cv2.INTER_AREA)
            fov = cv2.resize(fov, (a.size, a.size), interpolation=cv2.INTER_NEAREST)
        batch_imgs.append(to_tensor(img))
        batch_meta.append((rec, img, fov))
        if len(batch_imgs) >= a.batch_size:
            flush()
        if (i + 1) % 250 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(recs)}  ({el:.0f}s, {(i+1)/el:.1f}/s)", flush=True)
    flush()

    print(f"Cached {len(recs)} feature vectors to {feat_dir} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
