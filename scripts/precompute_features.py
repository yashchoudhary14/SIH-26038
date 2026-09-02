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
import itertools
import time
from pathlib import Path
from pathlib import Path

import cv2
import numpy as np
import torch
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from drscreen.constants import NUM_LESION_CLASSES
from drscreen.data.cohort import read_manifest
from drscreen.data.torch_data import to_tensor
from drscreen.models.lesion_features import extract, fit_lesion_thresholds
from drscreen.models.segmentation import build_unet
from drscreen.pipeline import DRScreeningPipeline
from drscreen.preprocess.landmarks import locate


def bounded_map(pool, fn, items, window):
    """Like ``pool.map`` but with at most ``window`` tasks in flight.

    ``ThreadPoolExecutor.map`` submits every task the moment it is called and
    holds each finished result until the consumer reaches it. Here a result
    carries a 3x1024x1024 float32 tensor (~12 MB), and the consumer is far
    slower than the loaders because it runs GPU inference and connected-
    component analysis per case. Finished-but-unconsumed results therefore
    accumulate without bound: on a 12,495-case cohort that is ~160 GB of
    retained buffers, and the process dies on a 1 MiB allocation deep inside
    ``cv2.connectedComponentsWithStats``.

    The 6,047-case cohort survived it only by being small enough. Peak memory
    must not scale with cohort size, so keep a fixed window of futures open
    and submit a new one for each result consumed.
    """
    it = iter(items)
    pending = deque(pool.submit(fn, x) for x in itertools.islice(it, window))
    while pending:
        yield pending.popleft().result()
        nxt = next(it, None)
        if nxt is not None:
            pending.append(pool.submit(fn, nxt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--seg", type=Path, required=True)
    ap.add_argument("--size", type=int, default=512,
                    help="resolution the segmentation model runs at")
    ap.add_argument("--feature-size", type=int, default=512,
                    help="resolution clinical features are extracted at. Lesion "
                         "probabilities are downscaled to this first. Landmarks "
                         "and connected components cost ~6x more at 1024 and "
                         "gain nothing: the features are per-region scalars.")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="fallback per-class cut-point used ONLY when --thr-cohort "
                         "is not given. A blanket 0.5 does NOT match the fitted "
                         "thresholds the live pipeline applies, so features cached "
                         "with it are a train/serve skew for the fusion grader.")
    ap.add_argument("--thr-cohort", type=Path, default=None,
                    help="cohort carrying the lesion masks (e.g. cohort_seg1024). "
                         "When given, per-class F1-optimal thresholds are fitted "
                         "from --seg on its seg_val split and used for feature "
                         "extraction -- the SAME values validate.py deploys, so "
                         "cached features match what the model is served live.")
    ap.add_argument("--also-gt", action="store_true",
                    help="also cache features computed from ground-truth masks")
    ap.add_argument("--save-preds", action="store_true",
                    help="persist the lesion probability maps (~5 MB each). Off "
                         "by default; only the report path reads them.")
    ap.add_argument("--only-missing", action="store_true",
                    help="skip cases that already have a cached feature vector")
    ap.add_argument("--splits", nargs="*", default=None,
                    help="restrict to these cohort splits")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    dev = torch.device(a.device)
    ck = torch.load(a.seg, map_location="cpu", weights_only=False)
    model = build_unet("lesion", width=int(ck.get("width", 24)))
    model.load_state_dict(ck["model"])
    model = model.to(dev).eval()
    print(f"Loaded segmentation (epoch {ck.get('epoch')}, dice {ck.get('dice', float('nan')):.4f})")
    print(f"Segmenting at {a.size}px, extracting features at {a.feature_size}px")

    # Lesion cut-points MUST match what the live pipeline applies, or the fusion
    # grader learns from lesion counts it will never see at inference. The live
    # thresholds are the F1-optimal per-class values validate.py fits from this
    # same segmentation model; fit them here from the same function so the two
    # cannot drift. Written to features/thresholds.json for validate.py to check.
    lesion_thr: dict = {}
    if a.thr_cohort is not None:
        lesion_thr = fit_lesion_thresholds(a.seg, a.thr_cohort, device=a.device)
        if lesion_thr:
            print(f"  fitted lesion thresholds (match live inference): {lesion_thr}")
        else:
            print("  !! --thr-cohort given but no thresholds fitted (no seg_val "
                  "masks found); falling back to the scalar --threshold")
    if not lesion_thr:
        print(f"  !! caching features at a BLANKET {a.threshold}. This will NOT "
              f"match live inference; pass --thr-cohort to fit per-class values.")
    feat_threshold = lesion_thr or a.threshold

    recs = read_manifest(a.cohort)
    feat_dir = a.cohort / "features"; feat_dir.mkdir(exist_ok=True)
    if a.splits:
        recs = [r for r in recs if r.split in set(a.splits)]
    if a.only_missing:
        before = len(recs)
        recs = [r for r in recs if not (feat_dir / f"{r.uid}.npy").exists()]
        print(f"{before - len(recs)} already cached, {len(recs)} to compute")
    pred_dir = a.cohort / "preds"
    if a.save_preds:
        pred_dir.mkdir(exist_ok=True)
    gt_dir = a.cohort / "features_gt"
    if a.also_gt:
        gt_dir.mkdir(exist_ok=True)

    t0 = time.time()
    done = 0

    def load(rec):
        """Read one case and prepare both resolutions (runs in worker threads)."""
        img = cv2.imread(str(a.cohort / "images" / f"{rec.uid}.png"), cv2.IMREAD_COLOR)
        fov = cv2.imread(str(a.cohort / "fov" / f"{rec.uid}.png"), cv2.IMREAD_GRAYSCALE)
        if img is None or fov is None:
            return None
        seg_img = img if img.shape[0] == a.size else cv2.resize(
            img, (a.size, a.size), interpolation=cv2.INTER_CUBIC
            if img.shape[0] < a.size else cv2.INTER_AREA)
        f_img = img if img.shape[0] == a.feature_size else cv2.resize(
            img, (a.feature_size, a.feature_size), interpolation=cv2.INTER_AREA)
        f_fov = fov if fov.shape[0] == a.feature_size else cv2.resize(
            fov, (a.feature_size, a.feature_size), interpolation=cv2.INTER_NEAREST)
        return rec, to_tensor(seg_img), f_img, f_fov

    batch_t, batch_meta = [], []

    def flush():
        nonlocal done
        if not batch_t:
            return
        x = torch.stack(batch_t).to(dev)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=dev.type == "cuda"):
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.sigmoid(logits.float()).permute(0, 2, 3, 1).cpu().numpy()

        for p, (rec, f_img, f_fov) in zip(probs, batch_meta):
            if p.shape[0] != a.feature_size:
                p = cv2.resize(p, (a.feature_size, a.feature_size),
                               interpolation=cv2.INTER_AREA)
            lm = locate(f_img, f_fov)
            vessel = DRScreeningPipeline._vessel_proxy(f_img, f_fov)
            f = extract(p, lm, f_fov, vessel, threshold=feat_threshold)
            np.save(feat_dir / f"{rec.uid}.npy", f.to_vector())

            if a.save_preds:
                np.savez_compressed(pred_dir / f"{rec.uid}.npz",
                                    lesions=(p * 255).astype(np.uint8))

            if a.also_gt and rec.has_masks:
                z = np.load(a.cohort / "masks" / f"{rec.uid}.npz")
                gt = (z["lesions"] > 127).astype(np.float32)
                if gt.shape[0] != a.feature_size:
                    gt = cv2.resize(gt, (a.feature_size, a.feature_size),
                                    interpolation=cv2.INTER_NEAREST)
                fg = extract(gt, lm, f_fov, vessel, threshold=0.5)
                np.save(gt_dir / f"{rec.uid}.npy", fg.to_vector())
            done += 1

        batch_t.clear(); batch_meta.clear()

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for item in bounded_map(pool, load, recs, a.workers * 4):
            if item is None:
                continue
            rec, tensor, f_img, f_fov = item
            batch_t.append(tensor); batch_meta.append((rec, f_img, f_fov))
            if len(batch_t) >= a.batch_size:
                flush()
                if done % 500 < a.batch_size:
                    el = time.time() - t0
                    print(f"  {done}/{len(recs)}  ({el:.0f}s, {done/max(el,1):.1f}/s)",
                          flush=True)
    flush()

    # Provenance: record exactly which thresholds these features were cached at,
    # so validate.py can assert the deployed thresholds are the same ones.
    import json
    (feat_dir / "thresholds.json").write_text(json.dumps({
        "lesion_thresholds": lesion_thr or None,
        "scalar_fallback": None if lesion_thr else a.threshold,
        "seg_checkpoint": str(a.seg),
        "feature_size": a.feature_size,
    }, indent=2))
    print(f"Cached {done} feature vectors to {feat_dir} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
