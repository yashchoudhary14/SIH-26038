"""Quantitative evaluation of explanations.

The problem statement asks for Grad-CAM outputs "rated as clinically useful".
Producing a heatmap is trivial; producing evidence that the heatmap means
anything is the actual work, and it is the step almost every DR paper skips.

Two independent questions, two families of metric:

**Is the explanation faithful to the model?**
    Deletion and insertion AUC (Petsiuk et al., RISE).  Progressively remove
    the pixels the map ranks highest and watch the predicted probability: a
    faithful map causes a steep drop (low deletion AUC).  Progressively add
    them to a blurred canvas: a faithful map causes a steep rise (high
    insertion AUC).  These need no ground truth, so they run on every dataset.

**Is the explanation clinically correct?**
    Pointing game, lesion-mask IoU and lesion-hit rate against IDRiD's
    pixel-level lesion annotations.  A map can be perfectly faithful to a
    model that is looking at the wrong thing; only ground truth catches that.
    A high faithfulness score with a low pointing-game score is the signature
    of a shortcut-learning model, and it is exactly the failure mode that
    makes black-box DR systems collapse on external data.

Both are aggregated into a single ``ExplanationQuality`` record that goes into
the validation report next to the accuracy numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn.functional as F

# ``np.trapezoid`` is the numpy >= 2.0 spelling; ``np.trapz`` is the 1.x one and
# is removed in 2.x. requirements.txt permits both major versions (some
# environments pin numpy < 2 for unrelated packages), so resolve it once here
# rather than letting the deletion/insertion metric fail on a valid install.
_trapz = getattr(np, "trapezoid", None) or np.trapz


@dataclass
class ExplanationQuality:
    deletion_auc: float
    insertion_auc: float
    faithfulness: float            # insertion - deletion, higher is better
    pointing_game: float | None    # fraction of maps whose peak lands on a lesion
    lesion_hit_rate: float | None  # fraction of map mass on annotated lesions
    lesion_iou: float | None
    sparsity: float                # Gini; a map that highlights everything explains nothing
    n_evaluated: int

    def to_dict(self) -> dict:
        return asdict(self)

    def verdict(self) -> str:
        bits = [f"faithfulness {self.faithfulness:+.3f} "
                f"(insertion {self.insertion_auc:.3f} - deletion {self.deletion_auc:.3f})"]
        if self.pointing_game is not None:
            bits.append(f"pointing game {self.pointing_game:.1%}")
        if self.lesion_hit_rate is not None:
            bits.append(f"{self.lesion_hit_rate:.1%} of attention mass on annotated lesions")
        return "; ".join(bits)


# --------------------------------------------------------------------------
# Faithfulness (no ground truth required)
# --------------------------------------------------------------------------
@torch.no_grad()
def deletion_insertion(model: torch.nn.Module, image: torch.Tensor,
                       cam: np.ndarray, clinical: torch.Tensor | None = None,
                       steps: int = 32, target: str = "referable",
                       referable_index: int = 1) -> tuple[float, float, dict]:
    """Deletion and insertion AUC for one image/CAM pair.

    Pixels are removed/added in the order the CAM ranks them.  The deletion
    baseline is a heavy blur rather than black: blacking out regions of a
    fundus image creates artificial edges that the model responds to, which
    inflates the apparent drop and flatters the explanation.
    """
    from .cam import _target_score

    model.eval()
    device = image.device
    _, _, H, W = image.shape
    order = np.argsort(cam.ravel())[::-1]              # most important first

    k = max(9, (min(H, W) // 8) | 1)
    blurred = F.avg_pool2d(image, kernel_size=k, stride=1, padding=k // 2)

    def score_of(x: torch.Tensor) -> float:
        lg = model(x, clinical) if clinical is not None else model(x)
        return float(torch.sigmoid(_target_score(lg, target, None, referable_index)).item())

    n_px = H * W
    chunk = max(1, n_px // steps)
    del_scores, ins_scores, fracs = [], [], []

    del_img = image.clone()
    ins_img = blurred.clone()
    flat_idx = torch.from_numpy(np.ascontiguousarray(order)).long().to(device)

    del_scores.append(score_of(del_img))
    ins_scores.append(score_of(ins_img))
    fracs.append(0.0)

    for s in range(steps):
        sel = flat_idx[s * chunk:(s + 1) * chunk]
        if sel.numel() == 0:
            break
        ys, xs = sel // W, sel % W
        del_img[..., ys, xs] = blurred[..., ys, xs]
        ins_img[..., ys, xs] = image[..., ys, xs]
        del_scores.append(score_of(del_img))
        ins_scores.append(score_of(ins_img))
        fracs.append(min(1.0, (s + 1) * chunk / n_px))

    d_auc = float(_trapz(del_scores, fracs)) if len(fracs) > 1 else 0.0
    i_auc = float(_trapz(ins_scores, fracs)) if len(fracs) > 1 else 0.0
    return d_auc, i_auc, {"fraction": fracs, "deletion": del_scores, "insertion": ins_scores}


def gini_sparsity(cam: np.ndarray) -> float:
    """Gini coefficient of the attention mass.

    0 = uniform (explains nothing), 1 = a single pixel.  A CAM that lights up
    the whole retina passes deletion/insertion trivially yet tells a clinician
    nothing, so sparsity is reported alongside faithfulness to catch that.
    """
    x = np.sort(np.abs(cam).ravel())
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


# --------------------------------------------------------------------------
# Clinical correctness (needs lesion ground truth)
# --------------------------------------------------------------------------
def pointing_game(cam: np.ndarray, lesion_mask: np.ndarray,
                  tolerance: int = 15) -> bool:
    """Does the CAM's peak fall on (or within `tolerance` px of) a lesion?"""
    if lesion_mask.sum() == 0:
        return False
    idx = int(np.argmax(cam))
    y, x = divmod(idx, cam.shape[1])
    h, w = lesion_mask.shape[:2]
    y0, y1 = max(0, y - tolerance), min(h, y + tolerance + 1)
    x0, x1 = max(0, x - tolerance), min(w, x + tolerance + 1)
    return bool(lesion_mask[y0:y1, x0:x1].any())


def attention_on_lesions(cam: np.ndarray, lesion_mask: np.ndarray,
                         dilate: int = 9) -> float:
    """Fraction of total CAM mass falling inside dilated lesion regions.

    Dilation is applied because a CAM's spatial resolution is that of the last
    conv layer (typically 1/32 of the input), so demanding pixel-exact overlap
    with a 3-pixel microaneurysm would score every method at zero and tell us
    nothing.
    """
    import cv2
    if lesion_mask.sum() == 0:
        return float("nan")
    m = (lesion_mask > 0).astype(np.uint8)
    if dilate > 1:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate)))
    total = float(cam.sum())
    return float((cam * m).sum() / total) if total > 0 else float("nan")


def cam_lesion_iou(cam: np.ndarray, lesion_mask: np.ndarray,
                   percentile: float = 90.0, dilate: int = 9) -> float:
    import cv2
    if lesion_mask.sum() == 0:
        return float("nan")
    thr = np.percentile(cam, percentile)
    pred = (cam >= thr)
    m = (lesion_mask > 0).astype(np.uint8)
    if dilate > 1:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate)))
    m = m > 0
    inter = float((pred & m).sum())
    union = float((pred | m).sum())
    return inter / union if union > 0 else float("nan")


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
def evaluate_explanations(model: torch.nn.Module,
                          images: list[torch.Tensor],
                          cams: list[np.ndarray],
                          clinicals: list[torch.Tensor | None] | None = None,
                          lesion_masks: list[np.ndarray] | None = None,
                          steps: int = 24) -> ExplanationQuality:
    d_aucs, i_aucs, ginis, hits, iousidx, points = [], [], [], [], [], []
    for i, (img, cam) in enumerate(zip(images, cams)):
        cl = clinicals[i] if clinicals else None
        d, ins, _ = deletion_insertion(model, img, cam, cl, steps=steps)
        d_aucs.append(d); i_aucs.append(ins); ginis.append(gini_sparsity(cam))
        if lesion_masks is not None:
            lm = lesion_masks[i]
            if lm is not None and lm.sum() > 0:
                points.append(pointing_game(cam, lm))
                hits.append(attention_on_lesions(cam, lm))
                iousidx.append(cam_lesion_iou(cam, lm))

    def _mean(v):
        v = [x for x in v if x == x]          # drop NaN
        return float(np.mean(v)) if v else None

    d = _mean(d_aucs) or 0.0
    ins = _mean(i_aucs) or 0.0
    return ExplanationQuality(
        deletion_auc=d, insertion_auc=ins, faithfulness=ins - d,
        pointing_game=(float(np.mean(points)) if points else None),
        lesion_hit_rate=_mean(hits), lesion_iou=_mean(iousidx),
        sparsity=_mean(ginis) or 0.0, n_evaluated=len(images))
