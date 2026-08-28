"""Class-activation mapping, implemented from scratch.

Written directly against autograd rather than pulled from ``pytorch-grad-cam``
for two reasons: the CORN head's target is a *cumulative* probability
P(y >= 2) rather than a class logit, which off-the-shelf CAM wrappers do not
express; and a screening system that ships an explanation has to be able to
say exactly what was differentiated with respect to what.

Three variants, because they answer different questions:

``gradcam``
    Channel weights = global-average-pooled gradients (Selvaraju et al.).
    The standard; smooth, and reliable for a single dominant object.

``gradcam++``
    Second-order weighting (Chattopadhay et al.).  Materially better when the
    evidence is *many small scattered objects* -- which is precisely what
    diabetic retinopathy looks like.  Plain Grad-CAM tends to collapse onto
    the single largest lesion cluster and under-weight the rest.

``hirescam``
    Element-wise gradient-activation product (Draelos & Carin) rather than a
    pooled weight.  It is provably faithful for the final layer -- the map
    sums to the score being explained -- so it is used as the audit map when
    the report needs a defensible attribution rather than a readable one.

Use the faithfulness metrics in :mod:`drscreen.explain.faithfulness` to choose
between them on *your* data instead of taking the above on trust.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F


class CAMExtractor:
    """Hook-based activation/gradient capture for one target layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles: list = []

    def __enter__(self):
        def fwd_hook(_m, _i, out):
            self.activations = out

        def bwd_hook(_m, _gi, gout):
            self.gradients = gout[0]

        self._handles.append(self.target_layer.register_forward_hook(fwd_hook))
        self._handles.append(self.target_layer.register_full_backward_hook(bwd_hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


def find_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Pick the last spatial convolution as the CAM target.

    Deliberately structural rather than a hard-coded ``model.conv_head``: the
    backbone is configurable, and a wrong-but-silent layer choice produces a
    plausible-looking heatmap that explains nothing.
    """
    candidates = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    if not candidates:
        raise ValueError("No Conv2d layer found; cannot compute a CAM.")
    return candidates[-1]


def _target_score(logits: torch.Tensor, mode: str, grade: int | None,
                  referable_index: int) -> torch.Tensor:
    """Scalar to differentiate.

    ``referable`` explains the actual clinical decision -- the log-odds of
    P(y >= 2) -- rather than an arbitrary class logit.  Because CORN gives
    P(y>k) = prod_j sigmoid(z_j), the log-odds of the cumulative event is the
    sum of log-sigmoids up to k, which is differentiable and exactly the
    quantity the referral threshold is applied to.
    """
    if mode == "referable":
        log_p = F.logsigmoid(logits[:, :referable_index + 1]).sum(dim=1)
        # log p - log(1 - p), stable via log1p on the exponentiated sum
        return log_p - torch.log1p(-torch.exp(log_p).clamp(max=1 - 1e-6))
    if mode == "grade" and grade is not None:
        from ..models.grader import corn_class_probs
        probs = corn_class_probs(logits).clamp_min(1e-9)
        return probs[:, grade].log()
    return logits.sum(dim=1)


def compute_cam(model: torch.nn.Module, image: torch.Tensor,
                clinical: torch.Tensor | None = None,
                method: str = "gradcam++",
                target: str = "referable",
                grade: int | None = None,
                referable_index: int = 1,
                target_layer: torch.nn.Module | None = None,
                relu: bool = True,
                fov_mask: np.ndarray | None = None) -> np.ndarray:
    """Compute a CAM for a single image.

    Returns an ``(H, W)`` float array in [0, 1] at the *input* resolution.

    ``fov_mask`` restricts attention to the retinal aperture. This is not
    cosmetic: the last conv layer has a receptive field covering a large
    fraction of the frame and the map is upsampled ~32x, so activation bleeds
    into the black surround and can even peak there. Attention outside the
    retina is meaningless by construction -- there is nothing to look at -- and
    leaving it in both alarms the reviewer and corrupts every localisation
    metric, because that mass sits in the denominator.
    """
    model.eval()
    layer = target_layer or find_target_layer(model)
    image = image.detach().requires_grad_(False)

    with CAMExtractor(model, layer) as ex:
        model.zero_grad(set_to_none=True)
        logits = model(image, clinical) if clinical is not None else model(image)
        score = _target_score(logits, target, grade, referable_index).sum()
        score.backward()

        acts = ex.activations           # (B, C, h, w)
        grads = ex.gradients            # (B, C, h, w)
        if acts is None or grads is None:
            raise RuntimeError("Hooks captured nothing; is the target layer in the graph?")

        if method == "gradcam":
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = (weights * acts).sum(dim=1)

        elif method == "gradcam++":
            g2 = grads.pow(2)
            g3 = grads.pow(3)
            sum_a = acts.sum(dim=(2, 3), keepdim=True)
            denom = 2 * g2 + sum_a * g3
            alpha = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
            weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
            cam = (weights * acts).sum(dim=1)

        elif method == "hirescam":
            cam = (grads * acts).sum(dim=1)

        else:
            raise ValueError(f"Unknown CAM method: {method}")

    if relu:
        cam = F.relu(cam)
    cam = cam.unsqueeze(1)
    cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam.squeeze(1)[0].detach().float().cpu().numpy()

    if fov_mask is not None:
        m = fov_mask
        if m.shape[:2] != cam.shape[:2]:
            import cv2
            m = cv2.resize(m, (cam.shape[1], cam.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        cam = np.where(m > 0, cam, 0.0)

    # Normalise *after* masking, so the scale is set by retinal attention
    # rather than by a corner artefact.
    lo, hi = float(cam.min()), float(cam.max())
    return (cam - lo) / (hi - lo) if hi > lo else np.zeros_like(cam)


@torch.no_grad()
def occlusion_map(model: torch.nn.Module, image: torch.Tensor,
                  clinical: torch.Tensor | None = None,
                  patch: int = 32, stride: int = 16,
                  target: str = "referable", referable_index: int = 1,
                  baseline: str = "blur") -> np.ndarray:
    """Model-agnostic occlusion sensitivity, as an independent cross-check.

    Gradient-based CAMs can be fooled by saturation; occlusion cannot, because
    it measures the actual output change.  It is ~100x slower, so it is used
    to validate the CAM offline rather than in the live report path.

    ``baseline='blur'`` occludes with a locally blurred patch rather than
    grey: a grey square is off-manifold for a fundus image and the resulting
    score drop partly measures "this looks nothing like a retina" instead of
    "the evidence was here".
    """
    model.eval()
    _, _, H, W = image.shape
    logits = model(image, clinical) if clinical is not None else model(image)
    base = float(_target_score(logits, target, None, referable_index).item())

    if baseline == "blur":
        k = max(3, (patch // 2) | 1)
        blurred = F.avg_pool2d(image, kernel_size=k, stride=1, padding=k // 2)
    else:
        blurred = torch.zeros_like(image)

    ys = list(range(0, max(H - patch + 1, 1), stride))
    xs = list(range(0, max(W - patch + 1, 1), stride))
    heat = np.zeros((len(ys), len(xs)), np.float32)

    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            occ = image.clone()
            occ[..., y:y + patch, x:x + patch] = blurred[..., y:y + patch, x:x + patch]
            lg = model(occ, clinical) if clinical is not None else model(occ)
            heat[i, j] = base - float(_target_score(lg, target, None, referable_index).item())

    t = torch.from_numpy(heat)[None, None]
    t = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
    out = t[0, 0].numpy()
    out = np.clip(out, 0, None)
    m = out.max()
    return out / m if m > 0 else out
