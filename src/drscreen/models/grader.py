"""DR severity grader: ordinal CNN + clinical-feature fusion.

Three modelling decisions distinguish this from the usual
"EfficientNet with 5-way softmax + QWK loss" recipe.

1. **Ordinal (CORN) head instead of softmax.**
   DR grades are ordered, and the screening decision is a *cumulative* one:
   "is this grade >= 2?".  A softmax has to reconstruct that by summing
   probabilities of classes it modelled as unordered, and nothing constrains
   the result to be monotone.  CORN (Shi, Cao & Raschka, 2021) predicts the
   conditional probabilities P(y > k | y > k-1) directly, so
   P(y > k) = prod_{j<=k} sigma(z_j) is monotone **by construction**.  The
   referable-DR probability the whole clinical target rests on is then a
   single number the model actually optimised, not a derived quantity, and it
   can be thresholded without ever breaking the ordering.

2. **Fusion of CNN features with explicit clinical features.**
   The lesion-derived vector (counts per quadrant, NV location, exudate
   distance from fovea) enters the classifier head alongside the pooled CNN
   embedding.  This is the "integrated pipeline" the problem statement asks
   for, and it is what makes the model's evidence auditable: the same
   features that drive the prediction are the ones printed in the report.

3. **Deliberate uncertainty.**
   Dropout is retained at inference for MC sampling, giving an epistemic
   variance that -- combined with the calibrated probability -- drives the
   selective-referral / "send to human" decision instead of forcing a
   confident answer on every image.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import NUM_GRADES, REFERABLE_THRESHOLD
from .lesion_features import ClinicalFeatures


# --------------------------------------------------------------------------
# CORN ordinal machinery
# --------------------------------------------------------------------------
def corn_loss(logits: torch.Tensor, targets: torch.Tensor,
              num_classes: int = NUM_GRADES,
              class_weights: torch.Tensor | None = None) -> torch.Tensor:
    """Conditional ordinal loss.

    ``logits`` is ``(B, num_classes - 1)``; task *k* models
    P(y > k | y > k-1) and is trained **only on the subset with y > k-1**,
    which is what makes the chained product a valid probability.

    Normalisation is a weighted mean over every ``(sample, task)`` conditional
    term, not a mean of per-task means. The previous form gave every task equal
    weight regardless of how many samples it had, and silently dropped tasks
    whose subset was empty. Both effects concentrate on the deep conditionals:
    task 3 only sees grades 3 and 4 (~17% of the cohort, two or three images in
    a batch of 16), so its "mean" was taken over a handful of samples yet
    carried the same weight as task 0's mean over the full batch, and the
    changing number of surviving tasks made the denominator -- and so the
    effective learning rate on z2 and z3 -- jitter from batch to batch. Those
    are exactly the two logits that decide severe NPDR and proliferative DR.

    ``class_weights`` are applied per sample and the result is divided by their
    sum. ``F.binary_cross_entropy_with_logits(..., weight=w)`` does *not*
    renormalise -- it returns ``mean(w * loss)`` -- so passing weights through
    it made each task's loss magnitude depend on which grades happened to land
    in the batch.
    """
    total = logits.new_zeros(())
    denom = logits.new_zeros(())
    for k in range(num_classes - 1):
        sel = (torch.ones_like(targets, dtype=torch.bool) if k == 0
               else targets > (k - 1))
        if not bool(sel.any()):
            continue
        z = logits[sel, k]
        y = (targets[sel] > k).float()
        terms = F.binary_cross_entropy_with_logits(z, y, reduction="none")
        if class_weights is not None:
            w = class_weights[targets[sel]]
            total = total + (terms * w).sum()
            denom = denom + w.sum()
        else:
            total = total + terms.sum()
            denom = denom + float(terms.numel())
    if float(denom) == 0.0:
        return logits.sum() * 0.0
    return total / denom


def corn_cumulative_probs(logits: torch.Tensor) -> torch.Tensor:
    """P(y > k) for k = 0..K-2, monotone non-increasing by construction."""
    return torch.cumprod(torch.sigmoid(logits), dim=1)


def corn_class_probs(logits: torch.Tensor) -> torch.Tensor:
    """Convert CORN logits to a proper distribution over the K grades."""
    cum = corn_cumulative_probs(logits)                      # (B, K-1)
    ones = torch.ones(cum.size(0), 1, device=cum.device, dtype=cum.dtype)
    zeros = torch.zeros_like(ones)
    upper = torch.cat([ones, cum], dim=1)                    # P(y > k-1)
    lower = torch.cat([cum, zeros], dim=1)                   # P(y > k)
    return (upper - lower).clamp_min(0.0)


def corn_predict(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (corn_cumulative_probs(logits) > threshold).sum(dim=1).long()


def referable_prob(logits: torch.Tensor,
                   threshold_index: int = REFERABLE_THRESHOLD - 1) -> torch.Tensor:
    """P(grade >= REFERABLE_THRESHOLD), i.e. P(y > 1) for a threshold of 2."""
    return corn_cumulative_probs(logits)[:, threshold_index]


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class ClinicalFeatureEncoder(nn.Module):
    """Small MLP over the hand-computed clinical vector."""

    def __init__(self, in_dim: int, out_dim: int = 128, p_drop: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 256), nn.SiLU(inplace=True), nn.Dropout(p_drop),
            nn.Linear(256, out_dim), nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DRGrader(nn.Module):
    """Backbone + optional clinical-feature fusion + CORN ordinal head."""

    def __init__(self, backbone: str = "tf_efficientnet_b0",
                 num_classes: int = NUM_GRADES,
                 pretrained: bool = True,
                 use_clinical: bool = True,
                 clinical_dim: int | None = None,
                 p_drop: float = 0.3,
                 in_chans: int = 3,
                 use_image: bool = True):
        super().__init__()
        import timm
        self.num_classes = num_classes
        self.use_clinical = use_clinical
        #: When False the CNN embedding is zeroed, so the model is forced to
        #: grade from the clinical feature vector alone. This is the
        #: classical-CV arm of the ablation, and it is a constructor flag
        #: rather than a monkeypatched ``backbone.forward`` so that
        #: ``copy.deepcopy`` (used by the EMA) reproduces it faithfully --
        #: deepcopy treats function objects as atomic, so a patched closure
        #: would silently keep pointing at the original module.
        self.use_image = use_image

        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0,
            global_pool="avg", in_chans=in_chans)
        feat_dim = self.backbone.num_features

        self.clinical_dim = clinical_dim or ClinicalFeatures.vector_size()
        fuse_dim = feat_dim
        if use_clinical:
            self.clinical_encoder = ClinicalFeatureEncoder(self.clinical_dim, 128, p_drop / 2)
            fuse_dim += 128
            # Learned gate: lets the model down-weight the clinical branch when
            # the segmentation head is unreliable (e.g. a borderline-quality
            # image), instead of trusting a lesion count that came from noise.
            self.gate = nn.Sequential(nn.Linear(feat_dim, 1), nn.Sigmoid())

        self.dropout = nn.Dropout(p_drop)
        self.head = nn.Sequential(
            nn.LayerNorm(fuse_dim),
            nn.Linear(fuse_dim, 256), nn.SiLU(inplace=True), nn.Dropout(p_drop),
            nn.Linear(256, num_classes - 1),
        )

    def features(self, image: torch.Tensor) -> torch.Tensor:
        if not self.use_image:
            with torch.no_grad():
                z = self.backbone(image)
            return torch.zeros_like(z)
        return self.backbone(image)

    def forward(self, image: torch.Tensor,
                clinical: torch.Tensor | None = None) -> torch.Tensor:
        z = self.dropout(self.features(image))
        if self.use_clinical:
            if clinical is None:
                clinical = torch.zeros(z.size(0), self.clinical_dim,
                                       device=z.device, dtype=z.dtype)
            c = self.clinical_encoder(clinical) * self.gate(z)
            z = torch.cat([z, c], dim=1)
        return self.head(z)

    # -- inference helpers -------------------------------------------------
    @torch.no_grad()
    def predict(self, image: torch.Tensor, clinical: torch.Tensor | None = None,
                mc_samples: int = 0, temperature: float = 1.0) -> dict:
        """Return grade, class probabilities, referable probability, uncertainty.

        ``mc_samples > 0`` keeps dropout active and samples the posterior, so
        the returned ``epistemic`` term reflects model uncertainty (what a
        deep ensemble would give, at 1/N the training cost).
        """
        was_training = self.training
        if mc_samples > 0:
            self.eval()
            for m in self.modules():                 # dropout only, not norms
                if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                    m.train()
            probs = []
            for _ in range(mc_samples):
                logits = self(image, clinical) / temperature
                probs.append(corn_class_probs(logits))
            P = torch.stack(probs)                    # (S, B, K)
            mean = P.mean(0)
            epistemic = P.var(0).sum(dim=1)
        else:
            self.eval()
            logits = self(image, clinical) / temperature
            mean = corn_class_probs(logits)
            epistemic = torch.zeros(mean.size(0), device=mean.device)
        self.train(was_training)

        grades = torch.arange(self.num_classes, device=mean.device, dtype=mean.dtype)
        expected = (mean * grades).sum(dim=1)
        p_ref = mean[:, REFERABLE_THRESHOLD:].sum(dim=1)
        entropy = -(mean.clamp_min(1e-9).log() * mean).sum(dim=1)
        return {
            "class_probs": mean,
            "grade": mean.argmax(dim=1),
            "expected_grade": expected,
            "referable_prob": p_ref,
            "entropy": entropy,
            "epistemic": epistemic,
        }


def build_grader(cfg: dict | None = None) -> DRGrader:
    cfg = cfg or {}
    return DRGrader(
        backbone=cfg.get("backbone", "tf_efficientnet_b0"),
        pretrained=cfg.get("pretrained", True),
        use_clinical=cfg.get("use_clinical", True),
        p_drop=cfg.get("dropout", 0.3),
        in_chans=cfg.get("in_chans", 3),
    )
