"""Modality dropout on the fusion arm's clinical branch.

The failure this guards is not a crash. Without it the fusion arm co-adapts:
the clinical vector is the easier signal to fit, gradient flows preferentially
there, and the image backbone under-trains. Measured on the validation split,
the co-adapted fusion model's image pathway alone scored AUC 0.8638 while the
image-only arm -- same backbone, same images -- scored 0.9562, and the fused
result (0.9523) landed below the better single modality.

The tell-tale was the *direction*: with poor lesion features the crutch was weak
and fusion tied the CNN arm (DeLong p = 0.92); improving the segmentation made
the crutch better and fusion lost outright (p = 0.04). Better inputs, worse
model.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from drscreen.models.grader import DRGrader


def _model(clinical_dropout: float) -> DRGrader:
    torch.manual_seed(0)
    return DRGrader(backbone="resnet18", pretrained=False, use_clinical=True,
                    clinical_dropout=clinical_dropout)


def _branch_variants(m: DRGrader, x: torch.Tensor, c: torch.Tensor):
    """Logits with the clinical branch fully on and fully off, at eval."""
    with torch.no_grad():
        z = m.features(x)
        enc = m.clinical_encoder(c) * m.gate(z)
        full = m.head(torch.cat([z, enc], 1))
        zeroed = m.head(torch.cat([z, torch.zeros_like(enc)], 1))
    return z, enc, full, zeroed


def test_modality_dropout_zeroes_the_whole_branch_per_sample():
    """Each sample is either fully fused or fully image-only -- never partial.

    Unit dropout would scale individual clinical dimensions, which the head can
    route around. Only removing the branch outright forces the image pathway to
    be independently accurate.
    """
    m = _model(0.5)
    x = torch.randn(512, 3, 64, 64)
    c = torch.randn(512, m.clinical_dim)

    m.eval()                       # BatchNorm on running stats, unit dropout off
    _, _, full, zeroed = _branch_variants(m, x, c)
    assert not torch.allclose(full, zeroed), "clinical branch has no effect"

    m.training = True              # flip only DRGrader's own flag
    with torch.no_grad():
        out = m(x, c)
    m.training = False

    is_zeroed = torch.isclose(out, zeroed, atol=1e-6).all(1)
    is_full = torch.isclose(out, full, atol=1e-6).all(1)
    assert bool((is_zeroed | is_full).all()), (
        "some samples are neither fully dropped nor fully kept: the branch is "
        "being scaled rather than removed")

    rate = is_zeroed.float().mean().item()
    assert 0.42 < rate < 0.58, f"drop rate {rate:.3f} is not near the 0.5 target"


def test_clinical_branch_is_always_used_at_inference():
    """Dropout is a training-time regulariser only."""
    m = _model(0.5).eval()
    x = torch.randn(32, 3, 64, 64)
    c = torch.randn(32, m.clinical_dim)
    with torch.no_grad():
        a, b = m(x, c), m(x, c)
    assert torch.allclose(a, b), "inference must be deterministic"

    _, _, full, _ = _branch_variants(m, x, c)
    assert torch.allclose(a, full, atol=1e-6), (
        "at eval the clinical branch must be used un-dropped and un-scaled")


def test_zero_rate_restores_the_previous_behaviour():
    """The fix must be switchable off, so the co-adaptation can be reproduced."""
    m = _model(0.0)
    x = torch.randn(64, 3, 64, 64)
    c = torch.randn(64, m.clinical_dim)
    m.eval()
    _, _, full, _ = _branch_variants(m, x, c)
    m.training = True
    with torch.no_grad():
        out = m(x, c)
    assert torch.allclose(out, full, atol=1e-6), (
        "clinical_dropout=0.0 must never drop the branch")


def test_clinical_only_arm_must_not_drop_its_only_modality():
    """With the image pathway zeroed there is nothing to fall back on.

    ``train_grader`` builds the clinical arm with ``clinical_dropout=0.0`` for
    this reason: dropping the branch there would leave the model grading from a
    zeroed embedding and a zeroed feature vector, which is not a baseline, it is
    noise.
    """
    m = DRGrader(backbone="resnet18", pretrained=False, use_clinical=True,
                 use_image=False, clinical_dropout=0.0)
    x = torch.randn(16, 3, 64, 64)
    c = torch.randn(16, m.clinical_dim)
    m.eval()
    with torch.no_grad():
        z = m.features(x)
        assert torch.count_nonzero(z) == 0, "image pathway should be zeroed"
        enc = m.clinical_encoder(c) * m.gate(z)
        full = m.head(torch.cat([z, enc], 1))
    m.training = True
    with torch.no_grad():
        out = m(x, c)
    assert torch.allclose(out, full, atol=1e-6)


def test_default_rate_is_on_for_fusion():
    """A default of 0 would silently reintroduce the co-adaptation."""
    m = DRGrader(backbone="resnet18", pretrained=False, use_clinical=True)
    assert m.clinical_dropout > 0.0
