"""U-Net style segmentation for retinal vessels, lesions and the optic disc.

One architecture serves three heads because the encoder features are shared;
what differs is the output channel count and the loss weighting:

===================  ========  =====================================
head                 channels  trained on
===================  ========  =====================================
``vessel``           1         DRIVE (or phantom vessel masks)
``lesion``           5         IDRiD pixel ground truth
``structure``        2         optic disc + fovea disc (IDRiD / phantom)
===================  ========  =====================================

Two design choices matter for microaneurysm sensitivity, which is what the
whole screening target rests on:

1. **The decoder keeps full input resolution.**  A 3-pixel microaneurysm at
   1024px becomes sub-pixel at 224px; there is no recovering it.  Segmentation
   therefore runs at the native tile resolution and the grader consumes the
   resulting masks, rather than everything being squeezed through one
   low-resolution backbone.

2. **Deep supervision on the two finest decoder stages.**  Tiny structures
   produce a vanishing gradient share in a plain U-Net because the loss is
   dominated by the (vastly more numerous) background pixels; supervising the
   high-resolution stages directly counteracts that.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(ch: int) -> nn.Module:
    # GroupNorm rather than BatchNorm: segmentation runs at batch sizes of 2-4
    # at high resolution, where BatchNorm statistics are unusable.
    return nn.GroupNorm(num_groups=min(32, max(1, ch // 8)), num_channels=ch)


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), _norm(cout), nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), _norm(cout), nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SqueezeExcite(nn.Module):
    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(ch, max(4, ch // r), 1), nn.SiLU(inplace=True),
            nn.Conv2d(max(4, ch // r), ch, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(F.adaptive_avg_pool2d(x, 1))


class AttentionGate(nn.Module):
    """Additive attention on the skip connection (Oktay et al., 2018).

    Suppresses the skip's contribution where the decoder has no evidence,
    which on fundus images means the large avascular background stops
    injecting texture noise into the finest decoder stage.
    """

    def __init__(self, ch_skip: int, ch_gate: int, ch_int: int):
        super().__init__()
        self.wg = nn.Conv2d(ch_gate, ch_int, 1)
        self.wx = nn.Conv2d(ch_skip, ch_int, 1)
        self.psi = nn.Sequential(nn.Conv2d(ch_int, 1, 1), nn.Sigmoid())

    def forward(self, skip, gate):
        g = self.wg(gate)
        x = self.wx(skip)
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return skip * self.psi(F.silu(g + x))


class UNet(nn.Module):
    """Attention U-Net with optional deep supervision.

    Kept dependency-free (no ``segmentation_models_pytorch``) so the exported
    ONNX graph is fully under our control -- the edge deployment in the
    Simulink/SimPy model assumes a fixed, quantisable graph.
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 1,
                 widths: tuple[int, ...] = (32, 64, 128, 256, 512),
                 deep_supervision: bool = True):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.out_ch = out_ch

        self.encoders = nn.ModuleList()
        prev = in_ch
        for w in widths:
            self.encoders.append(ConvBlock(prev, w))
            prev = w
        self.pool = nn.MaxPool2d(2)

        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.se = nn.ModuleList()
        rev = list(reversed(widths))
        for i in range(len(widths) - 1):
            cin, cskip = rev[i], rev[i + 1]
            self.ups.append(nn.ConvTranspose2d(cin, cskip, 2, stride=2))
            self.gates.append(AttentionGate(cskip, cskip, max(8, cskip // 2)))
            self.decoders.append(ConvBlock(cskip * 2, cskip))
            self.se.append(SqueezeExcite(cskip))

        self.head = nn.Conv2d(widths[0], out_ch, 1)
        # Auxiliary heads on the two finest decoder stages.
        self.aux_heads = nn.ModuleList([nn.Conv2d(widths[1], out_ch, 1),
                                        nn.Conv2d(widths[2], out_ch, 1)]) \
            if deep_supervision and len(widths) >= 3 else nn.ModuleList()

    def forward(self, x):
        skips = []
        for i, enc in enumerate(self.encoders):
            x = enc(x)
            if i < len(self.encoders) - 1:
                skips.append(x)
                x = self.pool(x)

        aux: list[torch.Tensor] = []
        for i, (up, gate, dec, se) in enumerate(
                zip(self.ups, self.gates, self.decoders, self.se)):
            x = up(x)
            skip = skips[-(i + 1)]
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            skip = gate(skip, x)
            x = se(dec(torch.cat([x, skip], dim=1)))
            # decoder stage index counted from the coarse end
            depth_from_fine = len(self.ups) - 1 - i
            if self.training and self.aux_heads and 1 <= depth_from_fine <= len(self.aux_heads):
                aux.append(self.aux_heads[depth_from_fine - 1](x))

        logits = self.head(x)
        if self.training and aux:
            return logits, aux
        return logits


# --------------------------------------------------------------------------
# Losses tuned for extreme foreground/background imbalance
# --------------------------------------------------------------------------
def dice_loss(logits: torch.Tensor, target: torch.Tensor,
              eps: float = 1e-6) -> torch.Tensor:
    p = torch.sigmoid(logits)
    dims = (0, 2, 3)
    inter = (p * target).sum(dims)
    denom = p.sum(dims) + target.sum(dims)
    return (1.0 - (2 * inter + eps) / (denom + eps)).mean()


def tversky_loss(logits: torch.Tensor, target: torch.Tensor,
                 alpha: float = 0.3, beta: float = 0.7,
                 gamma: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    """Focal Tversky loss.

    ``beta > alpha`` penalises false negatives harder than false positives.
    For microaneurysms that is the correct asymmetry: a missed MA can mean a
    missed referral, whereas a false MA costs a few seconds of a
    reviewer's time.
    """
    p = torch.sigmoid(logits)
    dims = (0, 2, 3)
    tp = (p * target).sum(dims)
    fp = (p * (1 - target)).sum(dims)
    fn = ((1 - p) * target).sum(dims)
    ti = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return torch.pow(1.0 - ti, gamma).mean()


def segmentation_loss(logits, target, aux: list | None = None,
                      bce_weight: float = 0.5, aux_weight: float = 0.4,
                      pos_weight: torch.Tensor | None = None,
                      channel_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Combined BCE + Tversky loss over the lesion channels.

    ``channel_mask`` restricts the loss to channels that carry real annotation.
    A channel whose target is all-zero across the whole corpus -- which is what
    neovascularisation is on IDRiD -- otherwise contributes a confident-negative
    gradient on every single step: the network spends capacity learning to never
    fire, and the loss the annotated channels are scored on is diluted by a term
    that can only ever go to zero.
    """
    if channel_mask is not None:
        logits = logits[:, channel_mask]
        target = target[:, channel_mask]
        if aux:
            aux = [a[:, channel_mask] for a in aux]
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    loss = bce_weight * bce + (1 - bce_weight) * tversky_loss(logits, target)
    if aux:
        for a in aux:
            a_up = F.interpolate(a, size=target.shape[-2:], mode="bilinear", align_corners=False)
            loss = loss + aux_weight * (
                bce_weight * F.binary_cross_entropy_with_logits(a_up, target, pos_weight=pos_weight)
                + (1 - bce_weight) * tversky_loss(a_up, target))
    return loss


def build_unet(task: str = "vessel", in_ch: int = 3, width: int = 32) -> UNet:
    from ..constants import NUM_LESION_CLASSES
    out = {"vessel": 1, "lesion": NUM_LESION_CLASSES, "structure": 2}[task]
    widths = tuple(width * m for m in (1, 2, 4, 8, 16))
    return UNet(in_ch=in_ch, out_ch=out, widths=widths)
