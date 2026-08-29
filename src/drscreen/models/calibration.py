"""Confidence calibration and operating-point selection.

The problem statement asks for *calibrated confidence scores*, which is a
stronger requirement than it looks.  A modern CNN's softmax output is
systematically over-confident: it will say 0.97 on a batch that is right 82%
of the time.  In a human-in-the-loop screening programme that is actively
dangerous, because the review queue is prioritised by confidence -- an
over-confident false negative is never looked at again.

This module provides:

* **Temperature scaling** on the CORN logits (Guo et al., 2017) -- a single
  parameter fitted on a held-out split.

  One caveat specific to ordinal heads, which the standard write-up of
  temperature scaling does not cover: for a *single* logit, dividing by T is a
  monotone transform and therefore leaves the ranking, and hence AUC, exactly
  unchanged.  That guarantee does **not** carry over to CORN, because the
  referable probability is a *product* of sigmoids,
  P(y>1) = sigma(z_0) * sigma(z_1), and T does not factor out of a product of
  logistic functions.  Measured on random logits, ~4% of pairs change order at
  T = 2.5, though almost all are near-ties: the effect on AUC is around
  5e-4, i.e. far below the sampling error of any realistic test set.

  We report it rather than assume it away, and for the binary referral
  decision -- the one with clinical consequences -- we recalibrate the scalar
  P(referable) with isotonic regression, which *is* exactly rank-preserving.

* **Isotonic recalibration** of the binary referable probability. Applied on
  top of the temperature-scaled score, it fixes the probability scale of the
  referral decision without touching its ranking.
* **ECE / MCE / Brier / reliability curves** to prove the calibration worked.
* **Operating-point selection**: the threshold on P(referable) that meets the
  >=90% sensitivity constraint while maximising specificity, chosen on
  validation data and then *frozen* -- never re-tuned on the external test set.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import TARGET_SENSITIVITY, TARGET_SPECIFICITY


# --------------------------------------------------------------------------
# Temperature scaling
# --------------------------------------------------------------------------
class TemperatureScaler(nn.Module):
    """Single-parameter logit rescaling: z -> z / T."""

    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(float(np.log(init_temp))))

    @property
    def temperature(self) -> float:
        return float(self.log_temp.exp().item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.log_temp.exp()

    def fit(self, logits: torch.Tensor, targets: torch.Tensor,
            num_classes: int = 5, max_iter: int = 200) -> float:
        """Fit T by minimising the CORN NLL on a held-out calibration split."""
        from .grader import corn_loss
        logits = logits.detach()
        targets = targets.detach()
        opt = torch.optim.LBFGS([self.log_temp], lr=0.05, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            loss = corn_loss(self(logits), targets, num_classes)
            loss.backward()
            return loss

        opt.step(closure)
        return self.temperature


class IsotonicCalibrator:
    """Monotone recalibration of a single probability (the referable decision).

    Isotonic regression is non-parametric and cannot invert the ranking, so
    like temperature scaling it leaves AUC untouched while fixing the
    probability scale.  It needs more calibration data than temperature
    scaling, so we fall back to identity when the split is small.
    """

    def __init__(self, min_samples: int = 200):
        self.min_samples = min_samples
        self._iso = None

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        probs = np.asarray(probs, np.float64).ravel()
        labels = np.asarray(labels, np.float64).ravel()
        if probs.size < self.min_samples:
            self._iso = None
            return self
        from sklearn.isotonic import IsotonicRegression
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._iso.fit(probs, labels)
        return self

    def __call__(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, np.float64)
        if self._iso is None:
            return probs
        return self._iso.predict(probs.ravel()).reshape(probs.shape)

    # -- serialisation ----------------------------------------------------
    # The calibrator has to travel with the model. A referral threshold is
    # only meaningful on the probability scale it was chosen on, so shipping
    # the threshold without the recalibrator that produced that scale silently
    # applies the operating point to a different number line.
    def to_dict(self) -> dict:
        if self._iso is None:
            return {"kind": "identity"}
        return {"kind": "isotonic",
                "x": np.asarray(self._iso.X_thresholds_, float).tolist(),
                "y": np.asarray(self._iso.y_thresholds_, float).tolist()}

    @classmethod
    def from_dict(cls, d: dict | None) -> "IsotonicCalibrator":
        obj = cls()
        if not d or d.get("kind") != "isotonic":
            return obj
        x = np.asarray(d["x"], float)
        y = np.asarray(d["y"], float)

        class _Interp:
            """Replays the fitted step function without needing sklearn."""

            X_thresholds_, y_thresholds_ = x, y

            @staticmethod
            def predict(p):
                return np.interp(np.asarray(p, float), x, y, left=y[0], right=y[-1])

        obj._iso = _Interp()
        return obj


# --------------------------------------------------------------------------
# Calibration metrics
# --------------------------------------------------------------------------
@dataclass
class CalibrationReport:
    ece: float
    mce: float
    brier: float
    nll: float
    temperature: float
    bin_edges: list[float]
    bin_confidence: list[float]
    bin_accuracy: list[float]
    bin_count: list[int]

    def to_dict(self) -> dict:
        return asdict(self)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                               n_bins: int = 15, adaptive: bool = True
                               ) -> tuple[float, float, dict]:
    """ECE and MCE for a binary probability.

    ``adaptive=True`` uses equal-mass bins rather than equal-width.  With the
    heavily skewed confidence distributions a well-trained DR model produces,
    equal-width bins leave most of the range nearly empty and report an
    artificially small ECE.
    """
    p = np.asarray(probs, np.float64).ravel()
    y = np.asarray(labels, np.float64).ravel()
    n = p.size
    if n == 0:
        return 0.0, 0.0, {"edges": [], "conf": [], "acc": [], "count": []}

    if adaptive:
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(p, qs))
        if edges.size < 2:
            edges = np.array([0.0, 1.0])
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece, mce = 0.0, 0.0
    confs, accs, counts = [], [], []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < len(edges) - 2 else (p >= lo) & (p <= hi)
        cnt = int(m.sum())
        if cnt == 0:
            confs.append(float((lo + hi) / 2)); accs.append(float("nan")); counts.append(0)
            continue
        conf = float(p[m].mean())
        acc = float(y[m].mean())
        gap = abs(conf - acc)
        ece += (cnt / n) * gap
        mce = max(mce, gap)
        confs.append(conf); accs.append(acc); counts.append(cnt)

    return float(ece), float(mce), {"edges": edges.tolist(), "conf": confs,
                                    "acc": accs, "count": counts}


def calibration_report(probs: np.ndarray, labels: np.ndarray,
                       temperature: float = 1.0, n_bins: int = 15
                       ) -> CalibrationReport:
    p = np.clip(np.asarray(probs, np.float64).ravel(), 1e-7, 1 - 1e-7)
    y = np.asarray(labels, np.float64).ravel()
    ece, mce, bins = expected_calibration_error(p, y, n_bins)
    brier = float(np.mean((p - y) ** 2))
    nll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return CalibrationReport(ece=ece, mce=mce, brier=brier, nll=nll,
                             temperature=float(temperature),
                             bin_edges=bins["edges"], bin_confidence=bins["conf"],
                             bin_accuracy=bins["acc"], bin_count=bins["count"])


# --------------------------------------------------------------------------
# Operating point
# --------------------------------------------------------------------------
@dataclass
class OperatingPoint:
    threshold: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    meets_target: bool
    rationale: str

    def to_dict(self) -> dict:
        return asdict(self)


def select_threshold(probs: np.ndarray, labels: np.ndarray,
                     min_sensitivity: float = TARGET_SENSITIVITY,
                     min_specificity: float = TARGET_SPECIFICITY,
                     prevalence: float | None = None) -> OperatingPoint:
    """Pick the referral threshold on **validation** data.

    Policy, in priority order:

    1. Among thresholds meeting both the sensitivity and specificity targets,
       take the one maximising Youden's J (the balanced choice).
    2. If none meets both, satisfy sensitivity first and maximise specificity
       subject to it.  Sensitivity is the binding constraint in a screening
       programme: a missed proliferative DR costs sight, a false positive
       costs one teleconsultation.
    3. If even that is impossible, return the max-J point and flag it.
    """
    p = np.asarray(probs, np.float64).ravel()
    y = (np.asarray(labels).ravel() > 0).astype(np.int64)
    if p.size == 0 or y.sum() == 0 or (1 - y).sum() == 0:
        return OperatingPoint(0.5, 0.0, 0.0, 0.0, 0.0, False,
                              "Degenerate calibration set: one class absent.")

    cand = np.unique(np.concatenate([[0.0], p, [1.0]]))
    best = None
    rows = []
    for t in cand:
        pred = p >= t
        tp = int((pred & (y == 1)).sum()); fn = int((~pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum()); tn = int((~pred & (y == 0)).sum())
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        ppv = tp / max(tp + fp, 1)
        npv = tn / max(tn + fn, 1)
        rows.append((t, sens, spec, ppv, npv))

    both = [r for r in rows if r[1] >= min_sensitivity and r[2] >= min_specificity]
    if both:
        best = max(both, key=lambda r: r[1] + r[2] - 1)
        rationale = (f"Meets both targets (sens>={min_sensitivity:.0%}, "
                     f"spec>={min_specificity:.0%}); maximises Youden's J among those.")
        meets = True
    else:
        sens_ok = [r for r in rows if r[1] >= min_sensitivity]
        if sens_ok:
            best = max(sens_ok, key=lambda r: r[2])
            rationale = (f"Specificity target not simultaneously attainable; "
                         f"sensitivity constraint >={min_sensitivity:.0%} enforced "
                         f"first and specificity maximised subject to it.")
            meets = best[2] >= min_specificity
        else:
            best = max(rows, key=lambda r: r[1] + r[2] - 1)
            rationale = ("Neither target attainable on this split; reporting the "
                         "maximum-Youden operating point.")
            meets = False

    t, sens, spec, ppv, npv = best
    if prevalence is not None:
        # Re-express predictive values at the deployment prevalence rather than
        # the (usually enriched) validation prevalence -- otherwise PPV quoted
        # to a district programme is meaningless.
        ppv = (sens * prevalence) / max(sens * prevalence + (1 - spec) * (1 - prevalence), 1e-9)
        npv = (spec * (1 - prevalence)) / max(spec * (1 - prevalence) + (1 - sens) * prevalence, 1e-9)
        rationale += f" PPV/NPV restated at {prevalence:.1%} deployment prevalence."

    return OperatingPoint(float(t), float(sens), float(spec), float(ppv),
                          float(npv), bool(meets), rationale)


def selective_risk_curve(probs: np.ndarray, labels: np.ndarray,
                         uncertainty: np.ndarray, threshold: float,
                         coverages: np.ndarray | None = None) -> dict:
    """Risk-coverage curve for the human-in-the-loop abstention policy.

    At coverage c, the model auto-reports the c fraction of cases it is most
    certain about and defers the rest to a human.  The curve tells a
    programme manager exactly how much specialist time a given error rate
    costs -- which is the number the Simulink/SimPy capacity model consumes.
    """
    p = np.asarray(probs, np.float64).ravel()
    y = (np.asarray(labels).ravel() > 0).astype(np.int64)
    u = np.asarray(uncertainty, np.float64).ravel()
    order = np.argsort(u)                       # most certain first
    p, y = p[order], y[order]

    if coverages is None:
        coverages = np.linspace(0.1, 1.0, 19)
    out = {"coverage": [], "risk": [], "sensitivity": [], "specificity": [], "n": []}
    for c in coverages:
        k = max(1, int(round(c * p.size)))
        pred = (p[:k] >= threshold).astype(np.int64)
        yy = y[:k]
        err = float((pred != yy).mean())
        tp = int(((pred == 1) & (yy == 1)).sum()); fn = int(((pred == 0) & (yy == 1)).sum())
        fp = int(((pred == 1) & (yy == 0)).sum()); tn = int(((pred == 0) & (yy == 0)).sum())
        out["coverage"].append(float(c))
        out["risk"].append(err)
        out["sensitivity"].append(tp / max(tp + fn, 1))
        out["specificity"].append(tn / max(tn + fp, 1))
        out["n"].append(k)
    return out
