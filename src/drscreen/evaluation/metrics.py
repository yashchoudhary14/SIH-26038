"""Clinical validation metrics with proper uncertainty quantification.

A screening claim of ">90% sensitivity" is meaningless without an interval.
On a 400-image test set, an observed sensitivity of 0.91 has a 95% CI of
roughly [0.86, 0.94] -- i.e. the data are consistent with the target being
missed.  Every metric here therefore returns a point estimate *and* an
interval, and comparisons between models use paired tests rather than
eyeballed differences.

Implemented:

* Wilson score intervals for proportions (sensitivity, specificity, PPV, NPV)
  -- correct at the extremes where the normal approximation fails, which is
  exactly where screening metrics live.
* Stratified bootstrap CIs for any metric, including QWK and AUC.
* DeLong's method for the variance of an AUC and for comparing two correlated
  AUCs on the same cases.
* McNemar's exact/continuity-corrected test for comparing two classifiers'
  errors on the same cases.
* Quadratic weighted kappa, the standard DR grading agreement statistic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

from ..constants import NUM_GRADES, REFERABLE_THRESHOLD


# --------------------------------------------------------------------------
# Interval estimators
# --------------------------------------------------------------------------
def wilson_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    from scipy.stats import norm
    z = float(norm.ppf(1 - alpha / 2))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass
class Proportion:
    value: float
    lower: float
    upper: float
    numerator: int
    denominator: int

    def __str__(self) -> str:
        return f"{self.value:.3f} [{self.lower:.3f}-{self.upper:.3f}] ({self.numerator}/{self.denominator})"

    def to_dict(self) -> dict:
        return asdict(self)


def proportion(k: int, n: int, alpha: float = 0.05) -> Proportion:
    lo, hi = wilson_interval(k, n, alpha)
    return Proportion(k / n if n else 0.0, lo, hi, int(k), int(n))


# --------------------------------------------------------------------------
# Binary screening metrics
# --------------------------------------------------------------------------
@dataclass
class BinaryMetrics:
    threshold: float
    sensitivity: Proportion
    specificity: Proportion
    ppv: Proportion
    npv: Proportion
    accuracy: Proportion
    f1: float
    youden_j: float
    auc: float
    auc_ci: tuple[float, float]
    auprc: float
    tp: int
    fp: int
    tn: int
    fn: int
    n_positive: int
    n_negative: int
    meets_sensitivity_target: bool
    meets_specificity_target: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("sensitivity", "specificity", "ppv", "npv", "accuracy"):
            d[k] = getattr(self, k).to_dict()
        return d

    def summary(self) -> str:
        return (f"Sens {self.sensitivity}  Spec {self.specificity}  "
                f"AUC {self.auc:.4f} [{self.auc_ci[0]:.4f}-{self.auc_ci[1]:.4f}]")


def binary_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float,
                   alpha: float = 0.05,
                   target_sens: float = 0.90,
                   target_spec: float = 0.85) -> BinaryMetrics:
    s = np.asarray(scores, np.float64).ravel()
    y = (np.asarray(labels).ravel() > 0).astype(np.int64)
    pred = (s >= threshold).astype(np.int64)

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    sens = proportion(tp, tp + fn, alpha)
    spec = proportion(tn, tn + fp, alpha)
    ppv = proportion(tp, tp + fp, alpha)
    npv = proportion(tn, tn + fn, alpha)
    acc = proportion(tp + tn, len(y), alpha)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)

    auc, auc_lo, auc_hi = delong_auc_ci(s, y, alpha)
    auprc = average_precision(s, y)

    return BinaryMetrics(
        threshold=float(threshold), sensitivity=sens, specificity=spec,
        ppv=ppv, npv=npv, accuracy=acc, f1=float(f1),
        youden_j=float(sens.value + spec.value - 1),
        auc=float(auc), auc_ci=(float(auc_lo), float(auc_hi)),
        auprc=float(auprc), tp=tp, fp=fp, tn=tn, fn=fn,
        n_positive=int(y.sum()), n_negative=int((1 - y).sum()),
        meets_sensitivity_target=bool(sens.lower >= 0 and sens.value >= target_sens),
        meets_specificity_target=bool(spec.value >= target_spec),
    )


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    y = np.asarray(labels).ravel()
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    return float(average_precision_score(y, np.asarray(scores).ravel()))


# --------------------------------------------------------------------------
# DeLong
# --------------------------------------------------------------------------
def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    ranks = np.empty(n, np.float64)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and xs[j + 1] == xs[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    out = np.empty(n, np.float64)
    out[order] = ranks
    return out


def _delong_structural(scores: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (auc_per_model, V10, V01) for one or more score vectors.

    ``scores`` is ``(K, N)`` for K models evaluated on the same N cases.
    Implements the fast O(N log N) algorithm of Sun & Xu (2014).
    """
    pos = scores[:, y == 1]
    neg = scores[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    K = scores.shape[0]

    tx = np.empty((K, m)); ty = np.empty((K, n)); tz = np.empty((K, m + n))
    for k in range(K):
        tx[k] = _midrank(pos[k])
        ty[k] = _midrank(neg[k])
        tz[k] = _midrank(np.concatenate([pos[k], neg[k]]))

    aucs = (tz[:, :m].sum(axis=1) / (m * n)) - (m + 1) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    return aucs, v10, v01


def delong_auc_ci(scores: np.ndarray, labels: np.ndarray,
                  alpha: float = 0.05) -> tuple[float, float, float]:
    """AUC with a DeLong 95% CI (logit-transformed, so it respects [0,1])."""
    s = np.asarray(scores, np.float64).ravel()[None, :]
    y = (np.asarray(labels).ravel() > 0).astype(np.int64)
    if y.sum() == 0 or (1 - y).sum() == 0:
        return (float("nan"), float("nan"), float("nan"))
    aucs, v10, v01 = _delong_structural(s, y)
    auc = float(aucs[0])
    var = np.var(v01[0], ddof=1) / v01.shape[1] + np.var(v10[0], ddof=1) / v10.shape[1]
    se = math.sqrt(max(var, 1e-12))

    from scipy.stats import norm
    z = float(norm.ppf(1 - alpha / 2))
    # Logit transform keeps the interval inside [0, 1] near AUC ~ 1.
    a = float(np.clip(auc, 1e-6, 1 - 1e-6))
    logit = math.log(a / (1 - a))
    se_logit = se / (a * (1 - a))
    lo = 1 / (1 + math.exp(-(logit - z * se_logit)))
    hi = 1 / (1 + math.exp(-(logit + z * se_logit)))
    return (auc, lo, hi)


@dataclass
class ComparisonResult:
    metric: str
    value_a: float
    value_b: float
    difference: float
    ci: tuple[float, float]
    p_value: float
    significant: bool
    interpretation: str

    def to_dict(self) -> dict:
        return asdict(self)


def delong_test(scores_a: np.ndarray, scores_b: np.ndarray,
                labels: np.ndarray, alpha: float = 0.05,
                name_a: str = "A", name_b: str = "B") -> ComparisonResult:
    """Paired comparison of two correlated AUCs on the same cases."""
    y = (np.asarray(labels).ravel() > 0).astype(np.int64)
    s = np.stack([np.asarray(scores_a, np.float64).ravel(),
                  np.asarray(scores_b, np.float64).ravel()])
    aucs, v10, v01 = _delong_structural(s, y)
    m, n = v01.shape[1], v10.shape[1]
    S = np.cov(v01, ddof=1) / m + np.cov(v10, ddof=1) / n
    diff = float(aucs[0] - aucs[1])
    var = float(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    se = math.sqrt(max(var, 1e-12))

    from scipy.stats import norm
    z_stat = diff / se
    p = float(2 * (1 - norm.cdf(abs(z_stat))))
    zc = float(norm.ppf(1 - alpha / 2))
    ci = (diff - zc * se, diff + zc * se)

    better = name_a if diff > 0 else name_b
    interp = (f"{better} has the higher AUC by {abs(diff):.4f}; "
              + ("the difference is statistically significant"
                 if p < alpha else "the difference is not statistically significant")
              + f" (DeLong p = {p:.4g}).")
    return ComparisonResult("AUC", float(aucs[0]), float(aucs[1]), diff, ci,
                            p, p < alpha, interp)


def mcnemar_test(pred_a: np.ndarray, pred_b: np.ndarray, labels: np.ndarray,
                 alpha: float = 0.05, name_a: str = "A", name_b: str = "B"
                 ) -> ComparisonResult:
    """Paired comparison of two classifiers' error patterns."""
    y = np.asarray(labels).ravel()
    a = np.asarray(pred_a).ravel() == y
    b = np.asarray(pred_b).ravel() == y
    n01 = int((~a & b).sum())      # A wrong, B right
    n10 = int((a & ~b).sum())      # A right, B wrong

    from scipy.stats import binomtest, chi2
    if n01 + n10 == 0:
        return ComparisonResult("accuracy", float(a.mean()), float(b.mean()), 0.0,
                                (0.0, 0.0), 1.0, False,
                                "The two models make identical predictions.")
    if n01 + n10 < 25:
        p = float(binomtest(n10, n01 + n10, 0.5).pvalue)   # exact
        method = "exact binomial"
    else:
        stat = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)     # continuity-corrected
        p = float(1 - chi2.cdf(stat, 1))
        method = "chi-square with continuity correction"

    diff = float(a.mean() - b.mean())
    se = math.sqrt(max(n01 + n10, 1)) / len(y)
    from scipy.stats import norm
    zc = float(norm.ppf(1 - alpha / 2))
    better = name_a if diff > 0 else name_b
    interp = (f"{better} is more accurate by {abs(diff):.4f} "
              f"({n10} vs {n01} discordant cases); "
              + ("significant" if p < alpha else "not significant")
              + f" (McNemar {method}, p = {p:.4g}).")
    return ComparisonResult("accuracy", float(a.mean()), float(b.mean()), diff,
                            (diff - zc * se, diff + zc * se), p, p < alpha, interp)


# --------------------------------------------------------------------------
# Ordinal agreement
# --------------------------------------------------------------------------
def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray,
                             num_classes: int = NUM_GRADES) -> float:
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(np.asarray(y_true).ravel(),
                                   np.asarray(y_pred).ravel(),
                                   labels=list(range(num_classes)),
                                   weights="quadratic"))


def confusion(y_true: np.ndarray, y_pred: np.ndarray,
              num_classes: int = NUM_GRADES) -> np.ndarray:
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(np.asarray(y_true).ravel(), np.asarray(y_pred).ravel(),
                            labels=list(range(num_classes)))


def per_grade_recall(y_true: np.ndarray, y_pred: np.ndarray,
                     num_classes: int = NUM_GRADES) -> dict[int, Proportion]:
    yt = np.asarray(y_true).ravel(); yp = np.asarray(y_pred).ravel()
    out = {}
    for g in range(num_classes):
        m = yt == g
        out[g] = proportion(int((yp[m] == g).sum()), int(m.sum()))
    return out


def adjacent_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> Proportion:
    """Fraction predicted within one grade -- the usual inter-grader benchmark.

    Human retinal graders agree exactly on ICDR grade only ~60-75% of the
    time, but within one grade >90%.  Quoting exact-match accuracy alone
    against a human-derived reference standard therefore understates
    performance, and quoting it *without* the adjacent figure is the most
    common way DR papers mislead.
    """
    yt = np.asarray(y_true).ravel(); yp = np.asarray(y_pred).ravel()
    return proportion(int((np.abs(yt - yp) <= 1).sum()), yt.size)


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------
def bootstrap_ci(metric_fn, *arrays, n_boot: int = 2000, alpha: float = 0.05,
                 stratify: np.ndarray | None = None, seed: int = 0
                 ) -> tuple[float, float, float]:
    """Stratified bootstrap CI for an arbitrary metric.

    Stratifying on the label preserves the class balance in every resample;
    without it, resamples of a set with 30 grade-4 cases routinely contain 15
    or 50, and the resulting interval reflects that sampling artefact rather
    than the model.
    """
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    point = float(metric_fn(*arrays))

    if stratify is None:
        idx_pool = [np.arange(n)]
    else:
        stratify = np.asarray(stratify).ravel()
        idx_pool = [np.nonzero(stratify == v)[0] for v in np.unique(stratify)]

    vals = []
    for _ in range(n_boot):
        take = np.concatenate([rng.choice(ix, size=len(ix), replace=True)
                               for ix in idx_pool if len(ix)])
        try:
            vals.append(float(metric_fn(*[a[take] for a in arrays])))
        except Exception:
            continue
    if not vals:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


# --------------------------------------------------------------------------
# Top-level evaluation
# --------------------------------------------------------------------------
def evaluate_grading(y_true: np.ndarray, y_pred: np.ndarray,
                     referable_scores: np.ndarray,
                     threshold: float,
                     alpha: float = 0.05, n_boot: int = 2000,
                     seed: int = 0) -> dict:
    """Full grading evaluation: ordinal agreement + the referable-DR decision."""
    yt = np.asarray(y_true).ravel().astype(int)
    yp = np.asarray(y_pred).ravel().astype(int)
    ref_true = (yt >= REFERABLE_THRESHOLD).astype(int)

    qwk, qwk_lo, qwk_hi = bootstrap_ci(
        lambda a, b: quadratic_weighted_kappa(a, b), yt, yp,
        n_boot=n_boot, alpha=alpha, stratify=yt, seed=seed)

    bm = binary_metrics(referable_scores, ref_true, threshold, alpha)
    adj = adjacent_accuracy(yt, yp)
    exact = proportion(int((yt == yp).sum()), yt.size, alpha)

    return {
        "n": int(yt.size),
        "grade_distribution": {int(g): int((yt == g).sum()) for g in range(NUM_GRADES)},
        "qwk": {"value": qwk, "lower": qwk_lo, "upper": qwk_hi},
        "exact_accuracy": exact.to_dict(),
        "adjacent_accuracy": adj.to_dict(),
        "per_grade_recall": {int(g): p.to_dict() for g, p in per_grade_recall(yt, yp).items()},
        "confusion_matrix": confusion(yt, yp).tolist(),
        "referable": bm.to_dict(),
        "referable_summary": bm.summary(),
    }
