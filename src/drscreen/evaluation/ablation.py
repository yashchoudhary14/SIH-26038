"""Ablation study: does the integrated pipeline beat any single technique?

The problem statement asks for exactly this claim, so it has to be *measured*,
with paired statistical tests on the same cases -- not asserted from two
numbers in a table.

Arms compared
-------------
===================  =====================================================
arm                  description
===================  =====================================================
``rule_based``       Classical CV only: segmentation -> lesion counts ->
                     published ICDR criteria (the 4-2-1 rule etc.).
``cnn_only``         Deep learning only: ordinal CNN on the image.
``clinical_only``    Lesion features only, through the ordinal head.
``fusion``           The integrated system: CNN + clinical features.
``no_preprocess``    Fusion, but on raw RGB instead of the adaptive
                     enhancement -- isolates the preprocessing contribution.
``no_calibration``   Fusion, uncalibrated -- isolates the calibration
                     contribution (changes ECE, not AUC).
===================  =====================================================

Every pairwise comparison against ``fusion`` reports a DeLong test on AUC and
a McNemar test on the referral decision, both paired on the same cases, so
the claim "the integrated pipeline outperforms any single technique" either
survives with a p-value attached or it does not get made.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from ..constants import REFERABLE_THRESHOLD
from .metrics import (binary_metrics, delong_test, mcnemar_test,
                      quadratic_weighted_kappa, bootstrap_ci, evaluate_grading)


@dataclass
class ArmResult:
    name: str
    description: str
    grades: list = field(default_factory=list)
    referable_scores: list = field(default_factory=list)
    threshold: float = 0.5
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("grades"); d.pop("referable_scores")
        return d


ARM_DESCRIPTIONS = {
    "rule_based": "Classical CV: lesion segmentation to ICDR criteria (no deep grader)",
    "cnn_only": "Deep learning only: ordinal CNN on the enhanced image",
    "clinical_only": "Lesion-feature vector only, through the ordinal head",
    "fusion": "Integrated: CNN image features fused with clinical lesion features",
    "no_preprocess": "Integrated, but trained/run on raw RGB (no adaptive enhancement)",
    "no_calibration": "Integrated, without temperature scaling",
}


def compare_arms(y_true: np.ndarray, arms: dict[str, dict],
                 reference: str = "fusion", alpha: float = 0.05,
                 n_boot: int = 1000) -> dict:
    """Evaluate every arm and test each against `reference` on the same cases.

    ``arms[name]`` must contain ``{"grades": ..., "referable_scores": ...,
    "threshold": ...}``.
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    ref_true = (y_true >= REFERABLE_THRESHOLD).astype(int)

    results: dict[str, ArmResult] = {}
    for name, d in arms.items():
        grades = np.asarray(d["grades"]).ravel().astype(int)
        scores = np.asarray(d["referable_scores"], np.float64).ravel()
        thr = float(d.get("threshold", 0.5))
        bm = binary_metrics(scores, ref_true, thr, alpha)
        qwk, qlo, qhi = bootstrap_ci(quadratic_weighted_kappa, y_true, grades,
                                     n_boot=n_boot, stratify=y_true)
        results[name] = ArmResult(
            name=name, description=ARM_DESCRIPTIONS.get(name, ""),
            grades=grades.tolist(), referable_scores=scores.tolist(), threshold=thr,
            metrics={
                "auc": bm.auc, "auc_ci": list(bm.auc_ci),
                "sensitivity": bm.sensitivity.to_dict(),
                "specificity": bm.specificity.to_dict(),
                "ppv": bm.ppv.to_dict(), "npv": bm.npv.to_dict(),
                "youden_j": bm.youden_j, "auprc": bm.auprc,
                "qwk": {"value": qwk, "lower": qlo, "upper": qhi},
                "exact_accuracy": float((grades == y_true).mean()),
                "adjacent_accuracy": float((np.abs(grades - y_true) <= 1).mean()),
                "meets_targets": bool(bm.meets_sensitivity_target and bm.meets_specificity_target),
            })

    comparisons = []
    if reference in results:
        ref = results[reference]
        for name, arm in results.items():
            if name == reference:
                continue
            dl = delong_test(np.asarray(ref.referable_scores),
                             np.asarray(arm.referable_scores), ref_true,
                             alpha, name_a=reference, name_b=name)
            mc = mcnemar_test(
                (np.asarray(ref.referable_scores) >= ref.threshold).astype(int),
                (np.asarray(arm.referable_scores) >= arm.threshold).astype(int),
                ref_true, alpha, name_a=reference, name_b=name)
            comparisons.append({
                "reference": reference, "arm": name,
                "auc_test": dl.to_dict(), "decision_test": mc.to_dict(),
                "reference_better": bool(dl.difference > 0),
                "significant": bool(dl.significant),
            })

    beats_all = bool(comparisons) and all(
        c["reference_better"] for c in comparisons)
    beats_all_sig = bool(comparisons) and all(
        c["reference_better"] and c["significant"] for c in comparisons)

    return {
        "n": int(y_true.size),
        "reference": reference,
        "arms": {k: v.to_dict() for k, v in results.items()},
        "comparisons": comparisons,
        "integrated_beats_every_single_technique": beats_all,
        "integrated_beats_every_single_technique_significantly": beats_all_sig,
        "verdict": _verdict(reference, comparisons, beats_all, beats_all_sig),
    }


def _verdict(reference: str, comparisons: list, beats_all: bool, sig: bool) -> str:
    if not comparisons:
        return "No comparison arms were supplied."
    if sig:
        return (f"The integrated pipeline ({reference}) has a higher referable-DR AUC "
                f"than every single-technique arm, and every difference is "
                f"statistically significant (DeLong, alpha=0.05).")
    if beats_all:
        losers = [c["arm"] for c in comparisons if not c["significant"]]
        return (f"The integrated pipeline ({reference}) has the highest referable-DR AUC "
                f"of all arms, but the margin over {', '.join(losers)} is not "
                f"statistically significant at this sample size.")
    beaten = [c["arm"] for c in comparisons if not c["reference_better"]]
    return (f"The integrated pipeline ({reference}) does NOT beat every single "
            f"technique: {', '.join(beaten)} scored higher. This must be "
            f"reported as-is.")


def preprocessing_ablation(evaluator, variants: dict[str, dict],
                           y_true: np.ndarray) -> dict:
    """Compare preprocessing configurations with the model held fixed.

    ``evaluator(variant_cfg) -> (grades, referable_scores)``.
    """
    arms = {}
    for name, cfg in variants.items():
        grades, scores = evaluator(cfg)
        arms[name] = {"grades": grades, "referable_scores": scores, "threshold": 0.5}
    return compare_arms(y_true, arms, reference=list(variants)[0])


def save_report(result: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, default=float))
    return p


def format_table(result: dict) -> str:
    """Plain-text table for the console and the README."""
    rows = []
    hdr = (f"{'arm':16s} {'AUC':>7s} {'95% CI':>17s} {'Sens':>7s} {'Spec':>7s} "
           f"{'QWK':>7s} {'targets':>8s}")
    rows.append(hdr)
    rows.append("-" * len(hdr))
    order = sorted(result["arms"].items(),
                   key=lambda kv: kv[1]["metrics"]["auc"], reverse=True)
    for name, arm in order:
        m = arm["metrics"]
        ci = m["auc_ci"]
        star = " *" if name == result["reference"] else "  "
        rows.append(f"{name+star:16s} {m['auc']:7.4f} "
                    f"[{ci[0]:.4f},{ci[1]:.4f}] "
                    f"{m['sensitivity']['value']:7.3f} {m['specificity']['value']:7.3f} "
                    f"{m['qwk']['value']:7.4f} "
                    f"{'PASS' if m['meets_targets'] else 'fail':>8s}")
    rows.append("")
    rows.append(f"* = integrated pipeline ({result['reference']})")
    rows.append("")
    rows.append(result["verdict"])
    if result["comparisons"]:
        rows.append("")
        rows.append("Paired tests vs the integrated pipeline:")
        for c in result["comparisons"]:
            rows.append(f"  vs {c['arm']:16s} {c['auc_test']['interpretation']}")
    return "\n".join(rows)
