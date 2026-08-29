"""Stage 4: calibration, operating-point selection, and the full validation report.

Discipline enforced here (this is the whole point of the script):

* **val** fits the temperature and selects the referral threshold.
* **test** is the internal held-out estimate.
* **external** is the zero-shot generalisation estimate (Messidor-2 for real
  data, the domain-shifted phantom split otherwise). Nothing is fitted on it.

Selecting a threshold on the same data you report it on is the single most
common way DR papers inflate their numbers; the split roles are therefore
positional arguments of the flow, not a convention someone has to remember.

    python scripts/validate.py --cohort data/cohort_synth \\
        --grader outputs/grader_fusion/best.pt --seg outputs/segmentation/best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from drscreen.constants import (ICDR_GRADES, REFERABLE_THRESHOLD,
                                TARGET_SENSITIVITY, TARGET_SPECIFICITY)
from drscreen.data.cohort import CohortDataset, clinical_from_batch
from drscreen.evaluation.ablation import compare_arms, format_table, save_report
from drscreen.evaluation.metrics import evaluate_grading, binary_metrics
from drscreen.models.calibration import (IsotonicCalibrator, TemperatureScaler,
                                         calibration_report, select_threshold,
                                         selective_risk_curve)
from drscreen.models.grader import (DRGrader, corn_class_probs, corn_predict,
                                    referable_prob)
from drscreen.training import collect_logits


def load_grader(path: Path) -> tuple[DRGrader, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = DRGrader(backbone=ck.get("backbone", "tf_efficientnet_b0"),
                     pretrained=False,
                     use_clinical=bool(ck.get("use_clinical", True)),
                     use_image=bool(ck.get("use_image", True)))
    model.load_state_dict(ck["model"])
    return model.eval(), ck


def run_split(model, cohort, split, size, use_clinical, workers, device):
    ds = CohortDataset(cohort, split, size=size, train=False, augment=False,
                       with_features=use_clinical)
    out = collect_logits(model, ds, num_workers=workers, device=device,
                         clinical_fn=clinical_from_batch if use_clinical else None)
    return out["logits"], out["labels"], ds


def summarise(logits: torch.Tensor, labels: torch.Tensor, temperature: float,
              threshold: float, iso=None) -> dict:
    z = logits / temperature
    probs = corn_class_probs(z).numpy()
    grades = corn_predict(z).numpy()
    ref = referable_prob(z).numpy()
    if iso is not None:
        ref = iso(ref)
    y = labels.numpy()
    res = evaluate_grading(y, grades, ref, threshold)
    res["calibration"] = calibration_report(
        ref, (y >= REFERABLE_THRESHOLD).astype(int), temperature).to_dict()
    entropy = -(np.clip(probs, 1e-9, 1) * np.log(np.clip(probs, 1e-9, 1))).sum(1)
    res["selective"] = selective_risk_curve(
        ref, (y >= REFERABLE_THRESHOLD).astype(int), entropy, threshold)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--grader", type=Path, default=Path("outputs/grader_fusion/best.pt"))
    ap.add_argument("--seg", type=Path, default=Path("outputs/segmentation/best.pt"))
    ap.add_argument("--arms", nargs="*", default=[],
                    help="extra arms as name=path/to/best.pt for the ablation")
    ap.add_argument("--out", type=Path, default=Path("outputs/validation"))
    ap.add_argument("--artifacts", type=Path, default=Path("outputs/artifacts"))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--prevalence", type=float, default=0.18,
                    help="deployment prevalence of referable DR, for PPV/NPV")
    ap.add_argument("--seg-cohort", type=Path, default=None,
                    help="cohort holding the segmentation splits, if separate "
                         "(the lesion model may run at a higher resolution)")
    ap.add_argument("--explain-n", type=int, default=40,
                    help="images used for the explanation-faithfulness study")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("DR SCREENING - CLINICAL VALIDATION")
    print("=" * 78)

    model, ck = load_grader(a.grader)
    use_clinical = bool(ck.get("use_clinical", True))
    print(f"\nGrader: {a.grader}  (arm={ck.get('arm','?')}, "
          f"backbone={ck.get('backbone','?')}, clinical={use_clinical})")

    # ---------------- 1. calibration on VAL --------------------------------
    print("\n[1/5] Calibration and operating point (fitted on VAL only)")
    v_logits, v_labels, _ = run_split(model, a.cohort, "val", a.size,
                                      use_clinical, a.workers, a.device)
    scaler = TemperatureScaler()
    T = scaler.fit(v_logits, v_labels)
    print(f"  temperature T = {T:.4f}")

    ref_val_raw = referable_prob(v_logits).numpy()
    ref_val_cal = referable_prob(v_logits / T).numpy()
    y_val_ref = (v_labels.numpy() >= REFERABLE_THRESHOLD).astype(int)

    cal_before = calibration_report(ref_val_raw, y_val_ref, 1.0)
    cal_after = calibration_report(ref_val_cal, y_val_ref, T)
    print(f"  ECE  {cal_before.ece:.4f} -> {cal_after.ece:.4f}")
    print(f"  MCE  {cal_before.mce:.4f} -> {cal_after.mce:.4f}")
    print(f"  Brier {cal_before.brier:.4f} -> {cal_after.brier:.4f}")

    # Temperature targets the multiclass CORN NLL; the referral decision uses
    # the binary P(referable). Recalibrate that number directly with isotonic
    # regression, which is monotone and so cannot reorder the review queue.
    #
    # Whether to adopt it must be decided OUT OF FOLD. Isotonic is a free-form
    # monotone fit: scored on the same data it was fitted to, it drives ECE to
    # ~0 by construction and would always look like an improvement. So fit on
    # one half of val, score the other, both ways, and compare like for like.
    rng = np.random.default_rng(0)
    fold = rng.permutation(len(ref_val_cal)) % 2
    oof = np.empty_like(ref_val_cal)
    for k in (0, 1):
        fit_m, score_m = (fold != k), (fold == k)
        if y_val_ref[fit_m].sum() in (0, int(fit_m.sum())):
            oof[score_m] = ref_val_cal[score_m]      # degenerate fold
            continue
        oof[score_m] = IsotonicCalibrator(min_samples=50).fit(
            ref_val_cal[fit_m], y_val_ref[fit_m])(ref_val_cal[score_m])

    ece_oof = calibration_report(oof, y_val_ref, T).ece
    use_iso = ece_oof < cal_after.ece
    print(f"  ECE for isotonic on P(referable), out-of-fold: {ece_oof:.4f} "
          f"vs {cal_after.ece:.4f} temperature-only "
          f"-> {'adopted' if use_iso else 'rejected'}")

    iso = IsotonicCalibrator(min_samples=50).fit(ref_val_cal, y_val_ref) if use_iso else None
    cal_iso = calibration_report(oof, y_val_ref, T)
    ref_val_final = iso(ref_val_cal) if iso is not None else ref_val_cal
    op = select_threshold(ref_val_final, y_val_ref, TARGET_SENSITIVITY,
                          TARGET_SPECIFICITY, prevalence=a.prevalence)
    print(f"  threshold = {op.threshold:.4f}   "
          f"sens {op.sensitivity:.3f}  spec {op.specificity:.3f}")
    print(f"  {op.rationale}")

    # ---------------- 2. internal test -------------------------------------
    print("\n[2/5] Internal held-out test")
    t_logits, t_labels, _ = run_split(model, a.cohort, "test", a.size,
                                      use_clinical, a.workers, a.device)
    test_res = summarise(t_logits, t_labels, T, op.threshold, iso)
    print(f"  n = {test_res['n']}   {test_res['referable_summary']}")
    print(f"  QWK {test_res['qwk']['value']:.4f} "
          f"[{test_res['qwk']['lower']:.4f}, {test_res['qwk']['upper']:.4f}]")
    print(f"  ECE {test_res['calibration']['ece']:.4f}")

    # ---------------- 3. external (zero-shot) ------------------------------
    print("\n[3/5] External validation (zero-shot; nothing fitted on this split)")
    try:
        e_logits, e_labels, _ = run_split(model, a.cohort, "external", a.size,
                                          use_clinical, a.workers, a.device)
        n_labelled = int((e_labels.numpy() >= 0).sum())
        if n_labelled == 0:
            raise ValueError(
                f"the external split has {len(e_labels)} images but no grades. "
                "Messidor-2 from ADCIS ships images plus an eye-pairing CSV; the "
                "adjudicated reference standard (Krause et al. 2018) is a "
                "separate download. Inference and reports work; metrics cannot "
                "be computed. See docs/DATASETS.md.")
        ext_res = summarise(e_logits, e_labels, T, op.threshold, iso)
        print(f"  n = {ext_res['n']}   {ext_res['referable_summary']}")
        print(f"  QWK {ext_res['qwk']['value']:.4f}")
        print(f"  ECE {ext_res['calibration']['ece']:.4f}")
        drop = test_res["referable"]["auc"] - ext_res["referable"]["auc"]
        print(f"  AUC drop internal -> external: {drop:+.4f}")
    except Exception as e:
        print(f"  NOT EVALUATED -- {e}")
        ext_res = None

    # ---------------- 4. ablation -------------------------------------------
    print("\n[4/5] Ablation: does the integrated pipeline beat single techniques?")
    arms: dict[str, dict] = {}
    y_test = t_labels.numpy()

    # Same transform as the threshold was chosen under, or the operating point
    # is being applied to a different number line than it was selected on.
    _fusion_ref = referable_prob(t_logits / T).numpy()
    if iso is not None:
        _fusion_ref = iso(_fusion_ref)
    arms["fusion"] = {
        "grades": corn_predict(t_logits / T).numpy().tolist(),
        "referable_scores": _fusion_ref.tolist(),
        "threshold": op.threshold}

    for spec in a.arms:
        if "=" not in spec:
            print(f"  ignoring malformed arm spec: {spec}")
            continue
        name, path = spec.split("=", 1)
        p = Path(path)
        if not p.exists():
            print(f"  arm '{name}': checkpoint not found ({p}), skipped")
            continue
        m, mck = load_grader(p)
        uc = bool(mck.get("use_clinical", True))
        lg, lb, _ = run_split(m, a.cohort, "test", a.size, uc, a.workers, a.device)
        # Each arm gets its own threshold chosen on VAL, so the comparison is
        # between best-configured systems rather than between one tuned model
        # and others handicapped by a threshold that does not suit them.
        vlg, vlb, _ = run_split(m, a.cohort, "val", a.size, uc, a.workers, a.device)
        s = TemperatureScaler(); t_arm = s.fit(vlg, vlb)
        v_ref = referable_prob(vlg / t_arm).numpy()
        v_y = (vlb.numpy() >= REFERABLE_THRESHOLD).astype(int)
        iso_arm = IsotonicCalibrator().fit(v_ref, v_y)
        if calibration_report(iso_arm(v_ref), v_y, t_arm).ece >= \
           calibration_report(v_ref, v_y, t_arm).ece:
            iso_arm = None
        t_ref = referable_prob(lg / t_arm).numpy()
        op_arm = select_threshold(iso_arm(v_ref) if iso_arm else v_ref, v_y,
                                  TARGET_SENSITIVITY, TARGET_SPECIFICITY)
        arms[name] = {"grades": corn_predict(lg / t_arm).numpy().tolist(),
                      "referable_scores": (iso_arm(t_ref) if iso_arm else t_ref).tolist(),
                      "threshold": op_arm.threshold}
        print(f"  loaded arm '{name}' (T={t_arm:.3f}, thr={op_arm.threshold:.3f})")

    # Rule-based arm: needs the cached clinical features, no network involved.
    rule = rule_based_arm(a.cohort, "test", y_test)
    if rule:
        arms["rule_based"] = rule
        print("  built arm 'rule_based' from cached lesion features")

    ablation = compare_arms(y_test, arms, reference="fusion") if len(arms) > 1 else None
    if ablation:
        print()
        print(format_table(ablation))
    else:
        print("  only one arm available; train the ablation arms to populate this "
              "section (see scripts/train_grader.py --arm)")

    # ---------------- 5. explanation quality ---------------------------------
    print("\n[5/6] Explanation quality (is the heatmap worth showing a clinician?)")
    xq_d = None
    try:
        xq = evaluate_explanation_quality(model, a.cohort, "test", a.size,
                                          use_clinical, a.device,
                                          n=a.explain_n, seed=0)
        print(f"  {xq.verdict()}")
        print(f"  attention sparsity (Gini) {xq.sparsity:.3f} "
              f"over {xq.n_evaluated} referable images")
        xq_d = xq.to_dict()
    except Exception as e:
        print(f"  skipped: {type(e).__name__}: {e}")

    # ---------------- 6. write artefacts -------------------------------------
    print("\n[6/6] Writing artefacts")
    report = {
        "grader_checkpoint": str(a.grader),
        "arm": ck.get("arm"),
        "calibration": {"temperature": T,
                        "isotonic_applied": bool(iso is not None),
                        "before": cal_before.to_dict(),
                        "after": (cal_iso if iso is not None else cal_after).to_dict(),
                        "after_temperature_only": cal_after.to_dict()},
        "operating_point": op.to_dict(),
        "targets": {"sensitivity": TARGET_SENSITIVITY,
                    "specificity": TARGET_SPECIFICITY,
                    "referable_threshold_grade": REFERABLE_THRESHOLD},
        "internal_test": test_res,
        "external": ext_res,
        "ablation": ablation,
        "explanation_quality": xq_d,
    }
    (a.out / "validation.json").write_text(json.dumps(report, indent=2, default=float))
    if ablation:
        save_report(ablation, a.out / "ablation.json")
        (a.out / "ablation.txt").write_text(format_table(ablation))
    (a.out / "summary.md").write_text(markdown_summary(report))

    from drscreen.evaluation.model_card import generate as make_card
    note = ("These figures were measured on procedurally generated fundus "
            "phantoms, not patients. They demonstrate that the pipeline is "
            "correctly wired and that the validation machinery works; they are "
            "NOT clinical performance. Re-run on APTOS/IDRiD with Messidor-2 "
            "held out for any clinical claim."
            if "synth" in str(a.cohort) else "")
    (a.out / "model_card.md").write_text(
        make_card(report, {"data_note": note} if note else None), encoding="utf-8")

    # Deployable artefact bundle
    a.artifacts.mkdir(parents=True, exist_ok=True)
    import shutil
    if a.seg.exists():
        shutil.copy(a.seg, a.artifacts / "segmentation.pt")
    shutil.copy(a.grader, a.artifacts / "grader.pt")
    lesion_thr = {}
    if a.seg.exists():
        try:
            lesion_thr = fit_lesion_thresholds(a.seg, a.seg_cohort or a.cohort, a.device)
            if lesion_thr:
                print(f"  lesion thresholds (F1-optimal on held-out IDRiD): {lesion_thr}")
        except Exception as e:
            print(f"  lesion-threshold fit skipped: {type(e).__name__}: {e}")

    seg_size = a.size
    if a.seg.exists():
        _sck = torch.load(a.seg, map_location="cpu", weights_only=False)
        seg_size = int(_sck.get("size", a.size))

    (a.artifacts / "pipeline.json").write_text(json.dumps({
        "referral_threshold": op.threshold,
        "temperature": T,
        "calibrator": (iso.to_dict() if iso is not None else {"kind": "identity"}),
        "lesion_thresholds": lesion_thr or None,
        "size": a.size,
        "seg_size": seg_size,
        "backbone": ck.get("backbone", "tf_efficientnet_b0"),
        "defer_band": [max(0.0, op.threshold - 0.15), min(1.0, op.threshold + 0.15)],
        "model_version": "drscreen-1.0.0",
        "validated_on": {"internal_n": test_res["n"],
                         "external_n": ext_res["n"] if ext_res else 0},
    }, indent=2))
    print(f"  {a.out/'validation.json'}")
    print(f"  {a.out/'summary.md'}")
    print(f"  {a.out/'model_card.md'}")
    print(f"  deployable bundle -> {a.artifacts}")

    # Headline verdict
    print("\n" + "=" * 78)
    bm = test_res["referable"]
    sens_ok = bm["sensitivity"]["value"] >= TARGET_SENSITIVITY
    spec_ok = bm["specificity"]["value"] >= TARGET_SPECIFICITY
    print(f"TARGET: sensitivity >= {TARGET_SENSITIVITY:.0%}, "
          f"specificity >= {TARGET_SPECIFICITY:.0%} for referable DR")
    print(f"INTERNAL TEST: sensitivity {bm['sensitivity']['value']:.3f} "
          f"[{bm['sensitivity']['lower']:.3f}-{bm['sensitivity']['upper']:.3f}] "
          f"{'PASS' if sens_ok else 'FAIL'}")
    print(f"               specificity {bm['specificity']['value']:.3f} "
          f"[{bm['specificity']['lower']:.3f}-{bm['specificity']['upper']:.3f}] "
          f"{'PASS' if spec_ok else 'FAIL'}")
    if ext_res:
        eb = ext_res["referable"]
        print(f"EXTERNAL:      sensitivity {eb['sensitivity']['value']:.3f} "
              f"specificity {eb['specificity']['value']:.3f}")
    print("=" * 78)


def evaluate_explanation_quality(model, cohort: Path, split: str, size: int,
                                 use_clinical: bool, device: str,
                                 n: int = 40, seed: int = 0):
    """Faithfulness + clinical-correctness of the Grad-CAM++ explanations.

    Faithfulness (deletion/insertion) needs no ground truth. The pointing game
    and attention-on-lesions do, and they are the ones that catch a model that
    is confidently explaining the wrong thing -- so they run against the
    cohort's lesion masks wherever those exist.
    """
    from drscreen.data.cohort import CohortDataset
    from drscreen.explain.cam import compute_cam
    from drscreen.explain.faithfulness import evaluate_explanations
    from drscreen.training import device_of

    dev = device_of(device)
    model = model.to(dev).eval()
    ds = CohortDataset(cohort, split, size=size, train=False, augment=False,
                       with_masks=True, with_features=use_clinical)
    rng = np.random.default_rng(seed)
    # Explanations are only meaningful where there is something to explain, so
    # sample from referable cases rather than uniformly (a grade-0 image has no
    # lesion to point at, and scoring the pointing game on it is meaningless).
    referable = [i for i, r in enumerate(ds.records) if r.grade >= REFERABLE_THRESHOLD]
    pool = referable if len(referable) >= n else list(range(len(ds)))
    idxs = rng.choice(pool, size=min(n, len(pool)), replace=False)

    images, cams, clinicals, masks = [], [], [], []
    for i in idxs:
        b = ds[int(i)]
        x = b["image"].unsqueeze(0).to(dev)
        c = b["clinical"].unsqueeze(0).to(dev) if use_clinical else None
        fov = b["fov_mask"][0].numpy()
        cam = compute_cam(model, x, c, method="gradcam++", target="referable",
                          referable_index=REFERABLE_THRESHOLD - 1,
                          fov_mask=fov)
        images.append(x); cams.append(cam); clinicals.append(c)
        lm = b.get("lesion_mask")
        masks.append(lm.numpy().max(axis=0) if lm is not None else None)

    return evaluate_explanations(model, images, cams, clinicals, masks, steps=16)


def fit_lesion_thresholds(seg_ckpt: Path, cohort: Path, device: str = "auto",
                          split: str = "seg_val") -> dict:
    """F1-optimal per-class cut-points on the lesion probability maps.

    A blanket 0.5 is not a neutral default, it is an unfitted parameter. On
    held-out IDRiD the optimum sits at 0.85-0.95, and at 0.5 the exudate
    channel produced enough false positives on healthy APTOS retinas to trip
    the macular-oedema rule and label them urgent.

    Fitted on the segmentation validation split -- never on the grading test
    split the operating point is reported against.
    """
    import cv2
    from drscreen.data.cohort import read_manifest
    from drscreen.data.torch_data import to_tensor
    from drscreen.models.segmentation import build_unet
    from drscreen.constants import LESION_CLASSES
    from drscreen.training import device_of

    ck = torch.load(seg_ckpt, map_location="cpu", weights_only=False)
    seg_size = int(ck.get("size", 512))
    seg_cohort = cohort
    if not (seg_cohort / "manifest.jsonl").exists():
        return {}
    recs = [r for r in read_manifest(seg_cohort) if r.split == split and r.has_masks]
    if not recs:
        return {}

    dev = device_of(device)
    m = build_unet("lesion", width=int(ck.get("width", 24)))
    m.load_state_dict(ck["model"]); m = m.to(dev).eval()

    P, Y = [], []
    for r in recs:
        img = cv2.imread(str(seg_cohort / "images" / f"{r.uid}.png"))
        z = np.load(seg_cohort / "masks" / f"{r.uid}.npz")
        if img.shape[0] != seg_size:
            img = cv2.resize(img, (seg_size, seg_size), interpolation=cv2.INTER_CUBIC)
        with torch.no_grad():
            lo = m(to_tensor(img).unsqueeze(0).to(dev))
            if isinstance(lo, tuple):
                lo = lo[0]
            pr = torch.sigmoid(lo.float())[0].permute(1, 2, 0).cpu().numpy()
        gt = (z["lesions"] > 127).astype(np.uint8)
        if gt.shape[0] != pr.shape[0]:
            gt = cv2.resize(gt, pr.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        P.append(pr); Y.append(gt)
    P, Y = np.stack(P), np.stack(Y)

    out = {}
    for c, name in enumerate(LESION_CLASSES):
        if Y[..., c].sum() == 0:
            continue                      # class not annotated in this corpus
        best_f1, best_t = -1.0, 0.5
        for t in np.arange(0.05, 0.99, 0.05):
            pred = P[..., c] >= t
            tp = float((pred & (Y[..., c] > 0)).sum())
            fp = float((pred & (Y[..., c] == 0)).sum())
            fn = float((~pred & (Y[..., c] > 0)).sum())
            f1 = 2 * tp / max(2 * tp + fp + fn, 1.0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        out[name] = round(best_t, 2)
    return out


def rule_based_arm(cohort: Path, split: str, y_true: np.ndarray) -> dict | None:
    """Reconstruct the rule-based grade from cached clinical feature vectors."""
    from drscreen.data.cohort import read_manifest
    from drscreen.models.lesion_features import ClinicalFeatures, QUADRANTS
    from drscreen.constants import LESION_CLASSES

    recs = read_manifest(cohort, split)
    feat_dir = cohort / "features"
    if not feat_dir.exists():
        return None

    names = ClinicalFeatures.feature_names()
    idx = {n: i for i, n in enumerate(names)}
    grades, scores = [], []
    for r in recs:
        f = feat_dir / f"{r.uid}.npy"
        if not f.exists():
            grades.append(0); scores.append(0.0); continue
        v = np.load(f)
        counts = {c: float(np.expm1(v[idx[f"log_count_{c}"]])) for c in LESION_CLASSES}
        q_hem = v[idx["quadrants_with_hemorrhage"]]
        q_bead = v[idx["quadrants_with_beading"]]
        nvd, nve = v[idx["nv_at_disc"]], v[idx["nv_elsewhere"]]

        if counts["neovascularization"] >= 1 or nvd > 0.5 or nve > 0.5:
            g = 4
        elif q_hem >= 4 or q_bead >= 2:
            g = 3
        elif (counts["hemorrhage"] >= 1 or counts["hard_exudate"] >= 1
              or counts["soft_exudate"] >= 1 or counts["microaneurysm"] >= 8):
            g = 2
        elif counts["microaneurysm"] >= 1:
            g = 1
        else:
            g = 0
        grades.append(g)
        # A rule engine gives a discrete grade; to place it on an ROC it needs a
        # score. Total lesion burden is the natural monotone surrogate, and it
        # is what a clinician's gestalt tracks.
        burden = sum(counts.values()) + 5 * counts["neovascularization"]
        scores.append(float(np.clip(np.log1p(burden) / np.log1p(120), 0, 1)))
    return {"grades": grades, "referable_scores": scores, "threshold": 0.5}


def markdown_summary(r: dict) -> str:
    t = r["internal_test"]; b = t["referable"]
    e = r.get("external")
    lines = [
        "# Clinical validation summary", "",
        f"Grader: `{r['grader_checkpoint']}` (arm: {r.get('arm')})", "",
        "## Targets", "",
        f"- Sensitivity for referable DR (grade >= {r['targets']['referable_threshold_grade']}): "
        f">= {r['targets']['sensitivity']:.0%}",
        f"- Specificity: >= {r['targets']['specificity']:.0%}", "",
        "## Operating point", "",
        f"- Threshold on P(referable): **{r['operating_point']['threshold']:.4f}** "
        f"(selected on the validation split only)",
        f"- Temperature: **{r['calibration']['temperature']:.4f}**",
        f"- Rationale: {r['operating_point']['rationale']}", "",
        "## Calibration", "",
        "| metric | before | after |", "|---|---|---|",
        f"| ECE | {r['calibration']['before']['ece']:.4f} | {r['calibration']['after']['ece']:.4f} |",
        f"| MCE | {r['calibration']['before']['mce']:.4f} | {r['calibration']['after']['mce']:.4f} |",
        f"| Brier | {r['calibration']['before']['brier']:.4f} | {r['calibration']['after']['brier']:.4f} |",
        "", "## Internal held-out test", "",
        f"n = {t['n']}", "",
        "| metric | value | 95% CI |", "|---|---|---|",
        f"| Sensitivity | {b['sensitivity']['value']:.4f} | "
        f"{b['sensitivity']['lower']:.4f}-{b['sensitivity']['upper']:.4f} |",
        f"| Specificity | {b['specificity']['value']:.4f} | "
        f"{b['specificity']['lower']:.4f}-{b['specificity']['upper']:.4f} |",
        f"| AUC | {b['auc']:.4f} | {b['auc_ci'][0]:.4f}-{b['auc_ci'][1]:.4f} |",
        f"| QWK | {t['qwk']['value']:.4f} | {t['qwk']['lower']:.4f}-{t['qwk']['upper']:.4f} |",
        f"| Exact accuracy | {t['exact_accuracy']['value']:.4f} | "
        f"{t['exact_accuracy']['lower']:.4f}-{t['exact_accuracy']['upper']:.4f} |",
        f"| Within-one-grade | {t['adjacent_accuracy']['value']:.4f} | "
        f"{t['adjacent_accuracy']['lower']:.4f}-{t['adjacent_accuracy']['upper']:.4f} |",
        "",
    ]
    if e:
        eb = e["referable"]
        lines += [
            "## External validation (zero-shot)", "",
            f"n = {e['n']}. Nothing was fitted on this split.", "",
            "| metric | value | 95% CI |", "|---|---|---|",
            f"| Sensitivity | {eb['sensitivity']['value']:.4f} | "
            f"{eb['sensitivity']['lower']:.4f}-{eb['sensitivity']['upper']:.4f} |",
            f"| Specificity | {eb['specificity']['value']:.4f} | "
            f"{eb['specificity']['lower']:.4f}-{eb['specificity']['upper']:.4f} |",
            f"| AUC | {eb['auc']:.4f} | {eb['auc_ci'][0]:.4f}-{eb['auc_ci'][1]:.4f} |",
            f"| QWK | {e['qwk']['value']:.4f} | {e['qwk']['lower']:.4f}-{e['qwk']['upper']:.4f} |",
            "",
            f"AUC change internal -> external: "
            f"**{eb['auc'] - b['auc']:+.4f}**", "",
        ]
    if r.get("ablation"):
        lines += ["## Ablation", "", "```", format_table(r["ablation"]), "```", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
