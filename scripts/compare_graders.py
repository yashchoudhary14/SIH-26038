"""Head-to-head comparison of two graders on images held out from both.

Retraining on a re-shuffled pool makes the obvious comparison invalid. The two
models were validated on different test splits -- 631 images against 1,852, with
different corpora and different grade mixes -- so their headline numbers are not
measuring the same thing, and the apparent movement is mostly the test set
changing underneath, not the model.

Worse, the naive fix is also wrong. Scoring the old model on the new test split
contaminates the comparison in the other direction: the re-shuffle moved 850 of
those 1,852 images out of the old training split, and the old model memorised
them.

What is left is the intersection that is genuinely blind to both: records whose
``origin_split`` was the old ``test`` or ``external`` -- never trained on, never
used to fit the old model's threshold -- and which the re-shuffle placed in the
new ``test`` split, so the new model has not seen them either. Each model is
scored with its own fitted temperature, calibrator and operating point, because
that is how each would actually be deployed.

    python scripts/compare_graders.py \\
        --cohort data/cohort_all \\
        --a outputs/artifacts_pre_messidor --a-name "pre-pool (deployed)" \\
        --b outputs/artifacts_all --b-name "pooled + Messidor-2"
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from drscreen.constants import REFERABLE_THRESHOLD
from drscreen.evaluation.metrics import delong_test, mcnemar_test, proportion
from drscreen.models.calibration import IsotonicCalibrator

#: Old-split labels that were never seen by the pre-pool model in any capacity.
#: ``val`` is deliberately excluded: the old model's temperature and referral
#: threshold were fitted on it, so it is held out from the weights but not from
#: the decision rule.
BLIND_ORIGINS = ("test", "external")


def _load_validate():
    path = Path(__file__).resolve().parent / "validate.py"
    spec = importlib.util.spec_from_file_location("_validate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cfg(bundle: Path) -> dict:
    return json.loads((bundle / "pipeline.json").read_text(encoding="utf-8"))


def _v(x):
    return x["value"] if isinstance(x, dict) and "value" in x else x


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--a", type=Path, required=True, help="artifact bundle A")
    ap.add_argument("--b", type=Path, required=True, help="artifact bundle B")
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--split", default="test")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--all-of-split", action="store_true",
                    help="score the whole split instead of the blind intersection "
                         "(contaminated for A; use only to show the size of the effect)")
    ap.add_argument("--by-source", action="store_true",
                    help="repeat the headline metrics within each source corpus. "
                         "Guards against Simpson's paradox: an overall win driven "
                         "entirely by the comparison set's mix rather than by the "
                         "model being better on any one corpus.")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    V = _load_validate()
    cfgs = {a.a_name: _cfg(a.a), a.b_name: _cfg(a.b)}

    scored: dict[str, dict] = {}
    ds = None
    for name, bundle in ((a.a_name, a.a), (a.b_name, a.b)):
        model, ck = V.load_grader(bundle / "grader.pt")
        use_clinical = bool(ck.get("use_clinical", True))
        logits, labels, ds = V.run_split(model, a.cohort, a.split, a.size,
                                         use_clinical, a.workers, a.device)
        scored[name] = {"logits": logits, "labels": labels}
        del model

    origins = [(r.meta or {}).get("origin_split", r.split) for r in ds.records]
    if a.all_of_split:
        idx = list(range(len(origins)))
        blurb = f"ENTIRE '{a.split}' split -- contaminated for {a.a_name}"
    else:
        idx = [i for i, o in enumerate(origins) if o in BLIND_ORIGINS]
        blurb = (f"blind intersection: old split in {BLIND_ORIGINS}, "
                 f"new split '{a.split}'")
    if not idx:
        print("No records in the comparison set.")
        return 1
    sel = torch.as_tensor(idx, dtype=torch.long)

    labels_np = scored[a.a_name]["labels"][sel].numpy()
    keep = labels_np >= 0
    y = labels_np[keep]
    print(f"Comparison set: n={len(y)}  ({blurb})")
    print("  grades " + str({int(g): int((y == g).sum()) for g in range(5)}))
    print(f"  referable prevalence {float((y >= REFERABLE_THRESHOLD).mean()):.1%}\n")

    results: dict[str, dict] = {}
    for name in (a.a_name, a.b_name):
        c = cfgs[name]
        res = V.summarise(scored[name]["logits"][sel], scored[name]["labels"][sel],
                          float(c["temperature"]), float(c["referral_threshold"]),
                          IsotonicCalibrator.from_dict(c.get("calibrator")),
                          c.get("grade_thresholds"))
        results[name] = res

    rows = [
        ("referable sensitivity", lambda r: _v(r["referable"]["sensitivity"])),
        ("referable specificity", lambda r: _v(r["referable"]["specificity"])),
        ("referable AUC", lambda r: r["referable"]["auc"]),
        ("sight-threat sensitivity",
         lambda r: _v(r["severity"]["sight_threatening"]["sensitivity"])),
        ("sight-threat specificity",
         lambda r: _v(r["severity"]["sight_threatening"]["specificity"])),
        ("QWK", lambda r: _v(r["qwk"])),
        ("exact accuracy", lambda r: _v(r.get("accuracy", r.get("exact_accuracy")))),
        ("within-one-grade", lambda r: _v(r["adjacent_accuracy"])),
        ("ECE", lambda r: r["calibration"]["ece"]),
    ]
    w = max(len(a.a_name), len(a.b_name), 12) + 2
    print(f"{'metric':<26}{a.a_name:>{w}}{a.b_name:>{w}}{'delta':>10}")
    print("-" * (26 + 2 * w + 10))
    table = {}
    for label, fn in rows:
        try:
            va, vb = fn(results[a.a_name]), fn(results[a.b_name])
        except (KeyError, TypeError):
            continue
        if va is None or vb is None:
            continue
        va, vb = float(va), float(vb)
        table[label] = {a.a_name: va, a.b_name: vb, "delta": vb - va}
        print(f"{label:<26}{va:>{w}.4f}{vb:>{w}.4f}{vb-va:>+10.4f}")

    print(f"\n{'per-grade recall':<26}{a.a_name:>{w}}{a.b_name:>{w}}{'delta':>10}")
    print("-" * (26 + 2 * w + 10))
    pg = {}
    for g in range(5):
        # summarise() keys this by int in memory; it only becomes a string
        # once it has been through JSON. Accept either.
        def _recall(name):
            d = results[name]["per_grade_recall"]
            return d.get(g, d.get(str(g)))

        ra, rb = _recall(a.a_name), _recall(a.b_name)
        va, vb = _v(ra), _v(rb)
        if va is None or vb is None:
            continue
        n_g = int((y == g).sum())
        pg[g] = {a.a_name: float(va), a.b_name: float(vb), "delta": float(vb) - float(va),
                 "n": n_g}
        print(f"{'grade '+str(g)+f'  (n={n_g})':<26}"
              f"{float(va):>{w}.4f}{float(vb):>{w}.4f}{float(vb)-float(va):>+10.4f}")

    # --- significance ------------------------------------------------------
    # Paired tests on the same cases: DeLong for the ranking (AUC), McNemar for
    # the realised referral decision at each model's own operating point. The
    # second is the one a screening programme actually experiences.
    print("\nPaired tests")
    sig: dict = {}
    sa = np.asarray(results[a.a_name]["referable"]["scores"], np.float64) \
        if "scores" in results[a.a_name]["referable"] else None
    if sa is None:
        # summarise() does not always retain raw scores; recompute them.
        from drscreen.models.grader import referable_prob
        def _scores(name):
            c = cfgs[name]
            z = scored[name]["logits"][sel] / float(c["temperature"])
            s = referable_prob(z).numpy()
            iso = IsotonicCalibrator.from_dict(c.get("calibrator"))
            return iso(s)[keep] if iso is not None else s[keep]
        sa, sb = _scores(a.a_name), _scores(a.b_name)
    y_bin = (y >= REFERABLE_THRESHOLD).astype(int)
    dl = delong_test(sa, sb, y_bin, name_a=a.a_name, name_b=a.b_name)
    print(f"  AUC     {dl.interpretation}")
    sig["delong_referable_auc"] = dl.to_dict()

    pred_a = (sa >= float(cfgs[a.a_name]["referral_threshold"])).astype(int)
    pred_b = (sb >= float(cfgs[a.b_name]["referral_threshold"])).astype(int)
    mc = mcnemar_test(pred_a, pred_b, y_bin)
    print(f"  Referral decision  {mc.interpretation}")
    sig["mcnemar_referral"] = mc.to_dict()

    # Small per-grade cells are the main way a comparison like this misleads:
    # grade 3 and 4 counts here are in the tens, so a swing of several points
    # is inside the sampling noise. State the interval rather than let the
    # delta column imply a precision it does not have.
    print()
    for g, row in pg.items():
        if row["n"] < 60:
            k = int(round(row[a.b_name] * row["n"]))
            ci = proportion(k, row["n"])
            lo, hi = ci.lower, ci.upper
            print(f"  grade {g}: n={row['n']}, so {a.b_name} recall "
                  f"{row[a.b_name]:.3f} carries a 95% CI of [{lo:.3f}, {hi:.3f}] "
                  f"-- treat the delta as indicative, not measured")

    by_source: dict = {}
    if a.by_source:
        # Per-source metrics go back through summarise() on the source's own
        # global indices, rather than slicing the already-filtered score arrays,
        # so exact accuracy and QWK are available on the same footing as the
        # pooled table -- and so the grade decoding uses each model's own
        # fitted cut-points rather than being recomputed by hand.
        def _refacc(r):
            se, sp = r["referable"]["sensitivity"], r["referable"]["specificity"]
            tp, pos = se["numerator"], se["denominator"]
            tn, neg = sp["numerator"], sp["denominator"]
            return (tp + tn) / max(pos + neg, 1)

        print()
        print(f"{'by source':<26}{a.a_name:>{w}}{a.b_name:>{w}}{'delta':>10}")
        print("-" * (26 + 2 * w + 10))
        for src in sorted({ds.records[i].source for i in idx}):
            gidx = [i for i in idx if ds.records[i].source == src]
            if len(gidx) < 20:
                continue
            gsel = torch.as_tensor(gidx, dtype=torch.long)
            ysub = scored[a.a_name]["labels"][gsel].numpy()
            ysub = ysub[ysub >= 0]
            nref = int((ysub >= REFERABLE_THRESHOLD).sum())
            if nref == 0 or nref == len(ysub):
                continue

            row = {"n": len(ysub)}
            for nm in (a.a_name, a.b_name):
                c = cfgs[nm]
                r = V.summarise(scored[nm]["logits"][gsel],
                                scored[nm]["labels"][gsel],
                                float(c["temperature"]),
                                float(c["referral_threshold"]),
                                IsotonicCalibrator.from_dict(c.get("calibrator")),
                                c.get("grade_thresholds"))
                row[nm] = {"auc": r["referable"]["auc"],
                           "sensitivity": float(_v(r["referable"]["sensitivity"])),
                           "specificity": float(_v(r["referable"]["specificity"])),
                           "exact_accuracy": float(_v(r["exact_accuracy"])),
                           "within_one": float(_v(r["adjacent_accuracy"])),
                           "referable_accuracy": _refacc(r),
                           "qwk": float(_v(r["qwk"]))}
            by_source[src] = row
            print(f"{src + f'  (n={len(ysub)})':<26}"
                  f"{'':>{w}}{'':>{w}}")
            for key, label in (("exact_accuracy", "exact accuracy"),
                               ("within_one", "within-one-grade"),
                               ("referable_accuracy", "referable accuracy"),
                               ("auc", "referable AUC"),
                               ("sensitivity", "referable sens"),
                               ("specificity", "referable spec"),
                               ("qwk", "QWK")):
                va, vb = row[a.a_name][key], row[a.b_name][key]
                print(f"{'   ' + label:<26}{va:>{w}.4f}{vb:>{w}.4f}{vb - va:>+10.4f}")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(
            {"cohort": str(a.cohort), "split": a.split, "n": int(len(y)),
             "comparison_set": blurb, "a": a.a_name, "b": a.b_name,
             "metrics": table, "per_grade": pg, "tests": sig,
             "by_source": by_source},
            indent=2, default=float), encoding="utf-8")
        print(f"\nWrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
