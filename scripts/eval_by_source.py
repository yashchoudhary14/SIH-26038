"""Break a validated split down by source dataset.

Pooling Messidor-2 into training removes the zero-shot cohort, and with it the
single number that made the old result worth quoting. What can still be
recovered is a *composition-matched* comparison: the Messidor-2 images that
landed in the new test split are the same kind of data the old external set was
made of, scored with the same fitted operating point.

Read it for what it is. These images were never trained on -- they are in the
held-out test split -- but Messidor-2 *as a corpus* was, so the model has seen
its cameras, its grading protocol and its population. This is an in-distribution
estimate on unseen images, which sits strictly between "internal test" and the
zero-shot number it replaces, and it is not a substitute for the latter.

    python scripts/eval_by_source.py --cohort data/cohort_all \\
        --grader outputs/grader_cnn_all/best.pt --artifacts outputs/artifacts_all
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from drscreen.models.calibration import IsotonicCalibrator


def _load_validate():
    """Import scripts/validate.py as a module (scripts/ is not a package)."""
    path = Path(__file__).resolve().parent / "validate.py"
    spec = importlib.util.spec_from_file_location("_validate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--grader", type=Path, required=True)
    ap.add_argument("--artifacts", type=Path, required=True,
                    help="bundle holding pipeline.json, for the fitted operating point")
    ap.add_argument("--split", default="test")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    V = _load_validate()
    cfg = json.loads((a.artifacts / "pipeline.json").read_text(encoding="utf-8"))
    T = float(cfg["temperature"])
    thr = float(cfg["referral_threshold"])
    grade_thr = cfg.get("grade_thresholds")
    iso = IsotonicCalibrator.from_dict(cfg.get("calibrator"))

    model, ck = V.load_grader(a.grader)
    use_clinical = bool(ck.get("use_clinical", True))
    logits, labels, ds = V.run_split(model, a.cohort, a.split, a.size,
                                     use_clinical, a.workers, a.device)

    sources = [r.source for r in ds.records]
    by_source: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(sources):
        by_source[s].append(i)

    out: dict[str, dict] = {}
    groups = [("ALL", list(range(len(sources))))]
    groups += [(s, idx) for s, idx in sorted(by_source.items())]
    # The old external set was Messidor-2 alone; its complement is the closest
    # thing left to "the cohort as it was before pooling".
    if "messidor2" in by_source:
        groups.append(("non-messidor2",
                       [i for i, s in enumerate(sources) if s != "messidor2"]))

    print(f"Split '{a.split}' of {a.cohort}  (T={T:.4f}, threshold={thr:.4f})\n")
    head = (f"{'group':<16}{'n':>6}{'ref sens':>10}{'ref spec':>10}{'AUC':>8}"
            f"{'ST sens':>9}{'QWK':>8}{'adj acc':>9}")
    print(head)
    print("-" * len(head))
    for name, idx in groups:
        if not idx:
            continue
        sel = torch.as_tensor(idx, dtype=torch.long)
        res = V.summarise(logits[sel], labels[sel], T, thr, iso, grade_thr)
        out[name] = res
        st = res["severity"]["sight_threatening"]["sensitivity"]

        def val(x):
            return x["value"] if isinstance(x, dict) and "value" in x else x

        print(f"{name:<16}{res['n']:>6}"
              f"{val(res['referable']['sensitivity']):>10.4f}"
              f"{val(res['referable']['specificity']):>10.4f}"
              f"{res['referable']['auc']:>8.4f}"
              f"{val(st):>9.4f}"
              f"{val(res['qwk']):>8.4f}"
              f"{val(res['adjacent_accuracy']):>9.4f}")

    print("\nPer-grade recall")
    grades = sorted({int(g) for g in out["ALL"]["per_grade_recall"]})
    print(f"{'group':<16}" + "".join(f"{'grade '+str(g):>10}" for g in grades))
    for name, _ in groups:
        if name not in out:
            continue
        row = out[name]["per_grade_recall"]
        cells = []
        for g in grades:
            r = row.get(str(g), row.get(g))
            v = (r["value"] if isinstance(r, dict) and "value" in r else r)
            cells.append("        --" if v is None else f"{float(v):>10.4f}")
        print(f"{name:<16}" + "".join(cells))

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(
            {"cohort": str(a.cohort), "split": a.split, "grader": str(a.grader),
             "temperature": T, "threshold": thr, "by_group": out},
            indent=2, default=float), encoding="utf-8")
        print(f"\nWrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
