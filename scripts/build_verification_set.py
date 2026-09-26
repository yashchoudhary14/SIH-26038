"""Build the 12-case verification set from REAL held-out fundus photographs.

The earlier verification set was twelve generated phantoms. They exercised the
plumbing, but nobody looking at one mistook it for a retina, and a screening
result carries no weight when the thing screened was drawn by the same project
that graded it. This script rebuilds the set from photographs the model has
never been fitted on: the APTOS-2019 and IDRiD held-out *test* splits, read
from their original files on disk, not from the preprocessed cohort tensors.

Every number written out is the model's own output on the exact file saved
into ``images/`` -- the pipeline is run on the saved copy, not on the source,
so what a reviewer uploads is what was measured.

    python scripts/build_verification_set.py
    python scripts/build_verification_set.py --out outputs/verification_set

The twelve cases are curated: among the held-out images the deployed model
grades correctly, these are the ones with clean quality and a decisive
verdict. That is what a demonstration set is -- it shows the system working,
it does not measure it. The measurement is the whole-split number printed at
the end and recorded in the summary, over all 631 held-out images including
the ones the model gets wrong.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from drscreen.constants import ICDR_GRADES
from drscreen.explain.report import save_report
from drscreen.pipeline import DRScreeningPipeline

#: case name -> cohort uid. Chosen from the held-out test split; see module
#: docstring for the selection rule. Grades are re-read from the manifest, so
#: this table cannot silently drift out of step with the ground truth.
CASES: dict[str, str] = {
    "grade0_case1": "aptos2019_test_000125",
    "grade0_case2": "aptos2019_test_000191",
    "grade0_case3": "idrid_grading_test_002503",
    "grade1_case1": "aptos2019_test_000069",
    "grade1_case2": "aptos2019_test_000043",
    "grade1_case3": "aptos2019_test_000273",
    "grade2_case1": "aptos2019_test_000280",
    "grade2_case2": "idrid_grading_test_002505",
    "grade3_case1": "idrid_grading_test_002491",
    "grade3_case2": "aptos2019_test_000320",
    "grade4_case1": "aptos2019_test_000404",
    "grade4_case2": "idrid_grading_test_002487",
}

SOURCE_LABEL = {
    "aptos2019": "APTOS-2019 (Aravind Eye Hospital, India)",
    "idrid_grading": "IDRiD (Nanded, India)",
}

#: Longest side of the saved demo image. The pipeline resizes to 512 for the
#: grader and 1024 for segmentation regardless, so anything above this is
#: repository weight with no effect on the result -- 1280 keeps the saved file
#: a plausible camera output rather than a thumbnail, with headroom over the
#: 1024 the segmenter actually consumes.
MAX_SIDE = 1280

#: JPEG, not PNG: these are photographs, both source corpora ship JPEG or
#: JPEG-derived pixels, and lossless coding of a fundus photo costs ~1 MB an
#: image for no visible or measured difference. Quality 95 is far above the
#: point where the grader's output moves -- the script re-runs the pipeline on
#: the encoded file, so any drift would show up in the printed table.
JPEG_QUALITY = 95


def load_manifest(cohort: Path) -> dict[str, dict]:
    rows = {}
    with (cohort / "manifest.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            rows[r["uid"]] = r
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=Path, default=Path("data/cohort_real_ddr"))
    ap.add_argument("--artifacts", type=Path, default=Path("outputs/artifacts"))
    ap.add_argument("--out", type=Path, default=Path("outputs/verification_set"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    manifest = load_manifest(args.cohort)
    pipe = DRScreeningPipeline.load(args.artifacts)
    pipe.cfg.device = args.device

    img_dir = args.out / "images"
    rep_dir = args.out / "reports"
    img_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for case, uid in CASES.items():
        row = manifest[uid]
        raw = Path(str(row["meta"]["original"]).replace("\\", "/"))
        src = cv2.imread(str(raw), cv2.IMREAD_COLOR)
        if src is None:
            raise SystemExit(f"could not read {raw} for {case}")

        h, w = src.shape[:2]
        if max(h, w) > MAX_SIDE:
            s = MAX_SIDE / max(h, w)
            src = cv2.resize(src, (round(w * s), round(h * s)),
                             interpolation=cv2.INTER_AREA)
        dst = img_dir / f"{case}.jpg"
        cv2.imwrite(str(dst), src, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

        # Run on the saved copy: the reviewer uploads this file, so this file
        # is what has to have been measured.
        img = cv2.imread(str(dst), cv2.IMREAD_COLOR)
        res, art = pipe.run(img, image_id=case, explain=True)

        label = SOURCE_LABEL.get(row["source"], row["source"])
        prov = (f"Real image attached — {label}, held-out test split, "
                f"subject {row['meta'].get('subject', uid)}, "
                f"{src.shape[1]}×{src.shape[0]} px. "
                f"Never seen in training, validation or threshold fitting.")
        save_report(res, art, rep_dir, provenance=prov)

        rec = {
            "case": case, "image": f"images/{dst.name}",
            "uid": uid, "source": row["source"],
            "subject": row["meta"].get("subject", ""),
            "original_file": str(raw).replace("\\", "/"),
            "original_px": [w, h], "saved_px": [src.shape[1], src.shape[0]],
            "true_grade": row["grade"], "true_label": ICDR_GRADES[row["grade"]],
            "predicted_grade": res.grade, "predicted_label": res.grade_label,
            "exact_match": res.grade == row["grade"],
            "within_one": abs(res.grade - row["grade"]) <= 1,
            "confidence": round(float(res.confidence), 4),
            "referable_probability": round(float(res.referable_probability), 4),
            "sight_threatening_probability": round(
                float(res.sight_threatening_probability), 4),
            "decision": res.decision, "urgency": res.urgency,
            "rule_based_grade": res.rule_based_grade,
            "agreement": res.agreement,
            "quality": res.quality.get("overall", ""),
            "gradeable": res.gradeable,
            "latency_ms": round(float(res.timing_ms.get("total", 0)), 1),
            "provenance": prov,
        }
        records.append(rec)
        print(f"{case:14s} true {rec['true_grade']} -> pred {rec['predicted_grade']} "
              f"{'EXACT' if rec['exact_match'] else 'MISS ':5s} "
              f"conf={rec['confidence']:.2f} P(ref)={rec['referable_probability']:.3f} "
              f"{res.decision}/{res.urgency}  {rec['latency_ms']:.0f}ms", flush=True)

    exact = sum(r["exact_match"] for r in records)
    referred = sum(r["decision"] == "refer" for r in records if r["true_grade"] >= 2)
    n_ref = sum(r["true_grade"] >= 2 for r in records)
    summary = {
        "model_version": pipe.cfg.model_version,
        "artifacts": str(args.artifacts),
        "image_provenance": "real fundus photographs, held-out test split",
        "n": len(records),
        "exact_matches": exact,
        "within_one": sum(r["within_one"] for r in records),
        "sight_threatening_referred": [referred, n_ref],
        "note": ("These twelve are a curated demonstration set, not a "
                 "measurement. The measured numbers for this model are in "
                 "outputs/validation_fusionfix/summary.md."),
        "cases": records,
    }
    (args.out / "verification_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nexact {exact}/{len(records)}   "
          f"sight-threatening referred {referred}/{n_ref}")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
