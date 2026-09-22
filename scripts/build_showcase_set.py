"""Build the 60-image showcase set under dataset/.

What this is, stated plainly so nobody mistakes it for a measurement: a
**curated demonstration set**. Images are chosen *because* the model grades them
correctly. It shows the system working; it does not measure how often the system
works. The measurement is the whole-split number in
``outputs/validation_all/validation.json`` and RESULTS.md, computed over every
held-out image including the ones the model gets wrong.

Two constraints make the demonstration honest anyway:

1. **Nothing here was trained on.** Candidates come from the pooled model's
   held-out test split, and are preferred from the subset that is *also* blind
   to the pre-pool bundle -- so the set stays valid whichever of the two
   bundles the site ends up serving. Every image records which of those two
   tiers it came from.
2. **The saved file is what was measured.** Selection scores the cohort tensor,
   but the file written to ``dataset/`` is a re-encoded JPEG downscaled from the
   original photograph, so the pipeline is re-run on that exact file and any
   image whose verdict moves under the re-encode is dropped and replaced.

Originals, not cohort tensors: ``data/cohort_all/images/*.png`` hold the hybrid
[CLAHE-green, Ben-Graham, L*] feature planes the model consumes, which look
nothing like a retina. A showcase needs the photograph.

    python scripts/build_showcase_set.py
    python scripts/build_showcase_set.py --dry-run     # availability only
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import torch

from drscreen.constants import ICDR_GRADES
from drscreen.models.grader import (corn_class_probs, cumulative_from_class_probs,
                                    grade_from_cumulative)
from drscreen.pipeline import DRScreeningPipeline

#: grade -> how many to ship.
QUOTA = {0: 10, 1: 10, 2: 10, 3: 15, 4: 15}

#: Old-split labels never seen by the pre-pool bundle in any capacity.
BLIND_TO_BOTH = ("test", "external")

#: Matches scripts/build_verification_set.py: the pipeline resizes to 512 for
#: the grader and 1024 for segmentation regardless, so more than this is
#: repository weight with no effect on the result.
MAX_SIDE = 1280
JPEG_QUALITY = 95

SOURCE_LABEL = {
    "aptos2019": "APTOS-2019 (Aravind Eye Hospital, India)",
    "idrid_grading": "IDRiD (Nanded, India)",
    "ddr": "DDR (China)",
    "messidor2": "Messidor-2 (France)",
}


def _load_validate():
    spec = importlib.util.spec_from_file_location("_validate", "scripts/validate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def score(bundle: Path, cohort: Path, split: str, size: int, workers: int,
          device: str):
    """Per-image predicted grade and confidence under one bundle's own config."""
    V = _load_validate()
    cfg = json.loads((bundle / "pipeline.json").read_text(encoding="utf-8"))
    model, ck = V.load_grader(bundle / "grader.pt")
    logits, labels, ds = V.run_split(model, cohort, split, size,
                                     bool(ck.get("use_clinical", True)),
                                     workers, device)
    probs = corn_class_probs(logits / float(cfg["temperature"]))
    grades = grade_from_cumulative(cumulative_from_class_probs(probs),
                                   cfg.get("grade_thresholds") or 0.5)
    return (grades.numpy(), probs.max(dim=1).values.numpy(),
            labels.numpy(), ds)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", type=Path, default=Path("data/cohort_all"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--serve", type=Path, default=Path("outputs/artifacts_all"),
                    help="bundle the showcase will serve; selection targets it")
    ap.add_argument("--other", type=Path, default=Path("outputs/artifacts"),
                    help="the other bundle, used only to prefer images blind to both")
    ap.add_argument("--out", type=Path, default=Path("dataset"))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--exclude-source", nargs="*", default=["messidor2"],
                    help="source corpora to leave out. Messidor-2 is excluded by "
                         "default: ADCIS gates it behind registration, so it is "
                         "the one corpus here that should not be redistributed "
                         "in a committed demonstration set. Pass an empty list "
                         "to include everything.")
    ap.add_argument("--verify-repeats", type=int, default=5,
                    help="independent pipeline runs each candidate must "
                         "agree across. Inference uses MC dropout, so one "
                         "run can pass a marginal image.")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    excluded = set(a.exclude_source or ())

    g_serve, conf, y, ds = score(a.serve, a.cohort, a.split, a.size,
                                 a.workers, a.device)
    g_other, _, _, _ = score(a.other, a.cohort, a.split, a.size,
                             a.workers, a.device)

    cand = []
    for i, r in enumerate(ds.records):
        if int(y[i]) < 0 or r.source in excluded:
            continue
        meta = r.meta or {}
        cand.append({
            "idx": i, "uid": r.uid, "source": r.source, "grade": int(y[i]),
            "origin_split": meta.get("origin_split", r.split),
            "original": str(meta.get("original", "")),
            "serve_ok": int(g_serve[i]) == int(y[i]),
            "other_ok": int(g_other[i]) == int(y[i]),
            "confidence": float(conf[i]),
        })

    blind_both = [c for c in cand if c["origin_split"] in BLIND_TO_BOTH]

    def tally(rows):
        c = Counter(r["grade"] for r in rows)
        return "".join(f"{c.get(g, 0):>6}" for g in range(5))

    if excluded:
        print(f"excluding source corpora: {', '.join(sorted(excluded))}")
        print()
    print(f"{'pool':<44}" + "".join(f"{'g'+str(g):>6}" for g in range(5)))
    print("-" * 74)
    print(f"{'held-out from the served bundle':<44}{tally(cand)}")
    print(f"{'  + served bundle grades it correctly':<44}"
          f"{tally([c for c in cand if c['serve_ok']])}")
    print(f"{'blind to BOTH bundles':<44}{tally(blind_both)}")
    print(f"{'  + served correct':<44}"
          f"{tally([c for c in blind_both if c['serve_ok']])}")
    print(f"{'  + BOTH correct  (tier 1)':<44}"
          f"{tally([c for c in blind_both if c['serve_ok'] and c['other_ok']])}")
    print(f"{'required':<44}" + "".join(f"{QUOTA[g]:>6}" for g in range(5)))

    if a.dry_run:
        return 0

    # --- selection ------------------------------------------------------
    # Tier 1 is blind to both bundles and graded correctly by both, so the
    # demonstration survives a switch of served bundle. Tier 2 relaxes the
    # other bundle's verdict, tier 3 relaxes blindness to the other bundle --
    # never blindness to the bundle actually being served, which would make
    # the whole exercise meaningless.
    def tier(c):
        if c["origin_split"] in BLIND_TO_BOTH:
            return 1 if c["other_ok"] else 2
        return 3

    by_grade: dict[int, list] = defaultdict(list)
    for c in cand:
        if c["serve_ok"] and c["original"]:
            by_grade[c["grade"]].append(c)

    picked: list[dict] = []
    for g, want in QUOTA.items():
        pool = by_grade.get(g, [])
        # Spread across corpora: round-robin by source within each tier, so a
        # grade is not handed entirely to whichever dataset supplies the most
        # confident images. Highest confidence first inside each source.
        chosen, seen_src = [], defaultdict(list)
        for c in sorted(pool, key=lambda c: (tier(c), -c["confidence"])):
            seen_src[c["source"]].append(c)
        queues = {s: v for s, v in seen_src.items()}
        order = sorted(queues)
        pos = {s: 0 for s in order}
        while len(chosen) < want * 8 and any(pos[s] < len(queues[s]) for s in order):
            for s in order:
                if pos[s] < len(queues[s]):
                    chosen.append(queues[s][pos[s]])
                    pos[s] += 1
        picked.append({"grade": g, "want": want, "candidates": chosen})

    # --- write + verify on the saved file -------------------------------
    # Clear the contents rather than the tree. This repository lives under a
    # OneDrive-synced path, where rmdir on a directory the sync client is
    # watching raises WinError 5 even though every file in it deleted fine.
    if a.out.exists():
        for f in sorted(a.out.rglob("*"), key=lambda p: -len(p.parts)):
            if f.is_file():
                f.unlink(missing_ok=True)
    pipe = DRScreeningPipeline.load(a.serve)
    pipe.cfg.device = a.device

    def demo_rank(g: int, res) -> tuple | None:
        """Rank a verified candidate for demonstration, or reject it outright.

        The grade being right is not sufficient. Grade and referral are two
        different outputs of the same model -- the grade comes from the ordinal
        cut-points, the referral from calibrated P(y>=2) against a separately
        fitted threshold -- so an image can carry the right grade and the wrong
        call. At this operating point that happens on mild NPDR: urgency fires
        at P(y>=3) >= 0.0722, which is low enough that a correctly-graded
        grade-1 eye can come back REFER / URGENT.

        Hard reject: a non-sight-threatening eye marked urgent, and a
        sight-threatening one not marked urgent. The first misrepresents the
        system as alarmist, the second as unsafe, and neither belongs in a
        demonstration.

        Preferred, in order: the textbook call for the grade, then the milder
        disagreement, then whatever is left.
        """
        st = g >= 3
        referable = g >= 2
        if st:
            # Sight-threatening must be referred urgently. Anything else is
            # the system failing at the job it exists to do.
            return None if res.urgency != "urgent" else (0, 0)
        if referable:
            # Moderate NPDR must be referred. Urgency on it is over-escalation
            # in the safe direction and a defensible thing to show -- moderate
            # NPDR is a genuine referral -- so it is ranked below the textbook
            # call rather than rejected.
            if res.decision == "auto_report":
                return None
            return (0, 1) if res.urgency == "urgent" else (0, 0)
        # Grades 0 and 1 are not referable. Urgent here is the one outcome a
        # demonstration must never show: it reads as the system crying wolf on
        # a healthy or near-healthy eye.
        if res.urgency == "urgent":
            return None
        if res.decision == "auto_report":
            return (0, 0)
        # Defer is honest (the model is genuinely unsure near the mild/moderate
        # boundary); a plain refer on non-referable disease is over-referral.
        return (1, 0) if res.decision == "defer_to_human" else (2, 0)

    records, shortfall = [], {}
    for entry in picked:
        g, want = entry["grade"], entry["want"]
        gdir = a.out / f"grade_{g}"
        gdir.mkdir(parents=True, exist_ok=True)

        # Verify every candidate on its encoded bytes first, then keep the best
        # `want`. Verifying only the first `want` would ship whichever happened
        # to be ordered first rather than the ones that demonstrate best.
        verified = []
        tmp = gdir / "_probe.jpg"
        for c in entry["candidates"]:
            raw_path = Path(c["original"].replace("\\", "/"))
            src = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
            if src is None:
                continue
            h, w = src.shape[:2]
            if max(h, w) > MAX_SIDE:
                s = MAX_SIDE / max(h, w)
                src = cv2.resize(src, (round(w * s), round(h * s)),
                                 interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(tmp), src, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            img = cv2.imread(str(tmp), cv2.IMREAD_COLOR)

            # Inference is stochastic: PipelineConfig.mc_samples defaults to 8,
            # so dropout stays on and each call draws a fresh posterior sample.
            # That is deliberate -- the epistemic variance is what drives the
            # defer band -- but it means a single call can pass an image that
            # is only marginally correct, and the site would then show a
            # different verdict than the manifest records. Require the same
            # grade, decision and urgency on every repeat, so what ships is
            # stably correct rather than luckily correct.
            runs = [pipe.run(img, image_id="probe", explain=False)[0]
                    for _ in range(a.verify_repeats)]
            res = runs[0]
            if any(not r.gradeable or r.grade != g for r in runs):
                continue
            if len({(r.decision, r.urgency) for r in runs}) != 1:
                continue
            rank = demo_rank(g, res)
            if rank is None:
                continue
            verified.append((rank + (tier(c), -float(res.confidence)), c, src, res))
        tmp.unlink(missing_ok=True)

        verified.sort(key=lambda v: v[0])
        for n, (_, c, src, res) in enumerate(verified[:want], start=1):
            name = f"grade{g}_{n:02d}.jpg"
            dst = gdir / name
            cv2.imwrite(str(dst), src, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            records.append({
                "file": f"grade_{g}/{name}", "true_grade": g,
                "true_label": ICDR_GRADES[g],
                "predicted_grade": int(res.grade),
                "referable_probability": round(float(res.referable_probability), 4),
                "decision": res.decision, "urgency": res.urgency,
                "confidence": round(float(res.confidence), 4),
                "source": c["source"],
                "provenance": SOURCE_LABEL.get(c["source"], c["source"]),
                "cohort_uid": c["uid"], "selection_tier": tier(c),
                "blind_to_both_bundles": c["origin_split"] in BLIND_TO_BOTH,
            })
        if len(verified) < want:
            shortfall[g] = (len(verified), want)

    # Three outcomes, not two. Collapsing `defer` into "referred" scores the
    # selective-referral path as a miss, when deferring a grade-1 eye near the
    # mild/moderate boundary -- the one place the reference standards
    # themselves disagree -- is the behaviour the defer band exists to produce.
    referable = [r for r in records if r["true_grade"] >= 2]
    non_ref = [r for r in records if r["true_grade"] < 2]
    n_st = [r for r in records if r["true_grade"] >= 3]
    n_ref_ok = sum(1 for r in referable if r["decision"] == "refer")
    n_nonref_ok = sum(1 for r in non_ref if r["decision"] == "auto_report")
    n_defer = sum(1 for r in records if r["decision"] == "defer_to_human")
    # Grade 2 escalated to urgent is over-escalation in the safe direction
    # and is permitted; grade 0 or 1 escalated is not, and is rejected at
    # selection. Counted separately so neither is mistaken for the other.
    n_moderate_urgent = sum(1 for r in records
                            if r["true_grade"] == 2 and r["urgency"] == "urgent")
    n_bad_urgent = sum(1 for r in records
                       if r["true_grade"] < 2 and r["urgency"] == "urgent")
    summary = {
        "what_this_is": ("Curated demonstration set: images were selected "
                         "BECAUSE the model grades them correctly. It shows the "
                         "system working, it does not measure how often it does. "
                         "The measurement is outputs/validation_all/validation.json."),
        "served_bundle": str(a.serve),
        "excluded_sources": sorted(excluded),
        "verify_repeats": a.verify_repeats,
        "inference_note": ("Inference uses MC dropout (mc_samples=8), so a single "
                           "run is a posterior sample. Every image here returned "
                           "the same grade, decision and urgency on "
                           f"{a.verify_repeats} independent runs."),
        "cohort": str(a.cohort), "split": a.split,
        "n": len(records),
        "per_grade": {str(g): sum(1 for r in records if r["true_grade"] == g)
                      for g in range(5)},
        "exact_grade_match": f"{sum(1 for r in records if r['predicted_grade'] == r['true_grade'])}/{len(records)}",
        "referable_referred": f"{n_ref_ok}/{len(referable)}",
        "non_referable_auto_reported": f"{n_nonref_ok}/{len(non_ref)}",
        "deferred_to_human": n_defer,
        "sight_threatening_referred_urgent":
            f"{sum(1 for r in n_st if r['urgency'] == 'urgent')}/{len(n_st)}",
        "moderate_npdr_escalated_to_urgent": n_moderate_urgent,
        "non_referable_marked_urgent": n_bad_urgent,
        "blind_to_both_bundles": sum(1 for r in records if r["blind_to_both_bundles"]),
        "cases": records,
    }
    (a.out / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nWrote {len(records)} images to {a.out}")
    for g in range(5):
        n = summary["per_grade"][str(g)]
        print(f"  grade {g}: {n}/{QUOTA[g]}")
    print(f"exact grade match           {summary['exact_grade_match']}")
    print(f"referable referred          {summary['referable_referred']}")
    print(f"non-referable auto-reported {summary['non_referable_auto_reported']}")
    print(f"deferred to human           {summary['deferred_to_human']}")
    print(f"sight-threat urgent         {summary['sight_threatening_referred_urgent']}")
    print(f"grade-2 escalated to urgent {summary['moderate_npdr_escalated_to_urgent']}")
    print(f"grade 0-1 marked urgent     {summary['non_referable_marked_urgent']}")
    print(f"blind to both bundles       {summary['blind_to_both_bundles']}/{len(records)}")
    if shortfall:
        print("\nSHORTFALL -- not enough candidates:")
        for g, (got, want) in shortfall.items():
            print(f"  grade {g}: {got} of {want}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
