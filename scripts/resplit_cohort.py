"""Re-split an existing materialised cohort, optionally pooling the held-out set.

This exists to answer one specific question: what happens if Messidor-2 stops
being a blind external set and joins the training pool, with train/val/test cut
from the shuffled union of all four graded corpora?

It is a **separate script on purpose**. ``build_cohort.py`` enforces the
opposite policy -- Messidor-2 always lands in ``external`` and
``assert_no_leakage`` refuses to let it near ``train`` -- and that guard stays
exactly as it is. Pooling the external set is a deliberate, named choice with a
real cost (see below), not something that should be reachable by fat-fingering a
flag on the normal build.

**What you give up.** After this runs there is no zero-shot cohort left. Every
number the validation script reports becomes in-distribution: the model has seen
the camera estate, the grading protocol and the population of all four corpora
during training. The previous headline -- 97.3% sight-threatening sensitivity on
a set the model had never encountered -- is not reproducible from a pooled split,
because no such set exists any more. Keep the old validation.json.

**Why re-split rather than rebuild.** The source cohort already holds every
grade 1, 3 and 4 image in the corpus; only grades 0 and 2 were thinned by
curation, and curation caps those again anyway. So the images and the
precomputed clinical features on disk are reused as-is (hardlinked, not copied)
and nothing is re-preprocessed. The one consequence is that the pooled val/test
prevalence sits a few points above the raw corpus prevalence, which is recorded
in ``resplit.json`` beside the counts.

Grouping is by subject, and for Messidor-2 by *patient*: the ADCIS distribution
ships a ``left;right`` pairing CSV, and without it fellow eyes -- which are
strongly correlated in severity -- would straddle the train/test boundary and
flatter the test estimate.

Example
-------
    python scripts/resplit_cohort.py --src data/cohort_real_ddr \\
        --out data/cohort_all --pool-external --curate
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from drscreen.data.registry import _hash_frac, format_grade_source_matrix

#: Splits that hold graded images and are therefore re-cut here. Everything
#: else in the manifest (the segmentation and vessel splits) is copied through
#: untouched -- those train a different model on a different corpus, and this
#: script must not perturb the lesion head that produced the cached features.
GRADED_SPLITS = ("train", "val", "test", "external")

LINKED_DIRS = ("images", "fov", "masks", "features")


@dataclass
class _Shim:
    """Minimal stand-in for registry.Sample so curate_training_pool applies."""
    grade: int
    dataset: str
    subject_id: str
    rec: dict


def messidor_patient_map(data_root: Path) -> dict[str, str]:
    """stem -> patient id, from the ADCIS ``left;right`` eye-pairing CSV.

    Messidor-2 filenames carry a per-image id, not a per-patient one, so the
    subject ids already in the manifest make every image its own group. The
    pairing file is the only thing that recovers the patient, and fellow eyes
    agree on DR grade far more often than two random eyes do -- splitting them
    across train and test leaks.
    """
    out: dict[str, str] = {}
    for cand in sorted(data_root.rglob("*.csv")):
        try:
            with cand.open(encoding="utf-8-sig", newline="") as fh:
                rows = list(csv.reader(fh, delimiter=";"))
        except Exception:
            continue
        if not rows:
            continue
        header = [c.strip().lower() for c in rows[0]]
        if header[:2] != ["left", "right"]:
            continue
        for i, row in enumerate(rows[1:]):
            if len(row) < 2:
                continue
            pid = f"pair{i:05d}"
            for cell in row[:2]:
                stem = Path(cell.strip()).stem.lower()
                if stem:
                    out[stem] = pid
        break
    return out


def group_key(rec: dict, pair_map: dict[str, str]) -> str:
    """Split-grouping key: everything sharing it lands in the same split."""
    meta = rec.get("meta") or {}
    source = rec["source"]
    if source == "messidor2" and pair_map:
        stem = Path(str(meta.get("original", ""))).stem.lower()
        pid = pair_map.get(stem)
        if pid:
            return f"messidor2/{pid}"
    return f"{source}/{meta.get('subject') or rec['uid']}"


def assign(key: str, salt: str, test_frac: float, val_frac: float) -> str:
    """Deterministic three-way assignment from one hash draw.

    One draw, two cut-points -- not two independent hashes -- so the fractions
    partition the unit interval exactly and a group cannot be claimed twice.
    """
    h = _hash_frac(key, salt)
    if h < test_frac:
        return "test"
    if h < test_frac + val_frac:
        return "val"
    return "train"


def link_tree(src: Path, dst: Path) -> tuple[int, int]:
    """Hardlink every file under src/<dir> into dst/<dir>.

    Hardlinks, not a copy: the image tree is 3.8 GB and this costs no extra
    disk. Not a directory junction either -- a junction makes ``rm -rf`` on the
    new cohort capable of walking into the source data and deleting it. With
    hardlinks, removing this cohort removes only these links.
    """
    linked = copied = 0
    for sub in LINKED_DIRS:
        s, d = src / sub, dst / sub
        if not s.is_dir():
            continue
        d.mkdir(parents=True, exist_ok=True)
        for f in s.iterdir():
            if not f.is_file():
                continue
            target = d / f.name
            if target.exists():
                continue
            try:
                os.link(f, target)
                linked += 1
            except OSError:
                shutil.copy2(f, target)
                copied += 1
    return linked, copied


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True,
                    help="existing materialised cohort to re-split")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("data/raw"),
                    help="raw datasets, read only for the Messidor-2 pairing CSV")
    ap.add_argument("--pool-external", action="store_true",
                    help="fold the external split (Messidor-2) into the pool. "
                         "This destroys the zero-shot estimate -- see module docstring.")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--salt", default="drscreen-pooled",
                    help="changing this reshuffles the split")
    ap.add_argument("--curate", action="store_true",
                    help="cap over-represented grades in TRAIN only, after the "
                         "split, matching build_cohort.py's policy")
    ap.add_argument("--curate-cap", type=int, default=0)
    a = ap.parse_args(argv)

    src_manifest = a.src / "manifest.jsonl"
    if not src_manifest.exists():
        print(f"No manifest at {src_manifest}", file=sys.stderr)
        return 1

    rows = [json.loads(l) for l in src_manifest.open(encoding="utf-8") if l.strip()]
    graded = [r for r in rows if r["split"] in GRADED_SPLITS]
    passthrough = [r for r in rows if r["split"] not in GRADED_SPLITS]

    if not a.pool_external:
        held = [r for r in graded if r["split"] == "external"]
        graded = [r for r in graded if r["split"] != "external"]
        passthrough += held

    # Ungradable images (Messidor-2 ships four) carry grade -1. They cannot
    # train an ordinal head and would corrupt the class weights, so they are
    # dropped explicitly here rather than surviving into a split.
    n_ungradable = sum(1 for r in graded if r["grade"] < 0)
    graded = [r for r in graded if r["grade"] >= 0]

    pair_map = messidor_patient_map(a.data_root) if a.pool_external else {}
    sources = sorted({r["source"] for r in graded})
    print(f"Pool: {len(graded)} graded images from {', '.join(sources)}")
    if n_ungradable:
        print(f"  dropped {n_ungradable} ungradable (grade -1)")
    if a.pool_external:
        n_pairs = len({v for v in pair_map.values()})
        print(f"  Messidor-2 eye pairing: {len(pair_map)} images -> {n_pairs} patients")
        if not pair_map:
            print("  WARNING: no pairing CSV found; fellow eyes may straddle splits")

    # --- split ------------------------------------------------------------
    by_split: dict[str, list[dict]] = defaultdict(list)
    for r in graded:
        by_split[assign(group_key(r, pair_map), a.salt, a.test_frac, a.val_frac)].append(r)

    # Leakage check on the realised split, not on the intent: every group must
    # be wholly inside one split. Cheap, and it catches a bad key immediately.
    seen: dict[str, str] = {}
    for split, recs in by_split.items():
        for r in recs:
            k = group_key(r, pair_map)
            if seen.setdefault(k, split) != split:
                print(f"LEAK: group {k} spans {seen[k]} and {split}", file=sys.stderr)
                return 1

    # --- curation: TRAIN only, after the split ----------------------------
    if a.curate:
        from drscreen.data.registry import curate_training_pool
        shims = [_Shim(r["grade"], r["source"],
                       (r.get("meta") or {}).get("subject") or r["uid"], r)
                 for r in by_split["train"]]
        before = len(shims)
        kept = curate_training_pool(shims, cap_per_grade=a.curate_cap, seed=0)
        by_split["train"] = [s.rec for s in kept]
        print(f"\nTrain curated: {before} -> {len(by_split['train'])}")

    # --- write ------------------------------------------------------------
    a.out.mkdir(parents=True, exist_ok=True)
    out_rows: list[dict] = []
    for split in ("train", "val", "test"):
        for r in by_split[split]:
            r = dict(r)
            r["meta"] = dict(r.get("meta") or {})
            # The uid encodes the split it was FIRST built under and is also the
            # image/feature filename, so it must not be rewritten. Record the
            # original split instead, so a record's provenance stays legible.
            r["meta"]["origin_split"] = r["split"]
            r["split"] = split
            out_rows.append(r)
    out_rows += passthrough

    with (a.out / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for r in out_rows:
            fh.write(json.dumps(r) + "\n")

    linked, copied = link_tree(a.src, a.out)
    # link_tree already brought thresholds.json across with the rest of
    # features/; copy it only if that did not happen.
    thr = a.src / "features" / "thresholds.json"
    dst_thr = a.out / "features" / "thresholds.json"
    if thr.exists() and not dst_thr.exists():
        shutil.copy2(thr, dst_thr)

    # --- report -----------------------------------------------------------
    summary: dict = {"src": str(a.src), "salt": a.salt,
                     "pool_external": bool(a.pool_external),
                     "curated": bool(a.curate), "splits": {}}
    print()
    for split in ("train", "val", "test"):
        recs = by_split[split]
        g = Counter(r["grade"] for r in recs)
        s = Counter(r["source"] for r in recs)
        ref = sum(v for k, v in g.items() if k >= 2)
        print(f"[{split}] n={len(recs)}  grades={dict(sorted(g.items()))}")
        print(f"         sources={dict(sorted(s.items()))}  referable={ref/max(len(recs),1):.1%}")
        summary["splits"][split] = {"n": len(recs), "grades": dict(sorted(g.items())),
                                    "sources": dict(sorted(s.items())),
                                    "referable_prevalence": round(ref / max(len(recs), 1), 4)}

    print("\nTrain pool by grade and source:")
    print(format_grade_source_matrix(
        [_Shim(r["grade"], r["source"], "", r) for r in by_split["train"]]))

    print(f"\nWrote {a.out/'manifest.jsonl'} ({len(out_rows)} records; "
          f"{len(passthrough)} passed through)")
    print(f"Linked {linked} files, copied {copied}")
    (a.out / "resplit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if a.pool_external:
        print("\nNOTE: no external split remains. Every metric from this cohort "
              "is in-distribution.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
