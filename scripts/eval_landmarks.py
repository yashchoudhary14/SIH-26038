"""Localisation accuracy of the analytic optic-disc / fovea detector, on real data.

Ground truth is IDRiD's ``C. Localization`` subset: 413 training + 103 testing
fundus photographs with hand-marked optic-disc and fovea centres.

This used to run on generated phantoms, which made it worthless as a measurement
and actively harmful as a fit. Phantoms draw vessels converging cleanly on the
disc and draw exudates sparse, so the one failure mode the disc detector's
vessel-convergence term exists to defeat -- a confluent bright plaque with no
vessels running into it -- barely appeared. The weight fitted on that set was
fitted against a problem real retinas do not pose.

Units
-----
The clinical criteria are expressed in disc diameters (DD), but the localization
CSVs give centres only, not a disc radius. DD is therefore derived per image from
the ground truth itself: the disc-to-fovea distance is anatomically ~2.5 DD, so
``DD = dist(disc, fovea) / 2.5``. That constant is the only assumption here and
it is applied identically to every method being compared, so a sweep over
``--vessel-weight`` is unaffected by it.

Literature reference points on real data: optic-disc detection is typically
reported at 95-99% within 1 DD, fovea at 90-96% within 1 DD.

    python scripts/eval_landmarks.py                     # score the shipped constant
    python scripts/eval_landmarks.py --sweep             # re-fit it on real data
    python scripts/eval_landmarks.py --split train --sweep
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from drscreen.preprocess.fov import standardize
from drscreen.preprocess.landmarks import DISC_VESSEL_WEIGHT, locate

#: Disc-to-fovea separation in disc diameters. Standard fundus anatomy; used
#: only to express errors in DD, never to place either landmark.
DISC_FOVEA_DD = 2.5


def _read_markup(path: Path) -> dict[str, tuple[float, float]]:
    """stem -> (x, y) from an IDRiD markup CSV.

    The files are padded with dozens of empty trailing columns and the headers
    carry stray spaces (``X- Coordinate``, ``Y - Coordinate``), so columns are
    taken positionally after checking the first three.
    """
    out: dict[str, tuple[float, float]] = {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for i, row in enumerate(csv.reader(fh)):
            if i == 0 or len(row) < 3:
                continue
            name, xs, ys = row[0].strip(), row[1].strip(), row[2].strip()
            if not name or not xs or not ys:
                continue
            try:
                out[name.lower()] = (float(xs), float(ys))
            except ValueError:
                continue
    return out


def load_localization(root: Path, split: str = "test") -> list[dict]:
    """IDRiD localization cases: image path plus disc and fovea centres."""
    loc = None
    for p in root.rglob("*"):
        if p.is_dir() and "localization" in p.name.lower():
            loc = p
            break
    if loc is None:
        return []

    want = "b." if split == "test" else "a."
    img_dir = None
    for p in (loc / "1. Original Images").rglob("*"):
        if p.is_dir() and p.name.lower().startswith(want):
            img_dir = p
            break
    if img_dir is None:
        return []

    disc_csv = fovea_csv = None
    for p in (loc / "2. Groundtruths").rglob("*.csv"):
        parent = p.parent.name.lower()
        stem = p.name.lower()
        if split == "test" and "test" not in stem:
            continue
        if split == "train" and "train" not in stem:
            continue
        if "optic" in parent or "od" in stem:
            disc_csv = p
        elif "fovea" in parent or "fovea" in stem:
            fovea_csv = p
    if disc_csv is None or fovea_csv is None:
        return []

    discs, foveas = _read_markup(disc_csv), _read_markup(fovea_csv)
    cases = []
    for img in sorted(img_dir.rglob("*")):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            continue
        key = img.stem.lower()
        if key in discs and key in foveas:
            cases.append({"path": img, "disc": discs[key], "fovea": foveas[key]})
    return cases


def project(pt, fov, raw_shape, size):
    """Map a raw-frame point through crop -> square-pad -> resize."""
    h, w = raw_shape[:2]
    x0, y0, x1, y1 = fov.bbox
    pad_x, pad_y = int(0.02 * (x1 - x0)), int(0.02 * (y1 - y0))
    X0, Y0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
    X1, Y1 = min(w, x1 + pad_x), min(h, y1 + pad_y)
    ch, cw = Y1 - Y0, X1 - X0
    side = max(ch, cw)
    top, left = (side - ch) // 2, (side - cw) // 2
    sc = size / side
    return ((pt[0] - X0 + left) * sc, (pt[1] - Y0 + top) * sc), sc


def prepare(cases: list[dict], size: int, limit: int | None = None) -> list[tuple]:
    """Read, standardise and project once, so a sweep does not redo it per point.

    IDRiD originals are 4288x2848; decoding one costs far more than running the
    detector on it, so an 11-point sweep that re-read every file would spend
    almost all its time in imread.
    """
    out = []
    for c in cases[:limit]:
        raw = cv2.imread(str(c["path"]), cv2.IMREAD_COLOR)
        if raw is None:
            continue
        img, msk, fov = standardize(raw, size=size)
        gd, _ = project(c["disc"], fov, raw.shape, size)
        gf, _ = project(c["fovea"], fov, raw.shape, size)
        # DD from the ground truth's own geometry, in the standardised frame.
        dd = math.dist(gd, gf) / DISC_FOVEA_DD
        if dd < 1.0:                      # degenerate markup; cannot normalise
            continue
        out.append((img, msk, gd, gf, dd))
    return out


def evaluate(prepared: list[tuple], vessel_weight: float | None) -> dict:
    d_err, f_err = [], []
    for img, msk, gd, gf, dd in prepared:
        lm = locate(img, msk, vessel_weight=vessel_weight)
        d_err.append(math.dist(lm.disc_xy, gd) / dd)
        f_err.append(math.dist(lm.fovea_xy, gf) / dd)

    d, f = np.asarray(d_err), np.asarray(f_err)
    if not d.size:
        return {"n": 0}
    skipped = 0
    return {
        "n": int(d.size), "skipped": skipped,
        "disc": {"median": float(np.median(d)),
                 "within_0_5": float((d <= 0.5).mean()),
                 "within_1": float((d <= 1.0).mean()),
                 "within_2": float((d <= 2.0).mean())},
        "fovea": {"median": float(np.median(f)),
                  "within_0_5": float((f <= 0.5).mean()),
                  "within_1": float((f <= 1.0).mean()),
                  "within_2": float((f <= 2.0).mean())},
    }


def _report(tag: str, r: dict) -> None:
    print(f"{tag}  (n = {r['n']} real photographs)")
    for name in ("disc", "fovea"):
        s = r[name]
        print(f"  {name:6s}  median {s['median']:.3f} DD | "
              f"<=0.5 DD {100*s['within_0_5']:5.1f}% | "
              f"<=1 DD {100*s['within_1']:5.1f}% | "
              f"<=2 DD {100*s['within_2']:5.1f}%")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("data/raw"))
    ap.add_argument("--split", choices=["train", "test"], default="train",
                    help="IDRiD localization split. Fit on train, confirm on test.")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("-n", "--limit", type=int, default=None)
    ap.add_argument("--vessel-weight", type=float, default=None,
                    help="override the shipped DISC_VESSEL_WEIGHT for one run")
    ap.add_argument("--sweep", action="store_true",
                    help="sweep the vessel weight and report the real-data optimum")
    ap.add_argument("--sweep-values", type=float, nargs="*", default=None)
    a = ap.parse_args(argv)

    idrid = a.data_root / "idrid"
    cases = load_localization(idrid if idrid.is_dir() else a.data_root, a.split)
    if not cases:
        print(f"No IDRiD localization ground truth under {a.data_root}.\n"
              "Expected 'C. Localization' with optic-disc and fovea markup CSVs; "
              "see docs/DATASETS.md.", file=sys.stderr)
        return 1

    print(f"Preprocessing {min(len(cases), a.limit or len(cases))} images...",
          flush=True)
    prepared = prepare(cases, a.size, a.limit)
    if not prepared:
        print("No usable cases after preprocessing.", file=sys.stderr)
        return 1

    if not a.sweep:
        w = DISC_VESSEL_WEIGHT if a.vessel_weight is None else a.vessel_weight
        r = evaluate(prepared, a.vessel_weight)
        _report(f"IDRiD localization / {a.split}  vessel_weight = {w:.2f}", r)
        return 0

    values = a.sweep_values or [round(x, 2) for x in np.linspace(0.0, 1.0, 11)]
    print(f"Sweeping vessel weight on IDRiD localization / {a.split} "
          f"({min(len(cases), a.limit or len(cases))} images per point)\n")
    head = (f"{'weight':>8}{'disc <=1DD':>12}{'disc med':>10}"
            f"{'fovea <=1DD':>13}{'fovea med':>11}")
    print(head)
    print("-" * len(head))
    rows = []
    for w in values:
        r = evaluate(prepared, float(w))
        if not r["n"]:
            continue
        rows.append((float(w), r))
        mark = "  <- shipped" if abs(w - DISC_VESSEL_WEIGHT) < 1e-9 else ""
        print(f"{w:>8.2f}{100*r['disc']['within_1']:>11.1f}%"
              f"{r['disc']['median']:>10.3f}"
              f"{100*r['fovea']['within_1']:>12.1f}%"
              f"{r['fovea']['median']:>11.3f}{mark}", flush=True)

    if rows:
        # Rank on disc <=1 DD -- the quantity this term controls -- and break
        # ties on median error, which keeps improving after the hit rate has
        # saturated. The fovea search starts from the disc, so a disc gain
        # carries into it; scoring on a blend would hide which one moved.
        best = max(rows, key=lambda kv: (kv[1]["disc"]["within_1"],
                                         -kv[1]["disc"]["median"]))
        cur = next((r for w, r in rows if abs(w - DISC_VESSEL_WEIGHT) < 1e-9), None)
        print(f"\nBest on real data: weight {best[0]:.2f} -> "
              f"disc <=1 DD {100*best[1]['disc']['within_1']:.1f}%, "
              f"median {best[1]['disc']['median']:.3f} DD")
        if cur is not None:
            print(f"Shipped ({DISC_VESSEL_WEIGHT:.2f}):    "
                  f"disc <=1 DD {100*cur['disc']['within_1']:.1f}%, "
                  f"median {cur['disc']['median']:.3f} DD")
            delta = best[1]["disc"]["within_1"] - cur["disc"]["within_1"]
            print(f"Difference: {100*delta:+.1f} percentage points within 1 DD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
