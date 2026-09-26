"""Unpack the downloaded archives into the layout the loaders expect.

The four corpora ship in four different shapes: APTOS as one Kaggle
competition zip, IDRiD as three topic zips, DRIVE as train/test zips, and
Messidor-2 as a multi-part split archive. This normalises all of them into

    data/raw/{aptos2019,idrid,drive,messidor2}/

which is what ``drscreen.data.registry.discover`` searches.

Only what is actually usable is extracted. APTOS ships 1,928 competition test
images with no public labels; they are skipped, saving ~3 GB and the time to
write them.

    python scripts/extract_datasets.py --src data --out data/raw
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def extract(zip_path: Path, dest: Path, members: list[str] | None = None,
            strip: int = 0, label: str = "") -> int:
    """Extract `members` (or everything) from `zip_path` into `dest`."""
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    written = 0
    with zipfile.ZipFile(zip_path) as z:
        names = members if members is not None else z.namelist()
        total = len(names)
        for i, name in enumerate(names, 1):
            if name.endswith("/"):
                continue
            rel = Path(name)
            if strip:
                parts = rel.parts[strip:]
                if not parts:
                    continue
                rel = Path(*parts)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size > 0:
                written += 1
                continue
            with z.open(name) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            written += 1
            if i % 500 == 0 or i == total:
                el = time.time() - t0
                print(f"    {label} {i}/{total}  ({el:.0f}s)", flush=True)
    return written


def find_one(root: Path, pattern: str) -> Path | None:
    rx = re.compile(pattern, re.IGNORECASE)
    for p in sorted(root.rglob("*")):
        if p.is_file() and rx.search(p.name):
            return p
    return None


# --------------------------------------------------------------------------
def do_aptos(src: Path, out: Path) -> None:
    z = find_one(src, r"^aptos.*\.zip$")
    if z is None:
        print("  APTOS zip not found, skipping")
        return
    dest = out / "aptos2019"
    print(f"  APTOS  <- {z.name} ({_human(z.stat().st_size)})")
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
    # train_images + train.csv only. test_images have no public labels.
    keep = [n for n in names
            if n.startswith("train_images/") or n == "train.csv"]
    skipped = len(names) - len(keep)
    print(f"    keeping {len(keep)} entries, skipping {skipped} "
          f"(unlabelled competition test set)")
    extract(z, dest, keep, label="APTOS")


def do_idrid(src: Path, out: Path) -> None:
    root = next((p for p in src.iterdir()
                 if p.is_dir() and re.search(r"idrid", p.name, re.I)), None)
    if root is None:
        print("  IDRiD folder not found, skipping")
        return
    dest = out / "idrid"
    for zp in sorted(root.glob("*.zip")):
        print(f"  IDRiD  <- {zp.name} ({_human(zp.stat().st_size)})")
        extract(zp, dest, label=zp.stem[:14])


def do_drive(src: Path, out: Path) -> None:
    root = next((p for p in src.iterdir()
                 if p.is_dir() and re.search(r"drive", p.name, re.I)), None)
    if root is None:
        print("  DRIVE folder not found, skipping")
        return
    dest = out / "drive"
    for zp in sorted(root.glob("*.zip")):
        print(f"  DRIVE  <- {zp.name} ({_human(zp.stat().st_size)})")
        # Archives may or may not already contain a training/ or test/ root.
        with zipfile.ZipFile(zp) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
        roots = {n.split("/")[0] for n in names}
        if len(roots) == 1 and roots.pop().lower() in ("training", "test"):
            extract(zp, dest, label=zp.stem)          # already rooted
        else:
            extract(zp, dest / zp.stem, label=zp.stem)


def do_messidor(src: Path, out: Path) -> None:
    """Held-out external test set. Extracted, never trained on."""
    root = next((p for p in src.iterdir()
                 if p.is_dir() and re.search(r"messidor", p.name, re.I)), None)
    if root is None:
        print("  Messidor-2 folder not found, skipping")
        return
    dest = out / "messidor2"
    dest.mkdir(parents=True, exist_ok=True)

    csv = find_one(root, r"\.csv$")
    if csv:
        shutil.copy(csv, dest / csv.name)
        print(f"  Messidor-2 <- {csv.name} (adjudicated grades)")

    for zp in sorted(root.glob("*.zip*")):
        try:
            with zipfile.ZipFile(zp) as zf:
                zf.testzip()
        except Exception as e:
            print(f"  Messidor-2 <- {zp.name}: not a readable zip on its own "
                  f"({type(e).__name__}). If this is a split archive, join the "
                  f"parts first (7-Zip: open the .001 and extract).")
            continue
        print(f"  Messidor-2 <- {zp.name} ({_human(zp.stat().st_size)})")
        extract(zp, dest, label=zp.stem[:14])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--only", nargs="*",
                    choices=["aptos", "idrid", "drive", "messidor"],
                    help="extract only these")
    a = ap.parse_args()

    if not a.src.exists():
        print(f"No such directory: {a.src}", file=sys.stderr)
        return 1
    a.out.mkdir(parents=True, exist_ok=True)

    jobs = {"aptos": do_aptos, "idrid": do_idrid,
            "drive": do_drive, "messidor": do_messidor}
    selected = a.only or list(jobs)

    t0 = time.time()
    for name in selected:
        print(f"\n=== {name} ===", flush=True)
        jobs[name](a.src, a.out)

    print(f"\nDone in {(time.time()-t0)/60:.1f} min. Layout under {a.out}:")
    for d in sorted(a.out.iterdir()):
        if d.is_dir():
            n = sum(1 for _ in d.rglob("*") if _.is_file())
            print(f"  {d.name:12s} {n:6d} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
