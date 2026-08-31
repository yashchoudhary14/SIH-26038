"""One-shot downloader for the extra DR corpora (EyePACS 2015, DDR, ...).

Run once and it pulls the public datasets into ``data/`` and, with ``--extract``,
unpacks them into ``data/raw/<name>/`` -- the layout ``build_cohort.py`` reads.

WHAT THIS SCRIPT CANNOT DO, AND WHY
-----------------------------------
Kaggle-hosted sets (EyePACS 2015, APTOS 2019) are licensed. Downloading them
requires, one time:

  1. a free Kaggle account,
  2. an API token (``kaggle.json``), and
  3. clicking "I accept" on each competition's Rules page.

Those are licence steps only a human can take, so this script does NOT try to
bypass them. It checks whether they are done, tells you exactly how to do them
if not, and then automates everything else -- download, resume, assemble the
multi-part archives, extract, and verify.

DDR is distributed from Google Drive; pass its folder link with ``--ddr-gdrive``
(the script uses ``gdown``), or download it by hand from the project page and
point ``--extract`` at it.

ONE-TIME SETUP (about three minutes)
------------------------------------
  pip install kaggle gdown
  # Kaggle -> Account -> "Create New API Token" downloads kaggle.json
  mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
  # open each page once and click Accept:
  #   https://www.kaggle.com/c/diabetic-retinopathy-detection/rules
  #   https://www.kaggle.com/c/aptos2019-blindness-detection/rules   (only if you also want APTOS)

THEN
----
  python scripts/fetch_datasets.py --datasets eyepacs ddr --extract \
      --ddr-gdrive "https://drive.google.com/drive/folders/<DDR_FOLDER_ID>"

Disk: EyePACS is ~82 GB zipped and ~88 GB extracted. Keep ~180 GB free while
both the archive and its extraction coexist; delete the archive afterwards.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


# Kaggle competition slugs. These are the licensed, human-accept-required sets.
KAGGLE_COMPETITIONS = {
    "eyepacs": "diabetic-retinopathy-detection",   # EyePACS 2015, ~88k, grades 0-4
    "aptos":   "aptos2019-blindness-detection",     # you likely have this already
}
# Rough compressed sizes, GB, for the disk-space preflight.
APPROX_GB = {"eyepacs": 82, "aptos": 9, "ddr": 8}


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------
def have_kaggle_cli() -> bool:
    return shutil.which("kaggle") is not None


def kaggle_credentials_present() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    token = Path.home() / ".kaggle" / "kaggle.json"
    return token.exists()


def kaggle_setup_help() -> str:
    return (
        "\n  Kaggle is not set up yet. One-time, ~3 minutes:\n"
        "    1) pip install kaggle\n"
        "    2) kaggle.com -> your avatar -> Settings -> 'Create New API Token'\n"
        "       (this downloads kaggle.json)\n"
        "    3) mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/\n"
        "       chmod 600 ~/.kaggle/kaggle.json\n"
        "    4) open the competition Rules page once and click Accept:\n"
        "       https://www.kaggle.com/c/diabetic-retinopathy-detection/rules\n"
        "  Then re-run this script.\n")


def check_disk(out: Path, need_gb: float) -> None:
    out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out).free
    # x2 because the archive and its extraction briefly coexist.
    want = need_gb * 2 * 1024 ** 3
    print(f"  disk at {out}: {human(free)} free; "
          f"~{human(want)} recommended (archive + extraction).")
    if free < want:
        print(f"  !! WARNING: low disk space. You may run out mid-extraction.")


# --------------------------------------------------------------------------
# Archive assembly / extraction
# --------------------------------------------------------------------------
_PART_RE = re.compile(r"^(?P<base>.+\.zip)\.(?P<idx>\d+)$")


def assemble_multipart(directory: Path) -> None:
    """Concatenate ``foo.zip.001, foo.zip.002, ...`` back into ``foo.zip``.

    EyePACS and Messidor-2 ship as split archives -- one zip cut into numbered
    byte-parts. Concatenating the parts in order reproduces the original zip.
    """
    groups: dict[str, list[Path]] = {}
    for p in directory.iterdir():
        m = _PART_RE.match(p.name)
        if m:
            groups.setdefault(m.group("base"), []).append(p)
    for base, parts in groups.items():
        parts.sort(key=lambda q: int(_PART_RE.match(q.name).group("idx")))
        target = directory / base
        if target.exists():
            continue
        print(f"    assembling {base} from {len(parts)} parts")
        with open(target, "wb") as out:
            for part in parts:
                with open(part, "rb") as fh:
                    shutil.copyfileobj(fh, out, length=16 * 1024 * 1024)
        # Reclaim the parts immediately -- for EyePACS they are ~82 GB and
        # keeping both them and the assembled zip overflows a tight disk.
        for part in parts:
            part.unlink()


def extract_all_zips(directory: Path, depth: int = 0) -> None:
    """Extract every .zip in ``directory`` in place, recursing into new ones."""
    if depth > 4:
        return
    assemble_multipart(directory)
    zips = [p for p in directory.iterdir()
            if p.suffix == ".zip" and not _PART_RE.match(p.name)]
    for z in zips:
        print(f"    unzip {z.name}")
        try:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(directory)
        except zipfile.BadZipFile:
            print(f"    !! {z.name} is not a valid zip (incomplete download?)")
            continue
        z.unlink()                       # reclaim space as we go
    # any nested zips that just appeared
    if any(p.suffix == ".zip" and not _PART_RE.match(p.name)
           for p in directory.iterdir()):
        extract_all_zips(directory, depth + 1)


# --------------------------------------------------------------------------
# Downloaders
# --------------------------------------------------------------------------
def run(cmd: list[str], dry: bool) -> int:
    print("    $ " + " ".join(cmd))
    if dry:
        return 0
    return subprocess.call(cmd)


def download_kaggle(name: str, out: Path, dry: bool, skip_existing: bool) -> bool:
    comp = KAGGLE_COMPETITIONS[name]
    dest = out / name
    dest.mkdir(parents=True, exist_ok=True)
    if skip_existing and any(dest.glob("*.zip")):
        print(f"  {name}: archive already present, skipping download.")
        return True
    print(f"  {name}: Kaggle competition '{comp}' -> {dest}")
    rc = run(["kaggle", "competitions", "download", "-c", comp, "-p", str(dest)], dry)
    if rc != 0 and not dry:
        print(f"  !! kaggle exited {rc}. The usual cause is not having clicked "
              f"'Accept' on:\n     https://www.kaggle.com/c/{comp}/rules")
        return False
    return True


def download_ddr(url: str | None, out: Path, dry: bool, skip_existing: bool) -> bool:
    dest = out / "ddr"
    dest.mkdir(parents=True, exist_ok=True)
    if skip_existing and any(dest.iterdir()):
        print("  ddr: target not empty, skipping download.")
        return True
    if not url:
        print("  ddr: no --ddr-gdrive link given. Download it by hand from the "
              "project page and unzip into data/raw/ddr/:\n"
              "     https://github.com/nkicsl/DDR-dataset")
        return False
    try:
        import gdown  # noqa: F401
    except ImportError:
        print("  ddr: needs gdown -> pip install gdown")
        return False
    print(f"  ddr: gdown folder -> {dest}")
    return run(["gdown", "--folder", url, "-O", str(dest)], dry) == 0


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/raw"),
                    help="where archives are downloaded and extracted (default data/raw)")
    ap.add_argument("--datasets", nargs="+", default=["eyepacs", "ddr"],
                    choices=["eyepacs", "ddr", "aptos"],
                    help="which corpora to fetch (default: eyepacs ddr)")
    ap.add_argument("--ddr-gdrive", default=None,
                    help="Google Drive folder URL/id for DDR (used with gdown)")
    ap.add_argument("--extract", action="store_true",
                    help="also assemble + unzip the archives after download")
    ap.add_argument("--skip-existing", action="store_true",
                    help="do not re-download a dataset whose files are already there")
    ap.add_argument("--dry-run", action="store_true",
                    help="print exactly what would run, download nothing")
    a = ap.parse_args()

    print("Fetching:", ", ".join(a.datasets))
    print("Target  :", a.out.resolve())
    if a.dry_run:
        print("(dry run -- nothing will be downloaded)\n")

    need = sum(APPROX_GB.get(d, 5) for d in a.datasets)
    check_disk(a.out, need)

    wants_kaggle = any(d in KAGGLE_COMPETITIONS for d in a.datasets)
    if wants_kaggle and not a.dry_run:
        if not have_kaggle_cli():
            print("  !! kaggle CLI not found -> pip install kaggle")
            print(kaggle_setup_help())
            return 2
        if not kaggle_credentials_present():
            print(kaggle_setup_help())
            return 2

    ok: dict[str, bool] = {}
    for d in a.datasets:
        print()
        if d in KAGGLE_COMPETITIONS:
            ok[d] = download_kaggle(d, a.out, a.dry_run, a.skip_existing)
        elif d == "ddr":
            ok[d] = download_ddr(a.ddr_gdrive, a.out, a.dry_run, a.skip_existing)

    if a.extract and not a.dry_run:
        print("\nExtracting archives...")
        for d in a.datasets:
            if ok.get(d):
                print(f"  {d}:")
                extract_all_zips(a.out / d)

    print("\nSummary:")
    for d in a.datasets:
        state = "ok" if ok.get(d) else "NOT fetched (see messages above)"
        print(f"  {d:<10} {state}")

    print(
        "\nNEXT STEP -- this only downloads. To turn it into a correct training\n"
        "cohort (all rare grades kept, common grades subsampled, sources mixed\n"
        "per grade so the model cannot shortcut on camera), the EyePACS/DDR\n"
        "loaders and the curated build must be added to registry.py and\n"
        "build_cohort.py. Ask Claude for that step; then:\n"
        "  python scripts/build_cohort.py --source real --data-root data/raw "
        "--out data/cohort_real --curate ...\n")
    return 0 if all(ok.get(d) for d in a.datasets) else 1


if __name__ == "__main__":
    raise SystemExit(main())
