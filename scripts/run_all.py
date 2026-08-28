"""One-command end-to-end build: cohort -> models -> calibration -> validation -> demo.

    python scripts/run_all.py --quick              # ~15 min on a modern GPU
    python scripts/run_all.py                      # full run
    python scripts/run_all.py --real data/raw      # use APTOS/IDRiD/DRIVE/Messidor-2

Every stage is skipped if its output already exists, so a failed run resumes
rather than restarting. Pass --force to rebuild from scratch.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], title: str) -> float:
    print("\n" + "=" * 78)
    print(f">>> {title}")
    print("=" * 78)
    print(" ".join(str(c) for c in cmd) + "\n", flush=True)
    t0 = time.time()
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT)
    dt = time.time() - t0
    if r.returncode != 0:
        raise SystemExit(f"\nFAILED: {title} (exit {r.returncode})")
    print(f"\n--- {title}: {dt/60:.1f} min")
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="small cohort and few epochs, for a fast smoke run")
    ap.add_argument("--real", type=Path, default=None,
                    help="path to data/raw with the real datasets")
    ap.add_argument("--cohort", type=Path, default=None)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gen-workers", type=int, default=16)
    ap.add_argument("--arms", action="store_true",
                    help="also train the cnn_only and clinical_only ablation arms")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    n = 1200 if a.quick else 6000
    seg_epochs = 4 if a.quick else 14
    grade_epochs = 5 if a.quick else 18
    size = 384 if a.quick else a.size
    cohort = a.cohort or (ROOT / ("data/cohort_real" if a.real else "data/cohort_synth"))

    if a.force:
        for p in (cohort, ROOT / "outputs"):
            if p.exists():
                shutil.rmtree(p)

    total = 0.0
    t_start = time.time()

    # ---- 1. cohort ------------------------------------------------------
    if not (cohort / "manifest.jsonl").exists():
        if a.real:
            total += run([PY, "scripts/build_cohort.py", "--source", "real",
                          "--data-root", a.real, "--out", cohort, "--size", size],
                         "1/6  Build cohort from real datasets")
        else:
            total += run([PY, "scripts/build_cohort.py", "--source", "synthetic",
                          "--n", n, "--out", cohort, "--size", size,
                          "--workers", a.gen_workers, "--seed", 7],
                         f"1/6  Generate {n} fundus phantoms")
    else:
        print(f"[skip] cohort exists at {cohort}")

    # ---- 2. segmentation --------------------------------------------------
    seg_ckpt = ROOT / "outputs/segmentation/best.pt"
    if not seg_ckpt.exists():
        total += run([PY, "scripts/train_seg.py", "--cohort", cohort,
                      "--epochs", seg_epochs, "--size", size,
                      "--batch-size", 8, "--workers", a.workers],
                     "2/6  Train lesion segmentation (attention U-Net)")
    else:
        print(f"[skip] segmentation checkpoint exists at {seg_ckpt}")

    # ---- 3. clinical features ---------------------------------------------
    if not (cohort / "features").exists() or not any((cohort / "features").iterdir()):
        total += run([PY, "scripts/precompute_features.py", "--cohort", cohort,
                      "--seg", seg_ckpt, "--size", size, "--also-gt"],
                     "3/6  Extract clinical features from predicted lesion masks")
    else:
        print("[skip] clinical features already cached")

    # ---- 4. grader (+ ablation arms) ---------------------------------------
    arms = ["fusion"] + (["cnn", "clinical"] if a.arms else [])
    for arm in arms:
        out = ROOT / f"outputs/grader_{arm}"
        if (out / "best.pt").exists():
            print(f"[skip] grader arm '{arm}' already trained")
            continue
        total += run([PY, "scripts/train_grader.py", "--cohort", cohort,
                      "--arm", arm, "--epochs", grade_epochs, "--size", size,
                      "--batch-size", 16, "--workers", a.workers],
                     f"4/6  Train grader (arm: {arm})")

    # ---- 5. validation -----------------------------------------------------
    extra: list[str] = []
    for arm in ("cnn", "clinical"):
        p = ROOT / f"outputs/grader_{arm}/best.pt"
        if p.exists():
            extra += [f"{arm}_only={p}"]
    cmd = [PY, "scripts/validate.py", "--cohort", cohort, "--size", size,
           "--workers", a.workers]
    if extra:
        cmd += ["--arms"] + extra
    total += run(cmd, "5/6  Calibration, operating point, validation, ablation")

    # ---- 6. demo + simulation ----------------------------------------------
    total += run([PY, "scripts/run_demo.py", "--demo", "--size", size],
                 "6/6  End-to-end demo reports")
    total += run([PY, "scripts/run_simulation.py", "--scenarios",
                  "--export-matlab", "matlab/"],
                 "6/6  Telemedicine simulation + Simulink export")

    print("\n" + "=" * 78)
    print(f"COMPLETE in {(time.time()-t_start)/60:.1f} min")
    print("=" * 78)
    print("""
Artefacts
  outputs/validation/summary.md      clinical validation report
  outputs/validation/ablation.txt    integrated vs single-technique
  outputs/reports/*.html             per-case clinical reports
  outputs/simulation/                scenarios, optimisation
  outputs/artifacts/                 deployable model bundle
  matlab/                            Simulink/SimEvents bridge

Next
  python -m uvicorn drscreen.api:app --port 8000     # review console
  python scripts/run_simulation.py --optimise        # cheapest feasible plan
""")


if __name__ == "__main__":
    main()
