"""End-to-end demonstration: image in, clinical report out.

    python scripts/run_demo.py --demo                    # one phantom per grade
    python scripts/run_demo.py --image path/to/fundus.jpg
    python scripts/run_demo.py --dir path/to/folder --out outputs/reports
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from drscreen.constants import ICDR_GRADES
from drscreen.explain.report import save_report
from drscreen.pipeline import DRScreeningPipeline, PipelineConfig

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def load_pipeline(artifacts: Path, size: int, device: str) -> DRScreeningPipeline:
    if artifacts.exists() and any(artifacts.iterdir()):
        pipe = DRScreeningPipeline.load(artifacts)
        pipe.cfg.device = device
        print(f"Loaded artefacts from {artifacts}")
        print(f"  segmentation: {'yes' if pipe.seg else 'no'}   "
              f"grader: {'yes' if pipe.grader else 'no'}   "
              f"threshold: {pipe.cfg.referral_threshold:.4f}   "
              f"T: {pipe.cfg.temperature:.3f}")
    else:
        print(f"No artefacts at {artifacts}; running with the rule-based grader only.")
        pipe = DRScreeningPipeline(None, None, PipelineConfig(size=size, device=device))
    return pipe


def show(res, prefix: str = ""):
    tag = f"{prefix}{res.image_id}"
    if not res.gradeable:
        print(f"{tag:28s} UNGRADEABLE  ({res.quality['overall']})")
        for a in res.recapture_advice:
            print(f"{'':28s}   -> {a}")
        return
    print(f"{tag:28s} grade {res.grade} ({ICDR_GRADES[res.grade]:16s}) "
          f"P(ref)={res.referable_probability:.3f} conf={res.confidence:.2f} "
          f"{res.decision:15s} {res.urgency:7s} rule={res.rule_based_grade} "
          f"{res.timing_ms['total']:6.0f}ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path)
    ap.add_argument("--dir", type=Path)
    ap.add_argument("--demo", action="store_true",
                    help="generate one phantom per ICDR grade")
    ap.add_argument("--severity", type=float, default=0.3)
    ap.add_argument("--artifacts", type=Path, default=Path("outputs/artifacts"))
    ap.add_argument("--out", type=Path, default=Path("outputs/reports"))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-report", action="store_true")
    a = ap.parse_args()

    pipe = load_pipeline(a.artifacts, a.size, a.device)
    a.out.mkdir(parents=True, exist_ok=True)
    print()

    jobs: list[tuple[str, np.ndarray, int | None]] = []
    if a.demo:
        from drscreen.data.synthetic import generate
        for g in range(5):
            ph = generate(grade=g, size=768, seed=2024 + g * 31, severity=a.severity)
            jobs.append((f"phantom_grade{g}", ph.image, ph.grade))
        # One deliberately unusable capture, to show the gate firing.
        ph = generate(grade=2, size=768, seed=99, severity=0.97)
        jobs.append(("phantom_ungradeable", ph.image, ph.grade))
    if a.image:
        img = cv2.imread(str(a.image), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"Cannot read {a.image}")
        jobs.append((a.image.stem, img, None))
    if a.dir:
        for p in sorted(a.dir.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is not None:
                    jobs.append((p.stem, img, None))
    if not jobs:
        ap.print_help()
        raise SystemExit(1)

    results, correct, gradeable_n = [], 0, 0
    t0 = time.time()
    for name, img, truth in jobs:
        res, art = pipe.run(img, image_id=name)
        show(res, prefix="")
        if truth is not None and res.gradeable:
            gradeable_n += 1
            correct += int(res.grade == truth)
            if res.grade != truth:
                print(f"{'':28s}   (ground truth {truth})")
        if not a.no_report:
            paths = save_report(res, art, a.out)
            print(f"{'':28s}   report: {paths['html']}")
        results.append(res.to_dict())

    dt = time.time() - t0
    print(f"\n{len(jobs)} images in {dt:.1f}s ({dt/len(jobs)*1000:.0f} ms/image)")
    if gradeable_n:
        print(f"exact grade match on phantoms: {correct}/{gradeable_n}")
    (a.out / "batch_results.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"Results: {a.out/'batch_results.json'}")


if __name__ == "__main__":
    main()
