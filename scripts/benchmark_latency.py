"""Measure real per-stage latency, and feed it back into the simulation config.

The telemedicine model's `edge_inference_s` and `gpu_batch_latency_s` start
life as literature estimates. Estimates are fine for a first pass and
indefensible in a capacity plan, so this script measures them on the actual
models and writes a config override the simulation can consume.

It also reports the CPU-only path, which is what a PHC edge device runs, and
the batched GPU path, which is what a district server runs -- these differ by
more than an order of magnitude and the deployment choice hinges on it.

    python scripts/benchmark_latency.py --artifacts outputs/artifacts
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from drscreen.data.synthetic import generate
from drscreen.pipeline import DRScreeningPipeline, PipelineConfig


def timeit(fn, n: int = 20, warmup: int = 3) -> dict:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    xs = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        xs.append((time.perf_counter() - t) * 1000)
    return {"mean_ms": statistics.mean(xs), "median_ms": statistics.median(xs),
            "p90_ms": float(np.percentile(xs, 90)),
            "min_ms": min(xs), "max_ms": max(xs), "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=Path("outputs/artifacts"))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("outputs/benchmark.json"))
    a = ap.parse_args()

    print("=" * 74)
    print("LATENCY BENCHMARK")
    print("=" * 74)
    print(f"CPU: {platform.processor() or platform.machine()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"resolution: {a.size}x{a.size}\n")

    phantom = generate(grade=2, size=1024, seed=1, severity=0.25)
    results: dict = {"size": a.size, "device_gpu": torch.cuda.is_available()}

    # ---- CPU-only: the PHC edge device path ------------------------------
    print("[edge / CPU-only]")
    pipe_cpu = DRScreeningPipeline.load(a.artifacts, PipelineConfig(
        size=a.size, device="cpu", enable_cam=False, mc_samples=0)) \
        if a.artifacts.exists() else DRScreeningPipeline(
            None, None, PipelineConfig(size=a.size, device="cpu", enable_cam=False))

    # The quality gate is what actually runs first on the device, and it is the
    # only thing that must run before a decision to recapture.
    from drscreen.preprocess.fov import standardize
    from drscreen.preprocess.landmarks import locate
    from drscreen.preprocess.quality import assess

    img, mask, fov = standardize(phantom.image, size=a.size)
    r = timeit(lambda: standardize(phantom.image, size=a.size), a.n)
    print(f"  geometry (FOV crop/resize)   {r['median_ms']:7.1f} ms")
    results["cpu_geometry"] = r

    r = timeit(lambda: locate(img, mask), a.n)
    print(f"  landmarks (disc + fovea)     {r['median_ms']:7.1f} ms")
    results["cpu_landmarks"] = r

    lm = locate(img, mask)
    r = timeit(lambda: assess(img, mask, fov, landmarks=lm), a.n)
    print(f"  quality gate (9 criteria)    {r['median_ms']:7.1f} ms")
    results["cpu_quality"] = r

    gate_ms = (results["cpu_geometry"]["median_ms"]
               + results["cpu_landmarks"]["median_ms"]
               + results["cpu_quality"]["median_ms"])
    print(f"  -> recapture decision in     {gate_ms:7.1f} ms  "
          f"(technician-facing feedback loop)")
    results["edge_gate_ms"] = gate_ms

    r = timeit(lambda: pipe_cpu.run(phantom.image, explain=False), max(5, a.n // 3))
    print(f"  full pipeline (CPU)          {r['median_ms']:7.1f} ms")
    results["cpu_full"] = r

    # ---- GPU: the district server path -----------------------------------
    if torch.cuda.is_available() and a.artifacts.exists():
        print("\n[district server / GPU]")
        pipe_gpu = DRScreeningPipeline.load(a.artifacts, PipelineConfig(
            size=a.size, device="cuda", enable_cam=False, mc_samples=0))
        r = timeit(lambda: pipe_gpu.run(phantom.image, explain=False), a.n)
        print(f"  full pipeline (GPU, bs=1)    {r['median_ms']:7.1f} ms")
        results["gpu_full"] = r

        pipe_cam = DRScreeningPipeline.load(a.artifacts, PipelineConfig(
            size=a.size, device="cuda", enable_cam=True, mc_samples=8))
        r = timeit(lambda: pipe_cam.run(phantom.image, explain=True), max(5, a.n // 2))
        print(f"  + Grad-CAM++ & MC dropout    {r['median_ms']:7.1f} ms")
        results["gpu_full_explained"] = r

        # Batched model-only throughput: what sizes the inference server.
        if pipe_gpu.seg is not None:
            x = torch.randn(a.batch, 3, a.size, a.size, device="cuda")
            with torch.no_grad():
                r = timeit(lambda: pipe_gpu.seg(x), a.n)
            per_img = r["median_ms"] / a.batch
            print(f"  segmentation (batch {a.batch})       "
                  f"{r['median_ms']:7.1f} ms  ->  {per_img:5.1f} ms/image")
            results["gpu_seg_batched_per_image_ms"] = per_img
        if pipe_gpu.grader is not None:
            x = torch.randn(a.batch, 3, a.size, a.size, device="cuda")
            c = torch.zeros(a.batch, pipe_gpu.grader.clinical_dim, device="cuda")
            with torch.no_grad():
                r = timeit(lambda: pipe_gpu.grader(x, c), a.n)
            per_img = r["median_ms"] / a.batch
            print(f"  grader (batch {a.batch})             "
                  f"{r['median_ms']:7.1f} ms  ->  {per_img:5.1f} ms/image")
            results["gpu_grader_batched_per_image_ms"] = per_img

    # ---- feed the measurements back into the simulation --------------------
    edge_s = results.get("cpu_full", {}).get("median_ms", 3000) / 1000.0
    gpu_per_img = (results.get("gpu_seg_batched_per_image_ms", 0)
                   + results.get("gpu_grader_batched_per_image_ms", 0))
    override = {
        "edge_iqa_latency_s": round(gate_ms / 1000.0, 4),
        "edge_inference_s": round(edge_s, 3),
        "gpu_batch_latency_s": round(gpu_per_img / 1000.0, 4) if gpu_per_img else None,
        "_measured_on": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "resolution": a.size,
        },
    }
    override = {k: v for k, v in override.items() if v is not None}

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"raw": results, "sim_overrides": override},
                                indent=2, default=float))

    print("\n" + "=" * 74)
    print("Measured overrides for SimConfig (replace the literature estimates):")
    for k, v in override.items():
        if not k.startswith("_"):
            print(f"  {k:24s} = {v}")
    print(f"\nWritten to {a.out}")
    print("Apply with:  simulate(SimConfig(**overrides))")


if __name__ == "__main__":
    main()
