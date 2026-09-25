"""Record what the Python pipeline does, so the MATLAB port can be held to it.

Three layers of fixtures, from narrow to wide, so a disagreement is located the
moment it appears rather than discovered as a wrong grade at the end:

``tests/fixtures/primitives.mat``
    OpenCV's output for every image primitive the pipeline uses -- Gaussian
    blur (8-bit and float kernels differ in OpenCV), resize (area, cubic,
    nearest), CLAHE, grey and Lab conversion, Otsu, elliptical morphology,
    distance transform, numpy's percentile. ``+drscreen/+cv`` re-implements
    each one, and this is what it is graded against.

``tests/fixtures/models.mat``
    Network outputs on fixed probe tensors: the CAM-layer activation and the
    logits of both grader bundles, and the U-Net logits.

``tests/fixtures/cases.json`` and ``tests/fixtures/arrays/*.mat``
    The full Python screening result for all 72 committed real photographs
    (12 verification cases + 60 showcase), with MC dropout off so the reference
    is deterministic, and the intermediate arrays for three of them.

    python matlab/tools/make_fixtures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drscreen.data.torch_data import to_tensor                      # noqa: E402
from drscreen.pipeline import DRScreeningPipeline, PipelineConfig    # noqa: E402
from drscreen.preprocess.fov import standardize                      # noqa: E402

# A reference must be true float32. On NVIDIA GPUs PyTorch runs convolutions in
# TF32 by default (10-bit mantissa), which moves activations by ~2e-3 relative
# -- the size of error the MATLAB tests are there to catch.
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

FIX = OUT / "tests" / "fixtures"
BUNDLES = {"prepool": ROOT / "outputs" / "artifacts",
           "pooled": ROOT / "outputs" / "artifacts_all"}

#: Cases whose intermediate arrays are recorded in full: one auto-report, one
#: sight-threatening referral, one moderate referral.
ARRAY_CASES = ("grade0_case1", "grade3_case1", "grade2_case1")


def rgb(bgr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(bgr[..., ::-1])


def ellipse(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)).astype(np.uint8)


# --------------------------------------------------------------------------
def primitives() -> None:
    src = cv2.imread(str(OUT / "data/verification_set/images/grade3_case1.jpg"))
    small = cv2.resize(src, (283, 301), interpolation=cv2.INTER_AREA)   # odd, non-square
    std, mask, _ = standardize(src, size=512)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    grayf = gray.astype(np.float32)
    green512 = std[..., 1]
    greenf = green512.astype(np.float32)

    P: dict[str, np.ndarray] = {
        "small_rgb": rgb(small), "std_rgb": rgb(std), "std_mask": mask,
        "gray": gray,
        "lab": cv2.cvtColor(small, cv2.COLOR_BGR2LAB),
    }
    P["lab2rgb"] = rgb(cv2.cvtColor(P["lab"], cv2.COLOR_LAB2BGR))
    # random colours in both directions: both conversions are exact on all
    # 16.7M values, and 262k random ones catch any regression
    rng = np.random.default_rng(7)
    P["rgb_rand"] = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
    P["rgb_rand_lab"] = cv2.cvtColor(P["rgb_rand"], cv2.COLOR_RGB2LAB)
    P["lab_rand"] = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
    P["lab_rand_rgb"] = cv2.cvtColor(P["lab_rand"], cv2.COLOR_LAB2RGB)

    sig_u8 = [1.5, 4.2, 10.3, 16.9]
    sig_f = [1.0, 4.0, 25.6, 23.3]
    P["blur_u8_sigmas"] = np.array(sig_u8)
    P["blur_f_sigmas"] = np.array(sig_f)
    for i, s in enumerate(sig_u8):
        P[f"blur_u8_{i}"] = cv2.GaussianBlur(gray, (0, 0), s)
    for i, s in enumerate(sig_f):
        P[f"blur_f_{i}"] = cv2.GaussianBlur(grayf, (0, 0), s)
    P["blur_rgb"] = rgb(cv2.GaussianBlur(small, (0, 0), 0.033 * small.shape[1]))

    P["resize_area_down"] = cv2.resize(gray, (151, 137), interpolation=cv2.INTER_AREA)
    P["resize_area_half"] = cv2.resize(grayf, (142, 151), interpolation=cv2.INTER_AREA)
    P["resize_area_std"] = cv2.resize(src, (512, 512), interpolation=cv2.INTER_AREA)[..., ::-1].copy()
    P["src_rgb"] = rgb(src)
    P["resize_cubic_up"] = cv2.resize(gray, (640, 600), interpolation=cv2.INTER_CUBIC)
    P["resize_cubic_2x"] = rgb(cv2.resize(std, (1024, 1024), interpolation=cv2.INTER_CUBIC))
    P["resize_nearest"] = cv2.resize(gray, (400, 97), interpolation=cv2.INTER_NEAREST)
    P["resize_area_float2x"] = cv2.resize(
        np.random.default_rng(0).random((64, 64, 5)).astype(np.float32), (32, 32),
        interpolation=cv2.INTER_AREA)
    P["float_probe"] = np.random.default_rng(0).random((64, 64, 5)).astype(np.float32)

    P["clahe_green_3"] = cv2.createCLAHE(3.0, (8, 8)).apply(green512)
    P["clahe_odd_25"] = cv2.createCLAHE(2.5, (8, 8)).apply(gray)

    blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
    P["otsu_in"] = blur
    P["otsu_t"] = np.array(float(cv2.threshold(blur, 0, 255,
                                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]))

    ks = list(range(3, 36, 2)) + [165, 187]
    P["ellipse_ks"] = np.array(ks, dtype=np.float64)
    for k in ks:
        P[f"ellipse_{k}"] = ellipse(k)

    P["close_f13"] = cv2.morphologyEx(greenf, cv2.MORPH_CLOSE, ellipse(13))
    P["close_f15"] = cv2.morphologyEx(greenf, cv2.MORPH_CLOSE, ellipse(15))
    binimg = (gray > 90).astype(np.uint8) * 255
    P["bin"] = binimg
    P["open_bin5"] = cv2.morphologyEx(binimg, cv2.MORPH_OPEN, ellipse(5))
    P["erode_mask31"] = cv2.erode(mask, ellipse(31))

    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    P["filter_noise"] = cv2.filter2D(greenf, -1, k)

    vessel = (cv2.morphologyEx(greenf, cv2.MORPH_CLOSE, ellipse(15)) - greenf > 6).astype(np.uint8)
    P["dt_in"] = vessel
    P["dt_out"] = cv2.distanceTransform(vessel, cv2.DIST_L2, 5)

    rng = np.random.default_rng(1)
    v = rng.normal(size=1001)
    P["pct_in"] = v
    P["pct_p"] = np.array([0, 5, 10, 50, 90, 95, 97, 99, 99.5, 100])
    P["pct_out"] = np.percentile(v, P["pct_p"])

    n, labels, stats, cents = cv2.connectedComponentsWithStats(binimg, connectivity=8)
    order = np.lexsort((cents[1:, 0], cents[1:, 1]))
    P["cc_area"] = stats[1:, cv2.CC_STAT_AREA][order].astype(np.float64)
    P["cc_cent"] = cents[1:][order]

    P["benG"] = cv2.addWeighted(small, 4, cv2.GaussianBlur(small, (0, 0), 9.3), -4, 128)[..., 1]
    P["benG_sigma"] = np.array(9.3)

    FIX.mkdir(parents=True, exist_ok=True)
    sio.savemat(FIX / "primitives.mat", P, do_compression=True)
    print(f"primitives.mat: {len(P)} arrays")


# --------------------------------------------------------------------------
def models() -> None:
    torch.manual_seed(0)
    M: dict[str, np.ndarray] = {}
    img = cv2.imread(str(OUT / "data/verification_set/images/grade3_case1.jpg"))
    std, _, _ = standardize(img, size=512)
    x512 = to_tensor(std).unsqueeze(0)
    x128 = torch.nn.functional.interpolate(x512, size=(128, 128), mode="area")
    M["probe512_hwc"] = x512[0].permute(1, 2, 0).numpy()
    M["probe128_hwc"] = x128[0].permute(1, 2, 0).numpy()

    for name, d in BUNDLES.items():
        pipe = DRScreeningPipeline.load(d, PipelineConfig(device="cpu"))
        g = pipe.grader.eval()
        bb = g.backbone
        with torch.no_grad():
            for tag, x in (("512", x512), ("128", x128)):
                a = bb.conv_head(bb.blocks(bb.bn1(bb.conv_stem(x))))
                M[f"{name}_act{tag}"] = a[0].permute(1, 2, 0).numpy()
                M[f"{name}_logits{tag}"] = g(x)[0].numpy()
            stem = bb.bn1(bb.conv_stem(x128))
            M[f"{name}_stem128"] = stem[0].permute(1, 2, 0).numpy()
            h = stem
            for si, stage in enumerate(bb.blocks):
                h = stage(h)
                M[f"{name}_stage{si}_128"] = h[0].permute(1, 2, 0).numpy()

    pipe = DRScreeningPipeline.load(BUNDLES["prepool"], PipelineConfig(device="cpu"))
    with torch.no_grad():
        M["unet_logits128"] = pipe.seg(x128)[0].permute(1, 2, 0).numpy()
    sio.savemat(FIX / "models.mat", M, do_compression=True)
    print(f"models.mat: {len(M)} arrays")


# --------------------------------------------------------------------------
def _images() -> list[tuple[str, Path, int | None, str]]:
    """(name, path, true_grade, set) for the 72 committed photographs.

    Read from matlab/data -- the very files the MATLAB tests read -- not
    from the repository's outputs/ and matlab/data/showcase, so a working-tree change there
    (a sync conflict once swapped two showcase images) cannot make Python and
    MATLAB screen different pixels.
    """
    out = []
    vset = OUT / "data" / "verification_set"
    vs = json.loads((vset / "verification_summary.json").read_text("utf-8"))
    for c in vs["cases"]:
        out.append((c["case"], vset / c["image"], c["true_grade"], "verification"))
    show = OUT / "data" / "showcase"
    man = json.loads((show / "manifest.json").read_text("utf-8"))
    for c in man["cases"]:
        stem = Path(c["file"]).stem
        out.append((stem, show / c["file"], c["true_grade"], "showcase"))
    return out


def _rel(p: Path) -> str:
    """Path relative to matlab, as the MATLAB tests open it."""
    return str(p.relative_to(OUT)).replace("\\", "/")


def cases() -> None:
    arr_dir = FIX / "arrays"
    arr_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    pipes = {}
    for name, d in BUNDLES.items():
        p = DRScreeningPipeline.load(d)
        p.cfg.mc_samples = 0                       # deterministic reference
        pipes[name] = p

    for case, path, true_grade, subset in _images():
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        row = {"case": case, "set": subset, "file": _rel(path), "true_grade": true_grade}
        for name, pipe in pipes.items():
            full = name == "prepool" and case in ARRAY_CASES
            res, art = pipe.run(img, image_id=case, explain=full)
            d = res.to_dict()
            d.pop("timing_ms", None)
            if res.gradeable and pipe.grader is not None:
                x = to_tensor(art["model_input"]).unsqueeze(0).to(pipe.device)
                with torch.no_grad():
                    logits = pipe.grader(x)[0].cpu().numpy()
                d["_logits"] = [float(v) for v in logits]
            row[name] = d
            if full:
                A = {
                    "standardized": rgb(art["standardized"]),
                    "fov_mask": art["fov_mask"],
                    "enhanced": rgb(art["enhanced"]),
                    "model_input": art["model_input"],
                    "lesion_probs": art["lesion_probs"].astype(np.float32),
                    "vessel_mask": art["vessel_mask"],
                }
                if "cam" in art:
                    A["cam"] = art["cam"].astype(np.float32)
                x = to_tensor(art["model_input"]).unsqueeze(0).to(pipe.device)
                bb = pipe.grader.backbone
                with torch.no_grad():
                    a = bb.conv_head(bb.blocks(bb.bn1(bb.conv_stem(x))))
                A["activation"] = a[0].permute(1, 2, 0).cpu().numpy()
                sio.savemat(arr_dir / f"{case}.mat", A, do_compression=True)
        rows.append(row)
        pp = row["prepool"]
        print(f"  {case:<16} true {true_grade}  grade {pp['grade']}  {pp['decision']}/{pp['urgency']}"
              f"  P(ref) {pp['referable_probability']:.4f}")

    (FIX / "cases.json").write_text(json.dumps({
        "note": ("Python reference results with MC dropout off (mc_samples=0). "
                 "Produced by tools/make_fixtures.py; do not edit by hand."),
        "cases": rows}, indent=1, default=float), encoding="utf-8")
    print(f"cases.json: {len(rows)} cases, arrays for {len(ARRAY_CASES)}")


if __name__ == "__main__":
    primitives()
    models()
    cases()
