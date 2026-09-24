"""Export the trained PyTorch networks into MATLAB-native form, and prove the export.

This is the only Python the MATLAB codebase depends on, and it runs once: it
turns ``grader.pt`` and ``segmentation.pt`` into ``.mat`` weight files plus a
JSON architecture spec that ``+drscreen/+models`` rebuilds the networks from
in plain MATLAB matrix code (``+drscreen/+nn``; no Deep Learning Toolbox).
After this runs, nothing in convrt_matlab imports Python.

Why a weight export and not ONNX
--------------------------------
MATLAB can import ONNX, but the importer needs a separate support package,
turns GroupNorm and TF-"same" padding into auto-generated custom layers, and
hides the network behind an opaque object. Rebuilding the two networks from
their weights keeps every layer readable in MATLAB, lets Grad-CAM differentiate
exactly the tensor the Python pipeline does, and lets MC dropout run in the head
without re-running the backbone -- the backbone has no dropout, so the Python
eight-sample loop was recomputing an identical tensor seven times.

Layout conventions written here, and relied on in MATLAB
--------------------------------------------------------
* conv weights: torch ``[out, in/g, kh, kw]`` -> MATLAB
  ``[kh, kw, in/g, out/g, g]`` (read by ``drscreen.nn.conv2d``)
* transposed conv: torch ``[in, out, kh, kw]`` -> MATLAB ``[kh, kw, out, in]``
  (read by ``Segmenter``)
* BatchNorm (eval) folded to a per-channel affine ``y = x .* s + t``, shaped
  ``[1 1 C]`` so it broadcasts over ``H x W x C x B``
* Linear stays ``[out, in]``; MATLAB computes ``W * x + b`` on a column

The self-check re-runs both networks from the *exported* arrays, reading them
back in the MATLAB layout and applying the padding rule the MATLAB code uses,
and compares against the original PyTorch modules. A wrong transpose, a missed
activation or a mis-described skip connection fails here, in Python, before it
can become a silent wrong answer in MATLAB.

    python convrt_matlab/tools/export_to_matlab.py
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]          # repository root
OUT = Path(__file__).resolve().parents[1]           # convrt_matlab/
sys.path.insert(0, str(ROOT / "src"))

from drscreen.data.torch_data import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from drscreen.models.grader import DRGrader                        # noqa: E402
from drscreen.models.segmentation import build_unet                # noqa: E402

#: bundle name in MATLAB -> artefact directory in the Python repo.
BUNDLES = {
    "prepool": ROOT / "outputs" / "artifacts",       # what the Python API serves
    "pooled": ROOT / "outputs" / "artifacts_all",    # all four corpora
}


# --------------------------------------------------------------------------
# Layout conversion
# --------------------------------------------------------------------------
def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()


def conv_w(w: torch.Tensor, groups: int = 1) -> np.ndarray:
    """torch [O, I/g, kh, kw] -> MATLAB [kh, kw, I/g, O/g, g]."""
    w = _np(w)
    o, ig, kh, kw = w.shape
    w = w.reshape(groups, o // groups, ig, kh, kw)
    return np.ascontiguousarray(w.transpose(3, 4, 2, 1, 0))


def tconv_w(w: torch.Tensor) -> np.ndarray:
    """torch ConvTranspose2d [I, O, kh, kw] -> MATLAB [kh, kw, O, I]."""
    return np.ascontiguousarray(_np(w).transpose(2, 3, 1, 0))


def bn_fold(bn) -> tuple[np.ndarray, np.ndarray]:
    """Eval-mode BatchNorm as y = x*s + t, each shaped [1, 1, C]."""
    s = _np(bn.weight) / np.sqrt(_np(bn.running_var) + bn.eps)
    t = _np(bn.bias) - _np(bn.running_mean) * s
    return s.reshape(1, 1, -1).astype(np.float32), t.reshape(1, 1, -1).astype(np.float32)


def act_name(m) -> str:
    n = type(getattr(m, "act", m)).__name__
    return {"SiLU": "silu", "Identity": "identity"}.get(n, n.lower())


# --------------------------------------------------------------------------
# Grader
# --------------------------------------------------------------------------
def export_grader(ckpt: Path, out_dir: Path) -> tuple[dict, dict, DRGrader]:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    if bool(ck.get("use_clinical", True)):
        raise SystemExit(f"{ckpt}: fusion arm. Only the image-only (cnn) arm is "
                         "deployed, and only it is ported.")
    model = DRGrader(backbone=ck.get("backbone", "tf_efficientnet_b0"),
                     pretrained=False, use_clinical=False)
    model.load_state_dict(ck["model"])
    model.eval()
    bb = model.backbone

    W: dict[str, np.ndarray] = {}
    W["stem_w"] = conv_w(bb.conv_stem.weight)
    W["stem_s"], W["stem_t"] = bn_fold(bb.bn1)
    spec = {
        "arch": "tf_efficientnet_b0",
        "stem": {"kernel": bb.conv_stem.kernel_size[0],
                 "stride": bb.conv_stem.stride[0], "act": act_name(bb.bn1)},
        "blocks": [],
    }

    def se(prefix, mod):
        W[f"{prefix}_se_rw"] = conv_w(mod.conv_reduce.weight)
        W[f"{prefix}_se_rb"] = _np(mod.conv_reduce.bias)
        W[f"{prefix}_se_ew"] = conv_w(mod.conv_expand.weight)
        W[f"{prefix}_se_eb"] = _np(mod.conv_expand.bias)

    idx = 0
    for stage in bb.blocks:
        for blk in stage:
            p = f"b{idx:02d}"
            name = type(blk).__name__
            dw = blk.conv_dw
            entry = {"name": p, "kernel": dw.kernel_size[0], "stride": dw.stride[0],
                     "skip": bool(blk.has_skip)}
            if type(blk.aa).__name__ != "Identity":
                raise SystemExit(f"{p}: anti-aliasing layer is not ported")
            if name == "DepthwiseSeparableConv":
                entry["type"] = "ds"
                W[f"{p}_dw_w"] = conv_w(dw.weight, groups=dw.groups)
                W[f"{p}_bn1_s"], W[f"{p}_bn1_t"] = bn_fold(blk.bn1)
                se(p, blk.se)
                W[f"{p}_pw_w"] = conv_w(blk.conv_pw.weight)
                W[f"{p}_bn2_s"], W[f"{p}_bn2_t"] = bn_fold(blk.bn2)
                entry["acts"] = [act_name(blk.bn1), act_name(blk.bn2)]
            elif name == "InvertedResidual":
                entry["type"] = "ir"
                W[f"{p}_pw_w"] = conv_w(blk.conv_pw.weight)
                W[f"{p}_bn1_s"], W[f"{p}_bn1_t"] = bn_fold(blk.bn1)
                W[f"{p}_dw_w"] = conv_w(dw.weight, groups=dw.groups)
                W[f"{p}_bn2_s"], W[f"{p}_bn2_t"] = bn_fold(blk.bn2)
                se(p, blk.se)
                W[f"{p}_pwl_w"] = conv_w(blk.conv_pwl.weight)
                W[f"{p}_bn3_s"], W[f"{p}_bn3_t"] = bn_fold(blk.bn3)
                entry["acts"] = [act_name(blk.bn1), act_name(blk.bn2), act_name(blk.bn3)]
            else:
                raise SystemExit(f"{p}: unsupported block type {name}")
            spec["blocks"].append(entry)
            idx += 1

    W["headconv_w"] = conv_w(bb.conv_head.weight)
    W["headconv_s"], W["headconv_t"] = bn_fold(bb.bn2)
    spec["head_conv"] = {"act": act_name(bb.bn2), "channels": int(bb.num_features)}

    ln, fc1, fc2 = model.head[0], model.head[1], model.head[4]
    W["ln_g"], W["ln_b"] = _np(ln.weight), _np(ln.bias)
    W["fc1_w"], W["fc1_b"] = _np(fc1.weight), _np(fc1.bias)
    W["fc2_w"], W["fc2_b"] = _np(fc2.weight), _np(fc2.bias)
    spec["mlp"] = {
        "layernorm_eps": ln.eps,
        "feature_dropout": model.dropout.p,       # on the pooled features
        "hidden_dropout": model.head[3].p,        # between fc1 and fc2
        "hidden": fc1.out_features,
        "outputs": fc2.out_features,
    }
    spec["num_classes"] = int(model.num_classes)
    spec["normalization"] = {"mean": IMAGENET_MEAN.tolist(), "std": IMAGENET_STD.tolist()}
    spec["checkpoint"] = {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                          for k, v in ck.items() if k not in ("model", "optimizer", "config")}

    out_dir.mkdir(parents=True, exist_ok=True)
    sio.savemat(out_dir / "grader.mat", {k: v.astype(np.float32) for k, v in W.items()},
                do_compression=True)
    (out_dir / "grader_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return W, spec, model


# --------------------------------------------------------------------------
# Segmentation U-Net
# --------------------------------------------------------------------------
def export_segmenter(ckpt: Path, out_dir: Path) -> tuple[dict, dict, torch.nn.Module]:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    width = int(ck.get("width", 24))
    unet = build_unet("lesion", width=width)
    unet.load_state_dict(ck["model"])
    unet.eval()

    W: dict[str, np.ndarray] = {}
    widths = [int(e.block[0].out_channels) for e in unet.encoders]
    groups = []

    def block(prefix, blk):
        seq = blk.block
        W[f"{prefix}_c1"] = conv_w(seq[0].weight)
        W[f"{prefix}_g1g"], W[f"{prefix}_g1b"] = _np(seq[1].weight), _np(seq[1].bias)
        W[f"{prefix}_c2"] = conv_w(seq[3].weight)
        W[f"{prefix}_g2g"], W[f"{prefix}_g2b"] = _np(seq[4].weight), _np(seq[4].bias)
        return int(seq[1].num_groups), float(seq[1].eps)

    for i, enc in enumerate(unet.encoders):
        g, eps = block(f"e{i}", enc)
        groups.append(g)

    dec_groups = []
    for i, (up, gate, dec, se) in enumerate(zip(unet.ups, unet.gates, unet.decoders, unet.se)):
        W[f"u{i}_w"], W[f"u{i}_b"] = tconv_w(up.weight), _np(up.bias)
        W[f"a{i}_gw"], W[f"a{i}_gb"] = conv_w(gate.wg.weight), _np(gate.wg.bias)
        W[f"a{i}_xw"], W[f"a{i}_xb"] = conv_w(gate.wx.weight), _np(gate.wx.bias)
        W[f"a{i}_pw"], W[f"a{i}_pb"] = conv_w(gate.psi[0].weight), _np(gate.psi[0].bias)
        g, _ = block(f"d{i}", dec)
        dec_groups.append(g)
        W[f"s{i}_rw"], W[f"s{i}_rb"] = conv_w(se.fc[0].weight), _np(se.fc[0].bias)
        W[f"s{i}_ew"], W[f"s{i}_eb"] = conv_w(se.fc[2].weight), _np(se.fc[2].bias)
    W["head_w"], W["head_b"] = conv_w(unet.head.weight), _np(unet.head.bias)

    spec = {
        "arch": "attention_unet",
        "widths": widths,
        "encoder_groups": groups,
        "decoder_groups": dec_groups,
        "gn_eps": eps,
        "out_channels": int(unet.out_ch),
        "size": int(ck.get("size", 1024)),
        "lesion_classes": list(ck.get("lesion_classes", [])),
        "supervised_lesion_classes": list(ck.get("supervised_lesion_classes") or []),
        "dice": float(ck.get("dice", float("nan"))),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    sio.savemat(out_dir / "segmentation.mat",
                {k: v.astype(np.float32) for k, v in W.items()}, do_compression=True)
    (out_dir / "segmentation_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return W, spec, unet


# --------------------------------------------------------------------------
# Self-check: re-run the networks from the exported arrays
# --------------------------------------------------------------------------
# Everything below reads weights back in the MATLAB layout and applies the same
# padding and activation rules +drscreen/+models does. It is the Python
# double of the MATLAB forward pass, so if it matches PyTorch the *description*
# handed to MATLAB is complete and correct.
def _back(w: np.ndarray) -> torch.Tensor:
    """MATLAB [kh, kw, I/g, O/g, g] -> torch [O, I/g, kh, kw] (+ groups)."""
    if w.ndim == 4:
        w = w[..., None]
    kh, kw, ig, og, g = w.shape
    t = torch.from_numpy(np.ascontiguousarray(w.transpose(4, 3, 2, 0, 1)))
    return t.reshape(g * og, ig, kh, kw), g


def tf_same(n: int, k: int, s: int) -> tuple[int, int]:
    """TF 'SAME' padding along one axis: (before, after). Mirrored in MATLAB."""
    out = math.ceil(n / s)
    total = max((out - 1) * s + k - n, 0)
    return total // 2, total - total // 2


def _conv(x, w, b=None, stride=1):
    wt, g = _back(w)
    kh, kw = wt.shape[-2:]
    ph = tf_same(x.shape[-2], kh, stride)
    pw = tf_same(x.shape[-1], kw, stride)
    x = F.pad(x, (pw[0], pw[1], ph[0], ph[1]))
    bt = None if b is None else torch.from_numpy(b.reshape(-1))
    return F.conv2d(x, wt, bt, stride=stride, groups=g)


def _aff(x, s, t, act):
    y = x * torch.from_numpy(s.reshape(1, -1, 1, 1)) + torch.from_numpy(t.reshape(1, -1, 1, 1))
    return F.silu(y) if act == "silu" else y


def _se(x, W, p):
    z = x.mean(dim=(2, 3), keepdim=True)
    z = F.silu(_conv(z, W[f"{p}_se_rw"], W[f"{p}_se_rb"]))
    z = torch.sigmoid(_conv(z, W[f"{p}_se_ew"], W[f"{p}_se_eb"]))
    return x * z


def ref_trunk(W, spec, x):
    x = _aff(_conv(x, W["stem_w"], stride=spec["stem"]["stride"]),
             W["stem_s"], W["stem_t"], spec["stem"]["act"])
    for b in spec["blocks"]:
        p, a = b["name"], b["acts"]
        inp = x
        if b["type"] == "ds":
            x = _aff(_conv(x, W[f"{p}_dw_w"], stride=b["stride"]), W[f"{p}_bn1_s"], W[f"{p}_bn1_t"], a[0])
            x = _se(x, W, p)
            x = _aff(_conv(x, W[f"{p}_pw_w"]), W[f"{p}_bn2_s"], W[f"{p}_bn2_t"], a[1])
        else:
            x = _aff(_conv(x, W[f"{p}_pw_w"]), W[f"{p}_bn1_s"], W[f"{p}_bn1_t"], a[0])
            x = _aff(_conv(x, W[f"{p}_dw_w"], stride=b["stride"]), W[f"{p}_bn2_s"], W[f"{p}_bn2_t"], a[1])
            x = _se(x, W, p)
            x = _aff(_conv(x, W[f"{p}_pwl_w"]), W[f"{p}_bn3_s"], W[f"{p}_bn3_t"], a[2])
        if b["skip"]:
            x = x + inp
    return _conv(x, W["headconv_w"])          # conv_head output = the CAM layer


def ref_head(W, spec, A):
    x = _aff(A, W["headconv_s"], W["headconv_t"], spec["head_conv"]["act"])
    f = x.mean(dim=(2, 3))                                    # [B, 1280]
    eps = spec["mlp"]["layernorm_eps"]
    f = F.layer_norm(f, f.shape[-1:], torch.from_numpy(W["ln_g"].reshape(-1)),
                     torch.from_numpy(W["ln_b"].reshape(-1)), eps)
    h = F.silu(f @ torch.from_numpy(W["fc1_w"]).T + torch.from_numpy(W["fc1_b"].reshape(-1)))
    return h @ torch.from_numpy(W["fc2_w"]).T + torch.from_numpy(W["fc2_b"].reshape(-1))


def _gn(x, g, b, groups, eps):
    return F.group_norm(x, groups, torch.from_numpy(g.reshape(-1)),
                        torch.from_numpy(b.reshape(-1)), eps)


def _cblock(x, W, p, groups, eps):
    x = F.silu(_gn(_conv(x, W[f"{p}_c1"]), W[f"{p}_g1g"], W[f"{p}_g1b"], groups, eps))
    return F.silu(_gn(_conv(x, W[f"{p}_c2"]), W[f"{p}_g2g"], W[f"{p}_g2b"], groups, eps))


def ref_unet(W, spec, x):
    eps = spec["gn_eps"]
    skips = []
    n = len(spec["widths"])
    for i in range(n):
        x = _cblock(x, W, f"e{i}", spec["encoder_groups"][i], eps)
        if i < n - 1:
            skips.append(x)
            x = F.max_pool2d(x, 2)
    for i in range(n - 1):
        w = W[f"u{i}_w"]                                      # [kh, kw, O, I]
        wt = torch.from_numpy(np.ascontiguousarray(w.transpose(3, 2, 0, 1)))
        x = F.conv_transpose2d(x, wt, torch.from_numpy(W[f"u{i}_b"].reshape(-1)), stride=2)
        skip = skips[-(i + 1)]
        g = _conv(x, W[f"a{i}_gw"], W[f"a{i}_gb"])
        s = _conv(skip, W[f"a{i}_xw"], W[f"a{i}_xb"])
        psi = torch.sigmoid(_conv(F.silu(g + s), W[f"a{i}_pw"], W[f"a{i}_pb"]))
        skip = skip * psi
        x = _cblock(torch.cat([x, skip], dim=1), W, f"d{i}", spec["decoder_groups"][i], eps)
        z = x.mean(dim=(2, 3), keepdim=True)
        z = torch.sigmoid(_conv(F.silu(_conv(z, W[f"s{i}_rw"], W[f"s{i}_rb"])),
                                W[f"s{i}_ew"], W[f"s{i}_eb"]))
        x = x * z
    return _conv(x, W["head_w"], W["head_b"])


def _check(name: str, a: torch.Tensor, b: torch.Tensor, tol: float) -> float:
    err = float((a - b).abs().max())
    scale = float(b.abs().max()) or 1.0
    status = "ok " if err <= tol * max(1.0, scale) else "FAIL"
    print(f"  [{status}] {name:<34} max|diff| {err:.2e}  (|ref| max {scale:.2e})")
    if status == "FAIL":
        raise SystemExit(f"export self-check failed: {name}")
    return err


def main() -> int:
    torch.manual_seed(0)
    models_dir = OUT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # One segmentation network serves both bundles (the files are identical).
    seg_src = BUNDLES["prepool"] / "segmentation.pt"
    print(f"segmentation  <- {seg_src.relative_to(ROOT)}")
    Ws, sspec, unet = export_segmenter(seg_src, models_dir)
    x = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        _check("U-Net logits (128x128 probe)", ref_unet(Ws, sspec, x), unet(x), 1e-4)

    for name, src in BUNDLES.items():
        if not (src / "grader.pt").exists():
            print(f"{name}: {src} missing, skipped")
            continue
        print(f"grader[{name}] <- {src.relative_to(ROOT)}")
        W, spec, model = export_grader(src / "grader.pt", models_dir / name)
        cfg = json.loads((src / "pipeline.json").read_text(encoding="utf-8"))
        cfg["bundle"] = name
        cfg["source_artifacts"] = str(src.relative_to(ROOT)).replace("\\", "/")
        cfg["supervised_lesion_classes"] = sspec["supervised_lesion_classes"]
        (models_dir / name / "pipeline.json").write_text(json.dumps(cfg, indent=2),
                                                        encoding="utf-8")
        x = torch.rand(1, 3, 512, 512) * 4 - 2
        with torch.no_grad():
            bb = model.backbone
            a_ref = bb.conv_head(bb.blocks(bb.bn1(bb.conv_stem(x))))
            a = ref_trunk(W, spec, x)
            _check("conv_head activation (CAM layer)", a, a_ref, 1e-4)
            _check("logits, trunk + MATLAB-layout head", ref_head(W, spec, a), model(x), 1e-4)

    (models_dir / "README.md").write_text(
        "# Exported weights\n\n"
        "Generated by `tools/export_to_matlab.py` from the Python checkpoints and\n"
        "self-checked against PyTorch at export time. Do not edit by hand.\n\n"
        "| file | contents |\n|---|---|\n"
        "| `segmentation.mat` / `segmentation_spec.json` | Attention U-Net, shared by both bundles |\n"
        "| `prepool/` | grader trained on APTOS-2019 + IDRiD; Messidor-2 held out. What the Python API serves. |\n"
        "| `pooled/` | grader trained on all four corpora pooled |\n",
        encoding="utf-8")
    print("\nexport complete; every exported network reproduced PyTorch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
