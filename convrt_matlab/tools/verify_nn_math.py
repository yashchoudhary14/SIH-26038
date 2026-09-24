"""Check the +drscreen/+nn and +drscreen/+models math against PyTorch, without MATLAB.

A numpy mirror of the MATLAB functions, written with Fortran-order (column-major)
reshapes so every reshape behaves exactly as the MATLAB one does -- that is where
a port of convolution code usually goes wrong. It loads the exported .mat weights
through scipy (which yields MATLAB's array shapes) and is compared with PyTorch on:

* the U-Net logits on the 128x128 probe,
* the EfficientNet CAM-layer activation and the CORN logits at 128x128 and 512x512,
* the analytic Grad-CAM gradient of Grader.gradCamPP against torch autograd, and
  the resulting Grad-CAM++ map against the Python explain.cam module.

    python convrt_matlab/tools/verify_nn_math.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parent
sys.path.insert(0, str(ROOT / "src"))

R = lambda a, *shape: np.reshape(a, shape, order="F")      # MATLAB reshape


def tf_same(n, k, s):
    out = math.ceil(n / s)
    total = max((out - 1) * s + k - n, 0)
    return total // 2, total - total // 2


def conv2d(X, W, b=None, stride=1):                        # mirrors +nn/conv2d.m
    H, Wd, C = X.shape
    kh, kw = W.shape[0], W.shape[1]
    pt, pb = tf_same(H, kh, stride)
    pl, pr = tf_same(Wd, kw, stride)
    Ho, Wo = math.ceil(H / stride), math.ceil(Wd / stride)
    Xp = np.zeros((H + pt + pb, Wd + pl + pr, C), X.dtype)
    Xp[pt:pt + H, pl:pl + Wd, :] = X
    rs = lambda u: slice(u, u + stride * (Ho - 1) + 1, stride)
    cs = lambda v: slice(v, v + stride * (Wo - 1) + 1, stride)
    depthwise = W.ndim == 5 and W.shape[4] > 1
    if depthwise:
        Y = np.zeros((Ho, Wo, C), X.dtype)
        for u in range(kh):
            for v in range(kw):
                Y += Xp[rs(u), cs(v), :] * R(W[u, v, 0, 0, :], 1, 1, C)
    else:
        O = W.shape[3]
        Y = np.zeros((Ho * Wo, O), X.dtype)
        for u in range(kh):
            for v in range(kw):
                Y += R(Xp[rs(u), cs(v), :], Ho * Wo, C) @ R(W[u, v, ...], C, O)
        Y = R(Y, Ho, Wo, O)
    if b is not None:
        Y = Y + R(b, 1, 1, b.size)
    return Y


silu = lambda x: x / (1 + np.exp(-x))
sig = lambda x: 1 / (1 + np.exp(-x))


def se(X, rw, rb, ew, eb):                                  # +nn/squeezeExcite.m
    C = X.shape[2]; r = rw.shape[3]
    z = R(X.mean(axis=(0, 1)), 1, C)
    z = silu(z @ R(rw, C, r) + R(rb, 1, r))
    z = sig(z @ R(ew, r, C) + R(eb, 1, C))
    return X * R(z, 1, 1, C)


def group_norm(X, G, g, b, eps):                            # +nn/groupNorm.m
    H, W, C = X.shape
    Y = R(X, H * W * (C // G), G).astype(np.float64)
    mu = Y.mean(axis=0, keepdims=True)
    D = Y - mu
    v = (D ** 2).mean(axis=0, keepdims=True)
    Y = (D / np.sqrt(v + eps)).astype(np.float32)
    return R(Y, H, W, C) * R(g, 1, 1, C) + R(b, 1, 1, C)


def max_pool2(x):                                           # Segmenter.m maxPool2
    H, W, C = x.shape
    y = R(x, 2, H // 2, 2, W // 2, C)
    return R(y.max(axis=0, keepdims=True).max(axis=2, keepdims=True), H // 2, W // 2, C)


def trans_conv(x, Wt, b):                                   # Segmenter.m transConv2x2
    H, W, C = x.shape; O = Wt.shape[2]
    Xm = R(x, H * W, C)
    y = np.zeros((2 * H, 2 * W, O), x.dtype)
    for di in range(2):
        for dj in range(2):
            Wm = R(Wt[di, dj, :, :], O, C)
            y[di::2, dj::2, :] = R(Xm @ Wm.T, H, W, O)
    return y + R(b, 1, 1, b.size)


def load(path):
    M = sio.loadmat(path)
    return {k: v for k, v in M.items() if not k.startswith("__")}


def unet(W, sp, X):
    n = len(sp["widths"]); skips = []
    def block(x, p, g):
        x = silu(group_norm(conv2d(x, W[f"{p}_c1"]), g, W[f"{p}_g1g"], W[f"{p}_g1b"], sp["gn_eps"]))
        return silu(group_norm(conv2d(x, W[f"{p}_c2"]), g, W[f"{p}_g2g"], W[f"{p}_g2b"], sp["gn_eps"]))
    x = X
    for i in range(n):
        x = block(x, f"e{i}", sp["encoder_groups"][i])
        if i < n - 1:
            skips.append(x); x = max_pool2(x)
    for i in range(n - 1):
        x = trans_conv(x, W[f"u{i}_w"], W[f"u{i}_b"])
        skip = skips[n - 2 - i]
        g = conv2d(x, W[f"a{i}_gw"], W[f"a{i}_gb"]); s = conv2d(skip, W[f"a{i}_xw"], W[f"a{i}_xb"])
        skip = skip * sig(conv2d(silu(g + s), W[f"a{i}_pw"], W[f"a{i}_pb"]))
        x = block(np.concatenate([x, skip], axis=2), f"d{i}", sp["decoder_groups"][i])
        x = se(x, W[f"s{i}_rw"], W[f"s{i}_rb"], W[f"s{i}_ew"], W[f"s{i}_eb"])
    return conv2d(x, W["head_w"], W["head_b"])


def trunk(W, sp, X):                                        # Grader.m trunk
    act = lambda x, a: silu(x) if a == "silu" else x
    X = act(conv2d(X, W["stem_w"], None, sp["stem"]["stride"]) * W["stem_s"] + W["stem_t"], sp["stem"]["act"])
    for b in sp["blocks"]:
        p, a, inp = b["name"], b["acts"], X
        if b["type"] == "ds":
            X = act(conv2d(X, W[f"{p}_dw_w"], None, b["stride"]) * W[f"{p}_bn1_s"] + W[f"{p}_bn1_t"], a[0])
            X = se(X, W[f"{p}_se_rw"], W[f"{p}_se_rb"], W[f"{p}_se_ew"], W[f"{p}_se_eb"])
            X = act(conv2d(X, W[f"{p}_pw_w"]) * W[f"{p}_bn2_s"] + W[f"{p}_bn2_t"], a[1])
        else:
            X = act(conv2d(X, W[f"{p}_pw_w"]) * W[f"{p}_bn1_s"] + W[f"{p}_bn1_t"], a[0])
            X = act(conv2d(X, W[f"{p}_dw_w"], None, b["stride"]) * W[f"{p}_bn2_s"] + W[f"{p}_bn2_t"], a[1])
            X = se(X, W[f"{p}_se_rw"], W[f"{p}_se_rb"], W[f"{p}_se_ew"], W[f"{p}_se_eb"])
            X = act(conv2d(X, W[f"{p}_pwl_w"]) * W[f"{p}_bn3_s"] + W[f"{p}_bn3_t"], a[2])
        if b["skip"]:
            X = X + inp
    return conv2d(X, W["headconv_w"])


def head(W, sp, A):                                         # Grader.m head (S = 1)
    m = sp["mlp"]
    a1 = A * W["headconv_s"] + W["headconv_t"]
    f = R(silu(a1).mean(axis=(0, 1)), -1, 1)
    N = f.shape[0]; mu = f.sum(0) / N; D = f - mu; v = (D ** 2).sum(0) / N
    rstd = 1 / np.sqrt(v + m["layernorm_eps"]); xhat = D * rstd
    n = xhat * W["ln_g"].reshape(-1, 1) + W["ln_b"].reshape(-1, 1)
    u = W["fc1_w"] @ n + W["fc1_b"].reshape(-1, 1)
    z = W["fc2_w"] @ silu(u) + W["fc2_b"].reshape(-1, 1)
    return z, dict(a1=a1, xhat=xhat, rstd=rstd, u=u, N=N)


def cam_grad(W, A, z, c):                                   # Grader.m gradCamPP (gradient part)
    z = z.astype(np.float64).ravel()
    logp = -np.log1p(np.exp(-z[0])) - np.log1p(np.exp(-z[1]))
    p = np.exp(logp)
    dS = 1 + p / (1 - p) if p < 1 - 1e-6 else 1.0
    dz = np.zeros((4, 1)); dz[0] = dS * sig(-z[0]); dz[1] = dS * sig(-z[1])
    dh = W["fc2_w"].astype(np.float64).T @ dz
    u = c["u"].astype(np.float64); su = sig(u)
    du = dh * su * (1 + u * (1 - su))
    dn = W["fc1_w"].astype(np.float64).T @ du
    dx = dn * W["ln_g"].astype(np.float64).reshape(-1, 1)
    xh = c["xhat"].astype(np.float64); N = c["N"]
    df = float(c["rstd"]) / N * (N * dx - dx.sum() - xh * (dx * xh).sum())
    h, w, C = A.shape
    da2 = R(df, 1, 1, C) / (h * w)
    a1 = c["a1"].astype(np.float64); sa = sig(a1)
    return da2 * sa * (1 + a1 * (1 - sa)) * W["headconv_s"].astype(np.float64)


def report(name, got, ref, tol):
    got = np.asarray(got, float); ref = np.asarray(ref, float)
    err = float(np.abs(got - ref).max()); scale = float(np.abs(ref).max()) or 1.0
    ok = err <= tol * max(1.0, scale)
    print(f"  [{'ok ' if ok else 'FAIL'}] {name:<44} max|d| {err:.2e}  (|ref| max {scale:.2e})")
    return ok


def main():
    M = sio.loadmat(OUT / "tests/fixtures/models.mat")
    ok = True
    Ws = load(OUT / "models/segmentation.mat")
    ss = json.loads((OUT / "models/segmentation_spec.json").read_text())
    ok &= report("U-Net logits, 128 probe", unet(Ws, ss, M["probe128_hwc"].astype(np.float32)),
                 M["unet_logits128"], 1e-4)

    for name in ("prepool", "pooled"):
        W = load(OUT / f"models/{name}/grader.mat")
        sp = json.loads((OUT / f"models/{name}/grader_spec.json").read_text())
        for tag in ("128", "512"):
            A = trunk(W, sp, M[f"probe{tag}_hwc"].astype(np.float32))
            ok &= report(f"{name} CAM-layer activation, {tag} probe", A, M[f"{name}_act{tag}"], 1e-4)
            z, _ = head(W, sp, A)
            ok &= report(f"{name} CORN logits, {tag} probe", z.ravel(), M[f"{name}_logits{tag}"].ravel(), 1e-4)

    # Analytic Grad-CAM gradient vs torch autograd on the real case, prepool bundle.
    from drscreen.explain.cam import compute_cam, _target_score
    from drscreen.pipeline import DRScreeningPipeline, PipelineConfig
    from drscreen.data.torch_data import to_tensor
    pipe = DRScreeningPipeline.load(ROOT / "outputs/artifacts", PipelineConfig(device="cpu"))
    arr = sio.loadmat(OUT / "tests/fixtures/arrays/grade3_case1.mat")
    mi = arr["model_input"]
    x = to_tensor(mi).unsqueeze(0)
    g = pipe.grader.eval()
    acts = {}
    h1 = g.backbone.conv_head.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = g.backbone.conv_head.register_full_backward_hook(lambda m, gi, go: acts.__setitem__("g", go[0]))
    g.zero_grad(); score = _target_score(g(x), "referable", None, 1).sum(); score.backward()
    h1.remove(); h2.remove()
    ref_grad = acts["g"][0].permute(1, 2, 0).detach().numpy()

    W = load(OUT / "models/prepool/grader.mat")
    sp = json.loads((OUT / "models/prepool/grader_spec.json").read_text())
    Xn = (mi.astype(np.float32) / 255 - np.array([0.485, 0.456, 0.406], np.float32)) / \
        np.array([0.229, 0.224, 0.225], np.float32)
    A = trunk(W, sp, Xn.astype(np.float32))
    z, c = head(W, sp, A)
    dA = cam_grad(W, A, z, c)
    ok &= report("analytic dScore/dA vs torch autograd", dA, ref_grad, 1e-3)

    # Grad-CAM++ map from the analytic gradient vs explain.cam.compute_cam
    import cv2
    a = A.astype(np.float64)
    g2, g3 = dA ** 2, dA ** 3
    denom = 2 * g2 + a.sum(axis=(0, 1), keepdims=True) * g3
    denom[denom == 0] = 1
    wts = (g2 / denom * np.maximum(dA, 0)).sum(axis=(0, 1), keepdims=True)
    m = np.maximum((wts * a).sum(axis=2), 0)
    t = torch.from_numpy(m)[None, None]
    m = torch.nn.functional.interpolate(t, size=(512, 512), mode="bilinear", align_corners=False)[0, 0].numpy()
    fov = arr["fov_mask"]
    m = np.where(fov > 0, m, 0)
    m = (m - m.min()) / (m.max() - m.min())
    ref_cam = compute_cam(g, x, None, "gradcam++", "referable", referable_index=1, fov_mask=fov)
    ok &= report("Grad-CAM++ map vs explain.cam.compute_cam", m, ref_cam, 1e-3)
    print("\nall network math reproduced" if ok else "\nMISMATCH -- see above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
