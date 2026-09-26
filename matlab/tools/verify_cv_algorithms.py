"""Check the +drscreen/+cv algorithms against real OpenCV, without MATLAB.

Each function below is a line-for-line numpy transliteration of the matching
MATLAB file in +drscreen/+cv (same variable names, same order of operations,
0-based where the MATLAB adds 1). Running them against the OpenCV outputs in
tests/fixtures/primitives.mat proves the *algorithm* each MATLAB file encodes
reproduces OpenCV. What this cannot prove is the MATLAB syntax and indexing;
tests/TestCv.m checks that once MATLAB is available, against the same fixtures.

    python matlab/tools/verify_cv_algorithms.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.signal import convolve2d

FIX = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "primitives.mat"


def cv_round(x):
    x = np.asarray(x, float)
    r = np.round(x)                # numpy rounds half to even already
    return r


def sat_u8(x):
    return np.clip(cv_round(x), 0, 255).astype(np.uint8)


def reflect101(n, lo, hi):
    p = np.arange(lo, hi + 1)
    if n == 1:
        return np.zeros_like(p)
    period = 2 * (n - 1)
    p = np.mod(p, period)
    return np.minimum(p, period - p)          # 0-based


def gaussian_blur(img, sigma):
    is_u8 = img.dtype == np.uint8
    k = int(cv_round(sigma * 3 * 2 + 1)) if is_u8 else int(cv_round(sigma * 4 * 2 + 1))
    k |= 1
    x = np.arange(k) - (k - 1) / 2
    g = np.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    if is_u8:
        kf = np.zeros(k); err = 0.0; s = 0; half = k // 2
        for i in range(half):
            adj = g[i] * 256 + err
            v0 = float(cv_round(adj)); err = adj - v0
            kf[i] = kf[k - 1 - i] = v0; s += v0
        kf[half] = 256 - 2 * s
        g = kf
    r = (k - 1) // 2
    h, w = img.shape[:2]
    ri = reflect101(h, -r, h - 1 + r)
    ci = reflect101(w, -r, w - 1 + r)
    chans = img[..., None] if img.ndim == 2 else img
    out = np.zeros(chans.shape, float)
    for c in range(chans.shape[2]):
        P = chans[ri][:, ci][..., c].astype(float)
        out[..., c] = convolve2d(convolve2d(P, g[:, None], "valid"), g[None, :], "valid")
    out = out[..., 0] if img.ndim == 2 else out
    if is_u8:
        return np.clip(np.floor((out + 32768) / 65536), 0, 255).astype(np.uint8)
    return out.astype(np.float32)


def resize_weights(n, N, method):
    scale = n / N
    rows, cols, vals = [], [], []
    d = np.arange(N)
    if method == "nearest":
        s = np.minimum(np.floor(d * scale), n - 1).astype(int)
        rows, cols, vals = d, s, np.ones(N)
    elif method == "linear":
        fx = (d + 0.5) * scale - 0.5
        sx = np.floor(fx); f = fx - sx
        lo = sx < 0; f[lo] = 0; sx[lo] = 0
        hi = sx >= n - 1; f[hi] = 0; sx[hi] = n - 1
        s2 = np.minimum(sx + 1, n - 1)
        rows = np.r_[d, d]; cols = np.r_[sx, s2].astype(int); vals = np.r_[1 - f, f]
    elif method == "cubic":
        A = -0.75
        fx = (d + 0.5) * scale - 0.5
        sx = np.floor(fx); f = fx - sx
        w0 = ((A * (f + 1) - 5 * A) * (f + 1) + 8 * A) * (f + 1) - 4 * A
        w1 = ((A + 2) * f - (A + 3)) * f * f + 1
        w2 = ((A + 2) * (1 - f) - (A + 3)) * (1 - f) * (1 - f) + 1
        w3 = 1 - w0 - w1 - w2
        for k, wk in enumerate((w0, w1, w2, w3)):
            idx = np.clip(sx + k - 1, 0, n - 1).astype(int)
            rows += list(d); cols += list(idx); vals += list(wk)
    elif method == "area":
        assert n >= N
        for dd in range(N):
            fsx1 = dd * scale; fsx2 = fsx1 + scale
            cellW = min(scale, n - fsx1)
            sx1 = int(np.ceil(fsx1)); sx2 = int(np.floor(fsx2))
            sx2 = min(sx2, n - 1); sx1 = min(sx1, sx2)
            if sx1 - fsx1 > 1e-3:
                rows.append(dd); cols.append(sx1 - 1); vals.append((sx1 - fsx1) / cellW)
            for s in range(sx1, sx2):
                rows.append(dd); cols.append(s); vals.append(1 / cellW)
            if fsx2 - sx2 > 1e-3:
                rows.append(dd); cols.append(sx2); vals.append(min(min(fsx2 - sx2, 1), cellW) / cellW)
    return sp.csr_matrix((vals, (rows, cols)), shape=(N, n))


def resize(img, H, W, method):
    h, w = img.shape[:2]
    Wy = resize_weights(h, H, method); Wx = resize_weights(w, W, method)
    chans = img[..., None] if img.ndim == 2 else img
    out = np.stack([Wy @ chans[..., c].astype(float) @ Wx.T for c in range(chans.shape[2])], -1)
    out = out[..., 0] if img.ndim == 2 else out
    return sat_u8(out) if img.dtype == np.uint8 else out.astype(img.dtype)


def clahe(I, clip_limit, tiles=8):
    h, w = I.shape
    tx = ty = tiles
    if w % tx == 0 and h % ty == 0:
        src = I
    else:
        ri = reflect101(h, 0, h + (ty - h % ty) - 1)
        ci = reflect101(w, 0, w + (tx - w % tx) - 1)
        src = I[ri][:, ci]
    tileH = src.shape[0] // ty; tileW = src.shape[1] // tx
    area = tileH * tileW
    clip = max(int(clip_limit * area / 256), 1) if clip_limit > 0 else 0
    lut_scale = np.float32(255) / np.float32(area)
    luts = np.zeros((256, ty, tx))
    for j in range(ty):
        for i in range(tx):
            block = src[j * tileH:(j + 1) * tileH, i * tileW:(i + 1) * tileW]
            hist = np.bincount(block.ravel(), minlength=256).astype(float)
            if clip > 0:
                excess = np.maximum(hist - clip, 0).sum()
                hist = np.minimum(hist, clip)
                redist = np.floor(excess / 256)
                residual = int(excess - redist * 256)
                hist = hist + redist
                if residual > 0:
                    step = max(256 // residual, 1)
                    bins = np.arange(0, 256, step)[:residual]
                    hist[bins] += 1
            cs = np.cumsum(hist)
            luts[:, j, i] = sat_u8((cs.astype(np.float32) * lut_scale).astype(float))
    inv_th = np.float32(1) / np.float32(tileH)
    inv_tw = np.float32(1) / np.float32(tileW)
    tyf = np.arange(h, dtype=np.float32)[:, None] * inv_th - np.float32(0.5)
    txf = np.arange(w, dtype=np.float32)[None, :] * inv_tw - np.float32(0.5)
    ty1 = np.floor(tyf); ya = tyf - ty1; ya1 = 1 - ya
    tx1 = np.floor(txf); xa = txf - tx1; xa1 = 1 - xa
    ty2 = np.minimum(ty1 + 1, ty - 1); ty1 = np.maximum(ty1, 0)
    tx2 = np.minimum(tx1 + 1, tx - 1); tx1 = np.maximum(tx1, 0)
    L = luts.astype(np.float32)
    p = I.astype(int)
    def g(TY, TX):
        return L[p, np.broadcast_to(TY, p.shape).astype(int), np.broadcast_to(TX, p.shape).astype(int)]
    res = (g(ty1, tx1) * xa1 + g(ty1, tx2) * xa) * ya1 + (g(ty2, tx1) * xa1 + g(ty2, tx2) * xa) * ya
    return sat_u8(res.astype(float))


def ellipse(k):
    r = c = k // 2
    se = np.zeros((k, k), np.uint8)
    inv_r2 = 1 / (r * r) if r > 0 else 0
    for i in range(k):
        dy = i - r
        if abs(dy) <= r:
            dx = int(cv_round(c * np.sqrt((r * r - dy * dy) * inv_r2)))
            j1 = max(c - dx, 0); j2 = min(c + dx + 1, k)
            se[i, j1:j2] = 1
    return se


def otsu(I):
    h = np.bincount(I.ravel(), minlength=256) / I.size
    mu = (np.arange(256) * h).sum()
    q1 = mu1 = max_sigma = 0.0; t = 0
    eps = np.finfo(np.float32).eps
    for i in range(256):
        p = h[i]
        mu1 *= q1
        q1 += p
        q2 = 1 - q1
        if min(q1, q2) < eps or max(q1, q2) > 1 - eps:
            continue
        mu1 = (mu1 + i * p) / q1
        mu2 = (mu - q1 * mu1) / q2
        sigma = q1 * q2 * (mu1 - mu2) ** 2
        if sigma > max_sigma:
            max_sigma, t = sigma, i
    return t


def rgb2gray(rgb):
    R, G, B = (rgb[..., i].astype(np.uint32) for i in range(3))
    return ((R * 9798 + G * 19235 + B * 3735 + 16384) >> 15).astype(np.uint8)


def rgb2lab(rgb):
    x = np.arange(256) / 255.0
    gam = np.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)
    tab = cv_round(255 * 8 * gam)
    n = 256 * 3 // 2 * 8
    xs = np.arange(n) / (255 * 8)
    f = np.where(xs >= 0.008856, np.cbrt(xs), xs * 7.787 + 16 / 116)
    cb = np.clip(cv_round(32768 * f), 0, 65535)
    cb[49], cb[628] = 9454, 22126          # OpenCV's softfloat cbrt at two near-ties
    M = np.array([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169],
                  [0.019334, 0.119193, 0.950227]])
    wp = np.array([0.950456, 1.0, 1.088754])
    C = cv_round(4096 * M / wp[:, None])
    desc = lambda v, s: np.floor((v + 2 ** (s - 1)) / 2 ** s)
    R, G, B = (tab[rgb[..., i].astype(int)] for i in range(3))
    fX = cb[desc(R * C[0, 0] + G * C[0, 1] + B * C[0, 2], 12).astype(int)]
    fY = cb[desc(R * C[1, 0] + G * C[1, 1] + B * C[1, 2], 12).astype(int)]
    fZ = cb[desc(R * C[2, 0] + G * C[2, 1] + B * C[2, 2], 12).astype(int)]
    Lscale = (116 * 255 + 50) // 100
    Lshift = -((16 * 255 * 2 ** 15 + 50) // 100)
    L = desc(Lscale * fY + Lshift, 15)
    a = desc(500 * (fX - fY) + 128 * 2 ** 15, 15)
    b = desc(200 * (fY - fZ) + 128 * 2 ** 15, 15)
    return np.clip(np.stack([L, a, b], -1), 0, 255).astype(np.uint8)


def lab2rgb(lab):
    lab = lab.astype(np.int64)
    f32 = np.float32
    base = 2 ** 14
    i = np.arange(256)
    y_lo = cv_round(f32(i * base * 20 * 9) / f32(17 * 29 ** 3))
    f_lo = cv_round(f32(base) * (f32(16) / f32(116) + f32(i * 5) / f32(3 * 17 * 29)))
    fy = f32(i * 100 * base) / f32(255 * 116) + f32(16 * base) / f32(116)
    YF = np.stack([np.where(i <= 20, y_lo, cv_round(fy * fy * fy / f32(base * base))),
                   np.where(i <= 20, f_lo, cv_round(fy))], 1)
    j = np.arange(-8145, -8145 + base * 9 // 4)
    cdiv = lambda a, b: np.sign(a) * (np.abs(a) // b)
    XZ = np.where(j <= 3390, cdiv(j * 108, 841) - (base * 16 // 116) * 108 // 841,
                  cdiv(cdiv(j * j, base) * j, base))
    x = np.arange(4096) / 4096
    g = np.where(x <= 0.0031308, x * 12.92, x ** (1 / 2.4) * 1.055 - 0.055).astype(np.float32)
    GAM = cv_round(f32(255) * g)
    M = np.array([[3.240479, -1.53715, -0.498535], [-0.969256, 1.875991, 0.041556],
                  [0.055648, -0.204043, 1.057311]])
    C = cv_round(4096 * M * np.array([0.950456, 1.0, 1.088754]))
    YF = YF.astype(np.int64); GAM = GAM.astype(np.int64); C = C.astype(np.int64)
    y = YF[lab[..., 0], 0]; f = YF[lab[..., 0], 1]
    adiv = ((5 * lab[..., 1] * 53687 + 2 ** 7) >> 13) - 4194
    bdiv = ((lab[..., 2] * 41943 + 2 ** 4) >> 9) - 10485 + 1
    X = XZ[f + adiv + 8145]; Z = XZ[f - bdiv + 8145]
    out = [GAM[np.clip((C[c, 0] * X + C[c, 1] * y + C[c, 2] * Z + 2 ** 13) >> 14, 0, 4095)] for c in range(3)]
    return np.stack(out, -1).astype(np.uint8)


def chamfer(bw):
    HV, DIAG, LONG, INF = 65536, 91750, 143976, 2147483647
    h, w = bw.shape; B = 2
    T = np.full((h + 2 * B, w + 2 * B), float(INF))
    for i in range(h):
        ii = i + B
        for j in range(w):
            jj = j + B
            if not bw[i, j]:
                T[ii, jj] = 0
            else:
                T[ii, jj] = min(T[ii-2, jj-1] + LONG, T[ii-2, jj+1] + LONG, T[ii-1, jj-2] + LONG,
                                T[ii-1, jj-1] + DIAG, T[ii-1, jj] + HV, T[ii-1, jj+1] + DIAG,
                                T[ii-1, jj+2] + LONG, T[ii, jj-1] + HV)
    D = np.zeros((h, w), np.float32)
    for i in range(h - 1, -1, -1):
        ii = i + B
        for j in range(w - 1, -1, -1):
            jj = j + B
            t0 = T[ii, jj]
            if t0 > HV:
                t0 = min(t0, T[ii+2, jj+1] + LONG, T[ii+2, jj-1] + LONG, T[ii+1, jj+2] + LONG,
                         T[ii+1, jj+1] + DIAG, T[ii+1, jj] + HV, T[ii+1, jj-1] + DIAG,
                         T[ii+1, jj-2] + LONG, T[ii, jj+1] + HV)
                T[ii, jj] = t0
            D[i, j] = t0 / 65536
    return D


def report(name, got, ref, tol=0):
    got = np.asarray(got, float); ref = np.asarray(ref, float)
    if got.shape != ref.shape:
        print(f"  [FAIL] {name}: shape {got.shape} vs {ref.shape}")
        return False
    d = np.abs(got - ref)
    exact = float((d == 0).mean())
    ok = d.max() <= tol
    print(f"  [{'ok ' if ok else 'OFF'}] {name:<26} max|d| {d.max():.4g}  exact {exact*100:6.2f}%"
          f"  mean|d| {d.mean():.3g}")
    return ok


def main():
    P = sio.loadmat(FIX)
    g = lambda k: P[k]
    print("Transliterated +drscreen/+cv algorithms vs OpenCV output")
    report("rgb2gray (bit-exact)", rgb2gray(g("small_rgb")), g("gray"))
    report("rgb2lab (bit-exact)", rgb2lab(g("small_rgb")), g("lab"))
    report("rgb2lab random colours (bit-exact)", rgb2lab(g("rgb_rand")), g("rgb_rand_lab"))
    report("lab2rgb (bit-exact)", lab2rgb(g("lab")), g("lab2rgb"))
    report("lab2rgb random Lab (bit-exact)", lab2rgb(g("lab_rand")), g("lab_rand_rgb"))
    for i, s in enumerate(g("blur_u8_sigmas").ravel()):
        report(f"blur u8 sigma={s}", gaussian_blur(g("gray"), s), g(f"blur_u8_{i}"), tol=1)
    for i, s in enumerate(g("blur_f_sigmas").ravel()):
        report(f"blur f32 sigma={s}", gaussian_blur(g("gray").astype(np.float32), s),
               g(f"blur_f_{i}"), tol=1e-3)
    report("resize area 283x301->151x137", resize(g("gray"), 137, 151, "area"),
           g("resize_area_down"), tol=1)
    report("resize area float", resize(g("gray").astype(np.float32), 151, 142, "area"),
           g("resize_area_half"), tol=1e-3)
    report("resize area raw->512", resize(g("src_rgb"), 512, 512, "area"),
           g("resize_area_std"), tol=1)
    report("resize area float 2x", resize(g("float_probe"), 32, 32, "area"),
           g("resize_area_float2x"), tol=1e-5)
    report("resize cubic up", resize(g("gray"), 600, 640, "cubic"), g("resize_cubic_up"), tol=1)
    report("resize cubic 512->1024", resize(g("std_rgb"), 1024, 1024, "cubic"),
           g("resize_cubic_2x"), tol=1)
    report("resize nearest", resize(g("gray"), 97, 400, "nearest"), g("resize_nearest"))
    report("clahe 512 green, clip 3", clahe(g("std_rgb")[..., 1], 3.0), g("clahe_green_3"), tol=1)
    report("clahe odd size, clip 2.5", clahe(g("gray"), 2.5), g("clahe_odd_25"), tol=1)
    ok = otsu(g("otsu_in")) == int(g("otsu_t").ravel()[0])
    print(f"  [{'ok ' if ok else 'OFF'}] otsu threshold             {otsu(g('otsu_in'))} vs {int(g('otsu_t').ravel()[0])}")
    bad = [int(k) for k in g("ellipse_ks").ravel() if not np.array_equal(ellipse(int(k)), g(f"ellipse_{int(k)}"))]
    print(f"  [{'ok ' if not bad else 'OFF'}] ellipse shapes (19 sizes)  mismatched: {bad}")
    report("chamfer distance 5x5", chamfer(g("dt_in")), g("dt_out"), tol=1e-4)


if __name__ == "__main__":
    main()
