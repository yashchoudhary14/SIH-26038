"""Localisation accuracy of the analytic optic-disc / fovea detector.

Reports error in disc diameters (DD), the unit the clinical criteria use.
Literature reference points on real data: optic-disc detection is typically
reported at 95-99% within 1 DD, fovea at 90-96% within 1 DD.
"""
import argparse, math, sys
import numpy as np

from drscreen.data.synthetic import generate
from drscreen.preprocess.fov import standardize
from drscreen.preprocess.landmarks import locate


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


def main(n=100, size=512, seed=0):
    d_err, f_err = [], []
    for i in range(n):
        p = generate(size=size, seed=seed + i)
        img, msk, fov = standardize(p.image, size=size)
        gd, sc = project(p.disc_xy, fov, p.image.shape, size)
        gf, _ = project(p.fovea_xy, fov, p.image.shape, size)
        lm = locate(img, msk)
        dd_px = p.disc_radius * 2 * sc
        d_err.append(math.dist(lm.disc_xy, gd) / dd_px)
        f_err.append(math.dist(lm.fovea_xy, gf) / dd_px)

    d_err, f_err = np.array(d_err), np.array(f_err)
    print(f"n = {n} phantoms\n")
    for name, e in (("optic disc", d_err), ("fovea", f_err)):
        print(f"{name:11s}  median {np.median(e):.3f} DD | "
              f"<=0.5 DD {100*(e<=0.5).mean():5.1f}% | "
              f"<=1 DD {100*(e<=1.0).mean():5.1f}% | "
              f"<=2 DD {100*(e<=2.0).mean():5.1f}%")
    return float((d_err <= 1).mean()), float((f_err <= 1).mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.n, a.size, a.seed)
