"""PyTorch datasets, augmentation and caching.

The augmentation policy is not the usual "throw everything at it" list.  Each
transform is chosen against a specific real-world nuisance variable, and
transforms that would destroy clinical signal are explicitly excluded:

============================  =========================================
included                      the field variation it simulates
============================  =========================================
horizontal flip               left vs right eye
rotation +-20 deg             head tilt / camera roll
scale 0.85-1.15               working-distance variation
brightness / contrast jitter  flash intensity, pupil size
gamma jitter                  sensor response curves
per-channel gain              camera white-balance differences
Gaussian blur (mild)          slight defocus
Gaussian noise                low-light sensor noise
============================  =========================================

**Excluded deliberately:**

* *Vertical flip* -- it maps the superior retina to the inferior, and the
  4-2-1 severity rule is defined per quadrant.  It also makes the
  fovea-disc geometry anatomically impossible, so the model would learn that
  impossible configurations are normal.
* *Elastic / grid distortion* -- microaneurysms are 2-3 px; elastic warping
  at any useful amplitude either destroys them or invents them.
* *Aggressive cutout* -- it can delete the only lesion in a mild-NPDR image
  while the label still says "mild NPDR", which teaches the model to
  hallucinate.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..constants import NUM_LESION_CLASSES
from ..preprocess.enhance import adaptive_enhance, to_model_input
from ..preprocess.fov import standardize
from ..preprocess.quality import assess

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


# --------------------------------------------------------------------------
# Augmentation
# --------------------------------------------------------------------------
class FundusAugment:
    """Geometric + photometric augmentation applied jointly to image and masks."""

    def __init__(self, size: int = 512, train: bool = True,
                 rotate_deg: float = 20.0, scale_range: tuple[float, float] = (0.85, 1.15)):
        self.size = size
        self.train = train
        self.rotate_deg = rotate_deg
        self.scale_range = scale_range

    def __call__(self, image: np.ndarray, masks: np.ndarray | None = None,
                 rng: np.random.Generator | None = None
                 ) -> tuple[np.ndarray, np.ndarray | None]:
        if not self.train:
            return image, masks
        rng = rng or np.random.default_rng()
        h, w = image.shape[:2]

        # --- geometric (single affine, so masks stay in register) ---------
        angle = float(rng.uniform(-self.rotate_deg, self.rotate_deg))
        scale = float(rng.uniform(*self.scale_range))
        tx = float(rng.uniform(-0.03, 0.03)) * w
        ty = float(rng.uniform(-0.03, 0.03)) * h
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if masks is not None:
            masks = cv2.warpAffine(masks, M, (w, h), flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            if masks.ndim == 2:
                masks = masks[..., None]

        if rng.random() < 0.5:                       # horizontal flip only
            image = np.ascontiguousarray(image[:, ::-1])
            if masks is not None:
                masks = np.ascontiguousarray(masks[:, ::-1])

        # --- photometric --------------------------------------------------
        img = image.astype(np.float32)
        img *= rng.uniform(0.85, 1.18)                                  # exposure
        img = (img - img.mean()) * rng.uniform(0.88, 1.14) + img.mean()  # contrast
        img *= rng.uniform(0.94, 1.06, size=3).astype(np.float32)        # white balance
        img = np.clip(img, 0, 255)
        gamma = float(rng.uniform(0.85, 1.18))
        img = np.power(img / 255.0, 1.0 / gamma) * 255.0

        if rng.random() < 0.25:
            img = cv2.GaussianBlur(img, (0, 0), float(rng.uniform(0.4, 1.3)))
        if rng.random() < 0.30:
            img += rng.normal(0, float(rng.uniform(1.5, 6.0)), img.shape).astype(np.float32)

        return np.clip(img, 0, 255).astype(np.uint8), masks


def to_tensor(image: np.ndarray, normalize: bool = True) -> torch.Tensor:
    x = image.astype(np.float32) / 255.0
    if normalize:
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------
class PhantomDataset(Dataset):
    """Synthetic phantoms generated on the fly (or pre-generated for stability).

    ``deterministic=True`` regenerates the same phantom for a given index every
    epoch, which is what you want for validation; training uses fresh draws so
    the effective dataset is unbounded.
    """

    def __init__(self, n: int, size: int = 384, seed: int = 0, train: bool = True,
                 deterministic: bool | None = None, domain_shift: bool = False,
                 return_masks: bool = True, augment: bool = True,
                 preprocess: bool = True):
        self.n = n
        self.size = size
        self.seed = seed
        self.train = train
        self.deterministic = (not train) if deterministic is None else deterministic
        self.domain_shift = domain_shift
        self.return_masks = return_masks
        self.preprocess = preprocess
        self.aug = FundusAugment(size, train=train and augment)
        self._epoch = 0

    def set_epoch(self, e: int):
        self._epoch = e

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        from .synthetic import generate, CAMERAS
        base = self.seed * 1_000_003 + idx * 7919
        s = base if self.deterministic else base + self._epoch * 104_729
        rng = np.random.default_rng(s)

        cams = [c.name for c in (CAMERAS[2:] if self.domain_shift else CAMERAS)]
        sev = float(np.clip(rng.beta(2.2, 2.0) if self.domain_shift else rng.beta(1.6, 3.2), 0, 1))
        p = generate(size=self.size, seed=int(rng.integers(1 << 31)), severity=sev,
                     camera=cams[int(rng.integers(len(cams)))])

        img, fov_mask, fov = standardize(p.image, size=self.size)

        # Keep every mask in register with the standardised image.
        stack = [p.vessel_mask, p.disc_mask] + [p.lesion_masks[..., c]
                                                for c in range(NUM_LESION_CLASSES)]
        stack = np.stack(stack, axis=-1)
        stack = _apply_same_geometry(stack, p.image.shape[:2], fov, self.size)

        if self.preprocess:
            q = assess(img, fov_mask, fov)
            img, _ = adaptive_enhance(img, fov_mask, q.issues)
            img = to_model_input(img, fov_mask, mode="hybrid")
        else:
            q = None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img, stack = self.aug(img, stack, rng)

        out = {
            "image": to_tensor(img),
            "grade": torch.tensor(p.grade, dtype=torch.long),
            "quality_label": torch.tensor(p.quality_label, dtype=torch.long),
            "fov_mask": torch.from_numpy((fov_mask > 0).astype(np.float32))[None],
        }
        if self.return_masks:
            m = (stack > 127).astype(np.float32)
            out["vessel_mask"] = torch.from_numpy(m[..., 0])[None]
            out["disc_mask"] = torch.from_numpy(m[..., 1])[None]
            out["lesion_mask"] = torch.from_numpy(
                np.ascontiguousarray(m[..., 2:].transpose(2, 0, 1)))
        return out


def _apply_same_geometry(masks: np.ndarray, raw_shape: tuple[int, int],
                         fov, size: int) -> np.ndarray:
    """Replicate crop -> square-pad -> resize on a mask stack."""
    h, w = raw_shape
    x0, y0, x1, y1 = fov.bbox
    px, py = int(0.02 * (x1 - x0)), int(0.02 * (y1 - y0))
    x0, y0 = max(0, x0 - px), max(0, y0 - py)
    x1, y1 = min(w, x1 + px), min(h, y1 + py)
    m = masks[y0:y1, x0:x1]
    ch, cw = m.shape[:2]
    side = max(ch, cw)
    top, left = (side - ch) // 2, (side - cw) // 2
    m = cv2.copyMakeBorder(m, top, side - ch - top, left, side - cw - left,
                           cv2.BORDER_CONSTANT, value=0)
    if m.ndim == 2:
        m = m[..., None]
    return cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST).reshape(
        size, size, masks.shape[-1] if masks.ndim == 3 else 1)


class FundusDataset(Dataset):
    """Real-image dataset over :class:`drscreen.data.registry.Sample` records.

    Preprocessed tensors are cached to disk keyed by (path, mtime, size,
    pipeline-version): the FOV + enhancement stage costs ~200 ms per image,
    which would otherwise dominate GPU-bound training.
    """

    PIPELINE_VERSION = "v1"

    def __init__(self, samples: list, size: int = 512, train: bool = True,
                 cache_dir: str | Path | None = None, augment: bool = True,
                 load_masks: bool = False, preprocess: bool = True):
        self.samples = samples
        self.size = size
        self.train = train
        self.load_masks = load_masks
        self.preprocess = preprocess
        self.aug = FundusAugment(size, train=train and augment)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.samples)

    def _cache_key(self, s) -> str:
        try:
            mtime = s.image_path.stat().st_mtime_ns
        except OSError:
            mtime = 0
        raw = f"{s.image_path}|{mtime}|{self.size}|{self.PIPELINE_VERSION}|{self.preprocess}"
        return hashlib.sha1(raw.encode()).hexdigest()

    def _load_processed(self, s) -> tuple[np.ndarray, np.ndarray, dict]:
        if self.cache_dir:
            f = self.cache_dir / f"{self._cache_key(s)}.pkl"
            if f.exists():
                try:
                    with f.open("rb") as fh:
                        return pickle.load(fh)
                except Exception:
                    pass

        raw = cv2.imread(str(s.image_path), cv2.IMREAD_COLOR)
        if raw is None:
            raise FileNotFoundError(f"Unreadable image: {s.image_path}")
        img, fov_mask, fov = standardize(raw, size=self.size)
        q = assess(img, fov_mask, fov)
        if self.preprocess:
            img, applied = adaptive_enhance(img, fov_mask, q.issues)
            img = to_model_input(img, fov_mask, mode="hybrid")
        else:
            applied = []
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        meta = {"quality": q.to_dict(), "applied": applied}

        payload = (img, fov_mask, meta)
        if self.cache_dir:
            try:
                with (self.cache_dir / f"{self._cache_key(s)}.pkl").open("wb") as fh:
                    pickle.dump(payload, fh, protocol=4)
            except Exception:
                pass
        return payload

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        img, fov_mask, meta = self._load_processed(s)
        rng = np.random.default_rng(idx * 7919 + (0 if not self.train else np.random.randint(1 << 30)))
        img, _ = self.aug(img, None, rng)

        out = {
            "image": to_tensor(img),
            "grade": torch.tensor(s.grade if s.grade is not None else -1, dtype=torch.long),
            "fov_mask": torch.from_numpy((fov_mask > 0).astype(np.float32))[None],
            "index": torch.tensor(idx, dtype=torch.long),
            "gradeable": torch.tensor(bool(meta["quality"]["gradeable"])),
        }
        if self.load_masks and s.masks:
            planes = []
            for key in ("microaneurysm", "hemorrhage", "hard_exudate",
                        "soft_exudate", "neovascularization"):
                p = s.masks.get(key)
                if p is None:
                    planes.append(np.zeros((self.size, self.size), np.float32))
                else:
                    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    m = cv2.resize(m, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
                    planes.append((m > 127).astype(np.float32))
            out["lesion_mask"] = torch.from_numpy(np.stack(planes))
            if "vessel" in s.masks:
                m = cv2.imread(str(s.masks["vessel"]), cv2.IMREAD_GRAYSCALE)
                m = cv2.resize(m, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
                out["vessel_mask"] = torch.from_numpy((m > 127).astype(np.float32))[None]
        return out
