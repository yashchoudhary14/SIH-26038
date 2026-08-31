"""Materialised cohorts: a single on-disk format for phantom and real data.

Training the fusion grader needs clinical features, which need lesion masks,
which need a trained segmentation model.  That is a genuine three-stage
dependency, and running the segmentation network inside the grader's training
loop would cost ~300 ms per image in connected-component analysis alone.

The resolution is to materialise: images and labels are written once, the
segmentation stage writes its predictions back into the same record, and the
grader reads cached feature vectors.  This is also exactly the workflow the
real datasets require, so the phantom path and the APTOS/IDRiD path share one
code path rather than diverging into a demo branch and a real branch.

Layout::

    <root>/
      manifest.jsonl        one JSON record per case
      images/<uid>.png      standardised, enhanced model input (hybrid planes,
                            stored in the exact channel order the model is
                            fed at inference -- NOT a viewable RGB/BGR image)
      fov/<uid>.png         field-of-view mask
      masks/<uid>.npz       ground-truth masks (phantom / IDRiD only)
      features/<uid>.npy    clinical feature vector (written by stage 2)
      preds/<uid>.npz       predicted lesion probabilities (stage 2)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..constants import NUM_LESION_CLASSES
from .torch_data import FundusAugment, to_tensor


@dataclass
class CohortRecord:
    uid: str
    grade: int
    split: str
    source: str
    quality_label: int = 2
    camera: str = ""
    has_masks: bool = False
    meta: dict | None = None

    def to_json(self) -> str:
        return json.dumps({k: v for k, v in self.__dict__.items()})


class CohortWriter:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        for sub in ("images", "fov", "masks", "features", "preds"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self._fh = (self.root / "manifest.jsonl").open("w", encoding="utf-8")

    def add(self, rec: CohortRecord, image: np.ndarray, fov: np.ndarray,
            masks: dict[str, np.ndarray] | None = None):
        cv2.imwrite(str(self.root / "images" / f"{rec.uid}.png"), image)
        cv2.imwrite(str(self.root / "fov" / f"{rec.uid}.png"), fov)
        if masks:
            np.savez_compressed(self.root / "masks" / f"{rec.uid}.npz",
                                **{k: v.astype(np.uint8) for k, v in masks.items()})
            rec.has_masks = True
        self._fh.write(rec.to_json() + "\n")

    def close(self):
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def read_manifest(root: str | Path, split: str | None = None) -> list[CohortRecord]:
    root = Path(root)
    out = []
    with (root / "manifest.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if split and d.get("split") != split:
                continue
            out.append(CohortRecord(**d))
    return out


class CohortDataset(Dataset):
    """Reads a materialised cohort. Optionally serves cached clinical features."""

    def __init__(self, root: str | Path, split: str, size: int = 384,
                 train: bool = True, augment: bool = True,
                 with_masks: bool = False, with_features: bool = False,
                 feature_mode: str = "predicted"):
        self.root = Path(root)
        self.records = read_manifest(self.root, split)
        self.size = size
        self.train = train
        self.with_masks = with_masks
        self.with_features = with_features
        self.feature_mode = feature_mode
        self.aug = FundusAugment(size, train=train and augment)
        if not self.records:
            raise ValueError(f"No records for split '{split}' under {self.root}")

    def __len__(self) -> int:
        return len(self.records)

    def grades(self) -> np.ndarray:
        return np.array([r.grade for r in self.records], np.int64)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        img = cv2.imread(str(self.root / "images" / f"{r.uid}.png"), cv2.IMREAD_COLOR)
        fov = cv2.imread(str(self.root / "fov" / f"{r.uid}.png"), cv2.IMREAD_GRAYSCALE)
        if img.shape[0] != self.size:
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
            fov = cv2.resize(fov, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        stack = None
        if self.with_masks and r.has_masks:
            z = np.load(self.root / "masks" / f"{r.uid}.npz")
            planes = [z["vessel"], z["disc"]] + [z["lesions"][..., c]
                                                 for c in range(NUM_LESION_CLASSES)]
            stack = np.stack(planes, axis=-1)
            if stack.shape[0] != self.size:
                stack = cv2.resize(stack, (self.size, self.size),
                                   interpolation=cv2.INTER_NEAREST)

        rng = np.random.default_rng(
            idx * 7919 + (np.random.randint(1 << 30) if self.train else 0))
        img_aug, stack = self.aug(img, stack, rng)

        out = {
            "image": to_tensor(img_aug),
            "grade": torch.tensor(int(r.grade), dtype=torch.long),
            "quality_label": torch.tensor(int(r.quality_label), dtype=torch.long),
            "fov_mask": torch.from_numpy((fov > 0).astype(np.float32))[None],
            "index": torch.tensor(idx, dtype=torch.long),
        }
        if stack is not None:
            m = (stack > 127).astype(np.float32)
            out["vessel_mask"] = torch.from_numpy(m[..., 0])[None]
            out["disc_mask"] = torch.from_numpy(m[..., 1])[None]
            out["lesion_mask"] = torch.from_numpy(
                np.ascontiguousarray(m[..., 2:].transpose(2, 0, 1)))

        if self.with_features:
            sub = "features" if self.feature_mode == "predicted" else "features_gt"
            f = self.root / sub / f"{r.uid}.npy"
            vec = np.load(f) if f.exists() else np.zeros(
                _clinical_dim(), np.float32)
            out["clinical"] = torch.from_numpy(vec.astype(np.float32))
        return out


def _clinical_dim() -> int:
    from ..models.lesion_features import ClinicalFeatures
    return ClinicalFeatures.vector_size()


def clinical_from_batch(batch: dict) -> torch.Tensor:
    """``clinical_fn`` for :func:`drscreen.training.train_grader`."""
    return batch["clinical"]
