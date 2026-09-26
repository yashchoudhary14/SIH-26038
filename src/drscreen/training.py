"""Training loops for the segmentation and grading stages.

Shared conventions: AMP on CUDA, cosine schedule with warmup, EMA weights,
gradient clipping, and early stopping on the metric that actually matters for
each stage (Dice for segmentation, referable-DR AUC for grading -- *not*
accuracy, which a model can maximise by never predicting the minority grades).
"""
from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def device_of(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EMA:
    """Exponential moving average of weights.

    Worth the few lines: with the small effective batch sizes that
    high-resolution fundus training forces, the raw weights bounce enough that
    the last checkpoint is often measurably worse than the average.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for s, m in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(self.decay).add_(m.detach(), alpha=1 - self.decay)
            else:
                s.copy_(m)


def cosine_warmup(optimizer, total_steps: int, warmup_frac: float = 0.05,
                  min_lr_frac: float = 0.02):
    warmup = max(1, int(total_steps * warmup_frac))

    def fn(step: int) -> float:
        if step < warmup:
            return step / warmup
        t = (step - warmup) / max(1, total_steps - warmup)
        return min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


@dataclass
class TrainLog:
    epochs: list = field(default_factory=list)
    best_metric: float = -1e9
    best_epoch: int = -1
    elapsed_s: float = 0.0
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------
@torch.no_grad()
def dice_per_class(logits: torch.Tensor, target: torch.Tensor,
                   thr: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """Per-class Dice, with undefined cases returned as NaN rather than 1.0.

    The smoothed form ``(2*inter + eps) / (denom + eps)`` returns **1.0** when
    both the prediction and the target are empty. That convention is common and
    it is a trap: a model that has learned to predict nothing, evaluated on a
    batch that happens to contain no positives, scores a perfect 1.0.

    That is not hypothetical. It hid a real bug here: IDRiD encodes mask
    foreground as pixel value 76, the loader thresholded at >127, every mask
    became empty, and segmentation reported mean Dice 1.0000 across all five
    lesion classes while the training loss sat flat at 0.95.

    Dice is genuinely undefined with no positives, so it is reported as NaN and
    the caller averages over the classes that were actually evaluable.
    """
    p = (torch.sigmoid(logits) > thr).float()
    dims = (0, 2, 3)
    inter = (p * target).sum(dims)
    denom = p.sum(dims) + target.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    # Undefined where the ground truth has no positives for that class.
    return torch.where(target.sum(dims) > 0, dice,
                       torch.full_like(dice, float("nan")))


def train_segmentation(model, train_ds, val_ds, *, mask_key: str = "lesion_mask",
                       epochs: int = 12, batch_size: int = 8, lr: float = 3e-4,
                       weight_decay: float = 1e-4, num_workers: int = 4,
                       device: str = "auto", out_dir: str | Path = "outputs/seg",
                       amp: bool = True, pos_weight: float | None = None,
                       log_every: int = 20, ema_decay: float = 0.999,
                       channel_mask=None) -> TrainLog:
    from .models.segmentation import segmentation_loss

    dev = device_of(device)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(dev)

    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                    pin_memory=(dev.type == "cuda"), drop_last=True,
                    persistent_workers=num_workers > 0)
    vl = DataLoader(val_ds, batch_size=max(1, batch_size), shuffle=False,
                    num_workers=num_workers, pin_memory=(dev.type == "cuda"),
                    persistent_workers=num_workers > 0)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = cosine_warmup(opt, epochs * max(1, len(tl)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp and dev.type == "cuda")
    ema = EMA(model, ema_decay)

    pw = None
    if pos_weight is not None:
        pw = torch.tensor(float(pos_weight), device=dev)

    cmask = None
    if channel_mask is not None:
        cmask = torch.as_tensor(channel_mask, dtype=torch.bool, device=dev)

    log = TrainLog(config={"epochs": epochs, "batch_size": batch_size, "lr": lr,
                           "mask_key": mask_key, "device": str(dev),
                           "supervised_channels": (None if channel_mask is None
                                                   else [bool(b) for b in channel_mask])})
    t_start = time.time()

    for ep in range(epochs):
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(ep)
        model.train()
        running, nb = 0.0, 0
        for i, batch in enumerate(tl):
            x = batch["image"].to(dev, non_blocking=True)
            y = batch[mask_key].to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp and dev.type == "cuda"):
                out = model(x)
                if isinstance(out, tuple):
                    logits, aux = out
                else:
                    logits, aux = out, None
                loss = segmentation_loss(logits.float(), y,
                                         [a.float() for a in aux] if aux else None,
                                         pos_weight=pw, channel_mask=cmask)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            ema.update(model)
            running += float(loss.item()); nb += 1
            if log_every and i % log_every == 0:
                print(f"  ep{ep+1} step {i}/{len(tl)} loss {running/max(nb,1):.4f}", flush=True)

        # --- validation on the EMA weights -------------------------------
        ema.shadow.eval()
        dices = []
        with torch.no_grad():
            for batch in vl:
                x = batch["image"].to(dev, non_blocking=True)
                y = batch[mask_key].to(dev, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=amp and dev.type == "cuda"):
                    logits = ema.shadow(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
                dices.append(dice_per_class(logits.float(), y).cpu().numpy())
        # nanmean: classes absent from the whole validation split are
        # undefined, not perfect. See dice_per_class.
        stacked = np.stack(dices) if dices else np.full((1, 1), np.nan)
        with np.errstate(invalid="ignore"):
            per_class = np.nanmean(stacked, axis=0)
        evaluable = int(np.sum(~np.isnan(per_class)))
        mean_dice = float(np.nanmean(per_class)) if evaluable else 0.0

        if evaluable == 0:
            raise RuntimeError(
                "No validation class has any positive ground-truth pixels, so "
                "Dice is undefined for every class. The masks are almost "
                "certainly empty -- check the foreground encoding of your "
                "annotation files (IDRiD uses 76, not 255).")

        rec = {"epoch": ep + 1, "train_loss": running / max(nb, 1),
               "val_dice_mean": mean_dice,
               "val_classes_evaluable": evaluable,
               "val_dice_per_class": [None if np.isnan(v) else round(float(v), 4)
                                      for v in per_class]}
        log.epochs.append(rec)
        print(f"[seg] epoch {ep+1}/{epochs} loss {rec['train_loss']:.4f} "
              f"dice {mean_dice:.4f} ({evaluable} cls) "
              f"{rec['val_dice_per_class']}", flush=True)

        if mean_dice > log.best_metric:
            log.best_metric, log.best_epoch = mean_dice, ep + 1
            torch.save({"model": ema.shadow.state_dict(), "epoch": ep + 1,
                        "dice": mean_dice, "config": log.config},
                       out_dir / "best.pt")

    log.elapsed_s = time.time() - t_start
    (out_dir / "train_log.json").write_text(json.dumps(log.to_dict(), indent=2))
    return log


# --------------------------------------------------------------------------
# Grading
# --------------------------------------------------------------------------
def train_grader(model, train_ds, val_ds, *, epochs: int = 15, batch_size: int = 16,
                 lr: float = 3e-4, backbone_lr_mult: float = 0.25,
                 weight_decay: float = 1e-4, num_workers: int = 4,
                 device: str = "auto", out_dir: str | Path = "outputs/grader",
                 amp: bool = True, class_weights: np.ndarray | None = None,
                 clinical_fn=None, log_every: int = 20,
                 ema_decay: float = 0.999, sampler=None,
                 select_on: str = "qwk") -> TrainLog:
    """Train the ordinal grader.

    ``clinical_fn(batch) -> Tensor`` supplies the clinical feature vector; pass
    ``None`` to train the image-only ablation arm.

    The backbone gets a lower learning rate than the head: the pretrained
    features are already good and a full-rate update in the first epochs
    destroys them, which is the usual reason transfer learning underperforms
    on medical images.
    """
    from .models.grader import corn_loss, corn_predict, referable_prob
    from .constants import REFERABLE_THRESHOLD, SIGHT_THREATENING_THRESHOLD
    from sklearn.metrics import roc_auc_score

    dev = device_of(device)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(dev)

    # A grade-stratified sampler is what keeps the deep CORN conditionals fed:
    # under natural sampling grades 3-4 are ~17% of the cohort, so task 3 sees
    # two or three images per batch of 16 and its gradient is mostly noise.
    tl = DataLoader(train_ds, batch_size=batch_size,
                    shuffle=(sampler is None), sampler=sampler,
                    num_workers=num_workers,
                    pin_memory=(dev.type == "cuda"), drop_last=True,
                    persistent_workers=num_workers > 0)
    vl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    pin_memory=(dev.type == "cuda"), persistent_workers=num_workers > 0)

    backbone_params, head_params = [], []
    for n, p in model.named_parameters():
        (backbone_params if n.startswith("backbone.") else head_params).append(p)
    opt = torch.optim.AdamW(
        [{"params": backbone_params, "lr": lr * backbone_lr_mult},
         {"params": head_params, "lr": lr}], weight_decay=weight_decay)
    sched = cosine_warmup(opt, epochs * max(1, len(tl)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp and dev.type == "cuda")
    ema = EMA(model, ema_decay)

    cw = None
    if class_weights is not None:
        cw = torch.tensor(np.asarray(class_weights, np.float32), device=dev)

    log = TrainLog(config={"epochs": epochs, "batch_size": batch_size, "lr": lr,
                           "device": str(dev), "use_clinical": clinical_fn is not None})
    t_start = time.time()

    for ep in range(epochs):
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(ep)
        model.train()
        running, nb = 0.0, 0
        for i, batch in enumerate(tl):
            x = batch["image"].to(dev, non_blocking=True)
            y = batch["grade"].to(dev, non_blocking=True)
            c = clinical_fn(batch).to(dev, non_blocking=True) if clinical_fn else None
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp and dev.type == "cuda"):
                logits = model(x, c)
                loss = corn_loss(logits.float(), y, model.num_classes, cw)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            ema.update(model)
            running += float(loss.item()); nb += 1
            if log_every and i % log_every == 0:
                print(f"  ep{ep+1} step {i}/{len(tl)} loss {running/max(nb,1):.4f}", flush=True)

        # --- validation ---------------------------------------------------
        ema.shadow.eval()
        ys, preds, refs = [], [], []
        with torch.no_grad():
            for batch in vl:
                x = batch["image"].to(dev, non_blocking=True)
                y = batch["grade"]
                c = clinical_fn(batch).to(dev, non_blocking=True) if clinical_fn else None
                with torch.amp.autocast("cuda", enabled=amp and dev.type == "cuda"):
                    logits = ema.shadow(x, c).float()
                ys.append(y.numpy())
                preds.append(corn_predict(logits).cpu().numpy())
                refs.append(referable_prob(logits).cpu().numpy())

        y_true = np.concatenate(ys); y_pred = np.concatenate(preds)
        ref_score = np.concatenate(refs)
        ref_true = (y_true >= REFERABLE_THRESHOLD).astype(int)

        auc = float(roc_auc_score(ref_true, ref_score)) if 0 < ref_true.sum() < len(ref_true) else float("nan")
        from .evaluation.metrics import quadratic_weighted_kappa
        qwk = quadratic_weighted_kappa(y_true, y_pred)
        acc = float((y_true == y_pred).mean())

        # Sight-threatening recall is tracked per epoch because it is the axis
        # the model was failing on (0.43 on grades 3 and 4) and the axis the
        # old selection metric was structurally unable to see.
        st_true = y_true >= SIGHT_THREATENING_THRESHOLD
        st_hit = int(((y_pred >= SIGHT_THREATENING_THRESHOLD) & st_true).sum())
        sens_st = float(st_hit / int(st_true.sum())) if int(st_true.sum()) else float("nan")
        per_grade = [float((y_pred[y_true == g] == g).mean())
                     if int((y_true == g).sum()) else float("nan")
                     for g in range(model.num_classes)]

        rec = {"epoch": ep + 1, "train_loss": running / max(nb, 1),
               "val_auc_referable": auc, "val_qwk": qwk, "val_accuracy": acc,
               "val_sens_sight_threatening": sens_st,
               "val_recall_per_grade": per_grade}
        log.epochs.append(rec)
        print(f"[grader] epoch {ep+1}/{epochs} loss {rec['train_loss']:.4f} "
              f"AUC(ref) {auc:.4f} QWK {qwk:.4f} acc {acc:.4f} "
              f"sens(g>=3) {sens_st:.4f}", flush=True)

        # Selection metric.
        #
        # This was referable-DR AUC, scored on ``referable_prob(logits)`` =
        # ``corn_cumulative_probs(logits)[:, 1]`` = sigma(z0) * sigma(z1).
        # That expression does not contain z2 or z3 -- the two output units
        # that decide severe NPDR and proliferative DR -- so the criterion was
        # mathematically incapable of observing the model collapse grades 3
        # and 4 into grade 2. It duly preferred epoch 30 to epoch 18 for a
        # 0.002 AUC gain while QWK fell. The old comment here claimed the
        # metric protected against "a model that never predicts grades 3-4";
        # it was the one metric on the list that could not.
        #
        # QWK weights errors quadratically, so calling a grade-4 eye "grade 2"
        # costs four times what "grade 3" costs -- it reads the severity axis
        # directly. Referable discrimination is not sacrificed: it plateaus by
        # epoch 10, and the referral threshold is fitted afterwards on the
        # validation split regardless of which epoch is kept.
        scores = {"qwk": qwk, "referable_auc": auc,
                  "composite": 0.5 * qwk + 0.5 * (sens_st if sens_st == sens_st else 0.0)}
        if select_on not in scores:
            raise ValueError(f"select_on must be one of {sorted(scores)}, got {select_on!r}")
        score = scores[select_on]
        if score != score:                            # NaN guard
            score = auc if auc == auc else 0.0
        ck = {"model": ema.shadow.state_dict(), "epoch": ep + 1,
              "auc": auc, "qwk": qwk, "sens_sight_threatening": sens_st,
              "select_on": select_on, "config": log.config}
        if score > log.best_metric:
            log.best_metric, log.best_epoch = score, ep + 1
            torch.save(ck, out_dir / "best.pt")
        # Keep the final epoch too. Previously only ``best.pt`` was written and
        # it was overwritten in place, so once a run finished, every other
        # epoch was gone and revisiting the selection metric meant retraining.
        torch.save(ck, out_dir / "last.pt")

    log.elapsed_s = time.time() - t_start
    (out_dir / "train_log.json").write_text(json.dumps(log.to_dict(), indent=2))
    return log


@torch.no_grad()
def collect_logits(model, dataset, *, batch_size: int = 16, num_workers: int = 4,
                   device: str = "auto", clinical_fn=None) -> dict:
    """Run a model over a dataset and return logits/labels for calibration."""
    dev = device_of(device)
    model = model.to(dev).eval()
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=(dev.type == "cuda"))
    L, Y = [], []
    for batch in dl:
        x = batch["image"].to(dev, non_blocking=True)
        c = clinical_fn(batch).to(dev, non_blocking=True) if clinical_fn else None
        with torch.amp.autocast("cuda", enabled=dev.type == "cuda"):
            logits = model(x, c).float()
        L.append(logits.cpu())
        Y.append(batch["grade"])
    return {"logits": torch.cat(L), "labels": torch.cat(Y)}
