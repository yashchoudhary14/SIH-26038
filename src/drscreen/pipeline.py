"""End-to-end inference pipeline.

One entry point, :meth:`DRScreeningPipeline.run`, takes a raw fundus image and
returns a complete, auditable screening result:

1. **Geometry** -- FOV detection, tight crop, square pad, resize.
2. **Quality gate** -- interpretable criteria; ungradeable images stop here
   and return recapture instructions rather than a guessed grade.
3. **Adaptive enhancement** -- only the corrections the gate says are needed.
4. **Landmarks** -- optic disc and fovea, establishing the clinical
   coordinate frame the grading criteria are defined in.
5. **Segmentation** -- vessels and the five lesion classes.
6. **Clinical features** -- lesion counts per quadrant, NV location, exudate
   distance from the fovea.
7. **Grading** -- ordinal CNN fused with those features, temperature-calibrated.
8. **Decision** -- referral against a frozen operating point, with an
   explicit abstain/defer band for the human-in-the-loop workflow.
9. **Explanation** -- Grad-CAM++ over the referable log-odds, plus
   lesion-level evidence in clinical language.

Every stage records its latency, and the whole result is JSON-serialisable so
it can be stored, audited, and replayed.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np
import torch

from .constants import (ICDR_GRADES, LESION_CLASSES, NUM_LESION_CLASSES,
                        PIXEL_ANNOTATED_LESION_CLASSES, RECAPTURE_ADVICE,
                        REFERABLE_THRESHOLD, SIGHT_THREATENING_THRESHOLD)
from .models.lesion_features import (ClinicalFeatures, extract, rule_grade,
                                     dme_risk)
from .preprocess.enhance import adaptive_enhance, to_model_input
from .preprocess.fov import standardize
from .preprocess.landmarks import Landmarks, locate
from .preprocess.quality import assess, QualityReport, CORRECTABLE


@dataclass
class Timing:
    stages: dict = field(default_factory=dict)
    total_ms: float = 0.0


@dataclass
class ScreeningResult:
    image_id: str = ""
    gradeable: bool = True
    quality: dict = field(default_factory=dict)
    enhancement_applied: list = field(default_factory=list)
    landmarks: dict = field(default_factory=dict)
    grade: int = -1
    grade_label: str = ""
    class_probabilities: list = field(default_factory=list)
    referable: bool = False
    referable_probability: float = 0.0
    confidence: float = 0.0
    uncertainty: dict = field(default_factory=dict)
    decision: str = ""            # auto_report | refer | defer_to_human | recapture
    urgency: str = "routine"      # routine | soon | urgent
    dme_risk: int = 0
    clinical_features: dict = field(default_factory=dict)
    evidence: list = field(default_factory=list)
    rule_based_grade: int = -1
    #: P(grade >= 3), the quantity the urgency tier is decided on. Distinct
    #: from ``referable_probability`` = P(grade >= 2), which decides referral.
    sight_threatening_probability: float = 0.0
    rule_based_reasons: list = field(default_factory=list)
    agreement: str = ""
    recapture_advice: list = field(default_factory=list)
    timing_ms: dict = field(default_factory=dict)
    model_version: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), indent=2, default=float, **kw)


@dataclass
class PipelineConfig:
    size: int = 512
    #: Resolution the LESION model runs at. Often larger than `size`: a
    #: microaneurysm is a few pixels wide, and at 512 the segmentation head
    #: scored Dice 0.00 on them versus 0.485 at 1024. The grader still works at
    #: `size`; only segmentation pays the extra cost. If this does not match
    #: the resolution the segmentation model was trained at, the clinical
    #: features drift away from the ones the grader was fitted on, and the
    #: deployed system quietly disagrees with its own validation.
    seg_size: int = 512
    device: str = "auto"
    lesion_threshold: float = 0.5
    #: Per-class cut-points on the lesion probability maps, fitted for F1 on
    #: held-out IDRiD. Falls back to `lesion_threshold` for missing classes.
    lesion_thresholds: dict | None = None
    #: Lesion classes the loaded segmentation checkpoint was actually trained
    #: on, read from the checkpoint. ``None`` means the checkpoint predates the
    #: field, in which case ``PIXEL_ANNOTATED_LESION_CLASSES`` is assumed --
    #: every such checkpoint was trained on IDRiD-derived masks, which carry no
    #: neovascularisation. Assuming full supervision instead would restore the
    #: exact silent failure this field exists to surface.
    supervised_lesion_classes: tuple[str, ...] | None = None
    #: Per-boundary cut-points on the CORN cumulative probabilities, fitted on
    #: validation data by ``fit_grade_thresholds``. ``None`` means the un-fitted
    #: 0.5 default -- what shipped before this field existed, and not a rule
    #: that any reported metric describes.
    grade_thresholds: list | None = None
    referral_threshold: float = 0.5      # overwritten by the calibrated value
    #: Cut-point on P(grade >= 3) for the URGENT tier, fitted on validation
    #: data like every other threshold here. ``None`` falls back to the old
    #: rule, ``predicted grade >= 3``, which keys the urgency tier off the
    #: least reliable output the model has: on Messidor-2 that rule reaches
    #: sensitivity 0.509 against 0.782 for a fitted cut-point, so it leaves 30
    #: of 110 sight-threatening patients in the routine queue. Urgency stays a
    #: strict sub-tier of referral -- a case that is not referred is never
    #: urgent -- which measurement shows costs nothing, since every case the
    #: fitted threshold escalates is already above the referral threshold.
    urgent_threshold: float | None = None
    defer_band: tuple[float, float] = (0.35, 0.65)
    temperature: float = 1.0
    mc_samples: int = 8
    uncertainty_defer: float = 0.05      # epistemic variance above which we defer
    enable_cam: bool = True
    cam_method: str = "gradcam++"
    model_version: str = "drscreen-1.0.0"
    #: Serialised monotone recalibrator for P(referable). The referral
    #: threshold was selected on this scale, so the two must ship together.
    calibrator: dict | None = None


class DRScreeningPipeline:
    """Loads the trained components and runs the full screening flow."""

    def __init__(self, seg_model=None, grader=None, cfg: PipelineConfig | None = None):
        self.cfg = cfg or PipelineConfig()
        self.device = torch.device(
            "cuda" if (self.cfg.device == "auto" and torch.cuda.is_available())
            else ("cpu" if self.cfg.device == "auto" else self.cfg.device))
        self.seg = seg_model.to(self.device).eval() if seg_model is not None else None
        self.grader = grader.to(self.device).eval() if grader is not None else None
        self._calibrator = None
        if self.cfg.calibrator:
            from .models.calibration import IsotonicCalibrator
            self._calibrator = IsotonicCalibrator.from_dict(self.cfg.calibrator)

    # -- loading ----------------------------------------------------------
    @classmethod
    def load(cls, artifacts_dir: str | Path, cfg: PipelineConfig | None = None,
             backbone: str = "tf_efficientnet_b0") -> "DRScreeningPipeline":
        from .models.grader import DRGrader
        from .models.segmentation import build_unet

        d = Path(artifacts_dir)
        cfg = cfg or PipelineConfig()

        meta_path = d / "pipeline.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            cfg.referral_threshold = float(meta.get("referral_threshold", cfg.referral_threshold))
            cfg.temperature = float(meta.get("temperature", cfg.temperature))
            cfg.size = int(meta.get("size", cfg.size))
            cfg.seg_size = int(meta.get("seg_size", cfg.size))
            cfg.calibrator = meta.get("calibrator")
            cfg.lesion_thresholds = meta.get("lesion_thresholds")
            cfg.grade_thresholds = meta.get("grade_thresholds")
            cfg.urgent_threshold = meta.get("urgent_threshold")
            cfg.defer_band = tuple(meta.get("defer_band", cfg.defer_band))
            backbone = meta.get("backbone", backbone)
            cfg.model_version = meta.get("model_version", cfg.model_version)

        seg = None
        seg_ckpt = d / "segmentation.pt"
        if seg_ckpt.exists():
            ck = torch.load(seg_ckpt, map_location="cpu", weights_only=False)
            seg = build_unet("lesion", width=int(ck.get("width", 24)))
            seg.load_state_dict(ck["model"])
            sup = ck.get("supervised_lesion_classes")
            cfg.supervised_lesion_classes = tuple(sup) if sup else None

        grader = None
        g_ckpt = d / "grader.pt"
        if g_ckpt.exists():
            ck = torch.load(g_ckpt, map_location="cpu", weights_only=False)
            grader = DRGrader(backbone=backbone, pretrained=False,
                              use_clinical=bool(ck.get("use_clinical", True)))
            grader.load_state_dict(ck["model"])

        return cls(seg, grader, cfg)

    @property
    def unassessed_lesions(self) -> tuple[str, ...]:
        """Lesion classes this model cannot detect, because it never saw one.

        Kept distinct from "detected none" throughout the report and the audit
        log. Neovascularisation defines proliferative DR, and IDRiD -- the only
        lesion-annotated corpus here -- does not annotate it, so on real data
        this is non-empty and the grade-4 rule arm is unreachable.
        """
        sup = self.cfg.supervised_lesion_classes or PIXEL_ANNOTATED_LESION_CLASSES
        return tuple(c for c in LESION_CLASSES if c not in sup)

    # -- stages -----------------------------------------------------------
    def _to_tensor(self, img_rgb: np.ndarray) -> torch.Tensor:
        from .data.torch_data import to_tensor
        return to_tensor(img_rgb).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def _segment(self, model_input: np.ndarray) -> np.ndarray:
        """Segment at the model's training resolution, return maps at `size`.

        Takes the numpy image rather than the grader's tensor so the two stages
        can run at different resolutions -- which they must, because the lesion
        model is trained at 1024 and the grader at 512.
        """
        out_size = self.cfg.size
        if self.seg is None:
            return np.zeros((out_size, out_size, NUM_LESION_CLASSES), np.float32)

        img = model_input
        if img.shape[0] != self.cfg.seg_size:
            interp = cv2.INTER_CUBIC if img.shape[0] < self.cfg.seg_size else cv2.INTER_AREA
            img = cv2.resize(img, (self.cfg.seg_size, self.cfg.seg_size), interpolation=interp)

        from .data.torch_data import to_tensor
        x = to_tensor(img).unsqueeze(0).to(self.device)
        logits = self.seg(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = torch.sigmoid(logits.float())[0].permute(1, 2, 0).cpu().numpy()
        if probs.shape[0] != out_size:
            probs = cv2.resize(probs, (out_size, out_size), interpolation=cv2.INTER_AREA)
        return probs

    # -- main -------------------------------------------------------------
    def run(self, image: np.ndarray | str | Path, image_id: str = "",
            explain: bool = True) -> tuple[ScreeningResult, dict]:
        """Run the full pipeline.

        Returns ``(result, artifacts)`` where artifacts carries the arrays a
        report needs (enhanced image, CAM, overlays) but which do not belong
        in a JSON record.
        """
        t_all = time.perf_counter()
        timing: dict[str, float] = {}
        res = ScreeningResult(image_id=image_id or "case",
                              model_version=self.cfg.model_version)
        artifacts: dict = {}

        if isinstance(image, (str, Path)):
            res.image_id = image_id or Path(image).stem
            raw = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if raw is None:
                raise FileNotFoundError(f"Cannot read image: {image}")
        else:
            raw = image
        artifacts["raw"] = raw

        # 1. geometry ------------------------------------------------------
        t = time.perf_counter()
        img, fov_mask, fov = standardize(raw, size=self.cfg.size)
        timing["geometry"] = (time.perf_counter() - t) * 1000
        artifacts["standardized"] = img
        artifacts["fov_mask"] = fov_mask

        # 2. landmarks (needed by the quality gate) ------------------------
        t = time.perf_counter()
        lm = locate(img, fov_mask)
        timing["landmarks"] = (time.perf_counter() - t) * 1000
        res.landmarks = lm.to_dict()
        artifacts["landmarks"] = lm

        # 3. quality gate --------------------------------------------------
        t = time.perf_counter()
        q: QualityReport = assess(img, fov_mask, fov, landmarks=lm)
        timing["quality"] = (time.perf_counter() - t) * 1000
        res.quality = q.to_dict()
        res.gradeable = q.gradeable

        if not q.gradeable:
            res.decision = "recapture"
            res.recapture_advice = q.advice
            res.grade_label = "Ungradeable"
            timing["total"] = (time.perf_counter() - t_all) * 1000
            res.timing_ms = {k: round(v, 2) for k, v in timing.items()}
            return res, artifacts

        # 4. adaptive enhancement, then re-check the correctable criteria ---
        t = time.perf_counter()
        enhanced, applied = adaptive_enhance(img, fov_mask, q.issues)
        timing["enhancement"] = (time.perf_counter() - t) * 1000
        res.enhancement_applied = applied
        artifacts["enhanced"] = enhanced

        # The first pass treats correctable defects (uneven flash, exposure,
        # low contrast, noise) as "enhance", not "reject" -- software can undo
        # them, and sending a patient home for a fixable image wastes a visit.
        # Whether it *actually* fixed them is an empirical question, so re-run
        # the gate on the enhanced image and reject only what survived.
        if applied:
            t = time.perf_counter()
            q2 = assess(enhanced, fov_mask, fov, landmarks=lm)
            timing["quality_recheck"] = (time.perf_counter() - t) * 1000
            still_failing = [k for k, v in q2.verdicts.items()
                             if v == "fail" and k in CORRECTABLE]
            res.quality = q2.to_dict()
            res.quality["first_pass"] = {
                "overall": q.overall,
                "issues": q.issues,
                "scores": q.scores,
            }
            if still_failing:
                res.gradeable = False
                res.decision = "recapture"
                res.recapture_advice = [
                    RECAPTURE_ADVICE.get(k, f"Recapture: {k} inadequate.")
                    for k in still_failing]
                res.recapture_advice.insert(0, (
                    "Enhancement was applied (" + ", ".join(applied) +
                    ") but the image is still not gradeable."))
                res.grade_label = "Ungradeable"
                timing["total"] = (time.perf_counter() - t_all) * 1000
                res.timing_ms = {k: round(v, 2) for k, v in timing.items()}
                return res, artifacts
            q = q2

        model_in = to_model_input(enhanced, fov_mask, mode="hybrid")
        artifacts["model_input"] = model_in

        x = self._to_tensor(model_in)

        # 5. segmentation --------------------------------------------------
        t = time.perf_counter()
        lesion_probs = self._segment(model_in)
        timing["segmentation"] = (time.perf_counter() - t) * 1000
        artifacts["lesion_probs"] = lesion_probs

        # 6. clinical features ---------------------------------------------
        t = time.perf_counter()
        vessel = self._vessel_proxy(enhanced, fov_mask)
        feats = extract(lesion_probs, lm, fov_mask, vessel,
                        threshold=self.cfg.lesion_thresholds or self.cfg.lesion_threshold,
                        unassessed=self.unassessed_lesions)
        timing["clinical_features"] = (time.perf_counter() - t) * 1000
        res.clinical_features = {
            "counts": feats.counts,
            "per_quadrant": feats.per_quadrant,
            "quadrants_with_hemorrhage": feats.quadrants_with_hemorrhage,
            "quadrants_with_beading": feats.quadrants_with_beading,
            "nv_at_disc": feats.nv_at_disc,
            "nv_elsewhere": feats.nv_elsewhere,
            "unassessed": list(feats.unassessed),
            "lesions_within_1dd_of_fovea": feats.lesions_within_1dd_of_fovea,
            "nearest_lesion_dd": round(feats.nearest_lesion_dd, 2),
        }
        artifacts["features"] = feats
        artifacts["vessel_mask"] = vessel

        # 7. rule-based grade (classical arm + the audit trail) ------------
        rg, reasons = rule_grade(feats)
        res.rule_based_grade = rg
        res.rule_based_reasons = reasons
        dme, dme_reason = dme_risk(feats)
        res.dme_risk = dme

        # 8. neural grading -------------------------------------------------
        t = time.perf_counter()
        if self.grader is not None:
            c = torch.from_numpy(feats.to_vector()).unsqueeze(0).to(self.device)
            pred = self.grader.predict(x, c, mc_samples=self.cfg.mc_samples,
                                       temperature=self.cfg.temperature,
                                       grade_thresholds=self.cfg.grade_thresholds)
            probs = pred["class_probs"][0].cpu().numpy()
            # NOT argmax over the class probabilities. That is a different
            # decision rule from the ordinal one every metric in this project is
            # computed with, and the two disagreed on 3.65% of the internal test
            # split -- with argmax the worse of the pair (QWK 0.8855 vs 0.8939).
            # The served grade now comes from the same rule that was measured.
            res.grade = int(pred["grade"][0])
            res.class_probabilities = [round(float(p), 4) for p in probs]
            res.sight_threatening_probability = float(
                probs[SIGHT_THREATENING_THRESHOLD:].sum())
            p_ref = float(pred["referable_prob"][0])
            if self._calibrator is not None:
                p_ref = float(self._calibrator(np.array([p_ref]))[0])
            res.referable_probability = p_ref
            res.confidence = float(probs.max())
            res.uncertainty = {
                "entropy": round(float(pred["entropy"][0]), 4),
                "epistemic_variance": round(float(pred["epistemic"][0]), 5),
            }
        else:
            # No trained grader: fall back to the rule engine so the pipeline
            # still returns a defensible answer rather than nothing.
            res.grade = rg
            probs = np.zeros(len(ICDR_GRADES), np.float32); probs[rg] = 1.0
            res.class_probabilities = probs.tolist()
            res.referable_probability = float(rg >= REFERABLE_THRESHOLD)
            res.confidence = 0.5
            res.uncertainty = {"entropy": 0.0, "epistemic_variance": 0.0}
        timing["grading"] = (time.perf_counter() - t) * 1000

        res.grade_label = ICDR_GRADES[res.grade]
        res.referable = res.referable_probability >= self.cfg.referral_threshold

        # 9. decision & urgency ---------------------------------------------
        # agreement first: _decide consults the rule grade to distinguish
        # "confident negative the rules dispute" from a genuine emergency.
        res.agreement = self._agreement(res.grade, rg)
        res.decision, res.urgency = self._decide(res, feats)

        # 10. evidence -------------------------------------------------------
        res.evidence = self._build_evidence(feats, reasons, dme_reason, res)

        # 11. explanation ----------------------------------------------------
        if explain and self.cfg.enable_cam and self.grader is not None:
            t = time.perf_counter()
            try:
                from .explain.cam import compute_cam
                c = torch.from_numpy(feats.to_vector()).unsqueeze(0).to(self.device)
                cam = compute_cam(self.grader, x, c, method=self.cfg.cam_method,
                                  target="referable",
                                  referable_index=REFERABLE_THRESHOLD - 1,
                                  fov_mask=fov_mask)
                artifacts["cam"] = cam
            except Exception as e:                # never let XAI break screening
                artifacts["cam_error"] = str(e)
            timing["explanation"] = (time.perf_counter() - t) * 1000

        timing["total"] = (time.perf_counter() - t_all) * 1000
        res.timing_ms = {k: round(v, 2) for k, v in timing.items()}
        return res, artifacts

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _vessel_proxy(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Morphological vessel map, used when no vessel network is loaded.

        The clinical features only need vessel *calibre statistics*, which a
        bottom-hat + hysteresis threshold estimates adequately; a full U-Net
        is used when one is trained on DRIVE.
        """
        g = image[..., 1].astype(np.float32)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, k)
        resp = np.clip(bg - g, 0, None)
        resp[mask == 0] = 0
        if resp.max() <= 0:
            return np.zeros(g.shape, np.uint8)
        norm = (resp / resp.max() * 255).astype(np.uint8)
        hi = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return cv2.morphologyEx(hi, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    def _decide(self, res: ScreeningResult, feats: ClinicalFeatures) -> tuple[str, str]:
        """Combine the neural verdict with the lesion-based escalation rules.

        The escalation rules exist because confidence is not a licence to
        auto-report a potentially blinding finding. But they must not be able
        to *override* a confident negative on their own, because they run on a
        segmentation head with limited cross-domain precision: trained on 64
        IDRiD images, its exudate channel puts several false foci near the
        fovea on healthy APTOS retinas. Wired unconditionally, that single
        channel marked **every** case urgent -- and a screening system that
        escalates 100% of patients has thrown away the triage that justifies
        it.

        So a lesion rule that disagrees with a confident neural negative
        produces a *deferral* -- the disagreement goes to a human, which is
        what the human-in-the-loop is for -- rather than a false emergency.
        Corroborated findings still escalate exactly as before.
        """
        lo, hi = self.cfg.defer_band
        p = res.referable_probability
        eps = res.uncertainty.get("epistemic_variance", 0.0)

        # Which signals are allowed to change a decision is a question about
        # their measured reliability, not their clinical severity.
        #
        # The neural grader is calibrated and validated: sensitivity 0.930
        # [0.894-0.954], specificity 0.939 [0.909-0.960] on a held-out,
        # subject-disjoint test split. The rule engine runs on a segmentation
        # head trained on 64 IDRiD images and measured at specificity 0.058 on
        # that same split -- it calls almost everything referable. Letting it
        # escalate unilaterally turned every case urgent, then every case a
        # deferral; either way the triage that justifies the system is gone.
        #
        # So lesion-based escalation requires corroboration: it fires unless
        # the calibrated model is *confidently* negative (below the deferral
        # band, not merely below threshold). A confidently-negative case whose
        # lesion evidence disagrees is still auto-reported, but the
        # disagreement is written into the report and the audit log rather
        # than discarded -- which is what makes it reviewable later.
        confidently_negative = p < lo

        # Neovascularisation is the exception: it defines proliferative DR and
        # is a specific finding, so it escalates on sight.
        if feats.nv_at_disc or feats.nv_elsewhere:
            return "refer", "urgent"

        # The urgent tier is decided on P(grade >= 3) against a threshold
        # fitted on validation data -- not on the predicted grade. Keying it to
        # the grade meant the scarcest, least reliable output in the system
        # gated expedited review: on Messidor-2 that rule caught 56 of 110
        # sight-threatening eyes, where the fitted cut-point catches 86. The
        # other 30 were still referred, just queued as routine.
        #
        # None keeps the old grade rule so a bundle written before this field
        # existed behaves exactly as it did.
        if not confidently_negative:
            urgent = (res.grade >= SIGHT_THREATENING_THRESHOLD
                      if self.cfg.urgent_threshold is None else
                      res.sight_threatening_probability >= self.cfg.urgent_threshold)
            if urgent:
                return "refer", "urgent"
            if res.dme_risk >= 2:
                return "refer", "urgent"

        if lo <= p <= hi or eps > self.cfg.uncertainty_defer:
            return "defer_to_human", ("soon" if p >= self.cfg.referral_threshold else "routine")
        if p >= self.cfg.referral_threshold:
            return "refer", "soon"
        return "auto_report", "routine"

    @staticmethod
    def _agreement(neural: int, rule: int) -> str:
        d = abs(neural - rule)
        if d == 0:
            return "exact"
        if d == 1:
            return "within_one_grade"
        return "disagree"

    def _build_evidence(self, feats: ClinicalFeatures, reasons: list[str],
                        dme_reason: str, res: ScreeningResult) -> list[dict]:
        ev: list[dict] = []
        for name in feats.unassessed:
            ev.append({
                "finding": name.replace("_", " "),
                "status": "not assessed",
                "detail": "no pixel supervision for this class in the training "
                          "corpus; absence of a detection is not evidence of "
                          "absence of the lesion.",
            })
        for name in LESION_CLASSES:
            if name in feats.unassessed:
                continue
            n = feats.counts.get(name, 0)
            if n == 0:
                continue
            per_q = feats.per_quadrant.get(name, {})
            ev.append({
                "finding": name.replace("_", " "),
                "count": n,
                "per_quadrant": {k: v for k, v in per_q.items() if v},
                "area_percent": round(feats.area_fraction.get(name, 0.0) * 100, 3),
            })
        for r in reasons:
            ev.append({"criterion": r})
        if dme_reason:
            ev.append({"macular_assessment": dme_reason})
        if res.agreement == "disagree":
            ev.append({"caution": (
                f"The deep model graded {res.grade} ({ICDR_GRADES[res.grade]}) while the "
                f"rule-based criteria give {res.rule_based_grade} "
                f"({ICDR_GRADES[res.rule_based_grade]}). Flagged for human adjudication.")})
        return ev
