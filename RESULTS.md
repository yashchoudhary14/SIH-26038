# Results and Conclusions

Problem statement: MATLAB-based retinal image analysis pipeline for automated
diabetic retinopathy screening (SIH-26038). Implemented in Python with a
generated Simulink/SimEvents bridge — see [§8](#8-simulink).

Every number below is read from `outputs/validation/validation.json`, produced
by `scripts/validate.py`. Nothing is hand-entered. Reproduce with the commands
in [§9](#9-reproducing-this).

---

## 1. The headline

| | sensitivity | specificity | AUC | QWK | targets |
|---|---|---|---|---|---|
| **Internal test** (n=631) | **0.930** | **0.939** | 0.986 | 0.888 | ✅ both met |
| **External, Messidor-2** (n=1,744) | 0.427 | 0.978 | 0.875 | 0.586 | ❌ not met |

Targets: sensitivity ≥ 90%, specificity ≥ 85% for referable DR (ICDR grade ≥ 2).

**The aggregate external sensitivity is misleading on its own.** Split by true
severity, the misses are almost entirely *moderate* NPDR, not blinding disease:

| true severity | n | flagged referable |
|---|---|---|
| 0 — no DR | 1,017 | 2.1% *(correctly ignored)* |
| 1 — mild NPDR | 270 | 2.6% *(correctly ignored)* |
| **2 — moderate NPDR** | 347 | **27.7%** ← the failure |
| **3 — severe NPDR** | 75 | **93.3%** |
| **4 — proliferative DR** | 35 | **82.9%** |

> **Sight-threatening disease (grade ≥ 3): sensitivity 0.900 [0.830–0.943] at
> 97.8% specificity** — zero-shot, on a French cohort with different cameras
> and a different grading panel, with nothing fitted on it.

Grade 2 accounts for 76% of referable cases, which is what drags the aggregate
to 0.427. The two failure modes do not carry equal clinical weight: a missed
proliferative DR can cost sight within months; a missed moderate NPDR is
picked up at the next annual screen.

---

## 2. Verdict — can this system detect DR?

**As a sight-saving triage tool: yes.** It detects 90% of blinding disease on
cameras it has never seen while correctly ignoring 98% of healthy eyes, so it
does not flood the ophthalmologist. That is the decision the problem statement
exists to serve.

**As a full ICDR grader on unseen hardware: not yet.** Moderate-NPDR detection
at 27.7% is not acceptable, and on that distribution no threshold satisfies
both targets simultaneously (at 90% overall sensitivity, specificity is 0.628).

**On populations resembling its training data: yes, comfortably.** Both
targets met with margin on a subject-disjoint held-out split.

---

## 3. Why moderate NPDR fails — the reference standards disagree

This was traced to root cause, and it is **not** a model or image problem.

Median total lesion burden (microaneurysms + haemorrhages + exudates) detected
by the *same* segmentation model, grouped by the label each cohort assigns:

| true grade | APTOS / IDRiD | Messidor-2 |
|---|---|---|
| 0 — no DR | 24 | 22 |
| 1 — mild NPDR | 47 | 20 |
| **2 — moderate NPDR** | **146** | **46** |
| 3 — severe NPDR | 198 | 162 |
| 4 — proliferative | 164 | 133 |

**Grade 0 is identical across cohorts (24 vs 22).** That single number rules
out the obvious suspect: the segmentation has *not* stopped working on
Messidor-2 images. Grades 3 and 4 are comparable too. Only grade 2 diverges —
by more than 3×.

And the decisive comparison: a Messidor-2 **"moderate"** NPDR carries 46
lesions, *fewer* than an APTOS **"mild"** NPDR at 47. By actual disease
content, Messidor-2's grade 2 resembles APTOS's grade 1.

**Cause.** APTOS ships one grader per image with documented label noise; its
graders reserved grade 2 for visibly heavy disease. Messidor-2 ships
three-specialist adjudicated consensus applying ICDR strictly, where moderate
NPDR is *"more than microaneurysms alone but less than severe"* — satisfiable
by a single haemorrhage. The model faithfully learned APTOS's *de facto*
definition and is tested against ICDR's *de jure* one. Those are different
populations wearing the same label.

**Why it is fixable.** The ranking survives. On Messidor-2 the median
P(referable) is 0.000 for grade 1 and 0.250 for grade 2 — correctly ordered,
sitting below a threshold learned from a stricter-looking population:

| threshold | sens grade 2 | sens grade ≥3 | specificity | % flagged |
|---|---|---|---|---|
| 0.657 *(frozen)* | 19.6% | 86.4% | 98.8% | 10.3% |
| 0.400 | 27.7% | 90.0% | 97.7% | 12.8% |
| **0.250** | 50.7% | **95.5%** | **92.3%** | 21.8% |
| 0.150 | 66.0% | 97.3% | 83.2% | 31.7% |
| 0.050 | 85.3% | 99.1% | 64.0% | 49.8% |

At 0.250 the system reaches **95.5% sensitivity on sight-threatening disease
at 92.3% specificity**, flagging only 22% of patients — a deployable operating
point reached by moving one number.

**Implication:** retraining will not fix this. More epochs, bigger backbones
and heavier augmentation do not address a label-definition mismatch. The fixes
are (a) per-site threshold calibration against a few hundred locally-graded
images, which the `/audit` endpoint is built to collect, or (b) multi-source
training with harmonised reference standards.

---

## 4. Full metrics

### Referable DR (grade ≥ 2)

| metric | internal test (n=631) | external / Messidor-2 (n=1,744) |
|---|---|---|
| Sensitivity | 0.9296 [0.8937–0.9540] | 0.4267 [0.3821–0.4725] |
| Specificity | 0.9395 [0.9093–0.9601] | 0.9782 [0.9687–0.9849] |
| PPV | 0.9263 | 0.8744 |
| NPV | 0.9422 | 0.8277 |
| AUC | 0.9859 [0.9763–0.9916] | 0.8751 [0.8548–0.8930] |
| QWK | 0.8878 [0.8646–0.9082] | 0.5859 |
| Within-one-grade | 0.9556 | 0.9106 |
| ECE | 0.0276 | 0.0960 |

Intervals are Wilson score for proportions and DeLong for AUC.

### Lesion segmentation (IDRiD, 64 training images)

| class | Dice @512 | Dice @1024 | published IDRiD range |
|---|---|---|---|
| Microaneurysm | 0.000 | **0.485** | 0.30–0.50 |
| Haemorrhage | 0.248 | **0.521** | 0.50–0.65 |
| Hard exudate | 0.272 | **0.575** | 0.70–0.80 |
| Cotton-wool spot | 0.187 | **0.623** | 0.55–0.70 |
| Neovascularisation | — | — | **not annotated in IDRiD** |

Resolution is decisive: at 512 a microaneurysm survives downsampling from
4288px as ~4 pixels and Dice is **0.000**. The lesion model runs at 1024; the
grader runs at 512.

IDRiD does not annotate neovascularisation at all, so that channel cannot be
learned. Proliferative DR is still graded from image features, but the
*explanation* cannot cite NV as evidence.

### Landmark localisation

Optic disc median error **0.015 DD**, fovea **0.077 DD**, 97% of foveae within
1 disc diameter (`scripts/eval_landmarks.py`). Closed-form, no training data,
~120 ms on CPU.

### Calibration

| | val ECE | external AUC | spec @ 90% sens |
|---|---|---|---|
| Temperature only (T = 3.797) | 0.0519 | 0.8751 | 0.628 |
| Isotonic, no tie-break | 0.0000 *(in-sample, meaningless)* | 0.8626 | **0.000** |
| **Isotonic + tie-break** *(deployed)* | **0.0064** | **0.8751** | **0.628** |

Adoption is decided **out-of-fold** (0.0130 vs 0.0519), because isotonic scored
on its own fitting split drives ECE to ~0 by construction.

### Explanation quality

Faithfulness **+0.127** (insertion 0.974 − deletion 0.848), attention sparsity
(Gini) 0.772 over 40 referable images. Pointing-game and lesion-IoU require
pixel annotations, which the APTOS test split does not have.

---

## 5. Ablation — does the integrated pipeline beat any single technique?

| arm | AUC | 95% CI | sens | spec | QWK |
|---|---|---|---|---|---|
| **fusion** *(deployed)* | **0.9859** | 0.9763–0.9916 | 0.930 | 0.939 | **0.8878** |
| cnn_only | 0.9857 | 0.9752–0.9918 | 0.972 | 0.905 | 0.8768 |
| clinical_only | 0.9310 | 0.9096–0.9476 | 0.912 | 0.807 | 0.7087 |
| rule_based | 0.9139 | 0.8890–0.9337 | 0.996 | 0.058 | 0.0305 |

Paired DeLong tests against fusion:

- vs `clinical_only`: fusion higher by 0.0549, **p = 3.0×10⁻¹⁰** ✅
- vs `rule_based`: fusion higher by 0.0720, **p = 1.3×10⁻¹²** ✅
- vs `cnn_only`: difference 0.0002, **p = 0.92 — not significant**

**Honest conclusion:** the integrated pipeline decisively beats both classical
arms, and is *statistically tied* with the CNN-only arm on referable-DR
discrimination. On this data the clinical-feature branch buys interpretability
and a better QWK (0.888 vs 0.877), not better referral discrimination. The
verdict string in `outputs/validation/ablation.txt` is generated, not written,
and reports this as a failure to beat every single technique.

---

## 6. Nine bugs that only real data exposed

Listed because most are *invisible* failures — they produce a plausible number
rather than a crash.

| # | bug | how it presented |
|---|---|---|
| 1 | IDRiD encodes mask foreground as **76**; loader thresholded at >127 | every mask loaded empty |
| 2 | Dice scored empty-prediction-vs-empty-target as **1.0** | hid #1 as "mean Dice 1.0000" while loss sat flat at 0.95 |
| 3 | Masks skipped the image's crop/pad/resize geometry | annotations offset from the pixels they describe |
| 4 | Segmentation at 512px | microaneurysm Dice 0.000 |
| 5 | Deployed pipeline segmented at 512 while features were trained at 1024 | deployment disagreed with its own validation |
| 6 | Calibrator not shipped with the model | threshold applied to a different probability scale |
| 7 | `lesion_threshold = 0.5` never fitted | true F1 optimum 0.85–0.95 |
| 8 | FOV clipping penalty | rejected **34%** of real images whose coverage was 0.90–1.00 |
| 9 | Isotonic pinned the score floor to exactly 0.0 | on Messidor-2 sent 10.3% of true positives to zero; spec @ 90% sens fell 0.628 → **0.000** |

Three deserve emphasis:

- **#2 is the dangerous shape** — a metric reporting a perfect score for a
  model that had learned nothing, from data that contained nothing.
- **#8 was structurally invisible to synthetic data.** Phantoms always render a
  black margin, but a real fundus aperture is wider than the sensor is tall, so
  the retina touches the frame on almost every correct capture; APTOS ships
  pre-cropped touching all four edges.
- **#9 only external validation could find.** Isotonic measurably improved
  in-distribution calibration while silently destroying the operating range
  under distribution shift, because its ties are harmless until the score
  distribution moves.

All three now have regression tests. Suite: **32 tests**, `pytest tests/ -q`.

### Effect on deployed behaviour (120 real test images)

| | before fixes | after |
|---|---|---|
| Recapture rate | 34% | **0%** |
| Cases flagged urgent | 100% | 23% |
| Exact grade match | 59.5% | **68.3%** |

The 100%-urgent figure was the rule engine — measured specificity **0.058** —
unilaterally overriding a calibrated model with specificity 0.939. Lesion-based
escalation now requires corroboration and cannot override a confidently
negative neural verdict, though the disagreement still reaches the report and
the audit log. Neovascularisation remains an unconditional escalation.

---

## 7. Data

| dataset | role | n |
|---|---|---|
| APTOS 2019 | train / val / test | 3,662 |
| IDRiD disease grading | train / val / test | 516 |
| IDRiD segmentation | lesion masks | 81 (64 train / 17 val) |
| DRIVE | vessel masks | 40 |
| **Messidor-2** | **held-out external, never trained on** | 1,748 |

Cohort: 6,047 cases — train 2,925 / val 622 / test 631 / external 1,748, plus
the segmentation and vessel splits. **Measured subject overlap between the
grading splits: zero.** `registry.assert_no_leakage` raises `SplitViolation`
if Messidor-2 reaches the training pool.

Split discipline: `val` fits the temperature and selects the referral
threshold; `test` is the internal estimate; `external` has nothing fitted on
it. Four Messidor-2 images its adjudicators marked ungradable are excluded from
metrics rather than scored as a sixth class.

---

## 8. Simulink

The problem statement names Simulink. The executable telemedicine model is
SimPy (`src/drscreen/sim/telemedicine.py`); `scripts/run_simulation.py
--export-matlab matlab/` generates the SimEvents realisation from the same
`SimConfig`, so the two cannot drift.

| file | status |
|---|---|
| `dr_screening_params.m` | ✅ **verified under GNU Octave** — lognormal mean/CV → μ/σ round-trips exactly; Markov stationary uptime reproduces declared availability |
| `build_dr_screening_model.m` | ⚠️ **never executed** — needs a MATLAB + SimEvents licence |
| `validate_against_simpy.m` | ⚠️ never executed |

SimEvents is proprietary with no Octave equivalent, so nothing touching
`add_block`/`set_param` can be tested without a licence. Because SimEvents
dialog parameter names drift between releases and `set_param` is atomic over
its name/value pairs, the build script applies every property individually and
prints a fix-list rather than throwing. Expect a handful of names to need
correcting on first run.

**Capacity findings** (100,000 patients/year, 12 PHCs, 2 ophthalmologist FTE):

| scenario | reviewer load | routine SLA | p90 turnaround | feasible |
|---|---|---|---|---|
| Manual review, no AI | **140.5%** | 66.6% | 17.9 d | **no** |
| AI-assisted | 11.0% | 100% | 0.01 d | yes |

Without AI triage the review queue is *unstable* — demand is 141% of available
reading capacity, so the backlog grows without bound. Constrained search over
1,024 configurations returns 8 PHCs, 1 ophthalmologist FTE, edge inference,
**₹53.9 lakh/year ≈ ₹54 per patient screened**. The optimiser *chose* AI-assisted
review and edge inference; both are outputs, not assumptions.

---

## 9. Reproducing this

```bash
python scripts/extract_datasets.py --src data --out data/raw

python scripts/build_cohort.py --source real --data-root data/raw --out data/cohort_real --size 512 --workers 14
python scripts/build_cohort.py --source real --data-root data/raw --out data/cohort_seg1024 --size 1024 --workers 12 --only-splits seg_train seg_val

python scripts/train_seg.py --cohort data/cohort_seg1024 --epochs 160 --batch-size 3 --size 1024 --pos-weight 12
python scripts/precompute_features.py --cohort data/cohort_real --seg outputs/segmentation/best.pt --size 1024 --feature-size 512
python scripts/train_grader.py --cohort data/cohort_real --arm fusion --epochs 30

python scripts/validate.py --cohort data/cohort_real --seg-cohort data/cohort_seg1024 \
    --arms cnn_only=outputs/grader_cnn/best.pt clinical_only=outputs/grader_clinical/best.pt
```

Artefacts: `outputs/validation/` (metrics, ablation, model card),
`outputs/reports/` (rendered clinical reports), `outputs/artifacts/`
(deployable bundle). Hardware: RTX 5080, ~1.5 h end to end.

---

## 10. What would move the numbers

In descending order of expected value:

1. **Per-site threshold calibration.** A few hundred locally-graded images per
   deployment site. The ranking already transfers (external AUC 0.875); only
   the operating point does not. Cheapest fix by a wide margin.
2. **Multi-source training** across APTOS + Messidor-2 + EyePACS with
   harmonised grades, to learn a reference standard rather than one panel's
   habits.
3. **More lesion annotation.** 64 training images is the binding constraint on
   segmentation, and no public set annotates neovascularisation.
4. **Higher grading resolution.** The grader runs at 512; the segmentation
   result suggests 768–1024 would help early disease.
5. **Test-time augmentation and ensembling** — reliable but small gains, and
   they cost latency the edge deployment cannot spare.

---

## 11. Standing limitations

- Trained on Indian cohorts (APTOS, IDRiD); external validation is French
  (Messidor-2). No African, East Asian or Latin American validation.
- Co-pathology is **not** detected. Glaucoma, AMD and retinal vein occlusion
  can co-occur; "no DR" is not a statement that the eye is healthy.
- Neovascularisation has no pixel supervision in any public dataset used here.
- The venous-beading cue runs on a morphological vessel proxy and is
  deliberately conservative, so the 4-2-1 rule's "2" arm under-fires.
- Cost figures in `DEFAULT_COSTS` are order-of-magnitude inputs, not findings.

**Decision-support output. Not a diagnosis.** Every referable and every
sight-threatening finding is reviewed by a qualified ophthalmologist before
clinical action.
