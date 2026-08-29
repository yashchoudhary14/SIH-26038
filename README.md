# Explainable Diabetic Retinopathy Screening for Rural India

An end-to-end, clinically-validated DR screening pipeline: image quality
gating with recapture feedback, retinal structure and lesion segmentation,
ordinal ICDR severity grading with calibrated confidence, lesion-grounded
explainability, and a discrete-event model of a district telemedicine
programme.

**Trained and validated on real patient data** — APTOS 2019 + IDRiD, with
Messidor-2 held out for blind external validation.

| | sensitivity | specificity | AUC | targets |
|---|---|---|---|---|
| Internal test (n=631) | **0.930** | **0.939** | 0.986 | ✅ both met |
| **External, Messidor-2 (n=1,744)** | **0.427** | 0.978 | 0.875 | ❌ **not met** |

**Read the second row before the first** — and then read the breakdown below
it, because the aggregate sensitivity is misleading on its own.

Externally the misses are almost entirely **moderate NPDR**, not blinding
disease:

| true severity | n | flagged referable |
|---|---|---|
| 0 — no DR | 1,017 | 2.1% *(correctly not flagged)* |
| 1 — mild NPDR | 270 | 2.6% *(correctly not flagged)* |
| 2 — moderate NPDR | 347 | **27.7%** ← the failure |
| 3 — severe NPDR | 75 | **93.3%** |
| 4 — proliferative DR | 35 | **82.9%** |

**Sight-threatening disease (grade ≥ 3): sensitivity 0.900 [0.830–0.943] at
97.8% specificity** — on a cohort from a different country, different cameras
and a different grading panel, with nothing fitted on it.

Grade 2 is 76% of the referable cases, which is why it drags the aggregate to
0.427. The clinical weight of those two failure modes is not equal: a missed
proliferative DR can cost sight within months, a missed moderate NPDR is
caught at the next annual screen.

It also **runs on a fresh clone with no downloads**: a procedural fundus
phantom generator exercises every stage for real, so the system is
demonstrable in minutes before any dataset arrives.

> **[→ RESULTS.md](RESULTS.md)** — full results, the verdict on whether the
> system works, root-cause analysis of the moderate-NPDR gap, and the nine
> bugs real data exposed. Start there if you want the findings rather than the
> build instructions.

---

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate       # Linux/macOS: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt && pip install -e .
```

> The `cu128` index is required for RTX 50-series (Blackwell) GPUs. On older
> CUDA or CPU-only machines, install PyTorch from the default index instead.

### On real data (what the reported numbers come from)

Place the corpora under `data/` and unpack them (see `docs/DATASETS.md`):

```bash
python scripts/extract_datasets.py --src data --out data/raw
```

```bash
# grading cohort at 512; lesion cohort at 1024, where microaneurysms survive
python scripts/build_cohort.py --source real --data-root data/raw --out data/cohort_real --size 512 --workers 14
python scripts/build_cohort.py --source real --data-root data/raw --out data/cohort_seg1024 --size 1024 --workers 12 --only-splits seg_train seg_val

python scripts/train_seg.py --cohort data/cohort_seg1024 --epochs 160 --batch-size 3 --size 1024 --pos-weight 12
python scripts/precompute_features.py --cohort data/cohort_real --seg outputs/segmentation/best.pt --size 1024 --feature-size 512
python scripts/train_grader.py --cohort data/cohort_real --arm fusion --epochs 30

python scripts/validate.py --cohort data/cohort_real --seg-cohort data/cohort_seg1024     --arms cnn_only=outputs/grader_cnn/best.pt clinical_only=outputs/grader_clinical/best.pt
```

### Without any downloads (synthetic phantoms)

```bash
python scripts/build_cohort.py --source synthetic --n 6000 --out data/cohort_synth --workers 16
python scripts/train_seg.py --cohort data/cohort_synth --epochs 14
python scripts/precompute_features.py --cohort data/cohort_synth --seg outputs/segmentation/best.pt
python scripts/train_grader.py --cohort data/cohort_synth --arm fusion
python scripts/validate.py --cohort data/cohort_synth
```

Then the demo and the console:

```bash
python scripts/run_demo.py --demo
python -m uvicorn drscreen.api:app --port 8000     # open http://localhost:8000
```

Simulation and the MATLAB bridge:

```bash
python scripts/run_simulation.py --scenarios
python scripts/run_simulation.py --optimise
python scripts/run_simulation.py --export-matlab matlab/
```

Tests:

```bash
python -m pytest tests/ -q
```

---

## What the pipeline does

```
raw fundus image
      |
  [1] geometry ........... circular FOV detection, tight crop, square pad, resize
      |                    (aspect ratio preserved -- disc diameter is the clinical unit)
  [2] quality gate ....... 9 interpretable criteria -> pass / borderline / ungradeable
      |                    ungradeable STOPS here and returns recapture instructions
  [3] enhancement ........ only the corrections the gate asked for
      |
  [4] landmarks .......... optic disc + fovea -> the clinical coordinate frame
      |
  [5] segmentation ....... vessels + 5 lesion classes (attention U-Net)
      |
  [6] clinical features .. counts per quadrant, NV location, exudate distance from fovea
      |
      +----> [7a] rule engine ...... ICDR criteria applied directly (4-2-1 rule etc.)
      |
  [7b] ordinal grader .... CNN fused with clinical features, CORN head, temperature-calibrated
      |
  [8] decision ........... auto-report / refer / defer-to-human, against a frozen threshold
      |
  [9] explanation ........ Grad-CAM++ over the referable log-odds + lesion evidence in words
      |
  annotated report (HTML + PNG panel + JSON)
```

Every stage records its latency; the whole result is JSON-serialisable and
auditable.

---

## Design decisions that differ from the standard recipe

Each of these was a deliberate choice, and the reasoning is in the module
docstring next to the code.

### 1. Ordinal (CORN) head, not a 5-way softmax

DR grades are ordered and the screening decision is *cumulative*: "is this
grade ≥ 2?". A softmax has to reconstruct that by summing probabilities of
classes it modelled as unordered, with nothing constraining the result to be
monotone. CORN predicts P(y > k | y > k−1) directly, so
P(y > k) = ∏ σ(z_j) is monotone **by construction** and the referable
probability is a quantity the model actually optimised.

*Verified in `tests/test_pipeline.py::test_corn_cumulative_is_monotone`.*

### 2. Interpretable quality gate, not a learned blur classifier

The PS requires *recapture feedback*. A CNN logit cannot tell a technician
"the macula is out of frame"; nine physics-based criteria can, and they run in
milliseconds on CPU before any heavy model. Real field failures — media
opacity, miosis, lens flare, uncleaned optics — do not look like the Gaussian
blur a synthetic quality classifier is trained on.

### 3. Analytic landmark localisation

Two ICDR criteria are geometric: severe NPDR is defined by lesion counts *per
quadrant*, and CSME by distance from the fovea *in disc diameters*. Both need
a coordinate frame, and one that fails silently is worse than none. The
closed-form detector needs no training data, returns a confidence, and runs in
~120 ms.

Measured on phantoms (`scripts/eval_landmarks.py`): optic disc median error
**0.015 DD**, fovea **0.077 DD**, 97% of foveae within 1 DD.

### 4. Fusion of CNN features with explicit clinical features

The lesion-derived vector (counts per quadrant, NV location, exudate distance
from fovea) enters the classifier head alongside the pooled CNN embedding,
behind a learned gate that down-weights it when segmentation is unreliable.
This is the "integrated pipeline" the PS asks for, and it makes the evidence
auditable: the features driving the prediction are the ones printed in the
report.

### 5. Calibration is a first-class requirement

A modern CNN says 0.97 on a batch it gets right 82% of the time. In a
human-in-the-loop programme that is dangerous, because the review queue is
prioritised by confidence and an over-confident false negative is never looked
at again. Temperature scaling is fitted on validation only, and ECE / MCE /
Brier / reliability curves are reported before and after.

**A subtlety we found and did not paper over:** temperature scaling is exactly
rank-preserving for a *single* logit, but not for CORN, because the referable
score is a *product* of sigmoids and T does not factor out of a product of
logistic functions. About 4% of pairs change order at T = 2.5 — almost all
near-ties, worth ~5×10⁻⁴ of AUC. For the binary referral decision we therefore
recalibrate with isotonic regression, which *is* exactly non-inverting. See
`src/drscreen/models/calibration.py` and
`tests/test_pipeline.py::test_temperature_effect_on_corn_ranking_is_negligible`.

### 6. Explanations are measured, not just rendered

Producing a heatmap is trivial; evidence that it means anything is the work.
`src/drscreen/explain/faithfulness.py` computes:

- **Deletion / insertion AUC** — is the map faithful to the *model*? (no
  ground truth needed, runs on any dataset)
- **Pointing game, lesion hit-rate, CAM–lesion IoU** — is it faithful to the
  *pathology*? (needs IDRiD pixel annotations)
- **Gini sparsity** — a map that highlights everything explains nothing.

High faithfulness with a low pointing-game score is the signature of
shortcut learning, and it is exactly what makes black-box DR models collapse
on external data.

### 7. Messidor-2 is held out in code, not by convention

`src/drscreen/data/registry.py` raises `SplitViolation` if Messidor-2 reaches
the training pool, and splits are subject-grouped by hash so fellow eyes never
straddle the train/val boundary. Selecting a threshold on the data you report
it on is the most common way DR results get inflated, so the split roles are
positional in the flow rather than something to remember.

### 8. The abstention band

Confidence is not a licence to auto-report a potentially blinding finding.
Anything sight-threatening (grade ≥ 3), any neovascularisation, and any
suspected CSME goes to a human **regardless** of model confidence. Cases in
the uncertainty band are deferred. The risk-coverage curve
(`selective_risk_curve`) converts a chosen error rate into the specialist
hours it costs — which is the number the capacity model consumes.

---
## Measured results (real data)

Trained on **APTOS 2019 + IDRiD**, with Messidor-2 held out. Produced by
`scripts/run_all.py` against `data/cohort_real`.

| split | n | source |
|---|---|---|
| train | 2,925 | APTOS + IDRiD grading |
| val (calibration + threshold) | 622 | APTOS + IDRiD grading |
| test (reported below) | 631 | APTOS + IDRiD grading |
| external | 1,748 | Messidor-2 — **held out, and unlabelled: see below** |
| lesion segmentation | 64 / 17 | IDRiD pixel annotations |
| vessels | 32 / 8 | DRIVE |

Splits are subject-grouped; measured overlap between train/val/test is **zero**.

### Referable DR (grade ≥ 2), internal held-out test — n = 631

| metric | value | 95% CI | target |
|---|---|---|---|
| **Sensitivity** | **0.930** | 0.894–0.954 | ≥ 0.90 ✅ |
| **Specificity** | **0.939** | 0.909–0.960 | ≥ 0.85 ✅ |
| AUC | 0.9850 | 0.9744–0.9912 | — |
| QWK | 0.8878 | 0.8646–0.9082 | — |
| ECE | 0.0285 | — | — |

**Both problem-statement targets are met on real patient data.**

Calibration: temperature (T = 3.80) is fitted to the multiclass CORN NLL and
actually made binary ECE slightly *worse* (0.0503 → 0.0519) while improving
MCE and Brier. Isotonic regression on P(referable) — the number the referral
decision uses — gives ECE **0.0172 out-of-fold** vs 0.0519, and is adopted.
The adoption test is out-of-fold by necessity: isotonic scored on its own
fitting split drives ECE to ~0 by construction.

### Lesion segmentation (IDRiD, 64 training images)

| class | Dice @512 | Dice @1024 | published IDRiD range |
|---|---|---|---|
| Microaneurysm | 0.000 | **0.485** | 0.30–0.50 |
| Haemorrhage | 0.248 | **0.521** | 0.50–0.65 |
| Hard exudate | 0.272 | **0.575** | 0.70–0.80 |
| Cotton-wool spot | 0.187 | **0.623** | 0.55–0.70 |
| Neovascularisation | — | — | **not annotated in IDRiD** |

Resolution is decisive: at 512 a microaneurysm survives downsampling from
4288px as ~4 pixels and Dice is **0.000**. The lesion model therefore runs at
1024 while the grader runs at 512.

### Ablation

```
arm                  AUC            95% CI    Sens    Spec     QWK  targets
cnn_only          0.9853 [0.9752,0.9913]   0.972   0.905  0.8768     PASS
fusion *          0.9850 [0.9744,0.9912]   0.930   0.939  0.8878     PASS
clinical_only     0.9293 [0.9076,0.9463]   0.923   0.795  0.7087     fail
rule_based        0.9139 [0.8890,0.9337]   0.996   0.058  0.0305     fail

The integrated pipeline (fusion) does NOT beat every single technique:
cnn_only scored higher. This must be reported as-is.
```

Fusion beats the two classical arms decisively (DeLong p = 4×10⁻¹⁰ and
4×10⁻¹²) and is **statistically tied with the CNN-only arm** (Δ AUC 0.0003,
p = 0.899). On this data the clinical-feature branch buys interpretability and
a better QWK, not better referral discrimination. That is the honest finding.

### External validation: Messidor-2, zero-shot — n = 1,744

Nothing was fitted on this split. Grades are the adjudicated reference
standard (Krause et al. 2018); 4 images their adjudicators marked ungradable
are excluded rather than scored as a sixth class.

| metric | internal test | **external** |
|---|---|---|
| Sensitivity | 0.930 [0.894–0.954] | **0.427** [0.382–0.472] |
| Specificity | 0.939 [0.909–0.960] | 0.978 [0.969–0.985] |
| AUC | 0.9859 [0.9763–0.9916] | 0.8751 [0.8548–0.8930] |
| QWK | 0.8878 | 0.5859 |
| ECE | 0.0276 | 0.0960 |

**What is actually failing.** Two things, and they need separating:

1. **Real discrimination loss.** AUC 0.986 → 0.875. Even at the
   sensitivity-optimal threshold *chosen on Messidor-2 itself* — an oracle, not
   an achievable result — specificity at 90% sensitivity is only **0.628**. No
   threshold on this distribution satisfies both targets. So this is not merely
   a mis-set operating point.

2. **Threshold transfer.** At the frozen threshold the model is far too
   conservative here: specificity *rose* to 0.978 while sensitivity collapsed.
   Median P(referable) among true positives is 0.889 internally and 0.370 on
   Messidor-2 — the score distribution shifts down bodily.

The referable prevalence also differs (45% internal vs 26% external), which
changes PPV but not sensitivity.

### Why moderate NPDR specifically — the reference standards disagree

Median total lesion burden (MA + haemorrhage + exudate) detected by the *same*
segmentation model, by the label each cohort assigns:

| true grade | APTOS / IDRiD | Messidor-2 |
|---|---|---|
| 0 — no DR | 24 | 22 |
| 1 — mild NPDR | 47 | 20 |
| **2 — moderate NPDR** | **146** | **46** |
| 3 — severe NPDR | 198 | 162 |
| 4 — proliferative | 164 | 133 |

Grade 0 is identical across cohorts (24 vs 22), which rules out the obvious
suspect: the segmentation has **not** stopped working on Messidor-2 images.
Grades 3 and 4 are comparable too. Only grade 2 diverges, by more than 3x —
and a Messidor-2 "moderate NPDR" (46 lesions) carries *less* disease than an
APTOS "mild NPDR" (47).

So the model is not misreading the images. It is applying APTOS's *de facto*
definition of moderate NPDR, which reserves grade 2 for visibly heavy disease,
to a cohort graded by three retinal specialists applying ICDR *de jure*, where
moderate NPDR can be a single haemorrhage. APTOS ships one grader per image
with documented label noise; Messidor-2 ships adjudicated consensus. Those are
different populations wearing the same label.

The ranking survives — on Messidor-2 the median P(referable) is 0.000 for
grade 1 and 0.250 for grade 2, correctly ordered, just sitting below a
threshold learned from a stricter-looking population. Which means the fix is
recalibration, not retraining:

| threshold | sens (all referable) | sens (grade 2) | sens (grade ≥3) | specificity | % flagged |
|---|---|---|---|---|---|
| 0.657 *(frozen)* | 35.7% | 19.6% | 86.4% | 98.8% | 10.3% |
| 0.400 | 42.7% | 27.7% | 90.0% | 97.7% | 12.8% |
| **0.250** | 61.5% | 50.7% | **95.5%** | **92.3%** | 21.8% |
| 0.150 | 73.5% | 66.0% | 97.3% | 83.2% | 31.7% |
| 0.050 | 88.6% | 85.3% | 99.1% | 64.0% | 49.8% |

At 0.250 the system catches **95.5% of sight-threatening disease at 92.3%
specificity** while flagging only 22% of the population for review — a
deployable operating point, reached by moving one number. This is the concrete
argument for fitting the threshold per site against a few hundred local
labels, and for the audit log that collects them.

**This is the honest state of the system**: usable as a triage aid on
populations resembling its training data, not yet deployable on unseen
cameras. Closing it needs domain adaptation, multi-source training, or
site-specific threshold re-fitting with local labels — and the audit log
(`/audit`) exists precisely to detect this in the field before it harms anyone.

## What real data broke that phantoms never could

Eight bugs surfaced only once real corpora were loaded. They are listed
because most are invisible failures — the kind that produce a plausible number
rather than a crash.

| # | bug | how it presented |
|---|---|---|
| 1 | IDRiD encodes mask foreground as **76**, loader thresholded at >127 | every mask loaded empty |
| 2 | Dice scored empty-prediction-vs-empty-target as **1.0** | hid #1 as "mean Dice 1.0000" while loss sat flat at 0.95 |
| 3 | Masks skipped the image's crop/pad/resize geometry | annotations offset from the pixels they describe |
| 4 | Segmentation at 512 | microaneurysm Dice 0.000 |
| 5 | Deployed pipeline segmented at 512 while features were trained at 1024 | deployed system disagreed with its own validation |
| 6 | Calibrator not shipped with the model | threshold applied to a different probability scale |
| 7 | `lesion_threshold = 0.5` never fitted | true optimum 0.85–0.95 |
| 8 | FOV clipping penalty | rejected **34%** of real images whose coverage was 0.90–1.00 |
| 9 | Isotonic calibration pinned the score floor to exactly 0.0 | on Messidor-2 that sent 10.3% of true positives to zero, unreachable by any threshold; specificity at 90% sensitivity fell 0.628 → **0.000** |

Bugs 2, 8 and 9 deserve emphasis. **#2 is the dangerous shape**: a metric
reporting a perfect score for a model that had learned nothing, from data that
contained nothing. **#8 was structurally invisible to the phantoms**, which
always render a black margin — but a real fundus aperture is wider than the
sensor is tall, so the retina touches the frame edge on almost every correct
capture, and APTOS ships pre-cropped touching all four. **#9 is the one that
only external validation could find**: isotonic recalibration measurably
improved in-distribution ECE while silently destroying the model's operating
range under distribution shift, because its ties are harmless until the score
distribution moves. Blending a sliver of the raw score back in restores a
strict total order and keeps both properties — ECE 0.0519 → 0.0130 *and*
external AUC 0.875 with specificity 0.628 at 90% sensitivity. All three have
regression tests.

### Deployed behaviour on 120 real test images

| | before fixes | after |
|---|---|---|
| Recapture rate | 34% | **0%** |
| Cases flagged urgent | 100% | 23% |
| Exact grade match | 59.5% | **68.3%** |

The 100%-urgent figure was the rule engine — measured specificity **0.058** —
unilaterally overriding a calibrated model with specificity 0.939. Lesion-based
escalation now requires corroboration: it cannot override a *confidently*
negative neural verdict, though the disagreement is still written into the
report and the audit log. Neovascularisation remains an unconditional
escalation, being both specific and sight-defining.

## Clinical validation

`scripts/validate.py` enforces the split discipline and produces:

| | fitted on | reported |
|---|---|---|
| temperature, referral threshold | **val** | — |
| internal performance | — | **test** |
| zero-shot generalisation | nothing | **external** (Messidor-2) |

Metrics carry intervals, and model comparisons use paired tests:

- Wilson score intervals for sensitivity/specificity/PPV/NPV — correct at the
  extremes, which is where screening metrics live.
- **DeLong** for AUC variance and for comparing two correlated AUCs
  (verified bit-exact against `sklearn.roc_auc_score`).
- **McNemar** (exact below 25 discordant pairs, continuity-corrected above)
  for comparing referral decisions.
- Stratified bootstrap for QWK.
- **Adjacent (within-one-grade) accuracy** alongside exact accuracy. Human
  graders agree exactly on ICDR grade only ~60–75% of the time but within one
  grade >90%; quoting exact match alone against a human reference understates
  performance, and quoting it *without* the adjacent figure is how DR papers
  mislead.

### The ablation the PS asks for

"the integrated pipeline outperforms any single technique" is a claim that has
to be measured, so `src/drscreen/evaluation/ablation.py` runs:

| arm | technique |
|---|---|
| `rule_based` | classical CV: segmentation → ICDR criteria, no deep grader |
| `cnn_only` | deep learning only |
| `clinical_only` | lesion features only, through the ordinal head |
| **`fusion`** | **the integrated system** |
| `no_preprocess` | fusion on raw RGB — isolates the enhancement contribution |

Each arm gets its own temperature and threshold chosen on val, so the
comparison is between best-configured systems rather than one tuned model
against handicapped rivals. Every pairwise test against `fusion` is DeLong on
AUC plus McNemar on the decision, paired on the same cases. The verdict string
reports honestly when a margin is *not* significant — and would report it if
fusion lost.

> **The ablation is at ceiling on phantom data and you should not read a
> clinical finding into it.** A phantom's grade is a near-deterministic
> function of its lesion counts, so every arm reaches referable-DR AUC ≈ 0.99–1.00
> and there is nothing left for fusion to win. This is a property of the data,
> not evidence that the arms are equivalent. The ablation is a *machine* that
> becomes informative when pointed at APTOS/IDRiD, where grading is genuinely
> ambiguous.
>
> To make the synthetic study less degenerate, `build_cohort.py --label-noise
> 0.25` injects ±1-grade reference-standard error, which is what real ICDR
> grading looks like: human graders agree exactly only ~60–75% of the time and
> almost all disagreement is by one grade. Validating against noiseless labels
> flatters any model.

---

## Telemedicine simulation (PS item 5)

`src/drscreen/sim/telemedicine.py` models a district programme as a queueing
network. The interesting behaviour is in the couplings, which is why a
spreadsheet cannot answer these questions:

- **Recapture loops.** A rejected image sends the patient back into the camera
  queue they just left, so a gate 5 points stricter can cost far more than 5%
  of throughput.
- **Bandwidth as a shared, intermittent resource,** with a two-state link model.
- **Reviewer capacity as the real bottleneck,** coupled to the model's
  selective-referral threshold.
- **Clinical urgency preemption** — a FIFO review queue meets its average SLA
  while failing the patients who matter.

Scenario results (100,000 patients/year, 12 PHCs, 2 ophthalmologist FTE):

| scenario | throughput/yr | bottleneck | reviewer load | routine SLA | p90 turnaround | feasible |
|---|---|---|---|---|---|---|
| `baseline_manual` | 99,968 | reviewer | **140.5%** | 66.6% | 17.9 d | **no** |
| `ai_assisted` | 99,636 | camera | 11.0% | 100% | 0.01 d | throughput only |
| `ai_edge_lowbandwidth` | 99,568 | camera | 10.9% | 100% | 0.02 d | throughput only |
| `stress_2x_demand` | 200,455 | camera | 21.9% | 100% | 0.02 d | yes |

The headline: **without AI triage the review queue is unstable** — demand is
141% of available reading capacity, so the backlog grows without bound and no
affordable staffing level meets the SLA. AI-assisted review takes that to 11%.

The residual throughput gap (99,636 vs 100,000) is not a rounding artefact: it
is the patients who remain ungradeable after the maximum number of recaptures.
To *screen* 100,000 you must *see* about 100,400. The optimiser accounts for
this by sizing capture capacity accordingly.

`scripts/run_simulation.py --optimise` searches configurations against a cost
model (`DEFAULT_COSTS`, in INR, stated explicitly so procurement figures can
replace them) and returns the cheapest plan meeting throughput, both SLAs, and
a utilisation ceiling of 85% — the ceiling matters because queueing systems
degrade super-linearly and a plan running one ophthalmologist at 97% "on
paper" fails the first week someone takes leave.

Over 1,024 configurations — 192 rejected analytically as unstable, 832
simulated, 532 feasible — the cheapest feasible plan is:

| lever | value |
|---|---|
| PHCs with a camera | 8 |
| cameras per PHC | 1 |
| ophthalmologist FTE | **1.0** |
| auto-report coverage | 0.50 |
| review mode | **AI-assisted (0.5 min/case)** |
| inference | on-device (edge) |
| connectivity | 3G |

100,394 screened/year, camera-bound at 54% utilisation, reviewer at 35%, both
SLAs met, **Rs 53.9 lakh/year — about Rs 54 per patient screened.**

Two things the search decided rather than assumed:

- It **chose AI-assisted review** over unaided reading. Every one of the 192
  analytically-rejected configurations was an unaided-review design. The value
  of AI triage is an output of the optimisation, not a premise of it.
- It **chose edge inference**, because removing the GPU server and most of the
  bandwidth requirement is worth more than the per-PHC device cost at this
  scale — which is the opposite of the cloud-first default most designs reach
  for.

Two notes on the search itself:

- The grid deliberately spans **whether to use AI triage at all**
  (`review_time_min` 2.5 = unaided reading, 0.5 = reading a pre-annotated
  case). Without that axis the optimiser only ever compares AI-assisted
  designs against each other and never has to show that AI assistance is what
  makes the programme affordable.
- Configurations are **pre-screened analytically** with Little's Law before
  any simulation runs (`offered_load`): if offered load ≥ 1 the queue is
  unstable and more simulated time only produces a larger backlog. On the
  1024-configuration grid this rejects 192 (19%) without simulating them —
  all of them unaided-review designs. Note that with AI triage enabled, the
  *worst* configuration in the space still only reaches 0.67 offered load,
  which is itself the finding: AI assistance is what keeps the design space
  stable at all.
- Every configuration is run under multiple random seeds and judged on its
  *worst* utilisation, because a plan that looks feasible under one kind
  random stream is not a plan.

### MATLAB / Simulink

The PS names Simulink. `scripts/run_simulation.py --export-matlab matlab/`
generates, from the same `SimConfig`:

- `dr_screening_params.m` — every parameter as a MATLAB struct
- `build_dr_screening_model.m` — builds the SimEvents block diagram
- `validate_against_simpy.m` — runs the Simulink model and diffs its outputs
  against the SimPy reference

One source of truth, two runtimes, and a script that keeps them honest.
`matlab/README.md` documents the block-level mapping and names the two
elements (the recapture feedback path and the link-state Markov chain) that
need the graphical editor.

---

## Using the real datasets

Download and accept the licence for each, then arrange as below:

| dataset | role | source |
|---|---|---|
| APTOS 2019 | train + val | <https://www.kaggle.com/c/aptos2019-blindness-detection> |
| IDRiD | train + val, pixel-level lesions | <https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid> |
| DRIVE | vessel segmentation | <https://drive.grand-challenge.org/> |
| **Messidor-2** | **held-out external test** | <https://www.adcis.net/en/third-party/messidor2/> |

```
data/raw/
  aptos2019/     train.csv, train_images/
  idrid/         A. Segmentation/, B. Disease Grading/
  drive/         training/, test/
  messidor2/     IMAGES/, messidor_data.csv
```

```bash
python scripts/build_cohort.py --source real --data-root data/raw --out data/cohort_real
# ...then the same train/validate commands, pointed at data/cohort_real
```

The loaders search rather than assume exact paths, since these archives unpack
differently on different systems. Messidor-2 adjudicated grades come from a
separate CSV (Krause et al. 2018); without it the loader still returns images
for inference but metrics are unavailable.

---

## Honest limitations

- **The trained weights in this repository are fitted on phantoms, not
  patients.** The numbers `validate.py` prints on synthetic data measure
  whether the pipeline is correctly wired, *not* clinical performance. Any
  clinical claim requires re-running the same scripts on APTOS/IDRiD with
  Messidor-2 held out. Nothing about the code changes; only the cohort path.
- The phantom generator is anatomically structured but is **not** a claim of
  photorealism. Its jobs are integration testing, demonstration, and catching
  training bugs.
- The venous-beading cue currently runs on a morphological vessel proxy and is
  deliberately conservative — it does not fire on healthy retinas
  (regression-tested), but it will under-detect real beading until a vessel
  U-Net is trained on DRIVE. The 4-2-1 "2" arm is therefore weaker than the
  "4" arm today.
- Severe-NPDR phantoms pick one of the three 4-2-1 arms at random
  (4-quadrant / beading / IRMA), and the 4-quadrant arm places haemorrhages
  and microaneurysms quadrant-uniformly so the label is actually satisfied by
  the pixels. An earlier version scattered them at random, which reached the
  ≥20-per-quadrant threshold in only ~1.6 quadrants and quietly penalised any
  grader that implements ICDR correctly. Worth knowing if you extend the
  generator: **a phantom whose label contradicts its own pixels makes the
  correct model look wrong.**
- Optic-disc localisation is ~89% within 1 DD on phantoms, below the 95–99%
  published on real fundus images. The vessel-convergence weight is tuned on
  phantoms and should be re-fit on IDRiD's optic-disc masks; the failure mode
  it defends against (a bright confluent exudate) is under-represented in
  phantoms.
- Cost figures in `DEFAULT_COSTS` are order-of-magnitude inputs, not findings.

---

## Layout

```
src/drscreen/
  constants.py              ICDR scale, lesion taxonomy, targets -- stated once
  pipeline.py               end-to-end orchestrator
  api.py                    FastAPI service + review audit log
  preprocess/
    fov.py                  circular FOV detection, crop, square pad, resize
    enhance.py              adaptive enhancement operators
    quality.py              9-criterion interpretable gate + recapture advice
    landmarks.py            analytic optic disc / fovea, clinical coordinate frame
  data/
    registry.py             dataset discovery + enforced split policy
    synthetic.py            procedural fundus phantom generator
    torch_data.py           datasets, augmentation policy, caching
    cohort.py               materialised on-disk cohort format
  models/
    segmentation.py         attention U-Net, focal Tversky, deep supervision
    grader.py               CORN ordinal head + clinical-feature fusion
    lesion_features.py      clinical features + rule-based ICDR grader
    calibration.py          temperature/isotonic, ECE, operating point, risk-coverage
  explain/
    cam.py                  Grad-CAM / Grad-CAM++ / HiResCAM, occlusion
    faithfulness.py         deletion, insertion, pointing game, sparsity
    report.py               annotated review panel + HTML clinical report
  evaluation/
    metrics.py              Wilson, DeLong, McNemar, QWK, bootstrap
    ablation.py             integrated-vs-single-technique study
  sim/
    telemedicine.py         SimPy discrete-event district model
    optimize.py             constrained cost minimisation + sensitivity
    simulink_export.py      SimEvents model + parameter generation
scripts/                    build_cohort, train_seg, precompute_features,
                            train_grader, validate, run_demo, run_simulation,
                            eval_landmarks
web/index.html              review console
matlab/                     generated Simulink bridge
tests/                      27 regression tests on clinical invariants
```

---

## Not a diagnosis

Decision-support output only. Every referable and every sight-threatening
finding is reviewed by a qualified ophthalmologist before clinical action.
