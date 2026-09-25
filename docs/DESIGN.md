# Design and validation

How the pipeline works, the design decisions that differ from the standard recipe and why, and how it is validated. Measured numbers are in [RESULTS.md](../RESULTS.md).

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

Measured on IDRiD's hand-marked disc and fovea centres
(`scripts/eval_landmarks.py`, 103 held-out real photographs): optic disc median
error **0.098 DD** with **98.1%** within 1 DD, fovea median **0.178 DD** with
**93.2%** within 1 DD — in line with the 95–99% / 90–96% published on real
fundus images.

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

## Why there are no generated images

This project used to ship a procedural fundus phantom generator. It is gone,
and the reason is worth stating because generating training data is a common
suggestion.

A phantom is drawn from the pipeline's own assumptions, so it agrees with them.
That makes it useless as evidence — a screening result carries no weight when
the thing screened was drawn by the same project that graded it — and worse than
useless as a test fixture, because a test that only ever sees agreement cannot
fail on a mistaken assumption. Four defects reached the deployed pipeline behind
a green suite for exactly that reason:

- The FOV criterion rejected **34% of genuinely gradeable real images**, because
  phantoms always render a black margin and real fundus apertures touch the
  sensor edge.
- The triage rule escalated **631 of 631** real photographs to urgent referral,
  because phantoms annotate neovascularisation and no real corpus here does.
- The focus criterion **passed severely defocused images** — it was non-monotone
  in blur, and the phantom set never explored past the range where it worked.
- The optic-disc detector's vessel-convergence weight was fitted on drawn
  vessels and was **2.5 points worse** than the real-data optimum.

The last two were found by deleting the generator and re-pointing everything at
real photographs. See [RESULTS.md §6](../RESULTS.md#6-eighteen-bugs-that-only-real-data-exposed).

What replaced it: twelve real APTOS-2019 and IDRiD held-out photographs
committed to the repository (2.2 MB, `drscreen.data.samples`), covering all five
ICDR grades. Where a test needs a *degraded* image, the degradation is applied to
one of those photographs rather than simulated from scratch — blurring a real
retina tests whether the gate can tell a defocused real image from a sharp one,
which is the actual question.

---

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
strict total order and keeps both properties — ECE 0.0306 → 0.0153 *and* an
external AUC of 0.908 with a usable operating range. All three have regression
tests.

### Deployed behaviour on 120 real test images

*Measured on the checkpoint preceding the grade-3/4 fixes; this before/after is
for bugs #1–#9 and has not been re-measured on the current checkpoint.*

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

---

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
| **`cnn`** | **deep learning only — the deployed grader** |
| `clinical_only` | lesion features only, through the ordinal head |
| `fusion` | CNN image features fused with clinical lesion features |
| `no_preprocess` | fusion on raw RGB — isolates the enhancement contribution |

The deployed arm is whichever `--grader` is passed; it is keyed in the ablation
by the checkpoint's own `arm` field, and a `--arms` label that collides with it
is a hard error. Hard-coding `"fusion"` as the reference previously let a
comparison arm silently replace the deployed model in its own ablation.

Each arm gets its own temperature and threshold chosen on val, so the
comparison is between best-configured systems rather than one tuned model
against handicapped rivals. Every pairwise test against `fusion` is DeLong on
AUC plus McNemar on the decision, paired on the same cases. The verdict string
reports honestly when a margin is *not* significant — and would report it if
fusion lost.

> The ablation only says something when the grading task is genuinely
> ambiguous, which is why it is run on APTOS/IDRiD/DDR/Messidor-2 and not on
> anything easier. On a corpus where the grade is a near-deterministic function
> of lesion counts every arm reaches referable-DR AUC ≈ 0.99–1.00 and there is
> nothing left for any arm to win — a property of the data, not evidence that
> the arms are equivalent.
