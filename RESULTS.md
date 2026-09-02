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
| **Internal test** (n=631) | **0.986** | **0.873** | 0.988 | 0.894 | ✅ both met |
| **External, Messidor-2** (n=1,744) | 0.707 | 0.922 | 0.908 | 0.593 | ❌ sensitivity not met |

Targets: sensitivity ≥ 90%, specificity ≥ 85% for referable DR (ICDR grade ≥ 2).

Deployed arm: **CNN (image-only ordinal grader)** at referral threshold
**0.1999**, selected on the validation split under the sensitivity-first policy
(`--threshold-policy max_sensitivity`). See [§5](#5-ablation--does-the-integrated-pipeline-beat-any-single-technique)
for why the image-only arm is deployed rather than the fusion arm.

**The aggregate external sensitivity is misleading on its own.** Split by true
severity, the misses are almost entirely *moderate* NPDR, not blinding disease:

| true severity | n | flagged referable |
|---|---|---|
| 0 — no DR | 1,017 | 6.1% *(correctly ignored)* |
| 1 — mild NPDR | 270 | 14.1% *(correctly ignored)* |
| **2 — moderate NPDR** | 347 | **62.3%** ← still the weak point |
| **3 — severe NPDR** | 75 | **98.7%** *(74/75)* |
| **4 — proliferative DR** | 35 | **94.3%** *(33/35)* |

> **Sight-threatening disease (grade ≥ 3): sensitivity 0.973 [0.923–0.991] at
> 92.2% specificity** — 107 of 110 blinding eyes referred, zero-shot, on a
> French cohort with different cameras and a different grading panel, with
> nothing fitted on it. On the internal test split it is **1.000
> [0.965–1.000]** — all 105 sight-threatening eyes referred, none missed.

Grade 2 accounts for 76% of referable cases, which is what drags the aggregate
to 0.707. The two failure modes do not carry equal clinical weight: a missed
proliferative DR can cost sight within months; a missed moderate NPDR is
picked up at the next annual screen.

---

## 2. Verdict — can this system detect DR?

**As a sight-saving triage tool: yes.** It detects **97.3% of blinding disease**
on cameras it has never seen while correctly ignoring 92% of non-referable eyes,
so it does not flood the ophthalmologist — 24% of the population is queued for
review. On the internal test split it misses none at all (105/105). That is the
decision the problem statement exists to serve.

**As a full ICDR grader on unseen hardware: not yet.** Moderate-NPDR detection
reaches 62.3% — up from 27.7% two checkpoints ago — but is still not acceptable.
And *exact grade assignment* stays weak even internally: the referral flag is
trustworthy, the printed grade much less so. See
[§4.1](#41-exact-grade-assignment-is-weaker-than-referral).

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

**Why it is fixable.** The ranking survives. On Messidor-2 the model still
separates the grades in the right order — it flags 6.1% of grade 0, 14.1% of
grade 1, 62.3% of grade 2 and 97.3% of grade ≥ 3 — the scores simply sit lower
against a threshold learned from a stricter-looking population:

| threshold | sens grade 2 | sens grade ≥3 | specificity | % flagged |
|---|---|---|---|---|
| 0.050 | 72.6% | 98.2% | 88.4% | 29.2% |
| 0.100 | 72.6% | 98.2% | 88.5% | 29.1% |
| 0.150 | 62.2% | 97.3% | 92.2% | 24.3% |
| **0.1999** *(deployed)* | **62.2%** | **97.3%** | **92.2%** | **24.3%** |
| 0.200 | 53.9% | 97.3% | 93.6% | 21.6% |
| 0.250 | 43.8% | 92.7% | 96.2% | 17.4% |
| 0.300 | 43.2% | 92.7% | 96.2% | 17.3% |
| 0.400 | 38.3% | 91.8% | 97.2% | 15.5% |
| 0.600 | 28.8% | 89.1% | 98.3% | 12.6% |
| 0.700 | 11.8% | 78.2% | 99.4% | 7.7% |
| 0.800 | 7.2% | 66.4% | 99.8% | 5.7% |

The 0.1999 and 0.200 rows are genuinely distinct points, not a duplicate: the
isotonic calibrator maps a run of raw scores onto a tie just above 0.1999, so a
threshold 0.0001 higher drops 2.7% of the cohort out of the flagged set. The
deployed point sits on the *low* side of that tie deliberately — the
sensitivity-first policy takes the most sensitive val point that still clears
the specificity floor.

> **This table is now generated**, by `metrics.threshold_sweep`, with every row
> computed at the threshold it is labelled with and the deployed point always
> present. The hand-assembled version that previously stood here was **shifted
> by one row**: the frozen threshold's true values (specificity 97.82%, 12.79%
> flagged) sat on the row labelled `0.400`, so the operating point it
> recommended had never actually been measured. Generating the table is exactly
> the fix for that class of error.

**Implication:** the operating point does more work than the weights. Across
three checkpoints, external referable sensitivity went 0.427 → 0.619 → 0.707 and
external AUC 0.875 → 0.901 → 0.908. Real discrimination was gained — the AUC is
threshold-free and it moved — but the larger share of the deployed-point
improvement came from the threshold relocating 0.657 → 0.400 → 0.200 as the
selection policy was corrected and then made sensitivity-first.

Critically, **none of those thresholds was chosen by looking at Messidor-2**.
Each was selected on the validation split; the external numbers stay an honest
zero-shot measurement. Reading the operating point off the external sweep would
have produced a slightly better-looking table and destroyed the only claim in
this document worth making.

More epochs, bigger backbones and heavier augmentation still do not address a
label-definition mismatch. The remaining fixes are (a) per-site threshold
calibration against a few hundred locally-graded images, which the `/audit`
endpoint is built to collect, or (b) multi-source training with harmonised
reference standards — for which the EyePACS and DDR loaders and `--curate` now
exist, pending the download.

---

## 4. Full metrics

### Referable DR (grade ≥ 2)

| metric | internal test (n=631) | external / Messidor-2 (n=1,744) |
|---|---|---|
| Sensitivity | 0.9859 [0.9644–0.9945] | 0.7068 [0.6635–0.7467] |
| Specificity | 0.8732 [0.8341–0.9042] | 0.9223 [0.9064–0.9357] |
| PPV | 0.8642 | 0.7636 |
| NPV | 0.9870 | 0.8986 |
| AUC | 0.9883 [0.9790–0.9935] | 0.9082 |
| QWK | 0.8939 [0.8703–0.9149] | 0.5930 [0.5549–0.6287] |
| Exact accuracy | 0.8003 [0.7673–0.8296] | 0.6439 [0.6212–0.6661] |
| Within-one-grade | 0.9604 | 0.8899 |
| ECE | 0.0280 | 0.1133 |

Intervals are Wilson score for proportions and DeLong for AUC.

### Sight-threatening DR (grade ≥ 3)

| metric | internal test | external / Messidor-2 |
|---|---|---|
| **Sensitivity** | **1.0000 [0.9647–1.0000]** (105/105) | **0.9727 [0.9229–0.9907]** (107/110) |
| Specificity vs grades 0–1 | 0.8732 | 0.9223 |
| Specificity vs grades 0–2 | 0.5837 | 0.8066 |

Both framings are stated because they differ materially. The 0–2 denominator is
the honest one for a grade ≥ 3 question; quoting the referable task's own
specificity against a grade ≥ 3 sensitivity counts flagged grade-2 eyes as true
positives for a question that treats them as negatives.

### 4.1 Exact grade assignment is weaker than referral

Referral and grading are two different decision rules over the same logits.
Referral thresholds P(grade ≥ 2) at **0.1999**, a value fitted on val; the
printed grade is the CORN cumulative assignment at a hard-coded **0.5**, which
nobody has ever fitted. They disagree, and the gap matters:

**Internal test (n=631)**

| true grade | exact-grade recall | referred |
|---|---|---|
| 0 | 0.962 (281/292) | 5.1% |
| 1 | 0.764 (42/55) | 52.7% |
| 2 | 0.799 (143/179) | 97.8% |
| **3** | **0.362 (17/47)** | **100%** (47/47) |
| **4** | **0.379 (22/58)** | **100%** (58/58) |

Internal-test confusion matrix, true grade 3: `[0, 0, 23, 17, 7]` — every
grade-3 eye lands at grade ≥ 2 and is referred, but 23 of 47 are *labelled*
moderate. True grade 4: `[0, 3, 17, 16, 22]` — three proliferative eyes are
labelled grade 1, yet all 58 still clear the referral threshold, because
P(referable) reaches 0.1999 long before the grade assignment flips at 0.5.

**Read the referral flag, not the printed grade.** The grade is a triage
convenience; the referral decision is the one that has been validated. Exact
grade-3 recall has now *fallen* across two successive checkpoints
(0.426 → 0.383 → 0.362) while sight-threatening referral rose to 1.000 — the
system keeps getting better at deciding *whether* to refer and no better at
saying *how bad* it is. That is a real limitation, not a presentational one,
and it is the strongest argument for the grade-aware decision rule in
[§10](#10-what-would-move-the-numbers).

### Lesion segmentation (IDRiD, 64 training images)

| class | Dice @512 | Dice @1024 | published IDRiD range |
|---|---|---|---|
| Microaneurysm | 0.000 | **0.481** | 0.30–0.50 |
| Haemorrhage | 0.248 | **0.539** | 0.50–0.65 |
| Hard exudate | 0.272 | **0.572** | 0.70–0.80 |
| Cotton-wool spot | 0.187 | **0.628** | 0.55–0.70 |
| Neovascularisation | — | — | **not annotated in IDRiD** |

Resolution is decisive: at 512 a microaneurysm survives downsampling from
4288px as ~4 pixels and Dice is **0.000**. The lesion model runs at 1024; the
grader runs at 512.

IDRiD does not annotate neovascularisation at all, so that channel cannot be
learned. Proliferative DR is still graded from image features, but the
*explanation* cannot cite NV as evidence. The channel is now excluded from the
segmentation loss and recorded in the checkpoint as `supervised_lesion_classes`,
so the pipeline reports NV as **"not assessed"** rather than as a zero count —
see bug #11 in [§6](#6-fifteen-bugs-that-only-real-data-exposed).

### Landmark localisation

Optic disc median error **0.015 DD**, fovea **0.077 DD**, 97% of foveae within
1 disc diameter (`scripts/eval_landmarks.py`). Closed-form, no training data,
~120 ms on CPU.

### Calibration

Current checkpoint:

| | value |
|---|---|
| Temperature (multiclass CORN NLL) | T = 2.7080 |
| val ECE, before → after temperature | 0.0500 → 0.0254 |
| val MCE | 0.2461 → 0.1453 |
| val Brier | 0.0701 → 0.0606 |
| **Isotonic on P(referable), out-of-fold** | **0.0160** vs 0.0254 → adopted |
| Test ECE / external ECE | 0.0280 / 0.1133 |

Adoption is decided **out-of-fold**, because isotonic scored on its own fitting
split drives ECE to ~0 by construction and would always look like a win.

The tie-break ablation below was run on the checkpoint preceding the grade-3/4
fixes and has not been repeated; it is retained because it is why the deployed
calibrator blends a sliver of raw score back in:

| | val ECE | external AUC | spec @ 90% sens |
|---|---|---|---|
| Temperature only (T = 3.797) | 0.0519 | 0.8751 | 0.628 |
| Isotonic, no tie-break | 0.0000 *(in-sample, meaningless)* | 0.8626 | **0.000** |
| **Isotonic + tie-break** *(deployed)* | **0.0064** | **0.8751** | **0.628** |

### Explanation quality

Faithfulness **+0.127** (insertion 0.974 − deletion 0.848), attention sparsity
(Gini) 0.772 over 40 referable images. Pointing-game and lesion-IoU require
pixel annotations, which the APTOS test split does not have.

---

## 5. Ablation — does the integrated pipeline beat any single technique?

| arm | AUC | 95% CI | sens | spec | QWK |
|---|---|---|---|---|---|
| **cnn** *(deployed)* | **0.9883** | 0.9790–0.9935 | 0.986 | 0.873 | **0.8939** |
| fusion | 0.9850 | 0.9757–0.9908 | 0.989 | 0.882 | 0.8817 |
| clinical_only | 0.9272 | 0.9058–0.9441 | 0.880 | 0.813 | 0.6904 |
| rule_based | 0.9115 | 0.8860–0.9317 | 1.000 | 0.075 | 0.0000 |

Paired DeLong tests against the deployed arm:

- vs `fusion`: cnn higher by 0.0032, **p = 0.034** ✅
- vs `clinical_only`: cnn higher by 0.0610, **p = 1.7×10⁻¹¹** ✅
- vs `rule_based`: cnn higher by 0.0768, **p = 1.5×10⁻¹²** ✅

**Honest conclusion: the fusion arm lost, and the image-only arm is deployed.**
Across three checkpoints the clinical branch went from a slight QWK advantage,
to a statistical tie, to a *significant deficit* (p = 0.034). Correcting the
lesion thresholds — caching features at the same F1-fitted values the live
pipeline applies, rather than a blanket 0.5 — made the lesion counts sparser
and more accurate, and the fusion head got worse. The most likely reading is
that it had been exploiting a spurious regularity in the mis-thresholded
counts.

**This does not mean the pipeline is not integrated.** The system is still
end-to-end: quality gate → vessel and lesion segmentation → landmarks →
clinical features → ordinal grading → lesion-grounded explanation → capacity
simulation. What changed is that the clinical-feature branch now powers the
*explanation and the rule trail* rather than the referral score. Every referral
still ships lesion counts, quadrant maps, 4-2-1 reasoning and a CSME estimate;
they are simply no longer inputs to the number that decides referral.

Note that `rule_based` reaches sensitivity 1.000 at specificity 0.075 and
QWK 0.0000 — it refers essentially everyone. That is what a classical
lesion-criteria arm does without a learned grader, and it is the baseline the
problem statement's "existing solutions" critique is aimed at.

The verdict string in `outputs/validation/ablation.txt` is generated, not
written. It now correctly stars `cnn` as the deployed arm — until this run it
hard-coded `"fusion"` as the reference, which silently dropped the deployed
model from its own ablation (bug #14 in [§6](#6-fifteen-bugs-that-only-real-data-exposed)).

---

## 6. Fifteen bugs that only real data exposed

Listed because most are *invisible* failures — they produce a plausible number
rather than a crash. The first nine were found during the first real-data run,
the next four by asking why grades 3 and 4 were collapsing, and the last two
while correcting a train/serve preprocessing skew.

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

All three now have regression tests.

### Four more, found by asking why grades 3–4 collapsed

The nine above were found before and during the first real-data run. A later
audit of the *internal* test split — same cameras and same graders as training,
so distribution shift explains none of it — found severe-NPDR recall at 0.426
and proliferative at 0.431, with the misses folded into grade 2. Nothing in the
training loop or the validation artefact reported it. Four independent causes:

| # | cause | why it was invisible |
|---|---|---|
| 10 | Checkpoint selection used referable-DR AUC = σ(z0)·σ(z1) | that expression contains neither z2 nor z3 — the units deciding grades 3 and 4 — so the criterion was *mathematically incapable* of observing the collapse; it preferred epoch 30 to epoch 18 for a 0.002 AUC gain while QWK fell |
| 11 | Neovascularisation channel trained against an all-zero target | IDRiD annotates no NV, so the channel learned to never fire, the grade-4 rule arm became unreachable dead code, and the report printed the resulting zero as a *negative finding* — claiming an exclusion the model never made |
| 12 | `corn_loss` averaged per-task means and passed class weights to `binary_cross_entropy_with_logits` | `weight=` returns `mean(w·loss)` without renormalising, and empty tasks were dropped, so the effective learning rate on z2/z3 moved with batch composition |
| 13 | Severity and sweep tables assembled by hand after the run | rows drifted from the thresholds they were labelled with (see [§3](#3-why-moderate-npdr-fails--the-reference-standards-disagree)) |

**Fixes.** QWK checkpoint selection (`--select-on`), with sight-threatening and
per-grade recall logged every epoch and `last.pt` written alongside `best.pt`;
unsupervised channels detected from the stored masks, excluded from the loss and
recorded as `supervised_lesion_classes`, which the pipeline surfaces as **"not
assessed"** rather than "none detected"; `corn_loss` reduced to a weighted mean
over every `(sample, task)` conditional term; a square-root-stratified sampler
lifting grades 3–4 from ~17% to 27.1% of each batch, with rebalancing split
evenly between sampler and loss weights so the two compose to full balance
rather than compounding into ~30× over-weighting; and `severity_breakdown` /
`threshold_sweep` generated into the validation artefact.

**Effect.** All three arms now select a mid-run epoch rather than the last;
under the old criterion every one would have shipped epoch 30, where AUC(ref)
peaked while QWK was already falling. Sight-threatening referral went
0.426/0.431 → **1.000** internally and **0.973** externally. Exact grade-3
recall did *not* improve — see [§4.1](#41-exact-grade-assignment-is-weaker-than-referral).

### Two more, from a train/serve preprocessing skew

Neither is visible in `validation.json`, because `validate.py` reads the same
cohort images training produced. The skew only exists between the cohort and
the *live* pipeline, so no metric on either side can see it.

| # | bug | how it presented |
|---|---|---|
| 14 | `build_cohort` applied `cv2.COLOR_RGB2BGR` to the output of `to_model_input` | that output is the hybrid feature stack `[CLAHE-green, Ben-Graham, L*]`, not an RGB image, so the conversion reversed planes 0 and 2. The cohort stored the mirror of what `pipeline.py` feeds the same model. Verified empirically: `stored == live[:, :, ::-1]`, channel 1 byte-identical |
| 15 | Feature caching used a blanket threshold of 0.5 while live inference used the F1-fitted per-class values | the fusion grader was trained on lesion counts it is never served |

A third, found while deploying the CNN arm: `validate.py` registered the
deployed grader under the literal key `"fusion"` and `compare_arms` defaulted
`reference="fusion"`, so a run whose deployed arm was `cnn` plus a
`--arms fusion=...` comparison silently **overwrote the deployed model with the
comparison arm** and starred it as the integrated pipeline. The deployed model
never appeared in its own ablation. It only bites when the deployed arm is not
fusion, which is exactly the change that exposed it. The integrated arm is now
keyed by the checkpoint's own `arm` field, and a colliding `--arms` label is a
hard error rather than a silent replacement.

**Consequence.** Bugs #14 and #15 change what `build_cohort` writes, so the
cohort was rebuilt from `data/raw` and the whole chain retrained — a
grader-only retrain would have paired a new grader with old-order images. The
segmentation Dice is unchanged by the swap (a conv net learns whichever order
it is given consistently), which is the point: the model was never broken, only
the thing it was served at inference.

Suite: **50 tests**, `pytest tests/ -q`.

### Effect on deployed behaviour (120 real test images)

*Measured on the checkpoint preceding the grade-3/4 fixes; the before/after
contrast is for bugs #1–#9 and has not been re-measured on the current
checkpoint.*

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
| EyePACS 2015 | loader ready, **not yet downloaded** | ~88,000 |
| DDR | loader ready, **not yet downloaded** | 13,673 |

Cohort: 6,047 cases — train 2,925 / val 622 / test 631 / external 1,748, plus
the segmentation and vessel splits. **Measured subject overlap between the
grading splits: zero.** `registry.assert_no_leakage` raises `SplitViolation`
if Messidor-2 reaches the training pool.

Training-grade counts are the binding constraint on the deep CORN conditionals:
train carries **207 grade-3 and 260 grade-4** images, and task 3 (grade 4 vs 3)
trains only on that subset — 467 images in total. `scripts/fetch_datasets.py`
downloads EyePACS and DDR, `registry.load_eyepacs` / `load_ddr` read them, and
`build_cohort --curate` down-samples over-represented grades in the **training
split only**, drawing subjects round-robin across sources so grade does not
correlate with imaging chain. EyePACS is still un-downloaded; DDR has been run,
and [§7.1](#71-adding-ddr-what-it-fixed-and-what-it-broke) reports the result.

Split discipline: `val` fits the temperature and selects the referral
threshold; `test` is the internal estimate; `external` has nothing fitted on
it. Four Messidor-2 images its adjudicators marked ungradable are excluded from
metrics rather than scored as a sixth class.

### 7.1 Adding DDR: what it fixed and what it broke

DDR (13,673 images, 147 Chinese hospitals) was added to the grading pool and the
whole chain retrained. **The deployed model is not this one** — the experiment
is reported because it separates two things that had been moving together, and
because it settles a question three checkpoints could not.

On load, 1,151 DDR images carry **grade 5, its ungradable marker**. Read as an
ordinal grade that is a sixth class in a five-class problem; the loader returns
them as `grade=None` and the cohort builder drops them, reporting the count.

Curation mattered. DDR brings 4,397 grade-0 training images against APTOS's
1,256, so grade 0 would have been 76% DDR. Round-robin selection pulled it to
843 APTOS / 842 DDR / 113 IDRiD at the cap. Grades 3 and 4 were left whole,
taking train from 207/260 to **376/899**.

**What it fixed — exact grading, decisively.** This was the standing limitation:
grade-3 exact recall had *fallen* across three checkpoints while referral rose.

| exact-grade recall | APTOS+IDRiD | +DDR |
|---|---|---|
| external grade 1 | 0.085 | **0.515** |
| external grade 2 | 0.228 | **0.611** |
| external grade 3 | 0.280 | **0.493** |
| external QWK | 0.593 | **0.702** |
| internal grade 3 | 0.362 | **0.525** |
| internal grade 4 | 0.379 | **0.699** |

That is the data-volume hypothesis confirmed. Oversampling had already been
tried — the stratified sampler lifted grades 3–4 to 27% of every batch and exact
recall did not move — so the constraint was genuinely the number of distinct
images, not the gradient share they received.

**What it broke — the operating point.** External specificity collapsed from
0.922 to 0.753, flagging 41.8% of the population instead of 24.3%. Two
threshold policies were tried; neither recovered it (`max_sensitivity` 0.719,
`youden` 0.753). At *matched* specificity the older model is better on both
axes:

| at external spec ≈ 0.93 | sens referable | sens grade ≥3 |
|---|---|---|
| **APTOS+IDRiD @ 0.20** (spec 0.922) | **0.707** | **0.973** |
| +DDR @ 0.80 (spec 0.931) | 0.639 | 0.918 |

So the higher headline sensitivity is the operating point, not better referral
discrimination — referable AUC barely moved, 0.9082 → 0.9122.

**The mechanism, and the reason this is worth recording.** Discrimination and
calibration moved in *opposite* directions:

* discrimination transferred **better** — the internal→external AUC gap halved,
  0.080 → 0.041, which is exactly what a third imaging domain should buy;
* calibration transferred **worse** — DDR is 75% of the pool, so val is now
  DDR-dominated and both the temperature and the isotonic fit are aimed at a
  score distribution further from Messidor-2 than before.

A better model, worse aimed. Adding a corpus that dominates the pool improves
what the model can distinguish and degrades where the threshold lands, and only
the second of those shows up in a deployed sensitivity/specificity pair.

**Why the clinical arm collapsed** (AUC 0.9272 → 0.8125, QWK 0.6904 → 0.5503):
it consumes lesion features from a segmentation model trained **only on IDRiD**,
now asked to find lesions in a domain it has never seen. The grading pool gained
DDR; the segmentation pool did not. DDR ships **383 lesion-segmentation training
images** against IDRiD's 64, from the same domain as the new grading data — that
is the missing half of this experiment, and it is now the top roadmap item.

Artefacts: `outputs/validation_ddr/` (`max_sensitivity`) and
`outputs/validation_ddr_youden/` (`youden`), with the cohort reproducible via
`build_cohort --source real --curate`.

### 7.2 Fixing the segmentation domain gap — and what it revealed

[§7.1](#71-adding-ddr-what-it-fixed-and-what-it-broke) proposed that the clinical
arm collapsed because its features come from a segmenter trained only on IDRiD,
now asked to read a domain it had never seen. DDR ships 532 lesion-annotated
images (383 train / 149 val) against IDRiD's 81, so the claim was testable.

**The domain gap, measured.** The IDRiD-only checkpoint scores mean Dice
**0.530 on IDRiD but 0.372 on DDR** — a 30% drop across the gap. That was the
hypothesis; it is now a number.

Retraining on both corpora (613 images) closes most of it. On 108 held-out DDR
images, macro Dice:

| class | IDRiD-only | IDRiD+DDR |
|---|---|---|
| microaneurysm | 0.3467 | **0.4036** |
| haemorrhage | 0.3379 | **0.4533** |
| hard exudate | 0.3513 | **0.5282** |
| cotton-wool spot | 0.4511 | **0.6567** |
| **mean** | 0.3718 | **0.5105** |

On the **same 17 IDRiD validation images** it is a wash: 0.5296 → 0.5193 macro,
0.5547 → 0.5637 micro. So 424 extra training images bought a large gain in the
new domain and nothing in the old one. The new run had 60 epochs against the
baseline's 160 and its best epoch was the last, so it is undertrained relative
to what it is compared against.

Two measurement traps were avoided rather than walked into. The val split grew
from 17 to 125 images, so a mixed mean against an IDRiD-only baseline would
credit the model for an easier eval set — the IDRiD subset is scored separately,
and `group_split` hashes per subject, so the 64/17 IDRiD assignment is identical
in both cohorts (verified before training). And `training.py` pools pixels
across the batch (micro-Dice) while a per-image average is macro; the two differ
by 0.05 on the same checkpoint, so `scripts/eval_segmentation.py` reports both
and every checkpoint is scored under each.

**Then the features were rebuilt and the feature-dependent arms retrained.** The
CNN arm is image-only, so it was reused unchanged as a control — and it scored
0.953315 in both runs, identical to six decimal places. Every difference below
is therefore attributable to the features alone.

| arm | old segmenter | new segmenter | Δ |
|---|---|---|---|
| `cnn` *(control)* | 0.9533 | 0.9533 | — |
| **`clinical_only`** | 0.8125 | **0.8965** | **+0.084** |
| **`rule_based`** | 0.7930 | **0.8820** | **+0.089** |
| `fusion` | 0.9535 | 0.9475 | −0.006 |

The two arms that consume *only* lesion features improved substantially, exactly
as predicted: `clinical_only` QWK 0.550 → 0.697, and `rule_based` went from
specificity **0.110 to 0.788** with QWK 0.000 → 0.357 — from referring
essentially everyone to being a genuinely discriminating classical baseline.
Lesion-threshold refitting was part of that: the F1-optimal microaneurysm
cut-point moved 0.6 → 0.9 under the new segmenter, and carrying the old value
over would have massively over-detected.

**The surprise is `fusion`.** Better features made it slightly *worse*, and it
now loses to the image-only arm significantly (DeLong p = 0.0397, against
p = 0.9152 when the features were poorer). Its problem is therefore not feature
quality — give it demonstrably better lesion counts and it does not improve.
That points at the fusion head itself rather than at the data feeding it, which
is a different repair from the one [§7.1](#71-adding-ddr-what-it-fixed-and-what-it-broke)
implied, and it is why the deployed grader remains image-only.

Artefacts: `outputs/validation_ddrseg/`, segmentation in
`outputs/segmentation_ddr/`, cohort `data/cohort_seg1024_ddr`.

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

# --thr-cohort is not optional: it fits the per-class lesion cut-points the LIVE
# pipeline applies and caches features at those same values. Without it caching
# falls back to a blanket 0.5 and the fusion grader learns counts it is never
# served (bug #15).
python scripts/precompute_features.py --cohort data/cohort_real --seg outputs/segmentation/best.pt \
    --size 1024 --feature-size 512 --thr-cohort data/cohort_seg1024

python scripts/train_grader.py --cohort data/cohort_real --arm fusion --epochs 30
python scripts/train_grader.py --cohort data/cohort_real --arm cnn --epochs 30
python scripts/train_grader.py --cohort data/cohort_real --arm clinical --epochs 30

# The deployed arm is cnn, so it is the --grader; fusion becomes a comparison
# arm. --threshold-policy max_sensitivity takes the most sensitive VAL point
# still clearing the specificity floor.
python scripts/validate.py --cohort data/cohort_real --seg-cohort data/cohort_seg1024 \
    --seg outputs/segmentation/best.pt --grader outputs/grader_cnn/best.pt \
    --arms fusion=outputs/grader_fusion/best.pt clinical_only=outputs/grader_clinical/best.pt \
    --threshold-policy max_sensitivity
```

All three arms must be retrained together whenever the loss, the sampler or the
preprocessing changes — an ablation mixing arms trained under different
objectives compares nothing, and a grader-only retrain after a cohort change
pairs a new grader with old images. QWK selection and the stratified sampler are
defaults; `--select-on`, `--no-balanced-sampler` and `--threshold-policy youden`
restore the previous behaviours for comparison.

Artefacts: `outputs/validation/` (metrics, ablation, model card),
`outputs/reports/` (rendered clinical reports), `outputs/artifacts/`
(deployable bundle, now recording `preprocess_mode` and `channel_order` so it
cannot be served with the wrong preprocessing silently), `outputs/logs/` (raw
stdout of every run behind the numbers above). Hardware: RTX 5080 — ~24 min
segmentation, ~21 min features, ~15 min per grader arm, ~15 min to validate;
~2 h end to end from `data/raw`.

---

## 10. What would move the numbers

In descending order of expected value:

1. **A grade-aware decision rule.** Referral is validated; the printed ICDR
   grade is not (see [§4.1](#41-exact-grade-assignment-is-weaker-than-referral)).
   The referral threshold is fitted on val and moved 0.5 → 0.1999, which
   transformed the referral decision. The *grade* boundaries still sit at a
   hard-coded 0.5 that nobody has ever fitted. Selecting per-boundary cut-points
   on the CORN cumulative probabilities, on val, is the cheapest remaining fix
   and needs no new data or retraining — moved to first place because the
   evidence for it is now three checkpoints deep.
2. **Per-site threshold calibration.** A few hundred locally-graded images per
   deployment site. The ranking already transfers (external AUC 0.908); only
   the operating point does not.
3. **Train segmentation on DDR's 383 lesion-annotated images**, alongside
   IDRiD's 64. Promoted to third on direct evidence: adding DDR to the *grading*
   pool without adding it to the *segmentation* pool dropped the clinical arm
   from AUC 0.9272 to 0.8125, because its features come from a segmenter that
   has never seen a DDR image ([§7.1](#71-adding-ddr-what-it-fixed-and-what-it-broke)).
   Six times the annotation, from the domain that now supplies most of the
   grading data. DDR annotates EX/HE/MA/SE and, like IDRiD, **no
   neovascularisation** — so that gap is now confirmed across two corpora rather
   than assumed.
4. **Re-fit calibration on a source-balanced val split.** DDR improved
   discrimination transfer (internal→external AUC gap 0.080 → 0.041) while
   degrading calibration transfer (external specificity 0.922 → 0.753), because
   val became DDR-dominated. Selecting the temperature and isotonic fit on a
   val subsample balanced across sources should recover the operating point
   without giving up the grading gains — the cheapest way to make the DDR model
   deployable.
5. **Download EyePACS.** DDR is done ([§7.1](#71-adding-ddr-what-it-fixed-and-what-it-broke));
   EyePACS still needs its one-time licence acceptance. It would add ~3,200
   grade-3 and ~2,600 grade-4 images, and — unlike DDR — enough grade-3 volume
   to matter, since DDR carries only 236 in total.
6. **Multi-source training** with harmonised grades, to learn a reference
   standard rather than one panel's habits — the root cause in [§3](#3-why-moderate-npdr-fails--the-reference-standards-disagree).
7. **Higher grading resolution.** The grader runs at 512; the segmentation
   result suggests 768–1024 would help early disease. Segmentation Dice was
   still improving at the final epoch, so more epochs may also help.
8. **Test-time augmentation and ensembling** — reliable but small gains, and
   they cost latency the edge deployment cannot spare.

---

## 11. Standing limitations

- Trained on Indian cohorts (APTOS, IDRiD); external validation is French
  (Messidor-2). No African, East Asian or Latin American validation.
- Co-pathology is **not** detected. Glaucoma, AMD and retinal vein occlusion
  can co-occur; "no DR" is not a statement that the eye is healthy.
- Neovascularisation has no pixel supervision in any public dataset used here,
  so the model cannot detect it. It is now reported as **"not assessed"** rather
  than as an absent finding, and proliferative DR therefore cannot be excluded
  on lesion evidence — only on the image-level grade.
- The printed ICDR grade is materially less reliable than the referral flag:
  exact recall is 0.362 on severe NPDR and 0.379 on proliferative DR, and three
  of 58 proliferative eyes are *labelled* grade 1 while still being referred.
  Consume the referral flag, not the grade.
- The deployed grader is the **image-only** arm. The clinical-feature branch is
  a significantly weaker discriminator (DeLong p = 0.034) and now serves the
  explanation and rule trail rather than the referral score, so the lesion
  counts shown in a report are evidence *for a reader*, not inputs to the
  decision that flagged the case.
- Three of 110 sight-threatening eyes are missed on Messidor-2 (one grade 3,
  two grade 4). External sensitivity on grade ≥ 3 is 0.973, not 1.000. Eleven
  of 35 proliferative eyes there carry a *printed grade* of 0 or 1 despite most
  of them being referred — the grade and the flag disagree, and the flag is the
  validated one.
- The venous-beading cue runs on a morphological vessel proxy and is
  deliberately conservative, so the 4-2-1 rule's "2" arm under-fires.
- Cost figures in `DEFAULT_COSTS` are order-of-magnitude inputs, not findings.

**Decision-support output. Not a diagnosis.** Every referable and every
sight-threatening finding is reviewed by a qualified ophthalmologist before
clinical action.
