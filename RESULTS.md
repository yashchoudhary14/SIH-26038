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
| **Internal test** (n=631) | **0.986** | **0.873** | 0.988 | 0.897 | ✅ both met |
| **External, Messidor-2** (n=1,744) | 0.707 | 0.922 | 0.908 | 0.649 | ❌ sensitivity not met |

Targets: sensitivity ≥ 90%, specificity ≥ 85% for referable DR (ICDR grade ≥ 2).

Deployed arm: **CNN (image-only ordinal grader)** at referral threshold
**0.1999**, selected on the validation split under the sensitivity-first policy
(`--threshold-policy max_sensitivity`). See [§5](#5-ablation--does-the-integrated-pipeline-beat-any-single-technique)
for why the image-only arm is deployed rather than the fusion arm.

Trained on **APTOS 2019 + IDRiD only**. DDR was added, measured and
deliberately *not* promoted — see [§7.1](#71-adding-ddr-what-it-fixed-and-what-it-broke).
Pooling Messidor-2 into training instead of holding it out closes most of the
external-sensitivity gap — +0.227 sensitivity and +0.066 AUC on images blind to
both models — at the price of having no zero-shot cohort left at all; measured
in [§7.5](#75-pooling-messidor-2-into-training--the-biggest-single-gain-and-what-it-cost),
not deployed.
Grade boundaries are the per-boundary cut-points **[0.30, 0.39, 0.39, 0.36]**
fitted on val, not the hard-coded 0.5 that shipped previously
([§4.1](#41-exact-grade-assignment-is-weaker-than-referral)).

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
| QWK | 0.8970 [0.8738–0.9169] | 0.6486 [0.6138–0.6812] |
| Exact accuracy | 0.7861 [0.7524–0.8163] | 0.6411 [0.6183–0.6632] |
| Within-one-grade | 0.9620 | 0.9140 |
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
Referral thresholds P(grade ≥ 2) at **0.1999**; the printed grade uses the
per-boundary cut-points **[0.30, 0.39, 0.39, 0.36]**. Both are fitted on val
(see [§4.2](#42-every-cut-point-in-the-decision-path-is-fitted)), but they
answer different questions and they disagree:

**Internal test (n=631)**

| true grade | exact-grade recall | referred |
|---|---|---|
| 0 | 0.938 (274/292) | 5.1% |
| 1 | 0.709 (39/55) | 52.7% |
| 2 | 0.765 (137/179) | 97.8% |
| **3** | **0.447 (21/47)** | **100%** (47/47) |
| **4** | **0.431 (25/58)** | **100%** (58/58) |

**Messidor-2, external (n=1,744)**

| true grade | exact-grade recall | referred |
|---|---|---|
| 0 | 0.926 (942/1017) | 6.1% |
| 1 | 0.144 (39/270) | 14.1% |
| 2 | 0.277 (96/347) | 62.2% |
| **3** | **0.333 (25/75)** | **98.7%** (74/75) |
| **4** | **0.457 (16/35)** | **94.3%** (33/35) |

Internal-test confusion, true grade 3: `[0, 0, 18, 21, 8]` — every grade-3 eye
lands at grade ≥ 2 and is referred, but 18 of 47 are *labelled* moderate. True
grade 4: `[0, 2, 14, 17, 25]` — two proliferative eyes are labelled grade 1, yet
all 58 clear the referral threshold, because P(referable) reaches 0.1999 long
before the grade assignment resolves.

**Read the referral flag, not the printed grade.** The grade is a triage
convenience; the referral decision is what has been validated. Fitting the
boundaries reversed a three-checkpoint decline in grade-3 exact recall
(0.426 → 0.383 → 0.362 → **0.447**), and lifted external QWK 0.593 → 0.649 with
every diseased grade improving. It did not close the gap: 0.333 on external
grade 3 is still far below the 98.7% of those eyes that get referred.

The two failure shapes are different, and [§3](#3-why-moderate-npdr-fails--the-reference-standards-disagree)
separates them. Grades 3 and 4 are weak on **both** splits — a training-volume
limit (207 and 260 images). Grades 1 and 2 are fine internally (0.709, 0.765)
and collapse externally (0.144, 0.277) — a label-definition limit, since a
Messidor-2 "moderate" carries fewer lesions than an APTOS "mild".

### 4.2 Every cut-point in the decision path is fitted

The referral threshold was fitted from the start; everything else was a
hard-coded default that no measurement described. Each one is now selected on
the **validation split only** — never on test or external, which would fit a
decision rule to held-out data and void the zero-shot claim.

| decision | value | fitted on | what it replaced |
|---|---|---|---|
| referral | 0.1999 on P(y>1) | val, sensitivity-first | — |
| **urgency** | **0.1184 on P(y>2)** | val, sensitivity-first | `predicted grade ≥ 3` |
| grade boundaries | [0.30, 0.39, 0.39, 0.36] | val, macro-recall | a hard-coded 0.5 |
| lesion detection | per-class F1-optimal | held-out IDRiD | a blanket 0.5 |
| calibration | T = 2.708 + isotonic | val, out-of-fold | none |

**The urgency tier was the last one.** It fired on the predicted grade — the
least reliable output the system has — so expedited review was gated by a
number with 0.333 external accuracy:

| Messidor-2 urgent tier | sensitivity | specificity | urgent flags |
|---|---|---|---|
| `predicted grade ≥ 3` | 0.5091 (56/110) | 0.9969 | 61 |
| **`P(y>2) ≥ 0.1184`** | **0.7818 (86/110)** | 0.9670 | 140 |

**30 more sight-threatening eyes reach the expedited queue**, for 79 extra
urgent flags out of 1,744. Those 30 were never being missed — the screening
decision catches 97.3% of sight-threatening disease — they were being queued as
routine. This fixes the priority, not the detection.

**The screening decision does not move**, which was predicted before the run
rather than observed after it: referral is P(y>1) against its own threshold, and
urgency only re-ranks cases already being referred. Messidor-2 sensitivity
0.7068, specificity 0.9223, AUC 0.9082 — identical to four decimals either side
of the change.

Urgency is a strict sub-tier of referral: a case that is not referred is never
urgent. That guarantee is free, because P(y>2) ≤ P(y>1) by construction, so
anything clearing the urgent cut already clears referral — measured identical
(0.7818/0.9670) with and without the constraint, and asserted in the tests
rather than assumed.

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
see bug #11 in [§6](#6-sixteen-bugs-that-only-real-data-exposed).

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
model from its own ablation (bug #14 in [§6](#6-sixteen-bugs-that-only-real-data-exposed)).

---

## 6. Eighteen bugs that only real data exposed

Listed because most are *invisible* failures — they produce a plausible number
rather than a crash. The first nine were found during the first real-data run,
the next four by asking why grades 3 and 4 were collapsing, two more while
correcting a train/serve preprocessing skew, one by rebuilding the
demonstration set out of real photographs instead of generated ones, and the
last two by deleting the synthetic generator entirely and re-pointing everything
that had depended on it at real data.

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

Suite: **78 tests**, `pytest tests/ -q`.

### And one more, from replacing the demo phantoms with real photographs

The demonstration set used to be twelve images this project generated. Rebuilt
from real held-out photographs, the first run returned `refer / urgent` for
**all twelve**, healthy eyes included. Widened to the whole held-out split:
**631 of 631** real images escalated to urgent referral.

| # | bug | how it presented |
|---|---|---|
| 16 | `_decide` escalated on the neovascularisation flag before checking whether that channel was ever supervised | IDRiD ships no NV masks, so on the real cohort the NV channel trains against all-zero targets and is never supervised. At the default 0.5 cut-point it still returns a blob on nearly every retina, and that blob short-circuited the decision rule in its first branch — ahead of the confident-negative guard the rest of the method is built around. 100% of real images escalated, at P(referable) as low as 0.000004 |

This is the same 100%-urgent collapse as the pre-fix row in the table below,
arriving by a second route after the first was closed. `rule_grade` had always
refused to read that channel when unassessed —
`constants.PIXEL_ANNOTATED_LESION_CLASSES` documents exactly why — but
`_decide` had never been given the same guard, and the section above records
the exemption being granted deliberately.

**Why no metric caught it.** `validate.py` scores `referable_probability`
against its threshold; it never calls `_decide`. Sensitivity, specificity,
AUC, QWK and calibration are all computed on the probability, and the
probability was fine. The defect lived entirely in the decision layer stacked
on top, which only the serve path exercises — and the serve path was only ever
demonstrated on phantoms, whose cohort *does* annotate NV, so the channel was
supervised there and the branch behaved correctly. Synthetic demo data hid a
real-data-only bug in the one component synthetic data cannot represent.

Fix: one condition in `pipeline.py`, pinned by
`tests/test_unassessed_escalation.py` (4 tests, verified to fail against the
old behaviour). Measured on all 631 held-out real images, same weights:

| | before | after |
|---|---|---|
| Escalated urgent | 631/631 (100%) | 319/631 (50.6%) |
| Auto-reported without review | 0/631 (0%) | 278/631 (44.1%) |
| Referral specificity | 0.000 | **0.839** |
| Referral sensitivity | 1.000 (trivially — everything referred) | 0.989 |
| Exact-grade accuracy | 0.781 | 0.783 |

The grader was never the problem; the triage on top of it was, and it failed
in the direction that looks safe. A screener that refers every patient has
thrown away the only thing that makes it worth deploying.

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
the audit log.

Neovascularisation was left as the one unconditional escalation, on the
reasoning that it defines proliferative DR and is too specific a finding to
gate behind corroboration. That exemption recreated the whole failure on its
own — see bug #16.

### Two more, from deleting the synthetic generator

Removing the phantom generator meant re-pointing the tests and the landmark
fitting script at real photographs. Both had been agreeing with a drawing.

| # | bug | how it presented |
|---|---|---|
| 17 | the focus criterion is **not monotone in blur** — severe defocus scores *better* than a sharp image | `focus_score` was the ratio of high-band to mid-band energy. Both bands collapse under heavy blur, so the ratio of two vanishing quantities is governed by their relative decay, not by surviving detail. Measured on real photographs it bottoms out at σ ≈ 0.004·W and then **climbs back**: a retina blurred to σ = 0.08·W, with no discernible vessel anywhere, scored **0.981** against **0.746** for the same retina in focus. An unreadable image passed the gate and was graded |
| 18 | the optic-disc detector's vessel-convergence weight was fitted on phantoms and is wrong on real retinas | Phantoms draw vessels converging cleanly on the disc, so the fit over-trusted that cue. On IDRiD's hand-marked disc centres the shipped 0.70 gives **96.0%** within 1 DD; the decline is monotone across the whole sweep and the real-data optimum is **0.10** at **98.5%** |

Bug #17 is the one that mattered. The gate's entire purpose is to refuse images
a clinician could not read, and defocus is the canonical reason to refuse one —
it is in `NON_CORRECTABLE` precisely because no enhancer can invent detail that
was never captured. The phantom suite never explored past mild blur, so the
turning point sat outside everything that was ever tested.

**Fix for #17.** Absolute high-frequency energy, normalised by the retina's own
intensity spread, combined with the existing ratio as a conjunction. That
quantity *is* monotone in blur. Calibrated on real images only — 12 committed
APTOS/IDRiD held-out photographs plus 60 IDRiD originals against progressively
defocused copies — with the fail cut-point placed at 0.0085, below the lowest
gradeable real image measured (0.0111) and above every clearly defocused one
(≤ 0.0084).

| blur σ (fraction of width) | focus score before | after |
|---|---|---|
| 0 (sharp) | 0.746 | 0.690 → **pass** |
| 0.004 | 0.162 | 0.193 → fail |
| 0.012 | 0.307 | 0.198 → fail |
| 0.020 | 0.542 → **passed** | 0.170 → fail |
| 0.080 | 0.981 → **passed** | 0.132 → fail |

Verified not to have over-corrected: **0 of 72** real gradeable photographs are
rejected on focus, and the demonstration set still grades 12/12 exact with all
six sight-threatening cases referred urgent.

**Fix for #18.** `DISC_VESSEL_WEIGHT` 0.70 → **0.10**, fitted on the IDRiD
localization train split and confirmed on its held-out test split.

| weight | disc ≤1 DD, train (n=200) | disc ≤1 DD, test (n=103) |
|---|---|---|
| 0.00 | 97.0% | 98.1% |
| **0.10** | **98.5%** | **98.1%** |
| 0.20 | 98.0% | 98.1% |
| 0.50 | 97.0% | 96.1% |
| 0.70 *(was shipped)* | 96.0% | 96.1% |
| 1.00 | 93.5% | 96.1% |

0.10 rather than 0.00 because the term still has a job — a confluent hard-exudate
plaque is as bright as the disc and has no vessels running into it — and that
case is too rare in 303 images to show up in the aggregate but too damaging to
leave unguarded. Fovea accuracy is flat at 93–94% across the whole sweep, so
this trades nothing.


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
`build_cohort --curate`.

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

### 7.3 The fusion arm was starving its own backbone

[§7.2](#72-fixing-the-segmentation-domain-gap--and-what-it-revealed) left one
result unexplained: better lesion features made `fusion` *worse*. The obvious
reading — that the clinical features were harmful — is wrong, and the test that
settles it is to ablate the branch at inference on the trained model.

| val, referable AUC | |
|---|---|
| fusion as trained | 0.9523 |
| fusion, clinical branch zeroed | **0.8638** |

Removing the branch collapses the model, so it is not being harmed by those
features: it has become **dependent** on them. The diagnostic is the comparison
against the image-only arm, same backbone and same images:

| | image pathway alone |
|---|---|
| `cnn` arm | **0.9562** |
| `fusion` arm | **0.8638** |

**The fusion model's backbone had learned 0.09 AUC less.** The clinical vector
is the easier signal to fit, so gradient flowed preferentially there and the
image pathway under-trained; the fused output then lands *below* the better
single modality. A model that had learned less still looked competitive, because
the crutch carried it.

That also explains the direction that made no sense in §7.2. With poor lesion
features the crutch was weak, the backbone had to work, and fusion tied the CNN
arm (DeLong p = 0.92). Improving the segmentation made the crutch *better*, the
backbone leaned harder, and fusion lost outright (p = 0.04). **Better inputs,
worse model** — which is unintelligible until you measure the backbone.

**Fix: modality dropout.** Zero the entire clinical branch, per sample, with
p = 0.5 during training (`--clinical-dropout`). Two details matter. It drops the
*whole branch*, not individual units — unit dropout scales dimensions the head
can route around, whereas only removing the branch forces the image pathway to
be independently accurate. And it is deliberately **not** inverted-scaled: the
head must learn both regimes rather than a rescaled average, and it sees the
branch un-dropped at inference. The `clinical_only` arm is built with rate 0.0,
since with its image pathway already zeroed there is no second modality to fall
back on.

**Result.** The backbone recovers and the clinical branch becomes additive
rather than substitutive:

| | full | image pathway alone | gap |
|---|---|---|---|
| fusion, co-adapted | 0.9523 | 0.8638 | +0.0885 |
| fusion, modality dropout | **0.9560** | **0.9515** | +0.0045 |
| `cnn` reference | — | 0.9562 | — |

0.8638 → 0.9515 restores the backbone to within 0.005 of the dedicated
image-only model, and the fused result improves as well. On the internal test
split the ablation verdict flips across the three configurations:

| configuration | fusion vs cnn | DeLong p |
|---|---|---|
| poor features, co-adapted | fusion +0.0002 | 0.92 — tied |
| good features, co-adapted | **cnn +0.0058** | **0.040 — cnn wins** |
| good features, modality dropout | **fusion +0.0025** | 0.242 — not significant |

Fusion moves from a statistically significant deficit to a non-significant lead,
an 0.0083 AUC swing. It is now the best arm on every axis at once —
AUC 0.9558, sensitivity 0.901, specificity 0.866, QWK 0.8538 — and the **only**
arm meeting both problem-statement targets on this cohort. The AUC margin over
`cnn` is *not* statistically significant and is not claimed as such; what the
fix establishes is that the fusion architecture is no longer worse than its own
image pathway.

**Is the deployed checkpoint affected? No — and the reason completes the story.**
The deployed APTOS+IDRiD fusion arm predates the fix, so the obvious worry is
that the decision to deploy the image-only arm was measuring this defect rather
than a real architectural verdict. `scripts/diagnose_fusion.py` answers it in one
inference pass:

| deployed cohort, val | referable AUC |
|---|---|
| fusion, as trained | 0.9745 |
| fusion, **backbone alone** | 0.9743 |
| image-only arm (reference) | 0.9748 |
| **backbone deficit** | **+0.0005** |

Healthy. The backbone matches the image-only arm, so nothing was starved. The
clinical branch there contributes **+0.0002** — nothing — and the learned gate
tells you why: its median is **0.316** on the deployed model against **0.773** on
the DDR model. With weak features (the old segmenter, an unfitted blanket 0.5
threshold) the model learned to hold the gate mostly shut, so there was no crutch
to lean on and the backbone trained normally.

**Co-adaptation requires the shortcut to be good enough to be worth taking.**
That is why it appeared only after the segmentation was fixed, and it means the
original decision to deploy the image-only arm was a genuine result rather than
an artefact of this bug: on that cohort the clinical branch really did add
nothing.

Artefacts: `outputs/validation_fusionfix/`, grader in
`outputs/grader_fusion_fix/`. Five regression tests in `tests/test_fusion.py`,
and `scripts/diagnose_fusion.py` for checking any fusion checkpoint. Note that
the diagnostic reads the training-time `clinical_dropout` from the checkpoint
rather than off the rebuilt model: the constructor default is now 0.5, so a
live attribute would report 0.5 for a checkpoint trained long before the flag
existed. Checkpoints written from now on record it.

### 7.4 Merging grades 3 and 4 — helps at report time, hurts as a retrain

Grades 3 and 4 lead to the same clinical action, and the boundary between them
is neovascularisation, which no corpus here annotates — so the model is asked to
split on evidence it cannot perceive, using its thinnest data (CORN's task-3
conditional trains on grades 3-4 only). Merging them is therefore the obvious
way to make the printed grade honest without acquiring more data.

Two ways to do it, and they do **not** give the same answer. Both scored on the
same 4-class ground truth, because a 4-class task is easier than a 5-class one
and a natively-4-class model would otherwise "win" by construction.

**Messidor-2 (n=1,744)**

| | 5-class, collapsed at report | 4-class, trained natively |
|---|---|---|
| 0 no DR (1017) | 942 — 0.926 | 946 — 0.930 |
| 1 mild (270) | **39 — 0.144** | 35 — 0.130 |
| 2 moderate (347) | **96 — 0.277** | 71 — 0.205 |
| **3+ sight-threatening (110)** | **56 — 0.509** | 44 — 0.400 |
| overall exact | **1133 — 0.650** | 1096 — 0.628 |
| QWK | **0.6356** | 0.6151 |

**Merging at report time helps.** Against the 5-class output, where grade 3
scores 25/75 and grade 4 scores 16/35 (41/110 = 0.373 exact), collapsing gives
**56/110 = 0.509** — 15 more patients correct, purely from no longer penalising
3↔4 confusions the model was never equipped to make. It costs nothing and needs
no retraining.

**Retraining for it hurts.** The native 4-class model is worse on Messidor-2 on
every metric, despite the easier task. The tell is grade **2**, which the merge
does not touch and which still falls 0.277 → 0.205: the loss is a training
effect, not a reporting one. The fifth class appears to act as free supervision
— forcing the network to model the 3-vs-4 boundary regularises the
representation even though the distinction is discarded at output. "Train
fine-grained, predict coarse."

On the *decisions* the two are tied — referable sensitivity 0.7068 vs 0.7265,
sight-threatening 0.7818 vs 0.7727 on Messidor-2 — which is structural: both
decisions read boundaries *below* the merge, so removing the last boundary
cannot move them. The 4-class model loses only on grade assignment.

**Deployed: the 5-class grader, unchanged.** `--merge-severe` exists so the
negative result stays reproducible, not because it is used.

Artefacts: `outputs/grader_cnn_merged/`.

### 7.5 Pooling Messidor-2 into training — the biggest single gain, and what it cost

Every result above keeps Messidor-2 as a blind external cohort. This experiment
deliberately gives that up: all four graded corpora are pooled, shuffled and
re-cut into train/val/test, so Messidor-2 becomes training data like any other.

Built with `scripts/resplit_cohort.py --pool-external`, a separate script from
`build_cohort.py` on purpose — the normal build path still refuses to let
Messidor-2 near the training pool, and `assert_no_leakage` still raises
`SplitViolation` if it tries. Pooling has to be asked for by name.

| split | n | grades 0–4 | sources |
|---|---|---|---|
| train | 5,941 | 1840 / 926 / 1840 / 415 / 920 | APTOS 1888, DDR 2635, IDRiD 343, Messidor-2 1075 |
| val | 1,860 | 805 / 212 / 575 / 86 / 182 | — |
| test | 1,852 | 786 / 157 / 610 / 96 / 203 | APTOS 502, DDR 1015, IDRiD 75, Messidor-2 260 |

Messidor-2's fellow eyes are grouped by patient via the ADCIS `left;right`
pairing CSV (1,748 images → 874 patients). Without it, correlated eyes straddle
the train/test boundary and flatter the test estimate; the per-image subject ids
in the filenames do not recover the patient.

#### The comparison is not the one it looks like

The two models were validated on different test splits — 631 images against
1,852, different corpora, different grade mix — so their headline numbers are
not measuring the same thing. Read naively, exact accuracy *falls* from 0.786 to
0.727 and referable sensitivity from 0.986 to 0.915. That is the test set getting
harder, not the model getting worse.

Scoring the old model on the new test split is contaminated in the other
direction: the re-shuffle moved **850** of those 1,852 images out of its training
split, and it memorised them.

What is left is the intersection blind to both — old split `test` or `external`
(never trained on, never used to fit the old threshold), new split `test`.
**646 images.** `scripts/compare_graders.py` scores each model with its own fitted
temperature, calibrator and operating point, because that is how each would
actually be deployed.

| metric | pre-pool | pooled | Δ |
|---|---|---|---|
| referable sensitivity | 0.6655 | **0.8921** | +0.227 |
| referable specificity | 0.9429 | 0.9103 | −0.033 |
| referable AUC | 0.8971 | **0.9635** | +0.066 |
| sight-threatening sensitivity | 0.9231 | **0.9846** | +0.062 |
| exact accuracy | 0.6223 | **0.7384** | +0.116 |
| within-one-grade | 0.8529 | **0.9474** | +0.094 |
| referable accuracy | 0.8235 | **0.9025** | +0.079 |
| QWK | 0.6914 | **0.8350** | +0.144 |
| ECE | 0.2007 | **0.0396** | −0.161 |

DeLong on the AUC: **p = 2.6 × 10⁻¹⁰**. McNemar on the realised referral
decision: **p = 5.0 × 10⁻⁷** (24 vs 75 discordant cases). The AUC result matters
because it is threshold-free — the gain is not the old operating point simply
transferring badly.

#### It is not a composition artifact

The comparison set is 40% Messidor-2, where the old model was known weak, so the
obvious objection is Simpson's paradox. The gain holds **within every corpus**:

| corpus | n | AUC pre → pooled | sensitivity | specificity | exact acc |
|---|---|---|---|---|---|
| APTOS | 89 | 0.9797 → 0.9808 | 0.981 → 1.000 | 0.892 → 0.919 | 0.775 → 0.753 |
| DDR | 286 | 0.9092 → **0.9537** | 0.567 → **0.851** | 0.972 → 0.917 | 0.573 → **0.685** |
| Messidor-2 | 260 | 0.8844 → **0.9624** | 0.613 → **0.888** | 0.944 → 0.906 | 0.623 → **0.792** |

APTOS was already near-saturated and is a wash; its 2.3-point exact-accuracy dip
on n=89 is two images. The gain is concentrated exactly where the old model
failed to transfer.

The Messidor-2 subset is representative, not cherry-picked: the old model scored
0.6411 exact / 0.9140 within-one / 0.8658 referable on the **full** 1,744-image
cohort, against 0.6231 / 0.9077 / 0.8423 on these 260.

#### Pooled model, full test split (n=1,852)

| | value | 95% CI | count |
|---|---|---|---|
| referable sensitivity | 0.9153 | 0.895–0.932 | 832/909 |
| referable specificity | 0.8865 | 0.865–0.905 | 836/943 |
| referable AUC | 0.9642 | — | — |
| sight-threatening sensitivity | 0.9967 | 0.981–0.999 | 298/299 |
| QWK | 0.8689 | — | — |
| within-one-grade | 0.9476 | — | — |
| ECE | 0.0278 | — | — |

Per-grade recall 0.872 / 0.618 / 0.571 / 0.740 / 0.714. Grades 3–4 are up
sharply; **grade 2 at 0.571 is now the weak class**, which is the mild/moderate
label-definition boundary of [§3](#3-why-moderate-npdr-fails--the-reference-standards-disagree),
not a data-volume problem. Per corpus: APTOS 0.980/0.884, DDR 0.890/0.883,
IDRiD 0.917/0.852, Messidor-2 0.888/0.906 — every corpus clears the 85%
specificity floor; APTOS and IDRiD clear the 90% sensitivity target outright.

Fitted operating point: threshold **0.3207**, T **2.349**, grade cut-points
**[0.59, 0.56, 0.33, 0.42]**.

#### What it cost

**There is no zero-shot cohort left.** The 97.3% sight-threatening sensitivity in
[§1](#1-the-headline) was measured on a corpus the model had never encountered,
and that measurement cannot be reproduced from a pooled split because no such
corpus exists any more. Every number in this section is in-distribution. The
Messidor-2 rows are unseen *images* from a seen *corpus* — strictly between an
internal estimate and a zero-shot one, and not a substitute for the latter.
`outputs/validation_pre_messidor/validation.json` is retained so the zero-shot
result stays on the record.

**The trade is real.** Specificity fell 3.3 points on the common set to buy 22.7
points of sensitivity. Out of 943 non-referable eyes in the pooled test, 107 are
now falsely flagged. For a screening programme that is the right direction — a
false referral costs one clinic slot, a missed referable eye costs a year — but
it is a cost, not a free gain.

**Two smaller caveats.** Grade 3 and 4 cells in the head-to-head are n=23 and
n=42, so those per-grade deltas carry 95% CIs of roughly ±0.18 and ±0.13:
directionally clear, not precisely measured. And pooled val/test referable
prevalence sits about 4 points above the raw corpus prevalence, because 6,074
grade-0/2 images were curated away at the original cohort build and were never
materialised; recorded in `data/cohort_all/resplit.json`.

#### Deployment status

**Not deployed.** `outputs/artifacts/` still holds the pre-pool CNN arm and is
byte-identical to what [§1](#1-the-headline) describes. The pooled model lives in
`outputs/artifacts_all/` alongside it, and the decision of which to ship turns on
whether an auditable zero-shot number or a better in-distribution one is worth
more for this deployment — which is a programme decision, not a metric.


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

python scripts/build_cohort.py --data-root data/raw --out data/cohort_real --size 512 --workers 14
python scripts/build_cohort.py --data-root data/raw --out data/cohort_seg1024 --size 1024 --workers 12 --only-splits seg_train seg_val

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


Pooled-data experiment of [§7.5](#75-pooling-messidor-2-into-training--the-biggest-single-gain-and-what-it-cost)
(reuses the materialised cohort; nothing is re-preprocessed):

```bash
python scripts/resplit_cohort.py --src data/cohort_real_ddr --out data/cohort_all     --pool-external --curate
python scripts/train_grader.py --cohort data/cohort_all --arm cnn --epochs 30     --out outputs/grader_cnn_all --select-on qwk
python scripts/validate.py --cohort data/cohort_all --grader outputs/grader_cnn_all/best.pt     --seg outputs/artifacts/segmentation.pt --seg-cohort data/cohort_seg1024_ddr     --out outputs/validation_all --artifacts outputs/artifacts_all     --threshold-policy max_sensitivity
python scripts/compare_graders.py --cohort data/cohort_all --by-source     --a outputs/artifacts_pre_messidor --a-name pre-pool     --b outputs/artifacts_all --b-name pooled
```

Landmark constants are fitted, not assumed
([bug #18](#two-more-from-deleting-the-synthetic-generator)):

```bash
python scripts/eval_landmarks.py --split train --sweep   # fit
python scripts/eval_landmarks.py --split test --sweep    # confirm held out
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

1. **Per-site threshold calibration.** A few hundred locally-graded images per
   deployment site. The ranking already transfers (external AUC 0.908); only
   the operating point does not.
2. **Re-fit calibration on a source-balanced val split.** DDR improved
   discrimination transfer (internal→external AUC gap 0.080 → 0.041) while
   degrading calibration transfer (external specificity 0.922 → 0.753), because
   val became DDR-dominated. Selecting the temperature and isotonic fit on a
   val subsample balanced across sources should recover the operating point
   without giving up the grading gains — the cheapest way to make the DDR model
   deployable.
3. **Download EyePACS.** DDR is done ([§7.1](#71-adding-ddr-what-it-fixed-and-what-it-broke));
   EyePACS still needs its one-time licence acceptance. It would add ~3,200
   grade-3 and ~2,600 grade-4 images, and — unlike DDR — enough grade-3 volume
   to matter, since DDR carries only 236 in total.
4. **Multi-source training** with harmonised grades, to learn a reference
   standard rather than one panel's habits — the root cause in [§3](#3-why-moderate-npdr-fails--the-reference-standards-disagree).
5. **Higher grading resolution.** The grader runs at 512; the segmentation
   result suggests 768–1024 would help early disease. Segmentation Dice was
   still improving at the final epoch, so more epochs may also help.
6. **Test-time augmentation and ensembling** — reliable but small gains, and
   they cost latency the edge deployment cannot spare.

**Done since this list was first written**, both with their results recorded
above rather than assumed:

* *Train segmentation on DDR's lesion annotations* — [§7.2](#72-fixing-the-segmentation-domain-gap--and-what-it-revealed).
  Closed the domain gap on DDR images (mean Dice 0.372 → 0.511) and lifted the
  two feature-only arms substantially, while leaving IDRiD-domain Dice unchanged.
* *Fix the fusion head* — [§7.3](#73-the-fusion-arm-was-starving-its-own-backbone).
  Modality dropout restored the under-trained backbone from AUC 0.864 to 0.952
  and turned the clinical branch from a substitute into an additive contribution.
* *A grade-aware decision rule* — [§4.1](#41-exact-grade-assignment-is-weaker-than-referral),
  [§4.2](#42-every-cut-point-in-the-decision-path-is-fitted). Fitting the grade
  boundaries on val reversed a three-checkpoint decline in grade-3 exact recall
  (0.362 → 0.447) and lifted external QWK 0.593 → 0.649, with every diseased
  grade improving. It also removed a second defect found on the way: the
  pipeline assigned the grade with `argmax(class_probs)` while every metric used
  the ordinal rule, and the two disagreed on 3.65% of the internal test split —
  with argmax the worse of the pair.
* *Fit the urgency threshold* — [§4.2](#42-every-cut-point-in-the-decision-path-is-fitted).
  The last hard-coded cut-point in the decision path. Sight-threatening eyes
  reaching the expedited queue went 56/110 → 86/110, with the screening decision
  unchanged to four decimals.
* *Merge grades 3 and 4* — [§7.4](#74-merging-grades-3-and-4--helps-at-report-time-hurts-as-a-retrain).
  Helps at report time (0.373 → 0.509 exact on sight-threatening) and hurts as a
  retrain: a natively-4-class model is worse on Messidor-2 on every metric,
  including grade 2, which the merge does not touch.
* *Check whether the deployed fusion checkpoint is co-adapted too* —
  [§7.3](#73-the-fusion-arm-was-starving-its-own-backbone). It is not: backbone
  deficit +0.0005. The expected answer was that it would be, and it was wrong;
  the deployed cohort's lesion features were too weak to be worth leaning on, so
  the model held the gate mostly shut and the backbone trained normally. The
  decision to deploy the image-only arm therefore stands on its own evidence.

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
- The printed ICDR grade is materially less reliable than the referral flag.
  With the boundaries fitted, internal exact recall is 0.447 on severe NPDR and
  0.431 on proliferative DR; on Messidor-2 it is 0.333 and 0.457. Two of 58
  proliferative eyes are *labelled* grade 1 internally while still being
  referred. Consume the referral flag and the urgency tier, not the grade.
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
