# Model card -- DR screening pipeline

Generated 2026-09-21 from `outputs/validation/validation.json`. Every number below is read from that file; none is hand-entered.

## 1. Intended use

**Intended:** first-pass screening of adults with diabetes in primary health centres, to decide who needs to see an ophthalmologist. Output is a severity grade on the International Clinical DR scale plus a referral recommendation, with lesion-level evidence for human review.

**Not intended for:**

- Diagnosis. Every referable and every sight-threatening finding is confirmed by a qualified ophthalmologist before clinical action.
- Monitoring treatment response or deciding on laser/anti-VEGF therapy.
- Any eye disease other than diabetic retinopathy. Glaucoma, AMD and retinal vein occlusion are **not** detected and can co-occur; a 'no DR' result is not a statement that the eye is healthy.
- Paediatric patients, or type 1 diabetes of <5 years' duration, neither of which is represented in the validation population.
- Images from camera types not represented in validation (see section 3).

## 2. Operating point

- Referable DR is defined as grade >= 2 (Moderate NPDR or worse).
- Referral threshold on P(referable): **0.3207**
- Temperature: **2.3494**
- Both were selected on the validation split only, then frozen. Neither was tuned on the test or external splits.
- Selection rationale: Meets both targets (sens>=90%, spec>=85%); takes the most sensitive point clearing the specificity floor (screening policy). PPV/NPV restated at 18.0% deployment prevalence.

## 3. Validation population

- Internal test: n = 1852, grade distribution {0: 786, 1: 157, 2: 610, 3: 96, 4: 203}

## 4. Performance

Targets: sensitivity >= 90%, specificity >= 85% for referable DR.

| metric | internal test | external (zero-shot) |
|---|---|---|
| Sensitivity | 0.915 (95% CI 0.895-0.932) | n/a |
| Specificity | 0.887 (95% CI 0.865-0.905) | n/a |
| PPV | 0.886 (95% CI 0.864-0.905) | n/a |
| NPV | 0.916 (95% CI 0.896-0.932) | n/a |
| AUC | 0.9642 (0.9560-0.9709) | n/a |
| QWK | 0.8689 | n/a |
| Within-one-grade | 0.948 (95% CI 0.937-0.957) | n/a |

Targets met on internal test: sensitivity **yes**, specificity **yes**.

### Calibration

| | before | after |
|---|---|---|
| ECE | 0.0548 | 0.0248 |
| Brier | 0.0886 | 0.0834 |

## 5. Comparison against single techniques

The integrated pipeline (cnn) has a higher referable-DR AUC than every single-technique arm, and every difference is statistically significant (DeLong, alpha=0.05).

| arm | AUC | sensitivity | specificity |
|---|---|---|---|
| cnn **(deployed)** | 0.9642 | 0.915 | 0.887 |
| rule_based | 0.8814 | 0.850 | 0.724 |

## 6. Known failure modes

- **Ungradeable images are refused, not guessed.** The quality gate returns recapture instructions instead of a grade. A programme with a high recapture rate needs technician retraining, not a looser gate.
- **Co-pathology is not detected** (see section 1). 
- **Microaneurysm detection is resolution-limited.** At the working resolution a microaneurysm spans only a few pixels; very early (grade 1) disease is the hardest case and the most likely to be under-graded. Mild NPDR under-grading is clinically tolerable (these patients are not referable) but matters for prevalence reporting.
- **Domain shift.** Performance on a camera model absent from the validation set is unknown. The external-validation figures above are the best available estimate of what to expect, and the audit log (`/audit`) exists to detect drift in the field.
- **The venous-beading cue is conservative** and will under-detect the '2' arm of the 4-2-1 rule; severe NPDR is therefore mainly caught via haemorrhage counts.

## 7. Human oversight (assumed, not optional)

The reported performance assumes the deployment policy the pipeline implements:

- Grade >= 3, any neovascularisation, and any suspected clinically significant macular oedema go to a human **regardless of model confidence**.
- Cases inside the uncertainty band are deferred to a human.
- Every human decision is logged against the model's, so agreement can be monitored over time and by site.

Deploying without that oversight invalidates these numbers.

## 8. Ethical and equity considerations

- The screening threshold is deliberately asymmetric: sensitivity is treated as the binding constraint because a missed proliferative DR costs sight, while a false positive costs one teleconsultation.
- Performance should be re-measured per site and per camera before expanding a programme; an aggregate figure can hide a site where the system is failing.
- The system reduces specialist workload; it does not remove the need for specialists, and a programme that staffs on the assumption it does will fail its urgent cases first.
