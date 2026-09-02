# Clinical validation summary

Grader: `outputs\grader_cnn\best.pt` (arm: cnn)

## Targets

- Sensitivity for referable DR (grade >= 2): >= 90%
- Specificity: >= 85%

## Operating point

- Threshold on P(referable): **0.1999** (selected on the validation split only)
- Temperature: **2.7080**
- Rationale: Meets both targets (sens>=90%, spec>=85%); takes the most sensitive point clearing the specificity floor (screening policy). PPV/NPV restated at 18.0% deployment prevalence.

## Calibration

| metric | before | after |
|---|---|---|
| ECE | 0.0500 | 0.0160 |
| MCE | 0.2461 | 0.0731 |
| Brier | 0.0701 | 0.0612 |

## Internal held-out test

n = 631

| metric | value | 95% CI |
|---|---|---|
| Sensitivity | 0.9859 | 0.9644-0.9945 |
| Specificity | 0.8732 | 0.8341-0.9042 |
| AUC | 0.9883 | 0.9790-0.9935 |
| QWK | 0.8939 | 0.8703-0.9149 |
| Exact accuracy | 0.8003 | 0.7673-0.8296 |
| Within-one-grade | 0.9604 | 0.9422-0.9730 |

## External validation (zero-shot)

n = 1744. Nothing was fitted on this split.

| metric | value | 95% CI |
|---|---|---|
| Sensitivity | 0.7068 | 0.6635-0.7467 |
| Specificity | 0.9223 | 0.9064-0.9357 |
| AUC | 0.9082 | 0.8903-0.9234 |
| QWK | 0.5930 | 0.5549-0.6287 |

AUC change internal -> external: **-0.0801**

## Ablation

```
arm                  AUC            95% CI    Sens    Spec     QWK  targets
---------------------------------------------------------------------------
cnn *             0.9883 [0.9790,0.9935]   0.986   0.873  0.8939     PASS
fusion            0.9850 [0.9757,0.9908]   0.989   0.882  0.8817     PASS
clinical_only     0.9272 [0.9058,0.9441]   0.880   0.813  0.6904     fail
rule_based        0.9115 [0.8860,0.9317]   1.000   0.075  0.0000     fail

* = integrated pipeline (cnn)

The integrated pipeline (cnn) has a higher referable-DR AUC than every single-technique arm, and every difference is statistically significant (DeLong, alpha=0.05).

Paired tests vs the integrated pipeline:
  vs fusion           cnn has the higher AUC by 0.0032; the difference is statistically significant (DeLong p = 0.03364).
  vs clinical_only    cnn has the higher AUC by 0.0610; the difference is statistically significant (DeLong p = 1.715e-11).
  vs rule_based       cnn has the higher AUC by 0.0768; the difference is statistically significant (DeLong p = 1.495e-12).
```
