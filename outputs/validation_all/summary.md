# Clinical validation summary

Grader: `outputs\grader_cnn_all\best.pt` (arm: cnn)

## Targets

- Sensitivity for referable DR (grade >= 2): >= 90%
- Specificity: >= 85%

## Operating point

- Threshold on P(referable): **0.3207** (selected on the validation split only)
- Temperature: **2.3494**
- Rationale: Meets both targets (sens>=90%, spec>=85%); takes the most sensitive point clearing the specificity floor (screening policy). PPV/NPV restated at 18.0% deployment prevalence.

## Calibration

| metric | before | after |
|---|---|---|
| ECE | 0.0548 | 0.0248 |
| MCE | 0.1913 | 0.0690 |
| Brier | 0.0886 | 0.0834 |

## Internal held-out test

n = 1852

| metric | value | 95% CI |
|---|---|---|
| Sensitivity | 0.9153 | 0.8954-0.9317 |
| Specificity | 0.8865 | 0.8647-0.9052 |
| AUC | 0.9642 | 0.9560-0.9709 |
| QWK | 0.8689 | 0.8528-0.8843 |
| Exact accuracy | 0.7268 | 0.7060-0.7466 |
| Within-one-grade | 0.9476 | 0.9365-0.9569 |

## Ablation

```
arm                  AUC            95% CI    Sens    Spec     QWK  targets
---------------------------------------------------------------------------
cnn *             0.9642 [0.9560,0.9709]   0.915   0.887  0.8667     PASS
rule_based        0.8814 [0.8654,0.8956]   0.850   0.724  0.3380     fail

* = integrated pipeline (cnn)

The integrated pipeline (cnn) has a higher referable-DR AUC than every single-technique arm, and every difference is statistically significant (DeLong, alpha=0.05).

Paired tests vs the integrated pipeline:
  vs rule_based       cnn has the higher AUC by 0.0828; the difference is statistically significant (DeLong p = 0).
```
