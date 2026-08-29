# Clinical validation summary

Grader: `outputs\grader_fusion\best.pt` (arm: fusion)

## Targets

- Sensitivity for referable DR (grade >= 2): >= 90%
- Specificity: >= 85%

## Operating point

- Threshold on P(referable): **0.2419** (selected on the validation split only)
- Temperature: **0.4453**
- Rationale: Meets both targets (sens>=90%, spec>=85%); maximises Youden's J among those. PPV/NPV restated at 18.0% deployment prevalence.

## Calibration

| metric | before | after |
|---|---|---|
| ECE | 0.0427 | 0.0080 |
| MCE | 0.1749 | 0.0959 |
| Brier | 0.0096 | 0.0030 |

## Internal held-out test

n = 780

| metric | value | 95% CI |
|---|---|---|
| Sensitivity | 1.0000 | 0.9824-1.0000 |
| Specificity | 1.0000 | 0.9933-1.0000 |
| AUC | 1.0000 | 1.0000-1.0000 |
| QWK | 0.9814 | 0.9761-0.9867 |
| Exact accuracy | 0.9513 | 0.9338-0.9643 |
| Within-one-grade | 1.0000 | 0.9951-1.0000 |

## External validation (zero-shot)

n = 720. Nothing was fitted on this split.

| metric | value | 95% CI |
|---|---|---|
| Sensitivity | 0.9668 | 0.9331-0.9838 |
| Specificity | 0.9961 | 0.9858-0.9989 |
| AUC | 0.9990 | 0.9968-0.9997 |
| QWK | 0.9552 | 0.9376-0.9689 |

AUC change internal -> external: **-0.0010**

## Ablation

```
arm                  AUC            95% CI    Sens    Spec     QWK  targets
---------------------------------------------------------------------------
fusion *          1.0000 [1.0000,1.0000]   1.000   1.000  0.9814     PASS
clinical_only     1.0000 [1.0000,1.0000]   1.000   1.000  0.9772     PASS
cnn_only          1.0000 [0.9998,1.0000]   0.995   1.000  0.9673     PASS
rule_based        0.9988 [0.9913,0.9998]   0.995   1.000  0.9589     PASS

* = integrated pipeline (fusion)

The integrated pipeline (fusion) does NOT beat every single technique: clinical_only scored higher. This must be reported as-is.

Paired tests vs the integrated pipeline:
  vs cnn_only         fusion has the higher AUC by 0.0000; the difference is not statistically significant (DeLong p = 0.4141).
  vs clinical_only    clinical_only has the higher AUC by 0.0000; the difference is not statistically significant (DeLong p = 1).
  vs rule_based       fusion has the higher AUC by 0.0012; the difference is not statistically significant (DeLong p = 0.3184).
```
