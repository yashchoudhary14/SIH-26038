# Clinical validation summary

Grader: `outputs\grader_fusion\best.pt` (arm: fusion)

## Targets

- Sensitivity for referable DR (grade >= 2): >= 90%
- Specificity: >= 85%

## Operating point

- Threshold on P(referable): **0.3998** (selected on the validation split only)
- Temperature: **2.4877**
- Rationale: Meets both targets (sens>=90%, spec>=85%); maximises Youden's J among those. PPV/NPV restated at 18.0% deployment prevalence.

## Calibration

| metric | before | after |
|---|---|---|
| ECE | 0.0587 | 0.0153 |
| MCE | 0.2446 | 0.0728 |
| Brier | 0.0745 | 0.0659 |

## Internal held-out test

n = 631

| metric | value | 95% CI |
|---|---|---|
| Sensitivity | 0.9754 | 0.9500-0.9880 |
| Specificity | 0.9049 | 0.8695-0.9315 |
| AUC | 0.9862 | 0.9760-0.9921 |
| QWK | 0.8912 | 0.8619-0.9149 |
| Exact accuracy | 0.8003 | 0.7673-0.8296 |
| Within-one-grade | 0.9620 | 0.9440-0.9743 |

## External validation (zero-shot)

n = 1744. Nothing was fitted on this split.

| metric | value | 95% CI |
|---|---|---|
| Sensitivity | 0.6193 | 0.5739-0.6626 |
| Specificity | 0.9409 | 0.9267-0.9526 |
| AUC | 0.9013 | 0.8833-0.9167 |
| QWK | 0.6029 | 0.5655-0.6364 |

AUC change internal -> external: **-0.0850**

## Ablation

```
arm                  AUC            95% CI    Sens    Spec     QWK  targets
---------------------------------------------------------------------------
fusion *          0.9862 [0.9760,0.9921]   0.975   0.905  0.8912     PASS
cnn_only          0.9859 [0.9751,0.9921]   0.982   0.925  0.8941     PASS
clinical_only     0.9329 [0.9118,0.9492]   0.891   0.830  0.7108     fail
rule_based        0.9139 [0.8890,0.9337]   0.996   0.058  0.0305     fail

* = integrated pipeline (fusion)

The integrated pipeline (fusion) has the highest referable-DR AUC of all arms, but the margin over cnn_only is not statistically significant at this sample size.

Paired tests vs the integrated pipeline:
  vs cnn_only         fusion has the higher AUC by 0.0003; the difference is not statistically significant (DeLong p = 0.8574).
  vs clinical_only    fusion has the higher AUC by 0.0533; the difference is statistically significant (DeLong p = 2.293e-09).
  vs rule_based       fusion has the higher AUC by 0.0723; the difference is statistically significant (DeLong p = 5.339e-12).
```
