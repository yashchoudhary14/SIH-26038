# Clinical validation summary

Grader: `outputs\grader_cnn_ddr\best.pt` (arm: cnn)

## Targets

- Sensitivity for referable DR (grade >= 2): >= 90%
- Specificity: >= 85%

## Operating point

- Threshold on P(referable): **0.4625** (selected on the validation split only)
- Temperature: **2.1716**
- Rationale: Meets both targets (sens>=90%, spec>=85%); maximises Youden's J among those. PPV/NPV restated at 18.0% deployment prevalence.

## Calibration

| metric | before | after |
|---|---|---|
| ECE | 0.0630 | 0.0153 |
| MCE | 0.2192 | 0.0362 |
| Brier | 0.0891 | 0.0781 |

## Internal held-out test

n = 2551

| metric | value | 95% CI |
|---|---|---|
| Sensitivity | 0.8902 | 0.8710-0.9069 |
| Specificity | 0.8570 | 0.8376-0.8745 |
| AUC | 0.9533 | 0.9454-0.9601 |
| QWK | 0.8459 | 0.8282-0.8631 |
| Exact accuracy | 0.7381 | 0.7207-0.7548 |
| Within-one-grade | 0.9479 | 0.9385-0.9558 |

## External validation (zero-shot)

n = 1744. Nothing was fitted on this split.

| metric | value | 95% CI |
|---|---|---|
| Sensitivity | 0.8993 | 0.8683-0.9237 |
| Specificity | 0.7529 | 0.7286-0.7757 |
| AUC | 0.9122 | 0.8963-0.9259 |
| QWK | 0.7021 | 0.6720-0.7339 |

AUC change internal -> external: **-0.0411**

## Ablation

```
arm                  AUC            95% CI    Sens    Spec     QWK  targets
---------------------------------------------------------------------------
fusion            0.9558 [0.9481,0.9624]   0.901   0.866  0.8538     PASS
cnn *             0.9533 [0.9454,0.9601]   0.890   0.857  0.8459     fail
clinical_only     0.8965 [0.8839,0.9079]   0.907   0.699  0.6973     fail
rule_based        0.8820 [0.8686,0.8943]   0.805   0.788  0.3574     fail

* = integrated pipeline (cnn)

The integrated pipeline (cnn) does NOT beat every single technique: fusion scored higher. This must be reported as-is.

Paired tests vs the integrated pipeline:
  vs fusion           fusion has the higher AUC by 0.0025; the difference is not statistically significant (DeLong p = 0.242).
  vs clinical_only    cnn has the higher AUC by 0.0568; the difference is statistically significant (DeLong p = 0).
  vs rule_based       cnn has the higher AUC by 0.0713; the difference is statistically significant (DeLong p = 0).
```
