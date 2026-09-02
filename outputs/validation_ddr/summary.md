# Clinical validation summary

Grader: `outputs\grader_cnn_ddr\best.pt` (arm: cnn)

## Targets

- Sensitivity for referable DR (grade >= 2): >= 90%
- Specificity: >= 85%

## Operating point

- Threshold on P(referable): **0.3388** (selected on the validation split only)
- Temperature: **2.1716**
- Rationale: Meets both targets (sens>=90%, spec>=85%); takes the most sensitive point clearing the specificity floor (screening policy). PPV/NPV restated at 18.0% deployment prevalence.

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
| Sensitivity | 0.8988 | 0.8802-0.9148 |
| Specificity | 0.8361 | 0.8157-0.8547 |
| AUC | 0.9533 | 0.9454-0.9601 |
| QWK | 0.8459 | 0.8282-0.8631 |
| Exact accuracy | 0.7381 | 0.7207-0.7548 |
| Within-one-grade | 0.9479 | 0.9385-0.9558 |

## External validation (zero-shot)

n = 1744. Nothing was fitted on this split.

| metric | value | 95% CI |
|---|---|---|
| Sensitivity | 0.9081 | 0.8781-0.9313 |
| Specificity | 0.7187 | 0.6935-0.7426 |
| AUC | 0.9122 | 0.8963-0.9259 |
| QWK | 0.7021 | 0.6720-0.7339 |

AUC change internal -> external: **-0.0411**

## Ablation

```
arm                  AUC            95% CI    Sens    Spec     QWK  targets
---------------------------------------------------------------------------
fusion            0.9535 [0.9456,0.9604]   0.906   0.844  0.8439     fail
cnn *             0.9533 [0.9454,0.9601]   0.899   0.836  0.8459     fail
clinical_only     0.8125 [0.7956,0.8282]   0.890   0.477  0.5503     fail
rule_based        0.7930 [0.7752,0.8097]   0.991   0.110  0.0000     fail

* = integrated pipeline (cnn)

The integrated pipeline (cnn) does NOT beat every single technique: fusion scored higher. This must be reported as-is.

Paired tests vs the integrated pipeline:
  vs fusion           fusion has the higher AUC by 0.0002; the difference is not statistically significant (DeLong p = 0.9152).
  vs clinical_only    cnn has the higher AUC by 0.1408; the difference is statistically significant (DeLong p = 0).
  vs rule_based       cnn has the higher AUC by 0.1603; the difference is statistically significant (DeLong p = 0).
```
