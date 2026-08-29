# Clinical validation summary

Grader: `outputs\grader_fusion\best.pt` (arm: fusion)

## Targets

- Sensitivity for referable DR (grade >= 2): >= 90%
- Specificity: >= 85%

## Operating point

- Threshold on P(referable): **0.6571** (selected on the validation split only)
- Temperature: **3.7969**
- Rationale: Meets both targets (sens>=90%, spec>=85%); maximises Youden's J among those. PPV/NPV restated at 18.0% deployment prevalence.

## Calibration

| metric | before | after |
|---|---|---|
| ECE | 0.0503 | 0.0172 |
| MCE | 0.2079 | 0.0534 |
| Brier | 0.0705 | 0.0633 |

## Internal held-out test

n = 631

| metric | value | 95% CI |
|---|---|---|
| Sensitivity | 0.9296 | 0.8937-0.9540 |
| Specificity | 0.9395 | 0.9093-0.9601 |
| AUC | 0.9850 | 0.9744-0.9912 |
| QWK | 0.8878 | 0.8646-0.9082 |
| Exact accuracy | 0.7750 | 0.7408-0.8058 |
| Within-one-grade | 0.9556 | 0.9366-0.9691 |

## Ablation

```
arm                  AUC            95% CI    Sens    Spec     QWK  targets
---------------------------------------------------------------------------
cnn_only          0.9853 [0.9752,0.9913]   0.972   0.905  0.8768     PASS
fusion *          0.9850 [0.9744,0.9912]   0.930   0.939  0.8878     PASS
clinical_only     0.9293 [0.9076,0.9463]   0.923   0.795  0.7087     fail
rule_based        0.9139 [0.8890,0.9337]   0.996   0.058  0.0305     fail

* = integrated pipeline (fusion)

The integrated pipeline (fusion) does NOT beat every single technique: cnn_only scored higher. This must be reported as-is.

Paired tests vs the integrated pipeline:
  vs cnn_only         cnn_only has the higher AUC by 0.0003; the difference is not statistically significant (DeLong p = 0.899).
  vs clinical_only    fusion has the higher AUC by 0.0556; the difference is statistically significant (DeLong p = 4.308e-10).
  vs rule_based       fusion has the higher AUC by 0.0710; the difference is statistically significant (DeLong p = 4.459e-12).
```
