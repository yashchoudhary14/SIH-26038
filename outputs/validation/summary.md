# Clinical validation summary

Grader: `outputs\grader_fusion\best.pt` (arm: fusion)

## Targets

- Sensitivity for referable DR (grade >= 2): >= 90%
- Specificity: >= 85%

## Operating point

- Threshold on P(referable): **0.6569** (selected on the validation split only)
- Temperature: **3.7969**
- Rationale: Meets both targets (sens>=90%, spec>=85%); maximises Youden's J among those. PPV/NPV restated at 18.0% deployment prevalence.

## Calibration

| metric | before | after |
|---|---|---|
| ECE | 0.0503 | 0.0130 |
| MCE | 0.2079 | 0.0621 |
| Brier | 0.0705 | 0.0633 |

## Internal held-out test

n = 631

| metric | value | 95% CI |
|---|---|---|
| Sensitivity | 0.9296 | 0.8937-0.9540 |
| Specificity | 0.9395 | 0.9093-0.9601 |
| AUC | 0.9859 | 0.9763-0.9916 |
| QWK | 0.8878 | 0.8646-0.9082 |
| Exact accuracy | 0.7750 | 0.7408-0.8058 |
| Within-one-grade | 0.9556 | 0.9366-0.9691 |

## External validation (zero-shot)

n = 1744. Nothing was fitted on this split.

| metric | value | 95% CI |
|---|---|---|
| Sensitivity | 0.4267 | 0.3821-0.4725 |
| Specificity | 0.9782 | 0.9687-0.9849 |
| AUC | 0.8751 | 0.8548-0.8930 |
| QWK | 0.5859 | 0.5507-0.6178 |

AUC change internal -> external: **-0.1108**

## Ablation

```
arm                  AUC            95% CI    Sens    Spec     QWK  targets
---------------------------------------------------------------------------
fusion *          0.9859 [0.9763,0.9916]   0.930   0.939  0.8878     PASS
cnn_only          0.9857 [0.9752,0.9918]   0.972   0.905  0.8768     PASS
clinical_only     0.9310 [0.9096,0.9476]   0.912   0.807  0.7087     fail
rule_based        0.9139 [0.8890,0.9337]   0.996   0.058  0.0305     fail

* = integrated pipeline (fusion)

The integrated pipeline (fusion) has the highest referable-DR AUC of all arms, but the margin over cnn_only is not statistically significant at this sample size.

Paired tests vs the integrated pipeline:
  vs cnn_only         fusion has the higher AUC by 0.0002; the difference is not statistically significant (DeLong p = 0.9212).
  vs clinical_only    fusion has the higher AUC by 0.0549; the difference is statistically significant (DeLong p = 2.961e-10).
  vs rule_based       fusion has the higher AUC by 0.0720; the difference is statistically significant (DeLong p = 1.307e-12).
```
