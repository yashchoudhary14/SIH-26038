# Showcase set — 60 real fundus photographs

A **curated demonstration set**. These images were selected *because* the model
handles them correctly. It shows the system working; it does not measure how
often the system works.

The measurement is `outputs/validation_all/validation.json` and
[RESULTS.md](../RESULTS.md), computed over all 1,852 held-out images including
the ones the model gets wrong. Quoting a number off this folder would be
circular — the selection criterion was the answer.

## Contents

| grade | ICDR | n | model decision |
|---|---|---|---|
| 0 | No apparent DR | 10 | 10 × auto-report, routine |
| 1 | Mild NPDR | 10 | 4 × auto-report · 6 × **defer to human** |
| 2 | Moderate NPDR | 10 | 8 × refer/soon · 2 × refer/urgent |
| 3 | Severe NPDR | 15 | 15 × refer, **urgent** |
| 4 | Proliferative DR | 15 | 15 × refer, **urgent** |

Served bundle: `outputs/artifacts_all/` (pooled).

| | |
|---|---|
| exact grade match | **60/60** |
| referable (≥2) referred | **40/40** |
| sight-threatening (≥3) referred urgent | **30/30** |
| grade 0–1 wrongly marked urgent | **0** |
| never trained on by the served bundle | **60/60** |
| also blind to `outputs/artifacts/` | 39/60 |

P(referable) separates cleanly: grade 0 spans 0.000–0.009 against 0.774–0.938
for grade 2 and 0.900–1.000 for grades 3–4.

`manifest.json` carries per-image provenance, true grade, predicted grade,
P(referable), decision, urgency and confidence.

### The six deferrals are the point, not a shortfall

Six grade-1 eyes return `defer_to_human` rather than `auto_report`. That is the
selective-referral band doing its job: grade 1 vs 2 is the boundary where the
reference standards themselves disagree — human graders agree exactly only
60–75% of the time, and it is why grade 2 recall sits at 0.571 on the full
split ([RESULTS.md §3](../RESULTS.md)). A screener that answered confidently
there would be overclaiming. None of the six is marked urgent.

Two grade-2 eyes escalate to `urgent`. Moderate NPDR is a genuine referral, so
that is over-escalation in the safe direction rather than an error. Escalation
of grade 0 or 1 *is* an error, and is rejected at selection — there are none.

## How these were chosen

`scripts/build_showcase_set.py`, which is what to re-run rather than editing
this folder by hand.

- **Held out.** Candidates come only from the served bundle's test split, so
  nothing here was fitted on. 39 of 60 are additionally blind to the pre-pool
  bundle, so the set stays valid if the served bundle is switched.
- **Judged on the decision, not just the grade.** Grade and referral are
  separate outputs — the grade from the ordinal cut-points, the referral from
  calibrated P(y≥2) against its own threshold — so an image can carry a correct
  grade and a wrong call. A first pass filtering on grade alone produced nine
  correctly-graded grade-1 eyes returning REFER / URGENT.
- **Stable, not lucky.** Inference uses MC dropout (`mc_samples=8`), so every
  run is a posterior sample and a single call can pass a marginal image. Each
  image here returned the **same grade, decision and urgency on 5 independent
  runs**. Expect the site to reproduce these verdicts; expect small variation in
  the confidence figure, which is a sample statistic.
- **Measured on the file that ships.** Each image is re-encoded to 1280 px JPEG
  and the pipeline re-run on *that* file.
- **Spread across corpora**, round-robin within each grade.

## Provenance and licences

These are third-party research images, redistributed here for demonstration.
**They remain under their original licences and are not this project's to
relicense.** For any use beyond viewing this demo, obtain them from the source
and accept its terms — see [docs/DATASETS.md](../docs/DATASETS.md).

| corpus | n here | source and terms |
|---|---|---|
| DDR | 36 | Li et al., *Information Sciences* 501 (2019) 511–522. Research use. <https://github.com/nkicsl/DDR-dataset> |
| APTOS-2019 | 22 | APTOS 2019 Blindness Detection, Aravind Eye Hospital, India. Kaggle competition terms. <https://www.kaggle.com/c/aptos2019-blindness-detection> |
| IDRiD | 2 | Porwal et al., IEEE Dataport, 2018. CC BY 4.0. <https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid> |

**Messidor-2 is deliberately excluded.** ADCIS distributes it under
registration, which makes it the one corpus here that should not travel in a
committed demonstration set. `build_showcase_set.py` excludes it by default.

Images are downscaled to 1280 px on the longest side and re-encoded as JPEG
quality 95; they are not the original files. Grades are each corpus's own
reference standard, not this project's labels.

If you hold rights to any image here and object to its inclusion, open an issue
and it will be removed.

## Regenerating

```bash
python scripts/build_showcase_set.py
```

Requires the corpora under `data/raw/` and a built `data/cohort_all`. Selection
is stochastic in the MC-dropout sense, so the exact 60 may differ slightly
between runs; the stability and correctness criteria above hold for any run.
