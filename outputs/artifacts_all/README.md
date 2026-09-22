# Pooled deployable bundle (all four corpora)

Three files, and you need all three:

| file | what it carries |
|---|---|
| `grader.pt` | EfficientNet-B0 + CORN ordinal head, 4.34 M params, `cnn` arm |
| `segmentation.pt` | Attention U-Net, width 24, 4 supervised lesion classes at 1024 px |
| `pipeline.json` | temperature, isotonic calibrator, referral/urgency thresholds, grade cut-points, channel order |

**The checkpoint alone is not deployable.** `grader.pt` carries no temperature,
no isotonic calibrator, no grade cut-points and no urgency threshold. A consumer
that loads the weights without `pipeline.json` falls back to a 0.5 cut-point that
nothing in this project ever measured, and serves a grade no reported number
describes. Ship the directory, not the file.

## Operating point

| field | value |
|---|---|
| `referral_threshold` | 0.3207 |
| `temperature` | 2.3494 |
| `grade_thresholds` | [0.59, 0.56, 0.33, 0.42] |
| `urgent_threshold` | 0.0722 |
| `defer_band` | [0.1707, 0.4707] |

If you are looking at a bundle showing `referral_threshold: 0.1999` and
`urgent_threshold: 0.1184`, that is the **pre-pool** bundle in `outputs/artifacts/`,
not this one.

## Which bundle should you serve?

There are two in this repository and they are not interchangeable.

| | `outputs/artifacts/` | `outputs/artifacts_all/` (this one) |
|---|---|---|
| grader trained on | APTOS-2019 + IDRiD (2,925 images) | all four pooled (5,941 images) |
| DDR | held out — added, measured, [not promoted](../../RESULTS.md#71-adding-ddr-what-it-fixed-and-what-it-broke) | in training |
| Messidor-2 | held out, blind | in training |
| referable sensitivity | 0.986 internal / **0.707 zero-shot** | 0.915 |
| sight-threatening sensitivity | 1.000 internal / **0.973 zero-shot** | 0.997 |
| referable sensitivity on DDR | 0.567 | **0.851** |
| has a zero-shot number | **yes** | no — no blind cohort remains |

Both bundles share the same `segmentation.pt`, which *is* trained on IDRiD + DDR
pixel masks; only the grader differs.

That DDR row is the practical difference. The pre-pool grader never saw DDR and
misses roughly four in ten referable DDR eyes, so if the deployment population
does not look like APTOS or IDRiD, its zero-shot Messidor-2 number is the
optimistic case rather than the typical one.

On images blind to both models the pooled bundle is better by every measure
(+0.227 referable sensitivity, +0.066 AUC, DeLong p = 2.6e-10). What it cannot
do is produce a generalisation estimate, because after pooling there is no
unseen corpus left to measure one on. Full comparison:
[RESULTS.md §7.5](../../RESULTS.md).

## Loading it

```python
from drscreen.pipeline import DRScreeningPipeline
pipe = DRScreeningPipeline.load("outputs/artifacts_all")
result, artifacts = pipe.run(bgr_image, image_id="case")
```

**The FastAPI service does not use this bundle by default.** `drscreen/api.py`
sets `_ARTIFACTS_DIR = Path("outputs/artifacts")`, so serving the pooled model
through the API means pointing that at this directory. Check `/health`, which
reports `referral_threshold` — if it says 0.1999 you are serving the pre-pool
model.

## Consume the referral flag, not the grade

`referable_probability` and `decision` are what the thresholds were fitted and
validated on. The 5-class `grade` is a secondary output: exact-grade accuracy is
0.727 against 0.901 for the referral decision, and grade 2 in particular sits at
0.571 recall because the reference standards themselves disagree there
([RESULTS.md §3](../../RESULTS.md)). Build the UI around refer / defer /
auto-report.
