# Reproducing the results (Python)

Training and validation run in the Python reference implementation (`src/drscreen/`). These commands regenerate every number in [RESULTS.md](../RESULTS.md) from the public corpora. To *run* the screening system you only need MATLAB — see the [main README](../README.md#run-it).

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate       # Linux/macOS: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt && pip install -e .
```

> The `cu128` index is required for RTX 50-series (Blackwell) GPUs. On older
> CUDA or CPU-only machines, install PyTorch from the default index instead.

### On real data (what the reported numbers come from)

Place the corpora under `data/` and unpack them (see `docs/DATASETS.md`):

```bash
python scripts/extract_datasets.py --src data --out data/raw
```

```bash
# grading cohort at 512; lesion cohort at 1024, where microaneurysms survive
python scripts/build_cohort.py --data-root data/raw --out data/cohort_real --size 512 --workers 14
python scripts/build_cohort.py --data-root data/raw --out data/cohort_seg1024 --size 1024 --workers 12 --only-splits seg_train seg_val

python scripts/train_seg.py --cohort data/cohort_seg1024 --epochs 160 --batch-size 3 --size 1024 --pos-weight 12

# --thr-cohort fits the per-class lesion cut-points live inference uses and
# caches features at those same values. Without it, caching falls back to a
# blanket 0.5 and the fusion grader learns counts it is never served.
python scripts/precompute_features.py --cohort data/cohort_real --seg outputs/segmentation/best.pt \
    --size 1024 --feature-size 512 --thr-cohort data/cohort_seg1024

# All three arms: the ablation is meaningless if they were trained under
# different objectives, and the deployed arm is chosen from their comparison.
python scripts/train_grader.py --cohort data/cohort_real --arm fusion --epochs 30
python scripts/train_grader.py --cohort data/cohort_real --arm cnn --epochs 30
python scripts/train_grader.py --cohort data/cohort_real --arm clinical --epochs 30

# cnn is the deployed arm, so it is --grader; fusion becomes a comparison arm.
python scripts/validate.py --cohort data/cohort_real --seg-cohort data/cohort_seg1024 \
    --seg outputs/segmentation/best.pt --grader outputs/grader_cnn/best.pt \
    --arms fusion=outputs/grader_fusion/best.pt clinical_only=outputs/grader_clinical/best.pt \
    --threshold-policy max_sensitivity
```

### Without any downloads

```bash
python scripts/run_demo.py --demo        # screens the 12 committed real photographs
python -m pytest                         # full suite, real images throughout
```

Training needs the real corpora; there is no generated-image substitute.

Then the demo and the console:

```bash
python scripts/run_demo.py --demo
```

The website runs on the **MATLAB backend** (see
[`matlab/README.md`](../matlab/README.md)); it serves this
repository's `web/` folder and the same API:

```matlab
cd matlab
launch_web                                        % open http://localhost:8000
```

The Python backend (`python -m uvicorn drscreen.api:app --port 8000`) serves
the same page and returns the same JSON, if you need it instead.

### The verification set (12 real photographs)

The console's demonstration set is twelve **real** fundus photographs from the
APTOS-2019 and IDRiD held-out test splits — never trained on, never validated
on, never used to fit a threshold. They ship in the repository, so the console
runs on real retinas with no dataset download:

```bash
matlab -batch "cd matlab; launch_web"       # then click any thumbnail
python scripts/build_verification_set.py           # rebuild from data/raw
```

Twelve hand-picked cases are a demonstration, not a measurement — the numbers
that count are the whole-split ones in
[`outputs/verification_set/README.md`](../outputs/verification_set/README.md),
which also documents the 100%-escalation bug rebuilding this set uncovered.

Simulation and the MATLAB bridge:

```bash
python scripts/run_simulation.py --scenarios
python scripts/run_simulation.py --optimise
python scripts/run_simulation.py --export-matlab outputs/simulink_bridge/
```

Tests:

```bash
python -m pytest tests/ -q
```

---

## Using the real datasets

Download and accept the licence for each, then arrange as below:

| dataset | role | source |
|---|---|---|
| APTOS 2019 | train + val | <https://www.kaggle.com/c/aptos2019-blindness-detection> |
| IDRiD | train + val, pixel-level lesions | <https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid> |
| DRIVE | vessel segmentation | <https://drive.grand-challenge.org/> |
| **Messidor-2** | **held-out external test** | <https://www.adcis.net/en/third-party/messidor2/> |

```
data/raw/
  aptos2019/     train.csv, train_images/
  idrid/         A. Segmentation/, B. Disease Grading/
  drive/         training/, test/
  messidor2/     IMAGES/, messidor_data.csv
```

```bash
python scripts/build_cohort.py --data-root data/raw --out data/cohort_real
# ...then the same train/validate commands, pointed at data/cohort_real
```

The loaders search rather than assume exact paths, since these archives unpack
differently on different systems. Messidor-2 adjudicated grades come from a
separate CSV (Krause et al. 2018); without it the loader still returns images
for inference but metrics are unavailable.
