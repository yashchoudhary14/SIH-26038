# Dataset setup

All four corpora require you to accept their licence terms individually. None
is redistributed with this repository.

| dataset | role in this project | link |
|---|---|---|
| APTOS 2019 | train + internal validation | https://www.kaggle.com/c/aptos2019-blindness-detection |
| IDRiD | train + validation; the only public pixel-level lesion masks | https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid |
| DRIVE | vessel segmentation | https://drive.grand-challenge.org/ |
| Messidor-2 | **held-out external test only** | https://www.adcis.net/en/third-party/messidor2/ |

## Expected layout

```
data/raw/
  aptos2019/
    train.csv                     # id_code, diagnosis
    train_images/*.png
  idrid/
    A. Segmentation/
      1. Original Images/{a. Training Set, b. Testing Set}/
      2. All Segmentation Groundtruths/{a. Training Set, b. Testing Set}/
          1. Microaneurysms/  2. Haemorrhages/  3. Hard Exudates/
          4. Soft Exudates/   5. Optic Disc/
    B. Disease Grading/
      1. Original Images/{a. Training Set, b. Testing Set}/
      2. Groundtruths/*.csv       # Image name, Retinopathy grade, Risk of macular edema
  drive/
    training/{images, 1st_manual, mask}/
    test/{images, 1st_manual, mask}/
  messidor2/
    IMAGES/*.jpg
    messidor_data.csv             # adjudicated grades, see below
```

The loaders in `src/drscreen/data/registry.py` search rather than assume exact
paths, because these archives unpack differently depending on the tool used.
If a dataset is not found, `build_cohort.py --source real` says which ones it
did find so you can see what is missing.

## Messidor-2 grades

The ADCIS distribution ships **images only**. The adjudicated reference
standard (Krause et al., *Ophthalmology* 2018) is a separate CSV with columns
`image_id, adjudicated_dr_grade, adjudicated_dme, adjudicated_gradable`:

  https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades

Place it anywhere under `data/raw/messidor2/`. Without it the loader still
returns images so you can run inference and generate reports, but external
validation metrics will be unavailable.

## Why Messidor-2 is the test set

APTOS and IDRiD are both Indian cohorts; Messidor-2 is French, captured on
different cameras and graded by a different panel. Reporting on it is the only
honest way to claim the sensitivity/specificity targets will survive
deployment on hardware and populations the model has not seen.

The split policy is enforced in code, not by convention:
`registry.assert_no_leakage` raises `SplitViolation` if any Messidor-2 sample
reaches the training pool, and `build_cohort.py` calls it before writing
anything.

## Class imbalance

APTOS is roughly 49% grade 0, 10% grade 1, 27% grade 2, 5% grade 3, 8% grade 4.
Two mechanisms handle this:

- **Effective-number class weighting** (Cui et al., 2019) in the CORN loss,
  which behaves far better than inverse frequency when grade 3 has only a few
  dozen examples.
- **Selection on referable-DR AUC**, not accuracy. Accuracy is maximised by a
  model that never predicts grades 3 and 4.

## Subject grouping

`registry.group_split` hashes a derived subject id so both eyes of one patient
land in the same split. Fellow eyes are highly correlated; letting them
straddle the train/val boundary inflates every metric and is one of the most
common silent bugs in published DR pipelines.
