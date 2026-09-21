# Expected grades — RetinaScope test images

These 12 images ship with the screening portal so the service can be checked
by anyone, including a reviewer with no retinal images of their own.

## What these images are

They are **synthetic fundus phantoms**, not photographs of patients. They are
generated from a seed by `drscreen.data.synthetic.generate`, which is why the
correct answer for each one is known exactly and why they can be redistributed
without any patient-consent or dataset-licence question. They are the right
tool for confirming the service runs and is wired to the trained model. They
are **not** evidence of clinical accuracy — for that, see the validation on
real corpora in `outputs/validation/summary.md` (n = 631 internal, n = 1744
external).

## The labels

| image | true ICDR grade | meaning | needs referral? |
|-------|-----------------|---------|-----------------|
| `image1.jpg`  | 0 | No apparent DR | no |
| `image2.jpg`  | 0 | No apparent DR | no |
| `image3.jpg`  | 0 | No apparent DR | no |
| `image4.jpg`  | 1 | Mild NPDR | no |
| `image5.jpg`  | 1 | Mild NPDR | no |
| `image6.jpg`  | 1 | Mild NPDR | no |
| `image7.jpg`  | 2 | Moderate NPDR | **yes** |
| `image8.jpg`  | 2 | Moderate NPDR | **yes** |
| `image9.jpg`  | 3 | Severe NPDR | **yes** |
| `image10.jpg` | 3 | Severe NPDR | **yes** |
| `image11.jpg` | 4 | Proliferative DR | **yes** |
| `image12.jpg` | 4 | Proliferative DR | **yes** |

"Needs referral" is ICDR grade >= 2, the threshold at which a patient should be
seen by an ophthalmologist. It is the decision the system is actually built to
make.

## What the deployed model scores on these 12

Measured by uploading all 12 to a running instance
(`python deploy/smoke_test.py <url>`):

| measure | result |
|---------|--------|
| **Referral decision correct** | **12 / 12** |
| Exact grade | 7 / 12 |
| Within one grade | 9 / 12 |

Read that honestly: the system gets the **referral call right on every case**,
including all six that need an ophthalmologist and all six that do not. It is
less reliable at naming the exact grade, and the errors are concentrated in
grades 3 and 4 — `image9`, `image11` and `image12` are graded too low even
though all three are still correctly referred as urgent.

This is a known, tracked weakness, not a surprise: severe NPDR (grade 3) is the
hardest class in this dataset and collapses onto its neighbours. Work on it
lives on the `fix/sight-threatening-grades` branch.

## Reproducibility

Screening the same image twice returns the same grade and the same
probability, every time. The MC-dropout masks are seeded from the image
content, so a result can be re-checked and audited rather than re-rolled.
