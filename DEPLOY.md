# Deploying RetinaScope so anyone can open it in a browser

This branch (`deploy/public-demo`) turns the project into a public web service:
a reviewer opens a URL, uploads a retinal image, and gets a grade, a referral
decision and a heatmap computed **on the server by the trained model**.

The thing that matters about this branch: the results are real. The container
loads `outputs/artifacts/grader.pt` and `segmentation.pt` and runs
`DRScreeningPipeline.run` on the uploaded pixels. There is no lookup table and
no pre-recorded playback anywhere in this path. If the weights are ever missing
from a deployment, the portal says so in red rather than quietly falling back to
the rule-based grader — that failure is invisible otherwise, because the page
looks identical either way.

> Not to be confused with `outputs/verification_set/RetinaScope_offline.html`,
> which *is* a pre-baked file with fixed results. That file is unchanged and is
> not what gets deployed.

---

## 1. Deploy it (Hugging Face Spaces — free, no credit card)

Spaces is the right host here: the free CPU tier gives 16 GB of RAM, runs a
Dockerfile directly, needs no payment card, and gives a permanent public URL.

**Step 1 — create the Space.** Go to <https://huggingface.co/new-space>.

| field | value |
|-------|-------|
| Space name | `retinascope` |
| License | whatever you prefer |
| SDK | **Docker → Blank** |
| Hardware | **CPU basic** (free) |
| Visibility | **Public** |

**Step 2 — get a write token** at
<https://huggingface.co/settings/tokens> (role: **write**).

**Step 3 — push.** From a clone of this branch:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
./deploy/deploy_hf_space.sh <your-hf-username>/retinascope
```

The first build takes roughly 5–10 minutes (it downloads PyTorch). When it goes
green the portal is live at:

```
https://<your-hf-username>-retinascope.hf.space
```

That is the link to put in the submission. Nothing needs to be installed by
whoever opens it.

`deploy_hf_space.sh` pushes a clean staging copy, not this repo — the project's
git history is ~130 MB of superseded checkpoints and report images that the
running service does not need.

### Cold starts

A free Space sleeps after ~48 hours idle and takes ~30 seconds to wake. The
portal handles this honestly: it polls `/health`, shows "Model starting…" in the
header, and tells the user the first screening may take up to a minute instead
of showing a spinner that looks stuck.

**Before a demo or a review deadline, open the URL once yourself** so the
container is awake.

---

## 2. About Netlify

Netlify can host the page but **cannot run the model** — it serves static files
and short-lived JS functions, with no PyTorch and no 18 MB weight files. A
Netlify-only deployment could only ever replay canned results, which is exactly
what this branch exists to avoid.

If you want the page on Netlify anyway, split it: page on Netlify, model on
Spaces. The code supports this already.

1. Deploy the API to Spaces as above.
2. Add this to `web/index.html`, immediately after `<script>`:
   ```js
   window.RETINASCOPE_API = 'https://<your-hf-username>-retinascope.hf.space';
   ```
3. Start the API with the page's origin allowed:
   `DRSCREEN_CORS_ORIGINS=https://yoursite.netlify.app`
4. Publish `web/index.html` to Netlify.

Simpler is better for a review: one URL, served by the container, is one fewer
thing that can break. Only split if you specifically want a custom domain.

---

## 3. Other hosts

The `Dockerfile` is at the repo root and honours `$PORT`, so it also runs
unchanged on Render, Railway and Fly.io — point them at this branch.

One caveat: **Render's free tier gives 512 MB of RAM**, and this pipeline peaks
well above that (segmentation runs at 1024×1024). It will be OOM-killed on the
free plan. Use Spaces, or a paid instance with ≥2 GB.

To run it locally with Docker:

```bash
docker build -t retinascope .
docker run --rm -p 7860:7860 retinascope
```

Then open <http://localhost:7860>.

Without Docker:

```bash
pip install -r requirements.txt && pip install -e .
DRSCREEN_WARMUP=1 python -m uvicorn drscreen.api:app --port 7860
```

---

## 4. Verify the deployment is real

Do not take the portal's word for it. Point the checker at the live URL:

```bash
python deploy/smoke_test.py https://<your-hf-username>-retinascope.hf.space
```

It uploads all 12 labelled images to the running service, scores the answers
against their known grades, and checks that screening the same image twice
returns the same grade. Standard library only — it runs on any machine with
Python and nothing installed.

It fails loudly if the service is answering from the rule-based fallback
instead of the trained weights.

---

## 5. What a reviewer does

No dataset needs to be handed over separately. The portal has a **"Download the
12 test images"** button, which serves them from the running container along
with `EXPECTED_GRADES.md` listing the correct grade for each.

1. Open the URL.
2. Click **Download the 12 test images**, unzip.
3. Drag any image into the upload box.
4. Read the grade, the referral decision and the heatmap.
5. Check the answer against `EXPECTED_GRADES.md`.

They can also upload their own fundus photograph, or press one of the
**Grade 0–4** buttons to have the server generate a fresh case and screen it.

---

## 6. Measured results

Produced by `deploy/smoke_test.py` against a running instance — reproduce it
yourself rather than trusting the table.

### On the 12 bundled test images

| measure | result |
|---------|--------|
| **Referral decision correct** | **12 / 12** |
| Exact ICDR grade | 7 / 12 |
| Within one grade | 9 / 12 |
| Time per image | ~2–4 s (CPU) |

Every case that needs an ophthalmologist is referred, and every case that does
not is not. Exact grading is weaker, and the misses cluster in grades 3–4:
`image9`, `image11` and `image12` are graded too low, though all three are
still correctly flagged urgent.

These 12 are **synthetic phantoms**, useful for proving the service works, not
for claiming clinical accuracy.

### On real data (from `outputs/validation/summary.md`)

| split | n | sensitivity | specificity | AUC | QWK |
|-------|---|-------------|-------------|-----|-----|
| internal held-out | 631 | 0.975 | 0.905 | 0.986 | 0.891 |
| external, zero-shot | 1744 | 0.619 | 0.941 | 0.901 | 0.603 |

The external drop is real and is the honest headline: sensitivity falls to 0.62
on a corpus nothing was fitted to. Quote the internal numbers with the external
ones beside them.

---

## 7. What changed in the code

Small, contained changes — no model was retrained and no weights were touched.

**`src/drscreen/api.py`** — the serving layer, hardened for a public host:
- Artefact paths resolve against the package, not the process working
  directory, and are overridable by environment variable. Previously a
  container with a different CWD would silently serve the rule-based fallback.
- `/health` reports `trained_weights` and warm-up state, so "is the real model
  answering?" is checkable from outside.
- Model warm-up on startup, on a background thread, so the port binds
  immediately and the first visitor does not pay ~20 s of lazy imports.
- Inference runs in a threadpool behind a semaphore. It used to run on the
  event loop, which blocked `/health` for the duration of every screening —
  under concurrent load the platform's health probe would kill the container.
- `GET /samples` and `GET /samples.zip` serve the labelled test images.
- Upload size cap (20 MB, streamed), CORS support, and an audit log that
  tolerates a read-only filesystem.

**`src/drscreen/models/grader.py`, `src/drscreen/pipeline.py`** — reproducibility:

MC-dropout drew its masks from the unseeded global RNG, so the posterior mean
moved about 1% between calls. On ambiguous cases the top two classes sit inside
that margin, and the reported grade flipped. Measured on `image11`, six
consecutive screenings of the *same file* returned grades 4, 4, 4, 4, 1, 1.

A reviewer who uploads one image twice and gets two different diagnoses will
stop trusting the system, correctly. The masks are now seeded from a hash of
the preprocessed image, so the same image always gives the same answer while
different images still get independent masks and the uncertainty estimate keeps
its meaning. Repeated screening is now bit-identical.

Two consequences, stated plainly:

- The exact-grade score on the 12 test images is **7/12, not the 10/12** recorded
  in `outputs/verification_set/README.md`. That 10/12 was one lucky draw from a
  distribution that also contained 7/12; it was never reproducible. The
  referral decision — the number that matters — is 12/12 either way.
- Raising `mc_samples` does not rescue it. Measured at 8/16/32/64/128 samples,
  exact accuracy went 7, 7, 8, 7, 9 out of 12 while latency rose to 18.5 s per
  image. `image9` and `image12` are wrong at *every* sample count. These are
  genuine model errors, not sampling noise, so the deployment keeps
  `mc_samples=8`.

**`web/index.html`** — the portal, for someone who has never seen the project:
- A "Download the 12 test images" button replaces a hint that pointed at a
  filesystem path only a developer could use.
- Warm-up-aware loading text and error messages that suggest what to do,
  instead of telling a reviewer to run a `uvicorn` command.
- A red header warning if the deployment is missing its trained weights.
- Optional `window.RETINASCOPE_API` for split hosting.

**New files:** `Dockerfile`, `.dockerignore`, `deploy/requirements-serve.txt`
(serving deps only — the training stack roughly doubles the image for code that
never runs), `deploy/deploy_hf_space.sh`, `deploy/smoke_test.py`,
`outputs/verification_set/EXPECTED_GRADES.md`.

---

## 8. What to do next

**Before submitting**

1. Deploy and get the URL (section 1).
2. Run `deploy/smoke_test.py` against it and keep the output — it is evidence.
3. Open the URL on a phone and on someone else's laptop.
4. Wake the Space shortly before anyone is due to look at it.

**If newer weights exist**

Nothing needs to change in the code. The weights are the only input:

1. Copy the new `grader.pt` / `segmentation.pt` / `pipeline.json` over
   `outputs/artifacts/`.
2. Re-run `deploy/deploy_hf_space.sh` — it force-pushes, so this is one command.
3. Re-run `deploy/smoke_test.py` and update the table in section 6 with what
   you actually measure.

`pipeline.json` carries the referral threshold and the calibrator together, and
the threshold is only meaningful on the probability scale its calibrator
produced — so always copy the three files as a set, never `grader.pt` alone.

**Worth fixing next**

- *Grades 3–4 collapse onto their neighbours.* This is the real accuracy gap,
  it predates this branch, and the `fix/sight-threatening-grades` branch already
  exists for it. One cheap thing to try there: the grade is currently
  `argmax(class_probs)`, which throws away the ordinal structure and is exactly
  what makes a near-flat distribution unstable. The pipeline already computes
  `expected_grade` (the probability-weighted mean); rounding that is the
  standard estimator for ordinal targets and is far steadier under sampling
  noise. It would change the published QWK and accuracy figures, so re-run
  `scripts/validate.py` and report the new numbers rather than mixing them.
- *`outputs/verification_set/README.md` is now stale* — it still lists the
  pre-rename `gradeN_caseM` filenames and the unreproducible 10/12. Left alone
  deliberately: rewriting the verification record is a separate decision from
  deploying, and `EXPECTED_GRADES.md` is what the portal actually ships.
- *The audit log is ephemeral* on a free Space — `/review` decisions vanish on
  restart. Fine for a demo; point `DRSCREEN_AUDIT_LOG` at a persistent volume
  before anyone relies on the drift monitoring.
