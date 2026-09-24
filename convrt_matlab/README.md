# Diabetic Retinopathy Screening — MATLAB edition

The screening system rebuilt in MATLAB: the same two trained networks, the same
preprocessing, quality gate, calibration and triage rules, and the same two
front ends. The **web console** (the site in `web/`, served with its REST API
by MATLAB) and the **desktop screening console** (a MATLAB app) both run from
this folder with nothing but MATLAB and the Image Processing Toolbox. No
Python and no GPU are needed. The Deep Learning Toolbox is optional; when
installed it makes screening about 3× faster.

The Python repository one level up is untouched. Training stays there; this
folder holds everything needed to *run* the screening system.

> **Status: tested in MATLAB R2026a** (Windows 11, 24-thread CPU). All 54
> tests pass. On all 72 committed photographs, MATLAB and Python agree on
> gradeability, grade, decision and urgency with both model bundles (144 of
> 144). P(referable) agrees to a median of 1e-9, and at worst 4e-4. See
> [Fidelity](#fidelity-to-the-python-pipeline).
>
> On a new machine:
>
> ```matlab
> cd convrt_matlab
> check_install      % which products are present
> run_tests('fast')  % image primitives + both networks, a few seconds
> run_tests          % everything, including all 72 photographs vs Python (~8 min)
> ```

---

## What you need

| | product | used for |
|---|---|---|
| **required** | MATLAB **R2021a or newer** | everything; the web server uses .NET on Windows and the JVM elsewhere, both part of MATLAB |
| **required** | **Image Processing Toolbox** | morphology, connected components, image I/O helpers, `imshow` |
| optional | Deep Learning Toolbox | faster convolutions, used automatically (see below) |
| optional | MATLAB Compiler | building standalone executables and installers (`deploy/`) |
| optional | Simulink + SimEvents | the telemedicine capacity model (`simulation/`) |
| not needed | ONNX converter, Parallel Computing Toolbox, a GPU | — |

The two networks (EfficientNet-B0 grader, attention U-Net segmenter) are
rebuilt from exported weights and run as plain MATLAB matrix code, so the
Deep Learning Toolbox is not required. If it is installed, its `dlconv` runs
the same convolutions 3–7× faster and agrees with the plain code to 1e-7.
`drscreen.nn.useToolbox` switches it on automatically. Set the environment
variable `DRSCREEN_PLAIN=1` to force the plain path.

Time per photograph on a 24-thread desktop CPU (MATLAB R2026a):

| | with Deep Learning Toolbox | plain MATLAB |
|---|---|---|
| whole pipeline | **≈ 2.6 s** | ≈ 8 s |
| of which lesion U-Net (at 1024×1024) | 2.0 s | 7.3 s |
| grader + MC dropout + Grad-CAM++ | 0.2 s | 0.5 s |

MC dropout adds almost nothing, because only the small head is resampled.

## Quick start

```matlab
cd convrt_matlab          % startup.m puts everything on the path
launch_console            % desktop screening console
launch_web                % web console + API at http://127.0.0.1:8000
run_demo                  % screen the 12 verification photographs, write HTML reports
run_validation            % all 72 committed photographs: metrics + agreement with Python
```

Useful variations:

```matlab
launch_console('prepool')                                 % the other model bundle
s = launch_web('Background', true);  ...  s.stop();       % serve while you keep working
launch_web('Host', '0.0.0.0', 'Port', 8080)               % reachable on the LAN (Windows: see urlacl below)
run_demo('Folder', 'data/showcase/grade_3')
run_demo('Image', 'C:\photos\patient17.jpg', 'Bundle', 'pooled')
```

Reports and the reviewer audit log are written to `outputs/` (in a compiled
app, `%LOCALAPPDATA%\DRScreen\`).

## Deployment

The plan is to deploy the website and the screening console, or to ship a
local running setup. All three routes are supported:

**1. Local setup on a machine with MATLAB.** Copy this folder, open MATLAB in
it, and run `launch_console` or `launch_web`. That is all.

**2. Standalone, no MATLAB licence on the target machines.** On a machine with
MATLAB Compiler:

```matlab
build_standalone            % both apps
build_standalone('console') % dist/console/DRScreenConsole.exe + dist/console_installer/DRScreenConsoleInstaller.exe
build_standalone('web')     % dist/web/DRScreenWeb.exe         + dist/web_installer/DRScreenWebInstaller.exe
```

Both apps were built and run with R2026a. Each executable is about 78 MB and
packages the models, the website, the sample photographs and the docs. The
installers download the free **MATLAB Runtime** (same release as the MATLAB
that built them) onto the target machine if it is missing. After that
everything runs offline, which makes this the practical route for a screening
camp.

- `DRScreenConsole.exe [pooled|prepool]` opens the console. It is ready about
  10 s after launch.
- `DRScreenWeb.exe [Port 8080] [Bundle pooled] [Host 0.0.0.0] [OpenBrowser false]`
  opens a small control window and serves until that window is closed. It
  answers `/health` about 15 s after launch.
- Reports, the reviewer audit log and the logs (`logs/console.log`,
  `logs/web_server.log`) go to `%LOCALAPPDATA%\DRScreen\`. A failure at startup,
  such as a port that is already taken, is shown in a dialog and written to
  the log.

The MATLAB Runtime of R2025 and later starts **without a JVM**, so the
compiled web server cannot use Java sockets. On Windows the server uses .NET's
`HttpListener` instead (in desktop MATLAB too), and the Java socket remains
the transport on Linux and macOS desktops. Serving on all interfaces
(`Host 0.0.0.0`) through .NET needs a one-time URL reservation, run as
administrator:

```bat
netsh http add urlacl url=http://+:8080/ user=Everyone
```

**3. Hosting the website.** `web/` is a static site. It finds its backend by
probing its own origin, then `http://127.0.0.1:8000`. Served by `launch_web`,
it talks to MATLAB. The same page also works against the Python FastAPI
backend, since both return identical JSON (only `/health` differs, reporting
`"runtime": "matlab"`). Before putting the MATLAB server on a network:

- It handles **one request at a time**. MATLAB runs on one thread, so a second
  upload waits until the first has been screened. That suits a clinic
  workstation or a LAN, not a public site under load.
- It has **no authentication and no TLS**. It binds to `127.0.0.1` by default.
  To expose it, put it behind a reverse proxy (nginx, Caddy, IIS) that adds
  HTTPS and access control. `POST /review` writes to the audit log, and
  `GET /audit` reads it back.
- Uploads are capped at 25 MB (`MaxBodyBytes`).
- If the port is taken (for example by the Python API on 8000), it says so
  and exits. It never shares the port.

### Endpoints

Identical to the Python API:

| method | path | |
|---|---|---|
| GET | `/` | redirects to the web console |
| GET | `/web/...`, `/outputs/verification_set/...` | static files |
| GET | `/health` | model status, image size, quality thresholds |
| POST | `/screen` | multipart `file` → screening result JSON (with review panel) |
| POST | `/screen/report` | multipart `file` → printable HTML report |
| GET | `/cases`, `/cases/{name}[?report=true]`, `/cases/{name}/image` | the 12 verification photographs |
| GET | `/demo/{grade}` | a real photograph of that grade, screened live |
| POST | `/review` | record a reviewer's grade (audit trail) |
| GET | `/audit[?limit=n]` | the audit trail |

## Model bundles

| bundle | grader trained on | notes |
|---|---|---|
| `pooled` (**default**) | all four corpora pooled | the bundle the showcase set was selected with |
| `prepool` | APTOS-2019 + IDRiD; Messidor-2 held out | its Messidor-2 results are genuine external validation |

Both bundles share the same segmentation network. The default for every
launcher is **one line**, `C.DEFAULT_BUNDLE` in `+drscreen/constants.m`, and
it is `'pooled'`. The 60 showcase photographs were chosen because the pooled
grader handles them well: on them, `pooled` gets 60/60 grades right, while
`prepool` gets 47/60 with 7 referral errors. The 12 verification photographs
are graded correctly by both. Pass `'prepool'` to any launcher to use the
other bundle.

## Layout

```
convrt_matlab/
├── launch_console.m  launch_web.m  run_demo.m  run_validation.m  run_tests.m
├── check_install.m   startup.m
├── +drscreen/                  the package
│   ├── Pipeline.m              load a bundle, screen an image, decide
│   ├── constants.m  paths.m
│   ├── +cv/                    OpenCV-compatible image primitives
│   ├── +preprocess/            field of view, quality gate, enhancement, landmarks
│   ├── +nn/  +models/          network layers; Grader (EfficientNet-B0 + CORN), Segmenter (U-Net), calibration
│   ├── +features/              39-dim clinical vector, rule-based grade, DME risk
│   ├── +report/                lesion overlay, Grad-CAM++ overlay, review panel, HTML report
│   ├── +server/Server.m        HTTP server + REST API
│   ├── +io/  +samples/
├── app/ScreeningConsole.m      the desktop console
├── web/                        the website (unchanged copy)
├── models/                     exported weights: segmentation + prepool/ + pooled/
├── data/verification_set/      12 real photographs with their Python reports
├── data/showcase/              60 real photographs, grades 0–4
├── tests/                      parity tests + fixtures recorded from Python
├── deploy/build_standalone.m   MATLAB Compiler build
├── simulation/                 Simulink/SimEvents capacity model
├── docs/                       RESULTS.md, DATASETS.md (from the Python repo)
└── tools/                      Python: export weights, record fixtures, offline checks
```

### Where each Python module went

| Python (`src/drscreen/`) | MATLAB |
|---|---|
| `pipeline.py` | `+drscreen/Pipeline.m` |
| `constants.py` | `+drscreen/constants.m` |
| `preprocess/fov.py`, `quality.py`, `enhance.py`, `landmarks.py` | `+drscreen/+preprocess/` |
| `models/grader.py`, `segmentation.py`, `calibration.py` | `+drscreen/+models/`, `+drscreen/+nn/` |
| `models/lesion_features.py` | `+drscreen/+features/` |
| `explain/cam.py`, `explain/report.py` | `Grader.gradCamPP`, `+drscreen/+report/` |
| `api.py` | `+drscreen/+server/Server.m` |
| `data/samples.py` | `+drscreen/+samples/` |
| `scripts/run_demo.py`, `validate.py` | `run_demo.m`, `run_validation.m` |
| `sim/simulink_export.py` output | `simulation/` |
| cv2 / numpy calls | `+drscreen/+cv/` |

**Not ported, deliberately:** training (`training.py`, `train_grader.py`,
`train_seg.py`), dataset download/extraction and cohort building
(`data/cohort.py`, `registry.py`, `build_cohort.py` …), evaluation and
ablation studies, and the SimPy model and its optimiser. These produce the
artefacts that the deployed system uses. They are not part of it, they need
PyTorch and a GPU, and porting them would duplicate the source of truth for
the published results. `docs/RESULTS.md` holds those results.

## Fidelity to the Python pipeline

The port reproduces the Python pipeline, not just its general idea:

- **Bit-exact image primitives.** OpenCV's integer greyscale (the 15-bit
  coefficients of OpenCV 5), 8-bit RGB↔Lab in both directions via OpenCV's
  integer lookup tables (checked on all 16.7M values each way), the
  fixed-point 8-bit Gaussian kernel, CLAHE (including its padding quirk),
  Otsu, elliptical morphology, the 5×5 chamfer distance transform, and
  numpy's percentile and row-major argmax. Area resize reproduces OpenCV's
  float32 accumulation order, and every resize uses OpenCV's `1/(dst/src)`
  step. With these, the geometry stage (crop, pad, resize, FOV mask) is
  bit-identical to Python on all 72 photographs.
- **The networks are the trained networks.** The weights are exported from the
  PyTorch checkpoints with BatchNorm folded, and re-run from the exported
  arrays at export time: U-Net identical, grader logits within 4e-6.
- **Same float32 arithmetic where it matters.** Grey-world colour constancy
  sums pixel⁶ over the retina (values up to 1e19) in float32, where the order
  of the additions changes the answer. The port reproduces numpy's sequential
  order, so the output is bit-exact.
- **Same numbers downstream.** Temperature scaling, the isotonic calibrator with
  its tie-break, the CORN ordinal decoding, the referral threshold, and every
  quality-gate and triage constant are read from the same `pipeline.json`.
- **Same result format.** The result struct has the fields of Python's
  `ScreeningResult`, in the same order, so the JSON is interchangeable.

**Known differences:**

| where | difference | when it runs |
|---|---|---|
| denoise | `imnlmfilt` instead of OpenCV `fastNlMeansDenoisingColored` | only when the gate flags sensor noise (none of the 72 committed photographs) |
| illumination normalisation | OpenCV 5's float32 Gaussian blur (σ = 25.6, 205 taps) is reproduced to ~1e-4, not bit for bit; after the division a few dozen pixels per image land one level apart | photographs the gate flags for illumination or exposure (38 of the 72) |
| JPEG decoding | MATLAB's `imread` and OpenCV decoded all 72 committed JPEGs identically; a different encoder could in principle decode differently | every image |
| MC dropout | MATLAB's random stream is not PyTorch's | uncertainty estimates vary run to run in both; parity tests switch MC off |
| network arithmetic | float32 summation order differs from PyTorch's | ~2e-6 relative in activations, ~1e-5 in lesion probabilities |

### Measured agreement

MATLAB against Python (float32 reference, MC dropout off in both), on all 72
committed photographs:

| | `prepool` | `pooled` |
|---|---|---|
| gradeability, grade, decision, urgency identical | **72 / 72** | **72 / 72** |
| every field of the result identical (to 1e-4) | 42 / 72 | 41 / 72 |
| \|ΔP(referable)\| median / max | 1e-9 / 4e-4 | 5e-10 / 3e-7 |
| largest class-probability difference | 3.5e-3 | 1.2e-3 |

The photographs that are not identical in every field are all ones where
illumination normalisation ran (table above). Their lesion areas differ in the
third decimal of a percent, and their class probabilities in the third or
fourth decimal.

### Tests

`run_tests` runs five layers against fixtures recorded from the Python pipeline
(`tools/make_fixtures.py`):

| test | checks | pass criterion |
|---|---|---|
| `TestCv` | every image primitive vs OpenCV output | 8-bit outputs exact (both Lab directions on 262k random colours too); float outputs within float noise; ±1 only for non-integer cubic upscale |
| `TestModels` | both graders and the U-Net on fixed probe tensors, on both convolution paths | every stage within 1e-4 relative; logits within 1e-3 |
| `TestPreprocess` | each stage on 3 real photographs, fed Python's output from the stage before | enhancement exact; lesion maps within 1e-4; activations within 2e-5 relative; Grad-CAM++ within 3e-3 |
| `TestPipelineParity` | the whole pipeline on all 72 photographs, both bundles | ≥ 70/72 identical grade + decision + urgency; median \|ΔP(referable)\| < 1e-3 |
| `TestServer` | every endpoint in-process, plus one real HTTP round trip via `curl` | status codes and JSON |

### Offline checks (no MATLAB needed)

Scripts in `tools/`, run with the Python repository's `.venv`:

- `verify_cv_algorithms.py`: a line-for-line numpy transliteration of each
  `+cv` file, run against real OpenCV. All exact, or within float noise for
  float outputs.
- `verify_nn_math.py`: mirrors the column-major network code (reshapes, layer
  layouts, the analytic Grad-CAM++ gradient) against PyTorch. All agree; the
  analytic gradient matches autograd to 2e-5.
- `lint_matlab.py`: a static check of all `.m` files for block/`end`
  structure, bracket balance, unterminated strings, references to package
  functions that don't exist, and calls with more inputs or outputs than the
  function declares. Clean.

These were written before MATLAB was installed, to catch mistakes early. The
MATLAB test suite is the authority. The reference fixtures are recorded with
TF32 disabled: by default PyTorch runs GPU convolutions in TF32, which alone
moves activations by ~2e-3.

## Updating the models after retraining

Training stays in Python. After a new checkpoint is trained there:

```bash
.venv/Scripts/python.exe convrt_matlab/tools/export_to_matlab.py
.venv/Scripts/python.exe convrt_matlab/tools/make_fixtures.py
```

The first command rewrites `models/` (and checks the export against PyTorch).
The second re-records the parity fixtures. Then run `run_tests` in MATLAB. Both
are run from the Python repository root.

## Data and licences

`data/verification_set/` and `data/showcase/` contain real fundus photographs
from public research datasets (APTOS-2019, IDRiD, DDR; see `docs/DATASETS.md`).
They are included for **demonstration only** and each image stays under its
source dataset's licence. The showcase set was selected for correct model
output, so it demonstrates the system but does not measure it. The measured
sensitivity and specificity are in `docs/RESULTS.md`.

This is decision-support software. Every referable and sight-threatening
finding is meant to be confirmed by a qualified ophthalmologist.
