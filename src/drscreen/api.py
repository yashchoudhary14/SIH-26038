"""FastAPI service for the screening pipeline.

Endpoints
---------
``GET  /``               the screening portal (single-page app)
``GET  /console``        the reviewer / audit console
``GET  /health``         liveness, warm-up state, and which artefacts are loaded
``POST /screen``         upload a fundus image, get the full JSON result
``POST /screen/report``  same, but returns the rendered HTML report
``GET  /demo/{grade}``   run a generated phantom of a given grade
``GET  /samples``        list the bundled labelled test images
``GET  /samples.zip``    download them, so a reviewer with no fundus data can
                         still exercise the service
``POST /review``         record an ophthalmologist's agree/disagree decision
``GET  /audit``          the review log, for programme-level monitoring

The review log is the piece most prototypes omit and every real deployment
needs: a screening AI that is never told when it was wrong cannot be
monitored for drift, and post-market surveillance is a regulatory
requirement, not a nice-to-have.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import threading
import time
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from .constants import ICDR_GRADES
from .explain.report import build_review_panel, render_html
from .pipeline import DRScreeningPipeline, PipelineConfig

# ---------------------------------------------------------------------------
# Paths.
#
# These used to be relative to the process CWD. That works when uvicorn is
# launched from the repo root, and silently falls back to the rule-based
# grader when it is not -- which in a container is the normal case, since the
# CWD is whatever the base image left behind. The two serve identical-looking
# pages, so the failure is invisible. Resolve against the package location
# instead, and let the deployment override via the environment.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _path_env(var: str, default: Path) -> Path:
    raw = os.environ.get(var)
    return Path(raw).expanduser().resolve() if raw else default


_ARTIFACTS_DIR = _path_env("DRSCREEN_ARTIFACTS", _REPO_ROOT / "outputs" / "artifacts")
_SAMPLES_DIR = _path_env("DRSCREEN_SAMPLES",
                         _REPO_ROOT / "outputs" / "verification_set" / "images")
_AUDIT_LOG = _path_env("DRSCREEN_AUDIT_LOG", _REPO_ROOT / "outputs" / "audit" / "reviews.jsonl")
_WEB_DIR = _REPO_ROOT / "web"

#: Largest upload accepted, in bytes. A fundus capture is a few MB; anything
#: far above that is a mistake or an attempt to exhaust a free-tier container.
_MAX_UPLOAD_BYTES = int(os.environ.get("DRSCREEN_MAX_UPLOAD_MB", "20")) * 1024 * 1024

#: Concurrent inferences. One by default: a single screening already saturates
#: the available cores, so admitting a second only makes both slower and
#: doubles peak RSS. Queued callers wait rather than compete.
_MAX_CONCURRENCY = int(os.environ.get("DRSCREEN_CONCURRENCY", "1"))

_PIPELINE: DRScreeningPipeline | None = None
_PIPELINE_LOCK = threading.Lock()
_INFER_SEM: asyncio.Semaphore | None = None
_WARM = {"state": "cold", "seconds": None, "error": None}


def get_pipeline() -> DRScreeningPipeline:
    """Load the trained bundle once, thread-safely.

    The lock matters now that inference runs in a threadpool: without it two
    simultaneous first requests would each build a pipeline and the loser's
    ~36 MB of weights would linger until GC.
    """
    global _PIPELINE
    with _PIPELINE_LOCK:
        if _PIPELINE is None:
            if _ARTIFACTS_DIR.exists():
                _PIPELINE = DRScreeningPipeline.load(_ARTIFACTS_DIR)
            else:
                # No trained artefacts yet: the pipeline still runs, falling
                # back to the rule-based grader, so the service is never dead
                # on arrival.
                _PIPELINE = DRScreeningPipeline(None, None, PipelineConfig())
        return _PIPELINE


def artifacts_present() -> bool:
    """Whether a real trained bundle is on disk (not the rule-based fallback)."""
    return (_ARTIFACTS_DIR / "grader.pt").exists()


def _warm_up() -> None:
    """Load the weights and push one frame through the whole graph.

    torch, timm and cv2 all defer a lot of work to first call. Without this a
    public visitor's first upload pays ~20 s of import and lazy-init on top of
    inference and reasonably concludes the site is broken. Runs on a background
    thread so the port binds immediately -- container platforms treat a slow
    first bind as a failed deploy.
    """
    t0 = time.time()
    try:
        _WARM["state"] = "warming"
        p = get_pipeline()
        probe = np.zeros((512, 512, 3), np.uint8)
        cv2.circle(probe, (256, 256), 240, (60, 40, 30), -1)
        p.run(probe, image_id="warmup", explain=True)
        _WARM.update(state="warm", seconds=round(time.time() - t0, 1), error=None)
    except Exception as exc:  # never let warm-up take the service down
        _WARM.update(state="failed", seconds=round(time.time() - t0, 1), error=str(exc))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _INFER_SEM
    _INFER_SEM = asyncio.Semaphore(_MAX_CONCURRENCY)
    if os.environ.get("DRSCREEN_WARMUP", "0") == "1":
        threading.Thread(target=_warm_up, daemon=True).start()
    yield


app = FastAPI(title="DR Screening", version="1.0.0",
              description="Explainable diabetic retinopathy screening for rural India",
              lifespan=lifespan)

# Same-origin is the normal deployment (the portal is served by this app), so
# this is empty by default. Set DRSCREEN_CORS_ORIGINS when the static page is
# hosted separately -- e.g. the portal on Netlify pointing at this API.
_CORS = [o.strip() for o in os.environ.get("DRSCREEN_CORS_ORIGINS", "").split(",") if o.strip()]
if _CORS:
    app.add_middleware(CORSMiddleware, allow_origins=_CORS,
                       allow_methods=["GET", "POST"], allow_headers=["*"])


async def _infer(img: np.ndarray, image_id: str, explain: bool = True):
    """Run the pipeline off the event loop, one at a time.

    ``DRScreeningPipeline.run`` is several seconds of blocking CPU work. Called
    directly from an async endpoint it stalls the whole loop, so /health stops
    answering while someone is being screened and the platform's health probe
    starts killing the container under exactly the load it is meant to survive.
    """
    assert _INFER_SEM is not None, "lifespan did not run"
    async with _INFER_SEM:
        return await run_in_threadpool(get_pipeline().run, img,
                                       image_id=image_id, explain=explain)


async def _read_capped(file: UploadFile) -> bytes:
    """Read an upload, refusing anything over the cap.

    Read in chunks rather than checking ``file.size`` afterwards: by the time
    a declared size can be verified the bytes are already buffered, which is
    the resource exhaustion the cap exists to prevent.
    """
    buf, total = bytearray(), 0
    while chunk := await file.read(1024 * 1024):
        total += len(chunk)
        if total > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                413, f"Image is larger than the {_MAX_UPLOAD_BYTES // (1024*1024)} MB limit.")
        buf.extend(chunk)
    if not buf:
        raise HTTPException(400, "Empty upload.")
    return bytes(buf)


def _decode(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode the uploaded file as an image.")
    return img


@app.get("/health")
def health() -> dict:
    """Liveness plus, crucially, *which* model is answering.

    ``trained_weights`` is the flag that matters in a deployment review: false
    means the artefacts never made it into the image and every result on the
    portal is coming from the rule-based fallback, which grades but does not
    represent the validated model.
    """
    # Don't force a load from the health probe -- report cold honestly instead,
    # or the probe itself pays the 20 s warm-up and times out.
    if _PIPELINE is None and _WARM["state"] in ("cold", "warming"):
        return {
            "status": "ok",
            "warm": _WARM["state"],
            "trained_weights": artifacts_present(),
            "segmentation_loaded": False,
            "grader_loaded": False,
            "device": "cpu",
            "model_version": None,
            "artifacts_dir": str(_ARTIFACTS_DIR),
        }
    p = get_pipeline()
    return {
        "status": "ok",
        "warm": _WARM["state"],
        "warmup_seconds": _WARM["seconds"],
        "warmup_error": _WARM["error"],
        "trained_weights": artifacts_present(),
        "segmentation_loaded": p.seg is not None,
        "grader_loaded": p.grader is not None,
        "device": str(p.device),
        "referral_threshold": p.cfg.referral_threshold,
        "temperature": p.cfg.temperature,
        "model_version": p.cfg.model_version,
        "artifacts_dir": str(_ARTIFACTS_DIR),
        "samples_available": _sample_files() != [],
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    f = _WEB_DIR / "index.html"
    if f.exists():
        return f.read_text(encoding="utf-8")
    return "<h1>DR Screening</h1><p>Console not found. POST an image to /screen.</p>"


@app.get("/console", response_class=HTMLResponse)
def console() -> str:
    """The original review console (single-page app). The landing page ``/`` is
    the presentation portal; this keeps the reviewer/audit workflow available."""
    f = _WEB_DIR / "console.html"
    if f.exists():
        return f.read_text(encoding="utf-8")
    raise HTTPException(404, "console.html not found")


_SAMPLE_EXT = {".jpg", ".jpeg", ".png"}


def _sample_files() -> list[Path]:
    if not _SAMPLES_DIR.is_dir():
        return []
    return sorted((f for f in _SAMPLES_DIR.iterdir()
                   if f.suffix.lower() in _SAMPLE_EXT and not f.name.startswith(".")),
                  key=lambda f: (len(f.stem), f.stem))


@app.get("/samples")
def samples() -> dict:
    """List the bundled test images.

    A reviewer who opens the portal with no fundus photographs of their own
    cannot evaluate anything. Shipping the labelled set with the service, and
    saying plainly that it is synthetic, is the difference between a demo that
    can be checked and one that has to be taken on trust.
    """
    files = _sample_files()
    return {
        "count": len(files),
        "files": [f.name for f in files],
        "zip_url": "/samples.zip",
        "synthetic": True,
        "note": ("Seed-reproducible synthetic fundus images used as the "
                 "verification set. Expected grades are listed in "
                 "outputs/verification_set/README.md."),
    }


@app.get("/samples.zip")
def samples_zip() -> Response:
    """The test images as one download, so the evaluator gets files to upload."""
    files = _sample_files()
    if not files:
        raise HTTPException(404, "No sample images are bundled with this deployment.")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, arcname=f"retinascope-samples/{f.name}")
        # Ship the label sheet keyed to these filenames. The sibling README.md
        # documents the same cases under their pre-rename names
        # (grade0_case1 ...), which would not help anyone holding image1.jpg.
        labels = _SAMPLES_DIR.parent / "EXPECTED_GRADES.md"
        if labels.exists():
            z.write(labels, arcname="retinascope-samples/EXPECTED_GRADES.md")
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition":
                             'attachment; filename="retinascope-samples.zip"'})


@app.post("/screen")
async def screen(file: UploadFile = File(...), explain: bool = True) -> JSONResponse:
    img = _decode(await _read_capped(file))
    result, artifacts = await _infer(img, Path(file.filename or "case").stem, explain)
    payload = result.to_dict()
    try:
        panel = build_review_panel(result, artifacts)
        ok, buf = cv2.imencode(".jpg", panel, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if ok:
            import base64
            payload["panel_jpeg_b64"] = base64.b64encode(buf.tobytes()).decode()
    except Exception:
        pass
    return JSONResponse(payload)


@app.post("/screen/report", response_class=HTMLResponse)
async def screen_report(file: UploadFile = File(...)) -> str:
    img = _decode(await _read_capped(file))
    result, artifacts = await _infer(img, Path(file.filename or "case").stem)
    return render_html(result, artifacts)


@app.get("/demo/{grade}")
async def demo(grade: int, severity: float = 0.3, seed: int | None = None,
               report: bool = False):
    """Run a generated phantom of the requested ICDR grade.

    Present so the service is demonstrable with no data on disk; it is never
    part of the clinical path and the response says so explicitly.
    """
    if grade not in ICDR_GRADES:
        raise HTTPException(400, f"grade must be one of {list(ICDR_GRADES)}")
    from .data.synthetic import generate
    ph = generate(grade=grade, size=768,
                  seed=seed if seed is not None else int(time.time()) % 100000,
                  severity=float(np.clip(severity, 0, 1)))
    result, artifacts = await _infer(ph.image, f"phantom_g{grade}")
    if report:
        return HTMLResponse(render_html(result, artifacts))
    payload = result.to_dict()
    payload["synthetic"] = True
    payload["ground_truth"] = {"grade": ph.grade, "lesion_counts": ph.lesion_counts,
                               "camera": ph.camera, "quality_label": ph.quality_label}
    try:
        import base64
        panel = build_review_panel(result, artifacts)
        ok, buf = cv2.imencode(".jpg", panel, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if ok:
            payload["panel_jpeg_b64"] = base64.b64encode(buf.tobytes()).decode()
    except Exception:
        pass
    return JSONResponse(payload)


@app.post("/review")
def record_review(image_id: str = Form(...), model_grade: int = Form(...),
                  reviewer_grade: int = Form(...), reviewer: str = Form("unknown"),
                  seconds: float = Form(0.0), notes: str = Form("")) -> dict:
    """Record a human grading decision against a model output.

    This is the drift-monitoring substrate: agreement rate over time, by
    grade and by site, is the earliest signal that a deployed model has
    started to fail on a new camera or a new population.
    """
    try:
        _AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    rec = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_id": image_id, "model_grade": int(model_grade),
        "reviewer_grade": int(reviewer_grade), "reviewer": reviewer,
        "review_seconds": float(seconds), "notes": notes,
        "agreement": "exact" if model_grade == reviewer_grade else
                     ("within_one" if abs(model_grade - reviewer_grade) == 1 else "disagree"),
    }
    # Container filesystems are frequently read-only, and on an ephemeral host
    # this log does not survive a restart anyway. Losing a review line is not a
    # reason to fail the reviewer's request; point a volume at
    # DRSCREEN_AUDIT_LOG when the log needs to be durable.
    try:
        with _AUDIT_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        persisted = True
    except OSError:
        persisted = False
    return {"recorded": True, "persisted": persisted, **rec}


@app.get("/audit")
def audit(limit: int = 500) -> dict:
    if not _AUDIT_LOG.exists():
        return {"n": 0, "reviews": [], "summary": {}}
    rows = [json.loads(l) for l in _AUDIT_LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = rows[-limit:]
    n = len(rows)
    if n == 0:
        return {"n": 0, "reviews": [], "summary": {}}
    exact = sum(r["agreement"] == "exact" for r in rows)
    within = sum(r["agreement"] in ("exact", "within_one") for r in rows)
    times = [r["review_seconds"] for r in rows if r["review_seconds"] > 0]
    return {
        "n": n, "reviews": rows[-50:],
        "summary": {
            "exact_agreement": exact / n,
            "within_one_grade": within / n,
            "median_review_seconds": float(np.median(times)) if times else None,
            "under_30s_fraction": float(np.mean([t <= 30 for t in times])) if times else None,
        },
    }
