"""FastAPI service for the screening pipeline.

Endpoints
---------
``GET  /``               the review console (single-page app)
``GET  /health``         liveness + which artefacts are loaded
``POST /screen``         upload a fundus image, get the full JSON result
``POST /screen/report``  same, but returns the rendered HTML report
``GET  /demo/{grade}``   run a generated phantom of a given grade
``POST /review``         record an ophthalmologist's agree/disagree decision
``GET  /audit``          the review log, for programme-level monitoring
``GET  /simulation``     district telemedicine scenario results
``GET  /optimisation``   cheapest feasible programme configuration
``GET  /validation``     clinical validation report
``GET  /assets``         which generated artefacts are present on disk

The review log is the piece most prototypes omit and every real deployment
needs: a screening AI that is never told when it was wrong cannot be
monitored for drift, and post-market surveillance is a regulatory
requirement, not a nice-to-have.
"""
from __future__ import annotations

import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .constants import ICDR_GRADES
from .explain.report import build_review_panel, render_html
from .pipeline import DRScreeningPipeline, PipelineConfig

app = FastAPI(title="DR Screening", version="1.0.0",
              description="Explainable diabetic retinopathy screening for rural India")

_PIPELINE: DRScreeningPipeline | None = None
_ARTIFACTS_DIR = Path("outputs/artifacts")
_AUDIT_LOG = Path("outputs/audit/reviews.jsonl")
_WEB_DIR = Path(__file__).resolve().parents[2] / "web"
_SIM_DIR = Path("outputs/simulation")
_VAL_DIR = Path("outputs/validation")

if _WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_WEB_DIR)), name="static")


def _load_json(path: Path, how: str) -> dict:
    if not path.exists():
        raise HTTPException(404, f"{path} not found. Generate it with: {how}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(500, f"{path} is not valid JSON: {exc}")


def _encode_capture(artifacts: dict) -> str | None:
    img = artifacts.get("standardized")
    if img is None:
        img = artifacts.get("raw")
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) > 640:
        scale = 640.0 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 86])
    if not ok:
        return None
    import base64
    return base64.b64encode(buf.tobytes()).decode()


def get_pipeline() -> DRScreeningPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        if _ARTIFACTS_DIR.exists():
            _PIPELINE = DRScreeningPipeline.load(_ARTIFACTS_DIR)
        else:
            # No trained artefacts yet: the pipeline still runs, falling back to
            # the rule-based grader, so the service is never dead on arrival.
            _PIPELINE = DRScreeningPipeline(None, None, PipelineConfig())
    return _PIPELINE


def _decode(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode the uploaded file as an image.")
    return img


@app.get("/health")
def health() -> dict:
    p = get_pipeline()
    return {
        "status": "ok",
        "segmentation_loaded": p.seg is not None,
        "grader_loaded": p.grader is not None,
        "device": str(p.device),
        "referral_threshold": p.cfg.referral_threshold,
        "temperature": p.cfg.temperature,
        "model_version": p.cfg.model_version,
        "artifacts_dir": str(_ARTIFACTS_DIR),
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    f = _WEB_DIR / "index.html"
    if f.exists():
        return f.read_text(encoding="utf-8")
    return "<h1>DR Screening</h1><p>Console not found. POST an image to /screen.</p>"


@app.post("/screen")
async def screen(file: UploadFile = File(...), explain: bool = True) -> JSONResponse:
    img = _decode(await file.read())
    p = get_pipeline()
    result, artifacts = p.run(img, image_id=Path(file.filename or "case").stem,
                              explain=explain)
    payload = result.to_dict()
    payload["capture_jpeg_b64"] = _encode_capture(artifacts)
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
    img = _decode(await file.read())
    p = get_pipeline()
    result, artifacts = p.run(img, image_id=Path(file.filename or "case").stem)
    return render_html(result, artifacts)


@app.get("/demo/{grade}")
def demo(grade: int, severity: float = 0.3, seed: int | None = None,
         report: bool = False, explain: bool = True):
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
    p = get_pipeline()
    result, artifacts = p.run(ph.image, image_id=f"phantom_g{grade}", explain=explain)
    if report:
        return HTMLResponse(render_html(result, artifacts))
    payload = result.to_dict()
    payload["synthetic"] = True
    payload["ground_truth"] = {"grade": ph.grade, "lesion_counts": ph.lesion_counts,
                               "camera": ph.camera, "quality_label": ph.quality_label}
    payload["capture_jpeg_b64"] = _encode_capture(artifacts)
    if not explain:
        return JSONResponse(payload)
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
    _AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_id": image_id, "model_grade": int(model_grade),
        "reviewer_grade": int(reviewer_grade), "reviewer": reviewer,
        "review_seconds": float(seconds), "notes": notes,
        "agreement": "exact" if model_grade == reviewer_grade else
                     ("within_one" if abs(model_grade - reviewer_grade) == 1 else "disagree"),
    }
    with _AUDIT_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    return {"recorded": True, **rec}


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


@app.get("/simulation")
def simulation() -> JSONResponse:
    return JSONResponse(_load_json(
        _SIM_DIR / "scenarios.json",
        "python scripts/run_simulation.py --scenarios"))


@app.get("/optimisation")
def optimisation() -> JSONResponse:
    return JSONResponse(_load_json(
        _SIM_DIR / "optimisation.json",
        "python scripts/run_simulation.py --optimise"))


@app.get("/validation")
def validation() -> JSONResponse:
    return JSONResponse(_load_json(
        _VAL_DIR / "validation.json",
        "python scripts/validate.py --cohort data/cohort_synth "
        "--grader outputs/artifacts/grader.pt --seg outputs/artifacts/segmentation.pt"))


@app.get("/assets")
def assets() -> dict:
    files = {
        "scenarios": _SIM_DIR / "scenarios.json",
        "optimisation": _SIM_DIR / "optimisation.json",
        "validation": _VAL_DIR / "validation.json",
        "matlab": Path("matlab/build_dr_screening_model.m"),
    }
    return {k: {"present": v.exists(),
                "path": str(v),
                "bytes": v.stat().st_size if v.exists() else 0}
            for k, v in files.items()}
