"""FastAPI service for the screening pipeline.

Endpoints
---------
``GET  /``               the review console (single-page app)
``GET  /health``         liveness + which artefacts are loaded
``POST /screen``         upload a fundus image, get the full JSON result
``POST /screen/report``  same, but returns the rendered HTML report
``GET  /cases``          list the real held-out verification photographs
``GET  /cases/{name}``   screen one of them, straight off disk
``GET  /demo/{grade}``   run a generated phantom of a given grade
``POST /review``         record an ophthalmologist's agree/disagree decision
``GET  /audit``          the review log, for programme-level monitoring

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

from .constants import ICDR_GRADES
from .explain.report import build_review_panel, render_html
from .pipeline import DRScreeningPipeline, PipelineConfig

app = FastAPI(title="DR Screening", version="1.0.0",
              description="Explainable diabetic retinopathy screening for rural India")

_PIPELINE: DRScreeningPipeline | None = None
_ARTIFACTS_DIR = Path("outputs/artifacts")
_CASES_DIR = Path("outputs/verification_set")
_AUDIT_LOG = Path("outputs/audit/reviews.jsonl")
_WEB_DIR = Path(__file__).resolve().parents[2] / "web"


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
    raw = await file.read()
    img = _decode(raw)
    p = get_pipeline()
    name = Path(file.filename or "case").stem
    result, artifacts = p.run(img, image_id=name)
    h, w = img.shape[:2]
    return render_html(result, artifacts,
                       provenance=f"Real image attached — uploaded file "
                                  f"'{file.filename or 'case'}', {w}×{h} px.")


# --------------------------------------------------------------------------
# Real held-out cases
# --------------------------------------------------------------------------
# The demo used to be twelve generated phantoms, and every viewer worked that
# out in about a second -- which is the worst possible outcome, because a
# system that only ever demonstrates on images it drew itself gives a reviewer
# no reason to believe any of it. These endpoints serve real photographs from
# the APTOS-2019 and IDRiD held-out test splits instead: never trained on,
# never validated on, never used to fit a threshold. The grade shown is the
# model's own live output on the file, computed on request -- nothing here is
# a stored answer keyed to a filename.
def _load_cases() -> dict:
    f = _CASES_DIR / "verification_summary.json"
    if not f.exists():
        return {}
    try:
        rows = json.loads(f.read_text(encoding="utf-8")).get("cases", [])
    except Exception:
        return {}
    return {r["case"]: r for r in rows}


@app.get("/cases")
def list_cases() -> dict:
    """The real verification photographs available to screen.

    ``true_grade`` is the reference standard shipped with the corpus, not
    anything this project produced; it is returned so the console can show
    the reviewer what the answer should have been.
    """
    cases = _load_cases()
    if not cases:
        return {"n": 0, "cases": [], "note": (
            "No verification set on disk. Build it with "
            "scripts/build_verification_set.py, or use /demo/{grade} for "
            "synthetic phantoms.")}
    out = []
    for name, r in cases.items():
        out.append({
            "case": name,
            "true_grade": r["true_grade"],
            "true_label": r["true_label"],
            "source": r["source"],
            "subject": r.get("subject", ""),
            "provenance": r.get("provenance", ""),
            "thumbnail": f"/cases/{name}/image",
        })
    out.sort(key=lambda c: (c["true_grade"], c["case"]))
    return {"n": len(out), "real_images": True, "cases": out}


def _case_path(name: str) -> Path:
    cases = _load_cases()
    if name not in cases:
        raise HTTPException(404, f"no such case: {name}")
    img = _CASES_DIR / cases[name].get("image", f"images/{name}.jpg")
    if not img.exists():
        raise HTTPException(404, f"image missing on disk for case {name}")
    return img


@app.get("/cases/{name}/image")
def case_image(name: str) -> Response:
    p = _case_path(name)
    media = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    return Response(p.read_bytes(), media_type=media)


@app.get("/cases/{name}")
def screen_case(name: str, report: bool = False):
    """Run the pipeline on one real held-out photograph, live."""
    meta = _load_cases()[name] if name in _load_cases() else {}
    img_path = _case_path(name)
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(500, f"could not decode {img_path}")
    p = get_pipeline()
    result, artifacts = p.run(img, image_id=name)
    prov = meta.get("provenance") or "Real image attached."
    if report:
        return HTMLResponse(render_html(result, artifacts, provenance=prov))
    payload = result.to_dict()
    payload["synthetic"] = False
    payload["provenance"] = prov
    payload["ground_truth"] = {"grade": meta.get("true_grade"),
                               "label": meta.get("true_label"),
                               "source": meta.get("source"),
                               "subject": meta.get("subject"),
                               "reference_standard": "corpus label, held-out split"}
    try:
        import base64
        panel = build_review_panel(result, artifacts)
        ok, buf = cv2.imencode(".jpg", panel, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if ok:
            payload["panel_jpeg_b64"] = base64.b64encode(buf.tobytes()).decode()
    except Exception:
        pass
    return JSONResponse(payload)


@app.get("/demo/{grade}")
def demo(grade: int, severity: float = 0.3, seed: int | None = None,
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
    p = get_pipeline()
    result, artifacts = p.run(ph.image, image_id=f"phantom_g{grade}")
    if report:
        return HTMLResponse(render_html(
            result, artifacts,
            provenance=f"SYNTHETIC PHANTOM — generated, not a photograph "
                       f"(grade {ph.grade}, seed {ph.seed if hasattr(ph, 'seed') else '?'}). "
                       f"Not clinical evidence."))
    payload = result.to_dict()
    payload["synthetic"] = True
    payload["provenance"] = ("SYNTHETIC PHANTOM — generated, not a "
                             "photograph. Not clinical evidence.")
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
