"""Emit web/case-data.js from the committed verification-set run.

The prototype page replays a *recorded* pipeline run rather than performing
inference in the browser, so every number it shows has to come from the same
artefacts `scripts/run_demo.py` wrote. This script trims those artefacts to
what the page renders and writes them as a single JS module-free global.

    python scripts/build_web_casedata.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VSET = ROOT / "outputs" / "verification_set"
OUT = ROOT / "web" / "case-data.js"

# stage key in timing_ms -> the diagram stage it belongs to
STAGE_MAP = [
    ("geometry", "1"),
    ("quality", "2"),
    ("enhancement", "3"),
    ("quality_recheck", "3"),
    ("landmarks", "4"),
    ("segmentation", "5"),
    ("clinical_features", "6"),
    ("grading", "7b"),
    ("explanation", "9"),
]

GRADE_LABELS = [
    "No apparent DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative DR",
]


def evidence_text(result: dict) -> dict:
    """Split the evidence list into lesion findings, criteria and macula note."""
    findings, criteria, macula = [], [], None
    for item in result.get("evidence", []):
        if "finding" in item:
            findings.append(item)
        elif "criterion" in item:
            criteria.append(item["criterion"])
        elif "macular_assessment" in item:
            macula = item["macular_assessment"]
    return {"findings": findings, "criteria": criteria, "macula": macula}


def build() -> list[dict]:
    summary = json.loads((VSET / "verification_summary.json").read_text(encoding="utf-8"))
    cases = []
    for meta in summary["cases"]:
        name = meta["case"]
        rp = VSET / "reports" / f"{name}_result.json"
        if not rp.exists():
            continue
        r = json.loads(rp.read_text(encoding="utf-8"))

        timings = r.get("timing_ms", {})
        stages = [
            {"stage": sid, "key": key, "ms": round(timings[key], 1)}
            for key, sid in STAGE_MAP
            if key in timings
        ]

        q = r.get("quality", {})
        lm = r.get("landmarks", {})
        cf = r.get("clinical_features", {})

        cases.append({
            "id": name,
            "image": f"../outputs/verification_set/images/{name}.jpg",
            "panel": f"../outputs/verification_set/reports/{name}_panel.png",
            "report": f"../outputs/verification_set/reports/{name}_report.html",
            "source": "APTOS-2019" if meta["source"] == "aptos2019" else "IDRiD",
            "px": meta.get("saved_px"),
            "trueGrade": meta["true_grade"],
            "grade": r.get("grade"),
            "gradeLabel": r.get("grade_label"),
            "classProbs": r.get("class_probabilities", []),
            "referable": r.get("referable"),
            "referableP": round(r.get("referable_probability", 0), 4),
            "sightThreatP": round(r.get("sight_threatening_probability", 0), 4),
            "confidence": round(r.get("confidence", 0), 4),
            "entropy": r.get("uncertainty", {}).get("entropy"),
            "epistemic": r.get("uncertainty", {}).get("epistemic_variance"),
            "decision": r.get("decision"),
            "urgency": r.get("urgency"),
            "dmeRisk": r.get("dme_risk"),
            "quality": {
                "overall": q.get("overall"),
                "gradeable": q.get("gradeable"),
                "confidence": q.get("confidence"),
                "scores": q.get("scores", {}),
                "verdicts": q.get("verdicts", {}),
                "issues": q.get("issues", []),
                "advice": q.get("advice", []),
            },
            "enhancement": r.get("enhancement_applied", []),
            "landmarks": {
                "discConf": lm.get("disc_confidence"),
                "foveaConf": lm.get("fovea_confidence"),
                "laterality": lm.get("laterality"),
                "discDiameterPx": round(lm.get("disc_diameter_px", 0), 1),
            },
            "counts": cf.get("counts", {}),
            "perQuadrant": cf.get("per_quadrant", {}),
            "nearestLesionDD": cf.get("nearest_lesion_dd"),
            "lesionsNearFovea": cf.get("lesions_within_1dd_of_fovea"),
            "quadrantsWithHaem": cf.get("quadrants_with_hemorrhage"),
            "ruleGrade": r.get("rule_based_grade"),
            "ruleReasons": r.get("rule_based_reasons", []),
            "agreement": r.get("agreement"),
            "recapture": r.get("recapture_advice", []),
            "evidence": evidence_text(r),
            "stages": stages,
            "totalMs": round(timings.get("total", meta.get("latency_ms", 0)), 1),
            "modelVersion": r.get("model_version"),
        })
    return cases


def main() -> None:
    cases = build()
    payload = {
        "generated": "scripts/build_web_casedata.py",
        "note": "Recorded output of a real pipeline run over the committed "
                "held-out photographs. The page replays it; it does not infer.",
        "gradeLabels": GRADE_LABELS,
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "/* Generated by scripts/build_web_casedata.py - do not edit by hand. */\n"
        "window.DR_CASES = " + json.dumps(payload, indent=1) + ";\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT.relative_to(ROOT)} - {len(cases)} cases, "
          f"{OUT.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
