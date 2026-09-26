"""Model card generation (Mitchell et al., 2019), specialised for screening AI.

A model card is the document a district health officer, an ethics committee or
a regulator reads *before* deployment. It is generated from the validation
JSON rather than written by hand, so it cannot drift away from the measured
numbers -- which is the failure mode that makes hand-written model cards
worthless.

Sections follow the structure expected for a clinical decision-support tool:
intended use and, equally important, **intended non-use**; the population it
was validated on; performance with intervals; known failure modes; and the
human oversight the system assumes.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from ..constants import (ICDR_GRADES, REFERABLE_THRESHOLD, TARGET_SENSITIVITY,
                         TARGET_SPECIFICITY)


def _fmt_prop(d: dict | None) -> str:
    if not d:
        return "n/a"
    return f"{d['value']:.3f} (95% CI {d['lower']:.3f}-{d['upper']:.3f})"


def generate(validation: dict, extra: dict | None = None) -> str:
    extra = extra or {}
    t = validation.get("internal_test", {})
    e = validation.get("external")
    b = t.get("referable", {})
    op = validation.get("operating_point", {})
    cal = validation.get("calibration", {})
    abl = validation.get("ablation")

    sens_ok = b.get("sensitivity", {}).get("value", 0) >= TARGET_SENSITIVITY
    spec_ok = b.get("specificity", {}).get("value", 0) >= TARGET_SPECIFICITY

    lines = [
        "# Model card -- DR screening pipeline",
        "",
        f"Generated {date.today().isoformat()} from `outputs/validation/validation.json`. "
        "Every number below is read from that file; none is hand-entered.",
        "",
        "## 1. Intended use",
        "",
        "**Intended:** first-pass screening of adults with diabetes in primary "
        "health centres, to decide who needs to see an ophthalmologist. Output "
        "is a severity grade on the International Clinical DR scale plus a "
        "referral recommendation, with lesion-level evidence for human review.",
        "",
        "**Not intended for:**",
        "",
        "- Diagnosis. Every referable and every sight-threatening finding is "
        "confirmed by a qualified ophthalmologist before clinical action.",
        "- Monitoring treatment response or deciding on laser/anti-VEGF therapy.",
        "- Any eye disease other than diabetic retinopathy. Glaucoma, AMD and "
        "retinal vein occlusion are **not** detected and can co-occur; a "
        "'no DR' result is not a statement that the eye is healthy.",
        "- Paediatric patients, or type 1 diabetes of <5 years' duration, "
        "neither of which is represented in the validation population.",
        "- Images from camera types not represented in validation (see section 3).",
        "",
        "## 2. Operating point",
        "",
        f"- Referable DR is defined as grade >= {REFERABLE_THRESHOLD} "
        f"({ICDR_GRADES[REFERABLE_THRESHOLD]} or worse).",
        f"- Referral threshold on P(referable): **{op.get('threshold', float('nan')):.4f}**",
        f"- Temperature: **{cal.get('temperature', float('nan')):.4f}**",
        "- Both were selected on the validation split only, then frozen. "
        "Neither was tuned on the test or external splits.",
        f"- Selection rationale: {op.get('rationale', 'n/a')}",
        "",
        "## 3. Validation population",
        "",
        f"- Internal test: n = {t.get('n', 0)}, grade distribution "
        f"{t.get('grade_distribution', {})}",
    ]
    if e:
        lines.append(f"- External (zero-shot): n = {e.get('n', 0)}, grade distribution "
                     f"{e.get('grade_distribution', {})}")
    if extra.get("data_note"):
        lines += ["", f"> {extra['data_note']}"]

    lines += [
        "",
        "## 4. Performance",
        "",
        f"Targets: sensitivity >= {TARGET_SENSITIVITY:.0%}, "
        f"specificity >= {TARGET_SPECIFICITY:.0%} for referable DR.",
        "",
        "| metric | internal test | external (zero-shot) |",
        "|---|---|---|",
        f"| Sensitivity | {_fmt_prop(b.get('sensitivity'))} | "
        f"{_fmt_prop((e or {}).get('referable', {}).get('sensitivity'))} |",
        f"| Specificity | {_fmt_prop(b.get('specificity'))} | "
        f"{_fmt_prop((e or {}).get('referable', {}).get('specificity'))} |",
        f"| PPV | {_fmt_prop(b.get('ppv'))} | "
        f"{_fmt_prop((e or {}).get('referable', {}).get('ppv'))} |",
        f"| NPV | {_fmt_prop(b.get('npv'))} | "
        f"{_fmt_prop((e or {}).get('referable', {}).get('npv'))} |",
        f"| AUC | {b.get('auc', float('nan')):.4f} "
        f"({b.get('auc_ci', [0,0])[0]:.4f}-{b.get('auc_ci', [0,0])[1]:.4f}) | "
        + (f"{e['referable']['auc']:.4f} "
           f"({e['referable']['auc_ci'][0]:.4f}-{e['referable']['auc_ci'][1]:.4f})"
           if e else "n/a") + " |",
        f"| QWK | {t.get('qwk', {}).get('value', float('nan')):.4f} | "
        + (f"{e['qwk']['value']:.4f}" if e else "n/a") + " |",
        f"| Within-one-grade | {_fmt_prop(t.get('adjacent_accuracy'))} | "
        + (_fmt_prop(e.get('adjacent_accuracy')) if e else "n/a") + " |",
        "",
        f"Targets met on internal test: sensitivity **{'yes' if sens_ok else 'NO'}**, "
        f"specificity **{'yes' if spec_ok else 'NO'}**.",
        "",
        "### Calibration",
        "",
        "| | before | after |",
        "|---|---|---|",
        f"| ECE | {cal.get('before', {}).get('ece', float('nan')):.4f} | "
        f"{cal.get('after', {}).get('ece', float('nan')):.4f} |",
        f"| Brier | {cal.get('before', {}).get('brier', float('nan')):.4f} | "
        f"{cal.get('after', {}).get('brier', float('nan')):.4f} |",
        "",
    ]

    if abl:
        lines += ["## 5. Comparison against single techniques", "",
                  abl.get("verdict", ""), "",
                  "| arm | AUC | sensitivity | specificity |", "|---|---|---|---|"]
        for name, arm in sorted(abl.get("arms", {}).items(),
                                key=lambda kv: -kv[1]["metrics"]["auc"]):
            m = arm["metrics"]
            star = " **(deployed)**" if name == abl.get("reference") else ""
            lines.append(f"| {name}{star} | {m['auc']:.4f} | "
                         f"{m['sensitivity']['value']:.3f} | "
                         f"{m['specificity']['value']:.3f} |")
        lines.append("")

    lines += [
        "## 6. Known failure modes",
        "",
        "- **Ungradeable images are refused, not guessed.** The quality gate "
        "returns recapture instructions instead of a grade. A programme with a "
        "high recapture rate needs technician retraining, not a looser gate.",
        "- **Co-pathology is not detected** (see section 1). ",
        "- **Microaneurysm detection is resolution-limited.** At the working "
        "resolution a microaneurysm spans only a few pixels; very early "
        "(grade 1) disease is the hardest case and the most likely to be "
        "under-graded. Mild NPDR under-grading is clinically tolerable "
        "(these patients are not referable) but matters for prevalence "
        "reporting.",
        "- **Domain shift.** Performance on a camera model absent from the "
        "validation set is unknown. The external-validation figures above are "
        "the best available estimate of what to expect, and the audit log "
        "(`/audit`) exists to detect drift in the field.",
        "- **The venous-beading cue is conservative** and will under-detect the "
        "'2' arm of the 4-2-1 rule; severe NPDR is therefore mainly caught via "
        "haemorrhage counts.",
        "",
        "## 7. Human oversight (assumed, not optional)",
        "",
        "The reported performance assumes the deployment policy the pipeline "
        "implements:",
        "",
        "- Grade >= 3, any neovascularisation, and any suspected clinically "
        "significant macular oedema go to a human **regardless of model "
        "confidence**.",
        "- Cases inside the uncertainty band are deferred to a human.",
        "- Every human decision is logged against the model's, so agreement "
        "can be monitored over time and by site.",
        "",
        "Deploying without that oversight invalidates these numbers.",
        "",
        "## 8. Ethical and equity considerations",
        "",
        "- The screening threshold is deliberately asymmetric: sensitivity is "
        "treated as the binding constraint because a missed proliferative DR "
        "costs sight, while a false positive costs one teleconsultation.",
        "- Performance should be re-measured per site and per camera before "
        "expanding a programme; an aggregate figure can hide a site where the "
        "system is failing.",
        "- The system reduces specialist workload; it does not remove the need "
        "for specialists, and a programme that staffs on the assumption it does "
        "will fail its urgent cases first.",
        "",
    ]
    return "\n".join(lines)


def write(validation_path: str | Path, out_path: str | Path,
          extra: dict | None = None) -> Path:
    v = json.loads(Path(validation_path).read_text())
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(generate(v, extra), encoding="utf-8")
    return p
