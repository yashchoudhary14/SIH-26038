"""Annotated clinical report generation.

Target: an ophthalmologist reaches a validated agree/disagree decision in
**under 30 seconds**.  That constraint drives every layout choice here.

What a reviewer needs, in the order they need it:

1. The verdict and its confidence -- one glance, top-left.
2. *Where* to look -- the annotated image, with lesions marked in a fixed
   colour code and the fovea/disc drawn so they can orient instantly.
3. *Why* -- the criteria in clinical language ("22 haemorrhages across 4
   quadrants"), which they can spot-check against the image.
4. The attention map, as a cross-check that the model looked at the lesions
   and not at the vignette.
5. Anything unusual -- borderline quality, model/rule disagreement,
   high uncertainty -- surfaced as a warning rather than buried.

Both a composited PNG review panel and a standalone HTML report are produced;
the PNG is what gets pushed to a low-bandwidth review client, the HTML is the
record.
"""
from __future__ import annotations

import base64
import html
import io
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from ..constants import ICDR_GRADES, LESION_COLORS, LESION_CLASSES


# --------------------------------------------------------------------------
# Image composition
# --------------------------------------------------------------------------
def overlay_cam(image: np.ndarray, cam: np.ndarray, alpha: float = 0.42,
                threshold: float = 0.25) -> np.ndarray:
    """Blend a CAM onto the image, suppressing the low-attention background.

    Thresholding matters: an un-thresholded jet colormap tints the entire
    retina, which both looks alarming and hides the actual peak.
    """
    cam_u8 = np.uint8(np.clip(cam, 0, 1) * 255)
    heat = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    mask = (cam >= threshold).astype(np.float32)[..., None]
    mask = cv2.GaussianBlur(mask, (0, 0), 3)[..., None] if mask.ndim == 3 else mask
    a = alpha * np.clip(mask, 0, 1)
    return np.clip(image.astype(np.float32) * (1 - a) + heat.astype(np.float32) * a,
                   0, 255).astype(np.uint8)


def annotate_lesions(image: np.ndarray, lesion_probs: np.ndarray,
                     landmarks=None, threshold: float = 0.5,
                     min_area: int = 3, draw_landmarks: bool = True) -> np.ndarray:
    """Draw lesion contours in the fixed colour code, plus the disc and fovea."""
    out = image.copy()
    for ci, cname in enumerate(LESION_CLASSES):
        binary = (lesion_probs[..., ci] >= threshold).astype(np.uint8)
        if binary.sum() == 0:
            continue
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        keep = np.zeros_like(binary)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                keep[labels == i] = 1
        contours, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        colour = LESION_COLORS[cname]
        for c in contours:
            area = cv2.contourArea(c)
            if area < 12:            # too small to outline legibly -- ring it
                (cx, cy), r = cv2.minEnclosingCircle(c)
                cv2.circle(out, (int(cx), int(cy)), max(4, int(r) + 3), colour, 1, cv2.LINE_AA)
            else:
                cv2.drawContours(out, [c], -1, colour, 1, cv2.LINE_AA)

    if draw_landmarks and landmarks is not None:
        dx, dy = landmarks.disc_xy
        fx, fy = landmarks.fovea_xy
        cv2.circle(out, (dx, dy), int(landmarks.disc_radius), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.drawMarker(out, (fx, fy), (0, 255, 0), cv2.MARKER_CROSS, 16, 1, cv2.LINE_AA)
        # 1-DD ring around the fovea: the CSME boundary a reviewer checks first.
        cv2.circle(out, (fx, fy), int(landmarks.disc_diameter_px), (0, 200, 0), 1, cv2.LINE_AA)
    return out


def legend_strip(width: int, height: int = 34) -> np.ndarray:
    strip = np.full((height, width, 3), 24, np.uint8)
    x = 10
    for cname in LESION_CLASSES:
        colour = LESION_COLORS[cname]
        cv2.rectangle(strip, (x, height // 2 - 6), (x + 14, height // 2 + 6), colour, -1)
        label = cname.replace("_", " ")
        cv2.putText(strip, label, (x + 20, height // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
        x += 26 + int(len(label) * 7.2)
    return strip


def build_review_panel(result, artifacts: dict, panel_width: int = 1440) -> np.ndarray:
    """Three-up panel: enhanced image | lesion annotations | attention map."""
    base = artifacts.get("enhanced", artifacts.get("standardized"))
    if base is None:
        raise ValueError("No image available for the review panel.")
    size = base.shape[0]

    tiles = [("Enhanced fundus", base)]
    if "lesion_probs" in artifacts:
        tiles.append(("Detected lesions (ICDR evidence)",
                      annotate_lesions(base, artifacts["lesion_probs"],
                                       artifacts.get("landmarks"))))
    if "cam" in artifacts:
        tiles.append(("Model attention (Grad-CAM++)", overlay_cam(base, artifacts["cam"])))

    tile_w = panel_width // len(tiles)
    scale = tile_w / size
    tile_h = int(size * scale)
    header_h = 30

    canvas = np.full((tile_h + header_h + 34, tile_w * len(tiles), 3), 18, np.uint8)
    for i, (title, img) in enumerate(tiles):
        t = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        x0 = i * tile_w
        canvas[header_h:header_h + tile_h, x0:x0 + tile_w] = t
        cv2.putText(canvas, title, (x0 + 10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.line(canvas, (x0, header_h), (x0, header_h + tile_h), (60, 60, 60), 1)

    canvas[header_h + tile_h:, :] = cv2.resize(
        legend_strip(canvas.shape[1]), (canvas.shape[1], 34))
    return canvas


# --------------------------------------------------------------------------
# HTML report
# --------------------------------------------------------------------------
def _b64_png(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


URGENCY_STYLE = {
    "urgent": ("#7f1d1d", "#fecaca", "URGENT REFERRAL"),
    "soon": ("#78350f", "#fed7aa", "REFER"),
    "routine": ("#14532d", "#bbf7d0", "ROUTINE"),
}
DECISION_LABEL = {
    "auto_report": "Auto-reported (model confident, no referral)",
    "refer": "Referral recommended",
    "defer_to_human": "Deferred for human grading (model uncertain)",
    "recapture": "Image ungradeable - recapture required",
}


def render_html(result, artifacts: dict, title: str = "DR Screening Report") -> str:
    r = result
    grade_colour, grade_bg, urgency_text = URGENCY_STYLE.get(
        r.urgency, URGENCY_STYLE["routine"])

    panel_img = ""
    try:
        panel_img = _b64_png(build_review_panel(r, artifacts))
    except Exception:
        pass

    prob_bars = ""
    for g, p in enumerate(r.class_probabilities or []):
        pct = max(0.4, p * 100)
        is_pred = (g == r.grade)
        prob_bars += f"""
        <div class="bar-row">
          <span class="bar-label">{g} &middot; {html.escape(ICDR_GRADES[g])}</span>
          <span class="bar-track"><span class="bar-fill{' pred' if is_pred else ''}"
                style="width:{pct:.1f}%"></span></span>
          <span class="bar-val">{p*100:.1f}%</span>
        </div>"""

    evidence_html = ""
    for e in r.evidence or []:
        if e.get("status") == "not assessed":
            # Rendered as its own class, never as "0 detected". The distinction
            # is the whole point: a lesion class with no pixel supervision is
            # one the model cannot see, and for neovascularisation that is the
            # difference between excluding proliferative DR and never having
            # looked for it.
            evidence_html += (f"<li class='caution'><b>{html.escape(str(e['finding']))}</b>: "
                              f"NOT ASSESSED &mdash; {html.escape(str(e.get('detail','')))}</li>")
        elif "finding" in e:
            q = ", ".join(f"{k} {v}" for k, v in (e.get("per_quadrant") or {}).items())
            evidence_html += (f"<li><b>{html.escape(str(e['finding']))}</b>: "
                              f"{e['count']} detected"
                              + (f" ({html.escape(q)})" if q else "")
                              + f" &mdash; {e['area_percent']}% of retinal area</li>")
        elif "criterion" in e:
            evidence_html += f"<li class='criterion'>{html.escape(str(e['criterion']))}</li>"
        elif "macular_assessment" in e:
            evidence_html += f"<li class='macula'>{html.escape(str(e['macular_assessment']))}</li>"
        elif "caution" in e:
            evidence_html += f"<li class='caution'>{html.escape(str(e['caution']))}</li>"

    quality_rows = ""
    for k, v in (r.quality.get("scores") or {}).items():
        verdict = (r.quality.get("verdicts") or {}).get(k, "pass")
        quality_rows += (f"<tr><td>{html.escape(k.replace('_',' '))}</td>"
                         f"<td class='num'>{v:.2f}</td>"
                         f"<td class='v-{verdict}'>{verdict}</td></tr>")

    warnings = []
    if r.quality.get("overall") == "borderline":
        warnings.append("Image quality is borderline; enhancement was applied. "
                        "Interpret subtle findings with caution.")
    if r.agreement == "disagree":
        warnings.append(f"Deep model and rule-based criteria disagree "
                        f"(model {r.grade}, rules {r.rule_based_grade}).")
    if r.uncertainty.get("epistemic_variance", 0) > 0.05:
        warnings.append("High model uncertainty on this image.")
    if not r.gradeable:
        warnings.append("Image was rejected by the quality gate.")
    warn_html = "".join(f"<div class='warn'>{html.escape(w)}</div>" for w in warnings)

    advice_html = ""
    if r.recapture_advice:
        advice_html = ("<div class='card'><h2>Recapture instructions</h2><ol>"
                       + "".join(f"<li>{html.escape(a)}</li>" for a in r.recapture_advice)
                       + "</ol></div>")

    timing = r.timing_ms or {}
    timing_rows = "".join(f"<tr><td>{html.escape(k)}</td><td class='num'>{v:.1f} ms</td></tr>"
                          for k, v in timing.items())

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} &mdash; {html.escape(r.image_id)}</title>
<style>
:root {{ --bg:#0b0f14; --card:#131a22; --line:#243040; --text:#e6edf3; --muted:#93a1b1; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text);
  font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
.wrap {{ max-width:1500px; margin:0 auto; padding:20px; }}
header {{ display:flex; justify-content:space-between; align-items:flex-start;
  gap:20px; flex-wrap:wrap; border-bottom:1px solid var(--line); padding-bottom:14px; }}
h1 {{ font-size:19px; margin:0 0 4px; letter-spacing:.2px; }}
.meta {{ color:var(--muted); font-size:12.5px; }}
.verdict {{ background:{grade_bg}; color:{grade_colour}; border-radius:10px;
  padding:12px 18px; min-width:290px; }}
.verdict .g {{ font-size:26px; font-weight:700; line-height:1.15; }}
.verdict .u {{ font-size:11px; font-weight:700; letter-spacing:1.4px; opacity:.85; }}
.verdict .d {{ font-size:12.5px; margin-top:5px; }}
.grid {{ display:grid; grid-template-columns:1fr 380px; gap:18px; margin-top:18px; }}
@media (max-width:1100px) {{ .grid {{ grid-template-columns:1fr; }} }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:14px 16px; margin-bottom:16px; }}
.card h2 {{ font-size:13px; text-transform:uppercase; letter-spacing:1px;
  color:var(--muted); margin:0 0 10px; font-weight:600; }}
img.panel {{ width:100%; border-radius:8px; display:block; }}
.bar-row {{ display:flex; align-items:center; gap:9px; margin:5px 0; font-size:12.5px; }}
.bar-label {{ width:150px; color:var(--muted); }}
.bar-track {{ flex:1; height:9px; background:#1e2732; border-radius:5px; overflow:hidden; }}
.bar-fill {{ display:block; height:100%; background:#3b6ea5; }}
.bar-fill.pred {{ background:#2f9e6b; }}
.bar-val {{ width:52px; text-align:right; font-variant-numeric:tabular-nums; }}
ul {{ margin:0; padding-left:19px; }} li {{ margin:5px 0; }}
li.criterion {{ color:#9fd3ff; }} li.macula {{ color:#ffd28a; }}
li.caution {{ color:#ffb0b0; }}
table {{ width:100%; border-collapse:collapse; font-size:12.5px; }}
td {{ padding:4px 6px; border-bottom:1px solid #1d2735; }}
td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.v-pass {{ color:#5fd39a; }} .v-borderline {{ color:#e8c468; }} .v-fail {{ color:#f08a8a; }}
.warn {{ background:#2a1d10; border-left:3px solid #e0a33e; padding:9px 12px;
  border-radius:5px; margin-bottom:9px; font-size:12.5px; }}
.kv {{ display:flex; justify-content:space-between; padding:3px 0;
  border-bottom:1px solid #1d2735; font-size:12.5px; }}
.kv span:first-child {{ color:var(--muted); }}
footer {{ color:var(--muted); font-size:11.5px; margin-top:22px;
  border-top:1px solid var(--line); padding-top:12px; }}
</style></head><body><div class="wrap">

<header>
  <div>
    <h1>Diabetic Retinopathy Screening Report</h1>
    <div class="meta">Case <b>{html.escape(r.image_id)}</b> &middot;
      {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} &middot;
      model {html.escape(r.model_version)} &middot;
      {timing.get('total', 0):.0f} ms total</div>
  </div>
  <div class="verdict">
    <div class="u">{urgency_text}</div>
    <div class="g">Grade {r.grade} &mdash; {html.escape(r.grade_label)}</div>
    <div class="d">{html.escape(DECISION_LABEL.get(r.decision, r.decision))}<br>
      P(referable) = <b>{r.referable_probability:.3f}</b>
      &middot; confidence {r.confidence:.1%}</div>
  </div>
</header>

{warn_html}

<div class="grid">
  <div>
    <div class="card">
      <h2>Image review</h2>
      {'<img class="panel" src="' + panel_img + '" alt="review panel">' if panel_img
       else '<p class="meta">No image panel available.</p>'}
    </div>
    <div class="card">
      <h2>Clinical evidence &mdash; ICDR criteria</h2>
      <ul>{evidence_html or '<li>No lesions detected.</li>'}</ul>
    </div>
    {advice_html}
  </div>

  <div>
    <div class="card">
      <h2>Severity distribution</h2>
      {prob_bars}
    </div>
    <div class="card">
      <h2>Cross-check</h2>
      <div class="kv"><span>Deep model grade</span><b>{r.grade}</b></div>
      <div class="kv"><span>Rule-based grade</span><b>{r.rule_based_grade}</b></div>
      <div class="kv"><span>Agreement</span><b>{html.escape(r.agreement)}</b></div>
      <div class="kv"><span>Macular oedema risk</span><b>{r.dme_risk}</b></div>
      <div class="kv"><span>Predictive entropy</span>
        <b>{r.uncertainty.get('entropy', 0):.3f}</b></div>
      <div class="kv"><span>Epistemic variance</span>
        <b>{r.uncertainty.get('epistemic_variance', 0):.4f}</b></div>
    </div>
    <div class="card">
      <h2>Image quality</h2>
      <table>{quality_rows}</table>
      <div class="kv" style="margin-top:8px"><span>Overall</span>
        <b>{html.escape(str(r.quality.get('overall','')))}</b></div>
      <div class="kv"><span>Enhancement applied</span>
        <b>{html.escape(', '.join(r.enhancement_applied) or 'none')}</b></div>
    </div>
    <div class="card">
      <h2>Latency breakdown</h2>
      <table>{timing_rows}</table>
    </div>
  </div>
</div>

<footer>
  Decision-support output. Not a diagnosis. Every referable and every
  sight-threatening finding is reviewed by a qualified ophthalmologist before
  any clinical action. Grading follows the International Clinical Diabetic
  Retinopathy severity scale; referable DR is grade &ge; 2.
</footer>
</div></body></html>"""


def save_report(result, artifacts: dict, out_dir: str | Path,
                write_panel: bool = True) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}

    html_path = out / f"{result.image_id}_report.html"
    html_path.write_text(render_html(result, artifacts), encoding="utf-8")
    paths["html"] = html_path

    if write_panel:
        try:
            panel = build_review_panel(result, artifacts)
            p = out / f"{result.image_id}_panel.png"
            cv2.imwrite(str(p), panel)
            paths["panel"] = p
        except Exception:
            pass

    json_path = out / f"{result.image_id}_result.json"
    json_path.write_text(result.to_json(), encoding="utf-8")
    paths["json"] = json_path
    return paths
