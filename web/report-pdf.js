/* The screening report as a PDF, built in the browser from a screening result.

   The console (and the dossier's replay) already hold everything a report
   needs: the result a run returns, the photograph, and the review panel
   (enhanced image, lesion outlines, Grad-CAM++). The PDF is assembled here
   with jsPDF (vendor/jspdf.umd.min.js, MIT licence), so the report is the
   same whether the page is served by MATLAB, by the Python API, or replays a
   recorded run with no backend, and nothing leaves the laptop.

     DRReport.download(result, {src, panel, health, review, label, recorded})

   result   the screening result JSON (see pipeline.py, ScreeningResult)
   src      the photograph: a URL, blob URL or data URL
   panel    the review panel (three frames side by side), same forms
   health   the backend's /health reply, for the threshold and runtime
   review   {grade, model, agreement, seconds} if a reviewer graded it here
*/
(function (root) {
  "use strict";
  const LIB = "vendor/jspdf.umd.min.js?v=4.2.1";

  const GRADES = ["No apparent DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"];
  const GRADE_LONG = [
    "No apparent diabetic retinopathy",
    "Mild non-proliferative diabetic retinopathy",
    "Moderate non-proliferative diabetic retinopathy",
    "Severe non-proliferative diabetic retinopathy",
    "Proliferative diabetic retinopathy"];
  const DECISION = {
    auto_report: "Auto-reported: the model is confident there is no referable disease",
    refer: "Referral recommended",
    defer_to_human: "Deferred to a human grader: the model is not certain enough to decide",
    recapture: "Ungradeable: the photograph must be taken again"};
  // the outline colours both backends draw the review panel in (constants.py, constants.m)
  const LESION = [
    ["microaneurysm", "Microaneurysms", [255, 0, 0]],
    ["hemorrhage", "Haemorrhages", [255, 96, 0]],
    ["hard_exudate", "Hard exudates", [255, 255, 0]],
    ["soft_exudate", "Soft exudates (cotton-wool spots)", [255, 255, 255]],
    ["neovascularization", "Neovascularisation", [255, 0, 255]]];
  const QUALITY = [["focus", "Focus"], ["illumination", "Illumination"], ["contrast", "Contrast"],
    ["fov", "Field of view"], ["macula", "Macula in view"], ["artifact", "Artefacts"],
    ["under_exposure", "Under-exposure"], ["over_exposure", "Over-exposure"], ["noise", "Noise"]];
  const QTH = {focus: [.25, .5], illumination: [.3, .55], contrast: [.25, .5], fov: [.55, .8], macula: [.2, .45],
    artifact: [.3, .6], under_exposure: [.25, .5], over_exposure: [.25, .5], noise: [.2, .45]};
  const STAGES = {geometry: "Geometry: crop and resize", landmarks: "Landmarks: optic disc and fovea",
    quality: "Quality gate", enhancement: "Enhancement", quality_recheck: "Quality gate, re-check",
    segmentation: "Lesion segmentation (U-Net)", clinical_features: "Clinical features", grading: "Grading (CORN ordinal)",
    explanation: "Explanation (Grad-CAM++)"};
  const ENHANCE = {grey_world: "grey-world colour balance", illumination_normalize: "illumination levelling",
    auto_exposure: "exposure correction", clahe_lab: "local contrast (CLAHE)", denoise: "denoising"};
  const corrections = r => (r.enhancement_applied || []).map(k => ENHANCE[k] || k.replace(/_/g, " ")).join(", ");

  // page geometry, millimetres (A4)
  const PW = 210, PH = 297, ML = 16, MR = 16, TOP = 16, BOTTOM = 20, CW = PW - ML - MR;
  const PT = 0.3528;                                   // one point in mm
  const C = {ink: [24, 29, 37], body: [52, 60, 72], muted: [96, 106, 120], faint: [150, 158, 170],
    rule: [214, 219, 226], soft: [244, 246, 248], accent: [10, 118, 111], white: [255, 255, 255]};
  const TONE = {urgent: [180, 35, 24], refer: [181, 71, 8], routine: [6, 118, 71], human: [88, 62, 160], recapture: [71, 84, 103]};
  const GRADE_RGB = [[18, 150, 90], [120, 170, 20], [214, 130, 10], [214, 60, 44], [140, 28, 22]];

  // The standard PDF fonts cover Latin-1 only; everything else is spelled out.
  const MAP = {"≥": ">=", "≤": "<=", "—": " - ", "–": "-", "‘": "'", "’": "'", "“": '"',
    "”": '"', "…": "...", "−": "-", "→": "->", " ": " ", " ": " ", "λ": "lambda",
    "σ": "sigma", "μ": "mu", "•": "-"};
  const clean = s => String(s ?? "").replace(/[^\x00-\xFF]/g, ch => MAP[ch] ?? "").replace(/[\x80-\x9F]/g, "");
  const pct = (v, d = 1) => Number.isFinite(v) ? (100 * v).toFixed(d) + "%" : "-";
  const num = (v, d = 3) => Number.isFinite(v) ? v.toFixed(d) : "-";
  const ms = v => Number.isFinite(v) ? (v >= 100 ? v.toFixed(0) : v.toFixed(1)) + " ms" : "-";
  const cap = s => s ? s.charAt(0).toUpperCase() + s.slice(1) : s;

  // ── the library, loaded on first use so the console stays light ───────────
  let libLoading = null;
  function jsPDFClass() {
    if (root.jspdf && root.jspdf.jsPDF) return Promise.resolve(root.jspdf.jsPDF);
    if (!libLoading) libLoading = new Promise((ok, fail) => {
      const s = document.createElement("script");
      s.src = LIB; s.async = true;
      s.onload = () => (root.jspdf && root.jspdf.jsPDF) ? ok(root.jspdf.jsPDF) : fail(new Error("the PDF library did not load"));
      s.onerror = () => { libLoading = null; fail(new Error("could not load " + LIB)); };
      document.head.appendChild(s);
    });
    return libLoading;
  }

  // ── images: drawn through a canvas into JPEG the PDF can embed ────────────
  function loadImage(url) {
    return new Promise(resolve => {
      if (!url) return resolve(null);
      const img = new Image();
      try { const u = new URL(url, location.href); if (/^https?:$/.test(u.protocol) && u.origin !== location.origin) img.crossOrigin = "anonymous"; } catch (e) {}
      img.onload = () => resolve(img); img.onerror = () => resolve(null);
      img.src = url;
    });
  }
  // a square JPEG of (part of) an image, letterboxed on black like a fundus frame
  function squareJpeg(img, sx, sy, sw, sh, px) {
    try {
      const cv = document.createElement("canvas"); cv.width = cv.height = px;
      const g = cv.getContext("2d"); g.fillStyle = "#000"; g.fillRect(0, 0, px, px);
      const k = Math.min(px / sw, px / sh), w = sw * k, h = sh * k;
      g.drawImage(img, sx, sy, sw, sh, (px - w) / 2, (px - h) / 2, w, h);
      return cv.toDataURL("image/jpeg", 0.9);
    } catch (e) { return null; }                          // a cross-origin image taints the canvas
  }
  async function images(src, panel) {
    const out = {};
    const photo = await loadImage(src);
    if (photo) out.original = squareJpeg(photo, 0, 0, photo.naturalWidth, photo.naturalHeight, 900);
    const p = await loadImage(panel);
    if (p) {
      // the panel is three square frames side by side under a title strip:
      // the same crop the console's viewer uses, 2 px inside each frame so the
      // black margin's edge (where the untrained NV channel draws a line) stays out
      const fw = p.naturalWidth / 3, y0 = Math.max(0, (p.naturalHeight - fw) * 0.469), e = 2;
      ["enhanced", "lesions", "attention"].forEach((k, i) => { out[k] = squareJpeg(p, i * fw + e, y0 + e, fw - 2 * e, Math.min(fw, p.naturalHeight) - 2 * e, 900); });
    }
    return out;
  }

  // ── a writer with a cursor and automatic page breaks ──────────────────────
  function Writer(doc) {
    const W = {doc, y: TOP};
    W.font = (style, size, color) => { doc.setFont("helvetica", style || "normal"); doc.setFontSize(size); doc.setTextColor(...(color || C.ink)); };
    W.lh = size => size * PT * 1.42;
    W.room = () => PH - BOTTOM - W.y;
    W.ensure = h => { if (W.y + h > PH - BOTTOM) { W.newPage(); return true; } return false; };
    W.newPage = () => { doc.addPage(); W.y = TOP + 1; };
    // right-aligned text with letter spacing (jsPDF leaves the spacing out of its own right-align)
    W.rtext = (text, x, y, cs = 0) => { const t = clean(text); doc.text(t, x - doc.getTextWidth(t) - cs * Math.max(0, t.length - 1), y, {charSpace: cs}); };
    W.lines = (text, width) => doc.splitTextToSize(clean(text), width);
    // a paragraph, wrapped to width; breaks across pages line by line
    W.para = (text, o = {}) => {
      const size = o.size || 9.5, x = o.x ?? ML, width = o.width ?? CW;
      W.font(o.style, size, o.color || C.body);
      W.lines(text, width).forEach(line => {
        W.ensure(W.lh(size));
        doc.text(line, x, W.y + size * PT * 0.82);
        W.y += W.lh(size);
      });
      W.y += o.after ?? 1.2;
    };
    // a paragraph with **bold** runs, wrapped word by word; o.mark draws a list marker
    W.rich = (text, o = {}) => {
      const size = o.size || 9.5, indent = o.mark ? 5 : 0, x0 = (o.x ?? ML) + indent, width = (o.width ?? CW) - indent;
      const lh = W.lh(size), base = o.style || "normal", color = o.color || C.body, strong = o.strong || C.ink;
      const measure = (t, bold) => { doc.setFont("helvetica", bold ? "bold" : base); doc.setFontSize(size); return doc.getTextWidth(t); };
      const lines = [[]]; let used = 0;
      clean(text).split("**").forEach((run, ri) => run.split(/(\s+)/).forEach(tok => {
        if (!tok) return;
        const bold = ri % 2 === 1, space = /^\s+$/.test(tok), t = space ? " " : tok, w = measure(t, bold);
        let line = lines[lines.length - 1];
        if (space && !line.length) return;
        if (!space && used + w > width && line.length) {
          // a token glued to the word before it ("doctor" + ".") moves down with that word
          let k = line.length;
          while (k > 0 && line[k - 1].t !== " ") k--;
          const carry = k > 0 ? line.splice(k) : [];
          while (line.length && line[line.length - 1].t === " ") line.pop();
          lines.push(line = carry); used = carry.reduce((a, c) => a + c.w, 0);
        }
        line.push({t, bold, w}); used += w;
      }));
      lines.forEach((line, i) => {
        W.ensure(lh);
        if (o.mark && i === 0) { doc.setFillColor(...o.mark); doc.circle(x0 - indent + 1.3, W.y + size * PT * 0.5, 0.8, "F"); }
        // each run of one style is drawn as one string, so the viewer spaces its words itself
        let x = x0;
        const segs = [];
        line.forEach(tk => { const s = segs[segs.length - 1]; if (s && s.bold === tk.bold) { s.t += tk.t; s.w += tk.w; } else segs.push({t: tk.t, bold: tk.bold, w: tk.w}); });
        segs.forEach(sg => { doc.setFont("helvetica", sg.bold ? "bold" : base); doc.setFontSize(size);
          doc.setTextColor(...(sg.bold ? strong : color)); doc.text(sg.t, x, W.y + size * PT * 0.82); x += sg.w; });
        W.y += lh;
      });
      W.y += o.after ?? (o.mark ? 1 : 1.2);
    };
    // a list item with a coloured marker
    W.item = (text, o = {}) => {
      const size = o.size || 9.5, indent = 5;
      W.font(o.style, size, o.color || C.body);
      const lines = W.lines(text, CW - indent);
      W.ensure(W.lh(size) * Math.min(lines.length, 2));
      doc.setFillColor(...(o.mark || C.accent));
      doc.circle(ML + 1.3, W.y + size * PT * 0.5, 0.8, "F");
      lines.forEach(line => { W.ensure(W.lh(size)); doc.text(line, ML + indent, W.y + size * PT * 0.82); W.y += W.lh(size); });
      W.y += 1;
    };
    // a numbered section; `keep` is how much of what follows must fit under the heading
    W.section = (n, title, lede, keep = 30) => {
      W.font("normal", 8.8);
      const ledeH = lede ? W.lines(lede, CW).length * W.lh(8.8) + 3 : 2;
      if (!W.ensure(15.2 + ledeH + keep)) W.y += 5;
      doc.setDrawColor(...C.ink); doc.setLineWidth(0.45); doc.line(ML, W.y, ML + CW, W.y);
      W.y += 6.2;
      W.font("bold", 12, C.accent); doc.text(String(n), ML, W.y);
      W.font("bold", 12, C.ink); doc.text(clean(title), ML + 8, W.y);
      W.y += 4;
      if (lede) { W.y += 1; W.para(lede, {size: 8.8, color: C.muted, after: 2}); } else W.y += 2;
    };
    W.label = (text, x, y, color) => { W.font("bold", 7, color || C.muted); doc.text(clean(text).toUpperCase(), x, y, {charSpace: 0.25}); };
    // a table: columns [{h, w, align}]; rows of cell strings, or {text, color, style, span, align}
    // where span is how many columns the cell covers
    W.table = (cols, rows, o = {}) => {
      const size = o.size || 9, pad = 1.8, lh = W.lh(size);
      const head = () => {
        W.font("bold", 7.2, C.muted);
        let x = ML; cols.forEach(c => { const h = clean(c.h).toUpperCase();
          if (c.align === "right") W.rtext(h, x + c.w - pad, W.y + 4, 0.2); else doc.text(h, x + pad, W.y + 4, {charSpace: 0.2});
          x += c.w; });
        W.y += 6; doc.setDrawColor(...C.ink); doc.setLineWidth(0.3); doc.line(ML, W.y, ML + CW, W.y); W.y += 0.6;
      };
      if (o.head !== false) { W.ensure(6 + lh * 2 + 4); head(); }
      rows.forEach((row, ri) => {
        let ci = 0;
        const cells = row.map((cell, i) => {
          const c = typeof cell === "object" && cell ? cell : {text: cell};
          const span = Math.max(1, c.span || 1), w = cols.slice(ci, ci + span).reduce((a, k) => a + k.w, 0);
          const align = c.align || cols[ci].align;
          ci += span;
          W.font(c.style || (i === 0 ? "bold" : "normal"), size);
          return Object.assign({}, c, {w, align, lines: W.lines(c.text ?? "", w - 2 * pad)});
        });
        const h = Math.max(...cells.map(c => c.lines.length)) * lh + 2 * pad;
        if (W.ensure(h) && o.head !== false) head();
        if (o.zebra && ri % 2 === 1) { doc.setFillColor(...C.soft); doc.rect(ML, W.y, CW, h, "F"); }
        let x = ML;
        cells.forEach((c, i) => {
          W.font(c.style || (i === 0 ? "bold" : "normal"), size, c.color || (i === 0 ? C.ink : C.body));
          const right = c.align === "right", tx = right ? x + c.w - pad : x + pad;
          c.lines.forEach((ln, k) => doc.text(ln, tx, W.y + pad + size * PT * 0.82 + k * lh, {align: right ? "right" : "left"}));
          x += c.w;
        });
        W.y += h;
        doc.setDrawColor(...C.rule); doc.setLineWidth(0.2); doc.line(ML, W.y, ML + CW, W.y);
      });
      W.y += o.after ?? 3;
    };
    return W;
  }

  // ── what the result means for the patient ─────────────────────────────────
  function verdict(r) {
    if (r.gradeable === false || r.decision === "recapture")
      return {key: "recapture", label: "RECAPTURE REQUIRED",
        next: "The photograph could not be graded. Take it again, following the recapture instructions in this report, while the patient is still at the centre."};
    if (r.decision === "defer_to_human")
      return {key: "human", label: "HUMAN REVIEW",
        next: "The model was not certain enough to decide. A human grader reviews this image before any result is given to the patient."};
    if (r.urgency === "urgent")
      return {key: "urgent", label: "URGENT REFERRAL",
        next: "Refer urgently to an ophthalmologist. Programme service target: seen within 2 days."};
    if (r.decision === "refer" || r.urgency === "soon")
      return {key: "refer", label: "REFER",
        next: "Refer to an ophthalmologist. Programme service target: seen within 14 days."};
    return {key: "routine", label: "ROUTINE",
      next: "No referable diabetic retinopathy found. Re-screen in 12 months, the programme's routine recall."};
  }
  function cautions(r, panel) {
    const w = [];
    if (r.quality?.overall === "borderline") w.push("Image quality was borderline and enhancement was applied. Interpret subtle findings with caution.");
    if (r.agreement === "disagree") w.push(`The deep model (grade ${r.grade}) and the ICDR rule-based criteria (grade ${r.rule_based_grade}) disagree.`);
    if ((r.uncertainty?.epistemic_variance || 0) > 0.05) w.push("The model is unusually uncertain about this image.");
    (r.clinical_features?.unassessed || []).forEach(k => w.push(`${cap(k.replace(/_/g, " ").replace("neovascularization", "neovascularisation"))} was not assessed: the model was not trained to outline it, so its absence here is not evidence that it is absent. A full examination is still advised.`
      + (panel && k === "neovascularization" ? " The magenta outlines in the lesion image come from this untrained channel and are not findings." : "")));
    return w;
  }

  // ── the report ────────────────────────────────────────────────────────────
  async function build(r, o = {}) {
    const JsPDF = await jsPDFClass();
    const doc = new JsPDF({unit: "mm", format: "a4", compress: true});
    const W = Writer(doc), now = new Date();
    const health = o.health || null, v = verdict(r), tone = TONE[v.key];
    const gradeable = r.gradeable !== false && r.grade != null;
    const threshold = health && Number.isFinite(health.referral_threshold) ? health.referral_threshold : 0.1999;
    const p2 = k => String(k).padStart(2, "0");               // local time, as "Generated" shows it
    const day = `${now.getFullYear()}-${p2(now.getMonth() + 1)}-${p2(now.getDate())}`;
    const stamp = `${day.replace(/-/g, "")}-${p2(now.getHours())}${p2(now.getMinutes())}${p2(now.getSeconds())}`;
    const reportId = `DRS-${stamp}-${String(r.image_id || "image").replace(/[^A-Za-z0-9_-]/g, "").slice(0, 24)}`;
    const when = clean(now.toLocaleString("en-IN", {day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit"}));
    const runtime = o.recorded ? "Recorded run, replayed without a backend"
      : health ? `${health.runtime === "matlab" ? "MATLAB" : "Python"} backend${health.device ? ", " + health.device : ""}` : "Backend";
    const bundle = health?.artifacts_dir ? (/all|pooled/.test(health.artifacts_dir) ? "pooled (APTOS-2019, IDRiD, DDR, Messidor-2)" : "prepool (APTOS-2019, IDRiD)")
      : o.recorded ? "prepool (APTOS-2019, IDRiD)" : "-";
    const pics = await images(o.src, o.panel);
    doc.setProperties({title: `DR screening report ${r.image_id || ""}`, subject: "Diabetic retinopathy screening report",
      author: "SixEyes DR screening", creator: "SixEyes screening console", keywords: "diabetic retinopathy, ICDR, screening"});

    // ── title block ──
    W.font("bold", 7.5, C.accent); doc.text("SIXEYES  ·  DIABETIC RETINOPATHY SCREENING", ML, W.y + 2.5, {charSpace: 0.3});
    W.font("bold", 19, C.ink); doc.text("Screening report", ML, W.y + 11.5);
    W.font("normal", 9, C.muted); doc.text("Screening support for a clinician. Not a diagnosis.", ML, W.y + 17);
    const meta = [["Report ID", reportId], ["Generated", when], ["Image", clean(o.label || r.image_id || "-")], ["Model", `${clean(r.model_version || "-")}, ${bundle.split(" ")[0]} bundle`]];
    let my = W.y + 1.5;
    meta.forEach(([k, val]) => {
      W.font("bold", 6.8, C.muted); doc.text(k.toUpperCase(), ML + CW - 76, my + 2.2, {charSpace: 0.2});
      W.font("normal", 8.2, C.ink); doc.text(doc.splitTextToSize(val, 56)[0], ML + CW, my + 2.2, {align: "right"});
      my += 4.6;
    });
    W.y += 24;

    // ── the result band ──
    const bandH = 27;
    doc.setFillColor(...tone); doc.roundedRect(ML, W.y, CW, bandH, 2, 2, "F");
    W.font("bold", 7.5, C.white); doc.text("SCREENING RESULT", ML + 6, W.y + 6.5, {charSpace: 0.3});
    W.font("bold", 18, C.white); doc.text(v.label, ML + 6, W.y + 15);
    W.font("normal", 9.2, C.white);
    doc.text(doc.splitTextToSize(clean(DECISION[r.decision] || r.decision || ""), 100), ML + 6, W.y + 21);
    // the grade, in a white box on the right of the band
    const gx = ML + CW - 62, gy = W.y + 4, gw = 56, gh = bandH - 8;
    doc.setFillColor(...C.white); doc.roundedRect(gx, gy, gw, gh, 1.6, 1.6, "F");
    W.font("bold", 6.8, C.muted); doc.text("ICDR GRADE", gx + 4, gy + 5, {charSpace: 0.25});
    const tx = gradeable ? gx + 16 : gx + 4;
    if (gradeable) { W.font("bold", 24, GRADE_RGB[r.grade]); doc.text(String(r.grade), gx + 4, gy + 15.5); }
    W.font("bold", 9, C.ink); doc.text(doc.splitTextToSize(gradeable ? GRADES[r.grade] : "Ungradeable", gw - 20)[0], tx, gy + 11);
    W.font("normal", 7.2, C.muted); doc.text(gradeable ? (r.grade >= 2 ? "referable (grade 2 or above)" : "not referable") : "no grade until the photograph is retaken", tx, gy + 15.5);
    W.y += bandH + 3;
    // next step
    doc.setFillColor(...C.soft); W.font("normal", 9.5); const nextLines = W.lines(v.next, CW - 34);
    const nh = Math.max(11, nextLines.length * W.lh(9.5) + 6);
    doc.roundedRect(ML, W.y, CW, nh, 1.6, 1.6, "F");
    W.font("bold", 7, tone); doc.text("NEXT STEP", ML + 5, W.y + 6.6, {charSpace: 0.3});
    W.font("normal", 9.5, C.ink); nextLines.forEach((ln, i) => doc.text(ln, ML + 29, W.y + 6.6 + i * W.lh(9.5)));
    W.y += nh + 5;

    // ── key measures ──
    if (gradeable) {
      const p = r.referable_probability ?? NaN, u = r.uncertainty || {};
      const dme = r.dme_risk ?? 0;
      const cells = [
        ["P(referable)", num(p, 3), `threshold ${threshold.toFixed(3)}: ${p >= threshold ? "above, refer" : "below"}`],
        ["Confidence in the grade", pct(r.confidence), `probability of grade ${r.grade}`],
        ["Sight-threatening", pct(r.sight_threatening_probability), "probability of severe or proliferative disease"],
        ["Macular oedema risk", `${dme} of 2`, ["no hard exudates near the macula", "exudates, but over 1 disc diameter from the fovea", "exudates within 1 disc diameter of the fovea"][dme] || ""],
        ["Rule-based grade", String(r.rule_based_grade ?? "-"), r.agreement ? `${String(r.agreement).replace(/_/g, " ")} with the model` : ""],
        ["Uncertainty", num(u.entropy, 3), `entropy; epistemic variance ${num(u.epistemic_variance, 4)}`]];
      const cw = CW / 3, ch = 16;
      W.ensure(ch * 2 + 2);
      cells.forEach(([k, val, sub], i) => {
        const x = ML + (i % 3) * cw, y = W.y + Math.floor(i / 3) * ch;
        doc.setDrawColor(...C.rule); doc.setLineWidth(0.25); doc.rect(x, y, cw, ch, "S");
        W.label(k, x + 3.5, y + 4.6);
        W.font("bold", 13, C.ink); doc.text(val, x + 3.5, y + 10.5);
        W.font("normal", 7, C.muted); doc.text(doc.splitTextToSize(clean(sub), cw - 7)[0], x + 3.5, y + 14);
      });
      W.y += ch * 2 + 4.5;
    }

    // ── cautions and recapture instructions ──
    const warn = gradeable ? cautions(r, !!pics.lesions) : [];
    const advice = [].concat(r.recapture_advice || [], r.quality?.advice || []).filter((a, i, all) => a && all.indexOf(a) === i);
    if (advice.length) {
      W.ensure(14); W.label("Recapture instructions", ML, W.y + 3, TONE.recapture); W.y += 5.5;
      advice.forEach(a => W.item(a, {mark: TONE.recapture}));
      W.y += 1;
    }
    if (warn.length) {
      W.ensure(14); W.label("Read with care", ML, W.y + 3, TONE.refer); W.y += 5.5;
      warn.forEach(a => W.item(a, {mark: TONE.refer, size: 9}));
    }

    // ── 1 images ──
    // two images side by side, each with a title, an optional colour key and a caption
    const nvUnassessed = (r.clinical_features?.unassessed || []).includes("neovascularization");
    const keyRows = width => {
      W.font("normal", 7.3);
      const rows = [[]]; let used = 0;
      LESION.forEach(([k, name, rgb]) => {
        const txt = k === "neovascularization" && nvUnassessed ? "Neovascularisation (not assessed)" : k === "soft_exudate" ? "Soft exudates" : name;
        const w = 3.9 + doc.getTextWidth(txt);
        if (used && used + w > width) { rows.push([]); used = 0; }
        rows[rows.length - 1].push([txt, rgb, used]); used += w + 4.5;
      });
      return rows;
    };
    const pair = (a, b) => {
      const size = 80, gap = 8, subLh = 3.3, keyLh = 4.2;
      const x0 = b ? ML + (CW - 2 * size - gap) / 2 : ML + (CW - size) / 2;   // one image sits in the middle
      const parts = [a, b].map(it => {
        if (!it) return null;
        W.font("normal", 7.6);
        const key = it.key ? keyRows(size) : [], subs = doc.splitTextToSize(clean(it.sub), size).slice(0, 3);
        const keyH = key.length ? key.length * keyLh + 0.6 : 0;
        return {it, key, subs, keyH, h: 9 + keyH + (subs.length - 1) * subLh + 2.5};
      });
      const capH = Math.max(...parts.map(p => p ? p.h : 0));
      W.ensure(size + capH);
      parts.forEach((p, i) => {
        if (!p) return;
        const x = x0 + i * (size + gap), y = W.y + size;
        if (p.it.img) doc.addImage(p.it.img, "JPEG", x, W.y, size, size, undefined, "FAST");
        else { doc.setFillColor(...C.soft); doc.rect(x, W.y, size, size, "F"); W.font("normal", 8.5, C.muted); doc.text("Not available for this run", x + size / 2, W.y + size / 2, {align: "center"}); }
        W.font("bold", 9, C.ink); doc.text(clean(p.it.title), x, y + 5);
        p.key.forEach((row, k) => row.forEach(([txt, rgb, dx]) => {
          const ky = y + 6.9 + k * keyLh;
          doc.setFillColor(...rgb); doc.setDrawColor(...C.muted); doc.setLineWidth(0.15); doc.rect(x + dx, ky, 2.7, 2.7, "FD");
          W.font("normal", 7.3, C.body); doc.text(txt, x + dx + 3.9, ky + 2.3);
        }));
        W.font("normal", 7.6, C.muted); p.subs.forEach((ln, k) => doc.text(ln, x, y + 9 + p.keyH + k * subLh));
      });
      W.y += size + capH;
    };
    let sec = 1;
    const shown = gradeable || pics.lesions;                  // an ungradeable image has nothing more to show
    W.section(sec++, "Images", shown ? "The photograph as received, and what the system saw on the enhanced 512-pixel image the measurements use."
      : "The photograph as received. It was stopped at the quality gate, so there are no lesion outlines or attention map.", 104);
    pair({img: pics.original, title: "Photograph as received", sub: clean(o.label || r.image_id || "")},
         shown ? {img: pics.lesions, title: "Detected lesions", key: !!pics.lesions, sub: "Green ring: optic disc; green cross: fovea; the wider green ring is 1 disc diameter around the fovea."} : null);
    if (pics.enhanced || pics.attention)
      pair({img: pics.enhanced, title: "Enhanced image", sub: (r.enhancement_applied || []).length ? `Corrections applied: ${corrections(r)}.` : "No enhancement was needed."},
           {img: pics.attention, title: "Model attention (Grad-CAM++)", sub: "Where the grader looked. Warm colours mark the regions that drove the grade."});

    if (gradeable) {
      // ── 2 severity ──
      W.section(sec++, "Severity", "Probability of each grade on the International Clinical Diabetic Retinopathy (ICDR) scale. Grade 2 and above is referable.");
      const probs = r.class_probabilities || [], top = probs.indexOf(Math.max(...probs));
      const barX = ML + 58, barW = CW - 58 - 18, rowH = 6.4, gap = 5;
      W.ensure(probs.length * rowH + gap + 8);
      probs.forEach((pv, g) => {
        const y = W.y + g * rowH + (g >= 2 ? gap : 0);
        if (g === 2) {
          // grades 2 to 4 are referable: a labelled rule in the gap above them
          const ly = y - gap / 2 + 0.6, lab = "REFERABLE: GRADE 2 AND ABOVE";
          W.font("bold", 6.3, C.muted);
          const lw = doc.getTextWidth(lab) + 0.2 * (lab.length - 1);
          doc.setDrawColor(...C.faint); doc.setLineDashPattern([0.8, 0.8], 0); doc.setLineWidth(0.25);
          doc.line(ML, ly, ML + CW - lw - 3, ly); doc.setLineDashPattern([], 0);
          W.rtext(lab, ML + CW, ly + 0.8, 0.2);
        }
        W.font(g === top ? "bold" : "normal", 9, g === top ? C.ink : C.body);
        doc.text(`${g}  ${GRADES[g]}`, ML, y + 4);
        doc.setFillColor(...C.soft); doc.rect(barX, y + 1, barW, 3.8, "F");
        doc.setFillColor(...GRADE_RGB[g]); doc.rect(barX, y + 1, Math.max(0.4, barW * pv), 3.8, "F");
        W.font(g === top ? "bold" : "normal", 9, C.ink); doc.text(pct(pv), ML + CW, y + 4, {align: "right"});
      });
      W.y += probs.length * rowH + gap + 4;

      // ── 3 lesions ──
      const cf = r.clinical_features || {}, pq = cf.per_quadrant || {}, counts = cf.counts || {};
      const unassessed = new Set(cf.unassessed || []);
      const area = {};
      (r.evidence || []).forEach(e => { if (e.finding && e.area_percent != null) area[e.finding.replace(/ /g, "_")] = e.area_percent; });
      W.section(sec++, "Lesions", "Counted by the lesion model in each quarter of the retina, around the fovea. The ICDR criteria for severe disease are written in these quantities.");
      const Q = ["superior", "inferior", "nasal", "temporal"];
      W.table([{h: "Lesion", w: 58}, {h: "Total", w: 18, align: "right"}, {h: "Superior", w: 21, align: "right"}, {h: "Inferior", w: 21, align: "right"},
        {h: "Nasal", w: 18, align: "right"}, {h: "Temporal", w: 21, align: "right"}, {h: "Retina", w: 21, align: "right"}],
        LESION.map(([k, name]) => unassessed.has(k)
          ? [name, {text: "not assessed: the lesion model was never trained to outline this class", span: 6, align: "left", color: TONE.refer, style: "bold"}]
          : [name, String(counts[k] ?? 0), ...Q.map(q => String((pq[k] || {})[q] ?? 0)), area[k] != null ? area[k].toFixed(2) + "%" : "-"]),
        {zebra: true});
      // the quadrants, as a diagram, beside the facts that go with it
      const lm = r.landmarks || {}, frame = o.frame || 512;
      const nasalRight = lm.disc_xy ? lm.disc_xy[0] > frame / 2 : true;
      const R = 22, cx = ML + R + 4;
      W.ensure(2 * R + 20);
      W.label("Where the lesions are", ML, W.y + 4); W.y += 5;
      const cy2 = W.y + R + 6;
      doc.setDrawColor(...C.muted); doc.setLineWidth(0.3); doc.circle(cx, cy2, R, "S");
      const d = R * Math.SQRT1_2;
      doc.setLineWidth(0.2); doc.line(cx - d, cy2 - d, cx + d, cy2 + d); doc.line(cx - d, cy2 + d, cx + d, cy2 - d);
      doc.setFillColor(...C.white); doc.circle(cx, cy2, 2.2, "FD");
      const at = {superior: [cx, cy2 - R * 0.55], inferior: [cx, cy2 + R * 0.62], nasal: [cx + (nasalRight ? 1 : -1) * R * 0.58, cy2 + 1], temporal: [cx + (nasalRight ? -1 : 1) * R * 0.58, cy2 + 1]};
      Q.forEach(q => {
        const [x, y] = at[q];
        W.font("bold", 5.6, C.muted); doc.text(q.toUpperCase(), x, y - 2.6, {align: "center", charSpace: 0.2});
        const parts = [["microaneurysm", LESION[0][2]], ["hemorrhage", LESION[1][2]], ["hard_exudate", [190, 150, 0]]];
        const strs = parts.map(([k]) => String((pq[k] || {})[q] ?? 0)), sep = " · ";
        W.font("bold", 7.2);
        const total = strs.reduce((s, t) => s + doc.getTextWidth(t), 0) + 2 * doc.getTextWidth(sep);
        let tx = x - total / 2;
        strs.forEach((t, i) => { doc.setTextColor(...parts[i][1]); doc.text(t, tx, y + 0.8); tx += doc.getTextWidth(t);
          if (i < 2) { doc.setTextColor(...C.faint); doc.text(sep, tx, y + 0.8); tx += doc.getTextWidth(sep); } });
      });
      W.font("normal", 6.5, C.muted); doc.text(`${lm.laterality || "?"} eye, nasal on the ${nasalRight ? "right" : "left"}`, cx, cy2 + R + 4.5, {align: "center"});
      // the facts
      const fx = ML + 2 * R + 16, fw = CW - (fx - ML);
      const facts = [
        ["Quadrants with haemorrhage", `${cf.quadrants_with_hemorrhage ?? "-"} of 4`],
        ["Lesions within 1 disc diameter of the fovea", String(cf.lesions_within_1dd_of_fovea ?? "-")],
        ["Nearest lesion to the fovea", cf.nearest_lesion_dd != null ? `${cf.nearest_lesion_dd} disc diameters` : "-"],
        ["Venous beading", `${cf.quadrants_with_beading ?? 0} quadrants`],
        ["Neovascularisation", unassessed.has("neovascularization") ? "not assessed" : (cf.nv_at_disc || cf.nv_elsewhere) ? `detected${cf.nv_at_disc ? " at the disc" : ""}${cf.nv_elsewhere ? " elsewhere" : ""}` : "none detected"]];
      let fy = cy2 - R + 2;
      W.font("normal", 6.5, C.muted); doc.text("Counts in each quarter:  microaneurysms · haemorrhages · hard exudates", fx, fy); fy += 6;
      facts.forEach(([k, val]) => {
        W.font("normal", 8.8, C.body); doc.text(clean(k), fx, fy);
        W.font("bold", 8.8, k === "Neovascularisation" && val === "not assessed" ? TONE.refer : C.ink); doc.text(clean(val), fx + fw, fy, {align: "right"});
        doc.setDrawColor(...C.rule); doc.setLineWidth(0.15); doc.line(fx, fy + 1.8, fx + fw, fy + 1.8);
        fy += 6.4;
      });
      W.y = Math.max(cy2 + R + 8, fy);

      // ── 4 clinical evidence ──
      W.section(sec++, "Clinical evidence", "The findings in words, as the ICDR criteria read them, and the rule-based grade the system checks the model against.");
      const ev = r.evidence || [];
      ev.filter(e => e.criterion).forEach(e => W.item(e.criterion));
      ev.filter(e => e.macular_assessment).forEach(e => W.item("Macula: " + e.macular_assessment, {mark: (r.dme_risk || 0) >= 2 ? TONE.urgent : C.accent}));
      ev.filter(e => e.caution).forEach(e => W.item(e.caution, {mark: TONE.refer}));
      if (!ev.some(e => e.criterion || e.macular_assessment || e.caution)) W.item("No lesions detected.");
      W.y += 2;
      W.table([{h: "Cross-check", w: 90}, {h: "Value", w: CW - 90}], [
        ["Grade from the deep model", `${r.grade} (${GRADES[r.grade]})`],
        ["Grade from the ICDR rules", r.rule_based_grade != null ? `${r.rule_based_grade} (${GRADES[r.rule_based_grade] || "-"})` : "-"],
        ["Agreement", String(r.agreement || "-").replace(/_/g, " ")],
        ["Macular oedema risk (0 to 2)", String(r.dme_risk ?? "-")]], {head: true});
    }

    // ── 5 landmarks ──
    const lm = r.landmarks || {};
    if (lm.disc_xy || lm.fovea_xy) {
      W.section(sec++, "Landmarks", "The optic disc and the fovea set the frame the lesions are measured in. Positions are in pixels of the 512-pixel working image.");
      const dd = lm.disc_diameter_px, fd = lm.disc_xy && lm.fovea_xy ? Math.hypot(lm.disc_xy[0] - lm.fovea_xy[0], lm.disc_xy[1] - lm.fovea_xy[1]) : NaN;
      W.table([{h: "Landmark", w: 58}, {h: "Value", w: CW - 58}], [
        ["Eye", lm.laterality === "OD" ? "OD (right eye)" : lm.laterality === "OS" ? "OS (left eye)" : String(lm.laterality || "-")],
        ["Optic disc centre", lm.disc_xy ? `x ${Math.round(lm.disc_xy[0])}, y ${Math.round(lm.disc_xy[1])}; detection confidence ${pct(lm.disc_confidence, 0)}` : "-"],
        ["Optic disc diameter", Number.isFinite(dd) ? `${dd.toFixed(1)} px (1 disc diameter, the unit distances are measured in)` : "-"],
        ["Fovea centre", lm.fovea_xy ? `x ${Math.round(lm.fovea_xy[0])}, y ${Math.round(lm.fovea_xy[1])}; detection confidence ${pct(lm.fovea_confidence, 0)}` : "-"],
        ["Disc to fovea", Number.isFinite(fd) && dd ? `${(fd / dd).toFixed(2)} disc diameters` : "-"]]);
    }

    // ── 6 image quality ──
    const q = r.quality || {}, s = q.scores || {}, vd = q.verdicts || {}, th = (health && health.quality_thresholds) || o.qualityThresholds || QTH;
    W.section(sec++, "Image quality", `Nine checks run before any grading. A score below the first threshold fails; below the second is borderline. Overall: ${q.overall || "-"}${q.first_pass && q.first_pass.overall && q.first_pass.overall !== q.overall ? ` (first pass: ${q.first_pass.overall}${(q.first_pass.issues || []).length ? ", " + q.first_pass.issues.join(", ") : ""}; re-checked after enhancement)` : ""}.`);
    const qx = ML + 44, qw = CW - 44 - 44, qh = 6.2;
    W.ensure(QUALITY.length * qh + 6);
    W.font("bold", 7.2, C.muted);
    doc.text("CHECK", ML, W.y + 3, {charSpace: 0.2}); doc.text("SCORE (0 TO 1)", qx, W.y + 3, {charSpace: 0.2});
    W.rtext("SCORE", ML + CW - 22, W.y + 3, 0.2); W.rtext("RESULT", ML + CW, W.y + 3, 0.2);
    W.y += 5;
    QUALITY.filter(([k]) => k in s).forEach(([k, name]) => {
      const y = W.y, val = s[k], [f, b] = th[k] || [0, 0], verdictK = vd[k] || "pass";
      const col = verdictK === "fail" ? TONE.urgent : verdictK === "borderline" ? TONE.refer : TONE.routine;
      W.font("normal", 8.8, C.body); doc.text(name, ML, y + 3.9);
      doc.setFillColor(...C.soft); doc.rect(qx, y + 1.3, qw, 3.2, "F");
      doc.setFillColor(...col); doc.rect(qx, y + 1.3, Math.max(0.4, qw * Math.min(1, val)), 3.2, "F");
      doc.setDrawColor(...C.ink); doc.setLineWidth(0.25);
      doc.line(qx + qw * f, y + 0.6, qx + qw * f, y + 5.2);
      doc.setLineDashPattern([0.6, 0.6], 0); doc.line(qx + qw * b, y + 0.6, qx + qw * b, y + 5.2); doc.setLineDashPattern([], 0);
      W.font("bold", 8.8, C.ink); doc.text(num(val, 2), ML + CW - 22, y + 3.9, {align: "right"});
      W.font("bold", 8.8, col); doc.text(verdictK, ML + CW, y + 3.9, {align: "right"});
      W.y += qh;
    });
    W.font("normal", 7, C.muted); doc.text("Solid tick: fails below. Dashed tick: borderline below.", qx, W.y + 3.2);
    W.y += 6;
    W.para(`Gate confidence ${pct(q.confidence, 0)}. ${(r.enhancement_applied || []).length ? `Corrections applied before grading: ${corrections(r)}.` : "No corrections were needed."}${(q.issues || []).length ? " Issues flagged: " + q.issues.join(", ") + "." : ""}`, {size: 8.8, color: C.body});

    // ── 7 processing ──
    const t = r.timing_ms || {};
    const stages = Object.keys(t).filter(k => k !== "total");
    if (stages.length) {
      W.section(sec++, "Processing record", `Every stage records its own time. ${runtime}.`);
      const total = t.total || stages.reduce((a, k) => a + t[k], 0);
      W.table([{h: "Stage", w: 70}, {h: "Share of the run", w: CW - 70 - 26}, {h: "Time", w: 26, align: "right"}],
        stages.map(k => [STAGES[k] || cap(k.replace(/_/g, " ")), `${(100 * t[k] / total).toFixed(1)}%`, ms(t[k])])
          .concat([[{text: "Total", style: "bold"}, "", {text: ms(total), style: "bold", color: C.ink}]]), {size: 8.6});
    }

    // ── 8 provenance and reference ──
    const gt = r.ground_truth, hasGt = gt && gt.grade != null;
    W.section(sec++, "Source and reference", "Where the photograph came from and, for the held-out example photographs, the grade the reference graders gave it.");
    const src = [["Image identifier", clean(r.image_id || "-")], ["Run", runtime],
      ["Model", `${clean(r.model_version || "-")}; bundle ${bundle}`], ["Referral threshold", `${threshold.toFixed(4)} on P(referable), fixed before testing`]];
    if (r.provenance) src.push(["Provenance", clean(r.provenance)]);
    if (hasGt) {
      const dg = gradeable ? Math.abs(gt.grade - r.grade) : null;
      src.push(["Reference grade", `${gt.grade} (${GRADES[gt.grade]})`],
        ["Agreement with the reference", dg == null ? "-" : dg === 0 ? "exact" : dg === 1 ? "within one grade" : `off by ${dg} grades`]);
      if (gt.source) src.push(["Corpus", clean(gt.source === "aptos2019" ? "APTOS-2019" : gt.source === "idrid_grading" ? "IDRiD" : gt.source)]);
      if (gt.subject) src.push(["Subject", clean(gt.subject)]);
    }
    W.table([{h: "Item", w: 58}, {h: "Detail", w: CW - 58}], src, {size: 8.8});

    // ── 9 ophthalmologist review and sign-off ──
    W.section(sec++, "Ophthalmologist review", "To be completed by the reviewing ophthalmologist. Every referable and every sight-threatening finding is reviewed before any clinical action.", o.review ? 68 : 60);
    if (o.review) {
      const ag = String(o.review.agreement || "").replace(/_/g, " ");
      W.para(`Recorded in the console: the reviewer graded ${o.review.grade} (${GRADES[o.review.grade] || "-"}) against the model's ${o.review.model ?? "-"}${ag ? `, agreement ${ag}` : ""}${Number.isFinite(o.review.seconds) ? `, in ${o.review.seconds.toFixed(1)} s` : ""}.`, {size: 9, color: C.ink, style: "bold"});
    }
    W.ensure(58);
    const fy0 = W.y + 2;
    W.font("normal", 8.8, C.body);
    doc.text("Confirmed ICDR grade", ML, fy0 + 4);
    [0, 1, 2, 3, 4].forEach(g => { const x = ML + 44 + g * 17; doc.setDrawColor(...C.muted); doc.setLineWidth(0.3); doc.rect(x, fy0 + 0.8, 4, 4, "S");
      if (o.review && o.review.grade === g) { doc.setFillColor(...C.ink); doc.rect(x + 0.9, fy0 + 1.7, 2.2, 2.2, "F"); }
      W.font("normal", 8.8, C.body); doc.text(String(g), x + 5.6, fy0 + 4); });
    doc.text("Plan", ML, fy0 + 13);
    ["Agree with the screening result", "Change the plan (write below)"].forEach((t2, i) => { const x = ML + 44 + i * 64;
      doc.setDrawColor(...C.muted); doc.rect(x, fy0 + 9.8, 4, 4, "S"); W.font("normal", 8.8, C.body); doc.text(t2, x + 5.6, fy0 + 13); });
    const line = (label, y, x = ML, w = CW) => { W.font("normal", 7.2, C.muted); doc.text(label, x, y); doc.setDrawColor(...C.rule); doc.setLineWidth(0.3); doc.line(x, y - 4.2, x + w, y - 4.2); };
    doc.setDrawColor(...C.rule); doc.setLineWidth(0.3);
    doc.line(ML, fy0 + 24, ML + CW, fy0 + 24); doc.line(ML, fy0 + 31, ML + CW, fy0 + 31);
    W.font("normal", 7.2, C.muted); doc.text("Notes", ML, fy0 + 19);
    const half = (CW - 10) / 2;
    line("Name of the ophthalmologist", fy0 + 44, ML, half); line("Registration number", fy0 + 44, ML + half + 10, half);
    line("Signature", fy0 + 55, ML, half); line("Date", fy0 + 55, ML + half + 10, half);
    W.y = fy0 + 60;

    // ── about this report ──
    W.section(sec++, "About this report");
    W.table([{h: "ICDR grade", w: 22}, {h: "Meaning", w: 96}, {h: "Referable", w: CW - 22 - 96}],
      GRADE_LONG.map((g, i) => [String(i), g, i >= 2 ? (i >= 3 ? "yes; sight-threatening" : "yes") : "no"]), {size: 8.4, zebra: true});
    [["P(referable)", "the calibrated probability that the grade is 2 or higher. The patient is referred when it reaches the referral threshold."],
     ["Confidence", "the probability the model gives to the grade it chose."],
     ["Uncertainty", "entropy across the five grades, and the epistemic variance from repeated passes with dropout; high values mean the model is unsure."],
     ["Macular oedema risk", "0: no hard exudates near the macula; 1: exudates more than 1 disc diameter from the fovea; 2: exudates within 1 disc diameter of the fovea, clinically significant."],
     ["Service targets", "urgent referrals seen within 2 days and others within 14 days, from the programme model; a clear screen is repeated at 12 months."]]
      .forEach(([k, val]) => W.item(`${k}: ${val}`, {size: 8.6}));
    W.y += 2;
    W.para("This report was produced by automated screening software to support a clinician. It is not a diagnosis and does not replace an eye examination by a qualified ophthalmologist. Neovascularisation is not assessed by the lesion model; a full examination is advised whenever proliferative disease is suspected.", {size: 8.4, color: C.muted});

    // ── running header and footer on every page ──
    const n = doc.getNumberOfPages();
    for (let i = 1; i <= n; i++) {
      doc.setPage(i);
      if (i > 1) {
        W.font("bold", 6.8, C.muted); doc.text("DR SCREENING REPORT", ML, 10, {charSpace: 0.25});
        W.font("normal", 7.5, C.muted); doc.text(clean(`${o.label || r.image_id || ""}  ·  ${v.label}${gradeable ? "  ·  grade " + r.grade : ""}`), ML + CW, 10, {align: "right"});
        doc.setDrawColor(...C.rule); doc.setLineWidth(0.2); doc.line(ML, 12, ML + CW, 12);
      }
      doc.setDrawColor(...C.rule); doc.setLineWidth(0.2); doc.line(ML, PH - 13.5, ML + CW, PH - 13.5);
      W.font("normal", 6.8, C.muted);
      doc.text("Decision support only, not a diagnosis. Review by a qualified ophthalmologist before clinical action.", ML, PH - 9.5);
      doc.text(`${reportId}   ·   page ${i} of ${n}`, ML + CW, PH - 9.5, {align: "right"});
    }
    return {doc, name: `DR-screening-report_${String(r.image_id || "image").replace(/[^A-Za-z0-9_-]/g, "_")}_${day}.pdf`};
  }

  async function download(r, o) {
    const {doc, name} = await build(r, o);
    doc.save(name);
    return name;
  }

  // the page kit, shared with the dossier's documents (docs-pdf.js)
  const kit = {load: jsPDFClass, Writer, clean, pct, num, loadImage,
    C, TONE, GRADE_RGB, GRADES, GRADE_LONG, geometry: {PW, PH, ML, MR, TOP, BOTTOM, CW, PT}};
  root.DRReport = {build, download, kit};
})(typeof self !== "undefined" ? self : globalThis);
