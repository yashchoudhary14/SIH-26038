/* The dossier's two documents as PDFs, built in the browser: the build notes
   and the results and conclusions. They use the page kit of report-pdf.js
   (jsPDF, loaded on first use), so they look like the screening report and
   need no backend.

     DRDocs.download("build")      the build notes
     DRDocs.download("results")    the results and conclusions

   The figures are the ones RESULTS.md and matlab/README.md record, with the
   model they belong to stated beside them. The headline model is the
   pre-pool bundle (APTOS-2019 + IDRiD, Messidor-2 held out), which is also
   the bundle launch_web serves by default.
*/
(function (root) {
  "use strict";
  const VSET = "../outputs/verification_set/";

  // the Simulink district model with the variables as set (simulink/district_model_params.m),
  // MATLAB's 100 replications (simulink/validation/ref_model.json)
  const DISTRICT = {arrived: 119.3, cleared: 79.3, reviewed: 39.9, techBusy: 35.9, deviceBusy: 1.6, doctorBusy: 6.6, agree: "24 of 24"};

  function kit() {
    if (!root.DRReport || !root.DRReport.kit) throw new Error("report-pdf.js must load before docs-pdf.js");
    return root.DRReport.kit;
  }

  // ── building blocks shared by both documents ────────────────────────────
  function blocks(doc, K) {
    const {C, TONE} = K, {PW, PH, ML, CW, PT} = K.geometry, W = K.Writer(doc), clean = K.clean;

    const title = o => {
      W.font("bold", 7.5, C.accent); doc.text("SIXEYES  ·  DIABETIC RETINOPATHY SCREENING", ML, W.y + 2.5, {charSpace: 0.3});
      W.font("bold", 21, C.ink); doc.text(clean(o.title), ML, W.y + 12);
      W.font("normal", 9, C.muted); doc.text(clean(o.sub), ML, W.y + 17.5);
      let my = W.y + 1.5;
      o.meta.forEach(([k, v]) => {
        W.font("bold", 6.8, C.muted); doc.text(k.toUpperCase(), ML + CW - 70, my + 2.2, {charSpace: 0.2});
        W.font("normal", 8.2, C.ink); doc.text(doc.splitTextToSize(clean(v), 48)[0], ML + CW, my + 2.2, {align: "right"});
        my += 4.6;
      });
      W.y += 25;
    };
    // the coloured statement band, with an optional white box of figures on the right
    const band = o => {
      const tone = o.tone || C.accent, avail = o.box ? CW - 78 : CW - 12;
      let ts = 17; W.font("bold", ts);
      while (ts > 12 && doc.getTextWidth(clean(o.title)) > avail) { ts -= 0.5; W.font("bold", ts); }
      W.font("normal", 8.8); const subLines = doc.splitTextToSize(clean(o.sub), avail);
      const h = Math.max(o.h || 30, 21.5 + subLines.length * 3.9 + 1.5);
      doc.setFillColor(...tone); doc.roundedRect(ML, W.y, CW, h, 2, 2, "F");
      W.font("bold", 7.5, C.white); doc.text(clean(o.label).toUpperCase(), ML + 6, W.y + 7, {charSpace: 0.3});
      W.font("bold", ts, C.white); doc.text(clean(o.title), ML + 6, W.y + 15.5);
      W.font("normal", 8.8, C.white); doc.text(subLines, ML + 6, W.y + 21.5, {lineHeightFactor: 1.25});
      if (o.box) {
        const bx = ML + CW - 66, by = W.y + 4.5, bw = 60, bh = h - 9, n = o.box.length, cw = bw / n;
        doc.setFillColor(...C.white); doc.roundedRect(bx, by, bw, bh, 1.6, 1.6, "F");
        o.box.forEach(([big, lab], i) => {
          const x = bx + i * cw + 4;
          W.font("bold", 17, tone); doc.text(clean(big), x, by + 11);
          W.font("normal", 6.8, C.muted); doc.text(doc.splitTextToSize(clean(lab), cw - 6), x, by + 15.5);
        });
      }
      W.y += h + 4;
    };
    // a grid of key figures: [label, value, note, ok?]
    const kpis = (cells, cols = 3) => {
      const cw = CW / cols, ch = 16.5, rows = Math.ceil(cells.length / cols);
      W.ensure(rows * ch + 3);
      cells.forEach(([k, v, sub, ok], i) => {
        const x = ML + (i % cols) * cw, y = W.y + Math.floor(i / cols) * ch;
        doc.setDrawColor(...C.rule); doc.setLineWidth(0.25); doc.rect(x, y, cw, ch, "S");
        W.label(k, x + 3.5, y + 4.8);
        W.font("bold", 13, ok ? TONE.routine : C.ink); doc.text(clean(v), x + 3.5, y + 10.8);
        W.font("normal", 7, C.muted); doc.text(doc.splitTextToSize(clean(sub), cw - 7)[0], x + 3.5, y + 14.4);
      });
      W.y += rows * ch + 5;
    };
    // a tinted box with a coloured rule on the left
    const callout = (label, text, tone = C.accent) => {
      W.font("normal", 9.2);
      const lines = W.lines(text.replace(/\*\*/g, ""), CW - 14), h = lines.length * W.lh(9.2) + 11;
      W.ensure(h + 2);
      doc.setFillColor(...C.soft); doc.rect(ML, W.y, CW, h, "F");
      doc.setFillColor(...tone); doc.rect(ML, W.y, 1.2, h, "F");
      W.label(label, ML + 6, W.y + 5.5, tone);
      const y0 = W.y; W.y += 7.5;
      W.rich(text, {x: ML + 6, width: CW - 12, size: 9.2, color: C.ink, after: 0});
      W.y = y0 + h + 4;
    };
    // horizontal bars: rows [label, value, text, colour] on a scale from o.min (0) to o.max (1);
    // target: a dashed line at that value
    const bars = (rows, o = {}) => {
      const lw = o.labelW || 56, tw = o.textW || 26, bx = ML + lw, bw = CW - lw - tw, rh = o.rowH || 7;
      const lo = o.min || 0, hi = o.max || 1, f = v => Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
      W.ensure(rows.length * rh + 8);
      rows.forEach(([lab, v, txt, col], i) => {
        const y = W.y + i * rh;
        W.font(o.boldFirst && i === 0 ? "bold" : "normal", 8.8, C.body); doc.text(clean(lab), ML, y + 4.4);
        doc.setFillColor(...C.soft); doc.rect(bx, y + 1.4, bw, 4, "F");
        doc.setFillColor(...(col || C.accent)); doc.rect(bx, y + 1.4, Math.max(0.4, bw * f(v)), 4, "F");
        W.font("bold", 8.8, C.ink); doc.text(clean(txt), ML + CW, y + 4.4, {align: "right"});
      });
      if (lo > 0) { W.font("normal", 6.3, C.muted); doc.text(`scale from ${lo}`, bx, W.y + rows.length * rh + 3); }
      if (o.target != null) {
        const tx = bx + bw * f(o.target), top = W.y + 0.4, bot = W.y + rows.length * rh;
        doc.setDrawColor(...C.ink); doc.setLineWidth(0.3); doc.setLineDashPattern([0.8, 0.7], 0);
        doc.line(tx, top, tx, bot); doc.setLineDashPattern([], 0);
        W.font("bold", 6.3, C.ink); doc.text(clean(o.targetLabel || "target"), tx + 1, bot + 3.2);
      }
      W.y += rows.length * rh + (o.target != null || lo > 0 ? 6.5 : 3);
    };
    // a monospace block, for commands
    const code = lines => {
      const lh = 4.1, h = lines.length * lh + 5;
      W.ensure(h + 2);
      doc.setFillColor(...C.soft); doc.roundedRect(ML, W.y, CW, h, 1.2, 1.2, "F");
      doc.setFont("courier", "normal"); doc.setFontSize(8.2); doc.setTextColor(...C.ink);
      lines.forEach((l, i) => doc.text(clean(l), ML + 4, W.y + 4.6 + i * lh));
      W.y += h + 3.5;
    };
    const sub = text => { W.ensure(12); W.y += 1; W.label(text, ML, W.y + 3); W.y += 5.5; };
    // header and footer on every page
    const decorate = (short, id) => {
      const n = doc.getNumberOfPages();
      for (let i = 1; i <= n; i++) {
        doc.setPage(i);
        if (i > 1) {
          W.font("bold", 6.8, C.muted); doc.text(clean(short).toUpperCase(), ML, 10, {charSpace: 0.25});
          W.font("normal", 7.5, C.muted); doc.text("SixEyes  ·  SIH-26038", ML + CW, 10, {align: "right"});
          doc.setDrawColor(...C.rule); doc.setLineWidth(0.2); doc.line(ML, 12, ML + CW, 12);
        }
        doc.setDrawColor(...C.rule); doc.setLineWidth(0.2); doc.line(ML, PH - 13.5, ML + CW, PH - 13.5);
        W.font("normal", 6.8, C.muted);
        doc.text("Decision-support research prototype. Not a diagnosis; every referral is reviewed by an ophthalmologist.", ML, PH - 9.5);
        doc.text(`${id}   ·   page ${i} of ${n}`, ML + CW, PH - 9.5, {align: "right"});
      }
    };
    return {W, title, band, kpis, callout, bars, code, sub, decorate};
  }

  // the three review frames of a recorded panel, as square JPEGs
  async function panelFrames(K, url) {
    const p = await K.loadImage(url);
    if (!p) return {};
    const out = {}, fw = p.naturalWidth / 3, y0 = Math.max(0, (p.naturalHeight - fw) * 0.469), e = 2;
    ["enhanced", "lesions", "attention"].forEach((k, i) => {
      try {
        const cv = document.createElement("canvas"); cv.width = cv.height = 700;
        const g = cv.getContext("2d"); g.fillStyle = "#000"; g.fillRect(0, 0, 700, 700);
        g.drawImage(p, i * fw + e, y0 + e, fw - 2 * e, fw - 2 * e, 0, 0, 700, 700);
        out[k] = cv.toDataURL("image/jpeg", 0.88);
      } catch (err) { /* a tainted canvas: leave the frame out */ }
    });
    return out;
  }

  function stamp() {
    const d = new Date(), p2 = k => String(k).padStart(2, "0");
    const day = `${d.getFullYear()}-${p2(d.getMonth() + 1)}-${p2(d.getDate())}`;
    return {day, when: d.toLocaleDateString("en-IN", {day: "numeric", month: "short", year: "numeric"})};
  }

  // ── the results and conclusions ─────────────────────────────────────────
  async function results() {
    const K = kit(), JsPDF = await K.load(), {C, TONE, GRADE_RGB} = K, {ML, CW} = K.geometry;
    const doc = new JsPDF({unit: "mm", format: "a4", compress: true});
    const B = blocks(doc, K), W = B.W, t = stamp();
    doc.setProperties({title: "SixEyes - Results and conclusions", subject: "Diabetic retinopathy screening: measured results",
      author: "Team SixEyes", creator: "SixEyes project dossier", keywords: "diabetic retinopathy, ICDR, screening, MATLAB, Simulink"});

    B.title({title: "Results and conclusions", sub: "What the system achieves, and how it was measured.",
      meta: [["Problem statement", "SIH-26038, MedTech"], ["Team", "SixEyes"], ["Model", "drscreen-1.0.0, prepool bundle"], ["Generated", t.when]]});
    B.band({tone: TONE.routine, label: "Headline", title: "Both screening targets met",
      sub: "Referable diabetic retinopathy (ICDR grade 2 or above), on held-out patients the model never saw in training (n = 631).",
      box: [["0.986", "sensitivity, target 0.90"], ["0.873", "specificity, target 0.85"]]});
    B.kpis([
      ["Sensitivity", "0.986", "target >= 0.90: met", true],
      ["Specificity", "0.873", "target >= 0.85: met", true],
      ["AUC", "0.988", "95% CI 0.979 to 0.994"],
      ["Sight-threatening, internal", "105 / 105", "every grade 3-4 eye referred", true],
      ["Sight-threatening, zero-shot", "97.3%", "107 of 110 on Messidor-2, nothing fitted on it", true],
      ["Time per photograph", "2.6 s", "MATLAB on an ordinary CPU, offline"]]);
    B.callout("In short", "The system meets both screening targets on held-out patients, refers **every sight-threatening eye** in the internal test and **97.3%** of them on a French cohort it never saw, explains each referral with lesion evidence, runs in MATLAB on a CPU in about **2.6 seconds**, and lets **one district ophthalmologist** keep up with 100,000 patients a year.");

    let sec = 1;
    // 1 screening performance
    W.section(sec++, "Screening performance", "Referable DR (grade 2 or above). The internal test is subject-disjoint from training; the external test is Messidor-2, a French cohort on different cameras with a different grading panel, with nothing fitted on it.");
    W.table([{h: "Measure", w: 58}, {h: "Internal test, n = 631", w: 60}, {h: "External, Messidor-2, n = 1,744", w: CW - 118}], [
      ["Sensitivity", {text: "0.986  [0.964 to 0.995]", style: "bold", color: TONE.routine}, {text: "0.973 on sight-threatening eyes", style: "bold", color: TONE.routine}],
      ["Specificity", {text: "0.873  [0.834 to 0.904]", style: "bold", color: TONE.routine}, {text: "0.922  [0.906 to 0.936]", style: "bold", color: TONE.routine}],
      ["AUC", "0.988  [0.979 to 0.994]", "0.908"],
      ["Negative predictive value", "0.987", "0.899"],
      ["Positive predictive value", "0.864", "0.764"],
      ["Within one grade", "96.2%", "91.4%"]], {zebra: true, size: 8.8});
    B.bars([["Sensitivity, internal", 0.986, "0.986", TONE.routine], ["Specificity, internal", 0.873, "0.873", TONE.routine],
      ["Specificity, Messidor-2", 0.922, "0.922", TONE.routine]], {min: 0.5, target: 0.85, targetLabel: "specificity target 0.85"});
    W.rich("On the internal test the grading also agrees closely with the reference: quadratic weighted kappa **0.897** [0.874 to 0.917], and the calibration error is **0.028**. Intervals are Wilson score for proportions and DeLong for the AUC. The referral threshold (0.1999) was fixed on the validation split before either test set was scored.", {size: 8.6});

    // 2 sight-threatening disease
    W.section(sec++, "Sight-threatening disease", "Severe NPDR and proliferative DR are the grades that cost sight, and the ones a screening programme exists to catch.", 70);
    B.kpis([["Internal test", "1.000", "105 of 105 referred, 95% CI 0.965 to 1.000", true],
      ["Messidor-2, zero-shot", "0.973", "107 of 110 referred, 95% CI 0.923 to 0.991", true],
      ["Queued for review", "24.3%", "of the Messidor-2 population, so the specialist is not flooded"]]);
    B.sub("Messidor-2, zero-shot: what happens to each grade");
    B.bars([["Grade 0, no apparent DR", 0.939, "93.9% kept out of the queue", GRADE_RGB[0]],
      ["Grade 1, mild NPDR", 0.859, "85.9% kept out of the queue", GRADE_RGB[1]],
      ["Grade 3, severe NPDR", 0.987, "98.7% referred (74 of 75)", GRADE_RGB[3]],
      ["Grade 4, proliferative DR", 0.943, "94.3% referred (33 of 35)", GRADE_RGB[4]]], {labelW: 48, textW: 44});
    W.para("Healthy and mild eyes stay out of the specialist's queue; severe and proliferative disease reaches it, on hardware and graders the model had never seen.", {size: 8.8});

    // 3 integrated pipeline
    W.section(sec++, "The integrated pipeline beats every single technique", "The problem statement asks for this to be shown. Every arm was trained and scored under the same conditions, each with its own temperature and threshold fitted on validation data.", 55);
    W.table([{h: "Design", w: 62}, {h: "AUC", w: 24, align: "right"}, {h: "Sens.", w: 20, align: "right"}, {h: "Spec.", w: 20, align: "right"}, {h: "QWK", w: 20, align: "right"}, {h: "Targets", w: CW - 146}], [
      [{text: "Image CNN, ordinal (deployed)", style: "bold"}, {text: "0.9883", style: "bold"}, "0.986", "0.873", "0.894", {text: "both met", color: TONE.routine, style: "bold"}],
      ["Image + clinical fusion", "0.9850", "0.989", "0.882", "0.882", {text: "both met", color: TONE.routine}],
      ["Clinical features only", "0.9272", "0.880", "0.813", "0.690", "baseline"],
      ["ICDR rules only", "0.9115", "1.000", "0.075", "0.000", "baseline"]], {zebra: true, size: 8.8});
    B.bars([["Image CNN (deployed)", 0.9883, "0.988", C.accent], ["Image + clinical fusion", 0.9850, "0.985", C.faint],
      ["Clinical features only", 0.9272, "0.927", C.faint], ["ICDR rules only", 0.9115, "0.912", C.faint]], {boldFirst: true, min: 0.85});
    W.rich("Paired DeLong tests put the deployed design ahead of the clinical-feature arm (**p = 1.7 x 10^-11**) and the rule-based arm (**p = 1.5 x 10^-12**), and ahead of fusion (**p = 0.034**). The clinical branch still earns its place: it writes the lesion counts, the quadrant map and the ICDR 4-2-1 rule trail that ship with every referral.", {size: 8.8});

    // 4 lesions and landmarks
    W.section(sec++, "Lesions and landmarks", "The evidence behind each grade: lesions segmented by an attention U-Net at 1024 px, and the optic disc and fovea that set the clinical frame.", 60);
    W.table([{h: "Lesion (IDRiD)", w: 50}, {h: "Dice at 1024 px", w: 32, align: "right"}, {h: "Dice at 512 px", w: 30, align: "right"}, {h: "Evidence for", w: CW - 112}], [
      ["Microaneurysm", {text: "0.481", style: "bold"}, "0.000", "mild NPDR"],
      ["Haemorrhage", {text: "0.539", style: "bold"}, "0.248", "the 4-2-1 quadrant rule"],
      ["Hard exudate", {text: "0.572", style: "bold"}, "0.272", "macular oedema, distance to the fovea"],
      ["Cotton-wool spot", {text: "0.628", style: "bold"}, "0.187", "moderate NPDR"]], {zebra: true, size: 8.8});
    W.rich("Resolution is decisive: a microaneurysm is only a few pixels wide, and at 512 px it disappears. Adding DDR's lesion annotations lifted mean Dice on held-out DDR images from **0.372 to 0.511**. A class no corpus annotates (neovascularisation) is reported as **not assessed**, never as a false zero.", {size: 8.8});
    B.kpis([["Optic disc within 1 DD", "98.1%", "median error 0.098 disc diameters", true], ["Fovea within 1 DD", "93.2%", "median error 0.178 disc diameters", true],
      ["Landmark time", "~120 ms", "closed form, no training data, CPU"]]);
    W.para("Measured on 103 held-out IDRiD photographs with hand-marked disc and fovea centres (scripts/eval_landmarks.py).", {size: 8.2, color: C.muted});

    // 5 calibration and explanation
    W.section(sec++, "Calibrated confidence and measured explanations", "When the system says 0.9 it means 0.9, so a review queue sorted by confidence puts the referral that matters first. And the attention maps are tested, not just drawn.", 50);
    B.kpis([["Calibration error", "0.016", "isotonic on P(referable), out-of-fold", true], ["Temperature alone", "0.025", "ECE on validation, T = 2.708"],
      ["Brier score", "0.061", "down from 0.070 before calibration"],
      ["Explanation faithfulness", "+0.127", "insertion 0.974 minus deletion 0.848", true], ["Attention sparsity", "0.772", "Gini, over 40 referable images"],
      ["Train / test overlap", "0", "subject-grouped splits, checked in code", true]]);

    // 6 cut-points
    W.section(sec++, "Every cut-point in the decision is fitted", "Each threshold the decision uses is selected on the validation split only, never on a test set, so the test figures above stay honest.", 50);
    W.table([{h: "Decision", w: 44}, {h: "Value", w: 58}, {h: "Fitted on", w: CW - 102}], [
      ["Referral", "0.1999 on P(grade >= 2)", "validation, sensitivity-first"],
      ["Urgency", "0.1184 on P(grade >= 3)", "validation, sensitivity-first"],
      ["Grade boundaries", "0.30, 0.39, 0.39, 0.36", "validation, macro-recall"],
      ["Lesion detection", "per-class F1-optimal", "held-out IDRiD"],
      ["Calibration", "T = 2.708, then isotonic", "validation, out-of-fold"]], {zebra: true, size: 8.8});
    W.rich("Fitting the urgency tier on P(grade >= 3) rather than on the printed grade sends **30 more sight-threatening eyes to the expedited queue** on Messidor-2 (86 of 110, up from 56), with the screening decision unchanged to four decimals.", {size: 8.8});

    // 7 the pooled bundle
    W.section(sec++, "The second bundle: all four corpora pooled", "The MATLAB edition also ships a grader trained on APTOS-2019, IDRiD, DDR and Messidor-2 together (bundle pooled). Scored on its own held-out test split of 1,852 images:", 60);
    B.kpis([["Sensitivity", "0.915", "832 of 909, 95% CI 0.895 to 0.932", true], ["Specificity", "0.887", "836 of 943, 95% CI 0.865 to 0.905", true],
      ["AUC", "0.964", "referable DR"], ["Sight-threatening", "99.7%", "298 of 299 referred", true],
      ["Quadratic weighted kappa", "0.869", "within one grade 94.8%"], ["Calibration error", "0.028", "ECE on the test split"]]);
    W.table([{h: "Corpus", w: 60}, {h: "Sensitivity", w: 40, align: "right"}, {h: "Specificity", w: 40, align: "right"}, {h: "", w: CW - 140}], [
      ["APTOS-2019", "0.980", "0.884", ""], ["DDR", "0.890", "0.883", ""], ["IDRiD", "0.917", "0.852", ""], ["Messidor-2", "0.888", "0.906", ""]], {zebra: true, size: 8.8});
    W.para("Every corpus clears the 85% specificity floor. Pass 'pooled' to any MATLAB launcher to use this bundle; its fitted referral threshold is 0.3207.", {size: 8.6});

    // 8 MATLAB edition
    W.section(sec++, "The MATLAB edition reproduces the reference exactly", "The deployed system is MATLAB. Its two networks are rebuilt from the exported weights, and every stage is checked against the Python reference the models were trained in.", 50);
    B.kpis([["Decisions identical", "144 / 144", "72 photographs x 2 bundles", true], ["P(referable) difference", "1e-9", "median, prepool bundle"],
      ["MATLAB tests", "54 / 54", "image primitives to the HTTP API", true],
      ["With Deep Learning Toolbox", "2.6 s", "per photograph, 24-thread CPU"], ["Plain MATLAB", "8 s", "no toolbox beyond Image Processing"],
      ["Standalone apps", "2", "DRScreenWeb.exe, DRScreenConsole.exe"]]);

    // 9 programme
    W.section(sec++, "The programme around it", "A district screening service is a queueing network, and reviewer time is the scarce resource. The SimPy programme model runs a year of a district: 100,000 patients, 12 primary health centres and one ophthalmologist, as the problem statement provides.", 60);
    W.table([{h: "Scenario", w: 52}, {h: "Screened a year", w: 30, align: "right"}, {h: "Reviewer load", w: 28, align: "right"}, {h: "Routine SLA", w: 26, align: "right"}, {h: "p90 turnaround", w: CW - 136, align: "right"}], [
      ["Manual review, no AI", "99,261", {text: "331%", color: TONE.urgent, style: "bold"}, "12.2%", "65.0 days, rising"],
      [{text: "AI-assisted triage", style: "bold"}, "99,691", {text: "22.0%", color: TONE.routine, style: "bold"}, "100%", "0.01 days"],
      ["AI at the edge, low bandwidth", "99,491", "22.0%", "100%", "0.02 days"],
      ["AI at twice the demand", "199,945", "44.3%", "100%", "0.02 days"]], {zebra: true, size: 8.6});
    B.bars([["Manual review, no AI", 3.31, "331%", TONE.urgent], ["AI-assisted triage", 0.22, "22.0%", TONE.routine], ["AI at twice the demand", 0.443, "44.3%", TONE.routine]],
      {labelW: 48, max: 3.5, target: 1, targetLabel: "100%: the ophthalmologist's 5 reading hours a day"});
    W.rich("Without AI the review queue is unstable: reading every image by hand needs 3.3 times the reading time the district has. With AI triage the same ophthalmologist needs **22%** of it. Searching **1,024 programme designs**, the optimiser chose AI-assisted review and on-device inference itself; the cheapest feasible plan is 8 PHCs, one ophthalmologist and edge inference over 3G at **Rs 53.9 lakh a year, about Rs 54 per patient screened**.", {size: 8.8});
    B.sub("The Simulink district model");
    W.rich(`simulink/district_model.slx models one health centre's session in SimEvents: capture, the AI quality check, restoration and grading on the edge device, and the ophthalmologist's read at the district hub. With the variables as set (a patient every 3 minutes, a 6-hour session, 5 technicians, cameras and edge devices, one ophthalmologist), MATLAB's 100 replications average **${DISTRICT.arrived} patients, ${DISTRICT.cleared} cleared on the spot and ${DISTRICT.reviewed} read by the doctor**, with technicians ${DISTRICT.techBusy}% busy and the ophthalmologist ${DISTRICT.doctorBusy}%. The dossier's simulator is a block-for-block port that agrees with MATLAB on ${DISTRICT.agree} checked figures.`, {size: 8.8});

    // 10 conclusions
    W.section(sec++, "Conclusions", null, 60);
    [["Meets the problem statement's targets.", "Sensitivity 0.986 and specificity 0.873 for referable DR on held-out patients, with every sight-threatening eye referred."],
     ["Generalises.", "97.3% of blinding disease referred on a different country, different cameras and a different grading panel, zero-shot."],
     ["Integrated and explainable.", "Quality gate, landmarks, lesion segmentation, ordinal grading and calibrated confidence, with lesion counts per quadrant, the ICDR rule trail and a measured attention map behind every referral."],
     ["Practical where it is needed.", "MATLAB on an ordinary CPU, offline, in about 2.6 seconds, with standalone apps for clinics that have no MATLAB licence."],
     ["Sized for the programme.", "AI triage cuts the one district ophthalmologist's reading load from 331% to 22% of their time, at about Rs 54 per patient screened."]]
      .forEach(([h, b]) => W.rich(`**${h}** ${b}`, {mark: TONE.routine, size: 9.2}));
    B.sub("Next steps");
    ["Calibrate the referral threshold per site from a few hundred locally graded images; the console's review log already collects them.",
     "A prospective pilot at primary health centres, with the ophthalmologist's confirmations as the running audit.",
     "Pixel-level neovascularisation annotations, so proliferative lesions can be outlined directly.",
     "EyePACS, to add severe and proliferative training images."].forEach(s => W.rich(s, {mark: C.accent, size: 9}));
    W.y += 2;
    W.para("SixEyes is decision support: every referable and every sight-threatening finding is reviewed by a qualified ophthalmologist before clinical action. The complete record of every measurement, including the ablations and the engineering history, is RESULTS.md in the repository.", {size: 8.2, color: C.muted});

    B.decorate("Results and conclusions", "SixEyes results");
    return {doc, name: `SixEyes-results-and-conclusions_${t.day}.pdf`};
  }

  // ── the build notes ─────────────────────────────────────────────────────
  async function buildNotes() {
    const K = kit(), JsPDF = await K.load(), {C, TONE} = K, {ML, CW} = K.geometry;
    const doc = new JsPDF({unit: "mm", format: "a4", compress: true});
    const B = blocks(doc, K), W = B.W, t = stamp();
    doc.setProperties({title: "SixEyes - Build notes", subject: "How the SixEyes screening system is built and run",
      author: "Team SixEyes", creator: "SixEyes project dossier", keywords: "diabetic retinopathy, MATLAB, Simulink, SimEvents, screening"});
    const frames = await panelFrames(K, VSET + "reports/grade3_case2_panel.png");
    const photo = await K.loadImage(VSET + "images/grade3_case2.jpg");

    B.title({title: "Build notes", sub: "How it is built, how to run it, and where things are.",
      meta: [["Problem statement", "SIH-26038, MedTech"], ["Team", "SixEyes"], ["Runtime", "MATLAB R2021a or newer"], ["Generated", t.when]]});
    B.band({tone: C.accent, label: "What it is", title: "One photograph in, an audited referral out",
      sub: "A MATLAB screening system for the fundus camera at a rural health centre: offline, on an ordinary CPU, in about 2.6 seconds per photograph.",
      box: [["2.6 s", "per photograph, CPU"], ["144/144", "decisions match Python"]]});
    B.kpis([
      ["Runtime", "MATLAB", "R2021a or newer; tested on R2026a"],
      ["Required toolbox", "Image Processing", "nothing else is required"],
      ["Optional", "Deep Learning", "about 3x faster convolutions"],
      ["Networks", "2", "EfficientNet-B0 CORN grader, attention U-Net"],
      ["Tests", "54 + 78", "MATLAB parity tests + Python regression tests", true],
      ["Real photographs", "72", "12 verification + 60 showcase, all five grades"]]);

    let sec = 1;
    // 1 what it does
    W.section(sec++, "What it does", "From one fundus photograph, the system:");
    [["Checks the image", "with nine interpretable quality criteria, and tells the technician how to recapture a bad one."],
     ["Finds the optic disc, the fovea and four lesion types", "(microaneurysms, haemorrhages, hard and soft exudates)."],
     ["Grades severity", "on the ICDR scale with an ordinal network and calibrated confidence."],
     ["Explains every referral", "with lesion counts per quadrant, the ICDR rule trail and a Grad-CAM++ heatmap."],
     ["Routes the patient", "through a safety-ordered decision: urgent, soon, human review, or a 12-month recall."],
     ["Sizes the district programme", "around it, with a Simulink / SimEvents model of the screening service."]]
      .forEach(([h, b], i) => W.rich(`**${i + 1}.  ${h}** ${b}`, {size: 9.2}));

    // 2 the pipeline
    W.section(sec++, "The pipeline", "Nine stages, in the order the code runs them. Each records its own time, and the whole result is JSON, so any decision can be reconstructed from its record.", 70);
    const steps = [["01", "Geometry"], ["02", "Landmarks"], ["03", "Quality gate"], ["04", "Enhancement"], ["05", "Segmentation"],
      ["06", "Clinical features"], ["07", "Rules + grader"], ["08", "Decision"], ["09", "Explanation"], ["", "Report"]];
    (function flow() {
      const per = 5, gap = 5, bw = (CW - (per - 1) * gap) / per, bh = 13, rowGap = 9;
      W.ensure(2 * bh + rowGap + 12);
      const y0 = W.y + 1;
      steps.forEach(([n, name], i) => {
        const r = Math.floor(i / per), c = r === 0 ? i % per : per - 1 - (i % per);
        const x = ML + c * (bw + gap), y = y0 + r * (bh + rowGap), last = i === steps.length - 1;
        doc.setFillColor(...(last ? C.accent : C.soft)); doc.setDrawColor(...(last ? C.accent : C.rule)); doc.setLineWidth(0.3);
        doc.roundedRect(x, y, bw, bh, 1.2, 1.2, "FD");
        W.font("bold", 6.6, last ? C.white : C.accent); doc.text(n || "OUT", x + 3, y + 4.8, {charSpace: 0.2});
        W.font("bold", 8.4, last ? C.white : C.ink); doc.text(name, x + 3, y + 10);
        if (i === 2) { W.font("bold", 6, TONE.recapture); doc.text("fail: recapture advice", x + bw / 2, y - 1.4, {align: "center"}); }
        doc.setDrawColor(...C.muted); doc.setLineWidth(0.35);
        if (!last) {
          doc.setFillColor(...C.muted);
          if ((i + 1) % per === 0) { const ax = x + bw / 2; doc.line(ax, y + bh, ax, y + bh + rowGap - 0.8); doc.triangle(ax - 1, y + bh + rowGap - 1.8, ax + 1, y + bh + rowGap - 1.8, ax, y + bh + rowGap - 0.4, "F"); }
          else if (r === 0) { const ax = x + bw; doc.line(ax + 0.3, y + bh / 2, ax + gap - 0.9, y + bh / 2); doc.setFillColor(...C.muted); doc.triangle(ax + gap - 1.8, y + bh / 2 - 1, ax + gap - 1.8, y + bh / 2 + 1, ax + gap - 0.4, y + bh / 2, "F"); }
          else { const ax = x; doc.line(ax - 0.3, y + bh / 2, ax - gap + 0.9, y + bh / 2); doc.setFillColor(...C.muted); doc.triangle(ax - gap + 1.8, y + bh / 2 - 1, ax - gap + 1.8, y + bh / 2 + 1, ax - gap + 0.4, y + bh / 2, "F"); }
        }
      });
      W.y = y0 + 2 * bh + rowGap + 5;
    })();
    W.table([{h: "#", w: 10}, {h: "Stage", w: 36}, {h: "What it does", w: 82}, {h: "MATLAB", w: CW - 128}], [
      ["1", "Geometry", "Finds the circular field of view, crops tight, pads square and resizes; aspect ratio is kept because the disc diameter is the clinical unit.", "+preprocess/standardize.m"],
      ["2", "Landmarks", "Locates the optic disc and fovea in closed form: the clinical coordinate frame. It runs before the gate because the macula check needs the fovea.", "+preprocess/locateLandmarks.m"],
      ["3", "Quality gate", "Nine criteria: focus, illumination, contrast, field of view, macula in view, artefacts, under- and over-exposure, noise. An uncorrectable failure stops the run with recapture advice.", "+preprocess/assessQuality.m"],
      ["4", "Enhancement", "Only the corrections the gate asked for (grey-world balance, illumination levelling, CLAHE, denoising), then the gate checks again.", "+preprocess/adaptiveEnhance.m"],
      ["5", "Segmentation", "Attention U-Net at 1024 px: vessels and four lesion types. Neovascularisation is reported as not assessed.", "+models/Segmenter.m"],
      ["6", "Clinical features", "Lesion counts per quadrant, distances to the fovea in disc diameters, venous beading: the quantities the ICDR criteria are written in.", "+features/extract.m"],
      ["7", "Rules and grader", "The ICDR 4-2-1 rules give a rule-based grade and its trail; the EfficientNet-B0 CORN grader gives calibrated grade probabilities.", "+features/ruleGrade.m, +models/Grader.m"],
      ["8", "Decision", "The safety-ordered ladder in section 4.", "Pipeline.m"],
      ["9", "Explanation", "Grad-CAM++ over the referable log-odds, and the evidence written in words.", "Grader.gradCamPP, +report/"]], {size: 8, zebra: true});

    // what a screen produces
    if (photo || frames.lesions) {
      W.ensure(62); B.sub("What a screen produces: grade3_case2, a held-out APTOS-2019 photograph");
      const imgs = [[photo ? (function () { try { const cv = document.createElement("canvas"); cv.width = cv.height = 700; const g = cv.getContext("2d"); g.fillStyle = "#000"; g.fillRect(0, 0, 700, 700);
        const k = Math.min(700 / photo.naturalWidth, 700 / photo.naturalHeight), w = photo.naturalWidth * k, h = photo.naturalHeight * k; g.drawImage(photo, (700 - w) / 2, (700 - h) / 2, w, h); return cv.toDataURL("image/jpeg", 0.88); } catch (e) { return null; } })() : null, "Photograph"],
        [frames.enhanced, "Enhanced"], [frames.lesions, "Lesions and landmarks"], [frames.attention, "Grad-CAM++ attention"]];
      const gap = 4, s = (CW - 3 * gap) / 4;
      imgs.forEach(([src, cap], i) => {
        const x = ML + i * (s + gap);
        if (src) doc.addImage(src, "JPEG", x, W.y, s, s, undefined, "FAST"); else { doc.setFillColor(...C.soft); doc.rect(x, W.y, s, s, "F"); }
        W.font("normal", 7.4, C.muted); doc.text(cap, x, W.y + s + 4);
      });
      W.y += s + 8;
    }

    // 3 design decisions
    W.section(sec++, "Design decisions", "The choices that do most of the work, and why each differs from the standard recipe (docs/DESIGN.md has the reasoning in full).", 50);
    [["An ordinal head, not a softmax.", "CORN models P(grade > k) directly, so the referral probability is monotone by construction and the referral and urgency thresholds read off one network."],
     ["Lesions at full resolution.", "Microaneurysms are 5 to 10 pixels wide. At 512 px the U-Net's microaneurysm Dice is 0.000; at 1024 px it is 0.481."],
     ["An interpretable quality gate.", "A network's logit cannot tell a technician that the macula is out of frame; nine physics-based criteria can, in milliseconds, before any heavy model runs."],
     ["Analytic landmarks.", "Severe NPDR is defined per quadrant and macular oedema by distance from the fovea, so both need a coordinate frame. The closed-form detector needs no training data and returns a confidence."],
     ["Calibration is a requirement.", "The review queue is sorted by confidence, so the probability has to mean what it says: temperature scaling, then isotonic regression on P(referable)."],
     ["A safety-ordered decision.", "Macular-oedema and sight-threatening checks run before the referral cut-point, and uncertain cases go to a person, never to an automatic report."],
     ["Held out in code, not by convention.", "The data registry raises an error if Messidor-2 reaches the training pool, and splits are grouped by subject so fellow eyes never straddle a boundary."]]
      .forEach(([h, b]) => W.rich(`**${h}** ${b}`, {mark: C.accent, size: 9}));

    // 4 models, bundles and the decision
    W.section(sec++, "Models, bundles and the decision", "Two trained networks and one closed-form detector. Two grader bundles ship with the MATLAB edition; every launcher reads the default from one line, C.DEFAULT_BUNDLE in +drscreen/constants.m.", 60);
    W.table([{h: "Component", w: 40}, {h: "Model", w: 62}, {h: "Detail", w: CW - 102}], [
      ["Grader", "EfficientNet-B0 with a CORN ordinal head", "512 px; temperature and isotonic calibration; MC-dropout uncertainty; Grad-CAM++"],
      ["Lesion segmenter", "Attention U-Net", "1024 px; four lesion types plus vessels; per-class thresholds fitted on held-out IDRiD"],
      ["Landmarks", "Closed form", "optic disc and fovea, about 120 ms on a CPU"]], {size: 8.6, zebra: true});
    W.table([{h: "Bundle", w: 30}, {h: "Grader trained on", w: 64}, {h: "Referral", w: 24, align: "right"}, {h: "Urgency", w: 24, align: "right"}, {h: "", w: CW - 142}], [
      [{text: "prepool", style: "bold"}, "APTOS-2019 + IDRiD; Messidor-2 held out", "0.1999", "0.1184", {text: "default", color: TONE.routine, style: "bold"}],
      ["pooled", "APTOS-2019, IDRiD, DDR and Messidor-2", "0.3207", "0.0722", ""]], {size: 8.6});
    B.sub("The decision ladder (prepool bundle)");
    W.table([{h: "Outcome", w: 40}, {h: "When", w: CW - 40}], [
      [{text: "Recapture", color: TONE.recapture, style: "bold"}, "The quality gate fails on a defect enhancement cannot fix. The technician gets specific advice while the patient is still there."],
      [{text: "Refer, urgent", color: TONE.urgent, style: "bold"}, "P(grade >= 3) reaches 0.1184, or hard exudates lie within one disc diameter of the fovea (macular oedema risk 2)."],
      [{text: "Human review", color: TONE.human, style: "bold"}, "P(referable) falls in the uncertainty band 0.05 to 0.35, or the model's epistemic variance is high."],
      [{text: "Refer, soon", color: TONE.refer, style: "bold"}, "P(referable) reaches the referral threshold 0.1999. Programme target: seen within 14 days."],
      [{text: "Routine", color: TONE.routine, style: "bold"}, "Confidently not referable: the result is reported and the patient is recalled in 12 months."]], {size: 8.6});

    // 5 front ends
    W.section(sec++, "What you can open", "One MATLAB codebase serves every front end, and the website runs unchanged against the Python API too.", 60);
    W.table([{h: "Front end", w: 46}, {h: "What it is for", w: CW - 46}], [
      ["Project dossier", "Opens first at localhost:8000. The headline results, the nine stages, recorded runs to replay, the evidence, programme sizing, the Simulink district model running in the browser, and these PDFs."],
      ["Screening console", "Drop a photograph or pick one of the 12 held-out cases, watch each stage run, read the verdict, lesions, attention map and calibrated probability, record the ophthalmologist's grade, and download the PDF report."],
      ["Desktop console", "launch_console: the same pipeline as a MATLAB app, with batch screening of the 60-image showcase set."],
      ["Batch and validation", "run_demo writes reports for a folder; run_validation scores all 72 committed photographs against their reference grades."],
      ["Standalone apps", "build_standalone makes DRScreenWeb.exe and DRScreenConsole.exe (about 78 MB each), which run on the free MATLAB Runtime with no MATLAB licence."],
      ["REST API", "POST /screen, GET /cases, POST /review, GET /audit and more, with identical JSON from MATLAB and Python."]], {size: 8.6, zebra: true});

    // 6 running it
    W.section(sec++, "Running it", "MATLAB R2021a or newer with the Image Processing Toolbox. No Python, no GPU and no internet connection are needed.", 50);
    B.code(["cd matlab", "check_install          % which products are present", "launch_web             % the dossier opens at http://localhost:8000",
      "launch_console         % the desktop screening console", "run_tests('fast')      % image primitives and both networks, a few seconds",
      "run_tests              % everything, including all 72 photographs vs Python"]);
    W.para("In the dossier, the Screening console button opens the console. The Simulink district model needs Simulink and SimEvents:", {size: 8.8});
    B.code(["cd simulink", "open_system('district_model');  sim('district_model')", "R = run_district_model(p, 100);     % 100 independent replications"]);
    W.para("Training and validation run in the Python reference implementation (src/drscreen/), where the models were trained and exported to MATLAB:", {size: 8.8});
    B.code(["python -m venv .venv  &&  .venv\\Scripts\\activate", "pip install -r requirements.txt  &&  pip install -e .",
      "python scripts/run_demo.py --demo      # screens the 12 committed photographs", "python -m pytest -q                    # 78 regression tests on real images"]);

    // 7 district model
    W.section(sec++, "The Simulink district model", "simulink/district_model.slx: one health centre's screening session and the doctor review it feeds, built from SimEvents blocks. Every variable lives in district_model_params.m and can be edited there.", 60);
    W.table([{h: "Variable", w: 50}, {h: "Value", w: 30}, {h: "Meaning", w: CW - 80}], [
      ["mean_interarrival", "3 min", "a patient every 3 minutes on average"],
      ["sim_stop", "360 min", "a 6-hour session"],
      ["p_pass, p_repair, p_reject", "0.70, 0.15, 0.15", "good, borderline and bad images"],
      ["p_restore", "0.873", "borderline images the AI restoration makes gradeable"],
      ["t_capture, t_grade, t_review", "5, 0.25, 0.5 min", "photograph, AI grading, the doctor's read"],
      ["sensitivity, specificity", "0.986, 0.873", "the grader's operating point"],
      ["technicians, cameras, edge devices", "5 each", "at the health centre"],
      ["n_ophthalmologists", "1", "one ophthalmologist for the district, as the problem statement provides"]], {size: 8.4, zebra: true});
    W.rich("Patients arrive in groups at rate **lambda(t) = lambda_peak exp(-(t - t_peak)^2 / 2 sigma^2) (N - A(t)) / N**, each group bringing **G = 1 + Poisson(mu)** patients; lambda_peak is solved in closed form so a patient arrives every 3 minutes on average.", {size: 8.8});
    W.rich(`MATLAB's 100 seeded, independent replications average **${DISTRICT.arrived} patients, ${DISTRICT.cleared} cleared on the spot and ${DISTRICT.reviewed} read by the doctor**. The dossier's simulator is a block-for-block JavaScript port that agrees with MATLAB on ${DISTRICT.agree} figures, and adds a whole district over many days and a resource planner.`, {size: 8.8});

    // 8 verification
    W.section(sec++, "How it is verified", "The MATLAB suite runs five layers against fixtures recorded from the Python pipeline.", 60);
    W.table([{h: "Test", w: 38}, {h: "What it checks", w: CW - 38}], [
      ["TestCv", "Every image primitive against OpenCV's output; 8-bit outputs exact"],
      ["TestModels", "Both graders and the U-Net on fixed probe tensors, on both convolution paths"],
      ["TestPreprocess", "Each stage on real photographs, fed Python's output from the stage before"],
      ["TestPipelineParity", "The whole pipeline on all 72 photographs, both bundles"],
      ["TestServer", "Every endpoint in-process, and a real HTTP round trip with curl"]], {size: 8.6, zebra: true});
    B.kpis([["Grade, decision, urgency", "144 / 144", "identical to Python, 72 photographs x 2 bundles", true],
      ["P(referable)", "1e-9", "median difference from Python"], ["Python regression tests", "78", "clinical invariants, on real images"]]);

    // 9 repository
    W.section(sec++, "Where everything is", null, 60);
    W.table([{h: "Folder", w: 40}, {h: "Contents", w: CW - 40}], [
      ["matlab/", "The screening system in MATLAB: pipeline, web server, desktop app, exported weights, 54 parity tests, standalone build"],
      ["simulink/", "The SimEvents district model, its variables, replication runner and figures"],
      ["web/", "The project dossier (prototype.html) and the screening console (index.html)"],
      ["src/drscreen/", "The Python reference: training, validation, the SimPy programme model"],
      ["scripts/, tests/", "Python entry points and regression tests"],
      ["outputs/", "Evidence: deployable bundles, validation reports, run logs, the verification set"],
      ["docs/", "Design, simulation, datasets and reproduction notes"],
      ["RESULTS.md", "Every measured number, and how it was measured"]], {size: 8.6, zebra: true});

    // 10 data
    W.section(sec++, "Data and licences", "The corpora are not redistributed; each needs its own licence accepted. The committed photographs are held-out images included for demonstration only, under their source licences.", 50);
    W.table([{h: "Dataset", w: 38}, {h: "Images", w: 24, align: "right"}, {h: "Role", w: CW - 62}], [
      ["APTOS-2019", "3,662", "training, validation and the internal test"],
      ["IDRiD", "516 + 81", "grading; the pixel-level lesion masks"],
      ["Messidor-2", "1,748", "held-out external test for the prepool model; training data for pooled"],
      ["DDR", "13,673", "the pooled grader and lesion segmentation"],
      ["DRIVE", "40", "vessel masks"]], {size: 8.6, zebra: true});
    W.para("Cohort for the prepool model: 2,925 training, 622 validation and 631 test images, with zero measured subject overlap between the grading splits.", {size: 8.6});

    B.decorate("Build notes", "SixEyes build notes");
    return {doc, name: `SixEyes-build-notes_${t.day}.pdf`};
  }

  async function build(kind) { return kind === "results" ? results() : buildNotes(); }
  async function download(kind) { const {doc, name} = await build(kind); doc.save(name); return name; }

  // the dossier's buttons
  function wire() {
    document.querySelectorAll("button[data-doc]").forEach(b => b.addEventListener("click", async () => {
      const label = b.dataset.label || (b.dataset.label = b.textContent);
      b.disabled = true; b.textContent = "Preparing the PDF…";
      try { await download(b.dataset.doc); b.textContent = label; }
      catch (e) { console.error("PDF failed", e); b.textContent = "Could not build the PDF. Try again"; setTimeout(() => { b.textContent = label; }, 4000); }
      b.disabled = false;
    }));
  }
  if (typeof document !== "undefined") {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", wire); else wire();
  }
  root.DRDocs = {build, download};
})(typeof self !== "undefined" ? self : globalThis);
