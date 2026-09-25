/* The dossier's district simulator: controls, diagram, results, and the
   check against MATLAB. The model itself is district-sim.js (run in a Web
   Worker so a long run never blocks scrolling); the diagram and MATLAB's
   reference results come from district-model.js, generated from the .slx. */
(function () {
  "use strict";
  const M = window.DISTRICT_MODEL, Sim = window.DistrictSim;
  const form = document.getElementById("simForm");
  if (!form || !Sim || !M) return;
  const $ = id => document.getElementById(id);
  const box = $("sim"), statusEl = $("simStatus");
  const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ── formatting ───────────────────────────────────────────────────────────
  const nf = (x, d = 0) => Number.isFinite(x) ? x.toLocaleString("en-IN", {maximumFractionDigits: d, minimumFractionDigits: d}) : "—";
  const pct = (x, d = 0) => Number.isFinite(x) ? nf(100 * x, d) + "%" : "—";
  function dur(min) {
    if (!Number.isFinite(min)) return "—";
    if (min < 1) return nf(min * 60) + " s";
    if (min < 90) return nf(min, min < 10 ? 1 : 0) + " min";
    if (min < 48 * 60) return nf(min / 60, 1) + " h";
    return nf(min / 1440, 1) + " days";
  }
  const esc = t => String(t).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
  const hrs = x => Number.isInteger(+x) ? String(+x) : nf(x, 1);          // 6, 6.5
  const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;

  // ── the model's variables (district_model_params.m) ─────────────────────
  // The page starts from the struct MATLAB last replicated
  // (references.ref_model.p), so editing district_model_params.m and
  // re-running the check moves the page with it.
  const fromP = p => ({
    technicians: p.n_technicians, cameras: p.n_cameras, devices: p.n_edge_devices, ophthalmologists: p.n_ophthalmologists,
    meanGap: p.mean_interarrival, tPeakH: p.t_peak / 60, sigmaH: p.sigma / 60, N: p.N, mu: p.mu, sessionHours: p.sim_stop / 60,
    pGood: p.p_pass / (p.p_pass + p.p_repair + p.p_reject), pBorder: p.p_repair / (p.p_pass + p.p_repair + p.p_reject),
    pRestore: p.p_restore, includeGate: !!p.include_gate, maxAttempts: p.max_attempts,
    captureMin: p.t_capture, qualityMin: p.t_quality, restoreMin: p.t_restore, gradeMin: p.t_grade,
    uploadMin: p.t_upload, reviewMin: p.t_review, sensitivity: p.sensitivity, specificity: p.specificity,
    prevalence: p.prevalence_referable});
  const P0 = (M.references && M.references.ref_model && M.references.ref_model.p) || null;
  const MODEL = Object.assign({technicians: 5, cameras: 5, devices: 5, ophthalmologists: 5, meanGap: 3, tPeakH: 2.5, sigmaH: 1.5,
    N: 200, mu: 0.5, sessionHours: 6, pGood: 0.70, pBorder: 0.15, pRestore: 0.873, includeGate: false, maxAttempts: 3,
    captureMin: 5, qualityMin: 0.01, restoreMin: 0.02, gradeMin: 0.25, uploadMin: 0.67, reviewMin: 0.5,
    sensitivity: 0.986, specificity: 0.873, prevalence: 0.08}, P0 ? fromP(P0) : {}, {ai: true, priority: false, stochastic: false, reps: 20});
  // a district of 12 health centres sharing a review hub, the same blocks and variables
  const DISTRICT = Object.assign({}, MODEL, {phcs: 12, technicians: 1, cameras: 1, devices: 1, ophthalmologists: 2,
    meanGap: 12.6, tPeakH: 2.5, sigmaH: 1.5, N: 46, sessionHours: 7, reviewHours: 5, days: 20,
    ai: true, priority: true, stochastic: true, reps: 10});                 // ~33 patients per centre per 7-hour day
  const PRESETS = {
    district: DISTRICT,
    double: Object.assign({}, DISTRICT, {meanGap: 6.3, N: 92}),            // twice the catchment, twice as often
    manual: Object.assign({}, DISTRICT, {ai: false}),
    thin: Object.assign({}, DISTRICT, {ophthalmologists: 1, reviewHours: 2})
  };

  // ── the form ─────────────────────────────────────────────────────────────
  const F = form.elements;
  const asBuilt = () => F.model.value === "asbuilt";
  const ARRIVAL_KEYS = new Set(["meanGap", "tPeakH", "sigmaH", "N", "mu", "sessionHours"]);

  function setField(name, v) {
    const el = F[name]; if (!el) return;
    if (el.type === "checkbox") el.checked = !!v; else el.value = v;
    syncField(el, true);
  }
  function syncField(el, quiet) {
    // good + borderline cannot exceed every image: the other one gives way
    if (!quiet && (el.name === "pGood" || el.name === "pBorder")) {
      const other = F[el.name === "pGood" ? "pBorder" : "pGood"];
      if (+F.pGood.value + +F.pBorder.value > 1) { other.value = Math.max(0, 1 - +el.value).toFixed(2); syncField(other, true); }
    }
    const out = $("v_" + el.name);
    if (out) out.textContent = el.type === "range"
      ? (el.dataset.pct ? nf(100 * el.value, +el.dataset.pct) + "%" : el.value + (el.dataset.unit || "")) : el.value;
    if (el.name === "pGood" || el.name === "pBorder") {
      $("simBad").textContent = `Bad: ${nf(100 * Math.max(0, 1 - +F.pGood.value - +F.pBorder.value), 0)}% of images cannot be graded`;
    }
    if (ARRIVAL_KEYS.has(el.name)) arrivals();
    if (el.name === "includeGate") drawDiagram();
    const st = el.closest && el.closest(".st");
    if (st) {
      st.querySelector('[data-d="-1"]').disabled = +el.value <= +el.dataset.min;
      st.querySelector('[data-d="1"]').disabled = +el.value >= +el.dataset.max;
    }
  }
  function readForm() {
    const n = k => +F[k].value, ab = asBuilt();
    const c = {asBuilt: ab, reps: n("reps"), seed: n("seed"),
      technicians: n("technicians"), cameras: n("cameras"), devices: n("devices"), ophthalmologists: n("ophthalmologists"),
      pGood: n("pGood"), pBorder: n("pBorder"), pBad: Math.max(0, 1 - n("pGood") - n("pBorder")), pRestore: n("pRestore"),
      includeGate: F.includeGate.checked, maxAttempts: n("maxAttempts"),
      captureMin: n("captureMin"), qualityMin: n("qualityMin"), restoreMin: n("restoreMin"), gradeMin: n("gradeMin"),
      uploadMin: n("uploadMin"), reviewMin: n("reviewMin"), sensitivity: n("sensitivity"), specificity: n("specificity"),
      prevalence: n("prevalence"), sessionHours: n("sessionHours"),
      ai: ab ? true : F.ai.checked, priority: ab ? false : F.priority.checked, stochastic: ab ? false : F.stochastic.checked};
    Object.assign(c, arrivalCfg());
    if (!ab) Object.assign(c, {phcs: n("phcs"), reviewHours: n("reviewHours"), days: n("days")});
    // keep any one run to a couple of seconds: fewer replications for a very large district
    const load = ab ? Sim.expected(c) : c.phcs * Sim.expected(c) * c.days;
    const maxReps = Math.max(2, Math.floor(420000 / Math.max(1, load)));
    c.cappedFrom = c.reps > maxReps ? c.reps : 0;
    c.reps = Math.min(c.reps, maxReps);
    return c;
  }
  // arrival parameters, from the form's hours into the model's minutes
  function arrivalCfg() {
    const n = k => +F[k].value;
    return {meanGap: n("meanGap"), tPeak: n("tPeakH") * 60, sigma: n("sigmaH") * 60, N: n("N"), mu: n("mu"), sessionHours: n("sessionHours")};
  }
  function arrivals() {
    const svgA = $("simArrCurve"); if (!svgA) return;
    const cfg = Object.assign({asBuilt: asBuilt()}, arrivalCfg());
    const pts = Sim.arrivalCurve(cfg, 60), T = pts[pts.length - 1][0] || 1;
    const top = Math.max(1, ...pts.map(q => q[1]));
    const xy = pts.map(([t, v]) => `${(t / T * 300).toFixed(1)},${(66 - v / top * 60).toFixed(1)}`);
    svgA.innerHTML = `<polygon points="0,68 ${xy.join(" ")} 300,68" fill="rgba(62,214,206,.12)"/>` +
      `<polyline points="${xy.join(" ")}" fill="none" stroke="var(--disc)" stroke-width="1.5" vector-effect="non-scaling-stroke"/>` +
      `<text x="4" y="11" fill="var(--faint)" font-family="var(--mono)" font-size="9">${nf(top, 1)} /h</text>`;
    const peakAt = pts.reduce((a, b) => (b[1] > a[1] ? b : a))[0];
    $("simArrPeak").textContent = "busiest ≈ " + nf(peakAt / 60, 1) + " h in";
    $("simArrEnd").textContent = "closing, " + nf(T / 60, 1) + " h";
    $("simArrGap").textContent = hrs(cfg.meanGap);
    const want = T / cfg.meanGap, e = Sim.expected(cfg);
    $("simArrExp").textContent = (want >= 0.999 * cfg.N)
      ? `The catchment N = ${cfg.N} is smaller than a patient every ${hrs(cfg.meanGap)} min needs (${nf(want)}): raise N`
      : `λ peak ≈ ${nf(60 * Sim.peakRate(cfg), 1)} events/h · ≈ ${nf(e, e < 10 ? 1 : 0)} patients per centre per ${cfg.asBuilt ? "session" : "day"}`;
    $("simArrExp").classList.toggle("warn", want >= 0.999 * cfg.N);
  }
  function applyModel() {
    const ab = asBuilt();
    form.querySelectorAll('[data-scope="improved"]').forEach(g => { g.hidden = ab; });
    $("simPresets").hidden = ab;
    $("simSessLabel").textContent = ab ? "Session length" : "Clinic hours per day";
    const m = ab ? MODEL : DISTRICT;
    Object.keys(m).forEach(k => setField(k, m[k]));
    ["pGood", "pBorder"].forEach(k => syncField(F[k], true));
    arrivals();
    $("tgWait").value = ab ? 15 : 30;                       // a session's wait, a district's
    $("tgRead").value = ab ? 1 : 24;
    drawDiagram();
    $("simDiaNote").textContent = ab ? "Figures inside blocks: the model's readings for one session" : "Figures inside blocks: rows 1–2 per centre per day, row 3 for the whole district";
    if (ab) $("simPresets").querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", "false"));
  }
  function applyPreset(name) {
    const pr = PRESETS[name]; if (!pr) return;
    Object.keys(pr).forEach(k => setField(k, pr[k]));
    ["pGood", "pBorder"].forEach(k => syncField(F[k], true));
    arrivals(); drawDiagram();
    $("simPresets").querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", b.dataset.preset === name ? "true" : "false"));
  }

  form.querySelectorAll(".st").forEach(st => {
    const input = st.querySelector("input");
    st.addEventListener("click", ev => {
      const b = ev.target.closest(".st-b"); if (!b || b.disabled) return;
      input.value = Math.max(+input.dataset.min, Math.min(+input.dataset.max, +input.value + +b.dataset.d));
      syncField(input); changed();
    });
    syncField(input, true);
  });
  form.querySelectorAll('input[type="range"]').forEach(r => { syncField(r, true); r.addEventListener("input", () => { syncField(r); changed(); }); });
  form.querySelectorAll('input[type="checkbox"], input[type="number"]').forEach(x => x.addEventListener("change", () => { syncField(x); changed(); }));
  form.querySelectorAll('input[name="model"]').forEach(x => x.addEventListener("change", () => { applyModel(); if (!asBuilt()) applyPreset("district"); run(); }));
  $("simPresets").addEventListener("click", ev => { const b = ev.target.closest("button[data-preset]"); if (b) { applyPreset(b.dataset.preset); run(); } });
  form.addEventListener("submit", ev => { ev.preventDefault(); run(); });
  let debounce = 0;
  function changed() {
    $("simPresets").querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", "false"));
    clearTimeout(debounce); debounce = setTimeout(run, 380);
  }

  // ── running: in a worker when there is one ──────────────────────────────
  let worker = null, busy = false, seq = 0, pendingCfg = null, watchdog = 0;
  // on the page thread: when there is no worker, or it fails or goes quiet
  function local(cfg, id) {
    setTimeout(() => { if (id === seq) { try { done(Sim.run(cfg)); } catch (e) { done(null, e.message); } } }, 30);
  }
  function makeWorker() {
    if (worker === false) return null;
    try {
      worker = new Worker("district-sim.js?v=8");
      worker.onmessage = ev => { if (ev.data.id === seq) done(ev.data.ok ? ev.data.result : null, ev.data.error); };
      worker.onerror = ev => { if (ev && ev.preventDefault) ev.preventDefault(); worker = false; if (busy) local(pendingCfg, seq); };
    } catch (e) { worker = false; }
    return worker || null;
  }
  function run() {
    const cfg = readForm(), id = ++seq;
    if (busy && worker) { worker.terminate(); worker = null; }   // a newer question replaces the old one
    busy = true; box.classList.toggle("running", !REDUCED);
    statusEl.textContent = "Running…"; statusEl.className = "sim-status busy";
    pendingCfg = cfg;
    const w = worker || makeWorker();
    clearTimeout(watchdog);
    if (w) {
      w.postMessage({id, config: cfg});
      watchdog = setTimeout(() => { if (busy && id === seq) { if (worker) worker.terminate(); worker = false; local(cfg, id); } }, 4000);
    } else local(cfg, id);
  }
  function done(res, err) {
    clearTimeout(watchdog); busy = false; box.classList.remove("running");
    if (!res) { statusEl.textContent = "The run failed: " + (err || "unknown error"); statusEl.className = "sim-status"; return; }
    const c = res.config;
    statusEl.textContent = `${c.reps} × ${c.asBuilt ? "one session" : c.days + " days"} in ${nf(res.ms / 1000, 2)} s` +
      (c.cappedFrom ? ` (replications reduced from ${c.cappedFrom})` : "");
    statusEl.className = "sim-status";
    render(res);
    schedulePlan(c);
  }

  // ── the diagram, drawn from the .slx ─────────────────────────────────────
  // Every block, connection, stage panel and branch label is where
  // build_district_model.m put it in the model file. Scopes and displays are
  // not drawn: their readings appear inside the blocks they measure.
  const svg = $("simDiagram"), NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
  let statNode = {};
  const NAMES = {Arrivals: "Arrivals", MergeRetakes: "Merge recaptures", WaitingRoom: "Waiting hall",
    AcqTechnician: "Get technician", AcqCamera: "Get camera", CaptureStation: "Capture", RelTechnician: "Free technician",
    RelCamera: "Free camera", AcqDevice: "Get edge device", QualityCheck: "AI quality check", QualitySwitch: "AI sorts",
    Restoration: "AI restoration", RestoreSwitch: "Restored?", GradeMerge: "Merge", GradeOnDevice: "AI grading",
    DeviceMerge: "Merge", RelDevice: "Free device", RouteDecision: "Route", ClearedOnSpot: "Cleared on the spot",
    UploadBuffer: "Upload buffer", Upload: "Upload", ReviewQueue: "Review queue", AcqOphthalmologist: "Get doctor",
    SpecialistReview: "Doctor review", RelOphthalmologist: "Free doctor", Reviewed: "Reviewed"};
  const KIND = {EntityGenerator: "Entity Generator", EntityInputSwitch: "Entity Input Switch", Queue: "Entity Queue",
    EntityResourceAcquirer: "Resource Acquirer", EntityServer: "Entity Server", EntityResourceReleaser: "Resource Releaser",
    EntityOutputSwitch: "Entity Output Switch", EntityTerminator: "Entity Terminator", EntityResourcePool: "Resource Pool"};
  const TINT = ["rgba(110,150,255,.055)", "rgba(62,214,206,.06)", "rgba(242,166,90,.06)", "rgba(255,255,255,.028)"];
  const diagram = () => (F.includeGate.checked && M.diagramGate) ? M.diagramGate : M.diagram;
  function wrap(text, max) {
    const words = text.split(" "), out = [""];
    words.forEach(w => { const cur = out[out.length - 1]; if (cur && (cur + " " + w).length > max) out.push(w); else out[out.length - 1] = cur ? cur + " " + w : w; });
    return out.slice(0, 2);
  }
  function glyph(g, t, cx, cy) {
    const G = {class: "b-glyph"};
    if (t === "Queue") for (let i = 0; i < 4; i++) el("rect", Object.assign({x: cx - 15 + i * 8, y: cy - 10, width: 6, height: 20}, G), g);
    else if (t === "EntityServer") el("ellipse", Object.assign({cx, cy, rx: 15, ry: 11}, G), g);
    else if (t === "EntityResourceAcquirer") el("path", Object.assign({d: `M${cx - 12} ${cy - 2}v11h24v-11M${cx} ${cy - 13}v13m-5 -5 5 5 5-5`}, G), g);
    else if (t === "EntityResourceReleaser") el("path", Object.assign({d: `M${cx - 12} ${cy - 2}v11h24v-11M${cx} ${cy + 4}v-17m-5 5 5-5 5 5`}, G), g);
    else if (t === "EntityOutputSwitch") el("path", Object.assign({d: `M${cx - 14} ${cy}h8l14-9M${cx - 6} ${cy}l14 9`}, G), g);
    else if (t === "EntityInputSwitch") el("path", Object.assign({d: `M${cx - 14} ${cy - 9}l14 9h8M${cx - 14} ${cy + 9}l14-9`}, G), g);
    else if (t === "EntityGenerator") el("path", Object.assign({d: `M${cx - 10} ${cy - 10}q-6 10 0 20M${cx + 10} ${cy - 10}q6 10 0 20`}, G), g);
    else if (t === "EntityTerminator") el("path", Object.assign({d: `M${cx - 9} ${cy - 9}l18 18M${cx + 9} ${cy - 9}l-18 18`}, G), g);
  }
  const POOL_W = 210, POOL_H = 62;
  const nameLines = b => b.type === "EntityResourcePool" ? 0 : wrap(NAMES[b.name] || b.name, 15).length;
  // The .slx panels also hold its scopes, displays and formula, which the
  // page does not draw; here each panel is fitted to the blocks, lines and
  // labels it holds, with room for its title above them.
  function fitPanels(D) {
    const drawn = D.blocks.filter(b => KIND[b.type]);
    return D.areas.map(a => {
      const inside = (x, y) => x >= a.x && x <= a.x + a.w && y >= a.y && y <= a.y + a.h;
      const bs = drawn.filter(b => inside(b.x + b.w / 2, b.y + b.h / 2));
      if (!bs.length) return Object.assign({}, a);
      const box = b => b.type === "EntityResourcePool"
        ? [b.y + b.h / 2 - POOL_H / 2, b.y + b.h / 2 + POOL_H / 2]
        : [b.y, b.y + b.h + (nameLines(b) ? 8 + 16 * nameLines(b) : 0)];
      let top = Math.min(...bs.map(b => box(b)[0])), bot = Math.max(...bs.map(b => box(b)[1]));
      D.notes.forEach(n => { if (inside(n.x, n.y)) { top = Math.min(top, n.y); bot = Math.max(bot, n.y + n.h); } });
      D.lines.forEach(l => { if (l.kind === "entity") l.pts.forEach(([x, y]) => { if (inside(x, y)) { top = Math.min(top, y); bot = Math.max(bot, y); } }); });
      return Object.assign({}, a, {y: top - 44, h: bot - top + 44 + 12});
    });
  }
  function drawDiagram() {
    const D = diagram();
    svg.textContent = ""; statNode = {};
    $("simDiaSrc").textContent = D.source + (asBuilt() ? "" : " · one health centre shown");
    const panels = fitPanels(D);
    const xs = [], ys = [];
    panels.forEach(a => { xs.push(a.x, a.x + a.w); ys.push(a.y, a.y + a.h); });
    D.lines.forEach(l => { if (l.kind === "entity") l.pts.forEach(([x, y]) => { xs.push(x); ys.push(y); }); });
    const x0 = Math.min(...xs), y0 = Math.min(...ys), x1 = Math.max(...xs), y1 = Math.max(...ys), pad = 6;
    svg.setAttribute("viewBox", `${x0 - pad} ${y0 - pad} ${x1 - x0 + 2 * pad} ${y1 - y0 + 2 * pad}`);
    const defs = el("defs", {}, svg);
    const mk = el("marker", {id: "dm-ah", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 11, markerHeight: 11,
      markerUnits: "userSpaceOnUse", orient: "auto"}, defs);
    el("path", {d: "M0 0 10 5 0 10z", fill: "rgba(242,242,238,.6)"}, mk);

    // stage panels
    panels.forEach((a, i) => {
      el("rect", {x: a.x, y: a.y, width: a.w, height: a.h, rx: 6, fill: TINT[i] || TINT[3], stroke: "rgba(255,255,255,.06)"}, svg);
      el("text", {x: a.x + 14, y: a.y + 26, class: "b-lane"}, svg).textContent = a.text.replace(/\s*\(.*\)$/, "").replace(/^(\d)\s+/, "$1  ");
    });
    // connections, as Simulink routes them
    const gl = el("g", {}, svg);
    D.lines.filter(l => l.kind === "entity").forEach(l => {
      el("polyline", {points: l.pts.map(q => q.join(",")).join(" "), class: "l-e", "marker-end": "url(#dm-ah)"}, gl);
    });
    D.notes.forEach(n => { el("text", {x: n.x, y: n.y + n.h - 2, class: "b-loop"}, gl).textContent = n.text; });
    // blocks
    const gb = el("g", {}, svg);
    D.blocks.filter(b => KIND[b.type]).forEach(b => {
      const g = el("g", {transform: `translate(${b.x},${b.y})`, class: "b"}, gb);
      el("title", {}, g).textContent = `${b.name} — ${KIND[b.type]}`;
      if (b.type === "EntityResourcePool") {           // a card: the resource, how many, how busy
        const c = el("g", {transform: `translate(${b.w / 2 - POOL_W / 2},${b.h / 2 - POOL_H / 2})`}, g);
        el("rect", {width: POOL_W, height: POOL_H, rx: 6, class: "b-box b-pool"}, c);
        el("text", {x: 16, y: 26, class: "b-chip"}, c).textContent = (b.resource || b.name).replace("EdgeDevice", "Edge devices")
          .replace("Ophthalmologist", "Doctors").replace(/^(Technician|Camera)$/, "$1s");
        statNode[b.name + ":n"] = el("text", {x: POOL_W - 16, y: 26, class: "b-chip-v"}, c);
        el("rect", {x: 16, y: 39, width: POOL_W - 84, height: 8, rx: 2, class: "b-pbar"}, c);
        statNode[b.name + ":bar"] = el("rect", {x: 16, y: 39, width: 0, height: 8, rx: 2, class: "b-pfill"}, c);
        statNode[b.name] = el("text", {x: POOL_W - 16, y: 48, class: "b-pstat"}, c);
        return;
      }
      el("rect", {width: b.w, height: b.h, rx: 4, class: "b-box"}, g);
      if (b.type === "EntityServer") statNode[b.name + ":heat"] = el("rect", {x: 1, y: 1, width: b.w - 2, height: b.h - 2, rx: 3, class: "b-heat", fill: "rgba(62,214,206,0)"}, g);
      glyph(g, b.type, b.w / 2, b.h < 70 ? 22 : b.h / 2 - 8);
      statNode[b.name] = el("text", {x: b.w / 2, y: b.h - 9, class: "b-stat"}, g);
      wrap(NAMES[b.name] || b.name, 15).forEach((ln, k) => { el("text", {x: b.w / 2, y: b.h + 17 + k * 16, class: "b-name"}, g).textContent = ln; });
    });
  }
  function setStat(name, text, level) {
    const n = statNode[name]; if (!n) return;
    n.textContent = text; n.setAttribute("class", (/^Pool/.test(name) ? "b-pstat" : "b-stat") + (level ? " " + level : ""));
  }
  function setPool(name, units, u) {
    setStat(name, pct(u), level(u));
    if (statNode[name + ":n"]) statNode[name + ":n"].textContent = "× " + units;
    const bar = statNode[name + ":bar"];
    if (bar) { bar.setAttribute("width", ((POOL_W - 84) * Math.min(1, u || 0)).toFixed(1)); bar.setAttribute("class", "b-pfill " + level(u)); }
  }
  const level = u => u >= 0.95 ? "crit" : u >= 0.85 ? "hot" : "";
  const heat = (name, u) => {
    const r = statNode[name + ":heat"]; if (!r) return;
    const c = u >= 0.95 ? "229,88,76" : u >= 0.85 ? "242,166,90" : "62,214,206";
    r.setAttribute("fill", `rgba(${c},${Math.min(0.42, 0.06 + 0.36 * Math.min(u, 1)).toFixed(2)})`);
  };

  // ── results ──────────────────────────────────────────────────────────────
  function render(res) {
    const c = res.config, S = res.summary, m = k => S[k] ? S[k].mean : NaN;
    const days = c.asBuilt ? 1 : c.days;
    const screened = m("cleared") + m("referred");
    const perUnit = c.asBuilt ? `per ${hrs(c.sessionHours)}-hour session` : "per working day";
    const units = c.asBuilt ? 1 : c.phcs;

    // headline figures
    const urgentOk = m("urgentSla"), backlog = m("backlog");
    const K = [
      [nf(screened / days), `Patients screened ${perUnit}`, `${nf(screened)} in total`],
      [pct(m("cleared") / screened), "Cleared at the health centre", "no specialist needed"],
      [nf(m("reviewed")), "Reviewed by a specialist", `of ${nf(m("referred"))} referred`],
      [c.ai ? nf(m("restored")) : "—", "Borderline images restored", c.ai ? `of ${nf(m("restorations"))} borderline` : "no AI restoration"],
      [dur(m("toReviewMedian")), "Median wait for a specialist", "from referral to review"],
      [dur(m("toReviewP90")), "90% reviewed within", c.asBuilt ? "within the session" : "including nights and weekends off"],
      c.asBuilt || !c.priority
        ? (c.includeGate ? [nf(m("retakes")), "Recaptures", `bad images photographed again, up to ${c.maxAttempts} captures`]
                         : [nf(m("ungradable")), "Sent to a doctor ungraded", "bad images, no recapture"])
        : [pct(urgentOk), "Urgent cases seen within 2 days", "urgent-first queue", urgentOk >= 0.98 ? "ok" : urgentOk >= 0.9 ? "warn" : "bad"],
      [pct(m("referableCaught"), 1), "Referable patients sent on", "graded referable, or ungradable"]
    ];
    $("simKpis").innerHTML = K.map(([v, l, s, cls]) => `<div class="kpi"><b class="${cls || ""}">${v}</b><span>${l}</span><small>${esc(s)}</small></div>`).join("");

    // utilisation
    const where = n => c.asBuilt ? `${n} at the centre` : `${n} × ${c.phcs} centres`;
    const U = [["Technicians", where(c.technicians), m("util_technician")], ["Cameras", where(c.cameras), m("util_camera")],
      ["Edge devices", where(c.devices), m("util_device")],
      ["Ophthalmologists", c.asBuilt ? `${c.ophthalmologists} at the hub` : `${c.ophthalmologists} at the hub, ${c.reviewHours} h/day`, m("util_review")]];
    const top = U.reduce((a, b) => (b[2] > a[2] ? b : a));
    $("simUtil").innerHTML = U.map(([n, s, u]) => {
      const col = u >= 0.95 ? "var(--g4)" : u >= 0.85 ? "#F2A65A" : "var(--disc)";
      return `<div class="ub"><span>${n}<small>${esc(s)}</small></span><span class="tr"><i style="--c:${col}" data-w="${Math.min(100, 100 * u).toFixed(1)}"></i><u title="85%: queues start to grow quickly"></u></span><output>${pct(u)}</output></div>`;
    }).join("");
    setTimeout(() => $("simUtil").querySelectorAll("i[data-w]").forEach(i => { i.style.width = i.dataset.w + "%"; }), 30);
    $("simBottleneck").textContent = `busiest: ${top[0].toLowerCase()} ${pct(top[2])}`;

    // queues over time
    drawChart(res.series, c);

    // the diagram, annotated for one health centre and the hub
    const per = x => nf(x / units / days, 1) + (c.asBuilt ? "" : " /day");
    const hub = x => nf(x / days, 0) + (c.asBuilt ? "" : " /day");
    setStat("Arrivals", per(m("arrived")));
    setStat("WaitingRoom", dur(m("captureWaitMean")));
    setStat("CaptureStation", pct(m("util_technician")), level(m("util_technician"))); heat("CaptureStation", m("util_technician"));
    setStat("QualityCheck", per(m("captured")));
    setStat("Restoration", per(m("restorations")));
    setStat("RestoreSwitch", c.ai && m("restorations") ? pct(m("restored") / m("restorations")) : "");
    setStat("GradeOnDevice", per(m("good") + m("restored")));
    setStat("RouteDecision", c.includeGate ? "↺ " + per(m("retakes")) : "");
    setStat("ClearedOnSpot", per(m("cleared")));
    setStat("UploadBuffer", hub(m("referred")));
    setStat("ReviewQueue", dur(m("toReviewMedian")));
    setStat("SpecialistReview", pct(m("util_review")), level(m("util_review"))); heat("SpecialistReview", m("util_review"));
    setStat("Reviewed", hub(m("reviewed")));
    const amt = {PoolTechnician: c.technicians, PoolCamera: c.cameras, PoolEdgeDevice: c.devices, PoolOphthalmologist: c.ophthalmologists};
    const pu = {PoolTechnician: m("util_technician"), PoolCamera: m("util_camera"), PoolEdgeDevice: m("util_device"), PoolOphthalmologist: m("util_review")};
    Object.keys(amt).forEach(k => setPool(k, amt[k], pu[k]));

    // in words
    const over = U.filter(u => u[2] > 1), hot = U.filter(u => u[2] >= 0.85 && u[2] <= 1);
    let words = c.asBuilt
      ? `In one ${hrs(c.sessionHours)}-hour session the model screens <b>${nf(screened, 1)}</b> patients and clears <b>${pct(m("cleared") / screened)}</b> of them on the spot. `
      : `Over ${days} working days the district screens <b>${nf(screened / days)}</b> patients a day across ${c.phcs} health centres and clears <b>${pct(m("cleared") / screened)}</b> of them there. `;
    if (c.ai && m("restorations")) words += `The restoration saves <b>${nf(m("restored") / days, 1)}</b> of ${nf(m("restorations") / days, 1)} borderline images${c.asBuilt ? "" : " a day"} that would otherwise need a doctor or another photograph. `;
    words += c.ai ? "" : "With no AI every captured image goes to a specialist, which is what the hub has to absorb. ";
    if (over.length) words += `<b>${over.map(u => u[0].toLowerCase()).join(" and ")}</b> ${over.length > 1 ? "work" : "works"} past ${over.length > 1 ? "their" : "its"} scheduled hours (${over.map(u => pct(u[2])).join(", ")}) — add capacity there first. `;
    else if (hot.length) words += `<b>${hot[0][0]}</b> ${hot.length > 1 ? "and others are" : "is"} above 85%, where queues start to lengthen quickly. `;
    else words += `The busiest resource is ${top[0].toLowerCase()} at ${pct(top[2])}, so nothing is near its limit. `;
    if (!c.asBuilt && backlog > 0.05 * Math.max(1, m("reviewed"))) words += `The review backlog is still ${nf(backlog)} at the end — add reviewers or review hours.`;
    else if (!c.asBuilt) words += `Half of all referrals are read within ${dur(m("toReviewMedian"))}, and 90% within ${dur(m("toReviewP90"))}.`;
    $("simRead").innerHTML = words;
  }
  function drawChart(series, c) {
    const ch = $("simChart"); ch.innerHTML = "";
    if (!series || series.length < 2) return;
    const W = 600, H = 200, tMax = series[series.length - 1][0] || 1;
    const qMax = Math.max(1, ...series.map(s => Math.max(s[1], s[2])));
    for (let i = 1; i < 4; i++) el("line", {x1: 0, x2: W, y1: H * i / 4, y2: H * i / 4, stroke: "rgba(255,255,255,.06)"}, ch);
    const line = (k, col) => el("polyline", {fill: "none", stroke: col, "stroke-width": 1.6, "vector-effect": "non-scaling-stroke",
      points: series.map(s => `${(s[0] / tMax * W).toFixed(1)},${(H - 4 - s[k] / qMax * (H - 12)).toFixed(1)}`).join(" ")}, ch);
    line(2, "var(--g1)"); line(1, "var(--disc)");
    el("text", {x: 4, y: 12, fill: "var(--faint)", "font-family": "var(--mono)", "font-size": 11}, ch).textContent = nf(qMax, 1) + " waiting";
    const days = c.asBuilt ? 0 : c.days, T = c.sessionHours * 60;
    $("simChartX").innerHTML = c.asBuilt ? `<span>0 min</span><span>${nf(T / 2)}</span><span>${nf(T)} min</span>`
      : `<span>day 1</span><span>day ${Math.ceil(days / 2)}</span><span>day ${days}</span>`;
    $("simChartTitle").textContent = c.asBuilt ? "QUEUES THROUGH THE SESSION" : "QUEUES OVER THE WORKING DAYS";
  }

  // ── the resource planner ─────────────────────────────────────────────────
  // After each run, the engine sizes every resource in turn (district-sim.js,
  // plan): the smallest number that meets its target, the others held where
  // they are. That answers what a district administrator allocates: how few
  // of each a centre needs, and what could serve another centre.
  const planBox = $("simPlan");
  let planWorker = null, planBusy = false, planSeq = 0, lastCfg = null, planDog = 0;
  const targets = () => ({waitMin: Math.max(0.1, +$("tgWait").value || 15), deviceMin: 1, readMin: 60 * Math.max(0.05, +$("tgRead").value || 1)});
  function schedulePlan(cfg) {
    if (!cfg) return;
    lastCfg = cfg;
    const id = ++planSeq, tg = targets();
    planBox.classList.add("busy"); $("simPlanNote").textContent = "sizing each resource…";
    const finish = res => { if (id !== planSeq) return; planBusy = false; clearTimeout(planDog); planBox.classList.remove("busy"); renderPlan(res, cfg); };
    const onThread = () => setTimeout(() => { if (id === planSeq) finish(Sim.plan(cfg, tg)); }, 30);
    if (planBusy && planWorker) { planWorker.terminate(); planWorker = null; }
    if (planWorker === null) {
      try {
        planWorker = new Worker("district-sim.js?v=8");
        planWorker.onmessage = ev => { if (ev.data.ok) finish(ev.data.result); else { planWorker = false; onThread(); } };
        planWorker.onerror = ev => { if (ev && ev.preventDefault) ev.preventDefault(); planWorker = false; onThread(); };
      } catch (e) { planWorker = false; }
    }
    planBusy = true;
    if (planWorker) {
      planWorker.postMessage({id, kind: "plan", config: cfg, targets: tg});
      clearTimeout(planDog);
      planDog = setTimeout(() => { if (planBusy && id === planSeq) { if (planWorker) planWorker.terminate(); planWorker = false; onThread(); } }, 8000);
    } else onThread();
  }
  ["tgWait", "tgRead"].forEach(k => $(k).addEventListener("change", () => schedulePlan(lastCfg)));

  function miniChart(pts, target, have, need, fmtV) {
    // bars for each size; the dashed line is the target; the ring marks what you have
    const W = 300, H = 132, B = 20, n = pts.length, bw = (W - 8) / n;
    const vals = pts.map(q => q.v).filter(Number.isFinite);
    // the scale fits the bars and the target, but a runaway first bar is cut off (↑)
    const cap = Math.min(Math.max(...vals, target) * 1.18, Math.max(target * 4, ...vals.slice(1)) * 1.18);
    const y = v => H - B - Math.min(v, cap) / cap * (H - B - 14);
    let g = `<line x1="0" x2="${W}" y1="${y(target).toFixed(1)}" y2="${y(target).toFixed(1)}" stroke="rgba(242,166,90,.8)" stroke-dasharray="4 4" vector-effect="non-scaling-stroke"/>`;
    pts.forEach((q, i) => {
      const x = 4 + i * bw, yy = y(q.v), cls = q.n === need ? "var(--disc)" : q.ok ? "rgba(62,214,206,.35)" : "rgba(242,242,238,.18)";
      g += `<rect x="${(x + bw * 0.18).toFixed(1)}" y="${Math.min(yy, H - B - 2).toFixed(1)}" width="${(bw * 0.64).toFixed(1)}" height="${Math.max(2, H - B - yy).toFixed(1)}" fill="${cls}"><title>${q.n}: ${esc(fmtV(q.v))}${q.left > 0.05 ? ` · ${nf(q.left, 1)} still waiting at closing` : ""}</title></rect>`;
      if (q.v > cap) g += `<text x="${(x + bw / 2).toFixed(1)}" y="10" text-anchor="middle" fill="var(--faint)" font-family="var(--mono)" font-size="9">↑</text>`;
      if (q.n === need) g += `<text x="${(x + bw / 2).toFixed(1)}" y="${(Math.min(yy, H - B - 2) - 4).toFixed(1)}" text-anchor="middle" fill="var(--disc)" font-family="var(--mono)" font-size="11">${esc(fmtV(q.v))}</text>`;
      g += `<text x="${(x + bw / 2).toFixed(1)}" y="${H - 4}" text-anchor="middle" fill="${q.n === have ? "var(--bone)" : "var(--faint)"}" font-family="var(--mono)" font-size="11.5"${q.n === have ? ' font-weight="600"' : ""}>${q.n}</text>`;
      if (q.n === have) g += `<rect x="${(x + 1).toFixed(1)}" y="${H - B + 3}" width="${(bw - 2).toFixed(1)}" height="${B - 3}" rx="3" fill="none" stroke="var(--bone-dim)"/>`;
    });
    g += `<text x="${W - 2}" y="${(y(target) - 4).toFixed(1)}" text-anchor="end" fill="rgba(242,166,90,.9)" font-family="var(--mono)" font-size="11">target ${esc(fmtV(target))}</text>`;
    return `<svg viewBox="0 0 ${W} ${H}" aria-hidden="true">${g}</svg>`;
  }
  function renderPlan(P, c) {
    const one = c.asBuilt, where = one ? "this centre" : "each centre";
    const hub = one ? "at the hub" : "at the district hub";
    const teamsHave = Math.min(c.technicians, c.cameras);
    const cards = [
      ["CAPTURE TEAMS", `a technician and a camera · average wait before capture`, P.teams, P.targets.waitMin, teamsHave, P.need.teams, v => dur(v), one ? "per centre" : "per centre"],
      ["EDGE DEVICES", `wait for the AI after capture`, P.devices, P.targets.deviceMin, c.devices, P.need.devices, v => dur(v), "per centre"],
      ["DOCTORS", `time to read 90% of referrals`, P.doctors, P.targets.readMin, c.ophthalmologists, P.need.doctors, v => dur(v), hub]
    ];
    $("simPlanCards").innerHTML = cards.map(([t, m, pts, tgt, have, need, fmtV, unit]) => {
      const verdict = need == null
        ? `<b class="short">${pts.length}+</b>needed<small>even ${pts.length} miss the target of ${esc(fmtV(tgt))}</small>`
        : `<b>${need}</b>needed ${esc(unit)}<small>you have ${have}${have > need ? ` · ${have - need} spare` : have < need ? ` · short by ${need - have}` : " · just right"}</small>`;
      return `<div class="plan-card"><h4>${t}</h4><p class="pc-m">${esc(m)}</p>${miniChart(pts, tgt, have, need, fmtV)}<p class="pc-r">${verdict}</p></div>`;
    }).join("");

    // what to keep and what to send elsewhere
    const k = one ? 1 : c.phcs;
    const needT = P.need.teams, needD = P.need.devices, needR = P.need.doctors;
    const rows = [["Technicians", c.technicians, needT, k], ["Fundus cameras", c.cameras, needT, k],
      ["Edge devices", c.devices, needD, k], ["Doctors", c.ophthalmologists, needR, 1]];
    const cell = (have, need, mult) => {
      if (need == null) return `<td class="short">more</td>`;
      const d = (have - need) * mult;
      return `<td class="${d > 0 ? "spare" : d < 0 ? "short" : ""}">${d > 0 ? "+" + nf(d) + " spare" : d < 0 ? nf(-d) + " short" : "none"}</td>`;
    };
    $("simPlanTable").innerHTML = `<thead><tr><th scope="col"></th><th scope="col">${one ? "You have" : "Per centre"}</th><th scope="col">Needed</th><th scope="col">${one ? "Free for others" : "Across " + c.phcs + " centres"}</th></tr></thead><tbody>` +
      rows.map(([n, have, need, mult]) => `<tr><th scope="row">${n}${mult === 1 && !one ? "<small>at the hub</small>" : ""}</th><td>${have}</td><td>${need == null ? "—" : need}</td>${cell(have, need, mult)}</tr>`).join("") + "</tbody>";
    const span = m => m >= 60 ? `${hrs(Math.round(m / 6) / 10)} h` : dur(m);
    const wantW = span(P.targets.waitMin), wantR = span(P.targets.readMin);
    let sum;
    if (needT == null || needR == null) {
      sum = `No size in the range meets both targets — an average wait of ${wantW} and 90% of referrals read within ${wantR}. Spread the demand over more centres or sessions, or relax a target.`;
    } else {
      const spare = rows.map(([n, have, need, mult]) => [n, (have - need) * mult]).filter(r => r[1] > 0);
      const short = rows.map(([n, have, need, mult]) => [n, (need - have) * mult]).filter(r => r[1] > 0);
      sum = `To keep the average wait before capture under <b>${wantW}</b> and read 90% of referrals within <b>${wantR}</b>, ${where} needs <b>${plural(needT, "technician", "technicians")} with ${plural(needT, "camera", "cameras")}</b> and <b>${plural(needD, "edge device", "edge devices")}</b>, and the hub <b>${plural(needR, "doctor", "doctors")}</b>. `;
      if (spare.length) sum += `That leaves ${spare.map(([n, d]) => `${nf(d)} ${n.toLowerCase().replace(/s$/, d === 1 ? "" : "s")}`).join(", ")} free to serve ${one ? "other centres" : "other districts or new centres"}. `;
      if (short.length) sum += `It is short of ${short.map(([n, d]) => `${nf(d)} ${n.toLowerCase().replace(/s$/, d === 1 ? "" : "s")}`).join(", ")}. `;
    }
    $("simPlanSum").innerHTML = sum + `<span class="dim"> Every size was run ${P.reps} times${one ? "" : `, over ${P.days} days`}, with everything else as set.</span>`;
    $("simPlanNote").textContent = one ? "one session, this centre" : `per centre, ${c.phcs} centres`;
    drawRush(c, needT);
  }
  // arrivals through the session against what the capture teams can photograph
  function drawRush(c, needT) {
    const ch = $("simRush"); ch.innerHTML = "";
    const pts = Sim.arrivalCurve(c, 60), T = pts[pts.length - 1][0] || 1;
    const perTeam = 60 / c.captureMin, now = Math.min(c.technicians, c.cameras) * perTeam, need = (needT || 0) * perTeam;
    const top = Math.max(...pts.map(q => q[1]), now, need) * 1.12 || 1;
    const W = 600, H = 180, X = t => t / T * W, Y = v => H - 6 - v / top * (H - 22);
    for (let i = 1; i < 4; i++) el("line", {x1: 0, x2: W, y1: H * i / 4, y2: H * i / 4, stroke: "rgba(255,255,255,.05)"}, ch);
    // where the arrivals outrun the capacity that is needed: a queue builds
    let over = null, overs = [];
    pts.forEach(([t, v]) => { if (need && v > need) { if (!over) over = [t, t]; over[1] = t; } else if (over) { overs.push(over); over = null; } });
    if (over) overs.push(over);
    overs.forEach(([a, b]) => el("rect", {x: X(a), y: 0, width: Math.max(1, X(b) - X(a)), height: H, fill: "rgba(242,166,90,.08)"}, ch));
    const line = pts.map(([t, v]) => `${X(t).toFixed(1)},${Y(v).toFixed(1)}`).join(" ");
    el("polygon", {points: `0,${H} ${line} ${W},${H}`, fill: "rgba(62,214,206,.10)"}, ch);
    el("polyline", {points: line, fill: "none", stroke: "var(--disc)", "stroke-width": 1.8, "vector-effect": "non-scaling-stroke"}, ch);
    const hline = (v, col, dash, label) => {
      el("line", {x1: 0, x2: W, y1: Y(v), y2: Y(v), stroke: col, "stroke-width": 1.4, "stroke-dasharray": dash || "", "vector-effect": "non-scaling-stroke"}, ch);
      el("text", {x: W - 4, y: Y(v) - 5, "text-anchor": "end", fill: col, "font-family": "var(--mono)", "font-size": 11}, ch).textContent = label;
    };
    hline(now, "var(--muted)", "5 5", `now: ${nf(Math.min(c.technicians, c.cameras))} × ${nf(perTeam, 0)}/h`);
    if (needT && needT !== Math.min(c.technicians, c.cameras)) hline(need, "var(--g1)", "", `needed: ${needT} × ${nf(perTeam, 0)}/h`);
    el("text", {x: 4, y: 12, fill: "var(--faint)", "font-family": "var(--mono)", "font-size": 11}, ch).textContent = nf(top / 1.12, 0) + " /h";
    $("simRushX").innerHTML = `<span>opening</span><span>${hrs(T / 120)} h</span><span>closing, ${hrs(T / 60)} h</span>`;
    const peak = pts.reduce((a, b) => (b[1] > a[1] ? b : a));
    let note = `Arrivals peak at <b>${nf(peak[1], 0)}</b> an hour, ${hrs(Math.round(peak[0] / 6) / 10)} h after opening${one(c) ? "" : ", at each centre"}. ` +
      `Each capture team photographs ${nf(perTeam, 0)} an hour, so ${plural(Math.min(c.technicians, c.cameras), "team", "teams")} can take ${nf(now, 0)}. `;
    if (needT && overs.length) note += `With the ${plural(needT, "team", "teams")} needed, arrivals outrun capture from ${hrs(Math.round(overs[0][0] / 6) / 10)} h to ${hrs(Math.round(overs[overs.length - 1][1] / 6) / 10)} h (shaded): a short queue builds in the rush and clears after it, within the waiting target. Extra staff for those hours alone would shorten it.`;
    else if (needT) note += `The ${plural(needT, "team", "teams")} needed ${needT === 1 ? "keeps" : "keep"} ahead of arrivals all session.`;
    $("simRushNote").innerHTML = note;
  }
  const one = c => c.asBuilt;

  // ── start when the section comes near ───────────────────────────────────
  applyModel();
  let started = false;
  function near() {
    const r = box.getBoundingClientRect();
    if (!started && r.top < innerHeight * 1.6) { started = true; run(); }
    if (started) { removeEventListener("scroll", near); clearInterval(poll); }
  }
  addEventListener("scroll", near, {passive: true});
  const poll = setInterval(near, 700);
  near();
})();
