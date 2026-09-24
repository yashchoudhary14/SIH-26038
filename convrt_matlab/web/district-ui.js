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

  // ── the form ─────────────────────────────────────────────────────────────
  const F = form.elements;
  const IMPROVED_MODEL = {stochastic: true, priority: true, ai: true, captureMin: 4, reviewMin: 0.5};
  const AS_BUILT_MODEL = {stochastic: false, priority: false, ai: true, captureMin: 5, reviewMin: 0.5};
  const PRESETS = {
    district: {phcs: 12, technicians: 1, cameras: 1, devices: 1, ophthalmologists: 2, patientsPerDay: 33, sessionHours: 7, reviewHours: 5, days: 20, ai: true},
    double:   {phcs: 12, technicians: 1, cameras: 1, devices: 1, ophthalmologists: 2, patientsPerDay: 67, sessionHours: 7, reviewHours: 5, days: 20, ai: true},
    manual:   {phcs: 12, technicians: 1, cameras: 1, devices: 1, ophthalmologists: 2, patientsPerDay: 33, sessionHours: 7, reviewHours: 5, days: 20, ai: false},
    thin:     {phcs: 12, technicians: 1, cameras: 1, devices: 1, ophthalmologists: 1, patientsPerDay: 33, sessionHours: 7, reviewHours: 2, days: 20, ai: true}
  };
  const asBuilt = () => F.model.value === "asbuilt";

  function setField(name, v) {
    const el = F[name]; if (!el) return;
    if (el.type === "checkbox") el.checked = !!v; else el.value = v;
    syncField(el);
  }
  function syncField(el) {
    const out = $("v_" + el.name);
    if (out) out.textContent = el.type === "range" ? el.value + (el.dataset.unit || "") : el.value;
    const st = el.closest && el.closest(".st");
    if (st) {
      st.querySelector('[data-d="-1"]').disabled = +el.value <= +el.dataset.min;
      st.querySelector('[data-d="1"]').disabled = +el.value >= +el.dataset.max;
    }
  }
  function readForm() {
    const n = k => +F[k].value;
    const c = {asBuilt: asBuilt(), reps: n("reps"), seed: n("seed"),
      ai: F.ai.checked, priority: F.priority.checked, stochastic: F.stochastic.checked,
      captureMin: n("captureMin"), reviewMin: n("reviewMin"), sensitivity: n("sensitivity"), specificity: n("specificity"),
      pPass: n("pPass"), pRepair: n("pRepair"), maxAttempts: n("maxAttempts")};
    if (!c.asBuilt) Object.assign(c, {phcs: n("phcs"), technicians: n("technicians"), cameras: n("cameras"),
      devices: n("devices"), ophthalmologists: n("ophthalmologists"), patientsPerDay: n("patientsPerDay"),
      sessionHours: n("sessionHours"), reviewHours: n("reviewHours"), days: n("days")});
    // keep any one run to a couple of seconds: fewer replications for a very large district
    const load = c.asBuilt ? 30 : c.phcs * c.patientsPerDay * c.days;
    const maxReps = Math.max(2, Math.floor(420000 / load));
    c.cappedFrom = c.reps > maxReps ? c.reps : 0;
    c.reps = Math.min(c.reps, maxReps);
    return c;
  }
  function applyModel() {
    const ab = asBuilt();
    form.querySelectorAll('[data-scope="improved"]').forEach(g => { g.disabled = ab; });
    ["priority", "stochastic", "ai"].forEach(k => { F[k].disabled = ab; });
    $("simPresets").querySelectorAll("button").forEach(b => { b.disabled = ab; });
    const m = ab ? AS_BUILT_MODEL : IMPROVED_MODEL;
    Object.keys(m).forEach(k => setField(k, m[k]));
    $("simModelNote").textContent = ab
      ? "district_model.slx exactly: one health centre, one of each resource, fixed service times, a first-come queue, one 6-hour session."
      : "district_model_v2: resources you set, servers limited by them, variable service times, urgent-first review, and health centres sharing one district hub over working days.";
    const D = M.diagrams[ab ? "asbuilt" : "improved"];
    drawDiagram(D);
    $("simDiaSrc").textContent = ab ? D.source : D.source + " · one health centre shown";
    $("simDiaNote").textContent = ab ? "Figures for the 6-hour session · displays: totals" : "Health-centre blocks: per centre per day · review hub: the district per day · displays: totals over the run";
  }
  function applyPreset(name) {
    const pr = PRESETS[name]; if (!pr) return;
    Object.keys(pr).forEach(k => setField(k, pr[k]));
    $("simPresets").querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", b.dataset.preset === name ? "true" : "false"));
  }

  form.querySelectorAll(".st").forEach(st => {
    const input = st.querySelector("input");
    st.addEventListener("click", ev => {
      const b = ev.target.closest(".st-b"); if (!b || b.disabled) return;
      input.value = Math.max(+input.dataset.min, Math.min(+input.dataset.max, +input.value + +b.dataset.d));
      syncField(input); changed();
    });
    syncField(input);
  });
  form.querySelectorAll('input[type="range"]').forEach(r => { syncField(r); r.addEventListener("input", () => { syncField(r); changed(); }); });
  form.querySelectorAll('input[type="checkbox"], input[type="number"]').forEach(x => x.addEventListener("change", changed));
  form.querySelectorAll('input[name="model"]').forEach(x => x.addEventListener("change", () => { applyModel(); run(); }));
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
      worker = new Worker("district-sim.js?v=5");
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
  }

  // ── the diagram ──────────────────────────────────────────────────────────
  const svg = $("simDiagram"), NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
  let statNode = {};
  function drawDiagram(D) {
    svg.textContent = ""; statNode = {};
    const [x0, y0, x1, y1] = D.bounds;
    svg.setAttribute("viewBox", `${x0 - 14} ${y0 - 18} ${x1 - x0 + 28} ${y1 - y0 + 50}`);
    const defs = el("defs", {}, svg);
    const mk = el("marker", {id: "dm-ah", viewBox: "0 0 8 8", refX: 7.5, refY: 4, markerWidth: 7, markerHeight: 7, orient: "auto"}, defs);
    el("path", {d: "M0 0 8 4 0 8z", fill: "rgba(242,242,238,.5)"}, mk);
    const gl = el("g", {}, svg);
    D.lines.forEach(l => el("polyline", {points: l.pts.map(p => p.join(",")).join(" "),
      class: l.kind === "signal" ? "l-s" : "l-e", "marker-end": l.kind === "signal" ? "" : "url(#dm-ah)"}, gl));
    const gb = el("g", {}, svg);
    D.blocks.forEach(b => {
      const g = el("g", {transform: `translate(${b.x},${b.y})`}, gb);
      const t = b.type, w = b.w, h = b.h, cx = w / 2, cy = h / 2;
      const cls = t === "EntityResourcePool" ? "b-box b-pool" : (t === "Display" || t === "Scope") ? "b-box b-sink" : "b-box";
      el("rect", {width: w, height: h, rx: 3, class: cls}, g);
      if (t === "EntityServer") statNode[b.name + ":heat"] = el("rect", {x: 1, y: 1, width: w - 2, height: h - 2, rx: 2, class: "b-heat", fill: "rgba(62,214,206,0)"}, g);
      // a small glyph per block type, after Simulink's own icons
      const G = {class: "b-glyph"};
      if (t === "Queue") for (let i = 0; i < 4; i++) el("rect", Object.assign({x: w - 34 + i * 7, y: cy - 12, width: 5, height: 24}, G), g);
      else if (t === "EntityServer") el("ellipse", Object.assign({cx, cy, rx: Math.min(18, w / 3), ry: Math.min(16, h / 3)}, G), g);
      else if (t === "EntityResourceAcquirer") el("path", Object.assign({d: `M${cx - 12} ${cy - 8}v14h24v-14M${cx} ${cy - 16}v14m-5 -5 5 5 5-5`}, G), g);
      else if (t === "EntityResourceReleaser") el("path", Object.assign({d: `M${cx - 12} ${cy - 2}v14h24v-14M${cx} ${cy + 6}v-20m-5 5 5-5 5 5`}, G), g);
      else if (t === "EntityOutputSwitch") el("path", Object.assign({d: `M${cx - 16} ${cy - 6}h8l16-10M${cx - 8} ${cy - 6}l16 14`}, G), g);
      else if (t === "EntityInputSwitch") el("path", Object.assign({d: `M${cx - 16} ${cy - 12}l16 10h8M${cx - 16} ${cy + 8}l16-10`}, G), g);
      else if (t === "EntityGenerator") el("path", Object.assign({d: `M${cx - 14} ${cy - 10}q-6 10 0 20M${cx + 14} ${cy - 10}q6 10 0 20`}, G), g);
      else if (t === "EntityTerminator") el("path", Object.assign({d: `M${cx - 10} ${cy - 10}l20 20M${cx + 10} ${cy - 10}l-20 20`}, G), g);
      else if (t === "Scope") el("path", Object.assign({d: `M6 ${cy + 4}l6-8 6 10 6-12 6 8`}, G), g);
      else if (t === "EntityResourcePool") el("path", Object.assign({d: `M${cx - 16} ${cy - 4}v12h32v-12`, stroke: "var(--g1)"}, G), g);
      const name = t === "EntityResourcePool" ? (b.resource || b.name) : b.name;
      el("text", {x: cx, y: h + 15, class: "b-name"}, g).textContent = name;
      statNode[b.name] = el("text", {x: cx, y: t === "EntityResourcePool" ? cy - 10 : (t === "Display" ? cy + 5 : -7), class: "b-stat"}, g);
    });
  }
  function setStat(name, text, level) {
    const n = statNode[name]; if (!n) return;
    n.textContent = text; n.setAttribute("class", "b-stat" + (level ? " " + level : ""));
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
    const perUnit = c.asBuilt ? "per 6-hour session" : "per working day";
    const units = c.asBuilt ? 1 : c.phcs;

    // headline figures
    const urgentOk = m("urgentSla"), backlog = m("backlog");
    const K = [
      [nf(screened / days), `Patients screened ${perUnit}`, `${nf(screened)} in total`],
      [pct(m("cleared") / screened), "Cleared at the health centre", "no specialist needed"],
      [nf(m("reviewed")), "Reviewed by a specialist", `of ${nf(m("referred"))} referred`],
      [nf(backlog), "Waiting for review at the end", backlog > 0.05 * Math.max(1, m("reviewed")) ? "the backlog is growing" : "the queue clears", backlog > 0.05 * Math.max(1, m("reviewed")) ? "bad" : "ok"],
      [dur(m("toReviewMedian")), "Median wait for a specialist", "from grading to review"],
      [dur(m("toReviewP90")), "90% reviewed within", c.asBuilt ? "within the session" : "including nights and weekends off"],
      [c.asBuilt ? "—" : pct(urgentOk), "Urgent cases seen within 2 days", c.priority ? "urgent-first queue" : "first-come queue", c.asBuilt ? "" : urgentOk >= 0.98 ? "ok" : urgentOk >= 0.9 ? "warn" : "bad"],
      [pct(m("referableCaught"), 1), "Referable patients sent on", "grader plus ungradeable images"]
    ];
    $("simKpis").innerHTML = K.map(([v, l, s, cls]) => `<div class="kpi"><b class="${cls || ""}">${v}</b><span>${l}</span><small>${s}</small></div>`).join("");

    // utilisation
    const U = c.asBuilt
      ? [["Capture station", "technician and camera", m("util_capture")], ["Edge device", "grading", m("util_device")], ["Specialist review", "ophthalmologist", m("util_review")]]
      : [["Technicians", `${c.technicians} × ${c.phcs} centres`, m("util_technician")], ["Cameras", `${c.cameras} × ${c.phcs} centres`, m("util_camera")],
         ["Edge devices", `${c.devices} × ${c.phcs} centres`, m("util_device")], ["Ophthalmologists", `${c.ophthalmologists} at the hub, ${c.reviewHours} h/day`, m("util_review")]];
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
    const per = x => x / units;
    setStat("Arrivals", nf(per(m("arrived")) / (c.asBuilt ? 1 : days), 1));
    setStat("WaitingRoom", dur(m("captureWaitMean")));
    const uCap = c.asBuilt ? m("util_capture") : m("util_camera");
    setStat("CaptureStation", pct(uCap), level(uCap)); heat("CaptureStation", uCap);
    setStat("QualityGate", "↺ " + nf(per(m("retakes")) / (c.asBuilt ? 1 : days), 1));
    setStat("GradeOnDevice", pct(m("util_device")), level(m("util_device"))); heat("GradeOnDevice", m("util_device"));
    setStat("ClearedOnSpot", nf(per(m("cleared")) / (c.asBuilt ? 1 : days), 1));
    setStat("ReviewQueue", dur(m("toReviewMedian")));
    setStat("SpecialistReview", pct(m("util_review")), level(m("util_review"))); heat("SpecialistReview", m("util_review"));
    setStat("Reviewed", nf(m("reviewed") / (c.asBuilt ? 1 : days), 1));
    setStat("Patients_Cleared", nf(m("cleared")));
    setStat("Patients_Reviewed", nf(m("reviewed")));
    const amt = {PoolTechnician: c.asBuilt ? 1 : c.technicians, PoolCamera: c.asBuilt ? 1 : c.cameras,
      PoolEdgeDevice: c.asBuilt ? 1 : c.devices, PoolOphthalmologist: c.asBuilt ? 1 : c.ophthalmologists};
    Object.keys(amt).forEach(k => setStat(k, "× " + amt[k]));

    // in words
    const over = U.filter(u => u[2] > 1), hot = U.filter(u => u[2] >= 0.85 && u[2] <= 1);
    let words = c.asBuilt
      ? `In one 6-hour session the model as built screens <b>${nf(screened, 1)}</b> patients and clears <b>${pct(m("cleared") / screened)}</b> of them on the spot. `
      : `Over ${days} working days the district screens <b>${nf(screened / days)}</b> patients a day across ${c.phcs} health centres and clears <b>${pct(m("cleared") / screened)}</b> of them there. `;
    words += c.ai ? "" : "With no AI triage every gradeable image goes to a specialist, which is what the hub has to absorb. ";
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
    const days = c.asBuilt ? 0 : c.days;
    $("simChartX").innerHTML = c.asBuilt ? "<span>0 min</span><span>180</span><span>360 min</span>"
      : `<span>day 1</span><span>day ${Math.ceil(days / 2)}</span><span>day ${days}</span>`;
    $("simChartTitle").textContent = c.asBuilt ? "QUEUES THROUGH THE SESSION" : "QUEUES OVER THE WORKING DAYS";
  }

  // ── the check against MATLAB ─────────────────────────────────────────────
  function cfgFromRef(ref) {
    const p = ref.p, T = ref.stop_min;
    const c = {reps: 100, seed: 11, prevalence: p.prevalence_referable, pPass: p.p_pass, pRepair: p.p_repair,
      maxAttempts: p.max_attempts, sensitivity: p.sensitivity, specificity: p.specificity, patientsPerDay: T / p.mean_interarrival};
    if (ref.model !== "district_model_v2") return Object.assign(c, {asBuilt: true, sessionHours: T / 60});
    return Object.assign(c, {asBuilt: false, phcs: 1, days: 1, sessionHours: T / 60, horizonMin: T, reviewHours: 24, finishQueue: false,
      technicians: p.n_technicians, cameras: p.n_cameras, devices: p.n_devices, ophthalmologists: p.n_ophthalmologists, uplinks: p.n_uplinks,
      stochastic: !!p.stochastic, priority: true, captureMin: p.capture_min, captureCv: p.capture_cv, gradeMin: p.grade_min,
      uploadMin: p.upload_min, uploadCv: p.upload_cv, reviewMin: p.review_min, reviewCv: p.review_cv, urgentShare: p.urgent_share});
  }
  function validate() {
    const refs = M.references || {}, table = $("simValid");
    const ROWS = [
      ["ref_v1", "Simulink as shipped", "every run replays the same quality-gate and grader draws"],
      ["ref_v1_seeded", "Simulink as built, streams seeded per run", "district_model.slx, one of each resource"],
      ["ref_v2", "Improved model", "district_model_v2.slx, variable service times, urgent-first"],
      ["ref_v2b", "Improved model, 2 technicians + 2 cameras, twice the demand", "district_model_v2.slx"]
    ].filter(r => refs[r[0]]);
    if (!ROWS.length) { $("simValidText").textContent = "Reference runs from MATLAB were not found next to the model."; return; }
    // [MATLAB's name, the engine's, label, scale]; utilisation is compared as
    // SimEvents reports it — averaged up to each block's last event
    const MET = [["cleared", "cleared", "Cleared on the spot", 1], ["reviewed", "reviewed", "Reviewed", 1],
      ["util_capture", "se_capture", "Capture busy", 100], ["util_device", "se_device", "Device busy", 100], ["util_review", "se_review", "Review busy", 100]];
    const N = 1000;
    const fmt = (v, s) => s === 100 ? nf(100 * v, 1) + "%" : nf(v, 2);
    let html = "<thead><tr><th scope=\"col\">Configuration</th><th scope=\"col\">Source</th>" + MET.map(x => `<th scope="col">${x[2]}</th>`).join("") + "</tr></thead><tbody>";
    let agree = 0, total = 0;
    ROWS.forEach(([key, label, note]) => {
      const ref = refs[key], mine = Sim.run(Object.assign(cfgFromRef(ref), {reps: N})).summary;
      const shipped = key === "ref_v1";
      html += `<tr><th scope="row" rowspan="${shipped ? 1 : 2}">${esc(label)}<small class="dim"> · ${esc(note)}</small></th><td class="dim">MATLAB, ${ref.reps} runs</td>` +
        MET.map(([k, , , s]) => `<td>${fmt(ref.stats[k].mean, s)}</td>`).join("") + "</tr>";
      if (shipped) return;
      html += `<tr><td class="dim">this page, ${nf(N)} runs</td>` + MET.map(([k, k2, , s]) => {
        const a = ref.stats[k], b = mine[k2];
        const se = Math.sqrt(a.sd * a.sd / ref.reps + b.sd * b.sd / N) || 1e-9;
        const ok = Math.abs(a.mean - b.mean) <= 3 * se + 1e-6; total++; if (ok) agree++;
        return `<td class="${ok ? "ok" : "off"}">${fmt(b.mean, s)}</td>`;
      }).join("") + "</tr>";
    });
    table.innerHTML = html + "</tbody>";
    $("simValidNote").textContent = `${agree} of ${total} figures agree within sampling error`;
    const v1 = refs.ref_v1, v1s = refs.ref_v1_seeded;
    $("simValidText").innerHTML = (v1 && v1s ? `One finding came out of this check. In the model as shipped, each MATLAB action keeps its own random stream and restarts it from MATLAB's default seed on every run, so the quality gate and the grader replay the same numbers every time — and the start of that sequence runs high. Across ${v1.reps} runs the shipped model reports ${nf(v1.stats.reviewed.mean, 1)} reviews a session where its own parameters imply ${nf(v1s.stats.reviewed.mean, 1)}. Seeding every stream per run fixes it; the improved model does this itself. ` : "") +
      "A figure is shown as agreeing when the two means are within three standard errors of each other.";
  }

  // ── start when the section comes near ───────────────────────────────────
  applyModel(); applyPreset("district");
  let started = false, validated = false;
  function near() {
    const r = box.getBoundingClientRect();
    if (!started && r.top < innerHeight * 1.6) { started = true; run(); }
    const v = document.querySelector(".sim-valid");
    if (!validated && v && v.getBoundingClientRect().top < innerHeight * 1.6) { validated = true; setTimeout(validate, 60); }
    if (started && validated) { removeEventListener("scroll", near); clearInterval(poll); }
  }
  addEventListener("scroll", near, {passive: true});
  const poll = setInterval(near, 700);
  near();
})();
