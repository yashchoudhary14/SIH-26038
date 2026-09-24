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
  const ARR_DISTRICT = {lambdaPeakH: 11, tPeakH: 2.5, sigmaH: 1.5, N: 46, mu: 0.5};      // ~33 a 7-hour day
  const IMPROVED_MODEL = Object.assign({stochastic: true, priority: true, ai: true, captureMin: 4, reviewMin: 0.5}, ARR_DISTRICT);
  const AS_BUILT_MODEL = {stochastic: false, priority: false, ai: true, captureMin: 5, reviewMin: 0.5,
    lambdaPeakH: 11.25, tPeakH: 2, sigmaH: 1.25, N: 40, mu: 0.5};                          // the .slx's p: ~28.6 a 6-hour session
  const PRESETS = {
    district: Object.assign({phcs: 12, technicians: 1, cameras: 1, devices: 1, ophthalmologists: 2, sessionHours: 7, reviewHours: 5, days: 20, ai: true}, ARR_DISTRICT),
    // twice the catchment arriving twice as fast: twice the expected patients
    double:   Object.assign({phcs: 12, technicians: 1, cameras: 1, devices: 1, ophthalmologists: 2, sessionHours: 7, reviewHours: 5, days: 20, ai: true}, ARR_DISTRICT, {lambdaPeakH: 22, N: 92}),
    manual:   Object.assign({phcs: 12, technicians: 1, cameras: 1, devices: 1, ophthalmologists: 2, sessionHours: 7, reviewHours: 5, days: 20, ai: false}, ARR_DISTRICT),
    thin:     Object.assign({phcs: 12, technicians: 1, cameras: 1, devices: 1, ophthalmologists: 1, sessionHours: 7, reviewHours: 2, days: 20, ai: true}, ARR_DISTRICT)
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
    if (el.name in ARR_DISTRICT || el.name === "sessionHours") arrivals();
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
    Object.assign(c, arrivalCfg());
    if (!c.asBuilt) Object.assign(c, {phcs: n("phcs"), technicians: n("technicians"), cameras: n("cameras"),
      devices: n("devices"), ophthalmologists: n("ophthalmologists"),
      sessionHours: n("sessionHours"), reviewHours: n("reviewHours"), days: n("days")});
    // keep any one run to a couple of seconds: fewer replications for a very large district
    const load = c.asBuilt ? 30 : c.phcs * Sim.expected(c) * c.days;
    const maxReps = Math.max(2, Math.floor(420000 / load));
    c.cappedFrom = c.reps > maxReps ? c.reps : 0;
    c.reps = Math.min(c.reps, maxReps);
    return c;
  }
  // arrival parameters, from the form's hours into the model's minutes
  function arrivalCfg() {
    const n = k => +F[k].value;
    return {lambdaPeak: n("lambdaPeakH") / 60, tPeak: n("tPeakH") * 60, sigma: n("sigmaH") * 60, N: n("N"), mu: n("mu")};
  }
  function arrivals() {
    const svgA = $("simArrCurve"); if (!svgA) return;
    const cfg = Object.assign({asBuilt: asBuilt()}, arrivalCfg());
    if (!cfg.asBuilt) cfg.sessionHours = +F.sessionHours.value;
    const pts = Sim.arrivalCurve(cfg, 60), T = pts[pts.length - 1][0] || 1;
    const top = Math.max(1, ...pts.map(q => q[1]));
    const xy = pts.map(([t, v]) => `${(t / T * 300).toFixed(1)},${(66 - v / top * 60).toFixed(1)}`);
    svgA.innerHTML = `<polygon points="0,68 ${xy.join(" ")} 300,68" fill="rgba(62,214,206,.12)"/>` +
      `<polyline points="${xy.join(" ")}" fill="none" stroke="var(--disc)" stroke-width="1.5" vector-effect="non-scaling-stroke"/>` +
      `<text x="4" y="11" fill="var(--faint)" font-family="var(--mono)" font-size="9">${nf(top, 1)} /h</text>`;
    const peakAt = pts.reduce((a, b) => (b[1] > a[1] ? b : a))[0];
    $("simArrPeak").textContent = "busiest ≈ " + nf(peakAt / 60, 1) + " h in";
    $("simArrEnd").textContent = "closing, " + nf(T / 60, 1) + " h";
    const e = Sim.expected(cfg);
    $("simArrExp").textContent = `≈ ${nf(e, 1)} patients expected per centre per ${cfg.asBuilt ? "session" : "day"} (of N = ${cfg.N})`;
  }
  function applyModel() {
    const ab = asBuilt();
    form.querySelectorAll('[data-scope="improved"]').forEach(g => { g.disabled = ab; });
    ["priority", "stochastic", "ai"].forEach(k => { F[k].disabled = ab; });
    $("simPresets").querySelectorAll("button").forEach(b => { b.disabled = ab; });
    const m = ab ? AS_BUILT_MODEL : IMPROVED_MODEL;
    Object.keys(m).forEach(k => setField(k, m[k]));
    arrivals();
    $("simModelNote").textContent = ab
      ? "district_model.slx exactly: one health centre, one of each resource, fixed service times, a first-come queue, one 6-hour session."
      : "district_model_v2: resources you set, servers limited by them, variable service times, urgent-first review, and health centres sharing one district hub over working days.";
    const D = M.diagrams[ab ? "asbuilt" : "improved"];
    drawDiagram(D);
    $("simDiaSrc").textContent = ab ? D.source : D.source + " · one health centre shown";
    $("simDiaNote").textContent = ab ? "Figures inside blocks: the model's readings for the 6-hour session" : "Figures inside blocks: lanes 1–2 per centre per day, lane 3 for the whole district";
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
  // Blocks keep the model's own lanes and order, but are laid out on an even
  // grid with readable names; connections are routed from the model's own
  // source and destination ports. The Display and Scope blocks are not drawn
  // as boxes: their readings appear inside the blocks they measure.
  const NAMES = {Arrivals: "Arrivals", MergeRetakes: "Merge retakes", WaitingRoom: "Waiting room",
    AcqTechnician: "Get technician", AcqCamera: "Get camera", CaptureStation: "Capture", RelTechnician: "Free technician",
    RelCamera: "Free camera", QualityGate: "Quality gate", AcqDevice: "Get device", GradeOnDevice: "Grade on device",
    RelDevice: "Free device", RouteDecision: "Route", ClearedOnSpot: "Cleared on spot", UploadBuffer: "Upload buffer",
    Upload: "Upload", ReviewQueue: "Review queue", AcqOphthalmologist: "Get ophthalmologist", SpecialistReview: "Specialist review",
    RelOphthalmologist: "Free ophthalmologist", Reviewed: "Reviewed"};
  const KIND = {EntityGenerator: "Entity Generator", EntityInputSwitch: "Entity Input Switch", Queue: "Entity Queue",
    EntityResourceAcquirer: "Resource Acquirer", EntityServer: "Entity Server", EntityResourceReleaser: "Resource Releaser",
    EntityOutputSwitch: "Entity Output Switch", EntityTerminator: "Entity Terminator"};
  const LANES = ["CAPTURE", "GRADE · on the edge device", "REVIEW · at the district hub"];
  function wrap(text, max) {
    const words = text.split(" "), out = [""];
    words.forEach(w => { const cur = out[out.length - 1]; if (cur && (cur + " " + w).length > max) out.push(w); else out[out.length - 1] = cur ? cur + " " + w : w; });
    return out.slice(0, 2);
  }
  function glyph(g, t, cx, cy) {
    const G = {class: "b-glyph"};
    if (t === "Queue") for (let i = 0; i < 4; i++) el("rect", Object.assign({x: cx - 13 + i * 7, y: cy - 9, width: 5, height: 18}, G), g);
    else if (t === "EntityServer") el("ellipse", Object.assign({cx, cy, rx: 13, ry: 10}, G), g);
    else if (t === "EntityResourceAcquirer") el("path", Object.assign({d: `M${cx - 10} ${cy - 2}v10h20v-10M${cx} ${cy - 12}v12m-4 -4 4 4 4-4`}, G), g);
    else if (t === "EntityResourceReleaser") el("path", Object.assign({d: `M${cx - 10} ${cy - 2}v10h20v-10M${cx} ${cy + 4}v-16m-4 4 4-4 4 4`}, G), g);
    else if (t === "EntityOutputSwitch") el("path", Object.assign({d: `M${cx - 12} ${cy}h7l12-8M${cx - 5} ${cy}l12 8`}, G), g);
    else if (t === "EntityInputSwitch") el("path", Object.assign({d: `M${cx - 12} ${cy - 8}l12 8h7M${cx - 12} ${cy + 8}l12-8`}, G), g);
    else if (t === "EntityGenerator") el("path", Object.assign({d: `M${cx - 9} ${cy - 9}q-5 9 0 18M${cx + 9} ${cy - 9}q5 9 0 18`}, G), g);
    else if (t === "EntityTerminator") el("path", Object.assign({d: `M${cx - 8} ${cy - 8}l16 16M${cx + 8} ${cy - 8}l-16 16`}, G), g);
  }
  function drawDiagram(D) {
    svg.textContent = ""; statNode = {};
    const W = 1200, LX = 46, GUT = 26, BW = 102, BH = 68, LABEL = 40, GAP = 56, T1 = 70;
    const flow = D.blocks.filter(b => KIND[b.type]);
    const rows = [...new Set(flow.map(b => Math.round(b.y / 50) * 50))].sort((a, b) => a - b);
    const lanes = rows.map(r => flow.filter(b => Math.round(b.y / 50) * 50 === r).sort((a, b) => a.x - b.x));
    const cols = Math.max(...lanes.map(l => l.length)), pitch = (W - LX - 18) / cols;
    const laneTop = i => T1 + i * (BH + LABEL + GAP);
    const pos = {};
    lanes.forEach((lane, li) => lane.forEach((b, ci) => { pos[b.name] = {b, lane: li, col: ci, x: LX + ci * pitch + (pitch - BW) / 2, y: laneTop(li)}; }));
    const poolTop = laneTop(lanes.length) - 6;
    svg.setAttribute("viewBox", `0 0 ${W} ${poolTop + 52}`);

    const defs = el("defs", {}, svg);
    const mk = el("marker", {id: "dm-ah", viewBox: "0 0 8 8", refX: 7.5, refY: 4, markerWidth: 7, markerHeight: 7, orient: "auto"}, defs);
    el("path", {d: "M0 0 8 4 0 8z", fill: "rgba(242,242,238,.55)"}, mk);

    // lane titles
    lanes.forEach((_, li) => { el("text", {x: LX, y: laneTop(li) - 14, class: "b-lane"}, svg).textContent = `${li + 1}  ${LANES[li] || ""}`; });

    // connections, from the model's own ports
    const gl = el("g", {}, svg);
    D.lines.filter(l => l.src && pos[l.src] && pos[l.dst]).forEach(l => {
      const a = pos[l.src], b = pos[l.dst], ay = a.y + BH / 2, by = b.y + BH / 2;
      let d;
      if (a.lane === b.lane && b.col === a.col + 1) d = `M${a.x + BW} ${ay}H${b.x - 2}`;
      else if (a.lane === b.lane && b.col < a.col) {                      // the retake loop, over the lane
        const top = a.y - 34;
        d = `M${a.x + BW / 2} ${a.y}V${top}H${b.x + BW / 2}V${b.y - 2}`;
        el("text", {x: (a.x + b.x + BW) / 2, y: top - 6, class: "b-loop"}, gl).textContent = "retake: back to the waiting room";
      } else if (a.lane === b.lane) d = `M${a.x + BW} ${ay}H${b.x - 2}`;
      else {                                                               // down to the next lane, through the gap
        const sy = l.srcPort > 1 && a.b.out > 1 ? a.y + BH * 0.78 : ay;
        const gy = a.y + BH + LABEL + GAP / 2 - 4;
        d = `M${a.x + BW} ${sy}H${a.x + BW + (pitch - BW) / 2}V${gy}H${GUT}V${by}H${b.x - 2}`;
      }
      el("path", {d, class: "l-e", "marker-end": "url(#dm-ah)"}, gl);
    });

    // blocks
    const gb = el("g", {}, svg);
    Object.values(pos).forEach(({b, x, y}) => {
      const g = el("g", {transform: `translate(${x.toFixed(1)},${y})`, class: "b"}, gb);
      el("title", {}, g).textContent = `${b.name} — ${KIND[b.type]}`;
      el("rect", {width: BW, height: BH, rx: 4, class: "b-box"}, g);
      if (b.type === "EntityServer") statNode[b.name + ":heat"] = el("rect", {x: 1, y: 1, width: BW - 2, height: BH - 2, rx: 3, class: "b-heat", fill: "rgba(62,214,206,0)"}, g);
      glyph(g, b.type, BW / 2, 24);
      statNode[b.name] = el("text", {x: BW / 2, y: 55, class: "b-stat"}, g);
      wrap(NAMES[b.name] || b.name, 15).forEach((ln, k) => { el("text", {x: BW / 2, y: BH + 17 + k * 15, class: "b-name"}, g).textContent = ln; });
    });

    // the resource pools, as the strip they are
    const ORDER = ["Technician", "Camera", "EdgeDevice", "Ophthalmologist"];
    const pools = D.blocks.filter(b => b.type === "EntityResourcePool")
      .sort((a, b) => ORDER.indexOf(a.resource) - ORDER.indexOf(b.resource));
    el("text", {x: LX, y: poolTop + 22, class: "b-lane"}, svg).textContent = "RESOURCE POOLS";
    pools.forEach((b, i) => {
      const g = el("g", {transform: `translate(${LX + 150 + i * 190},${poolTop})`}, svg);
      el("title", {}, g).textContent = `${b.name} — Entity Resource Pool`;
      el("rect", {width: 176, height: 34, rx: 4, class: "b-box b-pool"}, g);
      el("text", {x: 14, y: 22, class: "b-chip"}, g).textContent = (b.resource || b.name).replace("EdgeDevice", "Edge device");
      statNode[b.name] = el("text", {x: 162, y: 22, class: "b-chip-v"}, g);
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
    const u = c.asBuilt ? "" : " /day";
    setStat("Arrivals", nf(per(m("arrived")) / (c.asBuilt ? 1 : days), 1) + u);
    setStat("WaitingRoom", dur(m("captureWaitMean")));
    const uCap = c.asBuilt ? m("util_capture") : m("util_camera");
    setStat("CaptureStation", pct(uCap), level(uCap)); heat("CaptureStation", uCap);
    setStat("QualityGate", "↺ " + nf(per(m("retakes")) / (c.asBuilt ? 1 : days), 1) + u);
    setStat("GradeOnDevice", pct(m("util_device")), level(m("util_device"))); heat("GradeOnDevice", m("util_device"));
    setStat("ClearedOnSpot", nf(per(m("cleared")) / (c.asBuilt ? 1 : days), 1) + u);
    setStat("ReviewQueue", dur(m("toReviewMedian")));
    setStat("SpecialistReview", pct(m("util_review")), level(m("util_review"))); heat("SpecialistReview", m("util_review"));
    setStat("Reviewed", nf(m("reviewed") / (c.asBuilt ? 1 : days), 0) + u);
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
      maxAttempts: p.max_attempts, sensitivity: p.sensitivity, specificity: p.specificity,
      lambdaPeak: p.lambda_peak, tPeak: p.t_peak, sigma: p.sigma, N: p.N, mu: p.mu};
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
    const MET = [["arrived", "arrived", "Arrived", 1], ["cleared", "cleared", "Cleared on the spot", 1], ["reviewed", "reviewed", "Reviewed", 1],
      ["util_capture", "se_capture", "Capture busy", 100], ["util_device", "se_device", "Device busy", 100], ["util_review", "se_review", "Review busy", 100]];
    const N = 1000;
    // (older reference runs did not log arrivals)
    for (let i = MET.length - 1; i >= 0; i--) if (!ROWS.every(r => refs[r[0]].stats[MET[i][0]])) MET.splice(i, 1);
    const fmt = (v, s) => s === 100 ? nf(100 * v, 1) + "%" : nf(v, 2);
    let html = "<thead><tr><th scope=\"col\">Source</th>" + MET.map(x => `<th scope="col">${x[2]}</th>`).join("") + "</tr></thead><tbody>";
    let agree = 0, total = 0;
    ROWS.forEach(([key, label, note]) => {
      const ref = refs[key], mine = Sim.run(Object.assign(cfgFromRef(ref), {reps: N})).summary;
      const shipped = key === "ref_v1";
      html += `<tr class="cfg"><th scope="rowgroup" colspan="${MET.length + 1}">${esc(label)}<small>${esc(note)}</small></th></tr>` +
        `<tr><td class="src">MATLAB, ${ref.reps} runs</td>` +
        MET.map(([k, , , s]) => `<td>${fmt(ref.stats[k].mean, s)}</td>`).join("") + "</tr>";
      if (shipped) return;
      html += `<tr><td class="src">this page, ${nf(N)} runs</td>` + MET.map(([k, k2, , s]) => {
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
