const GRADES = ["No apparent DR","Mild NPDR","Moderate NPDR","Severe NPDR","Proliferative DR"];
const SHORT = ["No DR","Mild","Moderate","Severe","Proliferative"];
const DECISION = {
  auto_report: "Auto-reported — model confident, no referral",
  refer: "Referral recommended",
  defer_to_human: "Deferred for human grading — model uncertain",
  recapture: "Ungradeable — recapture required"
};
const CRITERION_LABEL = {
  focus:"Focus", illumination:"Illumination", contrast:"Contrast", fov:"Field of view",
  macula:"Macula in frame", artifact:"Artefacts", under_exposure:"Exposure (low)",
  over_exposure:"Exposure (high)", noise:"Noise"
};

const $ = s => document.querySelector(s);
const $$ = s => Array.from(document.querySelectorAll(s));
const esc = s => String(s).replace(/[&<>"']/g, c =>
  ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const pct = (v, d = 1) => (v == null || isNaN(v)) ? "—" : (v * 100).toFixed(d) + "%";
const num = (v, d = 3) => (v == null || isNaN(v)) ? "—" : Number(v).toFixed(d);
const int = v => (v == null || isNaN(v)) ? "—" : Math.round(v).toLocaleString("en-IN");

function inr(x){
  if (x == null || isNaN(x)) return "—";
  if (x >= 1e7) return "₹ " + (x / 1e7).toFixed(2) + " Cr";
  if (x >= 1e5) return "₹ " + (x / 1e5).toFixed(2) + " L";
  return "₹ " + Math.round(x).toLocaleString("en-IN");
}

function ciSpan(p){
  if (!p) return "";
  return `<span class="ci">[${num(p.lower)}–${num(p.upper)}]</span>`;
}

async function api(path){
  const r = await fetch(path);
  if (!r.ok){
    let msg = await r.text();
    try { msg = JSON.parse(msg).detail || msg; } catch(e){}
    const err = new Error(msg); err.status = r.status; throw err;
  }
  return r.json();
}

function emptyState(host, title, body, cmd){
  host.innerHTML = `<div class="card"><div class="empty">
    <h3>${esc(title)}</h3>
    <div style="max-width:620px;margin:0 auto 14px">${body}</div>
    ${cmd ? `<code>${esc(cmd)}</code>` : ""}
  </div></div>`;
}

const TABS = ["screen","capture","worklist","validation","programme","audit"];
let activeTab = "screen";
const loaded = {};

function showTab(name){
  if (!TABS.includes(name)) return;
  activeTab = name;
  $$("nav.tabs button").forEach(b => b.classList.toggle("active", b.dataset.tab === name));
  $$(".view").forEach(v => v.classList.toggle("active", v.id === "view-" + name));
  window.scrollTo({top:0, behavior:"instant"});
  if (!loaded[name]){
    loaded[name] = true;
    if (name === "validation") loadValidation();
    if (name === "programme") loadProgramme();
    if (name === "audit") loadAudit();
  }
}

document.addEventListener("keydown", e => {
  if (e.target.matches("input,textarea,select")) return;
  const i = parseInt(e.key, 10);
  if (i >= 1 && i <= TABS.length) showTab(TABS[i - 1]);
});

let HEALTH = null;

async function loadHealth(){
  try {
    const h = await api("/health");
    HEALTH = h;
    $("#status").innerHTML =
      `<span><span class="dot ${h.segmentation_loaded?'on':'off'}"></span>segmentation</span>
       <span><span class="dot ${h.grader_loaded?'on':'off'}"></span>grader</span>
       <span>device <b>${esc(h.device)}</b></span>
       <span>thr <b>${num(h.referral_threshold)}</b></span>
       <span>T <b>${num(h.temperature,2)}</b></span>
       <span>${esc(h.model_version)}</span>`;
    const missing = [];
    if (!h.segmentation_loaded) missing.push("lesion segmentation");
    if (!h.grader_loaded) missing.push("severity grader");
    if (missing.length){
      const el = $("#degraded");
      el.classList.remove("hidden");
      el.innerHTML = `<b>Running in degraded mode — ${esc(missing.join(" and "))} not loaded.</b><br>
        Screening still runs end to end, but severity comes from the rule engine.
        Build the artefacts with <code>python scripts/run_all.py</code> and restart.`;
    }
  } catch(e){
    $("#status").textContent = "backend unreachable";
    const el = $("#degraded");
    el.classList.remove("hidden");
    el.className = "banner err";
    el.innerHTML = `<b>Cannot reach the backend.</b> Start it from the project root with
      <code>python -m uvicorn drscreen.api:app --port 8000</code>, then reload.`;
  }
}

let currentFile = null, currentResult = null, reviewStart = 0, timerId = null;

const drop = $("#drop"), fileInput = $("#file");
drop.addEventListener("dragover", e => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", e => {
  e.preventDefault(); drop.classList.remove("over");
  if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", e => { if (e.target.files.length) setFile(e.target.files[0]); });

function setFile(f){
  currentFile = f;
  drop.innerHTML = `<b>${esc(f.name)}</b><br><small>${(f.size/1024).toFixed(0)} KB · click to change</small>`;
  drop.appendChild(fileInput);
  $("#btn-run").disabled = false;
  $("#btn-report").disabled = false;
}

$("#btn-run").addEventListener("click", () => {
  if (!currentFile) return;
  const fd = new FormData(); fd.append("file", currentFile);
  runRequest(fetch("/screen", {method:"POST", body:fd}));
});

$("#btn-report").addEventListener("click", () => {
  if (!currentFile) return;
  const fd = new FormData(); fd.append("file", currentFile);
  fetch("/screen/report", {method:"POST", body:fd})
    .then(r => r.text())
    .then(html => { const w = window.open("", "_blank"); w.document.write(html); w.document.close(); });
});

$("#sev").addEventListener("input", e => {
  const v = +e.target.value;
  $("#sevlab").textContent = v < 25 ? "good" : v < 55 ? "borderline" : v < 80 ? "poor" : "very poor";
});

$$(".demo").forEach(b => b.addEventListener("click", () => {
  const sev = (+$("#sev").value) / 100;
  runRequest(fetch(`/demo/${b.dataset.g}?severity=${sev}`));
}));

async function runRequest(promise){
  $("#empty").classList.remove("hidden");
  $("#empty").innerHTML = '<div class="empty"><span class="spin"></span>Running the pipeline…</div>';
  $("#result").classList.add("hidden");
  try {
    const res = await promise;
    if (!res.ok) throw new Error(await res.text());
    renderResult(await res.json());
  } catch(err){
    $("#empty").innerHTML = `<div class="empty"><h3>Request failed</h3>${esc(err.message)}</div>`;
  }
}

function renderResult(r){
  currentResult = r;
  $("#empty").classList.add("hidden");
  $("#result").classList.remove("hidden");

  const cls = r.decision === "recapture" ? "recapture" : r.urgency;
  const label = r.decision === "recapture" ? "RECAPTURE"
              : r.urgency === "urgent" ? "URGENT REFERRAL"
              : r.urgency === "soon" ? "REFER" : "ROUTINE";
  $("#verdict").className = "verdict v-" + cls;
  $("#verdict").innerHTML = `
    <div class="u">${label}</div>
    <div class="g">${r.gradeable ? `Grade ${r.grade} — ${GRADES[r.grade]}` : "Ungradeable"}</div>
    <div class="d">${DECISION[r.decision] || esc(r.decision)}
      ${r.gradeable ? ` · P(referable) = <b>${num(r.referable_probability)}</b>
        · confidence ${pct(r.confidence)}` : ""}</div>`;

  const warns = [];
  if (r.quality?.overall === "borderline")
    warns.push("Image quality borderline; enhancement applied — interpret subtle findings with caution.");
  if (r.agreement === "disagree")
    warns.push(`Deep model (grade ${r.grade}) and rule-based ICDR criteria (grade ${r.rule_based_grade}) disagree.`);
  if ((r.uncertainty?.epistemic_variance || 0) > 0.05)
    warns.push("High model uncertainty on this image.");
  (r.recapture_advice || []).forEach(a => warns.push(a));
  $("#warnings").innerHTML = warns.map(w => `<div class="warn">${esc(w)}</div>`).join("");

  const panel = $("#panel");
  panel.src = r.panel_jpeg_b64 ? "data:image/jpeg;base64," + r.panel_jpeg_b64 : "";
  panel.style.display = r.panel_jpeg_b64 ? "block" : "none";

  $("#probs").innerHTML = (r.class_probabilities || []).map((p, g) => `
    <div class="bar"><span class="lb">${g} · ${SHORT[g]}</span>
      <span class="tr"><span class="fl ${g===r.grade?'pred':''}" style="width:${Math.max(0.4,p*100)}%"></span></span>
      <span class="vl">${pct(p)}</span></div>`).join("");

  $("#evidence").innerHTML = (r.evidence || []).map(e => {
    if (e.finding){
      const q = Object.entries(e.per_quadrant || {}).map(([k,v]) => `${k} ${v}`).join(", ");
      return `<li><b>${esc(e.finding)}</b>: ${e.count} detected${q?` (${esc(q)})`:""}
              — ${e.area_percent}% of retinal area</li>`;
    }
    if (e.criterion) return `<li class="criterion">${esc(e.criterion)}</li>`;
    if (e.macular_assessment) return `<li class="macula">${esc(e.macular_assessment)}</li>`;
    if (e.caution) return `<li class="caution">${esc(e.caution)}</li>`;
    return "";
  }).join("") || "<li>No lesions detected.</li>";

  $("#cross").innerHTML = `
    <div class="kv"><span>Deep model grade</span><b>${r.grade}</b></div>
    <div class="kv"><span>Rule-based grade</span><b>${r.rule_based_grade}</b></div>
    <div class="kv"><span>Agreement</span><b>${esc(r.agreement || "—")}</b></div>
    <div class="kv"><span>Macular oedema risk</span><b>${esc(String(r.dme_risk))}</b></div>
    <div class="kv"><span>Predictive entropy</span><b>${num(r.uncertainty?.entropy)}</b></div>
    <div class="kv"><span>Epistemic variance</span><b>${num(r.uncertainty?.epistemic_variance, 4)}</b></div>`;

  $("#quality").innerHTML =
    `<tr><th>Criterion</th><th class="n">Score</th><th>Verdict</th></tr>` +
    Object.entries(r.quality?.scores || {}).map(([k,v]) => {
      const vd = (r.quality.verdicts || {})[k] || "pass";
      return `<tr><td>${esc(CRITERION_LABEL[k] || k.replace(/_/g," "))}</td>
              <td class="n">${num(v,2)}</td><td class="${vd}">${vd}</td></tr>`;
    }).join("");

  $("#timing").innerHTML = Object.entries(r.timing_ms || {}).map(([k,v]) =>
    `<tr><td>${esc(k)}</td><td class="n">${num(v,1)} ms</td></tr>`).join("");

  const gt = r.ground_truth;
  $("#gt-card").style.display = gt ? "block" : "none";
  if (gt){
    const hit = gt.grade === r.grade ? "pass" : (Math.abs(gt.grade - r.grade) === 1 ? "borderline" : "fail");
    $("#gt").innerHTML = `
      <div class="kv"><span>True grade</span><b class="${hit}">${gt.grade} · ${SHORT[gt.grade]}</b></div>
      <div class="kv"><span>Camera</span><b>${esc(gt.camera)}</b></div>
      ${Object.entries(gt.lesion_counts || {}).filter(([,v]) => v > 0)
        .map(([k,v]) => `<div class="kv"><span>${esc(k.replace(/_/g," "))}</span><b>${v}</b></div>`).join("")}`;
  }
  buildReview(r);
}

function buildReview(r){
  reviewStart = Date.now();
  clearInterval(timerId);
  timerId = setInterval(() => {
    $("#timer").textContent = `${((Date.now() - reviewStart) / 1000).toFixed(0)}s`;
  }, 200);
  $("#review").innerHTML = SHORT.map((g, i) =>
    `<button class="${i === r.grade ? '' : 'ghost'}" data-grade="${i}">${i === r.grade ? '✓ ' : ''}${i}</button>`
  ).join("");
  $$("#review button").forEach(b =>
    b.addEventListener("click", () => submitReview(+b.dataset.grade)));
}

async function submitReview(grade){
  if (!currentResult) return;
  clearInterval(timerId);
  const secs = (Date.now() - reviewStart) / 1000;
  const fd = new FormData();
  fd.append("image_id", currentResult.image_id);
  fd.append("model_grade", currentResult.grade);
  fd.append("reviewer_grade", grade);
  fd.append("seconds", secs.toFixed(1));
  const res = await fetch("/review", {method:"POST", body:fd}).then(r => r.json());
  $("#review").innerHTML =
    `<div style="font-size:12.5px;color:var(--muted)">Recorded: reviewer graded
     <b style="color:var(--text)">${grade}</b>, model said
     <b style="color:var(--text)">${currentResult.grade}</b>
     (${esc(res.agreement)}) in ${secs.toFixed(1)}s${secs <= 30 ? " — within the 30s target" : ""}.</div>`;
  loaded.audit = false;
}

$("#cap-sev").addEventListener("input", e => {
  const v = +e.target.value;
  $("#cap-sevlab").textContent = v < 25 ? "clean capture" : v < 55 ? "some degradation" : v < 80 ? "poor capture" : "very poor capture";
});

$("#cap-run").addEventListener("click", async () => {
  const sev = (+$("#cap-sev").value) / 100;
  const g = +$("#cap-grade").value;
  const body = $("#cap-body");
  body.innerHTML = '<div class="empty"><span class="spin"></span>Capturing…</div>';
  try {
    const r = await api(`/demo/${g}?severity=${sev}`);
    renderCapture(r);
  } catch(e){
    body.innerHTML = `<div class="empty"><h3>Capture failed</h3>${esc(e.message)}</div>`;
  }
});

function renderCapture(r){
  const q = r.quality || {};
  const ok = r.gradeable;
  const chips = Object.entries(q.scores || {}).map(([k,v]) => {
    const vd = (q.verdicts || {})[k] || "pass";
    return `<div class="chip ${vd}">${esc(CRITERION_LABEL[k] || k)}<span class="sc">${num(v,2)}</span></div>`;
  }).join("");

  const advice = (r.recapture_advice || []);
  $("#cap-body").innerHTML = `
    ${r.capture_jpeg_b64 ? `<img class="ph-shot" src="data:image/jpeg;base64,${r.capture_jpeg_b64}" alt="capture">` : ""}
    <div class="ph-verdict" style="background:${ok?'#0f2e1e':'#2c1414'};border:1px solid ${ok?'#1f5c39':'#6d2020'};color:${ok?'#b6f0cf':'#ffc9c9'}">
      <div class="t">${ok ? "Capture accepted" : "Recapture required"}</div>
      <div class="s">${ok
        ? `Quality ${esc(q.overall)} · uploaded for grading`
        : `Quality gate: ${esc(q.overall)} · not sent for grading`}</div>
    </div>
    ${advice.length ? `<div class="ph-advice"><b>Technician instruction</b>
      ${advice.map(a => `<div style="margin-bottom:6px">${esc(a)}</div>`).join("")}</div>` : ""}
    <div style="font-size:10.5px;text-transform:uppercase;letter-spacing:1.1px;color:var(--muted);
                font-weight:650;margin:14px 0 8px">Quality criteria</div>
    <div class="chips">${chips}</div>
    <div style="margin-top:14px;font-size:11.5px;color:var(--muted)">
      Gate ran in ${num(r.timing_ms?.quality, 0)} ms on CPU. This stage is designed to run
      on the capture device, before any image leaves the health centre.
    </div>`;
}

let queue = [], queueRunning = false;

$("#wl-run").addEventListener("click", runQueue);
$("#wl-clear").addEventListener("click", () => {
  queue = []; renderQueue();
  $("#wl-progress").classList.add("hidden");
});

async function runQueue(){
  if (queueRunning) return;
  queueRunning = true;
  queue = [];
  const n = +$("#wl-n").value;
  const sev = (+$("#wl-sev").value) / 100;
  const btn = $("#wl-run");
  btn.disabled = true;
  btn.textContent = "Running…";
  $("#wl-progress").classList.remove("hidden");

  const weights = [60, 22, 11, 4, 3];
  const total = weights.reduce((a,b) => a + b, 0);
  const draw = () => {
    let x = Math.random() * total;
    for (let g = 0; g < weights.length; g++){ x -= weights[g]; if (x <= 0) return g; }
    return 0;
  };
  for (let i = 0; i < n; i++){
    const g = draw();
    const seed = 10000 + i * 7 + Math.floor(Math.random() * 5000);
    try {
      const r = await api(`/demo/${g}?severity=${sev}&seed=${seed}&explain=false`);
      queue.push({
        id: "PHC-" + String(1001 + i),
        site: "PHC " + (1 + (i % 8)),
        truth: r.ground_truth?.grade,
        r
      });
    } catch(e){
      queue.push({id:"PHC-" + String(1001 + i), site:"PHC " + (1 + (i % 8)), error:e.message});
    }
    $("#wl-prog-fill").style.width = ((i + 1) / n * 100) + "%";
    $("#wl-prog-lab").textContent = `${i + 1} of ${n} screened`;
    renderQueue();
  }
  btn.disabled = false;
  btn.textContent = "Run screening queue";
  queueRunning = false;
}

function urgencyRank(c){
  if (!c.r) return 5;
  if (!c.r.gradeable) return 3;
  return c.r.urgency === "urgent" ? 0 : c.r.urgency === "soon" ? 1 : 2;
}

function renderQueue(){
  if (!queue.length){
    $("#wl-stats").innerHTML = "";
    $("#wl-table").innerHTML = "";
    $("#wl-empty").classList.remove("hidden");
    return;
  }
  $("#wl-empty").classList.add("hidden");

  const done = queue.filter(c => c.r);
  const auto = done.filter(c => c.r.decision === "auto_report").length;
  const refer = done.filter(c => c.r.decision === "refer").length;
  const defer = done.filter(c => c.r.decision === "defer_to_human").length;
  const ungr = done.filter(c => !c.r.gradeable).length;
  const human = refer + defer + ungr;
  const matched = done.filter(c => c.r.gradeable && c.truth != null && c.truth === c.r.grade).length;
  const gradeable = done.filter(c => c.r.gradeable && c.truth != null).length;

  $("#wl-stats").innerHTML = `
    <div class="stat good"><div class="lab">Auto-cleared</div><div class="big">${auto}</div>
      <div class="sub">${done.length ? pct(auto / done.length, 0) : "—"} of the queue needs no human</div></div>
    <div class="stat bad"><div class="lab">Escalated</div><div class="big">${human}</div>
      <div class="sub">${refer} referral · ${defer} deferred · ${ungr} ungradeable</div></div>
    <div class="stat accent"><div class="lab">Reviewer workload</div><div class="big">${done.length ? pct(human / done.length, 0) : "—"}</div>
      <div class="sub">of images reaching an ophthalmologist</div></div>
    <div class="stat"><div class="lab">Exact grade match</div><div class="big">${matched}/${gradeable}</div>
      <div class="sub">against synthetic ground truth</div></div>`;

  const sorted = queue.slice().sort((a,b) => urgencyRank(a) - urgencyRank(b));
  $("#wl-table").innerHTML = `
    <table>
      <tr><th>Case</th><th>Site</th><th>Grade</th><th>Decision</th><th class="n">P(ref)</th>
          <th class="n">Conf</th><th>Truth</th><th>Priority</th></tr>
      ${sorted.map(c => {
        if (c.error) return `<tr><td>${esc(c.id)}</td><td>${esc(c.site)}</td>
          <td colspan="6" class="fail">${esc(c.error)}</td></tr>`;
        const r = c.r;
        const pri = !r.gradeable ? `<span class="pill grey">recapture</span>`
          : r.urgency === "urgent" ? `<span class="pill urgent">urgent</span>`
          : r.urgency === "soon" ? `<span class="pill soon">soon</span>`
          : `<span class="pill routine">routine</span>`;
        const match = r.gradeable && c.truth != null
          ? (c.truth === r.grade ? `<span class="pass">${c.truth}</span>`
             : `<span class="${Math.abs(c.truth - r.grade) === 1 ? 'borderline' : 'fail'}">${c.truth}</span>`)
          : "—";
        return `<tr class="clickable" data-id="${esc(c.id)}">
          <td class="mono">${esc(c.id)}</td>
          <td>${esc(c.site)}</td>
          <td>${r.gradeable ? `${r.grade} · ${SHORT[r.grade]}` : "—"}</td>
          <td>${esc((DECISION[r.decision] || r.decision).split(" — ")[0])}</td>
          <td class="n">${r.gradeable ? num(r.referable_probability) : "—"}</td>
          <td class="n">${r.gradeable ? pct(r.confidence, 0) : "—"}</td>
          <td class="n">${match}</td>
          <td>${pri}</td></tr>`;
      }).join("")}
    </table>`;

  $$("#wl-table tr.clickable").forEach(tr => tr.addEventListener("click", () => {
    const c = queue.find(x => x.id === tr.dataset.id);
    if (c && c.r){ renderResult(c.r); showTab("screen"); }
  }));
}

async function loadValidation(){
  const host = $("#view-validation");
  host.innerHTML = '<div class="card"><div class="empty"><span class="spin"></span>Loading validation report…</div></div>';
  let v;
  try {
    v = await api("/validation");
  } catch(e){
    emptyState(host, "No validation report on disk",
      `The clinical validation report is generated by a script, not at runtime.
       Build a cohort and run the validator, then reload this tab.`,
      "python scripts/validate.py --cohort data/cohort_synth --grader outputs/artifacts/grader.pt --seg outputs/artifacts/segmentation.pt");
    return;
  }
  renderValidation(host, v);
}

function metricRow(label, p, target){
  if (!p) return "";
  const meets = target != null ? p.value >= target : null;
  return `<div class="kv"><span>${esc(label)}</span>
    <b class="${meets === null ? "" : meets ? "pass" : "fail"}">${num(p.value)} ${ciSpan(p)}</b></div>`;
}

function splitCard(title, res, subtitle){
  if (!res) return `<div class="card"><h2>${esc(title)}</h2><p class="note">Not available.</p></div>`;
  const b = res.referable || {};
  return `<div class="card">
    <h2>${esc(title)}<span class="side">n = ${int(res.n)}</span></h2>
    <p class="note">${esc(subtitle)}</p>
    ${metricRow("Sensitivity (referable)", b.sensitivity, 0.90)}
    ${metricRow("Specificity (referable)", b.specificity, 0.80)}
    ${metricRow("PPV", b.ppv)}
    ${metricRow("NPV", b.npv)}
    <div class="kv"><span>AUC</span><b>${num(b.auc, 4)}
      <span class="ci">[${num(b.auc_ci?.[0],4)}–${num(b.auc_ci?.[1],4)}]</span></b></div>
    <div class="kv"><span>QWK (ordinal agreement)</span><b>${num(res.qwk?.value, 4)}
      <span class="ci">[${num(res.qwk?.lower,3)}–${num(res.qwk?.upper,3)}]</span></b></div>
    <div class="kv"><span>Exact accuracy</span><b>${num(res.exact_accuracy?.value)}</b></div>
    <div class="kv"><span>Adjacent (within-one-grade)</span><b>${num(res.adjacent_accuracy?.value)}</b></div>
    <div class="kv"><span>ECE</span><b>${num(res.calibration?.ece, 4)}</b></div>
    <div class="kv"><span>Confusion (TP/FP/TN/FN)</span><b>${b.tp}/${b.fp}/${b.tn}/${b.fn}</b></div>
  </div>`;
}

function reliabilityChart(before, after){
  if (!before?.bin_confidence?.length) return "";
  const W = 300, H = 220, P = 34;
  const x = v => P + v * (W - P - 8);
  const y = v => H - P - v * (H - P - 12);
  const maxCount = Math.max(...before.bin_count, 1);
  const line = (bins, colour) => {
    const kept = bins.bin_confidence
      .map((c, i) => ({c, a: bins.bin_accuracy[i], n: bins.bin_count[i]}))
      .filter(b => b.n > 0);
    if (kept.length < 1) return "";
    const pts = kept.map(b => `${x(b.c)},${y(b.a)}`);
    const poly = pts.length > 1
      ? `<polyline points="${pts.join(" ")}" fill="none" stroke="${colour}" stroke-width="1.6"
         stroke-linejoin="round" opacity=".75"/>` : "";
    return poly + kept.map(b =>
      `<circle cx="${x(b.c)}" cy="${y(b.a)}" r="${2 + 6 * Math.sqrt(b.n / maxCount)}"
        fill="${colour}" fill-opacity=".55" stroke="${colour}" stroke-width="1"/>`).join("");
  };
  return `<svg class="chart" viewBox="0 0 ${W} ${H}">
    <line class="gridline" x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}"
      stroke-dasharray="4 4" stroke="#3a4b5f"/>
    <line class="axis" x1="${P}" y1="${H-P}" x2="${W-8}" y2="${H-P}"/>
    <line class="axis" x1="${P}" y1="${H-P}" x2="${P}" y2="12"/>
    ${line(before, "#d95757")}
    ${line(after, "#5fd39a")}
    <text x="${P}" y="${H-P+14}">0</text>
    <text x="${x(1)-6}" y="${H-P+14}">1</text>
    <text x="${P-20}" y="${y(0)+4}">0</text>
    <text x="${P-20}" y="${y(1)+4}">1</text>
    <text x="${W/2-40}" y="${H-6}">predicted probability</text>
    <text transform="rotate(-90 12 ${H/2})" x="12" y="${H/2}">observed frequency</text>
  </svg>
  <div class="row" style="gap:16px;font-size:11.5px;margin-top:6px">
    <span style="color:#f08a8a">● uncalibrated</span>
    <span style="color:#5fd39a">● temperature-scaled</span>
    <span style="color:var(--muted)">◌ perfect calibration</span>
  </div>
  <p class="note" style="margin-top:8px">Marker area is proportional to the number of cases in
    the bin. On phantom data the bins concentrate at the extremes because the task is very nearly
    separable — which is a property of the data, not evidence of good calibration. ECE is the
    number to read.</p>`;
}

function renderValidation(host, v){
  const cal = v.calibration || {};
  const before = cal.before || {}, after = cal.after || {};
  const op = v.operating_point || {};
  const ab = v.ablation;
  const xq = v.explanation_quality;
  const t = v.internal_test, e = v.external;

  const arms = ab ? Object.entries(ab.arms || {}) : [];
  arms.sort((a, b) => (b[1].metrics?.auc || 0) - (a[1].metrics?.auc || 0));

  host.innerHTML = `
    <div class="banner info">
      <b>These figures measure that the pipeline is correctly wired, not clinical performance.</b>
      The grader is fitted on procedurally generated phantoms. Any clinical claim requires
      re-running the identical scripts on APTOS/IDRiD with Messidor-2 held out.
    </div>

    <div class="cols c4" style="margin-bottom:16px">
      <div class="stat ${t?.referable?.sensitivity?.value >= 0.9 ? 'good' : 'bad'}">
        <div class="lab">Sensitivity</div>
        <div class="big">${num(t?.referable?.sensitivity?.value)}</div>
        <div class="sub">internal test · target ≥ 0.90</div></div>
      <div class="stat ${t?.referable?.specificity?.value >= 0.8 ? 'good' : 'bad'}">
        <div class="lab">Specificity</div>
        <div class="big">${num(t?.referable?.specificity?.value)}</div>
        <div class="sub">internal test · target ≥ 0.80</div></div>
      <div class="stat accent"><div class="lab">AUC</div>
        <div class="big">${num(t?.referable?.auc, 4)}</div>
        <div class="sub">referable DR, DeLong CI</div></div>
      <div class="stat"><div class="lab">Calibration error</div>
        <div class="big">${num(after.ece, 4)}</div>
        <div class="sub">ECE after temperature scaling</div></div>
    </div>

    <div class="cols c2">
      ${splitCard("Internal held-out test", t, "Nothing was fitted on this split.")}
      ${splitCard("External validation (zero-shot)", e, "Domain-shifted split. Nothing fitted on it, not even the threshold.")}
    </div>

    <div class="cols c2">
      <div class="card">
        <h2>Calibration<span class="side">fitted on validation only</span></h2>
        <div class="cols c2" style="gap:10px;margin-bottom:12px">
          <div>
            <div class="kv"><span>Temperature T</span><b>${num(cal.temperature, 4)}</b></div>
            <div class="kv"><span>ECE</span><b>${num(before.ece,4)} → <span class="pass">${num(after.ece,4)}</span></b></div>
            <div class="kv"><span>MCE</span><b>${num(before.mce,4)} → <span class="pass">${num(after.mce,4)}</span></b></div>
            <div class="kv"><span>Brier</span><b>${num(before.brier,4)} → <span class="pass">${num(after.brier,4)}</span></b></div>
            <div class="kv"><span>NLL</span><b>${num(before.nll,4)} → ${num(after.nll,4)}</b></div>
          </div>
          <div>
            <div class="kv"><span>Referral threshold</span><b>${num(op.threshold, 4)}</b></div>
            <div class="kv"><span>Sensitivity at threshold</span><b>${num(op.sensitivity)}</b></div>
            <div class="kv"><span>Specificity at threshold</span><b>${num(op.specificity)}</b></div>
          </div>
        </div>
        ${reliabilityChart(before, after)}
        ${op.rationale ? `<p class="note" style="margin-top:12px">${esc(op.rationale)}</p>` : ""}
      </div>

      <div class="card">
        <h2>Ablation — does the integrated pipeline win?</h2>
        ${ab ? `
          <table>
            <tr><th>Arm</th><th class="n">AUC</th><th class="n">Sens</th><th class="n">Spec</th><th class="n">QWK</th></tr>
            ${arms.map(([name, a]) => {
              const m = a.metrics || {};
              const isRef = name === ab.reference;
              return `<tr>
                <td>${isRef ? "<b>" : ""}${esc(name)}${isRef ? " ★</b>" : ""}</td>
                <td class="n">${num(m.auc, 4)}</td>
                <td class="n">${num(m.sensitivity?.value)}</td>
                <td class="n">${num(m.specificity?.value)}</td>
                <td class="n">${num(m.qwk?.value, 4)}</td></tr>`;
            }).join("")}
          </table>
          <div class="verdict-quote" style="margin-top:13px">${esc(ab.verdict)}</div>
          ${(ab.comparisons || []).length ? `
            <div style="margin-top:12px">
              ${ab.comparisons.map(c => `<div class="kv">
                <span>${esc(ab.reference)} vs ${esc(c.arm)}</span>
                <b>DeLong p = ${num(c.auc_test?.p_value, 3)}
                  <span class="ci">${c.significant ? "significant" : "not significant"}</span></b>
              </div>`).join("")}
            </div>` : ""}
        ` : `<p class="note">Only one arm was available. Train the comparison arms with
             <code>scripts/train_grader.py --arm cnn_only</code> to populate this section.</p>`}
      </div>
    </div>

    ${xq ? `<div class="card">
      <h2>Explanation quality<span class="side">${int(xq.n_evaluated)} referable images</span></h2>
      <div class="cols c4">
        <div class="stat"><div class="lab">Pointing game</div><div class="big">${pct(xq.pointing_game, 1)}</div>
          <div class="sub">CAM peak lands on a lesion</div></div>
        <div class="stat"><div class="lab">Faithfulness</div><div class="big">${num(xq.faithfulness, 3)}</div>
          <div class="sub">insertion − deletion AUC</div></div>
        <div class="stat"><div class="lab">Sparsity (Gini)</div><div class="big">${num(xq.sparsity, 3)}</div>
          <div class="sub">a map that highlights everything explains nothing</div></div>
        <div class="stat"><div class="lab">Lesion attention</div>
          <div class="big">${xq.lesion_hit_rate != null ? pct(xq.lesion_hit_rate, 1) : "n/a"}</div>
          <div class="sub">CAM mass inside lesion masks</div></div>
      </div>
      <div class="verdict-quote" style="margin-top:14px">
        Insertion AUC ${num(xq.insertion_auc, 3)} − deletion AUC ${num(xq.deletion_auc, 3)}
        = faithfulness ${num(xq.faithfulness, 3)}.
        High faithfulness with a low pointing-game score is the signature of shortcut learning,
        and it is exactly what makes black-box DR models collapse on external data. Producing a
        heatmap is trivial; evidence that it means anything is the work.
      </div>
    </div>` : ""}

    <div class="card">
      <h2>Split discipline</h2>
      <table>
        <tr><th></th><th>Fitted on</th><th>Reported on</th></tr>
        <tr><td>Temperature, referral threshold</td><td class="pass">validation</td><td>—</td></tr>
        <tr><td>Internal performance</td><td>—</td><td class="pass">test</td></tr>
        <tr><td>Zero-shot generalisation</td><td>nothing</td><td class="pass">external</td></tr>
      </table>
      <p class="note" style="margin-top:11px">
        Enforced in code: <code>registry.assert_no_leakage</code> raises <code>SplitViolation</code>
        if a held-out sample reaches the training pool, and splits are subject-grouped by hash so
        fellow eyes never straddle a boundary.
      </p>
    </div>`;
}

async function loadProgramme(){
  const host = $("#view-programme");
  host.innerHTML = '<div class="card"><div class="empty"><span class="spin"></span>Loading simulation results…</div></div>';
  let scen = null, opt = null;
  try {
    scen = await api("/simulation");
  } catch(e){
    emptyState(host, "No simulation results on disk",
      `The district telemedicine model is a discrete-event simulation run by a script.
       Run it once, then reload this tab.`,
      "python scripts/run_simulation.py --scenarios");
    return;
  }
  try { opt = await api("/optimisation"); } catch(e){ opt = null; }
  renderProgramme(host, scen, opt);
}

function loadBar(label, value, colour){
  const v = (value == null || isNaN(value)) ? 0 : value;
  const w = Math.min(v, 1.6) / 1.6 * 100;
  const capX = (0.85 / 1.6) * 100;
  return `<div style="margin-bottom:13px">
    <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px">
      <span style="color:var(--muted)">${esc(label)}</span>
      <b style="font-variant-numeric:tabular-nums;color:${colour}">${pct(value)}</b></div>
    <div class="loadbar">
      <div class="fill" style="width:${w}%;background:${colour}"></div>
      <div class="cap" style="left:${capX}%"></div>
      <div class="caplab" style="left:calc(${capX}% + 7px);color:#fff;opacity:.7">85%</div>
    </div>
  </div>`;
}

function searchVerdict(opt){
  const all = Array.isArray(opt.all) ? opt.all : [];
  const unaided = c => Number(c.params?.review_time_min) > 1.0;
  const rejected = all.filter(c => c.screened_out);
  const rejUnaided = rejected.filter(unaided).length;
  const gridUnaided = all.filter(unaided).length;
  const feasible = all.filter(c => c.feasible);
  const feasUnaided = feasible.filter(unaided).length;
  const bestUnaided = unaided(opt.best || {});
  if (!all.length)
    return `The search evaluated ${int(opt.n_evaluated)} configurations and returned the cheapest
            one meeting every constraint.`;
  const allRejectedUnaided = rejected.length > 0 && rejUnaided === rejected.length;
  return `The grid deliberately spans <b>whether to use AI triage at all</b> — half its
    ${int(all.length)} configurations assume unaided reading, half assume a pre-annotated case.
    ${allRejectedUnaided
      ? `Every one of the ${int(rejected.length)} configurations rejected analytically as unstable
         before simulation was an unaided-review design, out of ${int(gridUnaided)} such designs
         in the grid.`
      : `${int(rejUnaided)} of the ${int(rejected.length)} configurations rejected analytically were
         unaided-review designs.`}
    ${feasUnaided} unaided designs were still feasible, but the cheapest plan overall uses
    ${bestUnaided ? "unaided reading" : "<b>AI-assisted review</b>"} and
    ${opt.best?.params?.edge_inference ? "<b>edge inference</b> rather than a central GPU"
      : "central GPU inference"} — an output of the optimisation, not a premise of it.`;
}

function renderProgramme(host, scen, opt){
  const names = Object.keys(scen);
  const base = scen.baseline_manual, ai = scen.ai_assisted;
  const baseLoad = base?.results?.utilisation?.reviewer;
  const aiLoad = ai?.results?.utilisation?.reviewer;

  const best = opt?.best;
  const bp = best?.params || {};
  const bc = best?.cost || {};
  const br = best?.results || {};

  host.innerHTML = `
    <div class="cols c4" style="margin-bottom:18px">
      <div class="stat bad"><div class="lab">Without AI triage</div>
        <div class="big">${pct(baseLoad, 1)}</div>
        <div class="sub">of available reading capacity — queue is unstable</div></div>
      <div class="stat good"><div class="lab">With AI triage</div>
        <div class="big">${pct(aiLoad, 1)}</div>
        <div class="sub">same demand, same staffing</div></div>
      <div class="stat accent"><div class="lab">Cheapest feasible plan</div>
        <div class="big">${opt ? inr(bc.total) : "—"}</div>
        <div class="sub">${opt ? "per year, all constraints met" : "run --optimise to populate"}</div></div>
      <div class="stat"><div class="lab">Cost per patient</div>
        <div class="big">${opt ? "₹ " + num(opt.cost_per_patient, 0) : "—"}</div>
        <div class="sub">screened per year</div></div>
    </div>

    <div class="card">
      <h2>Ophthalmologist load — the constraint the whole programme turns on
        <span class="side">red line = 85% planning ceiling</span></h2>
      <p class="note">Queueing systems degrade super-linearly. A design that runs a single
        ophthalmologist near 100% on paper fails the first week someone takes leave, so the
        optimiser rejects anything above 85%.</p>
      ${names.map(n => {
        const u = scen[n].results?.utilisation?.reviewer;
        const colour = u > 1 ? "#d95757" : u > 0.85 ? "#d99a2b" : "#31a56f";
        return loadBar(n.replace(/_/g," "), u, colour);
      }).join("")}
      <p class="note" style="margin-top:6px">
        Above 100% the arrival rate exceeds the service rate: the backlog grows without bound and
        no affordable staffing level meets the SLA. That is a stability result, not a slow queue.
      </p>
    </div>

    <div class="card">
      <h2>Deployment scenarios<span class="side">100,000 patients/year · 12 PHCs · 2 ophthalmologist FTE</span></h2>
      <div style="overflow-x:auto">
      <table>
        <tr><th>Scenario</th><th class="n">Throughput/yr</th><th>Bottleneck</th>
            <th class="n">Reviewer</th><th class="n">Urgent SLA</th><th class="n">Routine SLA</th>
            <th class="n">p90 turnaround</th><th class="n">Cost/yr</th><th>Feasible</th></tr>
        ${names.map(n => {
          const s = scen[n], r = s.results || {};
          return `<tr>
            <td><b>${esc(n.replace(/_/g," "))}</b>${(s.violations||[]).length
              ? `<div class="ci" style="margin-top:4px">${s.violations.map(esc).join("<br>")}</div>` : ""}</td>
            <td class="n">${int(r.throughput_per_year)}</td>
            <td>${esc(r.bottleneck || "—")}</td>
            <td class="n ${r.utilisation?.reviewer > 0.85 ? "fail" : "pass"}">${pct(r.utilisation?.reviewer)}</td>
            <td class="n">${pct(r.sla_urgent)}</td>
            <td class="n ${r.sla_routine < 0.9 ? "fail" : "pass"}">${pct(r.sla_routine)}</td>
            <td class="n">${num(r.turnaround_days?.p90, 2)} d</td>
            <td class="n">${inr(s.cost?.total)}</td>
            <td>${s.feasible ? '<span class="pill routine">yes</span>' : '<span class="pill urgent">no</span>'}</td>
          </tr>`;
        }).join("")}
      </table>
      </div>
    </div>

    ${opt ? `
    <div class="cols c2">
      <div class="card">
        <h2>Optimised programme configuration
          <span class="side">${int(opt.n_evaluated)} configurations searched</span></h2>
        <p class="note">Constrained cost minimisation over the deployment grid. Each configuration
          is run under multiple seeds and judged on its worst utilisation.</p>
        ${Object.entries(bp).map(([k,v]) =>
          `<div class="kv"><span>${esc(k.replace(/_/g," "))}</span><b>${esc(String(v))}</b></div>`).join("")}
        <div class="kv" style="margin-top:8px"><span>Screened per year</span><b>${int(br.throughput_per_year)}</b></div>
        <div class="kv"><span>Bottleneck</span><b>${esc(br.bottleneck || "—")}</b></div>
        <div class="kv"><span>Reviewer utilisation</span><b class="pass">${pct(br.utilisation?.reviewer)}</b></div>
        <div class="kv"><span>Urgent SLA</span><b class="pass">${pct(br.sla_urgent)}</b></div>
        <div class="kv"><span>Routine SLA</span><b class="pass">${pct(br.sla_routine)}</b></div>
      </div>

      <div class="card">
        <h2>Annualised cost<span class="side">INR, order-of-magnitude inputs</span></h2>
        ${Object.entries(bc).filter(([k]) => k !== "total").map(([k,v]) => {
          const share = bc.total ? v / bc.total : 0;
          return `<div class="bar"><span class="lb">${esc(k.replace(/_/g," "))}</span>
            <span class="tr"><span class="fl" style="width:${share*100}%"></span></span>
            <span class="vl" style="width:80px">${inr(v)}</span></div>`;
        }).join("")}
        <div class="kv" style="margin-top:11px;border-top:1px solid var(--line2);padding-top:9px">
          <span>Total</span><b>${inr(bc.total)}</b></div>
        <div class="kv"><span>Per patient screened</span><b>₹ ${num(opt.cost_per_patient, 1)}</b></div>
        <div class="verdict-quote" style="margin-top:14px">${searchVerdict(opt)}</div>
      </div>
    </div>` : `
    <div class="banner">
      <b>Optimiser results not on disk.</b> The cheapest-feasible-plan search has not been run.
      Generate it with <code>python scripts/run_simulation.py --optimise</code> and reload.
    </div>`}

    <div class="card">
      <h2>MATLAB / Simulink bridge</h2>
      <p class="note">The same <code>SimConfig</code> generates the SimEvents model, so there is one
        source of truth and two runtimes.</p>
      <table>
        <tr><td><code>dr_screening_params.m</code></td><td>every parameter as a MATLAB struct</td></tr>
        <tr><td><code>build_dr_screening_model.m</code></td><td>builds the SimEvents block diagram</td></tr>
        <tr><td><code>validate_against_simpy.m</code></td><td>runs the model and diffs it against the SimPy reference</td></tr>
      </table>
      <p class="note" style="margin-top:11px">Regenerate with
        <code>python scripts/run_simulation.py --export-matlab matlab/</code></p>
    </div>`;
}

async function loadAudit(){
  const host = $("#view-audit");
  host.innerHTML = '<div class="card"><div class="empty"><span class="spin"></span>Loading review log…</div></div>';
  let a;
  try { a = await api("/audit?limit=500"); }
  catch(e){ emptyState(host, "Audit log unavailable", esc(e.message)); return; }

  if (!a.n){
    emptyState(host, "No reviews recorded yet",
      `Grade a case on the <b>Screen</b> tab and confirm or correct it. Every human decision is
       appended to <code>outputs/audit/reviews.jsonl</code>, which is the substrate for drift
       monitoring: a screening model that is never told when it was wrong cannot be monitored.`);
    return;
  }
  const s = a.summary || {};
  host.innerHTML = `
    <div class="cols c4" style="margin-bottom:18px">
      <div class="stat accent"><div class="lab">Reviews logged</div><div class="big">${int(a.n)}</div>
        <div class="sub">appended to reviews.jsonl</div></div>
      <div class="stat ${s.exact_agreement >= 0.7 ? "good" : "warnv"}">
        <div class="lab">Exact agreement</div><div class="big">${pct(s.exact_agreement, 1)}</div>
        <div class="sub">human graders agree 60–75% of the time</div></div>
      <div class="stat good"><div class="lab">Within one grade</div>
        <div class="big">${pct(s.within_one_grade, 1)}</div>
        <div class="sub">the clinically meaningful figure</div></div>
      <div class="stat"><div class="lab">Median review time</div>
        <div class="big">${s.median_review_seconds != null ? num(s.median_review_seconds,1) + "s" : "—"}</div>
        <div class="sub">${s.under_30s_fraction != null ? pct(s.under_30s_fraction,0) + " under the 30s target" : ""}</div></div>
    </div>

    <div class="card">
      <h2>Review log<span class="side">most recent ${Math.min(a.reviews.length, 50)}</span></h2>
      <p class="note">Agreement over time, by grade and by site, is the earliest signal that a
        deployed model has started to fail on a new camera or a new population. Post-market
        surveillance is a regulatory requirement, not a nice-to-have.</p>
      <div style="overflow-x:auto">
      <table>
        <tr><th>Time</th><th>Image</th><th class="n">Model</th><th class="n">Reviewer</th>
            <th>Agreement</th><th class="n">Seconds</th></tr>
        ${a.reviews.slice().reverse().map(r => `<tr>
          <td class="mono">${esc((r.timestamp || "").replace("T"," ").slice(0,19))}</td>
          <td class="mono">${esc(r.image_id)}</td>
          <td class="n">${r.model_grade}</td>
          <td class="n">${r.reviewer_grade}</td>
          <td class="${r.agreement === "exact" ? "pass" : r.agreement === "within_one" ? "borderline" : "fail"}">
            ${esc(r.agreement.replace(/_/g," "))}</td>
          <td class="n">${num(r.review_seconds, 1)}</td></tr>`).join("")}
      </table>
      </div>
    </div>`;
}

$$("nav.tabs button").forEach(b => b.addEventListener("click", () => showTab(b.dataset.tab)));
loadHealth();
showTab("screen");
