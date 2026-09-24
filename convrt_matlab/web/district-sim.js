/* The district screening model as a discrete-event simulation.

   A block-for-block port of `Simulink Model/district_model.slx` (SimEvents):

     Arrivals → MergeRetakes → WaitingRoom → AcqTechnician → AcqCamera →
     CaptureStation → RelTechnician → RelCamera → QualityGate
         ├─ retake → MergeRetakes
         └─ gradeable → AcqDevice → GradeOnDevice → RelDevice → RouteDecision
                ├─ cleared → ClearedOnSpot
                └─ flagged → UploadBuffer → Upload → ReviewQueue →
                   AcqOphthalmologist → SpecialistReview → RelOphthalmologist → Reviewed

   `asBuilt: true` reproduces the .slx exactly — one of each resource, servers
   of capacity 1, fixed service times, FIFO queues, one session — and is what
   is compared against MATLAB's own replications. The other settings are the
   improvements (see IMPROVED below). Time base: minutes, as in the model.

   Runs in a Web Worker (postMessage a config, receive results) or directly
   via DistrictSim.run(config). */
(function (root) {
  "use strict";

  // ── the model's own parameters (district_model_params.m) ───────────────────
  const AS_BUILT = {
    asBuilt: true, phcs: 1, days: 1, sessionHours: 6, reviewHours: 6,
    technicians: 1, cameras: 1, devices: 1, ophthalmologists: 1, uplinks: 1,
    patientsPerDay: (100000 / (12 * 250)) * (6 / 7),       // one arrival every 12.6 min
    prevalence: 0.18, urgentShare: 0.25,
    pPass: 0.80, pRepair: 0.08, maxAttempts: 3,
    sensitivity: 0.986, specificity: 0.873, ai: true,
    captureMin: 5, gradeMin: 0.25, uploadMin: 0.67, reviewMin: 0.5, reviewMinUnaided: 2.5,
    stochastic: false, captureCv: 0.35, reviewCv: 0.4, uploadCv: 0.55,
    priority: false, finishQueue: false, reps: 20, seed: 1
  };
  // what the improved model defaults to: a district of 12 health centres
  // sharing a review hub, over working days rather than one session
  const IMPROVED = Object.assign({}, AS_BUILT, {
    asBuilt: false, phcs: 12, days: 20, sessionHours: 7, reviewHours: 5,
    ophthalmologists: 2, patientsPerDay: 100000 / (12 * 250),
    captureMin: 4, stochastic: true, priority: true, finishQueue: true, reps: 10
  });

  // ── machinery ──────────────────────────────────────────────────────────────
  function mulberry32(seed) {
    let a = (seed >>> 0) || 1;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  // events by time, ties in scheduling order
  function Agenda() {
    const h = []; let seq = 0;
    const less = (a, b) => a.t < b.t || (a.t === b.t && a.s < b.s);
    return {
      get size() { return h.length; },
      push(t, fn) {
        const n = {t, s: seq++, fn}; h.push(n);
        let i = h.length - 1;
        while (i > 0) { const p = (i - 1) >> 1; if (!less(h[i], h[p])) break; [h[i], h[p]] = [h[p], h[i]]; i = p; }
      },
      pop() {
        const top = h[0], last = h.pop();
        if (h.length) {
          h[0] = last; let i = 0;
          for (;;) {
            const l = 2 * i + 1, r = l + 1; let m = i;
            if (l < h.length && less(h[l], h[m])) m = l;
            if (r < h.length && less(h[r], h[m])) m = r;
            if (m === i) break; [h[i], h[m]] = [h[m], h[i]]; i = m;
          }
        }
        return top;
      }
    };
  }
  // a pool of identical units (a resource, or a server's capacity). Waiting
  // entities queue FIFO, or urgent-first when `priority` is set. `area`
  // accumulates units-in-use × time, `open` units-on-shift × time.
  function Pool(clock, amount, priority) {
    const P = {amount, onShift: amount, used: 0, waiters: [], area: 0, open: 0, qArea: 0, last: 0, priority, evT: 0, evArea: 0};
    P.mark = function () {
      const t = clock.t, dt = t - P.last;
      if (dt > 0) { P.area += P.used * dt; P.open += Math.min(P.onShift, P.amount) * dt; P.qArea += P.waiters.length * dt; P.last = t; }
    };
    // SimEvents updates a block's statistics only when an entity enters or
    // leaves it, so the value it reports at the stop time is the average up
    // to that block's last event; evT / evArea keep the same reading.
    const stamp = () => { P.evT = clock.t; P.evArea = P.area; };
    P.acquire = function (e, next) {
      P.mark(); stamp();
      if (P.used < P.onShift && !P.waiters.length) { P.used++; next(e); return; }
      if (P.priority && e.urgent) {                      // behind other urgent cases, ahead of routine ones
        let i = 0; while (i < P.waiters.length && P.waiters[i][0].urgent) i++;
        P.waiters.splice(i, 0, [e, next]);
      } else P.waiters.push([e, next]);
    };
    P.release = function () {
      P.mark(); stamp(); P.used--;
      P.dispatch();
    };
    P.dispatch = function () {
      while (P.used < P.onShift && P.waiters.length) {
        const [e, next] = P.waiters.shift(); P.used++; next(e);
      }
    };
    P.setShift = function (n) { P.mark(); P.onShift = n; P.dispatch(); };
    return P;
  }
  function quantile(sorted, q) {
    if (!sorted.length) return NaN;
    const i = (sorted.length - 1) * q, lo = Math.floor(i), hi = Math.ceil(i);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
  }

  // ── one replication ────────────────────────────────────────────────────────
  function replicate(c, seed) {
    const rand = mulberry32(seed);
    const randn = () => { let u = 0, v = 0; while (u === 0) u = rand(); v = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    // log-normal with a given mean and coefficient of variation; fixed when not stochastic
    const service = (mean, cv) => {
      if (!c.stochastic || !cv) return mean;
      const s2 = Math.log(1 + cv * cv);
      return Math.exp(Math.log(mean) - s2 / 2 + Math.sqrt(s2) * randn());
    };
    const clock = {t: 0}, agenda = Agenda();
    const at = (t, fn) => agenda.push(t, fn);
    const INF = Infinity;
    const DAY = 1440, SESSION = c.sessionHours * 60, REVIEW = c.reviewHours * 60;
    // (horizonMin stops the clock early, as a Simulink run's stop time does)
    const horizon = c.asBuilt ? SESSION : (c.horizonMin || c.days * DAY);

    // district review hub, shared by every health centre
    const oph = Pool(clock, c.ophthalmologists, c.priority);
    const reviewSrv = Pool(clock, c.asBuilt ? 1 : INF, false);
    // one health centre
    const phcs = [];
    for (let k = 0; k < c.phcs; k++) phcs.push({
      waiting: [], tech: Pool(clock, c.technicians), cam: Pool(clock, c.cameras),
      captureSrv: Pool(clock, c.asBuilt ? 1 : INF), dev: Pool(clock, c.devices),
      gradeSrv: Pool(clock, c.asBuilt ? 1 : INF), link: Pool(clock, c.uplinks)
    });

    const S = {arrived: 0, referred: 0, captured: 0, retakes: 0, repaired: 0, ungradable: 0, cleared: 0, uploaded: 0,
      reviewed: 0, missed: 0, falsePos: 0, referableCaught: 0, referable: 0,
      captureWait: [], toReview: [], urgentToReview: [], urgentReviewed: 0, urgentSla: 0, routineSla: 0, routineReviewed: 0};
    const series = [];                                   // [t, review queue, waiting rooms]

    // ── the flow, block by block ──
    function arrive(ph) {
      const e = {ph, tArrive: clock.t, attempts: 0, ungradable: false};
      e.referable = rand() < c.prevalence;
      e.urgent = e.referable && rand() < c.urgentShare;
      S.arrived++; if (e.referable) S.referable++;
      toWaitingRoom(e);
    }
    function toWaitingRoom(e) {                          // MergeRetakes → WaitingRoom → AcqTechnician
      e.tQueue = clock.t;
      e.ph.tech.acquire(e, e2 => e2.ph.cam.acquire(e2, e3 => e3.ph.captureSrv.acquire(e3, startCapture)));
    }
    function startCapture(e) {
      if (e.attempts === 0) S.captureWait.push(clock.t - e.tArrive);
      at(clock.t + service(c.captureMin, c.captureCv), () => endCapture(e));
    }
    function endCapture(e) {                             // CaptureStation service-complete action
      e.ph.captureSrv.release(); e.ph.tech.release(); e.ph.cam.release();
      S.captured++;
      const r = rand();
      let retake = false;
      if (r < c.pPass) { /* gradeable */ }
      else if (r < c.pPass + c.pRepair) S.repaired++;
      else {
        e.attempts++;
        if (e.attempts < c.maxAttempts) retake = true;
        else { e.ungradable = true; S.ungradable++; }
      }
      if (retake) { S.retakes++; toWaitingRoom(e); return; }    // QualityGate port 2
      if (!c.ai) { flag(e, true); return; }                      // manual review: everyone gradeable is read by a person
      e.ph.dev.acquire(e, e2 => e2.ph.gradeSrv.acquire(e2, x => at(clock.t + c.gradeMin, () => endGrade(x))));
    }
    function endGrade(e) {                               // GradeOnDevice service-complete action
      e.ph.gradeSrv.release(); e.ph.dev.release();
      const flagged = e.ungradable || (e.referable ? rand() < c.sensitivity : rand() > c.specificity);
      flag(e, flagged);
    }
    function flag(e, flagged) {                          // RouteDecision
      e.tFlag = clock.t;
      if (e.referable && flagged) S.referableCaught++;
      if (!flagged) { S.cleared++; if (e.referable) S.missed++; return; }
      S.referred++;
      if (!e.referable) S.falsePos++;
      e.ph.link.acquire(e, e2 => at(clock.t + service(c.uploadMin, c.uploadCv), () => endUpload(e2)));
    }
    function endUpload(e) {                              // Upload → ReviewQueue → AcqOphthalmologist
      e.ph.link.release(); S.uploaded++;
      oph.acquire(e, e2 => reviewSrv.acquire(e2, x => {
        const mean = c.ai ? c.reviewMin : c.reviewMinUnaided;
        at(clock.t + service(mean, c.reviewCv), () => endReview(x));
      }));
    }
    function endReview(e) {
      reviewSrv.release(); oph.release();
      S.reviewed++;
      const w = clock.t - e.tFlag; S.toReview.push(w);
      if (e.urgent) { S.urgentReviewed++; S.urgentToReview.push(w); if (w <= 2 * DAY) S.urgentSla++; }
      else { S.routineReviewed++; if (w <= 14 * DAY) S.routineSla++; }
    }

    // ── arrivals and shifts ──
    const perMin = c.patientsPerDay / SESSION;           // per health centre, during the session
    const meanGap = 1 / perMin;
    for (let d = 0; d < (c.asBuilt ? 1 : c.days); d++) {
      const open = d * DAY, close = open + SESSION;
      for (const ph of phcs) {
        // the model generates its first patient at the start of the session
        for (let t = open; t < close && t < horizon; t += -meanGap * Math.log(1 - rand())) at(t, () => arrive(ph));
      }
      if (!c.asBuilt) {
        // reviewers work their hours each day; whoever is mid-review finishes it
        oph.onShift = 0;
        at(open, () => oph.setShift(c.ophthalmologists));
        at(open + REVIEW, () => oph.setShift(0));
      }
    }
    if (!c.asBuilt) oph.onShift = 0;
    // sample the queues every simulated hour (every 5 min for one session)
    const step = c.asBuilt ? 5 : 60;
    for (let t = 0; t <= horizon; t += step) at(t, () => series.push([clock.t,
      oph.waiters.length, phcs.reduce((s, p) => s + p.tech.waiters.length + p.cam.waiters.length, 0)]));

    // ── run ──
    let events = 0;
    while (agenda.size) {
      const ev = agenda.pop();
      if (ev.t > horizon && (c.asBuilt || !c.finishQueue)) break;
      if (ev.t > horizon + 60 * DAY) break;             // a safety stop for an overloaded district
      clock.t = ev.t; ev.fn(); events++;
    }
    const T = c.asBuilt || !c.finishQueue ? horizon : Math.max(horizon, clock.t);
    const all = [oph, reviewSrv];
    phcs.forEach(p => all.push(p.tech, p.cam, p.captureSrv, p.dev, p.gradeSrv, p.link));
    all.forEach(p => { clock.t = T; p.mark(); });

    // utilisation: units in use over units available. For one session that
    // is the model's own statistic (busy time over the run); over days it is
    // measured against the scheduled hours, so overtime shows as > 100%.
    const sched = c.asBuilt ? T : c.days * SESSION;
    const util = (pools, units) => pools.reduce((s, p) => s + p.area, 0) / (units * sched);
    const res = {
      arrived: S.arrived, captured: S.captured, retakes: S.retakes, repaired: S.repaired, ungradable: S.ungradable,
      cleared: S.cleared, referred: S.referred, uploaded: S.uploaded, reviewed: S.reviewed,
      backlog: oph.waiters.length,
      missed: S.missed, falsePositives: S.falsePos,
      referableCaught: S.referable ? S.referableCaught / S.referable : NaN,
      util_capture: c.asBuilt ? phcs[0].captureSrv.area / T : util(phcs.map(p => p.cam), c.phcs * c.cameras),
      util_technician: util(phcs.map(p => p.tech), c.phcs * c.technicians),
      util_camera: util(phcs.map(p => p.cam), c.phcs * c.cameras),
      util_device: c.asBuilt ? phcs[0].gradeSrv.area / T : util(phcs.map(p => p.dev), c.phcs * c.devices),
      util_review: c.asBuilt ? reviewSrv.area / T
        : (oph.open > 0 ? oph.area / oph.open : 0),
      queue_review: oph.qArea / T,
      // the same utilisations as SimEvents reports them (see Pool), for the check against MATLAB
      se_capture: phcs[0].captureSrv.evT ? phcs[0].captureSrv.evArea / ((c.asBuilt ? 1 : Math.min(c.technicians, c.cameras)) * phcs[0].captureSrv.evT) : 0,
      se_device: phcs[0].gradeSrv.evT ? phcs[0].gradeSrv.evArea / ((c.asBuilt ? 1 : c.devices) * phcs[0].gradeSrv.evT) : 0,
      se_review: reviewSrv.evT ? reviewSrv.evArea / ((c.asBuilt ? 1 : c.ophthalmologists) * reviewSrv.evT) : 0,
      captureWaitMean: mean(S.captureWait), captureWaitP90: quantile(S.captureWait.sort((a, b) => a - b), 0.9),
      toReviewMedian: quantile(S.toReview.sort((a, b) => a - b), 0.5),
      toReviewP90: quantile(S.toReview, 0.9),
      urgentSla: S.urgentReviewed ? S.urgentSla / S.urgentReviewed : NaN,
      routineSla: S.routineReviewed ? S.routineSla / S.routineReviewed : NaN,
      events, series
    };
    return res;
  }
  function mean(a) { let s = 0; for (const x of a) s += x; return a.length ? s / a.length : NaN; }

  // ── replications, summarised ───────────────────────────────────────────────
  function run(config) {
    const c = Object.assign({}, config.asBuilt === false ? IMPROVED : AS_BUILT, config);
    const t0 = (typeof performance !== "undefined" ? performance : Date).now();
    const reps = [];
    for (let r = 0; r < c.reps; r++) reps.push(replicate(c, (c.seed * 7919 + r * 104729) >>> 0));
    const keys = Object.keys(reps[0]).filter(k => typeof reps[0][k] === "number");
    const summary = {};
    for (const k of keys) {
      const v = reps.map(x => x[k]).filter(Number.isFinite).sort((a, b) => a - b);
      summary[k] = {mean: mean(v), lo: quantile(v, 0.05), hi: quantile(v, 0.95), sd: Math.sqrt(mean(v.map(x => (x - mean(v)) ** 2)))};
    }
    // the queue series, averaged across replications
    const n = Math.min(...reps.map(x => x.series.length));
    const series = [];
    for (let i = 0; i < n; i++) series.push([reps[0].series[i][0],
      mean(reps.map(x => x.series[i][1])), mean(reps.map(x => x.series[i][2]))]);
    const ms = (typeof performance !== "undefined" ? performance : Date).now() - t0;
    return {config: c, summary, series, reps: reps.length, ms, runs: reps.map(x => { const o = Object.assign({}, x); delete o.series; return o; })};
  }

  const api = {run, AS_BUILT, IMPROVED};
  root.DistrictSim = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  // as a worker: a config in, results out
  if (typeof WorkerGlobalScope !== "undefined" && root instanceof WorkerGlobalScope) {
    root.onmessage = ev => { try { root.postMessage({id: ev.data.id, ok: true, result: run(ev.data.config)}); }
      catch (err) { root.postMessage({id: ev.data.id, ok: false, error: String(err && err.message || err)}); } };
  }
})(typeof self !== "undefined" ? self : globalThis);
