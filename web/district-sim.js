/* The district screening model as a discrete-event simulation.

   A block-for-block port of `simulink/district_model.slx` (SimEvents),
   built by build_district_model.m from district_model_params.m:

     Arrivals → [MergeRetakes] → WaitingRoom → AcqTechnician → AcqCamera →
     CaptureStation → RelTechnician → RelCamera → AcqDevice → QualityCheck →
     QualitySwitch
       ├─ good ──────────────────────────────→ GradeMerge → GradeOnDevice ─┐
       ├─ borderline → Restoration → RestoreSwitch ─ restored ↗             │
       │                                           └ not restored ─┐        │
       └─ bad ──────────────────────────────────────────────→ DeviceMerge ←┘
     → RelDevice → RouteDecision
       ├─ cleared → ClearedOnSpot
       ├─ refer   → UploadBuffer → Upload → ReviewQueue → AcqOphthalmologist →
       │            SpecialistReview → RelOphthalmologist → Reviewed
       └─ recapture (include_gate = 1) → MergeRetakes

   A bad image, or a borderline one the restoration cannot save, goes back
   for another capture when the gate is in (up to maxAttempts captures), and
   otherwise to the doctor ungraded.

   `asBuilt: true` is the .slx exactly — one health centre, one session,
   fixed service times, first-come queues — and is what is compared against
   MATLAB's replications. With `asBuilt: false` the same blocks run a
   district: several centres sharing one review hub, working hours over many
   days, and optionally variable service times, urgent-first review and a
   no-AI baseline. Time base: minutes, as in the model.

   Runs in a Web Worker (postMessage a config, receive results) or directly
   via DistrictSim.run(config). */
(function (root) {
  "use strict";

  // ── the model's own variables (district_model_params.m) ───────────────────
  const AS_BUILT = {
    asBuilt: true, phcs: 1, days: 1, sessionHours: 6, reviewHours: 6,        // sim_stop = 360 min
    technicians: 5, cameras: 5, devices: 5, ophthalmologists: 5,
    // arrivals: group events at λ(t) = λ_peak·exp(−(t − t_peak)²/(2σ²))·(N − A(t))/N,
    // each bringing 1 + Poisson(μ) patients; λ_peak is set from meanGap so
    // the session brings a patient every meanGap minutes on average
    meanGap: 3, tPeak: 150, sigma: 90, N: 200, mu: 0.5,
    prevalence: 0.08, urgentShare: 0.25,
    pGood: 0.70, pBorder: 0.15, pBad: 0.15, pRestore: 0.873,
    includeGate: false, maxAttempts: 3,
    sensitivity: 0.986, specificity: 0.873, ai: true,
    captureMin: 5, qualityMin: 0.01, restoreMin: 0.02, gradeMin: 0.25, uploadMin: 0.67,
    reviewMin: 0.5, reviewMinUnaided: 2.5,
    stochastic: false, captureCv: 0.35, reviewCv: 0.4, uploadCv: 0.55,
    priority: false, finishQueue: false, reps: 20, seed: 1
  };
  // the district: 12 health centres sharing a review hub, over working days
  const IMPROVED = Object.assign({}, AS_BUILT, {
    asBuilt: false, phcs: 12, days: 20, sessionHours: 7, reviewHours: 5,
    technicians: 1, cameras: 1, devices: 1, ophthalmologists: 2,
    // ~33.3 patients per centre per 7-hour day (100,000 a year, 12 centres, 250 days)
    meanGap: 12.6, tPeak: 150, sigma: 90, N: 46,
    stochastic: true, priority: true, finishQueue: true, reps: 10
  });
  const defaults = config => Object.assign({}, config.asBuilt === false ? IMPROVED : AS_BUILT, config);

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
  // a resource pool of identical units. Waiting entities queue FIFO, or
  // urgent-first when `priority` is set. `area` accumulates units-in-use ×
  // time, `open` units-on-shift × time.
  function Pool(clock, amount, priority) {
    const P = {amount, onShift: amount, used: 0, waiters: [], area: 0, open: 0, qArea: 0, last: 0, priority, evT: 0, evArea: 0};
    P.mark = function () {
      const t = clock.t, dt = t - P.last;
      if (dt > 0) { P.area += P.used * dt; P.open += Math.min(P.onShift, P.amount) * dt; P.qArea += P.waiters.length * dt; P.last = t; }
    };
    // SimEvents updates a pool's utilisation statistic when units are taken
    // or given back, so the value it reports at the stop time is the average
    // up to the pool's last such event; evT / evArea keep the same reading.
    const stamp = () => { P.evT = clock.t; P.evArea = P.area; };
    P.acquire = function (e, next) {
      P.mark();
      if (P.used < P.onShift && !P.waiters.length) { stamp(); P.used++; next(e); return; }
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

  // ── the arrival process ────────────────────────────────────────────────────
  // With depletion, dE[A]/dt = (1 + μ)·λ_peak·g(t)·(N − E[A])/N, so over a
  // session of T minutes E[A] = N·(1 − exp(−(1 + μ)·λ_peak·G(T)/N)), G the
  // integral of the Gaussian bump. Setting E[A] = T / meanGap gives λ_peak —
  // the same calculation as district_peak_rate.m.
  function erf(x) {                                       // Abramowitz & Stegun 7.1.26
    const s = x < 0 ? -1 : 1; x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x));
  }
  const Phi = x => 0.5 * (1 + erf(x / Math.SQRT2));
  const gauss = (c, t) => c.sigma * Math.sqrt(2 * Math.PI) * (Phi((t - c.tPeak) / c.sigma) - Phi(-c.tPeak / c.sigma));
  function peakRate(config) {
    const c = defaults(config), T = c.sessionHours * 60;
    const target = Math.min(T / c.meanGap / c.N, 0.999);
    return -c.N * Math.log(1 - target) / ((1 + c.mu) * gauss(c, T));
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
    const DAY = 1440, SESSION = c.sessionHours * 60, REVIEW = c.reviewHours * 60;
    // (horizonMin stops the clock early, as a Simulink run's stop time does)
    const horizon = c.asBuilt ? SESSION : (c.horizonMin || c.days * DAY);
    const lam = peakRate(c);

    // the review hub, shared by every health centre
    const oph = Pool(clock, c.ophthalmologists, c.priority);
    // the health centres
    const phcs = [];
    for (let k = 0; k < c.phcs; k++) phcs.push({
      tech: Pool(clock, c.technicians), cam: Pool(clock, c.cameras), dev: Pool(clock, c.devices), link: Pool(clock, 1)
    });

    const S = {arrived: 0, captured: 0, retakes: 0, good: 0, restorations: 0, restored: 0, bad: 0, ungradable: 0,
      cleared: 0, referred: 0, uploaded: 0, reviewed: 0, missed: 0, falsePos: 0, referableCaught: 0, referable: 0,
      captureWait: [], toReview: [], devWait: [], urgentReviewed: 0, urgentSla: 0, routineSla: 0, routineReviewed: 0};
    const patients = [], referrals = [];                 // for waits that are still running at the end
    const series = [];                                   // [t, review queue, waiting rooms]

    // ── the flow, block by block ──
    function arrive(ph) {                                // Arrivals (GenerateAction)
      const e = {ph, tArrive: clock.t, attempts: 0};
      e.referable = rand() < c.prevalence;
      e.urgent = e.referable && rand() < c.urgentShare;
      S.arrived++; if (e.referable) S.referable++;
      patients.push(e);
      toWaitingRoom(e);
    }
    function toWaitingRoom(e) {                          // [MergeRetakes →] WaitingRoom → AcqTechnician → AcqCamera
      e.ph.tech.acquire(e, e2 => e2.ph.cam.acquire(e2, startCapture));
    }
    function startCapture(e) {                           // CaptureStation
      if (e.attempts === 0) { e.tStart = clock.t; S.captureWait.push(clock.t - e.tArrive); }
      at(clock.t + service(c.captureMin, c.captureCv), () => endCapture(e));
    }
    function endCapture(e) {                             // → RelTechnician → RelCamera → AcqDevice
      e.ph.tech.release(); e.ph.cam.release();
      e.attempts++; S.captured++;
      if (!c.ai) { route(e, true); return; }             // no AI: every image goes to a doctor
      e.tDevReq = clock.t;
      e.ph.dev.acquire(e, x => { S.devWait.push(clock.t - x.tDevReq); at(clock.t + c.qualityMin, () => qualityCheck(x)); });
    }
    function qualityCheck(e) {                           // QualityCheck → QualitySwitch
      const r = rand() * (c.pGood + c.pBorder + c.pBad);
      if (r < c.pGood) { S.good++; grade(e); }
      else if (r < c.pGood + c.pBorder) {                // Restoration → RestoreSwitch
        S.restorations++;
        at(clock.t + c.restoreMin, () => {
          if (rand() < c.pRestore) { S.restored++; grade(e); } else badImage(e);
        });
      } else { S.bad++; badImage(e); }
    }
    function grade(e) {                                  // GradeMerge → GradeOnDevice → DeviceMerge → RelDevice
      at(clock.t + c.gradeMin, () => {
        e.ph.dev.release();
        route(e, e.referable ? rand() < c.sensitivity : rand() > c.specificity);
      });
    }
    function badImage(e) {                               // DeviceMerge → RelDevice → RouteDecision
      e.ph.dev.release();
      if (c.includeGate && e.attempts < c.maxAttempts) { S.retakes++; toWaitingRoom(e); return; }   // recapture
      S.ungradable++; e.ungradable = true;
      route(e, true);                                    // to the doctor ungraded
    }
    function route(e, flagged) {                         // RouteDecision
      e.tFlag = clock.t;
      if (e.referable && flagged) S.referableCaught++;
      if (!flagged) { S.cleared++; if (e.referable) S.missed++; return; }   // ClearedOnSpot
      S.referred++; referrals.push(e);
      if (!e.referable) S.falsePos++;
      e.ph.link.acquire(e, e2 => at(clock.t + service(c.uploadMin, c.uploadCv), () => endUpload(e2)));   // UploadBuffer → Upload
    }
    function endUpload(e) {                              // → ReviewQueue → AcqOphthalmologist → SpecialistReview
      e.ph.link.release(); S.uploaded++;
      oph.acquire(e, x => {
        const mean = c.ai ? c.reviewMin : c.reviewMinUnaided;     // (the model reads every referral in t_review)
        at(clock.t + service(mean, c.reviewCv), () => endReview(x));
      });
    }
    function endReview(e) {                              // → RelOphthalmologist → Reviewed
      oph.release();
      S.reviewed++; e.tDone = clock.t;
      const w = clock.t - e.tFlag; S.toReview.push(w);
      if (e.urgent) { S.urgentReviewed++; if (w <= 2 * DAY) S.urgentSla++; }
      else { S.routineReviewed++; if (w <= 14 * DAY) S.routineSla++; }
    }

    // ── arrivals and shifts ──
    // Thinning: candidates at λ_peak, each kept with probability λ(t)/λ_peak.
    // A kept event brings 1 + Poisson(μ) patients at once, never more than
    // the catchment has left — the same steps as the model's Arrivals block.
    const poisson = mu => { const L = Math.exp(-mu); let k = 0, pr = rand(); while (pr > L) { k++; pr *= rand(); } return k; };
    for (let d = 0; d < (c.asBuilt ? 1 : c.days); d++) {
      const open = d * DAY;
      for (const ph of phcs) {
        let tc = 0, A = 0;                                           // minutes since opening; patients so far
        while (A < c.N && tc < SESSION) {
          tc += -Math.log(1 - rand()) / lam;
          if (tc >= SESSION || open + tc >= horizon) break;
          const rate = lam * Math.exp(-((tc - c.tPeak) ** 2) / (2 * c.sigma * c.sigma)) * (c.N - A) / c.N;
          if (rand() * lam <= rate) {
            const g = 1 + Math.min(poisson(c.mu), c.N - A - 1);
            for (let k = 0; k < g; k++) at(open + tc, () => arrive(ph));
            A += g;
          }
        }
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
    const all = [oph];
    phcs.forEach(p => all.push(p.tech, p.cam, p.dev, p.link));
    all.forEach(p => { clock.t = T; p.mark(); });

    // utilisation: units in use over units available. For one session that
    // is busy time over the run; over days it is measured against the
    // scheduled hours, so overtime shows as > 100%.
    const sched = c.asBuilt ? T : c.days * SESSION;
    const util = (pools, units) => pools.reduce((s, p) => s + p.area, 0) / (units * sched);
    const se = P => (P.evT ? P.evArea / (P.amount * P.evT) : 0);
    // waits counted to the end for whoever is still waiting then, so a team
    // too small to finish the session shows as long waits, not short ones
    const waitAll = patients.map(e => (e.tStart !== undefined ? e.tStart : T) - e.tArrive);
    const readAll = referrals.map(e => (e.tDone !== undefined ? e.tDone : T) - e.tFlag).sort((a, b) => a - b);
    return {
      arrived: S.arrived, captured: S.captured, retakes: S.retakes, good: S.good, restorations: S.restorations,
      restored: S.restored, bad: S.bad, ungradable: S.ungradable,
      cleared: S.cleared, referred: S.referred, uploaded: S.uploaded, reviewed: S.reviewed,
      backlog: oph.waiters.length,
      missed: S.missed, falsePositives: S.falsePos,
      referableCaught: S.referable ? S.referableCaught / S.referable : NaN,
      util_technician: util(phcs.map(p => p.tech), c.phcs * c.technicians),
      util_camera: util(phcs.map(p => p.cam), c.phcs * c.cameras),
      util_device: util(phcs.map(p => p.dev), c.phcs * c.devices),
      util_review: c.asBuilt ? oph.area / (c.ophthalmologists * T) : (oph.open > 0 ? oph.area / oph.open : 0),
      queue_review: oph.qArea / T,
      // the pools' utilisation as SimEvents reports it (see Pool), for the check against MATLAB
      se_technician: se(phcs[0].tech), se_device: se(phcs[0].dev), se_review: se(oph),
      captureWaitMean: mean(S.captureWait), captureWaitP90: quantile(S.captureWait.sort((a, b) => a - b), 0.9),
      captureWaitAll: mean(waitAll), unscreened: patients.filter(e => e.tStart === undefined).length,
      deviceWaitMean: S.devWait.length ? mean(S.devWait) : 0,
      toReviewP90All: readAll.length ? quantile(readAll, 0.9) : 0,
      toReviewMedian: quantile(S.toReview.sort((a, b) => a - b), 0.5),
      toReviewP90: quantile(S.toReview, 0.9),
      urgentSla: S.urgentReviewed ? S.urgentSla / S.urgentReviewed : NaN,
      routineSla: S.routineReviewed ? S.routineSla / S.routineReviewed : NaN,
      events, series
    };
  }
  function mean(a) { let s = 0; for (const x of a) s += x; return a.length ? s / a.length : NaN; }

  // ── replications, summarised ───────────────────────────────────────────────
  function run(config) {
    const c = defaults(config);
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
    return {config: c, lambdaPeak: peakRate(c), summary, series, reps: reps.length, ms,
      runs: reps.map(x => { const o = Object.assign({}, x); delete o.series; return o; })};
  }

  // what the arrival process implies: patients expected in a session
  function expected(config) {
    const c = defaults(config), T = c.sessionHours * 60;
    return c.N * (1 - Math.exp(-(1 + c.mu) * peakRate(c) * gauss(c, T) / c.N));
  }
  // expected patients per hour through the session, n points
  function arrivalCurve(config, n = 60) {
    const c = defaults(config), T = c.sessionHours * 60, lam = peakRate(c), out = [];
    for (let i = 0; i <= n; i++) {
      const t = T * i / n, A = c.N * (1 - Math.exp(-(1 + c.mu) * lam * gauss(c, t) / c.N));
      out.push([t, 60 * (1 + c.mu) * lam * Math.exp(-((t - c.tPeak) ** 2) / (2 * c.sigma * c.sigma)) * (c.N - A) / c.N]);
    }
    return out;
  }

  // ── the resource planner ───────────────────────────────────────────────────
  // For each resource in turn, the smallest number that meets its target,
  // the others held where they are: capture teams (a technician and a
  // camera) against the wait before capture, edge devices against the wait
  // for the AI, doctors against the time to read 90% of referrals. Each
  // step uses the numbers the step before settled on.
  function plan(config, tg) {
    const base = defaults(config), one = base.asBuilt;
    const quick = Object.assign({}, base, {reps: one ? 30 : 3, days: one ? 1 : Math.min(base.days, 10)});
    const sweep = (range, over, score) => range.map(n => { const s = run(Object.assign({}, quick, over(n))).summary; return Object.assign({n}, score(s)); });
    const first = pts => { const p = pts.find(q => q.ok); return p ? p.n : null; };
    const range = n => Array.from({length: n}, (_, i) => i + 1);
    const out = {targets: tg, reps: quick.reps, days: quick.days};
    out.teams = sweep(range(one ? 10 : 6), n => ({technicians: n, cameras: n}), s => {
      const wait = s.captureWaitAll.mean, left = s.unscreened.mean;
      return {v: wait, left, ok: wait <= tg.waitMin && left < 0.5};
    });
    const teams = first(out.teams) || out.teams.length;
    out.devices = sweep(range(one ? 5 : 4), n => ({technicians: teams, cameras: teams, devices: n}), s => {
      const v = s.deviceWaitMean.mean; return {v, ok: v <= tg.deviceMin};
    });
    const devices = first(out.devices) || out.devices.length;
    out.doctors = sweep(range(one ? 6 : 12), n => ({technicians: teams, cameras: teams, devices, ophthalmologists: n}), s => {
      const v = s.toReviewP90All.mean; return {v, backlog: s.backlog.mean, ok: v <= tg.readMin};
    });
    out.need = {teams: first(out.teams), devices: first(out.devices), doctors: first(out.doctors)};
    return out;
  }

  const api = {run, plan, expected, arrivalCurve, peakRate, AS_BUILT, IMPROVED};
  root.DistrictSim = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  // as a worker: a config in, results out
  if (typeof WorkerGlobalScope !== "undefined" && root instanceof WorkerGlobalScope) {
    root.onmessage = ev => { try { root.postMessage({id: ev.data.id, ok: true,
        result: ev.data.kind === "plan" ? plan(ev.data.config, ev.data.targets) : run(ev.data.config)}); }
      catch (err) { root.postMessage({id: ev.data.id, ok: false, error: String(err && err.message || err)}); } };
  }
})(typeof self !== "undefined" ? self : globalThis);
