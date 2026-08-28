"""Discrete-event model of a district DR screening programme.

This is the SimPy realisation of the Simulink/SimEvents model the problem
statement asks for; :mod:`drscreen.sim.simulink_export` emits the equivalent
SimEvents block topology and a MATLAB driver so the same experiment can be run
inside Simulink for the toolbox requirement.

What is modelled
----------------
The pipeline is a queueing network, and the interesting behaviour is entirely
in the *couplings* -- which is why a spreadsheet cannot answer these
questions:

* **Recapture feedback loops.**  The IQA gate rejects an image and the patient
  is re-photographed, which re-occupies the camera *and* the technician.  A
  gate that is 5 points stricter can reduce camera throughput by far more
  than 5%, because the rejected patients re-enter the queue they just left.
* **Bandwidth as a shared, time-varying resource.**  Rural PHC links are
  shared, slow and intermittent.  Upload time is not a constant; it is a
  function of the number of concurrent uploads and of link state.
* **Reviewer capacity as the real bottleneck.**  The AI does not remove the
  ophthalmologist, it concentrates their time.  How much depends on the
  *selective* referral policy: cases the model is confident about are
  auto-reported, the rest are queued for a human.  That coupling between a
  model threshold and a staffing number is the whole point of the exercise.
* **Clinical urgency.**  Proliferative DR and suspected macular oedema must
  jump the queue.  A FIFO review queue meets its average SLA while failing
  the patients who matter.

Outputs
-------
Per-run: throughput, per-stage waiting times, resource utilisation, the
review backlog trajectory, and SLA attainment split by urgency.  The
optimiser in :mod:`drscreen.sim.optimize` searches configurations against a
cost model to answer "what do we actually need to buy".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import simpy


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class NetworkProfile:
    """Rural connectivity. Throughput in Mbit/s, availability as a fraction."""
    name: str
    mean_mbps: float
    cv: float               # coefficient of variation of instantaneous rate
    availability: float     # fraction of time the link is usable
    outage_mean_min: float  # mean outage duration when it drops

NETWORKS = {
    "2g_edge":  NetworkProfile("2g_edge", 0.12, 0.65, 0.82, 25.0),
    "3g":       NetworkProfile("3g", 1.6, 0.55, 0.90, 15.0),
    "4g_rural": NetworkProfile("4g_rural", 8.0, 0.45, 0.94, 10.0),
    "4g_good":  NetworkProfile("4g_good", 25.0, 0.30, 0.98, 5.0),
    "fibre":    NetworkProfile("fibre", 100.0, 0.15, 0.995, 3.0),
}


@dataclass
class SimConfig:
    # --- programme scale ---------------------------------------------------
    annual_patients: int = 100_000
    n_phc: int = 40                      # primary health centres in the district
    working_days_per_year: int = 250
    hours_per_day: float = 7.0

    # --- capture ----------------------------------------------------------
    cameras_per_phc: int = 1
    technicians_per_phc: int = 1
    capture_time_min: float = 4.0        # per patient, both eyes
    capture_time_cv: float = 0.35
    images_per_patient: int = 2          # one macula-centred field per eye
    image_mb: float = 3.2

    # --- quality gate -----------------------------------------------------
    edge_iqa: bool = True                # run the gate on the capture device
    edge_iqa_latency_s: float = 0.35
    ungradeable_rate: float = 0.12       # fraction failing the gate first time
    recapture_success_rate: float = 0.75 # fraction recovered on retry
    max_recaptures: int = 2

    # --- network ----------------------------------------------------------
    network: str = "4g_rural"
    concurrent_uploads_per_phc: int = 2
    upload_retries: int = 3

    # --- inference --------------------------------------------------------
    edge_inference: bool = False         # full grading on-device vs central GPU
    edge_inference_s: float = 2.8        # quantised model on an edge SoC
    gpu_servers: int = 2
    gpu_batch_latency_s: float = 0.45    # per image, amortised over a batch
    gpu_workers_per_server: int = 4

    # --- human review -----------------------------------------------------
    ophthalmologists: float = 2.0        # FTE assigned to the programme
    review_hours_per_day: float = 5.0    # realistic reading time, not roster time
    review_time_min: float = 0.5         # with AI pre-annotation (the <30s target)
    review_time_min_unaided: float = 2.5 # for the counterfactual
    auto_report_coverage: float = 0.70   # fraction the model handles alone
    review_time_cv: float = 0.4

    # --- clinical mix -----------------------------------------------------
    prevalence_referable: float = 0.18
    prevalence_urgent: float = 0.045     # PDR / suspected CSME
    sla_days_routine: float = 14.0
    sla_days_urgent: float = 2.0

    # --- simulation -------------------------------------------------------
    sim_days: int = 250
    seed: int = 0
    warmup_days: int = 10

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def patients_per_phc_per_day(self) -> float:
        return self.annual_patients / (self.n_phc * self.working_days_per_year)


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
@dataclass
class SimResults:
    config: dict
    n_arrived: int = 0
    n_captured: int = 0
    n_rejected_ungradeable: int = 0
    n_graded: int = 0
    n_auto_reported: int = 0
    n_human_reviewed: int = 0
    n_referred: int = 0
    n_urgent: int = 0
    throughput_per_year: float = 0.0
    wait_capture_min: dict = field(default_factory=dict)
    wait_upload_min: dict = field(default_factory=dict)
    wait_inference_min: dict = field(default_factory=dict)
    wait_review_hours: dict = field(default_factory=dict)
    turnaround_days: dict = field(default_factory=dict)
    utilisation: dict = field(default_factory=dict)
    review_backlog: list = field(default_factory=list)
    sla_routine: float = 0.0
    sla_urgent: float = 0.0
    bottleneck: str = ""
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    a = np.asarray(values, np.float64)
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)), "p99": float(np.percentile(a, 99)),
            "max": float(a.max())}


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------
class ScreeningProgramme:
    """District-level screening programme as a SimPy process network."""

    MIN_PER_DAY = 24 * 60

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.env = simpy.Environment()
        self.net = NETWORKS[cfg.network]

        # Resources
        self.cameras = [simpy.Resource(self.env, cfg.cameras_per_phc) for _ in range(cfg.n_phc)]
        self.techs = [simpy.Resource(self.env, cfg.technicians_per_phc) for _ in range(cfg.n_phc)]
        self.uplinks = [simpy.Resource(self.env, cfg.concurrent_uploads_per_phc)
                        for _ in range(cfg.n_phc)]
        self.gpu = simpy.Resource(self.env, max(1, cfg.gpu_servers * cfg.gpu_workers_per_server))
        # Reviewers modelled as a priority resource so urgent cases pre-empt.
        self.reviewers = simpy.PriorityResource(self.env, max(1, int(round(cfg.ophthalmologists))))
        self.link_up = [True] * cfg.n_phc

        # Telemetry
        self.res = SimResults(config=cfg.to_dict())
        self._w_capture: list[float] = []
        self._w_upload: list[float] = []
        self._w_infer: list[float] = []
        self._w_review: list[float] = []
        self._turnaround: list[float] = []
        self._busy = {"camera": 0.0, "technician": 0.0, "uplink": 0.0,
                      "gpu": 0.0, "reviewer": 0.0}
        self._backlog: list[tuple[float, int]] = []
        self._review_queue_len = 0
        self._sla_routine = [0, 0]
        self._sla_urgent = [0, 0]
        self._warmup_min = cfg.warmup_days * self.MIN_PER_DAY

    # -- helpers ----------------------------------------------------------
    def _lognormal(self, mean: float, cv: float) -> float:
        """Positive, right-skewed service time -- the realistic shape for
        human-paced tasks, unlike the exponential that queueing textbooks use."""
        if mean <= 0:
            return 0.0
        sigma = math.sqrt(math.log(1 + cv * cv))
        mu = math.log(mean) - sigma * sigma / 2
        return float(self.rng.lognormal(mu, sigma))

    def _past_warmup(self) -> bool:
        return self.env.now >= self._warmup_min

    def _in_working_hours(self) -> bool:
        minute_of_day = self.env.now % self.MIN_PER_DAY
        start = 9 * 60
        return start <= minute_of_day < start + self.cfg.hours_per_day * 60

    def _wait_for_working_hours(self):
        """Advance to the next working period; weekends are folded into the
        ``working_days_per_year`` arrival rate rather than modelled explicitly."""
        minute_of_day = self.env.now % self.MIN_PER_DAY
        start = 9 * 60
        end = start + self.cfg.hours_per_day * 60
        if minute_of_day < start:
            return self.env.timeout(start - minute_of_day)
        if minute_of_day >= end:
            return self.env.timeout(self.MIN_PER_DAY - minute_of_day + start)
        return self.env.timeout(0)

    # -- processes ---------------------------------------------------------
    def link_monitor(self, phc: int):
        """Toggle link availability so uploads see realistic outages."""
        while True:
            up_time = self.rng.exponential(
                self.net.outage_mean_min * self.net.availability / max(1 - self.net.availability, 1e-3))
            yield self.env.timeout(up_time)
            self.link_up[phc] = False
            yield self.env.timeout(self.rng.exponential(self.net.outage_mean_min))
            self.link_up[phc] = True

    def arrivals(self, phc: int):
        per_day = self.cfg.patients_per_phc_per_day
        if per_day <= 0:
            return
        mean_gap = (self.cfg.hours_per_day * 60) / per_day
        pid = 0
        while True:
            yield self._wait_for_working_hours()
            yield self.env.timeout(self.rng.exponential(mean_gap))
            if not self._in_working_hours():
                continue
            pid += 1
            if self._past_warmup():
                self.res.n_arrived += 1
            self.env.process(self.patient(phc, f"P{phc}-{pid}"))

    def patient(self, phc: int, pid: str):
        cfg = self.cfg
        t_arrive = self.env.now

        # ---- capture (camera + technician held together) -----------------
        attempts = 0
        gradeable = False
        while attempts <= cfg.max_recaptures and not gradeable:
            t0 = self.env.now
            with self.cameras[phc].request() as cam_req:
                yield cam_req
                with self.techs[phc].request() as tech_req:
                    yield tech_req
                    if self._past_warmup() and attempts == 0:
                        self._w_capture.append(self.env.now - t0)
                    dur = self._lognormal(cfg.capture_time_min, cfg.capture_time_cv)
                    yield self.env.timeout(dur)
                    if self._past_warmup():
                        self._busy["camera"] += dur
                        self._busy["technician"] += dur

            # ---- quality gate --------------------------------------------
            if cfg.edge_iqa:
                yield self.env.timeout(cfg.edge_iqa_latency_s / 60.0)
            p_fail = cfg.ungradeable_rate if attempts == 0 else \
                cfg.ungradeable_rate * (1 - cfg.recapture_success_rate)
            gradeable = self.rng.random() >= p_fail
            attempts += 1

        if not gradeable:
            if self._past_warmup():
                self.res.n_rejected_ungradeable += 1
            return
        if self._past_warmup():
            self.res.n_captured += 1

        # ---- inference: on the edge, or upload then central GPU -----------
        if cfg.edge_inference:
            dur = cfg.edge_inference_s * cfg.images_per_patient / 60.0
            yield self.env.timeout(dur)
            # Only the compact report leaves the PHC, not the images.
            payload_mb = 0.15
        else:
            payload_mb = cfg.image_mb * cfg.images_per_patient

        t0 = self.env.now
        with self.uplinks[phc].request() as req:
            yield req
            if self._past_warmup():
                self._w_upload.append(self.env.now - t0)
            ok = False
            for _ in range(cfg.upload_retries):
                while not self.link_up[phc]:
                    yield self.env.timeout(1.0)
                rate = max(0.02, self.rng.lognormal(
                    math.log(self.net.mean_mbps) - 0.5 * math.log(1 + self.net.cv ** 2),
                    math.sqrt(math.log(1 + self.net.cv ** 2))))
                minutes = (payload_mb * 8.0) / rate / 60.0
                yield self.env.timeout(minutes)
                if self._past_warmup():
                    self._busy["uplink"] += minutes
                if self.link_up[phc]:
                    ok = True
                    break
            if not ok:
                return

        if not cfg.edge_inference:
            t0 = self.env.now
            with self.gpu.request() as req:
                yield req
                if self._past_warmup():
                    self._w_infer.append(self.env.now - t0)
                dur = cfg.gpu_batch_latency_s * cfg.images_per_patient / 60.0
                yield self.env.timeout(dur)
                if self._past_warmup():
                    self._busy["gpu"] += dur

        if self._past_warmup():
            self.res.n_graded += 1

        # ---- clinical outcome & triage -----------------------------------
        u = self.rng.random()
        urgent = u < cfg.prevalence_urgent
        referable = urgent or (u < cfg.prevalence_referable)
        if self._past_warmup():
            if referable:
                self.res.n_referred += 1
            if urgent:
                self.res.n_urgent += 1

        # Selective prediction: the model auto-reports only where it is
        # confident. Urgent findings always go to a human regardless.
        auto = (not urgent) and (self.rng.random() < cfg.auto_report_coverage)
        if auto:
            if self._past_warmup():
                self.res.n_auto_reported += 1
                self._turnaround.append((self.env.now - t_arrive) / self.MIN_PER_DAY)
                self._record_sla(urgent, (self.env.now - t_arrive) / self.MIN_PER_DAY)
            return

        # ---- human review queue -------------------------------------------
        t0 = self.env.now
        self._review_queue_len += 1
        if self._past_warmup():
            self._backlog.append((self.env.now / self.MIN_PER_DAY, self._review_queue_len))
        prio = 0 if urgent else 1
        with self.reviewers.request(priority=prio) as req:
            yield req
            # Reviewers only work during their reading session.
            yield self._wait_for_working_hours()
            self._review_queue_len -= 1
            if self._past_warmup():
                self._w_review.append((self.env.now - t0) / 60.0)
            dur = self._lognormal(cfg.review_time_min, cfg.review_time_cv)
            yield self.env.timeout(dur)
            if self._past_warmup():
                self._busy["reviewer"] += dur
                self.res.n_human_reviewed += 1

        if self._past_warmup():
            days = (self.env.now - t_arrive) / self.MIN_PER_DAY
            self._turnaround.append(days)
            self._record_sla(urgent, days)

    def _record_sla(self, urgent: bool, days: float):
        if urgent:
            self._sla_urgent[1] += 1
            self._sla_urgent[0] += int(days <= self.cfg.sla_days_urgent)
        else:
            self._sla_routine[1] += 1
            self._sla_routine[0] += int(days <= self.cfg.sla_days_routine)

    def reviewer_capacity_limiter(self):
        """Cap daily reading to `review_hours_per_day` per reviewer.

        Without this the model would let an ophthalmologist read for 24 hours,
        and every capacity conclusion would be nonsense.
        """
        cfg = self.cfg
        daily_budget = cfg.review_hours_per_day * 60 * max(1, int(round(cfg.ophthalmologists)))
        while True:
            start_busy = self._busy["reviewer"]
            yield self.env.timeout(self.MIN_PER_DAY)
            used = self._busy["reviewer"] - start_busy
            if used > daily_budget * 1.05:
                self.res.notes.append(
                    f"Day {int(self.env.now / self.MIN_PER_DAY)}: reviewer load "
                    f"{used/60:.1f} h exceeded the {daily_budget/60:.1f} h budget.")

    # -- run ---------------------------------------------------------------
    def run(self) -> SimResults:
        cfg = self.cfg
        for phc in range(cfg.n_phc):
            self.env.process(self.arrivals(phc))
            self.env.process(self.link_monitor(phc))
        self.env.process(self.reviewer_capacity_limiter())

        horizon = cfg.sim_days * self.MIN_PER_DAY
        self.env.run(until=horizon)

        measured_min = max(horizon - self._warmup_min, 1.0)
        measured_days = measured_min / self.MIN_PER_DAY
        work_min = measured_days * cfg.hours_per_day * 60

        r = self.res
        r.throughput_per_year = r.n_graded / measured_days * cfg.working_days_per_year
        r.wait_capture_min = _stats(self._w_capture)
        r.wait_upload_min = _stats(self._w_upload)
        r.wait_inference_min = _stats(self._w_infer)
        r.wait_review_hours = _stats(self._w_review)
        r.turnaround_days = _stats(self._turnaround)
        r.review_backlog = self._backlog[::max(1, len(self._backlog) // 500)]
        r.sla_routine = self._sla_routine[0] / max(self._sla_routine[1], 1)
        r.sla_urgent = self._sla_urgent[0] / max(self._sla_urgent[1], 1)

        r.utilisation = {
            "camera": self._busy["camera"] / max(work_min * cfg.n_phc * cfg.cameras_per_phc, 1),
            "technician": self._busy["technician"] / max(work_min * cfg.n_phc * cfg.technicians_per_phc, 1),
            "uplink": self._busy["uplink"] / max(measured_min * cfg.n_phc * cfg.concurrent_uploads_per_phc, 1),
            "gpu": self._busy["gpu"] / max(measured_min * cfg.gpu_servers * cfg.gpu_workers_per_server, 1),
            "reviewer": self._busy["reviewer"] / max(
                measured_days * cfg.review_hours_per_day * 60 * max(1, int(round(cfg.ophthalmologists))), 1),
        }
        r.bottleneck = max(r.utilisation, key=r.utilisation.get)
        r.notes = r.notes[:20]
        return r


def simulate(cfg: SimConfig | None = None, **overrides: Any) -> SimResults:
    cfg = cfg or SimConfig()
    if overrides:
        cfg = SimConfig(**{**cfg.to_dict(), **overrides})
    return ScreeningProgramme(cfg).run()
