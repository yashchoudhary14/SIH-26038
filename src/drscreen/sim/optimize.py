"""Resource-allocation optimisation for a district screening programme.

The simulation answers "what happens if we deploy X".  This module answers
the question a district health officer actually asks: **"what is the cheapest
configuration that screens 100,000 people a year without anyone waiting too
long for an urgent result?"**

Formulated as constrained minimisation:

    minimise    annual_cost(config)
    subject to  throughput      >= target
                sla_urgent      >= 0.95
                sla_routine     >= 0.90
                utilisation_i   <= 0.85  for every resource i

The utilisation cap is not cosmetic.  Queueing systems degrade
super-linearly: at 95% utilisation the expected wait is roughly four times
what it is at 80%, and any variance in arrivals tips it into unbounded
backlog.  A plan that runs a single ophthalmologist at 97% "on paper" fails
the first week someone takes leave.

Costs are order-of-magnitude figures for an Indian district programme, stated
explicitly in :data:`DEFAULT_COSTS` so they can be replaced with real
procurement numbers.  They are inputs, not findings.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field, asdict

import numpy as np

from .telemedicine import NETWORKS, SimConfig, simulate, SimResults


# --------------------------------------------------------------------------
# Cost model (INR, annualised). Replace with procurement figures.
# --------------------------------------------------------------------------
DEFAULT_COSTS = {
    "camera_capex": 450_000.0,        # portable non-mydriatic fundus camera
    "camera_life_years": 5.0,
    "camera_maintenance_pa": 35_000.0,
    "technician_pa": 300_000.0,
    "ophthalmologist_pa": 1_800_000.0,  # per FTE
    "gpu_server_capex": 350_000.0,
    "gpu_server_life_years": 4.0,
    "gpu_server_opex_pa": 60_000.0,
    "edge_device_capex": 45_000.0,      # per PHC, for on-device inference
    "edge_device_life_years": 4.0,
    "connectivity_pa_per_phc": {        # annual link cost by profile
        "2g_edge": 6_000.0, "3g": 12_000.0, "4g_rural": 24_000.0,
        "4g_good": 42_000.0, "fibre": 120_000.0,
    },
}


def annual_cost(cfg: SimConfig, costs: dict | None = None) -> dict:
    c = {**DEFAULT_COSTS, **(costs or {})}
    n_cameras = cfg.n_phc * cfg.cameras_per_phc
    n_techs = cfg.n_phc * cfg.technicians_per_phc

    camera = n_cameras * (c["camera_capex"] / c["camera_life_years"] + c["camera_maintenance_pa"])
    tech = n_techs * c["technician_pa"]
    ophthal = cfg.ophthalmologists * c["ophthalmologist_pa"]
    gpu = 0.0 if cfg.edge_inference else cfg.gpu_servers * (
        c["gpu_server_capex"] / c["gpu_server_life_years"] + c["gpu_server_opex_pa"])
    edge = cfg.n_phc * c["edge_device_capex"] / c["edge_device_life_years"] \
        if (cfg.edge_inference or cfg.edge_iqa) else 0.0
    link = cfg.n_phc * c["connectivity_pa_per_phc"].get(cfg.network, 24_000.0)

    total = camera + tech + ophthal + gpu + edge + link
    return {"camera": camera, "technician": tech, "ophthalmologist": ophthal,
            "gpu": gpu, "edge_device": edge, "connectivity": link, "total": total}


# --------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------
@dataclass
class Constraints:
    min_throughput: int = 100_000
    min_sla_urgent: float = 0.95
    min_sla_routine: float = 0.90
    max_utilisation: float = 0.85
    max_turnaround_p90_days: float = 14.0

    def check(self, r: SimResults) -> tuple[bool, list[str]]:
        v: list[str] = []
        if r.throughput_per_year < self.min_throughput:
            v.append(f"throughput {r.throughput_per_year:,.0f}/yr < {self.min_throughput:,} target")
        if r.sla_urgent < self.min_sla_urgent:
            v.append(f"urgent SLA {r.sla_urgent:.1%} < {self.min_sla_urgent:.0%}")
        if r.sla_routine < self.min_sla_routine:
            v.append(f"routine SLA {r.sla_routine:.1%} < {self.min_sla_routine:.0%}")
        for name, u in r.utilisation.items():
            if u > self.max_utilisation:
                v.append(f"{name} utilisation {u:.1%} > {self.max_utilisation:.0%} "
                         f"(queue becomes unstable)")
        p90 = r.turnaround_days.get("p90", 0.0)
        if p90 > self.max_turnaround_p90_days:
            v.append(f"p90 turnaround {p90:.1f} d > {self.max_turnaround_p90_days:.0f} d")
        return (len(v) == 0), v


@dataclass
class Candidate:
    params: dict
    cost: dict
    results: dict
    feasible: bool
    violations: list = field(default_factory=list)
    screened_out: bool = False       # rejected analytically, never simulated

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Analytic pre-screen
# --------------------------------------------------------------------------
def offered_load(cfg: SimConfig) -> dict[str, float]:
    """Utilisation each resource would need, from Little's Law -- no simulation.

    Offered load rho = arrival rate x service time / servers. When rho >= 1 the
    queue is unstable: the backlog grows without bound and no amount of
    simulated time reveals anything except a larger backlog.

    This exists because those configurations are also, by far, the *slowest* to
    simulate -- an unstable review queue accumulates tens of thousands of
    waiting entities, and a naive grid search spends almost all of its runtime
    (and memory) on plans it was always going to reject. Screening them out
    analytically turned a search that exhausted 700 MB and ran for the better
    part of an hour into one that finishes in a couple of minutes.
    """
    per_day = cfg.annual_patients / cfg.working_days_per_year

    # Capture: cameras and technicians are held together for the whole session,
    # and rejected captures are repeated, so demand is inflated by the expected
    # number of attempts.
    expected_attempts = 1.0 + cfg.ungradeable_rate * (
        1.0 + cfg.ungradeable_rate * (1 - cfg.recapture_success_rate))
    capture_min_needed = per_day * cfg.capture_time_min * expected_attempts
    capture_min_avail = cfg.n_phc * cfg.cameras_per_phc * cfg.hours_per_day * 60
    tech_min_avail = cfg.n_phc * cfg.technicians_per_phc * cfg.hours_per_day * 60

    # Review: urgent cases always see a human; the rest only when the model
    # declines to auto-report.
    human_frac = cfg.prevalence_urgent + (1 - cfg.prevalence_urgent) * (
        1 - cfg.auto_report_coverage)
    review_min_needed = per_day * human_frac * cfg.review_time_min
    review_min_avail = (max(1e-9, cfg.ophthalmologists)
                        * cfg.review_hours_per_day * 60)

    # Inference and uplink are always-on, so they get the full 24 h.
    gpu_min_needed = 0.0 if cfg.edge_inference else (
        per_day * cfg.gpu_batch_latency_s * cfg.images_per_patient / 60.0)
    gpu_min_avail = cfg.gpu_servers * cfg.gpu_workers_per_server * 24 * 60

    net = NETWORKS[cfg.network]
    payload_mb = 0.15 if cfg.edge_inference else cfg.image_mb * cfg.images_per_patient
    upload_min_each = (payload_mb * 8.0) / max(net.mean_mbps, 1e-6) / 60.0
    uplink_min_needed = per_day * upload_min_each / max(net.availability, 1e-6)
    uplink_min_avail = (cfg.n_phc * cfg.concurrent_uploads_per_phc * 24 * 60)

    return {
        "camera": capture_min_needed / max(capture_min_avail, 1e-9),
        "technician": capture_min_needed / max(tech_min_avail, 1e-9),
        "reviewer": review_min_needed / max(review_min_avail, 1e-9),
        "gpu": gpu_min_needed / max(gpu_min_avail, 1e-9),
        "uplink": uplink_min_needed / max(uplink_min_avail, 1e-9),
    }


def prescreen(cfg: SimConfig, max_utilisation: float = 0.85
              ) -> tuple[bool, list[str]]:
    """Cheap stability check. Returns (worth_simulating, violations)."""
    rho = offered_load(cfg)
    viol = [f"{k} offered load {v:.0%} > {max_utilisation:.0%} "
            f"(analytic; queue unstable)"
            for k, v in rho.items() if v > max_utilisation]
    return (len(viol) == 0), viol


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------
#: The search space deliberately spans *whether to use AI triage at all*:
#: ``review_time_min`` 2.5 is unaided reading, 0.5 is reading a pre-annotated
#: case. Without that axis the optimiser only ever compares AI-assisted designs
#: against each other and never has to demonstrate that AI assistance is what
#: makes the programme feasible.
DEFAULT_GRID = {
    "n_phc": [8, 12, 16, 20],
    "cameras_per_phc": [1, 2],
    "ophthalmologists": [1.0, 2.0, 3.0, 4.0],
    "auto_report_coverage": [0.0, 0.5, 0.7, 0.85],
    "review_time_min": [0.5, 2.5],
    "edge_inference": [False, True],
    "network": ["3g", "4g_rural"],
}


def optimise(base: SimConfig | None = None,
             grid: dict | None = None,
             constraints: Constraints | None = None,
             costs: dict | None = None,
             sim_days: int = 90,
             seeds: tuple[int, ...] = (0,),
             progress: bool = True) -> dict:
    """Exhaustive search over `grid`, returning the cheapest feasible design.

    Multiple `seeds` average out simulation noise; with one seed a
    configuration can look feasible purely because that particular random
    stream was kind, which is how capacity plans get built on sand.
    """
    base = base or SimConfig()
    grid = grid or DEFAULT_GRID
    constraints = constraints or Constraints()

    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    out: list[Candidate] = []

    n_screened = 0
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        cfg = SimConfig(**{**base.to_dict(), **params, "sim_days": sim_days})

        # Reject analytically-unstable designs before paying for a simulation.
        ok, viol = prescreen(cfg, constraints.max_utilisation)
        if not ok:
            n_screened += 1
            stub = SimResults(config=cfg.to_dict())
            stub.utilisation = offered_load(cfg)
            stub.bottleneck = max(stub.utilisation, key=stub.utilisation.get)
            out.append(Candidate(params=params, cost=annual_cost(cfg, costs),
                                 results=stub.to_dict(), feasible=False,
                                 violations=viol, screened_out=True))
            continue

        runs = []
        for s in seeds:
            cfg_s = SimConfig(**{**cfg.to_dict(), "seed": s})
            runs.append(simulate(cfg_s))

        # Average the decision-relevant scalars across seeds; keep the worst
        # utilisation, because a plan must hold in the bad stream too.
        agg = SimResults(config=cfg.to_dict())
        agg.throughput_per_year = float(np.mean([r.throughput_per_year for r in runs]))
        agg.sla_urgent = float(np.mean([r.sla_urgent for r in runs]))
        agg.sla_routine = float(np.mean([r.sla_routine for r in runs]))
        agg.utilisation = {k: max(r.utilisation.get(k, 0.0) for r in runs)
                           for k in runs[0].utilisation}
        agg.turnaround_days = {k: float(np.mean([r.turnaround_days.get(k, 0.0) for r in runs]))
                               for k in runs[0].turnaround_days}
        agg.n_graded = int(np.mean([r.n_graded for r in runs]))
        agg.n_human_reviewed = int(np.mean([r.n_human_reviewed for r in runs]))
        agg.n_auto_reported = int(np.mean([r.n_auto_reported for r in runs]))
        agg.bottleneck = max(agg.utilisation, key=agg.utilisation.get)

        feasible, viol = constraints.check(agg)
        out.append(Candidate(params=params, cost=annual_cost(cfg, costs),
                             results=agg.to_dict(), feasible=feasible, violations=viol))
        if progress and (i + 1) % max(1, len(combos) // 10) == 0:
            print(f"  [{i+1}/{len(combos)}] evaluated "
                  f"({n_screened} screened out analytically)", flush=True)

    feasible = [c for c in out if c.feasible]
    best = min(feasible, key=lambda c: c.cost["total"]) if feasible else None
    if best is None:
        # Nothing feasible: report the closest miss so the failure is diagnosable.
        best = min(out, key=lambda c: len(c.violations))

    return {
        "n_evaluated": len(out),
        "n_simulated": len(out) - n_screened,
        "n_screened_out": n_screened,
        "n_feasible": len(feasible),
        "best": best.to_dict(),
        "cost_per_patient": (best.cost["total"] /
                             max(best.results["throughput_per_year"], 1.0)),
        "all": [c.to_dict() for c in out],
        "constraints": asdict(constraints),
    }


def sensitivity(base: SimConfig, param: str, values: list,
                sim_days: int = 90, seed: int = 0) -> list[dict]:
    """One-at-a-time sensitivity sweep -- what each lever actually buys."""
    rows = []
    for v in values:
        cfg = SimConfig(**{**base.to_dict(), param: v, "sim_days": sim_days, "seed": seed})
        r = simulate(cfg)
        rows.append({
            param: v,
            "throughput_per_year": r.throughput_per_year,
            "sla_urgent": r.sla_urgent,
            "sla_routine": r.sla_routine,
            "turnaround_p90_days": r.turnaround_days.get("p90", 0.0),
            "review_wait_p90_h": r.wait_review_hours.get("p90", 0.0),
            "bottleneck": r.bottleneck,
            "utilisation": r.utilisation,
            "annual_cost": annual_cost(cfg)["total"],
            "human_reviewed": r.n_human_reviewed,
        })
    return rows


# --------------------------------------------------------------------------
# Named scenarios
# --------------------------------------------------------------------------
SCENARIOS = {
    "baseline_manual": SimConfig(
        n_phc=12, cameras_per_phc=1, ophthalmologists=2.0,
        auto_report_coverage=0.0, review_time_min=2.5,
        network="3g", edge_inference=False,
        annual_patients=100_000,
    ),
    "ai_assisted": SimConfig(
        n_phc=12, cameras_per_phc=1, ophthalmologists=2.0,
        auto_report_coverage=0.70, review_time_min=0.5,
        network="3g", edge_inference=False,
        annual_patients=100_000,
    ),
    "ai_edge_lowbandwidth": SimConfig(
        n_phc=12, cameras_per_phc=1, ophthalmologists=2.0,
        auto_report_coverage=0.70, review_time_min=0.5,
        network="2g_edge", edge_inference=True, edge_iqa=True,
        annual_patients=100_000,
    ),
    "stress_2x_demand": SimConfig(
        n_phc=12, cameras_per_phc=1, ophthalmologists=2.0,
        auto_report_coverage=0.70, review_time_min=0.5,
        network="3g", edge_inference=False,
        annual_patients=200_000,
    ),
}
