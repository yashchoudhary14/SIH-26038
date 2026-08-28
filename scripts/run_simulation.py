"""Telemedicine workflow simulation, optimisation and sensitivity analysis.

    python scripts/run_simulation.py --scenarios          # compare deployments
    python scripts/run_simulation.py --optimise           # cheapest feasible plan
    python scripts/run_simulation.py --sensitivity ophthalmologists 1 2 3 4
    python scripts/run_simulation.py --export-matlab matlab/
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from drscreen.sim.optimize import (Constraints, DEFAULT_GRID, SCENARIOS,
                                   annual_cost, optimise, sensitivity)
from drscreen.sim.simulink_export import export_all
from drscreen.sim.telemedicine import SimConfig, simulate


def fmt_inr(x: float) -> str:
    if x >= 1e7:
        return f"Rs {x/1e7:.2f} Cr"
    if x >= 1e5:
        return f"Rs {x/1e5:.2f} L"
    return f"Rs {x:,.0f}"


def run_scenarios(sim_days: int, out: Path) -> dict:
    print("=" * 96)
    print("DEPLOYMENT SCENARIOS - district programme, 100,000 patients/year")
    print("=" * 96)
    print(f"{'scenario':24s} {'thru/yr':>10s} {'bottleneck':>12s} {'reviewer':>9s} "
          f"{'SLA urg':>8s} {'SLA rout':>9s} {'p90 days':>9s} {'cost/yr':>13s} {'feasible':>9s}")
    print("-" * 96)

    rows = {}
    for name, base in SCENARIOS.items():
        cfg = SimConfig(**{**base.to_dict(), "sim_days": sim_days, "warmup_days": 10})
        r = simulate(cfg)
        ok, viol = Constraints().check(r)
        cost = annual_cost(cfg)["total"]
        print(f"{name:24s} {r.throughput_per_year:10,.0f} {r.bottleneck:>12s} "
              f"{r.utilisation['reviewer']:8.1%} {r.sla_urgent:8.1%} {r.sla_routine:9.1%} "
              f"{r.turnaround_days.get('p90',0):9.2f} {fmt_inr(cost):>13s} "
              f"{'yes' if ok else 'NO':>9s}")
        if viol:
            for v in viol:
                print(f"{'':24s}   - {v}")
        rows[name] = {"results": r.to_dict(), "cost": annual_cost(cfg),
                      "feasible": ok, "violations": viol}

    print()
    base = rows.get("baseline_manual"); ai = rows.get("ai_assisted")
    if base and ai:
        b_rev = base["results"]["utilisation"]["reviewer"]
        a_rev = ai["results"]["utilisation"]["reviewer"]
        print(f"AI-assisted review cuts ophthalmologist load from {b_rev:.0%} to "
              f"{a_rev:.0%} of available reading capacity")
        print(f"  ({base['results']['n_human_reviewed']:,} -> "
              f"{ai['results']['n_human_reviewed']:,} images needing human review "
              f"over the simulated period).")
        if b_rev > 1.0 and a_rev < 0.85:
            print("  Without AI triage the review queue is unstable (load > 100%): the "
                  "backlog grows without bound and the programme cannot meet its SLA "
                  "at any staffing level it can afford.")
    (out / "scenarios.json").write_text(json.dumps(rows, indent=2, default=float))
    return rows


def run_optimise(sim_days: int, out: Path, target: int, seeds: tuple) -> dict:
    print("\n" + "=" * 96)
    print("RESOURCE OPTIMISATION - cheapest configuration meeting every constraint")
    print("=" * 96)
    base = SCENARIOS["ai_assisted"]
    cons = Constraints(min_throughput=target)
    print(f"Constraints: throughput >= {target:,}/yr, urgent SLA >= "
          f"{cons.min_sla_urgent:.0%}, routine SLA >= {cons.min_sla_routine:.0%}, "
          f"all utilisation <= {cons.max_utilisation:.0%}")
    grid_size = 1
    for v in DEFAULT_GRID.values():
        grid_size *= len(v)
    print(f"Searching {grid_size} configurations x {len(seeds)} seeds "
          f"({sim_days}-day horizon each)...\n")

    res = optimise(base, DEFAULT_GRID, cons, sim_days=sim_days, seeds=seeds)
    best = res["best"]
    print(f"\nEvaluated {res['n_evaluated']} configurations: "
          f"{res.get('n_screened_out', 0)} rejected analytically (unstable queue), "
          f"{res.get('n_simulated', res['n_evaluated'])} simulated, "
          f"{res['n_feasible']} feasible")
    if res["n_feasible"] == 0:
        print("NO FEASIBLE CONFIGURATION in the search grid. Closest miss:")
        for v in best["violations"]:
            print(f"  - {v}")
    print("\nRecommended configuration:")
    for k, v in best["params"].items():
        print(f"  {k:24s} {v}")
    print("\nCost breakdown (annualised):")
    for k, v in best["cost"].items():
        if k != "total":
            print(f"  {k:24s} {fmt_inr(v):>14s}")
    print(f"  {'TOTAL':24s} {fmt_inr(best['cost']['total']):>14s}")
    print(f"  {'per patient screened':24s} "
          f"Rs {res['cost_per_patient']:.1f}")
    r = best["results"]
    print(f"\nPerformance: {r['throughput_per_year']:,.0f}/yr, bottleneck "
          f"'{r['bottleneck']}', urgent SLA {r['sla_urgent']:.1%}, "
          f"routine SLA {r['sla_routine']:.1%}")
    (out / "optimisation.json").write_text(json.dumps(res, indent=2, default=float))
    return res


def run_sensitivity(param: str, values: list, sim_days: int, out: Path):
    print("\n" + "=" * 96)
    print(f"SENSITIVITY - {param}")
    print("=" * 96)
    base = SCENARIOS["ai_assisted"]
    typed = []
    for v in values:
        try:
            typed.append(int(v) if float(v).is_integer() and "." not in str(v) else float(v))
        except ValueError:
            typed.append(v if v not in ("true", "false") else v == "true")
    rows = sensitivity(base, param, typed, sim_days=sim_days)
    print(f"{param:>18s} {'thru/yr':>10s} {'SLA urg':>9s} {'SLA rout':>9s} "
          f"{'p90 days':>9s} {'reviews':>9s} {'bottleneck':>12s} {'cost/yr':>13s}")
    print("-" * 96)
    for r in rows:
        print(f"{str(r[param]):>18s} {r['throughput_per_year']:10,.0f} "
              f"{r['sla_urgent']:9.1%} {r['sla_routine']:9.1%} "
              f"{r['turnaround_p90_days']:9.2f} {r['human_reviewed']:9,d} "
              f"{r['bottleneck']:>12s} {fmt_inr(r['annual_cost']):>13s}")
    (out / f"sensitivity_{param}.json").write_text(json.dumps(rows, indent=2, default=float))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", action="store_true")
    ap.add_argument("--optimise", action="store_true")
    ap.add_argument("--sensitivity", nargs="+", metavar=("PARAM", "VALUE"))
    ap.add_argument("--export-matlab", type=Path, default=None)
    ap.add_argument("--sim-days", type=int, default=120)
    ap.add_argument("--target", type=int, default=100_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out", type=Path, default=Path("outputs/simulation"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    did = False
    if a.scenarios or not (a.optimise or a.sensitivity or a.export_matlab):
        run_scenarios(a.sim_days, a.out); did = True
    if a.optimise:
        run_optimise(a.sim_days, a.out, a.target, tuple(a.seeds)); did = True
    if a.sensitivity:
        run_sensitivity(a.sensitivity[0], a.sensitivity[1:], a.sim_days, a.out); did = True
    if a.export_matlab:
        cfg = SCENARIOS["ai_assisted"]
        paths = export_all(cfg, a.export_matlab)
        # Also store the reference results the MATLAB validator diffs against.
        r = simulate(SimConfig(**{**cfg.to_dict(), "sim_days": a.sim_days}))
        (a.out / "results.json").write_text(json.dumps(r.to_dict(), indent=2, default=float))
        print("\nMATLAB/Simulink export:")
        for k, p in paths.items():
            print(f"  {k:10s} {p}")
        print(f"  reference  {a.out/'results.json'}")
        did = True
    if not did:
        ap.print_help()


if __name__ == "__main__":
    main()
