# Telemedicine programme simulation

The district screening programme as a queueing network (SimPy), the cost optimiser, and the Simulink/SimEvents realisation the problem statement asks for.

## Telemedicine simulation (PS item 5)

`src/drscreen/sim/telemedicine.py` models a district programme as a queueing
network. The interesting behaviour is in the couplings, which is why a
spreadsheet cannot answer these questions:

- **Recapture loops.** A rejected image sends the patient back into the camera
  queue they just left, so a gate 5 points stricter can cost far more than 5%
  of throughput.
- **Bandwidth as a shared, intermittent resource,** with a two-state link model.
- **Reviewer capacity as the real bottleneck,** coupled to the model's
  selective-referral threshold.
- **Clinical urgency preemption** — a FIFO review queue meets its average SLA
  while failing the patients who matter.

Scenario results (100,000 patients/year, 12 PHCs, 2 ophthalmologist FTE):

| scenario | throughput/yr | bottleneck | reviewer load | routine SLA | p90 turnaround | feasible |
|---|---|---|---|---|---|---|
| `baseline_manual` | 99,968 | reviewer | **140.5%** | 66.6% | 17.9 d | **no** |
| `ai_assisted` | 99,636 | camera | 11.0% | 100% | 0.01 d | throughput only |
| `ai_edge_lowbandwidth` | 99,568 | camera | 10.9% | 100% | 0.02 d | throughput only |
| `stress_2x_demand` | 200,455 | camera | 21.9% | 100% | 0.02 d | yes |

The headline: **without AI triage the review queue is unstable** — demand is
141% of available reading capacity, so the backlog grows without bound and no
affordable staffing level meets the SLA. AI-assisted review takes that to 11%.

The residual throughput gap (99,636 vs 100,000) is not a rounding artefact: it
is the patients who remain ungradeable after the maximum number of recaptures.
To *screen* 100,000 you must *see* about 100,400. The optimiser accounts for
this by sizing capture capacity accordingly.

`scripts/run_simulation.py --optimise` searches configurations against a cost
model (`DEFAULT_COSTS`, in INR, stated explicitly so procurement figures can
replace them) and returns the cheapest plan meeting throughput, both SLAs, and
a utilisation ceiling of 85% — the ceiling matters because queueing systems
degrade super-linearly and a plan running one ophthalmologist at 97% "on
paper" fails the first week someone takes leave.

Over 1,024 configurations — 192 rejected analytically as unstable, 832
simulated, 532 feasible — the cheapest feasible plan is:

| lever | value |
|---|---|
| PHCs with a camera | 8 |
| cameras per PHC | 1 |
| ophthalmologist FTE | **1.0** |
| auto-report coverage | 0.50 |
| review mode | **AI-assisted (0.5 min/case)** |
| inference | on-device (edge) |
| connectivity | 3G |

100,394 screened/year, camera-bound at 54% utilisation, reviewer at 35%, both
SLAs met, **Rs 53.9 lakh/year — about Rs 54 per patient screened.**

Two things the search decided rather than assumed:

- It **chose AI-assisted review** over unaided reading. Every one of the 192
  analytically-rejected configurations was an unaided-review design. The value
  of AI triage is an output of the optimisation, not a premise of it.
- It **chose edge inference**, because removing the GPU server and most of the
  bandwidth requirement is worth more than the per-PHC device cost at this
  scale — which is the opposite of the cloud-first default most designs reach
  for.

Two notes on the search itself:

- The grid deliberately spans **whether to use AI triage at all**
  (`review_time_min` 2.5 = unaided reading, 0.5 = reading a pre-annotated
  case). Without that axis the optimiser only ever compares AI-assisted
  designs against each other and never has to show that AI assistance is what
  makes the programme affordable.
- Configurations are **pre-screened analytically** with Little's Law before
  any simulation runs (`offered_load`): if offered load ≥ 1 the queue is
  unstable and more simulated time only produces a larger backlog. On the
  1024-configuration grid this rejects 192 (19%) without simulating them —
  all of them unaided-review designs. Note that with AI triage enabled, the
  *worst* configuration in the space still only reaches 0.67 offered load,
  which is itself the finding: AI assistance is what keeps the design space
  stable at all.
- Every configuration is run under multiple random seeds and judged on its
  *worst* utilisation, because a plan that looks feasible under one kind
  random stream is not a plan.

### MATLAB / Simulink

**The executed model is [`simulink/district_model.slx`](../simulink/README.md)**: a
SimEvents model of one health centre's session (capture, AI quality check and
grading on the edge device, upload, the ophthalmologist's read), built and run in
MATLAB R2026a, with seeded independent replications. The project dossier's
Simulator is a block-for-block browser port of it, checked against MATLAB's
replications. The SimPy model above covers the programme scale (a year, many
PHCs, costs) that a single-session model does not.

The generated bridge below keeps the SimPy configuration and a SimEvents
realisation in step. `scripts/run_simulation.py --export-matlab outputs/simulink_bridge/`
generates, from the same `SimConfig`:

- `dr_screening_params.m` — every parameter as a MATLAB struct
- `build_dr_screening_model.m` — builds the SimEvents block diagram
- `validate_against_simpy.m` — runs the Simulink model and diffs its outputs
  against the SimPy reference

One source of truth, two runtimes, and a script that keeps them honest.
`matlab/simulation/README.md` documents the block-level mapping and names the two
elements (the recapture feedback path and the link-state Markov chain) that
need the graphical editor.
