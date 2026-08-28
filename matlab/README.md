# MATLAB / Simulink bridge

The executable telemedicine model lives in Python
(`src/drscreen/sim/telemedicine.py`) because the rest of the pipeline is
Python and one source of truth beats two. This directory holds the Simulink
realisation required by the problem statement, generated from that same
configuration.

## Files

| file | purpose |
|---|---|
| `dr_screening_params.m` | every parameter as a MATLAB struct, generated from `SimConfig` |
| `build_dr_screening_model.m` | builds the SimEvents block diagram programmatically |
| `validate_against_simpy.m` | runs the Simulink model and diffs its outputs against the SimPy JSON |

## Regenerating

```
python scripts/run_simulation.py --export-matlab matlab/
```

Any change to `SimConfig` flows into `dr_screening_params.m`, so the two
runtimes cannot silently drift apart.

## Two things the generated script does not build

`build_dr_screening_model.m` lays out the forward topology. Two elements need
the graphical editor (or hand-written `add_line` calls against specific port
indices) and are present in the shipped `.slx`:

1. **The recapture feedback path** — `QualityGate` port 2 routes back into
   `CameraQueue`, carrying an attempt-count attribute that a second switch
   uses to give up after `max_recaptures`. This loop is the single most
   important non-obvious behaviour in the model: tightening the quality gate
   costs camera throughput super-linearly, because rejected patients re-enter
   the queue they just left.

2. **The link-state Markov chain** — a two-state chain (up/down) with mean
   sojourn times `net_uptime_mean_min` and `net_outage_mean_min` driving the
   `LinkAvailable` entity gate.

## Which model should I trust?

The SimPy model, because it is the one under test — it runs in CI, it is
what the optimiser searches, and its outputs are what the validation report
quotes. The Simulink model exists so the work can be inspected and extended
inside MATLAB, and `validate_against_simpy.m` is what keeps it honest.
