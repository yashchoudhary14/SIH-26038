# MATLAB / Simulink bridge

> **In the MATLAB edition:** these files are the MATLAB/Simulink side of the
> telemedicine capacity model, generated from the Python `SimConfig`, except that
> `validate_against_simpy.m` reads the bundled `simpy_results.json` by
> default. The SimPy model and its optimiser live in `src/drscreen/sim/`
> (paths below are relative to the repository root). Running this needs Simulink + SimEvents, which the
> screening console and web server do not.

The programme-scale telemedicine model lives in Python
(`src/drscreen/sim/telemedicine.py`), and this directory holds a SimEvents
realisation generated from that same configuration. The Simulink model that
was built, run and validated in MATLAB is the district model in
[`../../simulink/`](../../simulink/README.md) (`district_model.slx`).

## Files

| file | purpose |
|---|---|
| `dr_screening_params.m` | every parameter as a MATLAB struct, generated from `SimConfig` |
| `build_dr_screening_model.m` | builds the SimEvents block diagram programmatically |
| `validate_against_simpy.m` | runs the Simulink model and diffs its outputs against the SimPy JSON |

## Regenerating

```
python scripts/run_simulation.py --export-matlab outputs/simulink_bridge/
```

Any change to `SimConfig` flows into `dr_screening_params.m`, so the two
runtimes cannot silently drift apart.

## Two things the generated script does not build

`build_dr_screening_model.m` lays out the forward topology. Two elements need
the graphical editor (or hand-written `add_line` calls against specific port
indices); the district model in `simulink/` builds its own recapture loop:

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

For the programme scale (a year, many PHCs, costs), the SimPy model: it is
covered by the Python test suite, it is what the optimiser searches, and its
outputs are what the results quote. For one health centre's session, the
Simulink district model in `simulink/`, whose MATLAB replications the
dossier's browser simulator is validated against.
