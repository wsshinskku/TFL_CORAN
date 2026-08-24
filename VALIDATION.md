# Verification

Last checked: 2026-08-24

The source tree was tested on Windows with Python 3.12.13 and the CPU build of
PyTorch. GitHub Actions runs lint, tests, and a smoke experiment on Linux with
Python 3.10 and 3.12.

## Test suite

```text
python -m ruff check src tests scripts
All checks passed!

python -W error -m pytest -p no:cacheprovider
26 passed
```

The test matrix also covers the following `configs/smoke.yaml` workflows:

- one TFL-CORAN run;
- all five methods (`heuristic`, `drl`, `fdrl`, `cfdrl`, `tfl_coran`);
- all five component-ablation variants;
- a two-method, two-seed aggregate summary;
- the compatibility run, VGAE pretraining, and evaluation scripts.

The sdist and universal wheel were built and installed in a separate
environment with `--no-deps`. `pip check`, `tfl-coran doctor`, and an
installed-wheel smoke run were also checked.

## Scope

The tests cover DDQN online selection and target evaluation, the reward
definition, continuing-process episode handling, action encoding, replay
growth, QoS event tracking, symmetric graph construction, VGAE
log-standard-deviation KL, restricted checkpoint loading, GMM responsibilities,
FL invariants, transfer edge cases, config rejection, deterministic end-to-end
execution, and matched central/local warmup budgets.

Exact comparison with Tables 3 and 4 additionally requires the original
external traces and simulator settings. See `docs/ASSUMPTIONS.md` and
`paper_reported/README.md` for the parameters used by the included simulator.
