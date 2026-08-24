# Release validation

Validation date: 2026-08-19

The 1.0.0 source tree was validated on Windows with Python 3.12.13 and the
CPU build of PyTorch. The repository CI repeats lint, tests, and a smoke run on
Linux with Python 3.10 and 3.12.

## Checks completed

```text
python -m ruff check src tests scripts
All checks passed!

python -W error -m pytest -p no:cacheprovider
26 passed
```

The following executable workflows also completed successfully with
`configs/smoke.yaml`:

- one TFL-CORAN run;
- all five benchmark methods (`heuristic`, `drl`, `fdrl`, `cfdrl`,
  `tfl_coran`);
- all five component-ablation variants;
- a two-method, two-seed reproduction summary;
- the legacy-compatible run, VGAE pretraining, and evaluation scripts.

Both sdist and universal wheel were built from the final source tree. The wheel
was installed into a separate environment with `--no-deps`; `pip check`,
`tfl-coran doctor`, and an installed-wheel smoke run passed.

## Scientific validation boundary

The tests cover DDQN online-selection/target-evaluation, the exact reward,
continuing-process episode handling, action encoding, replay growth, QoS event
tracking, symmetric graph construction, VGAE log-standard-deviation KL,
restricted checkpoint loading, GMM responsibilities, FL invariants, transfer
edge cases, config rejection, deterministic end-to-end execution, and matched
central/local warmup budgets.

This validates implementation behavior; it is not a claim that manuscript
Tables 3 and 4 were numerically reproduced. The original external traces and
several simulator/training parameters were not supplied. See
`docs/ASSUMPTIONS.md` and `paper_reported/README.md`.
