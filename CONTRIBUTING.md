# Contributing

1. Create a focused branch and keep generated `runs/` artifacts out of commits.
2. Install `pip install -e '.[dev]'`.
3. Run `python -m ruff check src tests scripts` and `python -m pytest`.
4. Add a deterministic unit test for every mathematical or lifecycle change.
5. Document simulator settings and measurement protocols in the relevant
   config and method guide. Keep paper tables under `paper_reported/` and
   generated outputs under `runs/`. Identify the configuration and data used
   in each experiment.

Bug reports should include the resolved config, seed, platform metadata and the
smallest command that reproduces the issue.
