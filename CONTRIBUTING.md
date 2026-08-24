# Contributing

1. Create a focused branch and keep generated `runs/` artifacts out of commits.
2. Install `pip install -e '.[dev]'`.
3. Run `python -m ruff check src tests scripts` and `python -m pytest`.
4. Add a deterministic unit test for every mathematical or lifecycle change.
5. Mark new simulator choices as **reported**, **assumed**, or **measured** in
   the relevant config/documentation. Never copy a manuscript table into the
   generated-results path or claim exact reproduction without its source traces.

Bug reports should include the resolved config, seed, platform metadata and the
smallest command that reproduces the issue.
