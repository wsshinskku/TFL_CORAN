# Upgrading from earlier versions

Version 1.0 adopts a standard `src/` package layout. The former top-level
packages (`ue`, `ric`, `gnb`, and `envs`) are replaced by the typed
`tfl_coran` package.

When upgrading from a version before 1.0, replace the tracked source files
instead of copying the new tree over the old one. Otherwise obsolete top-level
packages can remain importable. Start from a clean working tree and create a
backup reference before making the change.

```bash
git status --short
# Continue only when the command above prints nothing.
git branch backup/pre-v1.0.0
git switch -c upgrade/v1.0.0
git rm -r --ignore-unmatch .github/workflows configs data envs gnb models ric scripts tests ue utils
git rm --ignore-unmatch README.md pyproject.toml TFL_CORAN_Elsevier.pdf
# Place the v1.0 source tree in the repository root.
git add -A
git status
git diff --cached --stat
```

The backup branch records the previous commit but does not include uncommitted
files. Review the staged deletions before committing. The manuscript PDF is not
part of the software repository because it has separate distribution terms.

| Earlier command | Current command |
|---|---|
| `python scripts/run_sim.py --config ... --rounds N` | Compatibility script, or `tfl-coran run -c ... -o ...` |
| `python scripts/pretrain_vgae.py ...` | Compatibility script, or `tfl-coran pretrain-vgae` |
| `python scripts/evaluate.py ...` | Read `summary.json`, or use the compatibility reporter |

Configuration keys were renamed so that every consumed value is explicit.
`configs/paper.yaml` is the canonical mapping. Checkpoints from versions before
1.0 are not binary-compatible because their action and state dimensions differ.

The compatibility scripts preserve command names, not the earlier YAML schema.
Unknown keys now raise an error. The former `pretrain_vgae.py --snapshots` and
`--gen-if-missing` flags remain accepted for command-line compatibility; the
historical contexts are generated from the active configuration unless a VGAE
checkpoint is passed to the main CLI.
