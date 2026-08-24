# Migration from the prototype repository

This release keeps the project name and research design but adopts a standard
`src/` package layout. The old top-level modules (`ue`, `ric`, `gnb`, `envs`)
are replaced by the typed `tfl_coran` package.

This archive is a **clean replacement**, not an overlay. Copying it over the
old working tree would leave legacy tracked packages in place. On a branch of
the existing repository, remove the old tracked implementation first, then
copy this archive's contents while preserving only the existing `.git/`
directory. Start only from a clean tree; commit or stash any local work first.
Create a backup reference before the replacement. A safe reviewable workflow
is (choose unused branch names if these already exist):

```bash
git status --short
# Stop here unless the command above prints nothing.
git branch backup/pre-completed-implementation
git switch -c completed-implementation
git rm -r --ignore-unmatch .github/workflows configs data envs gnb models ric scripts tests ue utils
git rm --ignore-unmatch README.md pyproject.toml TFL_CORAN_Elsevier.pdf
# Copy the completed archive contents into the repository root now.
git add -A
git status
git diff --cached --stat
```

The backup branch preserves the pre-migration commit, but it does not preserve
uncommitted files; that is why the clean-tree check is required. Review the
staged deletion list, including the legacy workflow deletion, before committing.
The manuscript PDF is not bundled because its sharing rights are separate from
the MIT software license.

| Prototype command | Completed release |
|---|---|
| `python scripts/run_sim.py --config ... --rounds N` | Same compatibility script, or `tfl-coran run -c ... -o ...` |
| `python scripts/pretrain_vgae.py ...` | Same compatibility script, or `tfl-coran pretrain-vgae` |
| `python scripts/evaluate.py ...` | Read the generated `summary.json` or use the compatibility reporter |

Configuration keys were renamed to make every consumed value explicit. Use
`configs/paper.yaml` as the canonical mapping. Existing checkpoints are not
binary-compatible because the prototype action/state dimensions were not
aligned with the revised manuscript.

The compatibility scripts preserve command names, not the old YAML schema.
Unknown/legacy keys now fail loudly instead of being silently ignored. The old
`pretrain_vgae.py --snapshots` and `--gen-if-missing` flags are accepted only so
automation does not crash; this release always generates the configured
synthetic historical contexts unless a completed checkpoint is supplied to the
main CLI.

The implementation was reconciled against prototype commit
`2480e703ba72eb13b931b9236ffa5de2ffb8f227` and the May 2026 manuscript.
