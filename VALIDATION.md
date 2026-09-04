# Quality validation — 2026-09-04

Validated changes: recoverable global publication, runtime-aware cache identity,
reassignment consistency, and reproducible installation/runtime checks.

## Automated checks

- Fresh CPython 3.12.12 environment with non-editable wheels for zmip, MSP,
  agent-harness-bridge and standissect-lite, plus `constraints-runtime.txt`.
- Dependency check: **105 packages compatible**.
- Tests copied outside the checkout: **99 passed, 3 warnings**.
- Runtime API checks and CLI help passed. Native runtime: glibc 2.17 and
  HDF5 1.14.4; this node's HDF5 build needed `openmpi/5.0.5` headers.
- Ruff (`--isolated --select E9,F`) and `git diff --check` passed.

Tests cover hard process exit during publication, rollback with and without
old outputs, figure/report failures, damaged reports versus damaged data,
unchanged-version source edits, and CSV/H5AD reassignment mismatches. Existing
coverage/removal, island, submission and process-cleanup regressions remain.

Reproduce the installation check on this cluster with:

```bash
module load openmpi/5.0.5
HDF5_DIR=/share/software/user/open/hdf5/1.14.4 \
  scripts/validate_install.sh /absolute/path/to/new-validation \
  /path/to/msp /path/to/agent-harness-bridge /path/to/standissect-lite
```

The configured index could not resolve the harness release. Source wheels
are required for this development revision. Wheel hashes, resolved packages,
runtime identities and test logs are archived in the `wheel-validation/`
subdirectory of the evidence directory below.

## Real-model end-to-end check

Input: a seed-42 subset of an existing Fu2022 MSP result, containing 128
endothelial cells and 128 myeloid cells, all **17,813 genes**, original counts,
expression, metadata and the global graph/embedding. The input manifest records
the source file/hash, subset hash and exact selected cell IDs.

Runtime: `HARNESS=openai`, model `doubao-seed-2-1-turbo-260628`, two parallel
lineages, four-CPU affinity within existing Slurm allocation 41891659.
Arguments: `--min-cells 64 --n-top-genes 500 --n-pcs 15 --n-neighbors 30
--max-turns 80`.

All stages actually ran: model planning, markers, lineage integration, foreign
scores, model annotation, audited merge, figures and HTML publication. The
process exited 0 and retained **256/256 cells**, with **0 removals and
0 reassignments**. This is a small workflow check, not biological-accuracy
validation or a full-size benchmark; removal/reassignment branches are covered
by synthetic regression tests.

`scripts/check_result.py` independently reopened the final H5AD and verified
cell coverage, original annotations, gene order, expression, raw expression
and all input layers. The stored runtime/source identity matches the current
implementation. Repeating the same command exited 0, reused both lineages and
made **zero model calls**. Independent report rebuilding also exited 0; data
checks and the global completion receipt passed afterward.

Evidence directory:
`/scratch/users/chensj16/zmip-validation/20260904-quality/`

- `live/report.html`: global report.
- `live.log`, `live-resume.log`, `report-rebuild.log`: execution records.
- `live-validation.json`: independent data checks and runtime identity.
- `input_manifest.json`: input origin and exact sample selection.
- `wheel-validation/`: isolated environment, wheels, hashes and 99-test log.

Input subset SHA-256:
`c5e02241817b021b20633707cc708067806222c8164cbabb5eee0adc2186228b`

The final H5AD hash is in `live-validation.json`; hashes for the whole output
set are in `live/.zmip-global.json`.
