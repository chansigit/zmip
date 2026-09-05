# Installation and runtime

## Install

The verified installation uses compatible local source checkouts of ZMIP,
MSP, agent-harness-bridge, and standissect-lite, with `uv` installed. From the
ZMIP checkout, run the commands below using absolute paths. The script builds
non-editable wheels, creates a CPython 3.12 environment, installs the pinned
scientific dependencies, and runs API and test checks outside the checkout.
During the 2026-09-04 validation, the configured index could not resolve the
required harness release, so a bare `pip install zmip` did not reproduce it.

```bash
./scripts/validate_install.sh /absolute/path/to/validation \
    /path/to/msp /path/to/agent-harness-bridge /path/to/standissect-lite
source /absolute/path/to/validation/env/bin/activate
```

## Native dependencies

Older Linux systems may need source builds for dependencies without
compatible wheels. For h5py, set `HDF5_DIR` to a supported HDF5 installation
with matching headers and libraries; an MPI-enabled build also needs MPI
development headers. The script records loaded HDF5/libc versions, resolved
packages, wheel hashes, and runtime identity. See the
[validated cluster setup](../VALIDATION.md) for an example.

## Agent configuration

The default `HARNESS=openai` adapter runs Doubao through Volcengine Ark and
requires `ARK_API_KEY`. Choose a model with `--model`; `HARNESS=deepseek` and
`HARNESS=claude` select alternative adapters with their own setup requirements
in the [shared harness guide](https://github.com/chansigit/agent-harness-bridge).
`--max-turns` and `--effort` configure agent execution; `--language` sets the
report language. Only the OpenAI/Doubao path was exercised in the latest
real-model ZMIP validation.

## Integration and resources

`--n-top-genes`, `--n-pcs`, `--n-neighbors`, `--resolutions`, and repeatable
`--harmony KEY=VALUE` options apply to every local re-embedding. Resolutions
must be unique, finite, positive, and include `1.0` and `2.0` for annotation.
Set `ZMIP_PARALLEL` to cap concurrent lineages; scheduling also considers
available CPUs and estimated memory. The parent loads the input H5AD, so
allow memory for the global dataset as well as active lineage analyses.

```bash
ZMIP_PARALLEL=2 python -m zmip input.h5ad --outdir zmip_out \
    --n-top-genes 2000 --n-pcs 50 --n-neighbors 15
```

## Resume identity

Private `.zmip-*.json` receipts record input and stage file hashes, analysis
options, Python/dependency versions, and source hashes for ZMIP, MSP, the
harness, and standissect-lite. Changes to editable sources also invalidate
reuse. Input, option, runtime changes, or legacy directories without receipts
require a new output directory or `--force`, which recomputes the plan,
markers, and all selected lineages. Hash verification adds sequential file
reads without loading a second expression matrix.

## Failed and interrupted runs

One failed lineage allows independent lineages to finish, but prevents the
final merge. Parent interruption or failure while preparing children causes
active process groups to terminate; cleanup escalates to SIGKILL after a
grace period and waits for children and log readers. Completed, verified
lineages remain reusable. Only one parent may write to an output directory
at a time.

## Publishing results

Global CSVs, figures, H5AD, and report are built in staging, and the H5AD is
reopened before publication. A journal guards file replacement, with the
completion receipt written last. Failure, or the next run after a hard
interruption, restores the previous set from backups. Report rebuilding
rejects incomplete, modified, or stale data. External readers should hold
`zmip.cache.lock_run(outdir)` and check `zmip.publication.complete(outdir)`
while reading; multi-file replacement is not atomic for unlocked readers.
This recovery covers process interruption, not storage hardware failure.
