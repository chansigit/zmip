<p align="center">
  <img src="assets/logo.svg" alt="zmip — zoom-in pipeline" width="640">
</p>

# zmip — zoom-in pipeline

The round after [msp](https://github.com/chansigit/msp): take msp's
`annotated.h5ad`, split it into lineages, re-embed each lineage on its own,
and let a per-lineage agent refine the annotation, clean noise and hand
misassigned cells to the lineage they belong to. Same pattern as osp/msp —
fixed computation, narrow agent decisions validated by the host, one
self-contained report per lineage plus a global one.

See [Validation status](#validation-status) for the tested scope and remaining
validation work, and [VALIDATION.md](VALIDATION.md) for reproducible checks and
execution records.

```
msp annotated.h5ad ──▶ plan ──▶ per lineage: re-embed → foreign scores → agent ──▶ merge
                       (agent)                  (msp.integrate_adata)      (agent)     annotated_zmip.h5ad
```

## Install

Run from this checkout with `uv` installed and local source checkouts of
MSP, agent-harness-bridge and standissect-lite:

```bash
./scripts/validate_install.sh /absolute/path/to/validation \
    /path/to/msp /path/to/agent-harness-bridge /path/to/standissect-lite
```

This development revision requires compatible source revisions of MSP and the
shared harness. During the 2026-09-04 validation, the configured package index did not supply
`agent-harness-bridge[all]==0.1.0`; a bare `pip install zmip` therefore does not
reproduce this revision. The script builds four non-editable wheels, creates a
fresh CPython 3.12 environment, installs with `constraints-runtime.txt`, runs
`pip check`, behavioral API checks and the tests outside the checkout. It
records wheel hashes, resolved dependencies and the runtime identity. On older
Linux systems, dependencies without compatible wheels may need source builds.
For a source-built h5py, set `HDF5_DIR` to a supported HDF5 installation
so its headers and shared library match. An MPI-enabled HDF5 also needs its
matching MPI development headers. The validation records the loaded
HDF5 and libc versions in `native-runtime.json`.
Use the resulting `validation/env/bin/python` to run zmip.

## Usage

```bash
python -m zmip msp_out/annotated.h5ad --outdir zmip_out \
    --model doubao-seed-2-1-turbo-260628 --min-cells 800
python -m zmip.report zmip_out          # rebuild the global report only
```

The default is `HARNESS=openai` with Ark/Doubao Turbo. Set
`HARNESS=claude MODEL=claude-sonnet-5` or `HARNESS=deepseek` to select another
adapter. ZMIP imports the shared `harness_bridge` directly instead of routing
agent execution through MSP.

Use an MSP `annotated.h5ad` with unique cell IDs, coarse/fine annotations,
the global graph and UMAP, and expression/counts required by MSP integration.
Annotation columns default to `msp_ann_coarse` and `msp_ann_fine`; override
them with `--coarse-col` and `--fine-col`. Batch and species metadata default
to `uns['msp']`; pass `--batch-col` and `--species` when needed.
Set `ZMIP_PARALLEL=2`, for example, to cap concurrent lineage processes;
the scheduler also considers available CPU and estimated memory.

### Resume and recompute

Re-running resumes only verified results for the same input and options.
Private `.zmip-*.json` receipts record an input SHA-256, options and stage
file hashes. The identity also includes Python/dependency versions and source
hashes for zmip, MSP, the harness and standissect-lite, including editable
changes without a version bump. A lineage depends on the current plan and marker list and is
complete only after its H5AD, proposal, report and both audit CSVs are written
and its cell coverage is validated. Interrupted or modified results rerun.
Input, option or runtime changes, and legacy directories without receipts, require a
new output directory or `--force`. **`--force` now recomputes the plan as well
as markers and all zoomed lineages**; it does not preserve the old plan.
Only one parent run may write to an output directory at a time. Hashing adds
a sequential read of the input and any lineage H5AD considered for reuse;
it does not load a second matrix into memory.
Integration knobs (`--resolutions`,
`--n-top-genes`, `--n-pcs`, `--n-neighbors`, `--harmony KEY=VALUE`) are the
same as msp's and apply to every per-lineage re-embedding.
`--resolutions` must include `1.0` and `2.0`, the parent and base clusterings
used by annotation. Missing, duplicate, non-finite or non-positive values
are rejected before reading the input or running an agent.

## Steps

### 1. plan (`zmip.plan`)

Host writes the evidence: cells/samples per coarse label, a kNN
cross-connectivity matrix between coarse labels (share of each label's
graph edges landing on every other label), PAGA on the same graph, and the
coarse-label UMAP. The agent **must read the UMAP** and pools coarse labels
that form one connected island into one lineage — even across cell types
when data quality fuses them (T/B/myeloid as one immune island) — and keeps
separate islands separate even when related; states (proliferating,
stressed) go with the island they sit in. Host rules: every coarse label
assigned exactly once; zoom only for lineages with at least `--min-cells`
(default 800 — below that leiden cannot resolve stable substates); the
plan must agree with the host's own islands (`lineage_islands.csv`:
connected components of the 2-D UMAP kNN graph, long edges pruned, as % of
each label's cells) — pooling labels that sit on separate islands is
rejected outright, splitting labels that share one island is pushed back
once and accepted on resubmission with `confirm_shared_islands: true`
(recorded as `host_warnings`); archived to `zmip_plan.json`. If no lineage is
selected for zoom, the msp labels pass through unchanged.

### 2. per lineage (`zmip.foreign`, `msp.integrate_adata`, `zmip.annotate`)

- Subset → `msp.integrate_adata`: HVG/PCA/harmony/leiden(0.3/1.0/2.0)/UMAP
  recomputed on the lineage alone, with every msp artifact (QC tables,
  cell-level outliers, standissect fragments, DEG at r1.0/r2.0,
  `preannotation_removal.csv`) in `<lineage>/`.
- **Foreign-lineage scores**: lineage-level markers (wilcoxon on the whole
  dataset at the plan's lineage level, specific genes only) → `sc.tl.score_genes`
  for every other lineage → `obs["foreign_<lineage>"]`, per-cluster summaries
  and UMAPs. Evidence only: close lineages share programs, so the agent
  decides between doublet, ambient, misassignment and genuine biology.
  Marker estimation requires at least two cells in both the lineage and
  its reference. Ineligible lineages are logged and receive no marker list;
  their cells remain in the reference for eligible lineages. When no lineage
  is selected for zoom, marker computation is skipped and the marker CSV
  retains its columns with no rows.
- Agent on `msp_leiden_r2.0` of the subset, one task per
  cluster, tools `cluster_context` / `check_genes` / `check_deg` /
  `check_stability` / `subcluster` (reclustering allowed). Per cluster:
  distinctness → identity → foreign signal → merge, and one action:
  `keep` (coarse label within the lineage), `remove` (with reason), or
  `reassign` to another lineage's coarse label (relabel only — the cells are
  not re-embedded there this round). Host validation as in msp.annotate plus
  the reassign rules. Removal is real: subset pre-annotation filtering ∪
  agent-removed clusters.
- Outputs: `annotation_proposal.json`, `annotation_removed.csv`,
  `annotation_reassigned.csv`, `annotated.h5ad`, `report.html` (msp's report
  with the lineage's Cell Type Annotation section).

### 3. merge (`zmip.merge`, `zmip.report`)

Fold every lineage back into the global object. `annotated_zmip.h5ad` keeps
the survivors with `zmip_lineage`, `zmip_cluster` (`<lineage>:<id>`),
`zmip_ann_coarse`, `zmip_ann_fine`, `zmip_reassigned_from`; `msp_ann_*`
stay for the audit trail. No global re-embedding here (next round's job):
the global figures use msp's UMAP. Archives `zmip_removed.csv` (every
removed cell with lineage, cluster, sources) and `zmip_reassigned.csv`.
`report.html`: plan · lineages (linked per-lineage reports) · final
annotation · removed & reassigned.

Before publishing merged annotations, the host requires exactly one result
for each zoomed lineage. Its survivors and removals must cover its input
cells exactly once; duplicates, missing cells, cells from other lineages,
and reassignment records outside the survivors are rejected. Input cell
identifiers must also be unique. Failed validation leaves global output
files untouched.

Reassignment records must match the H5AD cell set, destination coarse labels,
fine labels and original cluster membership (including merged clusters).
Unknown destinations, same-lineage moves and unrecorded foreign labels are
rejected before publication.

Global CSVs, figures, H5AD and the report are built in a private staging
directory and the H5AD is reopened before publication. `.zmip-publish.json`
guards the multi-file replacement window; `.zmip-global.json` is written last.
On failure or the next run after a hard interruption, the host restores the
previous set from private backups. Report rebuilding refuses incomplete,
modified or stale data. Public filenames and file schemas remain unchanged.
External readers should hold `zmip.cache.lock_run(outdir)`, then require
`zmip.publication.complete(outdir)` while opening the global output; independent
files cannot all be replaced by one filesystem operation. These checks cover process interruption, not storage
hardware failure or unsynchronized external writers.

For a small real-data validation run, independently check cell coverage,
original annotations, expression and counts with:

```bash
python scripts/check_result.py input.h5ad zmip_out --json validation.json
```

DEG references retain the string API: `5,1` selects that exact current
subcluster; use CSV quoting for a pool such as `"5,0","5,1"`. Ambiguous
unquoted references return a correction request to the agent. This uses the
same reference parser as the installed MSP `DegCache`.

A failed lineage does not cancel independent lineages. If the parent is
interrupted or fails while preparing or launching work, it terminates all
active child process groups, escalates to SIGKILL after a grace period,
and waits for the children and their log readers. Successfully completed
lineages remain eligible for resume.

## Validation status

As of 2026-09-04, the fixes identified in the code-quality review are committed.
The validation completed so far is:

| Check | Result |
| --- | --- |
| Isolated source-wheel installation | 105 compatible packages; 99 tests passed |
| Interruption and output integrity | Regression coverage for child cleanup, publication rollback, damaged caches and CSV/H5AD consistency |
| Real OpenAI/Doubao workflow | Planning through final report on 256 Fu2022 cells and 17,813 genes |
| Independent data checks | Cell coverage, gene order, original annotations, expression, raw and input layers preserved |
| Resume and report rebuilding | Successful; the repeated workflow made zero model calls |

Remaining work is validation and distribution: full-dataset performance and
memory measurements, fresh end-to-end checks for Claude and DeepSeek, and
package-index installation once the required harness release is available.
The real-model check retained all 256 cells, so removal and reassignment were
exercised by synthetic regression tests only. These results establish a small
workflow check, not biological accuracy or performance at full scale.

See [VALIDATION.md](VALIDATION.md) for the exact environment, run parameters,
input hash and execution-record locations.
