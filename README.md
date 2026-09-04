# zmip — zoom-in pipeline

The round after [msp](https://github.com/chansigit/msp): take msp's
`annotated.h5ad`, split it into lineages, re-embed each lineage on its own,
and let a per-lineage agent refine the annotation, clean noise and hand
misassigned cells to the lineage they belong to. Same pattern as osp/msp —
fixed computation, narrow agent decisions validated by the host, one
self-contained report per lineage plus a global one.

```
msp annotated.h5ad ──▶ plan ──▶ per lineage: re-embed → foreign scores → agent ──▶ merge
                       (agent)                  (msp.integrate_adata)      (agent)     annotated_zmip.h5ad
```

## Install

```bash
pip install msp-sc                                        # msp on PyPI (import name `msp`)
pip install zmip                                          # includes agent-harness-bridge runtime extras
```

## Usage

```bash
python -m zmip msp_out/annotated.h5ad --outdir zmip_out \
    --model doubao-seed-2-1-turbo-260628 [--min-cells 800]
python -m zmip.report zmip_out          # rebuild the global report only
```

The default is `HARNESS=openai` with Ark/Doubao Turbo. Set
`HARNESS=claude MODEL=claude-sonnet-5` or `HARNESS=deepseek` to select another
adapter. ZMIP imports the shared `harness_bridge` directly instead of routing
agent execution through MSP.

Re-running resumes only verified results for the same input and options.
Private `.zmip-*.json` receipts record an input SHA-256, options and stage
file hashes; a lineage depends on the current plan and marker list and is
complete only after its H5AD, proposal, report and both audit CSVs are written
and its cell coverage is validated. Interrupted or modified results rerun.
Input or option changes, and legacy directories without receipts, require a
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
- Agent on `msp_leiden_r2.0` of the subset, one Claude Code Task per
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

DEG references retain the string API: `5,1` selects that exact current
subcluster; use CSV quoting for a pool such as `"5,0","5,1"`. Ambiguous
unquoted references return a correction request to the agent. This uses the
same reference parser as the installed MSP `DegCache`.

A failed lineage does not cancel independent lineages. If the parent is
interrupted or fails while preparing or launching work, it terminates all
active child process groups, escalates to SIGKILL after a grace period,
and waits for the children and their log readers. Successfully completed
lineages remain eligible for resume.
