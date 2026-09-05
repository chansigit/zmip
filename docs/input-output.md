# Inputs and outputs

## Input dataset

Use MSP's `annotated.h5ad` with unique cell identifiers, expression and counts
needed by MSP integration, its global graph and UMAP, and coarse/fine labels.
The default annotation columns are `msp_ann_coarse` and `msp_ann_fine`;
`--coarse-col` and `--fine-col` override them. Batch and species default to
`uns['msp']['batch_col']` and `uns['msp']['species']`; supply `--batch-col`
and `--species` when needed.

## Lineage planning

The plan assigns every coarse label exactly once, using label counts,
UMAP islands, graph connectivity, and PAGA as evidence. Pooling labels from
separate islands is rejected; splitting labels on one island requires the
agent's explicit confirmation, recorded as a warning. `--min-cells` controls
which lineages are large enough to reanalyze. If none qualify, existing
labels pass through and marker computation is skipped.

## Local analysis

Each selected lineage receives new features, PCA, Harmony integration,
neighbors, Leiden clusters, and UMAP through MSP. Foreign-lineage scores
provide evidence for reviewing misassignment, shared biology, doublets, and
ambient RNA. Marker estimation requires at least two cells in both a lineage
and its reference; ineligible groups receive no markers but remain available
as reference cells for other groups.

## Annotation decisions

The agent checks cluster context, gene expression, differential expression,
and stability, and may subcluster mixed populations. It submits `keep`,
`remove`, or `reassign` decisions. Reassignment must target another lineage's
known coarse label. For DEG queries, `5,1` identifies an exact subcluster;
a pool uses CSV quoting, such as `"5,0","5,1"`. Ambiguous references require
correction before analysis continues.

## Final H5AD

`annotated_zmip.h5ad` contains surviving cells with the input gene order,
expression, layers, raw data when present, and global embedding. Local
embeddings stay in each lineage's output. Original `msp_ann_*` columns remain
available alongside the fields below; cells in skipped lineages retain their
input labels.

| Field in `obs` | Meaning |
| --- | --- |
| `zmip_lineage` | Final lineage assignment |
| `zmip_cluster` | Local cluster, formatted as `<lineage>:<id>`; missing for skipped lineages |
| `zmip_ann_coarse` | Refined coarse label |
| `zmip_ann_fine` | Refined fine label |
| `zmip_reassigned_from` | Source lineage for reassigned cells |
| `zmip_action` | `keep` for cells retained in the final H5AD |

## Reports and audit files

The global report shows the plan, lineage summaries, final labels, and
removed/reassigned cells. Each lineage directory contains its own
`annotated.h5ad`, `annotation_proposal.json`, `annotation_removed.csv`,
`annotation_reassigned.csv`, and `report.html`. Global `zmip_removed.csv`
combines pre-annotation filtering and agent removals; `zmip_reassigned.csv`
records destination labels. Preserve the directory structure when sharing
the linked reports.

## Output validation

Before merging, survivors and removals must cover each lineage's input
exactly once. Duplicate, missing, or foreign cells are rejected, as are
reassignment records that disagree with H5AD labels or cluster membership.
For a small validation dataset, the command below also compares the final
H5AD against the input for cell coverage, gene order, original annotations,
expression, raw data, and layers.

```bash
python scripts/check_result.py input.h5ad zmip_out --json validation.json
```
