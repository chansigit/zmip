<p align="center">
  <img src="assets/logo.svg" alt="zmip — zoom-in pipeline" width="640">
</p>

<h1 align="center">ZMIP: Zoom-In Pipeline</h1>

<p align="center">
  Refine cell populations within each lineage after multi-sample integration.
</p>

<p align="center">
  <a href="https://pypi.org/project/zmip/"><img src="https://img.shields.io/pypi/v/zmip?label=PyPI&amp;color=258B81&amp;style=flat" alt="PyPI version"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&amp;logo=python&amp;logoColor=white" alt="Python 3.10 or newer"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-173B49?style=flat" alt="MIT license"></a>
  <a href="https://github.com/chansigit/eca-rsi"><img src="https://img.shields.io/badge/Ecosystem-ECA--RSI-258B81?style=flat" alt="Part of the ECA-RSI ecosystem"></a>
</p>

<p align="center">
  <a href="#why-zoom-in">Why zoom in?</a> ·
  <a href="#get-started">Get started</a> ·
  <a href="#read-your-results">Results</a> ·
  <a href="#further-reading">Documentation</a>
</p>

ZMIP takes an annotated dataset from [MSP](https://github.com/chansigit/msp),
analyzes each lineage separately, and refines cell-type labels using marker
genes and quality evidence. It writes updated annotations, records removed
or reassigned cells, and produces reports for review.

## Why zoom in?

Differences between major cell types can dominate a global analysis.
Recomputing features, neighbors, and clusters within a lineage helps examine
its finer populations. ZMIP uses this local view to refine labels and review
remaining quality concerns. When there are too few cells for stable subgroup
analysis, small lineages retain their existing annotations.

## How it works

An AI assistant groups cells into lineages using the existing labels and
embedding. Each selected lineage is re-embedded independently, then reviewed
using marker genes, quality measurements, and signals from other lineages.
The program checks the submitted decisions before merging the results back
into the global dataset.

Click the diagram to open the interactive version, with pan, zoom, search,
and guided views.

<p align="center">
  <a href="https://raw.githack.com/chansigit/zmip/main/docs/architecture.html">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/chansigit/zmip/main/assets/architecture-dark.svg">
      <img src="https://raw.githubusercontent.com/chansigit/zmip/main/assets/architecture-light.svg" alt="ZMIP module structure: the input H5AD is planned into lineages, each lineage is re-embedded and annotated by an agent through harness_bridge, results are merged back with host checks and published atomically" width="960">
    </picture>
  </a>
</p>

## What you get

The output includes a refined H5AD, a global HTML report linking to detailed
lineage reports, and cell-level records of removals and reassignments.
Original MSP annotations remain available for comparison; retained cells
keep their input expression, counts, and global embedding.

## Get started

Start with MSP's `annotated.h5ad`, including its annotations, counts, batch
metadata, graph, and UMAP. Install from PyPI (Python 3.10 or newer; MSP and
the agent harness come with it), set your Volcengine Ark API key, then run:

```bash
python -m pip install zmip
export ARK_API_KEY="YOUR_ARK_API_KEY"
HARNESS=openai python -m zmip msp_out/annotated.h5ad --outdir zmip_out
```

## Read your results

After a successful run, open `zmip_out/report.html` in your browser. Check
the lineage plan, follow the links to review local labels and supporting
genes, then inspect removed and reassigned populations. To share all reports,
copy the output directory with its subdirectories so the links still work.

| File | Contents |
| --- | --- |
| `report.html` | Global summary and links to lineage reports |
| `annotated_zmip.h5ad` | Retained cells with refined annotations |
| `zmip_removed.csv` | Removed cells and their recorded sources |
| `zmip_reassigned.csv` | Cells assigned to another lineage's coarse label |
| `<lineage>/report.html` | Local analysis, evidence, and annotation decisions |

## Does ZMIP change my data?

Your input file stays unchanged. Cells removed by local filtering or the
annotation agent are excluded from the final H5AD. Reassigned cells receive
new labels; they are not re-embedded in the destination lineage during this
run. Global plots retain MSP's embedding. See the
[input and output reference](docs/input-output.md) for fields and details.

## Can I resume a run?

Repeat the same command to reuse completed, verified stages. Agent settings
such as `--max-turns` or `--model` may differ between runs; only unfinished
lineages use the new values. If you change the input, analysis settings, or
runtime, use a new output directory to keep both analyses, or add `--force`
to recompute the plan and all selected lineages. To rebuild only the global
report, run:

```bash
python -m zmip.report zmip_out
```

## Validation

An isolated installation passed 99 tests, and an OpenAI/Doubao run completed
the workflow on 256 Fu2022 cells, including independent data checks and
resume without model calls. This checks the workflow at small scale;
full-dataset performance and biological accuracy remain unvalidated.
See [validation records and remaining checks](VALIDATION.md).

## Further reading

ZMIP follows [OSP](https://github.com/chansigit/osp) for sample-level review
and [MSP](https://github.com/chansigit/msp) for integration within the
[ECA-RSI](https://github.com/chansigit/eca-rsi) workflow.
See [inputs and outputs](docs/input-output.md) for data conventions,
[runtime options](docs/runtime.md) for installation and configuration,
and [GitHub issues](https://github.com/chansigit/zmip/issues) for questions
or problems. ZMIP uses the [MIT license](LICENSE).
