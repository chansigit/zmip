"""zmip (zoom-in pipeline): per-lineage refinement of an msp annotation.

    plan      UMAP-connected lineages from the coarse labels (agent + host rules)
    zoom      each lineage re-embedded on its own (msp.integrate_adata), scored
              for foreign-lineage signal, annotated by its own agent
              (refine fine labels / remove noise / reassign / recluster)
    merge     fold back with real removal → annotated_zmip.h5ad + report.html

Command line:
    python -m zmip annotated.h5ad --outdir zmip_out [--min-cells 800] [--model ...]
    python -m zmip.report zmip_out

Depends on msp (integration core, plots, report machinery) and needs the
claude-agent-sdk for both agent steps.
"""

from .foreign import lineage_markers, score_foreign
from .merge import merge_back
from .plan import DEFAULT_MIN_CELLS, plan_lineages, validate_plan
from .report import generate_report

__all__ = ["DEFAULT_MIN_CELLS", "generate_report", "lineage_markers", "merge_back", "plan_lineages",
           "score_foreign", "validate_plan"]
