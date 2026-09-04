"""python -m zmip: zoom-in pass over an msp annotated.h5ad.

  plan      agent groups coarse labels into UMAP-connected lineages, picks
            which to zoom (>= --min-cells)                 → zmip_plan.json
  markers   lineage-level marker lists for foreign-lineage scores
                                                           → lineage_markers.csv
  per lineage (concurrently, see zmip.lineage): subset → msp.integrate_adata
            (re-embed) → foreign scores → annotation agent (refine / remove /
            reassign / recluster) → <lineage>/{annotation_proposal.json,
            annotated.h5ad, report.html}
  merge     fold back, real removal                        → annotated_zmip.h5ad,
            zmip_removed.csv, zmip_reassigned.csv, report.html

Re-running resumes only verified results for the same input and options.
--force recomputes the plan, markers and every zoomed lineage.
If no lineage is selected for zoom, annotated_zmip.h5ad carries
the msp labels unchanged.
"""

import argparse
import hashlib
import os
import sys

import pandas as pd
import scanpy as sc
from harness_bridge import resolve_agent_config
from msp.report import write_report_context

from . import cache, publication
from .runtime import check_runtime

check_runtime()
from .cli import add_integration_options, parse_harmony
from .foreign import MARKER_COLUMNS, lineage_markers
from .lineage import (
    contract_done,
    lineage_dir,
    load_result,
    run_lineage,
    run_lineages_parallel,
    subset_for,
    validate_resolutions,
)
from .merge import merge_back
from .plan import DEFAULT_MIN_CELLS, plan_lineages

parser = argparse.ArgumentParser(prog="zmip", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("h5ad", help="msp annotated.h5ad (survivors with msp_ann_coarse/msp_ann_fine)")
parser.add_argument("--outdir", required=True)
parser.add_argument("--coarse-col", default="msp_ann_coarse")
parser.add_argument("--fine-col", default="msp_ann_fine")
parser.add_argument("--batch-col", default=None, help="defaults to uns['msp']['batch_col']")
parser.add_argument("--species", default=None, help="defaults to uns['msp']['species']")
parser.add_argument("--min-cells", type=int, default=DEFAULT_MIN_CELLS,
                    help=f"smallest lineage that gets zoomed (default {DEFAULT_MIN_CELLS})")
add_integration_options(parser)
parser.add_argument("--language", default="English")
parser.add_argument("--model", default=None)
parser.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"])
parser.add_argument("--max-turns", type=int, default=200)
parser.add_argument("--report-context", default=None, metavar="TEXT",
                    help='where this run sits, for report titles (e.g. "round 2 · fu2022-meniscus")')
parser.add_argument("--force", action="store_true")
args = parser.parse_args()
try:
    args.resolutions = validate_resolutions(args.resolutions)
    harmony_kwargs = parse_harmony(args.harmony)
    agent_config = resolve_agent_config(model=args.model)
    args.model = agent_config.model
except ValueError as exc:
    parser.error(str(exc))


out = os.path.abspath(args.outdir)
with cache.lock_run(out):
    publication.recover(out)
    # Check the on-disk input before loading or modifying any scientific outputs.
    options = {k: v for k, v in vars(args).items() if k not in {"h5ad", "outdir", "force", "report_context"}}
    options["harness"] = agent_config.harness
    options["harness_options"] = {k: os.environ.get(k) for k in ("DSH_PROVIDER", "OPENAI_AGENTS_API")}
    # Endpoint identity matters, but do not copy URLs (possibly containing credentials) into receipts.
    options["endpoint_sha256"] = hashlib.sha256(os.environ.get("DOUBAO_BASE_URL", "").encode()).hexdigest()
    generation = cache.prepare_run(out, args.h5ad, options, force=args.force)
    write_report_context(out, args.report_context)
    ad = sc.read_h5ad(args.h5ad)
    if not ad.obs_names.is_unique:
        sys.exit("input cell identifiers must be unique")
    meta = ad.uns.get("msp", {})
    batch_col = args.batch_col or meta.get("batch_col")
    if not batch_col:
        sys.exit("no --batch-col and uns['msp']['batch_col'] absent")
    species = args.species or (meta.get("species") or None)
    for c in (args.coarse_col, args.fine_col):
        if c not in ad.obs:
            sys.exit(f"obs[{c!r}] missing — input must be msp's annotated.h5ad (or pass --coarse-col/--fine-col)")
    print(f"== {ad.n_obs} cells, batch={batch_col!r}, species={species}", flush=True)

    plan = plan_lineages(ad, args.coarse_col, batch_col, out, min_cells=args.min_cells, species=species,
                         model=args.model, effort=args.effort,
                         force=not cache.valid(out, "plan", generation, ["zmip_plan.json"]))
    cache.seal(out, "plan", generation, ["zmip_plan.json"])
    label_to_lineage = {lab: ln["name"] for ln in plan["lineages"] for lab in ln["coarse_labels"]}
    ad.obs["_zmip_lineage"] = ad.obs[args.coarse_col].astype(str).map(label_to_lineage).astype("category")
    for ln in plan["lineages"]:
        print(f"== lineage {ln['name']}: {ln['coarse_labels']} n={ln['n_cells']} zoom={ln['zoom']}", flush=True)

    markers_p = os.path.join(out, "lineage_markers.csv")
    zoomed = [ln for ln in plan["lineages"] if ln["zoom"]]
    marker_generation = {"run_id": generation, "plan": cache.file_digest(os.path.join(out, "zmip_plan.json"))}
    mk = None
    if zoomed and cache.valid(out, "markers", marker_generation, ["lineage_markers.csv"]):
        try:
            mk = pd.read_csv(markers_p, keep_default_na=False, dtype={"lineage": str, "gene": str})
        except pd.errors.EmptyDataError:  # a half-written/empty file from an interrupted run: recompute
            mk = None
    if not zoomed:
        print("== no lineage selected for zoom — skipping markers and passing msp labels through", flush=True)
        # Keep the existing CSV contract even when no marker computation is needed.
        pd.DataFrame(columns=MARKER_COLUMNS).to_csv(markers_p, index=False)
        markers = {}
    elif mk is not None:
        markers = {g: mk.loc[mk["lineage"] == g, "gene"].tolist() for g in mk["lineage"].unique()}
    else:
        print("== lineage-level markers (for foreign-lineage scores)", flush=True)
        markers = lineage_markers(ad, "_zmip_lineage", out)

    cache.seal(out, "markers", marker_generation, ["lineage_markers.csv"])

    all_labels = set(label_to_lineage)
    results = {}
    keys_for_foreign = [f"msp_leiden_r{r}" for r in args.resolutions if r in (1.0, 2.0)]

    todo = []
    for ln in zoomed:
        d = lineage_dir(out, ln["name"])
        if contract_done(d) and not args.force:
            print(f"== [{ln['name']}] already done — skipping (resume)", flush=True)
            results[ln["name"]] = load_result(d)
        else:
            todo.append(ln)

    common = dict(batch_col=batch_col, species=species, h5ad_path=args.h5ad, resolutions=args.resolutions,
                  n_top_genes=args.n_top_genes, n_pcs=args.n_pcs, n_neighbors=args.n_neighbors,
                  harmony_kwargs=harmony_kwargs, keys_for_foreign=keys_for_foreign, language=args.language,
                  model=args.model, effort=args.effort, max_turns=args.max_turns)
    if len(todo) == 1 or os.environ.get("ZMIP_PARALLEL", "").strip() == "1":
        for ln in todo:  # in-process, exactly the old sequential path
            sub = subset_for(ad, ln["coarse_labels"], args.coarse_col, args.fine_col)
            results[ln["name"]] = run_lineage(sub, ln["name"], ln["coarse_labels"], all_labels, markers, out, **common)
            del sub
    elif todo:
        child_args = ["--h5ad", args.h5ad, "--batch-col", batch_col, "--resolutions", *map(str, args.resolutions),
                      "--n-top-genes", str(args.n_top_genes), "--n-pcs", str(args.n_pcs),
                      "--n-neighbors", str(args.n_neighbors), "--language", args.language, "--max-turns", str(args.max_turns)]
        for kv in args.harmony:
            child_args += ["--harmony", kv]
        if species:
            child_args += ["--species", species]
        if args.model:
            child_args += ["--model", args.model]
        if args.effort:
            child_args += ["--effort", args.effort]
        results.update(run_lineages_parallel(ad, todo, all_labels, out, child_args,
                                             coarse_col=args.coarse_col, fine_col=args.fine_col))

    merge_back(ad, plan, results, out, coarse_col=args.coarse_col, fine_col=args.fine_col, with_report=True)
    print(f"== report: {os.path.join(out, 'report.html')}", flush=True)
