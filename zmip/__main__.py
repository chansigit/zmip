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

Re-running resumes: the plan is reused, lineages whose contract files exist
are skipped; --force redoes everything. One lineage (or none above the
threshold) → nothing is zoomed and annotated_zmip.h5ad carries the msp
labels unchanged.
"""

import argparse
import os
import sys

import pandas as pd
import scanpy as sc

from msp.plots import slug

from .foreign import lineage_markers
from .lineage import contract_done, lineage_dir, load_result, run_lineage, run_lineages_parallel, subset_for
from .merge import merge_back
from .plan import DEFAULT_MIN_CELLS, plan_lineages
from .report import generate_report
from msp.report import write_report_context

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
parser.add_argument("--resolutions", type=float, nargs="+", default=[0.3, 1.0, 2.0])
parser.add_argument("--n-top-genes", type=int, default=2000)
parser.add_argument("--n-pcs", type=int, default=50)
parser.add_argument("--n-neighbors", type=int, default=15)
parser.add_argument("--harmony", action="append", default=[], metavar="KEY=VALUE",
                    help="harmonypy override for the per-lineage re-embedding, repeatable")
parser.add_argument("--language", default="English")
parser.add_argument("--model", default=None)
parser.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"])
parser.add_argument("--max-turns", type=int, default=200)
parser.add_argument("--report-context", default=None, metavar="TEXT",
                    help='where this run sits, for report titles (e.g. "round 2 · fu2022-meniscus")')
parser.add_argument("--force", action="store_true")
args = parser.parse_args()


def _kv(items):
    def conv(v):
        for cast in (int, float):
            try:
                return cast(v)
            except ValueError:
                pass
        return v
    out = {}
    for it in items:
        if "=" not in it:
            sys.exit(f"--harmony expects KEY=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = [conv(x) for x in v.split(",")] if "," in v else conv(v)
    return out


out = os.path.abspath(args.outdir)
os.makedirs(out, exist_ok=True)
write_report_context(out, args.report_context)
ad = sc.read_h5ad(args.h5ad)
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
                     model=args.model, effort=args.effort)
label_to_lineage = {lab: ln["name"] for ln in plan["lineages"] for lab in ln["coarse_labels"]}
ad.obs["_zmip_lineage"] = ad.obs[args.coarse_col].astype(str).map(label_to_lineage).astype("category")
for ln in plan["lineages"]:
    print(f"== lineage {ln['name']}: {ln['coarse_labels']} n={ln['n_cells']} zoom={ln['zoom']}", flush=True)

markers_p = os.path.join(out, "lineage_markers.csv")
mk = None
if os.path.exists(markers_p) and not args.force:
    try:
        mk = pd.read_csv(markers_p)
    except pd.errors.EmptyDataError:  # a half-written/empty file from an interrupted run: recompute
        mk = None
if mk is not None:
    markers = {g: mk.loc[mk["lineage"] == g, "gene"].tolist() for g in mk["lineage"].unique()}
else:
    print("== lineage-level markers (for foreign-lineage scores)", flush=True)
    markers = lineage_markers(ad, "_zmip_lineage", out)

all_labels = set(label_to_lineage)
zoomed = [ln for ln in plan["lineages"] if ln["zoom"]]
if not zoomed:
    print("== no lineage reaches min_cells — nothing to zoom; passing msp labels through", flush=True)
results = {}
harmony_kwargs = _kv(args.harmony)
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

merge_back(ad, plan, results, out, coarse_col=args.coarse_col, fine_col=args.fine_col)
print(f"== report: {generate_report(out)}", flush=True)
