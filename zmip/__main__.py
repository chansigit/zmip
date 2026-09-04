"""python -m zmip: zoom-in pass over an msp annotated.h5ad.

  plan      agent groups coarse labels into UMAP-connected lineages, picks
            which to zoom (>= --min-cells)                 → zmip_plan.json
  markers   lineage-level marker lists for foreign-lineage scores
                                                           → lineage_markers.csv
  per lineage (sequential): subset → msp.integrate_adata (re-embed) →
            foreign scores → annotation agent (refine / remove / reassign /
            recluster) → <lineage>/{annotation_proposal.json, annotated.h5ad,
            report.html}
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

from msp.integrate import integrate_adata
from msp.plots import save_single_umap, slug

from .annotate import PREV_SUFFIX, PREVIOUS_COLS, annotate_lineage
from .foreign import lineage_markers, score_foreign
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

for ln in zoomed:
    name, labels = ln["name"], ln["coarse_labels"]
    d = os.path.join(out, slug(name))
    contract = [os.path.join(d, f) for f in ("annotation_proposal.json", "annotated.h5ad", "report.html")]
    if all(os.path.exists(p) for p in contract) and not args.force:
        print(f"== [{name}] already done — skipping (resume)", flush=True)
        results[name] = {"dir": d, "removed": pd.read_csv(os.path.join(d, "annotation_removed.csv")),
                         "reassigned": pd.read_csv(os.path.join(d, "annotation_reassigned.csv"))}
        continue
    sub = ad[ad.obs[args.coarse_col].astype(str).isin(labels).values].copy()
    for c in PREVIOUS_COLS:
        src = {"msp_ann_coarse": args.coarse_col, "msp_ann_fine": args.fine_col}[c]
        sub.obs[c + PREV_SUFFIX] = sub.obs[src].astype(str).astype("category")
    del sub.obs["_zmip_lineage"]
    print(f"== [{name}] re-embedding {sub.n_obs} cells", flush=True)
    integrate_adata(sub, batch_col, d, species=species, resolutions=tuple(args.resolutions),
                    n_top_genes=args.n_top_genes, n_pcs=args.n_pcs, n_neighbors=args.n_neighbors,
                    harmony_kwargs=harmony_kwargs, inputs=[args.h5ad],
                    meta_extra={"zmip_lineage": name, "zmip_coarse_labels": list(labels)})
    figdir = os.path.join(d, "figures")
    print(f"== [{name}] foreign-lineage scores", flush=True)
    foreign_cols = score_foreign(sub, markers, name, keys_for_foreign, d, figdir)
    for c in PREVIOUS_COLS:
        col = c + PREV_SUFFIX
        n = sub.obs[col].nunique()
        save_single_umap(sub, col, os.path.join(figdir, f"umap_{col}.png"), repel=True,
                         repel_fontsize=8 if n > 15 else 11, figsize=(9, 9) if n > 15 else None)
    proposal, rm, ra = annotate_lineage(sub, d, name, labels, sorted(all_labels - set(labels)), foreign_cols,
                                        species=species, language=args.language, model=args.model,
                                        effort=args.effort, max_turns=args.max_turns)
    results[name] = {"dir": d, "removed": rm, "reassigned": ra}
    del sub

merge_back(ad, plan, results, out, coarse_col=args.coarse_col, fine_col=args.fine_col)
print(f"== report: {generate_report(out)}", flush=True)
