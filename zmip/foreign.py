"""
zmip.foreign — "foreign lineage signal": how much a cell inside one lineage
subset looks like a different lineage.

Deterministic evidence only — never a verdict. Some lineages are
transcriptionally close (fibroblast vs fibrochondrocyte, monocyte vs
macrophage), so a foreign score is compatible with a doublet, ambient
contamination, a misassigned cluster, OR genuine shared biology; the
per-lineage agent weighs it together with markers, doublet_score and
decontX.

  lineage_markers(ad, lineage_col, outdir)  wilcoxon one-vs-rest on the
        whole dataset at LINEAGE level (the plan's pooling), top genes per
        lineage filtered to be specific (pct_out low) → lineage_markers.csv
  score_foreign(sub, markers, own)          sc.tl.score_genes for every
        OTHER lineage's markers → obs["foreign_<lineage>"], plus per-cluster
        summary tables and qc_umap_foreign_*.png (qc_ prefix so msp's report
        shows them in the QC grid)
"""

import logging
import os

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from msp.plots import save_single_umap, slug

log = logging.getLogger(__name__)

TOP_N = 30
MAX_PCT_OUT = 0.35  # a marker must be expressed in <35% of cells outside its lineage
MARKER_COLUMNS = ["lineage", "rank", "gene", "logFC", "pct_in", "pct_out"]


def lineage_markers(ad, lineage_col, outdir, top_n=TOP_N):
    """One marker list per lineage (all lineages, zoomed or not — a skipped
    tiny lineage is still something a zoomed one can be contaminated by)."""
    groups = list(ad.obs[lineage_col].astype(str).unique())
    sizes = ad.obs[lineage_col].astype(str).value_counts()
    eligible = []
    for g in groups:
        n_group, n_rest = int(sizes[g]), ad.n_obs - int(sizes[g])
        if min(n_group, n_rest) < 2:
            log.warning(
                f"== lineage {g!r}: cannot estimate markers with {n_group} lineage cell(s) and "
                f"{n_rest} reference cell(s); need at least 2 each — no foreign score for it"
            )
        else:
            eligible.append(g)
    # Exclude tiny groups from testing, not from the other groups' reference cells.
    if eligible:
        # Ranking needs X and the grouping only, not raw/counts, graphs or embeddings.
        tmp = AnnData(X=ad.X, obs=ad.obs[[lineage_col]].copy(), var=pd.DataFrame(index=ad.var_names.copy()))
        if "log1p" in ad.uns:
            tmp.uns["log1p"] = dict(ad.uns["log1p"])
        sc.tl.rank_genes_groups(tmp, lineage_col, groups=eligible, method="wilcoxon", use_raw=False, pts=True)
    rows = []
    for g in eligible:
        df = sc.get.rank_genes_groups_df(tmp, group=g)
        df = df.rename(columns={"pct_nz_group": "pct_in", "pct_nz_reference": "pct_out"})
        df = df[(df["logfoldchanges"] > 0) & (df["pct_out"] < MAX_PCT_OUT)].head(top_n)
        for rank, r in enumerate(df.itertuples(index=False), 1):
            rows.append(
                {
                    "lineage": g,
                    "rank": rank,
                    "gene": r.names,
                    "logFC": float(r.logfoldchanges),
                    "pct_in": float(r.pct_in),
                    "pct_out": float(r.pct_out),
                }
            )
    # explicit columns: a run where no gene passes the pct_out filter for any
    # lineage (tiny/synthetic data, or one lineage swamping the rest) must
    # still yield an empty-but-well-formed table, not a KeyError below
    out = pd.DataFrame(rows, columns=MARKER_COLUMNS)
    out.to_csv(os.path.join(outdir, "lineage_markers.csv"), index=False)
    for g in eligible:
        if not (out["lineage"] == g).any():
            log.info(f"== no lineage markers pass pct_out<{MAX_PCT_OUT} for {g!r} — no foreign score for it")
    return {g: out.loc[out["lineage"] == g, "gene"].tolist() for g in groups}


def score_foreign(sub, markers, own_lineage, cluster_keys, outdir, figdir):
    """obs["foreign_<lineage>"] for every other lineage; per-cluster summary
    per clustering key (mean, p90, share of cells above the subset's own
    99th percentile of that score — i.e. clearly outlying cells); one UMAP
    per foreign lineage. Returns the list of score columns."""
    cols = []
    for lin, genes in markers.items():
        if lin == own_lineage:
            continue
        genes = [g for g in genes if g in sub.var_names]
        col = f"foreign_{slug(lin)}"
        if len(genes) < 5:
            log.info(f"== foreign {lin}: only {len(genes)} markers present, skipped")
            continue
        sc.tl.score_genes(sub, genes, score_name=col, random_state=0)
        cols.append(col)
        save_single_umap(sub, col, os.path.join(figdir, f"qc_umap_{col}.png"), color_map="viridis")
    if not cols:
        return cols
    for key in cluster_keys:
        g = sub.obs.groupby(key, observed=True)
        parts = []
        for col in cols:
            hi = np.quantile(sub.obs[col], 0.99)
            parts.append(
                pd.DataFrame(
                    {
                        f"{col}_mean": g[col].mean().round(3),
                        f"{col}_p90": g[col].quantile(0.9).round(3),
                        f"{col}_frac_top1pct": g[col]
                        .apply(lambda s, threshold=hi: float((s > threshold).mean()))
                        .round(3),
                    }
                )
            )
        tab = pd.concat(parts, axis=1)
        tab.insert(0, "n_cells", g.size())
        tab.to_csv(os.path.join(outdir, f"foreign_signal_{key}.csv"))
    return cols
