"""
zmip.merge — fold every lineage's zoom-in result back into the global object.

Global columns (survivors only, in annotated_zmip.h5ad):
  zmip_lineage      the plan's lineage name (skipped lineages included)
  zmip_cluster      "<lineage>:<cluster id>" of the zoomed subset; NaN for skipped
  zmip_ann_coarse   zoomed coarse label; reassigned cells carry their new
                    lineage's coarse label; skipped lineages keep msp_ann_coarse
  zmip_ann_fine     zoomed fine label; skipped lineages keep msp_ann_fine
  zmip_reassigned_from  the lineage a reassigned cell came from (else NaN)
msp_ann_* columns are kept verbatim for the audit trail. No global
re-embedding here (that is the next round's job): figures use the msp UMAP
the input already carries.
Archives: zmip_removed.csv / zmip_reassigned.csv (every cell, with sources).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from msp.annotate import _palette
from msp.plots import UMAP_DPI, save_single_umap, umap_axes


def merge_back(ad, plan, results, outdir, coarse_col="msp_ann_coarse", fine_col="msp_ann_fine"):
    """results: {lineage_name: {"dir": path, "removed": df, "reassigned": df}}
    for every ZOOMED lineage (the lineage dir holds annotated.h5ad whose obs
    carries msp_ann_cluster/coarse/fine for the survivors)."""
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    label_to_lineage = {lab: ln["name"] for ln in plan["lineages"] for lab in ln["coarse_labels"]}
    obs = ad.obs
    coarse0 = obs[coarse_col].astype(str)
    zl = coarse0.map(label_to_lineage).astype(object)
    zcoarse = coarse0.astype(object).copy()
    zfine = obs[fine_col].astype(str).astype(object).copy()
    zcluster = pd.Series(np.nan, index=obs.index, dtype=object)
    zfrom = pd.Series(np.nan, index=obs.index, dtype=object)
    removed = np.zeros(ad.n_obs, dtype=bool)
    rm_frames, ra_frames = [], []

    for name, r in results.items():
        sub = sc.read_h5ad(os.path.join(r["dir"], "annotated.h5ad"), backed="r")
        so = sub.obs
        idx = so.index.intersection(obs.index)
        zcluster.loc[idx] = name + ":" + so.loc[idx, "msp_ann_cluster"].astype(str)
        zcoarse.loc[idx] = so.loc[idx, "msp_ann_coarse"].astype(str)
        zfine.loc[idx] = so.loc[idx, "msp_ann_fine"].astype(str)
        if "zmip_reassigned_to" in so:
            ra = so.loc[idx, "zmip_reassigned_to"]
            ra_idx = ra.dropna().index
            zfrom.loc[ra_idx] = name
            zl.loc[ra_idx] = ra.loc[ra_idx].astype(str).map(label_to_lineage).values
        sub.file.close()
        rm = r["removed"]
        removed |= obs.index.isin(rm["cell"])
        rm_frames.append(rm)
        ra_frames.append(r["reassigned"])

    obs["zmip_lineage"] = zl.astype("category")
    obs["zmip_cluster"] = zcluster.astype("category")
    obs["zmip_ann_coarse"] = zcoarse.astype("category")
    obs["zmip_ann_fine"] = zfine.astype("category")
    obs["zmip_reassigned_from"] = zfrom.astype("category")
    obs["zmip_action"] = pd.Categorical(np.where(removed, "remove", "keep"), categories=["keep", "remove"])

    rm_all = pd.concat(rm_frames, ignore_index=True) if rm_frames else pd.DataFrame(
        columns=["cell", "lineage", "cluster", "preannotation", "annotate_remove", "remove_reason"])
    ra_all = pd.concat(ra_frames, ignore_index=True) if ra_frames else pd.DataFrame(
        columns=["cell", "lineage", "cluster", "reassign_to", "fine_label"])
    rm_all.to_csv(os.path.join(outdir, "zmip_removed.csv"), index=False)
    ra_all.to_csv(os.path.join(outdir, "zmip_reassigned.csv"), index=False)

    kept = ad[~removed].copy()
    for col in ("zmip_ann_coarse", "zmip_ann_fine", "zmip_lineage"):
        kept.obs[col] = kept.obs[col].cat.remove_unused_categories()
    _figures(ad, kept, figdir)
    tmp = os.path.join(outdir, "annotated_zmip.tmp.h5ad")
    kept.write_h5ad(tmp)
    os.replace(tmp, os.path.join(outdir, "annotated_zmip.h5ad"))
    print(f"== merged: removed {int(removed.sum())}, reassigned {len(ra_all)}, "
          f"annotated_zmip.h5ad keeps {kept.n_obs}/{ad.n_obs}", flush=True)
    return kept, rm_all, ra_all


def _figures(ad, kept, figdir):
    for col, fname in (("zmip_ann_coarse", "zmip_umap_coarse.png"), ("zmip_lineage", "zmip_umap_lineage.png")):
        pal = _palette(kept, col)
        if pal:
            kept.uns[f"{col}_colors"] = pal
        n = kept.obs[col].nunique()
        save_single_umap(kept, col, os.path.join(figdir, fname), repel=True,
                         repel_fontsize=9 if n > 12 else 11, figsize=(9, 9) if n > 12 else None)
    # fine labels are far too many for on-data text (dozens across lineages):
    # number them (ordered by coarse label, then size) and ship the legend
    # as zmip_fine_legend.csv for the report to render next to the panel
    legend = (kept.obs.groupby(["zmip_ann_coarse", "zmip_ann_fine"], observed=True).size().rename("n_cells")
              .reset_index().sort_values(["zmip_ann_coarse", "n_cells"], ascending=[True, False]))
    legend.insert(0, "id", range(1, len(legend) + 1))
    legend.to_csv(os.path.join(os.path.dirname(figdir), "zmip_fine_legend.csv"), index=False)
    id_of = dict(zip(legend["zmip_ann_fine"], legend["id"].astype(str)))
    kept.obs["_fine_id"] = pd.Categorical(kept.obs["zmip_ann_fine"].astype(str).map(id_of),
                                          categories=legend["id"].astype(str).tolist())
    pal = _palette(kept, "zmip_ann_fine")
    if pal:  # same colour per fine label as its id
        by_fine = dict(zip(kept.obs["zmip_ann_fine"].cat.categories, pal))
        kept.uns["_fine_id_colors"] = [by_fine[f] for f in legend["zmip_ann_fine"]]
    save_single_umap(kept, "_fine_id", os.path.join(figdir, "zmip_umap_fine.png"), repel=True,
                     repel_fontsize=9, figsize=(9, 9), title="zmip_ann_fine (ids: see legend)")
    del kept.obs["_fine_id"]
    kept.uns.pop("_fine_id_colors", None)
    xy = np.asarray(ad.obsm["X_umap"])
    base = 120000 / ad.n_obs
    act = ad.obs["zmip_action"].astype(str).values
    fig, ax = umap_axes(ad)
    for name, color, size in (("keep", "#d3d3d3", base), ("remove", "#c0392b", 1.5 * base)):
        m = act == name
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=size, c=color, linewidths=0, label=f"{name} (n={int(m.sum())})")
    ax.set_title("UMAP: cells removed at zoom-in")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.savefig(os.path.join(figdir, "zmip_umap_removed.png"), dpi=UMAP_DPI)
    plt.close(fig)

    frm = ad.obs["zmip_reassigned_from"]
    fig, ax = umap_axes(ad)
    ax.scatter(xy[:, 0], xy[:, 1], s=base, c="#d3d3d3", linewidths=0, label=f"unchanged (n={int(frm.isna().sum())})")
    if frm.notna().any():
        tgt = ad.obs["zmip_ann_coarse"].astype(str)
        moved = frm.notna().values
        pairs = (frm.astype(str) + " → " + tgt).where(moved)
        cmap = plt.get_cmap("tab10")
        for i, p in enumerate(pd.unique(pairs.dropna())):
            m = (pairs == p).values
            ax.scatter(xy[m, 0], xy[m, 1], s=2 * base, c=[cmap(i % 10)], linewidths=0,
                       label=f"{p} (n={int(m.sum())})")
    ax.set_title("UMAP: cells reassigned between lineages")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    fig.savefig(os.path.join(figdir, "zmip_umap_reassigned.png"), dpi=UMAP_DPI)
    plt.close(fig)
