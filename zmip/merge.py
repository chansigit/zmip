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

import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from msp.plots import UMAP_DPI, save_single_umap, umap_axes

from . import publication
from .msp_compat import palette

log = logging.getLogger(__name__)


def _validate_partition(name, expected, survivors, removed, reassigned):
    """A lineage's survivors and removals must partition its original cells."""
    kept_ids = survivors.index
    removed_ids = pd.Index(removed["cell"])
    reassigned_ids = pd.Index(reassigned["cell"])
    problems = []

    def report(label, ids):
        if len(ids):
            problems.append(f"{label}: {len(ids)} cell(s), examples {ids[:5].tolist()}")

    for label, ids in (("survivors", kept_ids), ("removed", removed_ids), ("reassigned", reassigned_ids)):
        report(f"duplicate {label}", ids[ids.duplicated()].unique())
        report(f"{label} outside the input lineage", ids.difference(expected))
    report("cells both kept and removed", kept_ids.intersection(removed_ids))
    report("input cells missing from both survivors and removals", expected.difference(kept_ids.union(removed_ids)))
    report("reassigned cells absent from survivors", reassigned_ids.difference(kept_ids))
    for label, table in (("removed", removed), ("reassigned", reassigned)):
        wrong_source = table["lineage"].ne(name) | table["lineage"].isna()
        report(f"{label} records with the wrong source lineage", pd.Index(table.loc[wrong_source, "cell"]))
    if problems:
        raise ValueError(f"lineage {name!r} has inconsistent cell coverage:\n- " + "\n- ".join(problems))


def _validate_annotation(name, survivors, reassigned, own_labels, all_labels):
    """Audit CSVs and H5AD must describe exactly the same reassignment decisions."""
    required = {"msp_ann_cluster", "msp_ann_coarse", "msp_ann_fine"}
    if not required.issubset(survivors) or not {"cell", "cluster", "reassign_to", "fine_label"}.issubset(reassigned):
        raise ValueError(f"lineage {name!r}: missing annotation/audit columns")
    # The H5AD returns these as categoricals with differing category sets, which pandas
    # refuses to compare. Work on plain text throughout; missing values stay missing.
    survivors = survivors.copy()
    for col in (*required, "zmip_reassigned_to"):
        if col in survivors:
            survivors[col] = survivors[col].astype(object)
    reassigned = reassigned.astype(object)
    problems = []

    def report(label, mask):
        ids = survivors.index[mask]
        if len(ids):
            problems.append(f"{label}: {len(ids)} cell(s), examples {ids[:5].tolist()}")

    target = survivors.get("zmip_reassigned_to", pd.Series(None, index=survivors.index, dtype=object))
    moved = target.notna()
    audited = survivors.index.isin(reassigned["cell"])
    report("reassignment membership differs between H5AD and CSV", moved != audited)
    report("invalid or same-lineage reassignment target", moved & ~target.isin(set(all_labels) - set(own_labels)))
    report("coarse label differs from reassignment target", moved & survivors["msp_ann_coarse"].ne(target))
    report("non-reassigned cell has a foreign coarse label", ~moved & ~survivors["msp_ann_coarse"].isin(own_labels))
    for col in required:
        report(f"empty {col}", survivors[col].isna() | survivors[col].astype(str).str.strip().eq(""))
    audit = reassigned.set_index("cell").reindex(survivors.index)
    report("CSV target differs from H5AD", moved & audited & audit["reassign_to"].ne(target))
    report("CSV fine label differs from H5AD", moved & audited & audit["fine_label"].ne(survivors["msp_ann_fine"]))
    # Audit cluster IDs are pre-merge; msp_ann_cluster can contain a '+'-joined component.
    cluster_matches = pd.Series(
        [
            str(c) in str(merged).split("+")
            for c, merged in zip(audit["cluster"], survivors["msp_ann_cluster"], strict=False)
        ],
        index=survivors.index,
    )
    report("CSV cluster is not in the H5AD merged cluster", moved & audited & ~cluster_matches)
    if problems:
        raise ValueError(f"lineage {name!r} has inconsistent annotation decisions:\n- " + "\n- ".join(problems))


def merge_back(ad, plan, results, outdir, coarse_col="msp_ann_coarse", fine_col="msp_ann_fine", *, with_report=False):
    """results: {lineage_name: {"dir": path, "removed": df, "reassigned": df}}
    for every ZOOMED lineage (the lineage dir holds annotated.h5ad whose obs
    carries msp_ann_cluster/coarse/fine for the survivors)."""
    if not ad.obs_names.is_unique:
        raise ValueError("input cell identifiers must be unique before merging")
    names = [ln["name"] for ln in plan["lineages"]]
    labels = [lab for ln in plan["lineages"] for lab in ln["coarse_labels"]]
    if len(set(names)) != len(names) or len(set(labels)) != len(labels):
        raise ValueError("plan must assign unique lineage names and each coarse label exactly once")
    zoomed = {ln["name"] for ln in plan["lineages"] if ln["zoom"]}
    if set(results) != zoomed:
        raise ValueError(
            f"lineage results do not match the zoom plan: missing {sorted(zoomed - set(results))}, "
            f"unexpected {sorted(set(results) - zoomed)}"
        )
    label_to_lineage = {lab: ln["name"] for ln in plan["lineages"] for lab in ln["coarse_labels"]}
    obs = ad.obs
    coarse0 = obs[coarse_col].astype(str)
    unassigned = set(coarse0) - set(label_to_lineage)
    if unassigned:
        raise ValueError(f"input coarse labels missing from plan: {sorted(unassigned)}")
    zl = coarse0.map(label_to_lineage).astype(object)
    zcoarse = coarse0.astype(object).copy()
    zfine = obs[fine_col].astype(str).astype(object).copy()
    zcluster = pd.Series(np.nan, index=obs.index, dtype=object)
    zfrom = pd.Series(np.nan, index=obs.index, dtype=object)
    removed = np.zeros(ad.n_obs, dtype=bool)
    rm_frames, ra_frames = [], []

    for name, r in results.items():
        sub = sc.read_h5ad(os.path.join(r["dir"], "annotated.h5ad"), backed="r")
        try:
            # Only metadata is needed; release the backed file even on failure.
            so = sub.obs.copy()
        finally:
            sub.file.close()
        expected = obs.index[coarse0.map(label_to_lineage).eq(name)]
        _validate_partition(name, expected, so, r["removed"], r["reassigned"])
        own_labels = [lab for lab, owner in label_to_lineage.items() if owner == name]
        _validate_annotation(name, so, r["reassigned"], own_labels, label_to_lineage)
        idx = so.index
        zcluster.loc[idx] = name + ":" + so.loc[idx, "msp_ann_cluster"].astype(str)
        zcoarse.loc[idx] = so.loc[idx, "msp_ann_coarse"].astype(str)
        zfine.loc[idx] = so.loc[idx, "msp_ann_fine"].astype(str)
        if "zmip_reassigned_to" in so:
            ra = so.loc[idx, "zmip_reassigned_to"]
            ra_idx = ra.dropna().index
            zfrom.loc[ra_idx] = name
            zl.loc[ra_idx] = ra.loc[ra_idx].astype(str).map(label_to_lineage).values
        rm = r["removed"]
        removed |= obs.index.isin(rm["cell"])
        rm_frames.append(rm)
        ra_frames.append(r["reassigned"])

    # Publish global annotations only after every lineage passes validation.
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    obs["zmip_lineage"] = zl.astype("category")
    obs["zmip_cluster"] = zcluster.astype("category")
    obs["zmip_ann_coarse"] = zcoarse.astype("category")
    obs["zmip_ann_fine"] = zfine.astype("category")
    obs["zmip_reassigned_from"] = zfrom.astype("category")
    obs["zmip_action"] = pd.Categorical(np.where(removed, "remove", "keep"), categories=["keep", "remove"])

    rm_all = (
        pd.concat(rm_frames, ignore_index=True)
        if rm_frames
        else pd.DataFrame(columns=["cell", "lineage", "cluster", "preannotation", "annotate_remove", "remove_reason"])
    )
    ra_all = (
        pd.concat(ra_frames, ignore_index=True)
        if ra_frames
        else pd.DataFrame(columns=["cell", "lineage", "cluster", "reassign_to", "fine_label"])
    )
    kept = ad[~removed].copy()
    for col in ("zmip_ann_coarse", "zmip_ann_fine", "zmip_lineage"):
        kept.obs[col] = kept.obs[col].cat.remove_unused_categories()
    with publication.staging(outdir) as stage:
        (stage / "figures").mkdir()
        rm_all.to_csv(stage / "zmip_removed.csv", index=False)
        ra_all.to_csv(stage / "zmip_reassigned.csv", index=False)
        _figures(ad, kept, str(stage / "figures"))
        kept.write_h5ad(stage / "annotated_zmip.h5ad")
        # Reopen the actual bytes before replacing any public output.
        disk = sc.read_h5ad(stage / "annotated_zmip.h5ad", backed="r")
        try:
            if not disk.obs_names.equals(kept.obs_names) or disk.shape != kept.shape:
                raise ValueError("written global H5AD does not match the validated survivors")
        finally:
            disk.file.close()
        if with_report:
            from .report import generate_report

            generate_report(outdir, out_html=str(stage / "report.html"), result_dir=str(stage))
            if not (stage / "report.html").is_file():
                raise ValueError("global report was not written")
        publication.publish(outdir, stage)
    log.info(
        f"== merged: removed {int(removed.sum())}, reassigned {len(ra_all)}, "
        f"annotated_zmip.h5ad keeps {kept.n_obs}/{ad.n_obs}"
    )
    return kept, rm_all, ra_all


def _figures(ad, kept, figdir):
    for col, fname in (("zmip_ann_coarse", "zmip_umap_coarse.png"), ("zmip_lineage", "zmip_umap_lineage.png")):
        pal = palette(kept, col)
        if pal:
            kept.uns[f"{col}_colors"] = pal
        n = kept.obs[col].nunique()
        save_single_umap(
            kept,
            col,
            os.path.join(figdir, fname),
            repel=True,
            repel_fontsize=9 if n > 12 else 11,
            figsize=(9, 9) if n > 12 else None,
        )
    # fine labels are far too many for on-data text (dozens across lineages):
    # number them (ordered by coarse label, then size) and ship the legend
    # as zmip_fine_legend.csv for the report to render next to the panel
    legend = (
        kept.obs.groupby(["zmip_ann_coarse", "zmip_ann_fine"], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
        .sort_values(["zmip_ann_coarse", "n_cells"], ascending=[True, False])
    )
    legend.insert(0, "id", range(1, len(legend) + 1))
    legend.to_csv(os.path.join(os.path.dirname(figdir), "zmip_fine_legend.csv"), index=False)
    # Fine names may repeat across independently annotated lineages.
    label_cols = ["zmip_ann_coarse", "zmip_ann_fine"]
    id_of = dict(zip(legend[label_cols].itertuples(index=False, name=None), legend["id"].astype(str), strict=False))
    fine_ids = [id_of[pair] for pair in kept.obs[label_cols].itertuples(index=False, name=None)]
    kept.obs["_fine_id"] = pd.Categorical(fine_ids, categories=legend["id"].astype(str).tolist())
    pal = palette(kept, "zmip_ann_fine")
    if pal:  # same colour per fine label as its id
        by_fine = dict(zip(kept.obs["zmip_ann_fine"].cat.categories, pal, strict=False))
        kept.uns["_fine_id_colors"] = [by_fine[f] for f in legend["zmip_ann_fine"]]
    save_single_umap(
        kept,
        "_fine_id",
        os.path.join(figdir, "zmip_umap_fine.png"),
        repel=True,
        repel_fontsize=9,
        figsize=(9, 9),
        title="zmip_ann_fine (ids: see legend)",
    )
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
            ax.scatter(xy[m, 0], xy[m, 1], s=2 * base, c=[cmap(i % 10)], linewidths=0, label=f"{p} (n={int(m.sum())})")
    ax.set_title("UMAP: cells reassigned between lineages")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    fig.savefig(os.path.join(figdir, "zmip_umap_reassigned.png"), dpi=UMAP_DPI)
    plt.close(fig)
