"""
zmip.annotate — the per-lineage agent: refine annotation inside one
re-embedded lineage subset, clean noise, hand misassigned cells to the
lineage they belong to.

Runs on the lineage directory msp.integrate_adata just produced (same
artifacts as an msp run: deg_global_*/deg_local_* at r1.0/r2.0, cluster
QC, cell-level outliers, standissect fragments, preannotation_removal.csv,
figures) plus zmip's foreign-lineage scores. Base clustering is
msp_leiden_r2.0 of the SUBSET; the agent may recluster (subcluster tool,
ids like "5,0") and every tool + the submission follow the refined ids.

Per cluster the agent answers a fixed chain — distinctness vs r1.0 parent /
siblings; identity (coarse label within this lineage, fine label); foreign
signal vs genuine biology; merge — and chooses ONE action:
  keep      coarse_label must be one of this lineage's coarse labels
  remove    doublet / low-quality / ambient / stress / batch / other
  reassign  the cluster is another lineage's cell type: reassign_to = a
            coarse label of a DIFFERENT lineage (relabel only — the cells
            are not re-embedded there this round)

Host-side validation mirrors msp.annotate (coverage of the CURRENT
clustering, union-find over merge_target, label hierarchy) plus the
reassign rules. Outputs in the lineage dir: annotation_proposal.json (msp
schema + reassign_to, so msp's report section renders it),
annotation_removed.csv, annotation_reassigned.csv, annotation UMAPs,
annotated.h5ad (survivors incl. reassigned, with msp_ann_* columns holding
this round's labels), report.html.
"""

import asyncio
import json
import os

import numpy as np
import pandas as pd
import scanpy as sc

from msp.annotate import (
    BASE_KEY, CONFIDENCES, PARENT_KEY, REMOVE_REASONS, _components, _load_paga_neighbors,
    _plot, _prior_label_columns,
)
from msp.inspect import (
    _cluster_order, _deg_table, _file_inventory, _gene_table, _load_removal_mask,
    _stability_table, _subcluster_once,
)
from msp.report import generate_report
from msp.agent_util import run_query

REMOVE_BUDGET = 0.10  # agent-removed share of a lineage above which finalize asks for a second look

PREVIOUS_COLS = ("msp_ann_coarse", "msp_ann_fine")  # last round's labels, carried on the subset as
PREV_SUFFIX = "_prev"                                 # msp_ann_coarse_prev / msp_ann_fine_prev


# ---------------------------------------------------------------- context

def _context(ad, key, cluster, batch_col, prior_cols, paga, pre_removed, foreign_cols, lineage_labels):
    lab = ad.obs[key].astype(str)
    m = (lab == cluster).values
    if not m.any():
        return f"unknown cluster {cluster!r}"
    sub = ad.obs.loc[m]
    n = int(m.sum())
    lines = [f"cluster {cluster} ({key}): n={n} cells, {int(pre_removed[m].sum())} "
             f"({100 * pre_removed[m].mean():.1f}%) already slated for removal (subset preannotation filtering)"]
    if key != BASE_KEY:
        base = sub[BASE_KEY].astype(str).value_counts()
        lines.append(f"  {BASE_KEY} composition: " + ", ".join(f"{i}:{v}" for i, v in base.head(5).items()))
    if PARENT_KEY in ad.obs:
        par = sub[PARENT_KEY].astype(str).value_counts()
        lines.append(f"  {PARENT_KEY} parent composition: " +
                     ", ".join(f"{i}:{v} ({100 * v / n:.0f}%)" for i, v in par.head(5).items()))
        main_parent = par.index[0]
        sib = ad.obs.loc[(ad.obs[PARENT_KEY].astype(str) == main_parent).values, key].astype(str).value_counts()
        sib = sib.drop(cluster, errors="ignore")
        lines.append(f"  siblings under parent {main_parent} ({key}): " +
                     (", ".join(f"{i}:{v}" for i, v in sib.items()) if len(sib) else "none"))
    if key == BASE_KEY and cluster in paga:
        lines.append(f"  PAGA nearest neighbours ({BASE_KEY}): {', '.join(paga[cluster])}")
    vc = sub[batch_col].value_counts(normalize=True)
    lines.append(f"  samples: {sub[batch_col].nunique()}/{ad.obs[batch_col].nunique()} present, "
                 f"dominant sample share {vc.iloc[0]:.2f} ({vc.index[0]})")
    qc = [c for c in ("doublet_score", "decontX_contamination", "pct_counts_mt", "n_genes_by_counts",
                      "total_counts", "dissociation_score") if c in ad.obs]
    lines.append("  QC medians: " + ", ".join(f"{c}={sub[c].median():.3g}" for c in qc))
    if foreign_cols:
        lines.append("  foreign-lineage scores (mean | p90 | share above subset 99th pct) — evidence, not verdict:")
        for c in foreign_cols:
            hi = np.quantile(ad.obs[c], 0.99)
            lines.append(f"    {c[len('foreign_'):]}: {sub[c].mean():.3f} | {sub[c].quantile(0.9):.3f} | "
                         f"{(sub[c] > hi).mean():.2f}")
    prev = [c + PREV_SUFFIX for c in PREVIOUS_COLS if c + PREV_SUFFIX in ad.obs]
    for c in prev:
        cc = sub[c].astype(str).value_counts(normalize=True)
        lines.append(f"  previous round {c[:-len(PREV_SUFFIX)]}: " +
                     ", ".join(f"{i}:{v:.2f}" for i, v in cc.head(5).items()))
    lines.append(f"  this lineage's coarse labels (allowed for keep): {lineage_labels}")
    if prior_cols:
        lines.append("  prior label compositions (reference only, NOT ground truth; top 5 per column):")
        for c in prior_cols:
            vals = sub[c].dropna().astype(str)
            vals = vals[~vals.str.lower().isin(["nan", "none", ""])]
            if vals.empty:
                continue
            cc = vals.value_counts(normalize=True)
            lines.append(f"    {c}: " + ", ".join(f"{i}:{v:.2f}" for i, v in cc.head(5).items()))
    return "\n".join(lines)


# ---------------------------------------------------------------- schema / validation

_CLUSTER_SCHEMA_DOC = """{
  "cluster_id": "<current cluster id (subcluster ids like "5,0" once you split)>",
  "coarse_label": "<keep: one of THIS lineage's coarse labels; reassign: equal to reassign_to; remove: descriptive>",
  "fine_label": "<subtype label in English, e.g. 'CTHRC1+ matrix fibroblast'; for removed clusters what it is>",
  "merge_target": null | "<another current cluster id this one is part of>",
  "action": "keep" | "remove" | "reassign",
  "remove_reason": null | "doublet" | "low-quality" | "ambient" | "stress" | "batch" | "other",
  "reassign_to": null | "<a coarse label belonging to a DIFFERENT lineage>",
  "confidence": "high" | "medium" | "low",
  "evidence": {
    "distinctness": "<step 1: distinct from r1.0 parent / siblings, or a splinter?>",
    "markers": "<step 2: positive markers verified with check_genes/check_deg>",
    "foreign": "<step 3: foreign-lineage scores — doublet / ambient / misassignment / shared biology?>",
    "merge": "<step 4: why merge or keep separate>"
  },
  "rationale": "<one or two sentences>"
}"""


def _validate_cluster(e, clusters, lineage_labels, other_labels):
    problems = []
    for k in ("cluster_id", "coarse_label", "fine_label", "merge_target", "action", "confidence",
              "evidence", "rationale"):
        if k not in e:
            problems.append(f"missing field {k!r}")
    if problems:
        return problems
    cid = str(e["cluster_id"])
    if cid not in clusters:
        problems.append(f"cluster_id {cid!r} is not a current cluster; current: {clusters}")
    for k in ("coarse_label", "fine_label"):
        if not isinstance(e[k], str) or not e[k].strip():
            problems.append(f"{k} must be a non-empty string")
    act = e["action"]
    if act not in ("keep", "remove", "reassign"):
        problems.append("action must be keep|remove|reassign")
    elif act == "keep" and str(e["coarse_label"]).strip() not in lineage_labels:
        problems.append(f"keep requires coarse_label in this lineage's labels {lineage_labels}; "
                        f"got {e['coarse_label']!r} — if it is really another lineage's cell type, use "
                        f"action=reassign with reassign_to")
    elif act == "remove" and e.get("remove_reason") not in REMOVE_REASONS:
        problems.append(f"remove requires remove_reason in {REMOVE_REASONS}")
    elif act == "reassign":
        tgt = e.get("reassign_to")
        if tgt not in other_labels:
            problems.append(f"reassign_to must be a coarse label of another lineage {sorted(other_labels)}; got {tgt!r}")
        elif str(e["coarse_label"]).strip() != tgt:
            problems.append(f"for reassign, coarse_label must equal reassign_to ({tgt!r})")
    if e["confidence"] not in CONFIDENCES:
        problems.append(f"confidence must be one of {CONFIDENCES}")
    mt = e["merge_target"]
    if mt is not None:
        mt = str(mt)
        if mt not in clusters:
            problems.append(f"merge_target {mt!r} is not a current cluster")
        elif mt == cid:
            problems.append("merge_target cannot be the cluster itself")
    ev = e["evidence"]
    if not isinstance(ev, dict) or not all(k in ev for k in ("distinctness", "markers", "foreign", "merge")):
        problems.append("evidence must be an object with distinctness / markers / foreign / merge")
    return problems


def _validate_final(entries, clusters):
    problems = []
    missing = [c for c in clusters if c not in entries]
    stale = [c for c in entries if c not in clusters]
    if missing or stale:
        if missing:
            problems.append(f"no submission for current clusters {missing}")
        if stale:
            problems.append(f"submissions for clusters that no longer exist (split since): {stale} — resubmit their subclusters")
        return problems
    for c, e in entries.items():
        mt = e.get("merge_target")
        if mt is None:
            continue
        tgt = entries[str(mt)]
        if tgt["action"] != e["action"] or tgt.get("reassign_to") != e.get("reassign_to"):
            problems.append(f"cluster {c} ({e['action']}) merges into {mt} ({tgt['action']}"
                            f"{', → ' + str(tgt.get('reassign_to')) if tgt.get('reassign_to') else ''}) — "
                            "merged clusters must share the same action (and reassign target)")
    comp = _components(entries)
    seen = set()
    for c, members in comp.items():
        key = tuple(members)
        if key in seen or len(members) < 2:
            continue
        seen.add(key)
        live = [m for m in members if entries[m]["action"] != "remove"]
        for field in ("coarse_label", "fine_label"):
            vals = {entries[m][field].strip() for m in live}
            if len(vals) > 1:
                problems.append(f"merged group {'+'.join(members)} disagrees on {field}: "
                                + "; ".join(f"{m}={entries[m][field]!r}" for m in live))
    by_fine = {}
    for c, e in entries.items():
        if e["action"] == "remove":
            continue
        by_fine.setdefault(e["fine_label"].strip(), []).append(c)
    for fine, members in by_fine.items():
        coarse = {entries[m]["coarse_label"].strip() for m in members}
        if len(coarse) > 1:
            problems.append(f"fine label {fine!r} sits under several coarse labels {sorted(coarse)} (clusters {members})")
        comps = {tuple(comp[m]) for m in members}
        if len(comps) > 1:
            problems.append(f"clusters {members} share fine label {fine!r} but are not merged — set merge_target "
                            "between them or give them distinct fine labels")
    return problems


# ---------------------------------------------------------------- apply

def _apply(ad, key, proposal, pre_removed, lineage):
    """msp_ann_* columns on the subset (so msp's plotting/report code renders
    this round), plus the removal / reassignment archives."""
    entries = {str(e["cluster_id"]): e for e in proposal["clusters"]}
    comp = _components(entries)
    lab = ad.obs[key].astype(str)
    ad.obs["msp_ann_cluster"] = lab.map({c: "+".join(v) for c, v in comp.items()}).astype("category")
    ad.obs["msp_ann_coarse"] = lab.map({c: e["coarse_label"].strip() for c, e in entries.items()}).astype("category")
    ad.obs["msp_ann_fine"] = lab.map({c: e["fine_label"].strip() for c, e in entries.items()}).astype("category")
    agent_remove = lab.isin([c for c, e in entries.items() if e["action"] == "remove"]).values
    removed = pre_removed | agent_remove
    reassign = lab.isin([c for c, e in entries.items() if e["action"] == "reassign"]).values & ~removed
    ad.obs["msp_ann_action"] = pd.Categorical(np.where(removed, "remove", "keep"), categories=["keep", "remove"])
    ad.obs["zmip_reassigned_to"] = lab.map({c: e.get("reassign_to") for c, e in entries.items()
                                            if e["action"] == "reassign"}).where(reassign, None)
    rm = pd.DataFrame({"cell": ad.obs_names, "lineage": lineage, "cluster": lab.values,
                       "preannotation": pre_removed, "annotate_remove": agent_remove,
                       "remove_reason": lab.map({c: e.get("remove_reason") for c, e in entries.items()
                                                 if e["action"] == "remove"}).values})
    ra = pd.DataFrame({"cell": ad.obs_names, "lineage": lineage, "cluster": lab.values,
                       "reassign_to": ad.obs["zmip_reassigned_to"].values,
                       "fine_label": ad.obs["msp_ann_fine"].astype(str).values})
    return rm.loc[removed].reset_index(drop=True), ra.loc[reassign].reset_index(drop=True)


# ---------------------------------------------------------------- agent

def _system_prompt(outdir, lineage, lineage_labels, other_labels, clusters, batch_col, species,
                   prior_cols, foreign_cols, language):
    context = (f"Context — species: {species}." if species else "No species context was provided.")
    context += f" Sample/batch column: {batch_col!r}."
    foreign = ", ".join(c[len("foreign_"):] for c in foreign_cols) or "none"
    return f"""You are a single-cell RNA-seq annotation expert doing the ZOOM-IN pass on ONE lineage. \
The working directory holds lineage {lineage!r} (coarse labels {lineage_labels}) re-embedded on its own \
— HVG/PCA/harmony/leiden/UMAP recomputed on just these cells, so substructure invisible in the global \
embedding is now resolvable. The global round's labels are on every cell as msp_ann_coarse_prev / \
msp_ann_fine_prev (previous round, to be improved, not copied). Task: annotate EVERY cluster of the base \
clustering {BASE_KEY} ({len(clusters)} clusters: {clusters}) — refine fine labels, remove noise, hand \
misassigned cells to their real lineage — and submit one JSON per cluster.
{context}
Other lineages' coarse labels (the only valid reassign_to targets): {sorted(other_labels)}.
Foreign-lineage scores available on every cell (obs foreign_<lineage>, sc.tl.score_genes on that \
lineage's top markers computed on the whole dataset): {foreign}. They are EVIDENCE, not verdicts: \
lineages can be transcriptionally close, so a high foreign score is compatible with a doublet (also high \
doublet_score, intermediate profile, two lineages' markers co-expressed), ambient contamination (decontX), \
a misassigned cluster (a clean, coherent population whose markers are simply the other lineage's — \
reassign), or shared biology (keep). Decide from the full picture.
Prior label columns in obs (reference only, never ground truth): {', '.join(prior_cols) or 'none'}.

Reasoning chain per cluster:
1. Distinctness — what does this cluster add over its {PARENT_KEY} parent? check_deg against its siblings; \
real substate (specific positive markers) vs resolution splinter (only depth/QC/cycle/stress genes).
2. Identity — coarse label (one of this lineage's, unless reassigning) and a fine label in English.
3. Foreign signal — read the foreign_* scores together with doublet_score/decontX/markers; conclude \
keep / remove (doublet, low-quality, ambient, stress, batch) / reassign.
4. Merge — same population as another current cluster → merge_target. Merge is explicit: equal fine labels \
must be merged (or made distinct); a merged group shares one coarse + one fine label and one action.
If a cluster is heterogeneous, split it with subcluster (ids like "5,0"); tools and submissions then use \
the refined ids, and any earlier submission for the split parent is discarded — resubmit its parts.

All relevant files (paths relative to the working directory):
{_file_inventory(outdir)}

What they are: deg_global_{{key}}.csv / deg_local_{{key}}.csv (subset DEG at r1.0/r2.0, computed after \
excluding preannotation_removal.csv cells), paga_neighbors_*.csv, cluster_qc_*.csv, \
cell_outlier_summary.csv, stress_clusters.csv, minor_sibling_qc.csv, foreign_signal_{{key}}.csv (per-cluster \
foreign score summary), figures/umap_*.png (subset UMAPs by sample / resolution / previous labels), \
figures/qc_umap_*.png (QC metrics and foreign scores on the subset UMAP), \
figures/umap_preannotation_removal.png (cells already slated for removal in this subset).

Mandatory workflow:
1. TaskCreate one task per base cluster ("annotate cluster <id>") before analysis; TaskUpdate to completed \
only after its submit_cluster succeeded (a split parent's task becomes its subclusters' tasks).
2. Read the figures first (resolution UMAPs, umap_msp_ann_fine_prev, sample mixing, foreign score UMAPs), \
then deg_global_{BASE_KEY}.csv, deg_local_{BASE_KEY}.csv, foreign_signal_{BASE_KEY}.csv once.
3. Per cluster: cluster_context, check_genes (batch dozens of genes), check_deg / check_stability when in \
doubt, then submit_cluster (resubmit to revise; last wins).
4. finalize_annotation when every task is completed; fix what it reports and call again.

Efficiency: parallel Reads; batch genes; don't re-read files.
Principles: labels in English; evidence/rationale in {language}. Weak evidence → low confidence, never a \
forced guess. Removal is for cells, not states: stress / immediate-early / interferon / cycling programs on \
top of a real cell type are biology to LABEL (fine label 'stressed ...'), not to delete, unless QC metrics \
are decisive; a lineage losing more than {int(100 * REMOVE_BUDGET)}% of its cells at this step triggers a \
second look. Be conservative with reassign: only a clean, coherent population with the other lineage's \
markers and low doublet evidence; mixed profiles are doublets, not reassignments."""


async def _run_agent(ad, outdir, lineage, lineage_labels, other_labels, batch_col, species, prior_cols,
                     paga, pre_removed, foreign_cols, other_keys, language, model, effort, max_turns):
    from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, ResultMessage, ToolUseBlock,
                                  create_sdk_mcp_server, tool)

    state = {"key": BASE_KEY, "n_sub": 0}
    entries, holder = {}, {}

    def current():
        return _cluster_order(ad.obs[state["key"]].astype(str))

    @tool("cluster_context", "Non-expression context for one current cluster: size, removal share, r1.0 "
          "parent + siblings, PAGA neighbours, samples, QC medians, foreign-lineage scores, previous-round "
          "labels, prior label compositions.", {"cluster": str})
    async def cluster_context(args):
        return {"content": [{"type": "text", "text": _context(
            ad, state["key"], str(args["cluster"]), batch_col, prior_cols, paga, pre_removed, foreign_cols,
            lineage_labels)}]}

    @tool("check_genes", "Per-cluster mean expression and expressing-cell fraction for the given genes "
          "(case-insensitive), on the current clustering.", {"genes": list})
    async def check_genes(args):
        genes = args["genes"]
        if isinstance(genes, str):
            genes = [g for g in genes.replace(",", " ").split() if g]
        return {"content": [{"type": "text", "text": _gene_table(ad, genes, state["key"])}]}

    @tool("check_deg", "On-demand wilcoxon DEG for one current cluster. reference='rest' (default) or a "
          "comma-separated list of other current cluster ids (e.g. its siblings). Cells already slated for "
          "removal are excluded.", {"cluster": str, "reference": str, "top_n": int})
    async def check_deg(args):
        c = str(args["cluster"])
        cur = current()
        if c not in cur:
            return {"content": [{"type": "text", "text": f"unknown cluster {c!r}; current: {cur}"}], "is_error": True}
        reference = str(args.get("reference") or "rest").strip() or "rest"
        if reference != "rest":
            unknown = [g.strip() for g in reference.split(",") if g.strip() and g.strip() not in cur]
            if unknown:
                return {"content": [{"type": "text", "text": f"unknown reference cluster(s) {unknown}; current: {cur}"}],
                        "is_error": True}
        return {"content": [{"type": "text", "text": _deg_table(
            ad, state["key"], c, reference, int(args.get("top_n") or 20), pre_removed)}]}

    @tool("check_stability", "How one current cluster decomposes across the other leiden resolutions "
          "of this subset (r0.3/r1.0/r2.0).", {"cluster": str})
    async def check_stability(args):
        return {"content": [{"type": "text",
                             "text": _stability_table(ad, str(args["cluster"]), state["key"],
                                                      [k for k in other_keys if k != state["key"]])}]}

    @tool("subcluster", "Split one heterogeneous current cluster with leiden restrict_to at the given "
          'resolution (0.3-1.0 typical). New ids look like "5,0"; tools and submissions follow the refined '
          "clustering; the parent's submission (if any) is discarded.", {"cluster": str, "resolution": float})
    async def subcluster(args):
        c = str(args["cluster"])
        if c not in current():
            return {"content": [{"type": "text", "text": f"unknown cluster {c!r}; current: {current()}"}],
                    "is_error": True}
        new_key = f"zmip_sub{state['n_sub'] + 1}"
        n, text = _subcluster_once(ad, state["key"], c, float(args["resolution"]), new_key, pre_removed)
        if n >= 2:
            state["n_sub"] += 1
            state["key"] = new_key
            if c in entries:
                del entries[c]
                text += f"\n(discarded the earlier submission for {c}; submit its subclusters)"
            text += "\n(working clustering refined; all tools and submissions now use the new ids)"
        return {"content": [{"type": "text", "text": text}]}

    @tool("submit_cluster", "Submit (or resubmit — last wins) ONE current cluster. cluster_json schema:\n"
          + _CLUSTER_SCHEMA_DOC, {"cluster_json": str})
    async def submit_cluster(args):
        try:
            e = json.loads(args["cluster_json"])
        except json.JSONDecodeError as exc:
            return {"content": [{"type": "text", "text": f"JSON parse error: {exc}"}], "is_error": True}
        cur = current()
        problems = _validate_cluster(e, cur, lineage_labels, other_labels)
        if problems:
            return {"content": [{"type": "text", "text": "invalid, fix and resubmit:\n- " + "\n- ".join(problems)}],
                    "is_error": True}
        e["cluster_id"] = str(e["cluster_id"])
        e["merge_target"] = None if e["merge_target"] is None else str(e["merge_target"])
        e.setdefault("remove_reason", None)
        e.setdefault("reassign_to", None)
        entries[e["cluster_id"]] = e
        left = [c for c in cur if c not in entries]
        tag = e["action"] + (f"→{e['reassign_to']}" if e["action"] == "reassign" else "")
        print(f"== [{lineage}] cluster {e['cluster_id']}: {e['coarse_label']} / {e['fine_label']} [{tag}"
              f"{', merge→' + e['merge_target'] if e['merge_target'] else ''}]", flush=True)
        return {"content": [{"type": "text", "text": f"recorded {e['cluster_id']}; {len(entries)}/{len(cur)} "
                             + (f"submitted, remaining: {left}" if left else "submitted — call finalize_annotation")}]}

    @tool("finalize_annotation", "Validate everything together and finish. overall = short assessment "
          "of this lineage.", {"overall": str})
    async def finalize_annotation(args):
        cur = current()
        problems = _validate_final(entries, cur)
        if problems:
            return {"content": [{"type": "text", "text": "not final yet:\n- " + "\n- ".join(problems)}],
                    "is_error": True}
        # removal budget (msp/eca-rsi lesson: a round that deletes >~10% is
        # usually deleting biology) — soft: one forced second look, then the
        # agent's reaffirmed decision stands and is recorded as over budget
        lab = ad.obs[state["key"]].astype(str)
        rm_clusters = [c for c, e in entries.items() if e["action"] == "remove"]
        rm_mask = lab.isin(rm_clusters).values & ~pre_removed
        frac = float(rm_mask.mean())
        if frac > REMOVE_BUDGET and not holder.get("budget_warned"):
            holder["budget_warned"] = True
            sizes = ", ".join(f"{c} (n={int((lab == c).sum())}, {entries[c].get('remove_reason')})" for c in rm_clusters)
            return {"content": [{"type": "text", "text":
                f"removal budget: you are removing {100 * frac:.1f}% of this lineage's cells beyond the pre-slated "
                f"ones (budget {100 * REMOVE_BUDGET:.0f}%): {sizes}. Reconsider each: stress/immediate-early/ISG/"
                "cycling STATES of a real cell type are biology unless QC is decisive (doublet_score, decontX, "
                "mt%, depth) — keep them with a descriptive fine label (e.g. 'stressed ...') so a later round can "
                "judge; remove only clusters whose evidence is decisive (doublets, low-quality, ambient). Resubmit "
                "what you change, then call finalize_annotation again; if you reaffirm all removals, the second "
                "call is accepted and recorded as over budget."}], "is_error": True}
        comp = _components(entries)
        groups = sorted({tuple(v) for v in comp.values() if len(v) > 1}, key=lambda t: float(str(t[0]).split(",")[0]))
        holder["proposal"] = {"cluster_key": state["key"], "parent_key": PARENT_KEY, "lineage": lineage,
                              "lineage_labels": list(lineage_labels),
                              "clusters": [entries[c] for c in cur],
                              "merged_groups": ["+".join(g) for g in groups],
                              "agent_removed_fraction": round(frac, 4),
                              "budget_exceeded": frac > REMOVE_BUDGET,
                              "overall": str(args.get("overall") or "")}
        if frac > REMOVE_BUDGET:
            print(f"== [{lineage}] WARNING: agent removes {100 * frac:.1f}% of the lineage (budget "
                  f"{100 * REMOVE_BUDGET:.0f}%) — reaffirmed, recorded as budget_exceeded", flush=True)
        with open(os.path.join(outdir, "annotation_proposal.json"), "w") as fh:
            json.dump(holder["proposal"], fh, ensure_ascii=False, indent=2)
        return {"content": [{"type": "text", "text": "accepted"}]}

    server = create_sdk_mcp_server(name="zmip", version="1.0.0",
                                   tools=[cluster_context, check_genes, check_deg, check_stability,
                                          subcluster, submit_cluster, finalize_annotation])
    options = ClaudeAgentOptions(
        mcp_servers={"zmip": server},
        allowed_tools=["Read", "Glob", "Grep", "TaskCreate", "TaskUpdate", "TaskList", "TaskGet",
                       "mcp__zmip__cluster_context", "mcp__zmip__check_genes", "mcp__zmip__check_deg",
                       "mcp__zmip__check_stability", "mcp__zmip__subcluster", "mcp__zmip__submit_cluster",
                       "mcp__zmip__finalize_annotation"],
        permission_mode="bypassPermissions",
        disallowed_tools=["Bash", "Write", "Edit", "MultiEdit", "NotebookEdit", "WebFetch", "WebSearch"],
        max_buffer_size=50_000_000,
        system_prompt=_system_prompt(outdir, lineage, lineage_labels, other_labels, current(), batch_col,
                                     species, prior_cols, foreign_cols, language),
        cwd=os.path.abspath(outdir),
        max_turns=max_turns,
        **({"model": model} if model else {}),
        **({"effort": effort} if effort else {}),
    )
    result_text = None
    async for message in run_query(f"Zoom-in annotate lineage {lineage!r}: one Task per base cluster, "
                                   "submit_cluster each, then finalize_annotation.", options, label=f"zmip {lineage}"):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    print(f"== [{lineage}] agent: {block.name}({str(next(iter(block.input.values()), ''))[:80]})",
                          flush=True)
        elif isinstance(message, ResultMessage):
            result_text = message.result
            if message.total_cost_usd:
                print(f"== [{lineage}] agent cost: ${message.total_cost_usd:.2f}", flush=True)
    if "proposal" not in holder:
        raise RuntimeError(f"[{lineage}] agent finished without finalize_annotation "
                           f"({len(entries)} submitted). Final reply:\n{result_text}")
    if result_text:
        with open(os.path.join(outdir, "annotation_notes.md"), "w") as fh:
            fh.write(result_text)
    return holder["proposal"]


# ---------------------------------------------------------------- entry

def annotate_lineage(ad, outdir, lineage, lineage_labels, other_labels, foreign_cols, species=None,
                     language="English", model=None, effort=None, max_turns=200):
    """ad: the lineage subset AFTER msp.integrate_adata (+ foreign scores).
    Writes annotation_* artifacts, annotated.h5ad and report.html into
    outdir; returns (proposal, removed_df, reassigned_df)."""
    batch_col = ad.uns["msp"]["batch_col"]
    other_keys = [k for k in ad.obs.columns if k.startswith("msp_leiden_r")]
    pre_removed = _load_removal_mask(outdir, ad)
    prior_cols = [c for c in _prior_label_columns(ad, batch_col) if not c.endswith(PREV_SUFFIX)]
    paga = _load_paga_neighbors(outdir, BASE_KEY)
    print(f"== [{lineage}] {int(pre_removed.sum())}/{ad.n_obs} cells pre-slated for removal; "
          f"prior cols {prior_cols}; foreign {foreign_cols}", flush=True)
    proposal = asyncio.run(_run_agent(ad, outdir, lineage, lineage_labels, other_labels, batch_col, species,
                                      prior_cols, paga, pre_removed, foreign_cols, other_keys, language,
                                      model, effort, max_turns))
    removed, reassigned = _apply(ad, proposal["cluster_key"], proposal, pre_removed, lineage)
    removed.to_csv(os.path.join(outdir, "annotation_removed.csv"), index=False)
    reassigned.to_csv(os.path.join(outdir, "annotation_reassigned.csv"), index=False)
    kept = ad[(ad.obs["msp_ann_action"] == "keep").values].copy()
    _plot(ad, kept, os.path.join(outdir, "figures"))
    tmp = os.path.join(outdir, "annotated.tmp.h5ad")
    kept.write_h5ad(tmp)
    os.replace(tmp, os.path.join(outdir, "annotated.h5ad"))
    generate_report(outdir, title=f"zmip lineage report — {lineage} ({', '.join(lineage_labels)})")
    print(f"== [{lineage}] removed {len(removed)}, reassigned {len(reassigned)}, kept {kept.n_obs}/{ad.n_obs}",
          flush=True)
    return proposal, removed, reassigned
