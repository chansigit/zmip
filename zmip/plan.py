"""
zmip.plan — decide which coarse lineages to zoom into, and how to pool them.

Evidence (all deterministic, written to outdir before the agent runs):
  - lineage_counts.csv      cells / samples per coarse label
  - lineage_knn.csv         kNN cross-connectivity between coarse labels
                            (row-normalised share of each label's graph
                            edges that land on every other label — high
                            off-diagonal = the two labels form one island)
  - lineage_paga.csv        PAGA connectivity on the same graph
  - figures/umap_<coarse>.png  the coarse-label UMAP the agent MUST look at

The agent (claude-agent-sdk) pools coarse labels that form ONE connected
island on the UMAP into a lineage and keeps separate islands separate —
even when they are related lineages — because zoom-in re-embeds each
lineage on its own and a disconnected island dragged in becomes a
permanent "foreign" cluster. States (Proliferating, stressed) go with the
island they sit in. The host enforces: every coarse label assigned exactly
once; zoom=true only for lineages with at least min_cells cells (below
that, leiden cannot resolve stable substates — default 800); the plan is
archived to zmip_plan.json and reused on resume.
"""

import asyncio
import json
import os

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from msp.plots import save_single_umap, slug

DEFAULT_MIN_CELLS = 800


def lineage_evidence(ad, coarse_col, batch_col, outdir):
    """Write the three evidence tables + the coarse UMAP; return (counts_df,
    knn_df, paga_df) for the prompt."""
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    lab = ad.obs[coarse_col].astype(str)
    labels = list(lab.value_counts().index)

    counts = pd.DataFrame({
        "n_cells": lab.value_counts(),
        "n_samples": ad.obs.groupby(lab, observed=True)[batch_col].nunique(),
        "pct": (100 * lab.value_counts() / ad.n_obs).round(2),
    }).loc[labels]
    counts.index.name = coarse_col
    counts.to_csv(os.path.join(outdir, "lineage_counts.csv"))

    if "connectivities" not in ad.obsp:
        raise ValueError("input lacks obsp['connectivities'] — run msp first (annotated.h5ad has it)")
    C = ad.obsp["connectivities"]
    C = C.tocsr() if sp.issparse(C) else sp.csr_matrix(C)
    codes = pd.Categorical(lab, categories=labels).codes
    M = sp.csr_matrix((np.ones(ad.n_obs), (np.arange(ad.n_obs), codes)), shape=(ad.n_obs, len(labels)))
    W = np.asarray((M.T @ C @ M).todense())  # label x label summed edge weight
    row = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
    knn = pd.DataFrame((100 * row).round(2), index=labels, columns=labels)
    knn.index.name = "from \\ to (% of edges)"
    knn.to_csv(os.path.join(outdir, "lineage_knn.csv"))

    paga = None
    if "neighbors" in ad.uns:
        try:
            tmp = ad.copy() if ad.n_obs < 200_000 else ad
            tmp.obs["_zmip_coarse"] = pd.Categorical(lab, categories=labels)
            sc.tl.paga(tmp, groups="_zmip_coarse")
            P = np.asarray(tmp.uns["paga"]["connectivities"].todense())
            paga = pd.DataFrame(P.round(3), index=labels, columns=labels)
            paga.index.name = "paga connectivity"
            paga.to_csv(os.path.join(outdir, "lineage_paga.csv"))
        except Exception as e:  # PAGA is a convenience, never a blocker
            print(f"== paga skipped: {e}", flush=True)

    n = len(labels)
    save_single_umap(ad, coarse_col, os.path.join(figdir, f"umap_{slug(coarse_col)}.png"),
                     repel=True, repel_fontsize=9 if n > 12 else 11,
                     figsize=(9, 9) if n > 12 else None)
    return counts, knn, paga


def validate_plan(plan, labels, counts, min_cells):
    """Host rules. Returns (problems, normalised_plan). Below-threshold
    lineages are forced to zoom=false (recorded), not rejected."""
    problems = []
    lineages = plan.get("lineages")
    if not isinstance(lineages, list) or not lineages:
        return ['"lineages" must be a non-empty list'], None
    seen_labels, seen_names, out = {}, set(), []
    for ln in lineages:
        name = str(ln.get("name", "")).strip()
        members = ln.get("coarse_labels")
        if not name:
            problems.append(f"lineage without a name: {ln}")
            continue
        if name in seen_names:
            problems.append(f"duplicate lineage name {name!r}")
        seen_names.add(name)
        if not isinstance(members, list) or not members:
            problems.append(f"lineage {name!r}: coarse_labels must be a non-empty list")
            continue
        for m in members:
            if m not in labels:
                problems.append(f"lineage {name!r}: {m!r} is not a coarse label; labels: {labels}")
            elif m in seen_labels:
                problems.append(f"coarse label {m!r} assigned to both {seen_labels[m]!r} and {name!r}")
            seen_labels[m] = name
        n = int(sum(counts.loc[m, "n_cells"] for m in members if m in counts.index))
        zoom = bool(ln.get("zoom", True))
        forced = ""
        if zoom and n < min_cells:
            zoom, forced = False, f" (host: {n} < min_cells={min_cells}, zoom disabled)"
        out.append({"name": name, "coarse_labels": list(members), "n_cells": n, "zoom": zoom,
                    "reason": str(ln.get("reason", "")) + forced})
    missing = [m for m in labels if m not in seen_labels]
    if missing:
        problems.append(f"coarse labels not assigned to any lineage: {missing}")
    if problems:
        return problems, None
    return [], {"lineages": out, "notes": str(plan.get("notes", "")), "min_cells": min_cells}


_PLAN_SCHEMA_DOC = """{
  "lineages": [
    {"name": "<short English lineage name, e.g. 'Stromal' / 'Immune' / 'Endothelial'>",
     "coarse_labels": ["<coarse label>", ...],   // every coarse label exactly once across all lineages
     "zoom": true | false,                        // false = keep as is (too small, or nothing to gain)
     "reason": "<why these labels are one island / one lineage, and why zoom or not>"}
  ],
  "notes": "<overall reading of the UMAP: which islands exist, how clean the separation is>"
}"""


def _prompt(coarse_col, labels, counts, knn, paga, min_cells, fig_rel, species):
    ctx = f"Species: {species}. " if species else ""
    return f"""You are planning the zoom-in stage of a single-cell analysis. {ctx}The dataset has been \
integrated (harmony) and annotated at coarse level (obs column {coarse_col!r}: {labels}). Zoom-in will \
take each LINEAGE you define, re-embed it on its own (HVG/PCA/harmony/leiden/UMAP on that subset) and \
refine its annotation. Decide how to group the coarse labels into lineages and which lineages to zoom.

The one rule that matters: a lineage must be ONE connected island on the current UMAP. Read the figure \
{fig_rel} FIRST and describe what you see. Labels that sit in the same island or form a continuum \
(no gap between them) belong together, even if they are different cell types — e.g. when data quality \
is modest, T, B and myeloid cells may fuse into one immune island and must be zoomed together; with clean \
data they form separate islands and become separate lineages. Related labels sitting on DIFFERENT islands \
stay separate: a disconnected island dragged into a re-embedding just becomes a permanent foreign cluster. \
States (Proliferating, stressed, cycling) go with the island they sit in — do NOT make a state its own \
lineage; likewise a small label (below the zoom threshold) that sits inside or on the edge of a bigger island \
is pooled into that island rather than left out (it would otherwise never be re-examined). Use the kNN cross-connectivity \
table (share of each label's graph edges landing on other labels) and PAGA as quantitative corroboration \
of what the picture shows — the picture decides ties.

Zoom only lineages with at least {min_cells} cells (below that leiden cannot resolve stable substates); \
smaller ones still get a lineage entry with zoom=false. Every coarse label must appear exactly once.

Cells and samples per coarse label:
{counts.to_string()}

kNN cross-connectivity (% of each row label's edges that land on the column label; diagonal = within):
{knn.to_string()}
{('PAGA connectivity:' + chr(10) + paga.to_string()) if paga is not None else ''}

Finish by calling submit_plan with JSON of this schema:
{_PLAN_SCHEMA_DOC}
If validation fails, fix the named problems and call it again."""


async def _run(ad, coarse_col, labels, counts, knn, paga, outdir, min_cells, species, model, effort):
    from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, ResultMessage, ToolUseBlock,
                                  create_sdk_mcp_server, query, tool)

    holder = {}

    @tool("submit_plan", "Submit the lineage plan (JSON string, schema in the task).", {"plan_json": str})
    async def submit_plan(args):
        try:
            plan = json.loads(args["plan_json"])
        except json.JSONDecodeError as e:
            return {"content": [{"type": "text", "text": f"JSON parse error: {e}"}], "is_error": True}
        problems, norm = validate_plan(plan, labels, counts, min_cells)
        if problems:
            return {"content": [{"type": "text", "text": "fix and resubmit:\n- " + "\n- ".join(problems)}],
                    "is_error": True}
        holder["plan"] = norm
        return {"content": [{"type": "text", "text": "plan accepted: " + ", ".join(
            f"{ln['name']}={ln['coarse_labels']} ({ln['n_cells']} cells, zoom={ln['zoom']})"
            for ln in norm["lineages"])}]}

    server = create_sdk_mcp_server(name="zmip", version="1.0.0", tools=[submit_plan])
    fig_rel = os.path.join("figures", f"umap_{slug(coarse_col)}.png")
    options = ClaudeAgentOptions(
        mcp_servers={"zmip": server},
        allowed_tools=["Read", "mcp__zmip__submit_plan"],
        permission_mode="bypassPermissions",
        disallowed_tools=["Bash", "Write", "Edit", "MultiEdit", "NotebookEdit", "WebFetch", "WebSearch"],
        max_buffer_size=50_000_000,
        system_prompt=_prompt(coarse_col, labels, counts, knn, paga, min_cells, fig_rel, species),
        cwd=os.path.abspath(outdir),
        max_turns=30,
        **({"model": model} if model else {}),
        **({"effort": effort} if effort else {}),
    )
    result_text = None
    async for message in query(prompt=f"Read {fig_rel}, then plan the lineages and call submit_plan.",
                               options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    print(f"== plan agent: {block.name}({str(next(iter(block.input.values()), ''))[:80]})",
                          flush=True)
        elif isinstance(message, ResultMessage):
            result_text = message.result
            if message.total_cost_usd:
                print(f"== plan agent cost: ${message.total_cost_usd:.2f}", flush=True)
    if "plan" not in holder:
        raise RuntimeError(f"plan agent finished without an accepted submit_plan. Final reply:\n{result_text}")
    holder["plan"]["agent_notes"] = result_text or ""
    return holder["plan"]


def plan_lineages(ad, coarse_col, batch_col, outdir, min_cells=DEFAULT_MIN_CELLS, species=None,
                  model=None, effort=None):
    """Evidence → agent → validated plan, archived to outdir/zmip_plan.json
    (reused when present)."""
    path = os.path.join(outdir, "zmip_plan.json")
    counts, knn, paga = lineage_evidence(ad, coarse_col, batch_col, outdir)
    if os.path.exists(path):
        with open(path) as f:
            plan = json.load(f)
        print(f"== reusing recorded plan {path}", flush=True)
        return plan
    labels = list(counts.index)
    plan = asyncio.run(_run(ad, coarse_col, labels, counts, knn, paga, outdir, min_cells, species,
                            model, effort))
    plan["coarse_col"] = coarse_col
    with open(path, "w") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    return plan
