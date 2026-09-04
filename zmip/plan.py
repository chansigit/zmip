"""
zmip.plan — decide which coarse lineages to zoom into, and how to pool them.

Evidence (all deterministic, written to outdir before the agent runs):
  - lineage_counts.csv      cells / samples per coarse label
  - lineage_knn.csv         kNN cross-connectivity between coarse labels
                            (row-normalised share of each label's graph
                            edges that land on every other label — high
                            off-diagonal = the two labels form one island)
  - lineage_paga.csv        PAGA connectivity on the same graph
  - lineage_islands.csv     host-computed UMAP islands: connected components
                            of the 2-D UMAP kNN graph (long edges pruned),
                            as % of each coarse label's cells per island
  - figures/umap_<coarse>.png  the coarse-label UMAP the agent MUST look at

The agent (via harness_bridge) pools coarse labels that form ONE connected
island on the UMAP into a lineage and keeps separate islands separate —
even when they are related lineages — because zoom-in re-embeds each
lineage on its own and a disconnected island dragged in becomes a
permanent "foreign" cluster. States (Proliferating, stressed) go with the
island they sit in. The host enforces: every coarse label assigned exactly
once; zoom=true only for lineages with at least min_cells cells (below
that, leiden cannot resolve stable substates — default 800); a lineage may
not pool labels whose cells sit on different UMAP islands (the rule the
prompt states, checked against lineage_islands.csv — a weak model once
pooled every label into one lineage); labels sharing one island but split
across lineages are pushed back once for confirmation (confirm_shared_islands:
true) since a touching-but-distinct pair is a judgement the picture decides.
The plan is archived to zmip_plan.json and reused on resume.
"""

import asyncio
import copy
import json
import os

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData
from msp.plots import save_single_umap, slug

from .cache import write_json

DEFAULT_MIN_CELLS = 800

# UMAP islands (host evidence + validation)
ISLAND_K = 15            # kNN in 2-D UMAP space
ISLAND_EDGE_FACTOR = 4.0  # prune edges longer than this × the median k-th-neighbour distance
ISLAND_MIN_FRAC = 0.002  # components smaller than max(20, this × n_obs) are noise (island 0)
HOME_FRAC = 0.30         # an island holding ≥ this share of a label's cells is a home island of the label


def _lineage_slugs(names):
    """Keep existing directory names, rejecting unsafe or ambiguous mappings."""
    result, owners = {}, {}
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("lineage name must be a non-empty string")
        directory = slug(name)
        if directory in (".", "..", "figures"):
            raise ValueError(f"lineage name {name!r} maps to reserved directory {directory!r}")
        if directory in owners:
            raise ValueError(f"lineage names {owners[directory]!r} and {name!r} share directory {directory!r}")
        owners[directory] = name
        result[name] = directory
    return result

def umap_islands(ad, coarse_col, outdir):
    """Connected components of the UMAP kNN graph (k=ISLAND_K, edges longer
    than ISLAND_EDGE_FACTOR × the median k-th-neighbour distance dropped so
    empty space separates components while a continuum does not), tiny
    components folded into island 0 ("noise"). Returns a DataFrame: coarse
    label × island → % of the label's cells (columns "island_1".. by size,
    "noise"), written to lineage_islands.csv; None when there is no UMAP."""
    if "X_umap" not in ad.obsm or ad.n_obs < ISLAND_K + 2:
        return None
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    U = np.asarray(ad.obsm["X_umap"])[:, :2].astype(float)
    n = U.shape[0]
    d, idx = cKDTree(U).query(U, k=ISLAND_K + 1)
    d, idx = d[:, 1:], idx[:, 1:]
    cutoff = ISLAND_EDGE_FACTOR * float(np.median(d[:, -1]))
    keep = (d <= cutoff).ravel()
    rows = np.repeat(np.arange(n), ISLAND_K)[keep]
    cols = idx.ravel()[keep]
    G = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    _, comp = connected_components(G, directed=False)
    sizes = np.bincount(comp)
    min_size = max(20, int(ISLAND_MIN_FRAC * n))
    big = [c for c in np.argsort(-sizes, kind="stable") if sizes[c] >= min_size]
    remap = {c: i + 1 for i, c in enumerate(big)}
    island = np.array([remap.get(c, 0) for c in comp])
    lab = ad.obs[coarse_col].astype(str).values
    labels = list(pd.Series(lab).value_counts().index)
    cols_out = [f"island_{i}" for i in range(1, len(big) + 1)] + ["noise"]
    tab = pd.DataFrame(0.0, index=labels, columns=cols_out)
    for l in labels:
        m = lab == l
        counts = np.bincount(island[m], minlength=len(big) + 1)
        pct = 100 * counts / max(1, m.sum())
        tab.loc[l, cols_out[:-1]] = pct[1:]
        tab.loc[l, "noise"] = pct[0]
    tab = tab.round(1)
    tab.index.name = f"{coarse_col} \\ % of cells per UMAP island"
    tab.attrs["island_sizes"] = {f"island_{i + 1}": int(sizes[c]) for i, c in enumerate(big)}
    tab.attrs["noise_cells"] = int((island == 0).sum())
    tab.to_csv(os.path.join(outdir, "lineage_islands.csv"))
    return tab


def home_islands(islands, label):
    """Islands holding ≥ HOME_FRAC of the label's cells (noise never counts)."""
    if islands is None or label not in islands.index:
        return set()
    row = islands.loc[label]
    return {c for c in islands.columns if c != "noise" and row[c] >= 100 * HOME_FRAC}


def island_problems(lineages, islands):
    """(hard, soft): hard = a lineage pools labels whose home islands are
    disjoint (separate islands on the UMAP); soft = labels with a common home
    island split across lineages. A label with no home island (scattered or
    all noise) constrains nothing; a label spanning several islands links
    them."""
    hard, soft = [], []
    if islands is None:
        return hard, soft
    for ln in lineages:
        homes = {m: home_islands(islands, m) for m in ln["coarse_labels"]}
        anchored = {m: h for m, h in homes.items() if h}
        if len(anchored) < 2:
            continue
        # union-find over islands via the labels that span them
        parent = {}
        def find(x):
            while parent.setdefault(x, x) != x:
                x = parent[x]
            return x
        for h in anchored.values():
            h = sorted(h)
            for other in h[1:]:
                parent[find(other)] = find(h[0])
        groups = {}
        for m, h in anchored.items():
            groups.setdefault(find(next(iter(h))), []).append(m)
        if len(groups) > 1:
            parts = "; ".join(f"{sorted(ms)} on {sorted(set().union(*(homes[m] for m in ms)))}"
                              for ms in groups.values())
            hard.append(f"lineage {ln['name']!r} pools labels that sit on separate UMAP islands: {parts} "
                        f"(see lineage_islands.csv) — separate islands must be separate lineages")
    by_label = {m: ln["name"] for ln in lineages for m in ln["coarse_labels"]}
    for isl in [c for c in islands.columns if c != "noise"]:
        owners = {}
        for m in islands.index:
            if isl in home_islands(islands, m) and m in by_label:
                owners.setdefault(by_label[m], []).append(m)
        if len(owners) > 1:
            soft.append(f"{isl} ({islands.attrs.get('island_sizes', {}).get(isl, '?')} cells) is shared by labels "
                        f"in different lineages: " + "; ".join(f"{n!r}: {ms}" for n, ms in owners.items()))
    return hard, soft


def lineage_evidence(ad, coarse_col, batch_col, outdir):
    """Write evidence and the coarse UMAP; return counts, kNN, PAGA and islands."""
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
            # PAGA uses the neighbor graph and labels, never expression matrices.
            neighbors = copy.deepcopy(ad.uns["neighbors"])
            keys = [neighbors.get("connectivities_key", "connectivities"),
                    neighbors.get("distances_key", "distances")]
            tmp = AnnData(obs=pd.DataFrame({"_zmip_coarse": pd.Categorical(lab, categories=labels)},
                                          index=ad.obs_names.copy()),
                          obsp={key: ad.obsp[key] for key in keys if key in ad.obsp},
                          uns={"neighbors": neighbors})
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
    islands = umap_islands(ad, coarse_col, outdir)
    return counts, knn, paga, islands


def validate_plan(plan, labels, counts, min_cells, islands=None):
    """Host rules. Returns (problems, normalised_plan). Below-threshold
    lineages are forced to zoom=false (recorded), not rejected. With an
    islands table: pooling separate islands is rejected; splitting a shared
    island is rejected once, accepted when the plan carries
    confirm_shared_islands: true (recorded under host_warnings)."""
    problems = []
    if not isinstance(plan, dict):
        return ["plan must be a JSON object"], None
    if not isinstance(plan.get("confirm_shared_islands", False), bool):
        problems.append("confirm_shared_islands must be a boolean")
    lineages = plan.get("lineages")
    if not isinstance(lineages, list) or not lineages:
        return ['"lineages" must be a non-empty list'], None
    seen_labels, seen_names, out = {}, set(), []
    for ln in lineages:
        if not isinstance(ln, dict):
            problems.append("each lineage must be a JSON object")
            continue
        if not isinstance(ln.get("name"), str):
            problems.append("lineage name must be a non-empty string")
            continue
        name = ln["name"].strip()
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
        if not all(isinstance(m, str) for m in members):
            problems.append(f"lineage {name!r}: coarse_labels must contain strings only")
            continue
        if not isinstance(ln.get("zoom", True), bool):
            problems.append(f"lineage {name!r}: zoom must be a boolean")
            continue
        for m in members:
            if m not in labels:
                problems.append(f"lineage {name!r}: {m!r} is not a coarse label; labels: {labels}")
            elif m in seen_labels:
                problems.append(f"coarse label {m!r} assigned to both {seen_labels[m]!r} and {name!r}")
            seen_labels[m] = name
        n = int(sum(counts.loc[m, "n_cells"] for m in members if m in counts.index))
        zoom = ln.get("zoom", True)
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
    try:
        _lineage_slugs(ln["name"] for ln in out)
    except ValueError as exc:
        return [str(exc)], None
    hard, soft = island_problems(out, islands)
    problems += hard
    if soft and not plan.get("confirm_shared_islands", False):
        problems += [f"{w} — labels on one island belong to one lineage unless the picture shows a real gap; "
                     f"either merge them or resubmit unchanged with \"confirm_shared_islands\": true" for w in soft]
    if problems:
        return problems, None
    norm = {"lineages": out, "notes": str(plan.get("notes", "")), "min_cells": min_cells}
    if soft:
        norm["host_warnings"] = soft
    return [], norm


_PLAN_SCHEMA_DOC = """{
  "lineages": [
    {"name": "<short English lineage name, e.g. 'Stromal' / 'Immune' / 'Endothelial'>",
     "coarse_labels": ["<coarse label>", ...],   // every coarse label exactly once across all lineages
     "zoom": true | false,                        // false = keep as is (too small, or nothing to gain)
     "reason": "<why these labels are one island / one lineage, and why zoom or not>"}
  ],
  "notes": "<overall reading of the UMAP: which islands exist, how clean the separation is>",
  "confirm_shared_islands": false   // only set true when resubmitting a plan the host flagged for splitting one island
}"""


def _islands_text(islands):
    if islands is None:
        return ""
    sizes = islands.attrs.get("island_sizes", {})
    return ("\nUMAP islands computed by the host (connected components of the 2-D UMAP kNN graph; % of each label's "
            "cells per island; island sizes " + ", ".join(f"{k}={v}" for k, v in sizes.items()) + "):\n"
            + islands.to_string() + "\nThe host rejects a lineage that pools labels sitting on different islands, "
            "and asks once for confirmation when labels sharing an island are split across lineages.\n")


def _prompt(coarse_col, labels, counts, knn, paga, islands, min_cells, fig_rel, species):
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
{_islands_text(islands)}
Finish by calling submit_plan with JSON of this schema:
{_PLAN_SCHEMA_DOC}
If validation fails, fix the named problems and call it again."""


async def _run(coarse_col, labels, counts, knn, paga, islands, outdir, min_cells, species, model, effort):
    from harness_bridge import ToolSpec, run_agent

    async def submit_plan(args):
        try:
            plan = json.loads(args["plan_json"])
        except json.JSONDecodeError as e:
            return {"content": [{"type": "text", "text": f"JSON parse error: {e}"}], "is_error": True}
        problems, norm = validate_plan(plan, labels, counts, min_cells, islands)
        if problems:
            return {"content": [{"type": "text", "text": "fix and resubmit:\n- " + "\n- ".join(problems)}],
                    "is_error": True}
        return {"content": [{"type": "text", "text": "plan accepted: " + ", ".join(
            f"{ln['name']}={ln['coarse_labels']} ({ln['n_cells']} cells, zoom={ln['zoom']})"
            for ln in norm["lineages"])}], "_submitted": norm}

    tool = ToolSpec(
        name="submit_plan",
        description="Submit the lineage plan (JSON string, schema in the task).",
        input_schema={"plan_json": str}, handler=submit_plan,
    )
    fig_rel = os.path.join("figures", f"umap_{slug(coarse_col)}.png")
    result = await run_agent(
        tools=[tool], submit_tool="submit_plan",
        prompt=f"Read {fig_rel}, then plan the lineages and call submit_plan.",
        system_prompt=_prompt(coarse_col, labels, counts, knn, paga, islands, min_cells, fig_rel, species),
        cwd=os.path.abspath(outdir), model=model, effort=effort, max_turns=30,
        allowed_builtin=("read",), max_buffer_size=50_000_000, label="zmip plan",
    )
    result.submitted["agent_notes"] = result.transcript_text or ""
    return result.submitted


def plan_lineages(ad, coarse_col, batch_col, outdir, min_cells=DEFAULT_MIN_CELLS, species=None,
                  model=None, effort=None, force=False):
    """Evidence → agent → validated plan, archived to outdir/zmip_plan.json
    (reused when present)."""
    from harness_bridge import default_model

    path = os.path.join(outdir, "zmip_plan.json")
    counts, knn, paga, islands = lineage_evidence(ad, coarse_col, batch_col, outdir)
    if os.path.exists(path) and not force:
        with open(path) as f:
            plan = json.load(f)
        # Recheck current evidence, including the archived explicit island review.
        candidate = dict(plan)
        candidate["confirm_shared_islands"] = bool(plan.get("host_warnings"))
        problems, normalized = validate_plan(candidate, list(counts.index), counts, min_cells, islands)
        if problems or plan.get("min_cells") != min_cells or plan.get("coarse_col") != coarse_col:
            raise ValueError(f"recorded plan does not match current input/options: {problems}; use --force")
        if normalized["lineages"] != plan["lineages"]:
            raise ValueError("recorded lineage counts or zoom decisions changed; use --force")
        print(f"== reusing recorded plan {path}", flush=True)
        return plan
    labels = list(counts.index)
    plan = asyncio.run(_run(coarse_col, labels, counts, knn, paga, islands, outdir, min_cells, species,
                            model or default_model(), effort))
    plan["coarse_col"] = coarse_col
    write_json(path, plan)
    return plan
