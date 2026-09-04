"""
zmip.report — the global zoom-in report (self-contained HTML, msp's CSS).

Sections (omitted when their artifacts are absent):
  1. Lineage plan       — coarse UMAP the agent read, counts / kNN / PAGA
                          tables, the plan with reasons, agent notes
  2. Lineages           — one subsection per zoomed lineage: summary,
                          its coarse/fine UMAPs, link to <lineage>/report.html
  3. Final annotation   — zmip coarse/fine/lineage UMAPs on the msp global
                          embedding, fine-label counts per coarse label
  4. Removed & reassigned — UMAPs + per-lineage counts from the archives

Usage:
    python -m zmip.report <zmip_outdir> [--out report.html]
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re

from msp.plots import slug
from msp.report import CSS, TOC_PIN_SCRIPT, _csv_table, _img

from . import cache, publication

_LABELS = {
    "plan": "Lineage plan",
    "lineages": "Lineages",
    "final": "Final annotation",
    "removed": "Removed & reassigned",
}


def _h2(anchor):
    return f'<h2 id="{anchor}">{_LABELS[anchor]}</h2>'


def _rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _table(rows, cols):
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{html.escape('' if r.get(c) is None else str(r.get(c)))}</td>"
                                    for c in cols) + "</tr>" for r in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def _section_plan(outdir, plan):
    coarse = plan.get("coarse_col", "msp_ann_coarse")
    parts = [_h2("plan"),
             '<p class="hint">Lineages are pooled by UMAP connectivity (one island = one lineage, even '
             "across cell types when they fuse; separate islands stay separate even when related). "
             f"Lineages below min_cells={plan.get('min_cells')} are kept as is (zoom=false).</p>"]
    fig = os.path.join(outdir, "figures", f"umap_{re.sub(r'[^A-Za-z0-9_.-]+', '_', coarse)}.png")
    if os.path.exists(fig):
        parts.append('<div class="trio">' + _img(fig) + "</div>")
    parts += ["<h3>Plan</h3>", '<div id="annotation-tables">',
              _table([{"lineage": ln["name"], "coarse_labels": ", ".join(ln["coarse_labels"]),
                       "n_cells": ln["n_cells"], "zoom": ln["zoom"], "reason": ln["reason"]}
                      for ln in plan["lineages"]],
                     ("lineage", "coarse_labels", "n_cells", "zoom", "reason")), "</div>"]
    if plan.get("notes"):
        parts.append(f"<p><b>Agent reading of the UMAP:</b> {html.escape(plan['notes'])}</p>")
    for w in plan.get("host_warnings", []):
        parts.append(f"<p><b>Host warning (confirmed by the agent):</b> {html.escape(w)}</p>")
    for name, label in (("lineage_counts.csv", "Cells per coarse label"),
                        ("lineage_islands.csv", "UMAP islands (host-computed; % of each label's cells)"),
                        ("lineage_knn.csv", "kNN cross-connectivity (% of edges)"),
                        ("lineage_paga.csv", "PAGA connectivity")):
        p = os.path.join(outdir, name)
        if os.path.exists(p):
            parts += [f"<details><summary>{label}</summary>{_csv_table(p)}</details>"]
    return "".join(parts)


def _section_lineages(outdir, plan):
    zoomed = [ln for ln in plan["lineages"] if ln["zoom"]]
    if not zoomed:
        return ""
    parts = [_h2("lineages"),
             '<p class="hint">Each zoomed lineage was re-embedded on its own (HVG/PCA/harmony/leiden/UMAP '
             "on the subset) and annotated by its own agent; the full per-lineage report (QC, DEG, "
             "foreign-lineage scores, per-cluster decisions) is linked.</p>"]
    for ln in zoomed:
        d = os.path.join(outdir, slug(ln["name"]))
        prop_p = os.path.join(d, "annotation_proposal.json")
        parts.append(f"<h3>{html.escape(ln['name'])}</h3>")
        if not os.path.exists(prop_p):
            parts.append("<p>(not run yet)</p>")
            continue
        with open(prop_p) as f:
            prop = json.load(f)
        rm = _rows(os.path.join(d, "annotation_removed.csv"))
        ra = _rows(os.path.join(d, "annotation_reassigned.csv"))
        n_clusters = len(prop.get("clusters", []))
        fine = sorted({e["fine_label"] for e in prop["clusters"] if e["action"] == "keep"})
        parts.append("<p>" + html.escape(
            f"{ln['n_cells']} cells · {n_clusters} clusters on {prop.get('cluster_key')} · "
            f"{len(fine)} fine labels kept · removed {len(rm)} · reassigned {len(ra)}"
            + (f" · merged groups {', '.join(prop['merged_groups'])}" if prop.get("merged_groups") else ""))
            + f' · <a href="{html.escape(slug(ln["name"]))}/report.html">full lineage report</a></p>')
        figs = [os.path.join(d, "figures", n) for n in ("annotation_umap_coarse.png", "annotation_umap_fine.png",
                                                        "annotation_umap_removed.png")]
        figs = [p for p in figs if os.path.exists(p)]
        if figs:
            parts.append('<div class="trio">' + "".join(_img(p) for p in figs) + "</div>")
        if prop.get("overall"):
            parts.append(f"<p><b>Overall:</b> {html.escape(prop['overall'])}</p>")
        rows = [{"cluster": e["cluster_id"], "coarse": e["coarse_label"], "fine": e["fine_label"],
                 "action": e["action"] + (f" → {e.get('reassign_to')}" if e["action"] == "reassign" else "")
                 + (f" ({e.get('remove_reason')})" if e["action"] == "remove" else ""),
                 "merge_target": e.get("merge_target") or "", "confidence": e["confidence"],
                 "rationale": e["rationale"]} for e in prop["clusters"]]
        parts.append('<details><summary>per-cluster decisions</summary><div id="annotation-tables">'
                     + _table(rows, ("cluster", "coarse", "fine", "action", "merge_target", "confidence", "rationale"))
                     + "</div></details>")
    return "".join(parts)


def _section_final(outdir, kept_counts):
    figdir = os.path.join(outdir, "figures")
    figs = [os.path.join(figdir, n) for n in ("zmip_umap_lineage.png", "zmip_umap_coarse.png", "zmip_umap_fine.png")]
    figs = [p for p in figs if os.path.exists(p)]
    if not figs and not kept_counts:
        return ""
    parts = [_h2("final"),
             '<p class="hint">zmip_ann_coarse / zmip_ann_fine on the msp global UMAP (no global '
             "re-embedding at this step — that is the next round's job). Skipped lineages carry their "
             "msp labels unchanged; reassigned cells carry their new lineage's coarse label.</p>"]
    if figs:
        parts.append('<div class="trio">' + "".join(_img(p) for p in figs) + "</div>")
    if kept_counts:
        parts += ["<h3>Fine-label legend</h3>",
                  '<p class="hint">Numbers on the fine UMAP above; ordered by coarse label, then size.</p>',
                  '<div id="annotation-tables">',
                  _table(kept_counts, ("id", "zmip_ann_coarse", "zmip_ann_fine", "n_cells")), "</div>"]
    return "".join(parts)


def _section_removed(outdir):
    rm = _rows(os.path.join(outdir, "zmip_removed.csv"))
    ra = _rows(os.path.join(outdir, "zmip_reassigned.csv"))
    figdir = os.path.join(outdir, "figures")
    figs = [os.path.join(figdir, n) for n in ("zmip_umap_removed.png", "zmip_umap_reassigned.png")]
    figs = [p for p in figs if os.path.exists(p)]
    if not figs and not rm and not ra:
        return ""
    parts = [_h2("removed")]
    if figs:
        parts.append('<div class="trio">' + "".join(_img(p) for p in figs) + "</div>")

    def count(rows, keys):
        out = {}
        for r in rows:
            k = tuple(r.get(x, "") for x in keys)
            out[k] = out.get(k, 0) + 1
        return [dict(zip(keys, k), n_cells=v) for k, v in sorted(out.items(), key=lambda kv: -kv[1])]

    if rm:
        src = count([{"lineage": r["lineage"],
                      "source": ("agent: " + (r.get("remove_reason") or "")) if r.get("annotate_remove") == "True"
                      else "subset preannotation filtering"} for r in rm], ("lineage", "source"))
        parts += [f"<h3>Removed ({len(rm)} cells)</h3>", _table(src, ("lineage", "source", "n_cells"))]
    if ra:
        parts += [f"<h3>Reassigned ({len(ra)} cells)</h3>",
                  _table(count(ra, ("lineage", "reassign_to", "fine_label")),
                         ("lineage", "reassign_to", "fine_label", "n_cells"))]
    return "".join(parts)


def _number(sections):
    present = [(a, l) for a, l in _LABELS.items() if any(f'<h2 id="{a}">{l}</h2>' in s for s in sections)]
    numbered = {a: f"{i}. {l}" for i, (a, l) in enumerate(present, 1)}
    out, toc = [], []
    for s in sections:
        anchor = next((a for a, l in present if f'<h2 id="{a}">{l}</h2>' in s), None)
        if anchor:
            s = s.replace(f'<h2 id="{anchor}">{_LABELS[anchor]}</h2>',
                          f'<h2 id="{anchor}">{numbered[anchor]}</h2>', 1)
            subs, n = [], [0]

            def repl(m, anchor=anchor, subs=subs, n=n):
                n[0] += 1
                sid = f"{anchor}-{n[0]}"
                subs.append((sid, m.group(1)))
                return f'<h3 id="{sid}">{m.group(1)}</h3>'
            s = re.sub(r"<h3>(.*?)</h3>", repl, s)
            toc.append(f'<a href="#{anchor}">{html.escape(numbered[anchor])}</a>')
            if subs:
                toc.append('<div class="toc-sub">' + "".join(
                    f'<a href="#{sid}">{html.escape(re.sub("<.*?>", "", t))}</a>' for sid, t in subs) + "</div>")
        out.append(s)
    return out, f'<nav class="toc">{"".join(toc)}</nav>'


def generate_report(outdir, out_html=None, title=None, *, result_dir=None):
    if result_dir is None:
        publication.require_complete(outdir)
    results = result_dir or outdir
    out_html = out_html or os.path.join(outdir, "report.html")
    plan_p = os.path.join(outdir, "zmip_plan.json")
    plan = None
    if os.path.exists(plan_p):
        with open(plan_p) as f:
            plan = json.load(f)
    kept_counts = _rows(os.path.join(results, "zmip_fine_legend.csv"))
    sections = []
    if plan:
        sections += [_section_plan(outdir, plan), _section_lineages(outdir, plan)]
    sections += [_section_final(results, kept_counts), _section_removed(results)]
    sections, toc = _number([s for s in sections if s])
    from msp.report import compose_title

    if title is None:
        parent_plan = os.path.join(os.path.dirname(os.path.abspath(outdir)), "zmip_plan.json")
        if plan is None and os.path.isfile(parent_plan):  # a per-lineage sub-report
            title = compose_title("zoom-in lineage (zmip)", outdir, subject=os.path.basename(os.path.abspath(outdir)))
        else:
            title = compose_title("zoom-in by lineage (zmip)", outdir)
    header = (f"<h1>{html.escape(title)}</h1>"
              f'<p class="meta">source dir: {html.escape(os.path.abspath(outdir))}</p>')
    body = f'{header}<div class="layout">{toc}<div class="content">{"".join(sections)}</div></div>'
    doc = ("<!DOCTYPE html><html><head><meta charset='utf-8'>"
           f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
           f"<body>{body}{TOC_PIN_SCRIPT}</body></html>")
    temporary = str(out_html) + ".tmp"
    with open(temporary, "w") as fh:
        fh.write(doc)
    os.replace(temporary, out_html)
    if result_dir is None:
        publication.refresh_report_receipt(outdir, out_html)
    return out_html


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="zmip.report", description=__doc__)
    ap.add_argument("outdir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    with cache.lock_run(a.outdir):
        publication.recover(a.outdir)
        print(f"wrote {generate_report(a.outdir, out_html=a.out)}")
