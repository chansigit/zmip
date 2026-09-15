"""Worker-side zoom-in primitives. No model calls or child-process scheduling."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from .annotate import _guard_batch_action, _validate_cluster, _validate_final, components

TYPE_KEY = 'msp_leiden_r1.0'
QUALITY_KEY = 'msp_leiden_r2.0'


def partitions(obs):
    """Preserve the two actual partitions; neither is assumed to nest in the other."""
    if not obs.index.is_unique or obs.index.hasnans:
        raise ValueError('Cell IDs must be unique and non-null')
    if any(key not in obs or obs[key].isna().any() for key in (TYPE_KEY, QUALITY_KEY)):
        raise ValueError('Fresh type and quality clusterings are required')
    return pd.crosstab(obs[QUALITY_KEY].astype(str), obs[TYPE_KEY].astype(str))


def validate_types(proposal, obs, own_labels):
    """Validate identity at 1.0, independently of quality/removal at 2.0."""
    partitions(obs)
    if not isinstance(proposal, dict) or proposal.get('cluster_key') != TYPE_KEY or not isinstance(proposal.get('clusters'), list):
        raise ValueError('Type proposal must name the 1.0 clustering and contain clusters')
    clusters = sorted(obs[TYPE_KEY].astype(str).unique())
    entries = {}
    for entry in proposal['clusters']:
        problems = _validate_cluster(entry, clusters, own_labels, [])
        if problems:
            raise ValueError('; '.join(problems))
        if entry['action'] != 'keep':
            raise ValueError('Type proposals assign identity; use quality intersections for removal/reassignment')
        cid = str(entry['cluster_id'])
        if cid in entries:
            raise ValueError('Duplicate type cluster: ' + cid)
        entries[cid] = deepcopy(entry)
    problems = _validate_final(entries, clusters)
    if problems:
        raise ValueError('; '.join(problems))
    return entries


def validate_quality(proposal, obs, other_labels):
    """Each QC group partitions into disjoint, explicitly named type intersections."""
    table = partitions(obs)
    if not isinstance(proposal, dict) or proposal.get('cluster_key') != QUALITY_KEY or not isinstance(proposal.get('clusters'), list):
        raise ValueError('Quality proposal must name the 2.0 clustering and contain clusters')
    normalized, seen = [], set()
    for group in proposal['clusters']:
        if not isinstance(group, dict) or not isinstance(group.get('decisions'), list):
            raise ValueError('Each quality group needs a decisions list')
        qid = str(group.get('cluster_id', ''))
        if qid not in table.index or qid in seen:
            raise ValueError('Unknown or duplicate quality cluster: ' + qid)
        seen.add(qid)
        expected = set(table.columns[table.loc[qid].gt(0)])
        covered, decisions = set(), []
        for original in group['decisions']:
            if not isinstance(original, dict):
                raise ValueError('A quality decision must be an object')
            entry = deepcopy(original)
            members = entry.get('type_clusters')
            if (not isinstance(members, list) or not members or any(not isinstance(c, str) for c in members)
                    or len(set(members)) != len(members) or not set(members) <= expected or covered & set(members)):
                raise ValueError('Quality intersections must name disjoint, present 1.0 clusters')
            covered.update(members)
            if entry.get('action') not in {'keep', 'remove', 'reassign'} or entry.get('confidence') not in {'high', 'medium', 'low'}:
                raise ValueError('Invalid quality action/confidence')
            for field in ('rationale', 'evidence'):
                if not isinstance(entry.get(field), str) or not entry[field].strip():
                    raise ValueError('Quality decisions need specific rationale and evidence')
            if entry['action'] == 'remove':
                from msp.annotate import REMOVE_REASONS
                if entry.get('remove_reason') not in {*REMOVE_REASONS, 'dissociation', 'dying'}:
                    raise ValueError('Removal needs an explicit supported reason')
                entry = _guard_batch_action(entry)
            if entry['action'] == 'reassign':
                if entry.get('reassign_to') not in other_labels:
                    raise ValueError('Reassignment must target another planned lineage label')
                if not isinstance(entry.get('fine_label'), str) or not entry['fine_label'].strip():
                    raise ValueError('Reassignment needs an explicit fine label')
            decisions.append(entry)
        if covered != expected:
            raise ValueError('Quality decisions must cover every type intersection exactly once')
        normalized.append({'cluster_id': qid, 'decisions': decisions})
    if seen != set(table.index):
        raise ValueError('Quality decisions must cover every 2.0 cluster')
    return normalized


def apply_decisions(obs, types, quality, own_labels, other_labels, lineage, pre_reasons=None):
    """Return labeled metadata and exact removal/reassignment audits without changing inputs.

    pre_reasons maps cells to already supported numerical exclusions. The caller
    writes the resulting H5AD and ledger together before accepting the operation.
    """
    entries = validate_types(types, obs, own_labels)
    decisions = validate_quality(quality, obs, other_labels)
    out = obs.copy()
    t, q = out[TYPE_KEY].astype(str), out[QUALITY_KEY].astype(str)
    merged = components(entries)
    out['msp_ann_cluster'] = t.map({c: '+'.join(v) for c, v in merged.items()})
    for column, field in [('msp_ann_coarse', 'coarse_label'), ('msp_ann_fine', 'fine_label')]:
        out[column] = t.map({c: e[field].strip() for c, e in entries.items()})
    reasons = deepcopy(pre_reasons or {})
    if not set(reasons) <= set(obs.index) or any(not v for v in reasons.values()):
        raise ValueError('Numerical exclusions must identify input cells and supported reasons')
    reassigned = {}
    for group in decisions:
        for decision in group['decisions']:
            ids = obs.index[q.eq(group['cluster_id']) & t.isin(decision['type_clusters'])]
            if decision['action'] == 'remove':
                for cell in ids:
                    reasons.setdefault(cell, []).append({'code': decision['remove_reason'], 'decision': decision})
            elif decision['action'] == 'reassign':
                out.loc[ids, 'msp_ann_coarse'] = decision['reassign_to']
                out.loc[ids, 'msp_ann_fine'] = decision['fine_label'].strip()
                reassigned.update({cell: decision['reassign_to'] for cell in ids})
    removed = out.index.isin(reasons)
    out['msp_ann_action'] = pd.Categorical(np.where(removed, 'remove', 'keep'))
    out['zmip_reassigned_to'] = pd.Series(reassigned, dtype=object).reindex(out.index).where(~removed, None).astype('category')
    rm = pd.DataFrame({'cell': out.index[removed], 'lineage': lineage,
        'cluster': t.loc[removed].to_numpy(), 'quality_cluster': q.loc[removed].to_numpy()})
    rm['reasons'] = [reasons[c] for c in rm.cell]
    ra = out.loc[out.zmip_reassigned_to.notna(), ['zmip_reassigned_to', 'msp_ann_fine']].rename(
        columns={'zmip_reassigned_to': 'reassign_to', 'msp_ann_fine': 'fine_label'})
    ra.insert(0, 'cluster', t.loc[ra.index]);ra.insert(0, 'lineage', lineage)
    ra = ra.rename_axis('cell').reset_index()
    from .merge import _validate_annotation, _validate_partition
    kept = out.loc[~removed]
    _validate_partition(lineage, obs.index, kept, rm, ra)
    _validate_annotation(lineage, kept, ra, own_labels, [*own_labels, *other_labels])
    return out, rm, ra, decisions


def compute_lineage(sub, name, labels, markers, outdir, *, batch_col, species,
                    n_top_genes=3000, n_pcs=50, n_neighbors=15):
    """One numerical unit up through UMAP/QC; persist the independent DEG plan."""
    from msp.integrate import integrate_adata
    from .foreign import score_foreign
    if not sub.obs_names.is_unique or not len(sub):
        raise ValueError('Lineage input must be nonempty with unique cell IDs')
    if 'counts' not in sub.layers:
        raise ValueError('Re-embedding requires accepted counts')
    if not set(sub.obs['msp_ann_coarse_prev'].astype(str)) <= set(labels):
        raise ValueError('Lineage input contains labels outside the accepted plan')
    expected = sub.obs_names.copy()
    integrate_adata(sub, batch_col, str(outdir), species=species, resolutions=(1., 2.),
        n_top_genes=n_top_genes, n_pcs=n_pcs, n_neighbors=n_neighbors,
        meta_extra={'zmip_lineage': name, 'zmip_coarse_labels': list(labels)}, defer_deg=True)
    if not sub.obs_names.equals(expected):
        raise ValueError('Integration changed lineage membership without an exclusion ledger')
    columns = score_foreign(sub, markers, name, (TYPE_KEY, QUALITY_KEY), str(outdir), str(Path(outdir)/'figures'))
    partitions(sub.obs).to_csv(Path(outdir)/'type_quality_intersections.csv')
    sub.obs[[TYPE_KEY, QUALITY_KEY]].rename_axis('cell').to_csv(Path(outdir)/'cell_partitions.csv.gz')
    sub.write_h5ad(Path(outdir)/'integrated.h5ad')
    return columns
