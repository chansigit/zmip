import copy

import pandas as pd
import pytest

from zmip.scheduled import TYPE_KEY, QUALITY_KEY, apply_decisions, validate_quality


def test_nonnested_quality_intersections_preserve_types_and_reassignments():
    obs = pd.DataFrame({TYPE_KEY: ['0', '1', '0', '1'], QUALITY_KEY: ['a', 'a', 'b', 'b']},
                       index=['001', 'NA', '003', '004'])
    types = {'cluster_key': TYPE_KEY, 'clusters': [dict(cluster_id=c, coarse_label='Epithelial',
        fine_label='Fine '+c, merge_target=None, action='keep', confidence='high',
        evidence={k:'observed evidence' for k in ('distinctness','markers','foreign','merge')},
        rationale='observed identity') for c in ('0','1')]}
    def decision(ids, action='keep', **extra):
        return dict(type_clusters=ids, action=action, confidence='high',
                    evidence='specific marker and QC evidence', rationale='observed evidence', **extra)
    quality = {'cluster_key': QUALITY_KEY, 'clusters': [
        dict(cluster_id='a', decisions=[decision(['0'], 'remove', remove_reason='dying'),decision(['1'])]),
        dict(cluster_id='b', decisions=[decision(['0']),decision(['1'], 'reassign', reassign_to='Immune',fine_label='T cell')])]}
    original = obs.copy(deep=True)
    out, rm, ra, _ = apply_decisions(obs,types,quality,['Epithelial'],['Immune'],'Epithelial')
    pd.testing.assert_frame_equal(obs,original)
    assert list(rm.cell)==['001'] and list(ra.cell)==['004']
    assert out.loc['NA','msp_ann_fine']=='Fine 1' and out.loc['003','msp_ann_fine']=='Fine 0'
    assert out.loc['004','msp_ann_coarse']=='Immune'
    bad=copy.deepcopy(quality);bad['clusters'][0]['decisions'].append(decision(['0']))
    with pytest.raises(ValueError,match='disjoint'):validate_quality(bad,obs,['Immune'])
    bad=copy.deepcopy(quality);bad['clusters'][0]['decisions'].pop()
    with pytest.raises(ValueError,match='every type intersection'):validate_quality(bad,obs,['Immune'])
    bad=copy.deepcopy(quality);bad['cluster_key']=TYPE_KEY
    with pytest.raises(ValueError,match='2.0'):validate_quality(bad,obs,['Immune'])
    quality['clusters'][0]['decisions'][0]['remove_reason']='batch'
    _,rm,_,normalized=apply_decisions(obs,types,quality,['Epithelial'],['Immune'],'Epithelial')
    assert rm.empty and normalized[0]['decisions'][0]['review_required']


def test_lineage_compute_persists_two_fresh_partitions_and_deferred_deg(tmp_path):
    import json
    import anndata as an
    import numpy as np
    from zmip.scheduled import compute_lineage
    rng=np.random.default_rng(7)
    counts=rng.poisson(1.,(150,60)).astype('float32')
    for group in range(3):counts[group*50:(group+1)*50,group*10:(group+1)*10]+=8
    obs=pd.DataFrame({'sample_id':['a','b']*75,'pct_counts_mt':[2.]*150,
        'n_genes_by_counts':(counts>0).sum(axis=1).astype(float),'total_counts':counts.sum(axis=1),
        'doublet_score':[.05]*150,'msp_ann_coarse_prev':['Epithelial']*150,
        TYPE_KEY:['inherited']*150,QUALITY_KEY:['inherited']*150},index=[f'c{i}' for i in range(150)])
    data=an.AnnData(counts,obs=obs);data.layers['counts']=counts.copy()
    compute_lineage(data,'Epithelial',['Epithelial'],{},tmp_path,batch_col='sample_id',species='human',
                    n_top_genes=30,n_pcs=10,n_neighbors=10)
    saved=an.read_h5ad(tmp_path/'integrated.h5ad')
    assert saved.obs_names.equals(obs.index)
    assert all('inherited' not in set(saved.obs[k]) for k in (TYPE_KEY,QUALITY_KEY))
    plan=json.loads((tmp_path/'deg_plan.json').read_text())
    assert set(plan['keys'])=={TYPE_KEY,QUALITY_KEY} and plan['plan']
    assert (tmp_path/'deg_input/metadata.h5ad').exists()
    assert pd.read_csv(tmp_path/'type_quality_intersections.csv',index_col=0).to_numpy().sum()==150
