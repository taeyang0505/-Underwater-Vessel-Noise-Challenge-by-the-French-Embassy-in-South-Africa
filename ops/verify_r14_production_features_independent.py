"""Bounded independent CPU14 parity on every TRAIN mask pattern, all folds.

Only one deterministic representative per observed TRAIN-mask pattern is
synthesized. This is not full-row numerical parity or model performance.
"""
from pathlib import Path
import gc,hashlib,json,time
import numpy as np
import torch
from pipeline.full_data_r13 import R13Sources,DynamicR13Dataset
from pipeline.features_r14_anchor import channels
from pipeline.features_r14_production import ProductionFrontend

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    torch.set_num_threads(2)
    config=ROOT/'configs/r14_production_mac_v1.json';c=json.loads(config.read_text())
    names=('pipeline/features_r14_production.py','pipeline/features_r14_torch.py','pipeline/features_r14_anchor.py',
        'pipeline/features_r12.py','pipeline/models_r14_production.py','pipeline/full_data_r13.py')
    pins={p:sha(ROOT/p) for p in names};out=ROOT/'reports/R14_PRODUCTION_14_FEATURE_INDEPENDENT_QA.json'
    if out.exists():raise FileExistsError('Fresh feature QA receipt required')
    policy={'data_variant':'baseline','include_shape':True,'production_recipe_approved':True,'QA_only':True}
    policy['policy_sha256']=hashlib.sha256(json.dumps(policy,sort_keys=True).encode()).hexdigest()
    source=R13Sources(ROOT/c['source_config_path'],bundle=ROOT/c['bundle_path'],verify_banks=True,runtime_policy=policy)
    folds={};maxx=maxw=poolgap=0.;cases=0;beg=time.perf_counter()
    for fold in range(5):
        ds=DynamicR13Dataset(source,fold,epoch=1);frame=ds.metadata;first=frame.groupby('effective_mask_bits',sort=True).head(1).index.to_numpy(int)
        inverse=np.empty(len(frame),np.int64);inverse[ds.schedule['primary_order']]=np.arange(len(frame))
        selected=[];record=[]
        for idx in first:
            row=ds.raw_item(int(inverse[idx]));db=np.asarray(row['db'],np.float32);bits=sum((int(v)<<t) for t,v in enumerate(np.isfinite(db).all(1)))
            assert bits==int(frame.iloc[idx].effective_mask_bits) and int(bits).bit_count()>=3
            selected.append(row);record.append({'TRAIN_metadata_index':int(idx),'base_window_index':int(row['base_window_index']),
                'effective_mask_bits':bits,'observed_minutes':int(bits).bit_count(),'source_id':row['source_id']})
        db=np.stack([a['db'] for a in selected]);q=np.stack([a['cell_share'] for a in selected]);y=np.asarray([a['share'] for a in selected])
        basis=source.basis(fold);front=ProductionFrontend(basis);expected,ew=channels(db,basis)
        actual,w=front(torch.from_numpy(db));actual=actual.numpy();w=w.numpy()
        err=float(abs(expected-actual).max());we=float(abs(ew-w).max());assert err<=3e-6 and we<=3e-8
        assert np.array_equal(actual[:,5],np.isfinite(db).astype(np.float32)) and np.all(actual.transpose(0,2,3,1)[~np.isfinite(db)]==0)
        gap=float(abs((w*q).sum((1,2))-y).max());assert gap<=.0058
        chunk=np.concatenate([front(torch.from_numpy(db[i:i+1]))[0].numpy() for i in range(len(db))]);assert np.max(abs(chunk-actual))<=3e-6
        gained,gw=front(torch.from_numpy(db.astype(float)+20));assert np.max(abs(gained.numpy()-actual))<=3e-6 and np.max(abs(gw.numpy()-w))<=3e-8
        maxx=max(maxx,err);maxw=max(maxw,we);poolgap=max(poolgap,gap);cases+=len(first)
        assert len(first)==frame.effective_mask_bits.nunique()
        folds[str(fold)]={'TRAIN_rows':len(frame),'unique_TRAIN_mask_patterns':len(first),'actual_synthesized_representatives':len(first),
            'all_mask_patterns_represented':True,'basis_training_fold':basis['training_binding']['fold'],
            'feature_max_abs_error':err,'weight_max_abs_error':we,'representatives':record}
        del ds;gc.collect()
    rng=np.random.default_rng(2026102714);synthetic=np.round(80+rng.normal(0,5,(9,15,25)),1)
    synthetic[0,:10]=np.nan;synthetic[1,:,23:]=np.nan;synthetic[2,:,3]=np.nan;synthetic[3,:,0:8]=np.nan
    synthetic[4,:,1::2]=np.nan;synthetic[5,0:12]=np.nan;synthetic[6]=80.;synthetic[7]=np.tile([-.15,-.05,.05,.15,99.95],5)[None,:]
    basis=source.basis(0);front=ProductionFrontend(basis);nx,nw=channels(synthetic,basis);tx,tw=front(torch.from_numpy(synthetic))
    se=float(np.max(abs(nx-tx.numpy())));sw=float(np.max(abs(nw-tw.numpy())));assert se<=3e-6 and sw<=3e-8
    changed=synthetic.copy();changed[:,:,23:]+=40;changed_x,_=front(torch.from_numpy(changed));assert torch.equal(changed_x[:,11:],tx[:,11:])
    assert not bool(tx[3,11:].any())
    try:front({'db':synthetic,'share':np.ones(9)})
    except (TypeError,ValueError):pass
    else:raise AssertionError('Label dictionary accepted')
    source.assert_unchanged();assert all(sha(ROOT/n)==h for n,h in pins.items())
    result={'format':'r14_production14_independent_TRAIN_mask_parity_v1','status':'passed','config_sha256':sha(config),
        'source_sha256':pins,'data_hash':c['bundle_data_hash'],'folds':folds,'actual_representatives':cases,
        'max_feature_abs_error':maxx,'max_weight_abs_error':maxw,'raw_label_rounded_pool_gap_max':poolgap,
        'synthetic_partial_rounding_URN_max_feature_error':se,'synthetic_max_weight_error':sw,
        'common_gain20db_and_chunk1_checks':True,'input_labels_dict_rejected':True,'source_unchanged':True,
        'elapsed_seconds':time.perf_counter()-beg,'new_model_fit_or_inference_or_MPS':False,'Test_or_heldout_arrays':False,
        'scope':'Complete fivefold TRAIN-mask-pattern coverage, one actual mixed-input representative per pattern; not all TRAIN rows numerically compared.'}
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'passed','representatives':cases,'max_feature_error':maxx,'max_weight_error':maxw,'report_sha256':sha(out)}))
if __name__=='__main__':main()
