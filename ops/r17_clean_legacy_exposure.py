"""Metadata-only legacy model exposure audit for the new clean fit.

Never reads predictions or target values.  No data set is regenerated.
"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ops.r17_clean_data import sha,write
OUT=ROOT/'reports/r17_clean_pipeline_r1/data/fresh_fit'


def compare(fresh,other):
    exact=fresh.base_window_index.isin(other.base_window_index).to_numpy()
    group=fresh.background_group.isin(other.background_group).to_numpy()
    gap=np.full(len(fresh),np.inf)
    for sensor,indices in fresh.groupby('sensor_group').indices.items():
        b=other[other.sensor_group.eq(sensor)]
        if not len(b):continue
        ids=np.asarray(indices)
        x=pd.to_datetime(fresh.iloc[ids].start_export_clock).astype('int64').to_numpy()/1e9
        y=np.sort(pd.to_datetime(b.start_export_clock).astype('int64').to_numpy()/1e9)
        pos=np.searchsorted(y,x)
        distance=np.minimum(abs(x-y[np.clip(pos-1,0,len(y)-1)]),abs(x-y[np.clip(pos,0,len(y)-1)]))
        gap[ids]=np.maximum(distance-900,0)/3600
    nearby=gap<24-1e-8
    any_exposure=exact|group|nearby
    return dict(exact=exact,group=group,gap=gap,nearby=nearby,any=any_exposure)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=ROOT/'data/synthetic/r17_clean_freshfit_v1/ensemble_fit_fresh_backgrounds.parquet'
    fresh=pd.read_parquet(source).reset_index(drop=True)
    comparisons={};pins={str(source.relative_to(ROOT)):sha(source)}
    for fold in range(5):
        for role in ['train','cv']:
            p=ROOT/f'data/synthetic/r16_multisupport_v1/fold_{fold:02d}/{role}_backgrounds.parquet'
            frame=pd.read_parquet(p)
            name=f'legacy_fold{fold}_{role}'
            comparisons[name]=(frame,compare(fresh,frame),str(p.relative_to(ROOT)))
            pins[str(p.relative_to(ROOT))]=sha(p)
    old=ROOT/'reports/r16_support_temporal_r1/data/local_fit_backgrounds.parquet'
    oldframe=pd.read_parquet(old)
    comparisons['legacy_shared_local_fit_ES']=(oldframe,compare(fresh,oldframe),str(old.relative_to(ROOT)))
    pins[str(old.relative_to(ROOT))]=sha(old)
    # Verify Cat's own-fold prediction receipt has the same source sample IDs.
    cv=ROOT/'data/synthetic/r16_multisupport_v1/fold_00/cv_D0.npz'
    cat=ROOT/'models/r16_catboost_peft_cpu8_r1/D0/fold_00/ownfold_cv.npz'
    with np.load(cv,allow_pickle=False)as a,np.load(cat,allow_pickle=False)as b:
        cat_keys=list(b.files)
        key='sample_id' if 'sample_id'in b.files else None
        same=(bool(np.array_equal(a['sample_id'],b[key]))if key else None)
        if same is False:raise AssertionError('Cat ownCV sample ids differ from frozen CV source')
    for p in [cv,cat]:pins[str(p.relative_to(ROOT))]=sha(p)
    all_cat_cv_checks=[]
    for fold in range(5):
        for regime in ['D0','H0']:
            cp=ROOT/f'data/synthetic/r16_multisupport_v1/fold_{fold:02d}/cv_{regime}.npz'
            mp=ROOT/f'models/r16_catboost_peft_cpu8_r1/{regime}/fold_{fold:02d}/ownfold_cv.npz'
            with np.load(cp,allow_pickle=False)as a,np.load(mp,allow_pickle=False)as b:
                match=bool(np.array_equal(a['sample_id'],b['sample_id']))
                source_ids=a['base_window_index']
                same_regime_metadata=bool(np.array_equal(source_ids,comparisons[f'legacy_fold{fold}_cv'][0].base_window_index.to_numpy()))
                if not match or not same_regime_metadata:raise AssertionError('Cat ownCV/source metadata identity differs')
                all_cat_cv_checks.append({'fold':fold,'regime':regime,'sample_ids_match':match,'background_metadata_ids_match':same_regime_metadata,'n':len(source_ids)})
            pins[str(cp.relative_to(ROOT))]=sha(cp);pins[str(mp.relative_to(ROOT))]=sha(mp)
    rows=[];detail=fresh[['base_window_index','site','recorder','sensor_group','background_group','start_export_clock']].copy()
    for name,(frame,v,path) in comparisons.items():
        finite=v['gap'][np.isfinite(v['gap'])]
        rows.append(dict(exposure_set=name,source_rows=len(frame),fresh_backgrounds=len(fresh),
                         exact_source_overlap=int(v['exact'].sum()),same_background_group=int(v['group'].sum()),
                         support_gap_lt24h=int(v['nearby'].sum()),any_source_group_or24h_exposure=int(v['any'].sum()),
                         minimum_support_gap_hours=float(finite.min())if len(finite)else None,metadata_path=path))
        for tag in ['exact','group','nearby','any']:
            detail[f'{name}__{tag}']=v[tag]
        detail[f'{name}__minimum_support_gap_hours']=np.where(np.isfinite(v['gap']),v['gap'],np.nan)
    train_union=np.logical_or.reduce([comparisons[f'legacy_fold{f}_train'][1]['exact']for f in range(5)])
    train_group_union=np.logical_or.reduce([comparisons[f'legacy_fold{f}_train'][1]['any']for f in range(5)])
    fold0_primary=~(comparisons['legacy_fold0_train'][1]['any']|comparisons['legacy_shared_local_fit_ES'][1]['any'])
    # For Cat require absence from its own-CV ES source at group and 24h level,
    # not merely absence of exactly matching base_window_index.
    fold0_all4=fold0_primary&~comparisons['legacy_fold0_cv'][1]['any']
    fold0_all4_exactonly=fold0_primary&~comparisons['legacy_fold0_cv'][1]['exact']
    detail['legacy_all5_train_exact_union']=train_union
    detail['legacy_all5_train_source_group_time_union']=train_group_union
    detail['strict_fold0_LGB_DINO_Dash_unexposed']=fold0_primary
    detail['strict_fold0_LGB_Cat_DINO_Dash_unexposed']=fold0_all4
    detail['exact_only_CatCV_exclusion_not_sufficient']=fold0_all4_exactonly
    detail.to_csv(OUT/'LEGACY_EXPOSURE_PER_BACKGROUND.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT/'LEGACY_EXPOSURE_AUDIT.csv',index=False)
    with np.load(ROOT/'data/synthetic/r17_clean_freshfit_v1/ensemble_fit_fresh_D0.npz',allow_pickle=False)as a:
        ids=a['base_window_index'];rowids=a['sample_id']
    valid_primary=np.isin(ids,fresh.loc[fold0_primary,'base_window_index'])
    valid_all4=np.isin(ids,fresh.loc[fold0_all4,'base_window_index'])
    exactonly=np.isin(ids,fresh.loc[fold0_all4_exactonly,'base_window_index'])
    mask_path=OUT/'COMMON_FOLD0_UNEXPOSED_MASKS.npz'
    np.savez_compressed(mask_path,sample_id=rowids,base_window_index=ids,
                        strict_LGB_DINO_Dash=valid_primary,strict_LGB_Cat_DINO_Dash=valid_all4,
                        exact_only_CatCV_exclusion_NOT_strict=exactonly,
                        strict_LGB_DINO_Dash_indices=np.flatnonzero(valid_primary),
                        strict_LGB_Cat_DINO_Dash_indices=np.flatnonzero(valid_all4))
    receipt={'metadata_only':True,'no_targets_or_predictions_read':True,'fresh_backgrounds':len(fresh),
             'cat_own_cv_keys':cat_keys,'CatFold0_ownCV_sample_id_matches_frozen_CV1000':same,
             'Cat_all10_ownCV_source_identity_checks':all_cat_cv_checks,
             'all5_legacy_train_exact_union_backgrounds':int(train_union.sum()),
             'all5_legacy_train_group_or_time_union_backgrounds':int(train_group_union.sum()),
             'strict_Fold0_LGB_DINO_Dash_backgrounds':int(fold0_primary.sum()),
             'strict_Fold0_LGB_DINO_Dash_generated_rows':int(valid_primary.sum()),
             'strict_Fold0_LGB_Cat_DINO_Dash_backgrounds':int(fold0_all4.sum()),
             'strict_Fold0_LGB_Cat_DINO_Dash_generated_rows':int(valid_all4.sum()),
             'exact_only_exclusion_backgrounds_NOT_group_independent':int(fold0_all4_exactonly.sum()),
             'comparisons':rows,'source_sha256':pins,'mask_path':str(mask_path.relative_to(ROOT)),
             'mask_sha256':sha(mask_path),
             'interpretation':'Old five-fold Tree averages are exposed to this pool through other folds TRAIN. CatFold0 ownCV may expose the entire pool at source-group level even when exact IDs are excluded. Primary legacy comparison uses Fold0 LGB/DINO/Dash with no Fold0TRAIN/sharedlocalfit exposure; Cat is secondary selection-exposed. New clean models use separate TRAIN and earlystop, so clean refit evaluation remains source-disjoint.'}
    write(OUT/'LEGACY_EXPOSURE_AUDIT.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
