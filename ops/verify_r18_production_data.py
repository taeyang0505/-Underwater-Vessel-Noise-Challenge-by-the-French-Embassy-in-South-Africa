"""Independent NPZ readback and deterministic raw-source replay for R18 data."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from pipeline.r18_production_data import ROOT,OUT,REPORT,SEED,sha,write,guard

def main():
    sys.addaudithook(guard)
    from pipeline.dataset_r9 import R9Sources
    from pipeline.sampler_r9 import smooth_eq,energy_weights
    from ops.r16_forward_data import generate_operator,channel_zero_mean
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert manifest['status']=='completed' and not manifest['Test_used']
    for name,h in manifest['code_sha256'].items():assert sha(ROOT/name)==h
    source=R9Sources(verify_banks=False)
    reserved=pd.read_parquet(OUT/'reserved_B19_Cprime25.parquet')
    eligible=pd.read_parquet(OUT/'reference_pool_465.parquet')
    eligible_ids=set(eligible.reference_row_index.astype(int))
    assert len(eligible_ids)==465 and not eligible_ids.intersection(reserved.reference_row_index.astype(int))
    records=[];role_sets={};replayed=0
    for role,regimes in manifest['roles'].items():
        backgrounds=pd.read_parquet(OUT/f'{role}_backgrounds.parquet')
        role_sets[role]=set(backgrounds.base_window_index.astype(int))
        previous=None
        for regime,record in regimes.items():
            path=ROOT/record['path'];assert sha(path)==record['sha256']
            with np.load(path,allow_pickle=False)as z:a={k:z[k]for k in z.files}
            n=len(a['share']);assert n==record['n']
            assert len(set(a['sample_id']))==n
            observed=np.isfinite(a['db']);z=a['presence'];positive=z.astype(bool)
            assert np.array_equal(observed,np.isfinite(a['db']).all(2)[:,:,None].repeat(25,2))
            assert np.all(a['cell_share'][~observed]==0)
            assert np.all(np.isfinite(a['cell_share'])) and np.all((a['cell_share']>=0)&(a['cell_share']<=1))
            assert np.array_equal(a['share']>0,positive)
            assert np.array_equal(z.reshape(-1,5),np.tile([0,1,1,1,1],(n//5,1)))
            assert np.array_equal(observed.reshape(-1,5,15,25),observed.reshape(-1,5,15,25)[:,0:1].repeat(5,axis=1))
            assert np.all(a['db'][observed]*10==np.rint(a['db'][observed]*10))
            assert set(a['reference_index'][positive])==eligible_ids
            use=np.bincount(a['reference_index'][positive],minlength=509);ess=float(use.sum()**2/(use*use).sum())
            assert abs(ess-record['reference_concentration_Kish_ESS'])<1e-10
            if regime=='H0':assert int(np.sum(a['active_minutes'][positive]>=12))==round(positive.sum()*.135)
            if previous is not None:
                assert np.array_equal(a['sample_id'],previous['sample_id'])
                assert np.array_equal(a['reference_index'],previous['reference_index'])
                assert np.array_equal(a['operator'],previous['operator'])
                assert np.array_equal(observed,previous['observed'])
            previous={k:a[k]for k in ['sample_id','reference_index','operator']};previous['observed']=observed
            rng=np.random.default_rng([SEED,sum(role.encode()),0x564552])
            pick=set(rng.choice(n,16,replace=False).tolist())
            # Explicitly replay zero and long examples as well as random rows.
            pick.update([0,int(np.flatnonzero(a['support_stratum']=='long_ge12')[0])])
            max_replay=0.
            for i in sorted(pick):
                row=backgrounds.iloc[i//5];bg=np.asarray(source.background(row),float)
                crng=np.random.default_rng([SEED,sum(role.encode()),i,0x4348]);zero=not bool(z[i])
                native=None if zero else source.reference_minutes[int(a['reference_index'][i])]
                geometry=None if zero else {'crop_begin_minute':int(a['crop_begin'][i]),'selected_minutes':int(a['selected_minutes'][i]),
                    'window_start_minute':int(a['window_start'][i]),'family':str(a['family'][i])}
                eq=np.zeros(25)if zero else smooth_eq(crng,6.)
                post=np.zeros(25)if zero else channel_zero_mean(crng,6. if role=='train'else 12.)
                value,_=generate_operator(bg,native,geometry,eq,post,float(crng.uniform(-10,10)),crng.uniform(-10,10,2),str(a['operator'][i]),zero=zero)
                assert np.array_equal(value['db'],a['db'][i],equal_nan=True)
                assert np.array_equal(value['cell_share'].astype(np.float32),a['cell_share'][i])
                max_replay=max(max_replay,abs(float(value['share'])-float(a['share'][i])))
                assert max_replay==0.;replayed+=1
            # Row-wise stable normalization, separately implemented from training.
            offset=np.nanmax(a['db'],axis=(1,2),keepdims=True)
            power=np.where(observed,10.**((a['db'].astype(float)-offset)/10.),0.)
            weights=power/power.sum((1,2),keepdims=True)
            pooled=(weights*a['cell_share']).sum((1,2))
            records.append({'role':role,'regime':regime,'n':n,'sha256':record['sha256'],'actual_reference_clips':len(eligible_ids),
                'Kish_reference_ESS':ess,'max_per_clip_use':int(use.max()),'H0_ge12_fraction':float((a['active_minutes'][positive]>=12).mean()),
                'nobs5_fraction':float((a['n_obs']==5).mean()),'min_nobs':int(a['n_obs'].min()),
                'source_replay_rows':len(pick),'source_replay_exact':True,
                'rounded_pool_vs_raw_target_max_error':float(abs(pooled-a['share']).max())})
    for role,ids in role_sets.items():
        for other,otherids in role_sets.items():
            if role!=other:assert not ids.intersection(otherids)
    result={'status':'passed','manifest_sha256':sha(OUT/'manifest.json'),'datasets':records,'total_raw_source_replays':replayed,
        'reserved44_used_in_new_train_ES_fit':False,'reference_shared_between_train_ES_fit':True,
        'independent_reference_claimed_for_clean_ES_fresh_fit':False,'Test_used':False,
        'verification_code_sha256':sha(Path(__file__))}
    write(REPORT/'INDEPENDENT_READBACK_VERIFICATION.json',result)
    pd.DataFrame(records).to_csv(REPORT/'INDEPENDENT_READBACK_VERIFICATION.csv',index=False)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
