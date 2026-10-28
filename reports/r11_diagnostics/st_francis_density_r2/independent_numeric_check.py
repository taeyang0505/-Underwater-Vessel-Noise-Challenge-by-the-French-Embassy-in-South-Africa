"""Read-only independent arithmetic check; no model inference or fitting."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
D=Path(__file__).resolve().parent
SEED=2026102614
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pair_auc(z,p):
    a=p[z];b=p[~z];wins=0.
    for start in range(0,len(a),100):
        x=a[start:start+100,None];wins+=float((x>b).sum())+.5*float((x==b).sum())
    return wins/(len(a)*len(b))


def main():
    from pipeline.dataset_r9 import R9Sources
    from ops.infer_r9_fold0_all5_pilot import verify_pins
    m=json.loads((D/'results.json').read_text());plan=json.loads((D/'plan.json').read_text())
    verify_pins(plan['original_source_pins'])
    sources=R9Sources(ROOT/'configs/r9_full_training_v1.json',project_root=ROOT,verify_banks=True)
    table=json.loads((D/'treatment_bank.json').read_text())
    source_frames={n:pd.read_parquet(D/(n+'_sources.parquet')) for n in ['sparse','dense']}
    assert not set(source_frames['sparse'].base_window_index)&set(source_frames['dense'].base_window_index)
    source_lookup={int(row['base_window_index']):row for frame in source_frames.values() for row in frame.to_dict('records')}
    components={};preds={};recipe={};reconstruction=[];max_gap=0.;errors=[];boot={};point={}
    for panel in ['sparse','dense','dense_mask5']:
        assert sha(D/(panel+'_known_components.npz'))==m['component_npz_sha256'][panel]
        assert sha(D/(panel+'_predictions.npz'))==m['prediction_npz_sha256'][panel]
        with np.load(D/(panel+'_known_components.npz'),allow_pickle=False) as z:components[panel]={k:z[k] for k in z.files}
        with np.load(D/(panel+'_predictions.npz'),allow_pickle=False) as z:preds[panel]={k:z[k] for k in z.files}
        recipe[panel]=[json.loads(x) for x in (D/(panel+'_recipes.jsonl')).read_text().splitlines()]
        c=components[panel];n=len(c['share']);nb=n//5
        assert n%5==0 and np.array_equal(c['presence'],(c['share']>0).astype(np.float32))
        assert np.array_equal(c['presence'].reshape(nb,5),np.tile([0,1,1,1,1],(nb,1)))
        assert int((c['presence']==0).sum())==nb
        observed=np.isfinite(c['db']);assert np.all(observed.all(2).sum(1)==(15 if panel=='dense' else 5))
        assert (c['cell_share'][~observed]==0).all() and np.isfinite(c['cell_share']).all()
        p=np.where(observed,10.**((c['db'].astype(float)-np.nanmax(c['db'],axis=(1,2),keepdims=True))/10),0.)
        w=p/p.sum((1,2),keepdims=True);gap=np.abs((w*c['cell_share']).sum((1,2))-c['share']);max_gap=max(max_gap,float(gap.max()))
        bounds=np.asarray([x['rounding_bound'] for x in recipe[panel]])
        assert (gap<=bounds+3e-7).all()
        for i in np.unique(np.linspace(0,n-1,64,dtype=int)):
            rr=recipe[panel][int(i)];bg=sources.background(source_lookup[rr['base_window_index']]);o=np.isfinite(bg);peak=float(np.nanmax(bg))
            b=np.where(o,10.**((bg.astype(float)-peak)/10),0.);a=np.zeros_like(b)
            if not rr['explicit_no_source']:
                t=table[rr['treatment_id']];native=sources.reference_minutes[t['reference_index']]*10.**(np.asarray(t['EQ_db'])[None]/10)
                tr=rr['placement_trajectory'];begin=tr['crop_begin_minute'];length=tr['selected_minutes'];start=tr['window_start_minute']
                part=native[begin:begin+length];placed=np.zeros((15,25));left=max(0,start);right=min(15,start+length)
                placed[left:right]=part[left-start:right-start];a=np.where(o,rr['alpha']*placed,0.)
                snr=10*np.log10(rr['alpha']*native.sum(1).max()/np.median(b.sum(1)[o.any(1)]))
                assert abs(snr-t['snr_db'])<1e-10
            if panel=='dense_mask5':
                mask=np.zeros_like(o);mask[rr['mask_minutes']]=True;o&=mask;b=np.where(o,b,0.);a=np.where(o,a,0.)
            e=b+a;q=np.divide(a,e,out=np.zeros_like(e),where=o);y=np.float32(a.sum()/e.sum())
            db=np.full((15,25),np.nan);db[o]=10*np.log10(e[o])+peak;db=np.round(db,1).astype(np.float32)
            assert np.array_equal(db,c['db'][i],equal_nan=True)
            assert y==c['share'][i] and np.array_equal(q.astype(np.float32),c['cell_share'][i])
            reconstruction.append({'panel':panel,'row':int(i),'db_q_share_exact':True})
        ix=0 if panel=='sparse' else 1
        bootstrap=np.random.default_rng(np.random.SeedSequence([SEED,0x4253,ix])).integers(0,nb,(200,nb))
        boot[panel]={};point[panel]={}
        for j,kind in enumerate(m['member_order']):
            y=c['share'].astype(float);ph=preds[panel]['share'][:,j].astype(float);s=preds[panel]['score'][:,j].astype(float);err=ph-y;zero=y==0
            calc={'RMSE':float(np.sqrt(np.mean(err*err))),'MAE':float(abs(err).mean()),'bias':float(err.mean()),
                'AUC_share':pair_auc(~zero,ph),'AUC_score':pair_auc(~zero,s),
                'zero_RMSE':float(np.sqrt(np.mean(ph[zero]**2))),'zero_score_ge_.5_rate':float((s[zero]>=.5).mean())}
            br=np.sqrt((err*err).reshape(nb,5).mean(1)[bootstrap].mean(1));boot[panel][kind]=br
            calc['sourceID_cluster_bootstrap_RMSE_SE']=float(br.std(ddof=1))
            for k,v in calc.items():
                error=abs(v-m['metrics'][panel][kind][k]);errors.append(error);assert error<5e-15,(panel,kind,k,error)
            assert np.array_equal(np.quantile(br,[.025,.975]),m['metrics'][panel][kind]['sourceID_cluster_bootstrap_RMSE_CI95'])
    dense=components['dense'];paired=components['dense_mask5'];mask=np.isfinite(paired['db'])
    assert np.array_equal(paired['db'][mask],dense['db'][mask])
    assert np.array_equal(paired['cell_share'][mask],dense['cell_share'][mask])
    assert all(a['alpha']==b['alpha'] for a,b in zip(recipe['dense'],recipe['dense_mask5']))
    for kind in m['member_order']:
        delta=boot['dense_mask5'][kind]-boot['dense'][kind]
        assert np.array_equal(np.quantile(delta,[.025,.975]),m['differences'][kind]['paired_delta_sourceID_cluster_CI95'])
    sources.assert_unchanged(verify_banks=True);verify_pins(plan['original_source_pins'])
    result={'status':'passed','results_sha256':sha(D/'results.json'),'maximum_metric_recomputation_error':max(errors),
        'AUC_oracle':'direct pairwise positive-negative comparisons with half-credit ties',
        'independent_component_recipe_reconstruction_examples':len(reconstruction),'reconstructed_db_q_share_exact':True,
        'paired_A_B_alpha_preserved_q_and_observed_db_exact':True,'exact20percent_zeros_and5view_clusters_verified':True,
        'cluster_bootstrap_RMSE_SE_CI_and_paired_delta_replayed_exact':True,'max_savedq_rounded_energy_pool_gap':max_gap,
        'all_original_source_model_policy_pins_unchanged':True,'new_model_inference_or_fit_Test_report_network':False,
        'verification_code_sha256':sha(__file__)}
    with (D/'independent_numeric_verification.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
