"""Evaluate ALL four metadata/physics-feasible extension conditions.

The extension strengthens LF/mid attenuation and changes the scale anchor
to strict3.16--7.94kHz. It is not evidence that the originalLF-10..-15
hypothesis met both constraints, or that any condition describes Test.
No fitting, policy changes, Test/unused-report/GPU/SSH access.
"""
from __future__ import annotations
import argparse,gc,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
WEIGHTS=(.1,.1,.1,.1,.6)

def run(out):
    from ops.r11_asymmetric_eq_audit import guard,sha,write,render,panel_summary,metric,utc
    from ops.infer_r9_fold0_all5_pilot import environment,inputs,local,verify_pins,combine,MEMBERS
    environment(4);sys.addaudithook(guard)
    import numpy as np,pandas as pd,torch
    from torch.utils.data import Dataset,DataLoader
    from pipeline.dataset_r9 import R9Sources
    from pipeline.sampler_r9 import place_minutes,energy_weights
    from pipeline.relative_grid_r6 import quantization_pooling_bound
    from pipeline.relative_grid_r6_v2 import relative_grid_v2
    from pipeline.runner_r9 import restore_neural
    from pipeline.training_r9 import predict_loader
    out=out.resolve()
    if out.exists():raise FileExistsError('Fresh audit output required')
    out.mkdir(parents=True);start=time.perf_counter()
    old=ROOT/'reports/r11_diagnostics/asymmetric_eq_r1';ext=ROOT/'reports/r11_diagnostics/asymmetric_forward_feasibility_r1'
    extdata=json.loads((ext/'RESULTS.json').read_text());selected=[c for c in extdata['conditions'] if c['all']['both_panel_targets_met']]
    if len(selected)!=4:raise ValueError('Exactly4predeclaredphysical-feasibleconditionsrequired')
    feasible={'format':'r11_all_physics_feasible_conditions_v1','selection':'ALL conditions satisfying only model-free panel constraints; no error-based selection',
              'conditions':selected,'extension_results_sha256':sha(ext/'RESULTS.json'),
              'extension_plan_sha256':sha(ext/'predeclared_plan.json')}
    write(out/'feasible_manifest.json',feasible)
    sourceplan=json.loads((old/'predeclared_plan.json').read_text())
    pins=dict(sourceplan['code_and_source_pins'])
    for path in [ext/'RESULTS.json',ext/'predeclared_plan.json',old/'shared_specs.json',old/'background_source_manifest.parquet',
                 ROOT/'ops/r11_asymmetric_feasible_models.py',ROOT/'ops/r11_asymmetric_forward_feasibility.py']:
        pins[str(path.relative_to(ROOT))]=sha(path)
    plan={'format':'r11_all_feasible_models_plan_v1','created_utc':utc(),'seed':2026102612,'n_each':1000,
          'feasible_manifest_sha256':sha(out/'feasible_manifest.json'),'allfour_feasible_conditions_evaluated':True,
          'condition_names':[c['name'] for c in selected],'condition_selection_uses_model_outputs':False,
          'physical_assumption_change':'LF-15/-25/-35 andmid316..1995attenuationup to-20; STRICTHFanchor3162..7943ratherthanmidhigh316..7943; beyondoriginalLF-10..-15onlyhypothesis',
          'original4factorial_result_unchanged':True,'new_production_or_Test_claim':False,
          'baseline':'sameB/ref/nativegeometry/physicaly withR9smoothEQ6 andalpha solvedfortarget; notoriginalR9SNRlaw',
          'region_indices':sourceplan['region_indices'],'weights':WEIGHTS,'NNthreads':2,'treethreads':4,
          'code_and_source_pins':pins,'no_fitting_production_change_or_Test':True}
    write(out/'predeclared_plan.json',plan)
    # Freeze the complete feasible list BEFORE any model is invoked.
    print(json.dumps({'stage':'all4feasible_plan_frozen','plan_sha256':sha(out/'predeclared_plan.json'),'feasible_manifest_sha256':sha(out/'feasible_manifest.json')}),flush=True)
    sources=R9Sources();meta=pd.read_parquet(old/'background_source_manifest.parquet');specs=json.loads((old/'shared_specs.json').read_text());n=len(specs)
    cache=[]
    for s in specs:
        row=meta.iloc[s['row']];db=sources.background(row);peak=float(db.max());B=10**((db-peak)/10)
        native=sources.reference_minutes[max(s['ref'],0)];P,trajectory=place_minutes(native,np.random.default_rng(s['placement_seed']),s['family'])
        cache.append((db,B,peak,native,P,trajectory,row))
    panels={};physical={}
    for c in selected:
        name=c['name'];lf=c['LF_EQ_db'];mid=c['mid_EQ_db'];snr=c['strict_HF_peak_SNR_db']
        for control in (False,True):
            panel=name+('_matched_R9eq6' if control else '_asym')
            data={k:np.empty((n,15,25),np.float64 if k=='unrounded_db' else np.float32) for k in ('db','unrounded_db','cell_share')}
            data.update(share=np.empty(n,np.float32),presence=np.empty(n,np.int64),base_window_index=np.empty(n,np.int64),reference_index=np.empty(n,np.int64))
            recipes=[]
            for i,(s,(db,B,peak,native,P,trajectory,row)) in enumerate(zip(specs,cache)):
                if control:
                    d,recipe=render(db,native,s,lf,snr,matched_target=float(panels[name+'_asym']['share'][i]))
                else:
                    eq=np.zeros(25);eq[:9]=lf;eq[9]=(lf+mid)/2;eq[10:19]=mid;eq[19]=mid/2
                    alpha=0. if s['no_source'] else 10**(snr/10)*np.median(B[:,20:25].sum(1))/native[:,20:25].sum(1).max()
                    A=alpha*P*10**(eq[None,:]/10);E=B+A;q=A/E;y=float(A.sum()/E.sum());raw=10*np.log10(E)+peak;rounded=np.round(raw,1).astype(np.float32);w=energy_weights(rounded)
                    identity=abs(float((E/E.sum()*q).sum())-y);gap=abs(float((w*q).sum())-y)
                    bound=quantization_pooling_bound(.05+float(np.max(abs(np.spacing(rounded)))/2)+1e-12)
                    if identity>1e-10 or gap>bound+2e-7:raise AssertionError('Physicalidentityboundfailed')
                    regions={}
                    for reg,inds in plan['region_indices'].items():
                        e=E[:,inds].sum(1);a=A[:,inds].sum(1);level=10*np.log10(e)+peak
                        regions[reg]={'bump_db':float(level.max()-np.median(level)),
                            'true_region_energy_fraction':float(a.sum()/e.sum()),'added_region_contribution_to_global_y':float(a.sum()/E.sum())}
                    recipe={'alpha':float(alpha),'eq_db':eq.tolist(),'trajectory':trajectory,'raw_identity_error':identity,'rounded_pool_gap':gap,
                            'rounding_bound':bound,'physical_regions':regions,'no_source':bool(s['no_source'])}
                    d={'db':rounded,'unrounded_db':raw,'cell_share':q.astype(np.float32),'share':np.float32(y),'presence':np.int64(y>0)}
                for k in d:data[k][i]=d[k]
                data['base_window_index'][i]=int(row.base_window_index);data['reference_index'][i]=s['ref'];recipes.append(recipe)
            panels[panel]=data;np.savez_compressed(out/(panel+'_samples.npz'),**data);write(out/(panel+'_recipes.json'),recipes)
            physical[panel]=panel_summary(data,recipes)
        # Published model-free forward aggregate must be reproduced, not
        # silently changed when preparing model input.
        if abs(physical[name+'_asym']['target_rms']-c['all']['y_rms'])>5e-8:raise AssertionError('FeasibilityyRMSchanged')
        if abs(physical[name+'_asym']['regions']['high3kto8k']['bump_all']['median']-c['all']['high_bump_median'])>1e-8:raise AssertionError('Feasibilitybumpchanged')
    write(out/'FORWARD_BEFORE_MODEL.json',{'plan_sha256':sha(out/'predeclared_plan.json'),'feasible_manifest_sha256':sha(out/'feasible_manifest.json'),
          'created_utc':utc(),'model_invoked':False,'panels':physical,'artifact_sha256':{f.name:sha(f) for f in out.iterdir() if f.is_file()}})
    pilot=ROOT/'submissions/r9_fold0_all5_cpu_r1';sp=json.loads((pilot/'prequery_plan.json').read_text());policy=json.loads((pilot/'frozen_policy.json').read_text())
    if tuple(policy['weights'])!=WEIGHTS:raise ValueError('Weightschanged')
    modelpins,models,binding,tree=inputs(sp['nn_dirs'],local(sp['tree_job']),local(sp['bundle']),local(sp['config']))
    modelpins[str((pilot/'frozen_policy.json').relative_to(ROOT))]=sha(pilot/'frozen_policy.json')
    write(out/'MODEL_EVALUATION_STARTED.json',{'created_utc':utc(),'forward_sha256':sha(out/'FORWARD_BEFORE_MODEL.json'),'pins':modelpins,'models':models,'binding':binding})
    names=list(panels);db=np.concatenate([panels[k]['db'] for k in names]);x,w=relative_grid_v2(db)
    class Query(Dataset):
        def __len__(self):return len(x)
        def __getitem__(self,i):return {'x':x[i],'w':w[i]}
    predictions={k:{} for k in names};timing={};torch.set_num_threads(2)
    for kind in MEMBERS[:-1]:
        model,summary=restore_neural(local(sp['nn_dirs'][kind]),device='cpu',fixture=False,expected_identity=binding['selection_identity'])
        t=time.perf_counter();p=predict_loader(model,DataLoader(Query(),batch_size=128,shuffle=False,num_workers=0),device='cpu',bf16=False,collect_targets=False)
        gap=float(abs(p['share'].astype(float)-(w*p['cell_mask']).sum((1,2))).max())
        if gap>3e-6:raise AssertionError('Predictionphysicalpoolfailed')
        timing[kind]={'seconds':time.perf_counter()-t,'physical_pool_max_error':gap,'epoch':summary['best_window_epoch'],'CPUFP32threads':2}
        for j,k in enumerate(names):predictions[k][kind]={f:p[f][j*n:(j+1)*n].copy() for f in ('share','score','cell_mask')}
        del model,p;gc.collect();print(json.dumps({'stage':'model_done','kind':kind,'elapsed_seconds':time.perf_counter()-start}),flush=True)
    t=time.perf_counter();p=tree.predict(db,batch_size=256);timing['lgbm_physics']={'seconds':time.perf_counter()-t,'CPUthreads':4}
    results={};contrasts={}
    for j,name in enumerate(names):
        ps=predictions[name];ps['lgbm_physics']={f:a[j*n:(j+1)*n] for f,a in p.items()};ps['frozen_all5']=combine(ps,WEIGHTS);ms={}
        for kind,pred in ps.items():
            m=metric(panels[name],pred);pr=pred['share'];m['share_saturation']={'fraction_le001':float(np.mean(pr<=.01)),'fraction_ge099':float(np.mean(pr>=.99))}
            if kind in MEMBERS[:-1]:
                m['region_q']={};ww=w[j*n:(j+1)*n]
                for region,inds in plan['region_indices'].items():
                    q=pred['cell_mask'][:,:,inds];qt=panels[name]['cell_share'][:,:,inds];weight=ww[:,:,inds];den=weight.sum((1,2))
                    pq=(q*weight).sum((1,2))/den;tq=(qt*weight).sum((1,2))/den
                    m['region_q'][region]={'prediction_mean':float(pq.mean()),'true_mean':float(tq.mean()),'bias':float((pq-tq).mean()),
                        'rmse':float(np.sqrt(np.mean((pq-tq)**2))),'cell_fraction_le001':float(np.mean(q<=.01)),'cell_fraction_ge099':float(np.mean(q>=.99))}
            ms[kind]=m
        results[name]=ms;np.savez_compressed(out/(name+'_predictions.npz'),**{kind+'_'+f:a for kind,pred in ps.items() for f,a in pred.items()})
    for c in selected:
        name=c['name'];a=name+'_asym';b=name+'_matched_R9eq6'
        if not np.array_equal(panels[a]['share'],panels[b]['share']):raise AssertionError('Matchedtargetnotexact')
        contrasts[name]={kind:{'RMSE_asym_minus_matched':results[a][kind]['rmse']-results[b][kind]['rmse'],
            'AUC_asym_minus_matched':results[a][kind]['controlled_addition_AUC']-results[b][kind]['controlled_addition_AUC']} for kind in results[a]}
    verify_pins(modelpins);verify_pins(pins)
    report={'format':'r11_all_feasible_models_results_v1','status':'completed','created_utc':utc(),
        'plan_sha256':sha(out/'predeclared_plan.json'),'feasible_manifest_sha256':sha(out/'feasible_manifest.json'),'forward_sha256':sha(out/'FORWARD_BEFORE_MODEL.json'),
        'physical_assumption_change':plan['physical_assumption_change'],'allfour_model_free_feasible_conditions_evaluated':True,
        'evaluations':results,'paired_contrasts':contrasts,'physical_before_model':physical,'timing':timing,'elapsed_seconds':time.perf_counter()-start,
        'model_pins':modelpins,'source_pins':pins,'model_models_code_policy_unchanged':True,'weights':WEIGHTS,
        'no_Test_report_fitting_GPU_SSH_or_production_mutation':True,
        'not_a_Test_label_alignment_proof':True,'selection_error_not_used_to_select_extension_conditions':True}
    write(out/'RESULTS.json',report)
    lines=['# R11 두 물리 목표를 만족한 네 확장 조건: 동결 모델 비교','',
        '**물리 가정이 확장됐습니다.** LF −15/−25/−35 dB뿐 아니라 316–1995 Hz mid도 −10/−20 dB 감쇠합니다. SNR 기준도 3162–7943 Hz로 바뀌었습니다. 원래 LF −10∼−15 dB만으로 목표를 달성했다고 해석하면 안 됩니다.',
        '모델과 무관한 정방향 18조건 중 두 집계 제약을 통과한 네 조건을 모두 평가했습니다. 모델 오차로 한 조건을 선택하지 않았으며 Test 분포와 일치한다는 증거는 아닙니다.',
        '모든 모델과 고정 가중치(4NN 각0.1/tree0.6)를 유지했습니다. 각 대조는 같은 배경·참조·native 시간 지지·실제 y를 갖고 smoothEQ6만 적용한 조건입니다.',
        '', '|조건|모델|패널|정답RMS|예측평균|예측RMS|bias|RMSE|AUC|share≤.01|share≥.99|','|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in selected:
        name=c['name']
        for kind in [*MEMBERS,'frozen_all5']:
            for suffix,tag in [('_asym','asym'),('_matched_R9eq6','matchedEQ6')]:
                m=results[name+suffix][kind];sat=m['share_saturation'];lines.append(f"|{name}|{kind}|{tag}|{m['target_rms']:.5f}|{m['prediction_mean']:.5f}|{m['prediction_rms']:.5f}|{m['bias']:+.5f}|{m['rmse']:.5f}|{m['controlled_addition_AUC']:.5f}|{sat['fraction_le001']:.3f}|{sat['fraction_ge099']:.3f}|")
    lines+=['', '원래 네 조건 결과는 별도 파일에 그대로 있습니다. 이 결과는 개발용 counterfactual challenge이며 자연 선박 정답이나 실제 대회 Test 라벨 정합을 증명하지 않습니다. 공유 참조·기존 selection 배경이라는 한계가 있습니다.',
        '저역·중역·고역 q의 에너지 가중 평균·bias·RMSE·cell saturation과 전체 출처 SHA는 JSON에 있습니다. tree와 ensemble에는 cell-mask 정답을 임의로 만들어 붙이지 않았습니다.',
        '', '[전체 수치와 source/model SHA](RESULTS.json) / [모델 예측 전 물리 요약](FORWARD_BEFORE_MODEL.json) / [네 조건 사전 동결](feasible_manifest.json)']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'stage':'completed','elapsed_seconds':report['elapsed_seconds']}),flush=True)
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'reports/r11_diagnostics/asymmetric_feasible_models_r1');a=p.parse_args()
    try:run(a.output)
    except BaseException as e:
        if a.output.exists():
            from ops.r11_asymmetric_eq_audit import write
            write(a.output/'FAILED.json',{'type':type(e).__name__,'error':str(e),'automatic_retry':False})
        raise
