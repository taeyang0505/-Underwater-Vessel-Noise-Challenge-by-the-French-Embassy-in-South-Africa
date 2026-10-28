"""Frozen-model, CPU-only asymmetric reference-EQ stress audit.

All controls are fixed before predictions. This is a user-specified
diagnostic challenge, never an inferred Test distribution. No fitting,
weight selection, production edits, Test/unused-report/GPU/SSH access.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import gc,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SEED=2026102612
CONDITIONS=(('lf10_hf10',-10.,10.),('lf15_hf10',-15.,10.),
            ('lf10_hf20',-10.,20.),('lf15_hf20',-15.,20.))
WEIGHTS=(.1,.1,.1,.1,.6)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    p=Path(p);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def utc():return datetime.now(timezone.utc).isoformat()
def guard(event,args):
    if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
    p=os.fsdecode(args[0]).replace('\\','/').lower()
    if Path(p).name in {'test.csv','sample_submission.csv','report.npz','outer_validation.npz'} or '/sealed/' in p:
        raise PermissionError('Test/unused report/oldOUTER/sealed values forbidden: '+p)
def qtiles(x):
    import numpy as np
    a=np.asarray(x,float)
    return dict(zip(('min','p10','median','p90','max'),np.quantile(a,[0,.1,.5,.9,1]).tolist())) if len(a) else None

def asymmetric_curve(lf_db):
    import numpy as np
    if lf_db not in (-10.,-15.):raise ValueError('Only predeclared LF factors permitted')
    eq=np.zeros(25);eq[:9]=lf_db;eq[9]=lf_db/2
    return eq

def render(bg,native,spec,lf_db,hf_snr,*,matched_target=None):
    """R9 geometry; modify only spectral response and the explicit scale law.

    HF scale anchor here means nominal300Hz..8kHz (indices10..24), not
    the narrower bump_high diagnostic3.16..7.94kHz (indices20..24).
    alpha is based on the FULL native trajectory peak BEFORE crop, as R9.
    A target-matched control instead uses R9 smoothEQ<=6 and solves alpha
    from that SAME physical whole-added-energy target after placement.
    """
    import numpy as np
    from pipeline.sampler_r9 import place_minutes,smooth_eq,energy_weights
    from pipeline.relative_grid_r6 import quantization_pooling_bound
    x=np.asarray(bg,np.float64);o=np.isfinite(x)
    if x.shape!=(15,25) or not o.all():raise ValueError('This diagnostic requires15completeobservedminutes')
    peak=float(x.max());b=10.**((x-peak)/10.)
    if matched_target is None:eq=asymmetric_curve(float(lf_db))
    else:eq=smooth_eq(np.random.default_rng(spec['eq_seed']),6.)
    transformed=np.asarray(native,np.float64)*10.**(eq[None,:]/10.)
    placed,trajectory=place_minutes(transformed,np.random.default_rng(spec['placement_seed']),spec['family'])
    if spec['no_source']:alpha=0.
    elif matched_target is not None:
        target=float(matched_target)
        if not 0<target<1:raise ValueError('Strict positive target needed')
        alpha=target*b.sum()/((1-target)*placed.sum())
    else:alpha=10.**(hf_snr/10.)*float(np.median(b[:,10:25].sum(1)))/float(transformed[:,10:25].sum(1).max())
    a=alpha*placed;e=b+a;q=a/e;y=float(a.sum()/e.sum())
    raw=10.*np.log10(e)+peak;db=np.round(raw,1).astype(np.float32);w=energy_weights(db)
    identity=abs(float((e/e.sum()*q).sum())-y);rounded_gap=abs(float((w*q).sum())-y)
    ulp=float(np.max(abs(np.spacing(db)))/2);bound=quantization_pooling_bound(.05+ulp+1e-12)
    if identity>1e-10 or rounded_gap>bound+2e-7 or not np.isfinite(db).all():raise AssertionError('Physical pooling/rounding failed')
    if matched_target is not None and not spec['no_source'] and abs(y-matched_target)>1e-10:raise AssertionError('Target-matched control failed')
    regions={'low_le200':slice(0,9),'mid300to3k':slice(10,20),'high3kto8k':slice(20,25),'safe1kto5k':slice(15,23)}
    physical={}
    for name,sl in regions.items():
        es=e[:,sl].sum(1);bs=b[:,sl].sum(1);rs=a[:,sl].sum(1)
        level=10*np.log10(es)+peak
        physical[name]={'bump_db':float(level.max()-np.median(level)),
            'true_region_energy_fraction':float(rs.sum()/es.sum()),
            'added_region_contribution_to_global_y':float(rs.sum()/e.sum()),
            'observed_added_peak_over_background_median_db':float(10*np.log10(rs.max()/np.median(bs))) if rs.max()>0 else None}
    return {'db':db,'unrounded_db':raw,'cell_share':q.astype(np.float32),'share':np.float32(y),'presence':np.int64(y>0)}, {
        'alpha':alpha,'lf_eq_db_requested':lf_db,'mid_high_peak_snr_requested_db':hf_snr,
        'target_matched_control':matched_target is not None,'eq_db':eq.tolist(),'trajectory':trajectory,
        'no_source':bool(spec['no_source']),'raw_identity_error':identity,'rounded_pool_gap':rounded_gap,
        'rounding_bound':bound,'physical_regions':physical}

def fixture():
    import numpy as np
    bg=np.full((15,25),70.);native=np.arange(1,151,dtype=float).reshape(6,25)
    s={'eq_seed':2,'placement_seed':3,'family':'native_inside','no_source':False};gaps=[]
    for _,lf,snr in CONDITIONS:
        a,r=render(bg,native,s,lf,snr);b,br=render(bg,native,s,lf,snr,matched_target=float(a['share']))
        assert abs(float(a['share'])-float(b['share']))<3e-8
        assert (np.array(r['eq_db'])[:9]==lf).all() and (np.array(r['eq_db'])[10:]==0).all()
        z,zr=render(bg,native,{**s,'no_source':True},lf,snr);assert z['share']==0 and not z['cell_share'].any()
        gaps.append(r['raw_identity_error'])
    return {'status':'passed','four_factorial_cases':4,'matched_target_cases':4,'zero_cases':4,'max_raw_identity_error':max(gaps)}

def panel_summary(data,recipes):
    import numpy as np
    y=data['share'].astype(float);positive=y>0
    out={'n':len(y),'positive':int(positive.sum()),'no_source':int((~positive).sum()),
         'target_mean':float(y.mean()),'target_rms':float(np.sqrt(np.mean(y*y))),
         'target_quantiles':qtiles(y),'regions':{},
         'raw_pool_identity_max_error':max(v['raw_identity_error'] for v in recipes),
         'rounded_pool_label_gap_max':max(v['rounded_pool_gap'] for v in recipes)}
    for name in recipes[0]['physical_regions']:
        reg=[r['physical_regions'][name] for r in recipes]
        out['regions'][name]={'bump_all':qtiles([v['bump_db'] for v in reg]),
            'bump_positive':qtiles([reg[i]['bump_db'] for i in np.flatnonzero(positive)]),
            'true_region_energy_fraction_mean':float(np.mean([v['true_region_energy_fraction'] for v in reg])),
            'added_region_contribution_global_y_mean':float(np.mean([v['added_region_contribution_to_global_y'] for v in reg]))}
    out['specified_target_y_RMS_015_to_020_met']=.15<=out['target_rms']<=.20
    out['bump_high_median_within16plusminus2']=14<=out['regions']['high3kto8k']['bump_all']['median']<=18
    out['both_diagnostic_targets_met']=out['specified_target_y_RMS_015_to_020_met'] and out['bump_high_median_within16plusminus2']
    return out

def prepare(args):
    import numpy as np
    from pipeline.dataset_r9 import R9Sources,make_schedule,DATA_CODE
    from pipeline.sampler_r9 import energy_weights
    out=args.output.resolve()
    if out.exists():raise FileExistsError('Fresh diagnostic directory required')
    out.mkdir(parents=True);started=time.perf_counter()
    sources=R9Sources();meta=sources.role_metadata(0,'selection');complete=meta[meta.qc_observed_minutes.eq(15)].reset_index(drop=True)
    if len(complete)<1000:raise ValueError('Need1000distinctcompleteobservations')
    selected=complete.iloc[np.random.default_rng(SEED).choice(len(complete),1000,replace=False)].reset_index(drop=True)
    names=sorted(set(DATA_CODE)|{'ops/r11_asymmetric_eq_audit.py','ops/infer_r9_fold0_all5_pilot.py','pipeline/relative_grid_r6.py'})
    pins={n:sha(ROOT/n) for n in names}
    for p in [sources.bg_dir/'fold_00/selection_manifest.parquet',sources.ref_dir/'reference_manifest.parquet',ROOT/'configs/r9_full_training_v1.json']:
        pins[str(p.relative_to(ROOT))]=sha(p)
    plan={'format':'r11_asymmetric_eq_plan_v1','created_utc':utc(),'seed':SEED,'n':1000,
          'source':'Fold0selectionmetadata; complete15minutesafterminuteQC; uniqueindices; noTestmatching',
          'conditions':[{'name':n,'LF_EQ_db':l,'mid_high_full_native_peak_SNR_db':s} for n,l,s in CONDITIONS],
          'reference_policy':'all509admittedwhole-recordingreference; cycles independentofclass/modelerror',
          'families':['native_inside','native_short','native_edge'],'family_probabilities':[.5,.25,.25],
          'no_source_fraction':.2,'placement':'nativeR9geometry;notimewarp;fullnativepeakbeforecrop',
          'EQ_definition':'indices0..8 gain LFfactor; index9 halfgain; indices10..24 zeroEQ; alpha anchors reference10..24peak to medianbackground10..24 power',
          'band_centres_Hz':(10.**(1.5+.1*np.arange(25))).tolist(),
          'region_indices':{'low_le200':list(range(9)),'mid300to3k':list(range(10,20)),
             'mid_high300to8k':list(range(10,25)),'high3kto8k':list(range(20,25)),'safe1kto5k':list(range(15,23))},
          'bump_definition':'10log10(sum linearbandenergy per minute): max over15 - median over15; measured on raw mixture; all andpositive separately',
          'model_free_target_checks':{'bump_high_median_goal':16,'report_tolerance_db':2,'actual_y_RMS_interval':[.15,.20]},
          'matched_controls':'ForEACHasymcondition,sameB/ref/nativegeometry withR9smoothEQ6; alpha solvedtoSAMErawphysicaltarget. ThiscontrolisnotstandardR9SNRdistribution.',
          'selection_policy':'evaluateallfourpredeclaredconditionsandallfourpairedcontrols;nonechosenusingmodeloutputs',
          'code_and_source_pins':pins,'source_identity':sources.identity,
          'model_weights':[.1,.1,.1,.1,.6],'model_threads':2,'tree_threads':4,
          'not_Test_distribution_estimate':True,'no_fitting_or_production_changes':True}
    write(out/'predeclared_plan.json',plan);write(out/'numeric_fixture.json',fixture());selected.to_parquet(out/'background_source_manifest.parquet',index=False)
    sched=make_schedule(1000,1,0,SEED,tag=1112);specs=[]
    for i in range(1000):
        r=np.random.default_rng(np.random.SeedSequence([SEED,i,511]))
        specs.append({'eq_seed':int(r.integers(2**31)),'placement_seed':int(r.integers(2**31)),
                      'family':sources.config['generator']['family_names'][int(sched['family_index'][i])],
                      'no_source':bool(sched['explicit_no_source'][i]),
                      'ref':int(sched['reference_index'][i]),'row':int(sched['primary_order'][i])})
    write(out/'shared_specs.json',specs);forward={}
    for name,lf,snr in CONDITIONS:
        for control in (False,True):
            arm=name+('_matched_R9eq6' if control else '_asym')
            data={k:np.empty((1000,15,25),np.float64 if k=='unrounded_db' else np.float32) for k in ('db','unrounded_db','cell_share')}
            data.update(share=np.empty(1000,np.float32),presence=np.empty(1000,np.int64),base_window_index=np.empty(1000,np.int64),reference_index=np.empty(1000,np.int64))
            recipes=[]
            targets=None
            if control:
                with np.load(out/(name+'_asym_samples.npz'),allow_pickle=False) as z:targets=z['share'].copy()
            for i,spec in enumerate(specs):
                row=selected.iloc[spec['row']];native=sources.reference_minutes[max(spec['ref'],0)]
                d,r=render(sources.background(row),native,spec,lf,snr,
                           matched_target=None if not control else float(targets[i]))
                for k in ('db','unrounded_db','cell_share','share','presence'):data[k][i]=d[k]
                data['base_window_index'][i]=int(row.base_window_index);data['reference_index'][i]=spec['ref'];recipes.append(r)
            np.savez_compressed(out/(arm+'_samples.npz'),**data);write(out/(arm+'_recipes.json'),recipes)
            forward[arm]=panel_summary(data,recipes)
    write(out/'FORWARD_BEFORE_MODEL.json',{'format':'r11_asymmetric_forward_v1','created_utc':utc(),'model_outputs_read':False,
        'plan_sha256':sha(out/'predeclared_plan.json'),'panels':forward,'elapsed_seconds':time.perf_counter()-started,
        'any_asym_condition_meets_both_targets':any(forward[n+'_asym']['both_diagnostic_targets_met'] for n,_,_ in CONDITIONS),
        'data_artifact_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()}})
    for n,d in forward.items():print(json.dumps({'forward_panel':n,'y_RMS':d['target_rms'],'bump_high_median':d['regions']['high3kto8k']['bump_all']['median'],'both_requested_targets_met':d['both_diagnostic_targets_met']}),flush=True)
    return forward

def metric(data,pred):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    y=data['share'].astype(float);z=data['presence'];p=pred['share'].astype(float);e=p-y
    out={'n':len(y),'target_mean':float(y.mean()),'target_rms':float(np.sqrt(np.mean(y*y))),
         'prediction_mean':float(p.mean()),'prediction_rms':float(np.sqrt(np.mean(p*p))),
         'bias':float(e.mean()),'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(abs(e).mean()),
         'controlled_addition_AUC':float(roc_auc_score(z,pred['score']))}
    return out

def infer(args):
    from ops.infer_r9_fold0_all5_pilot import environment,inputs,local,verify_pins,combine,MEMBERS
    environment(4)
    import numpy as np,torch
    from torch.utils.data import Dataset,DataLoader
    from pipeline.runner_r9 import restore_neural
    from pipeline.relative_grid_r6_v2 import relative_grid_v2
    from pipeline.training_r9 import predict_loader
    out=args.output.resolve();start=time.perf_counter()
    if (out/'RESULTS.json').exists() or (out/'MODEL_EVALUATION_STARTED.json').exists():raise FileExistsError('Do not repeat/adapt completed or partial model evaluation')
    forward=json.loads((out/'FORWARD_BEFORE_MODEL.json').read_text());plan=json.loads((out/'predeclared_plan.json').read_text())
    if forward['plan_sha256']!=sha(out/'predeclared_plan.json'):raise ValueError('ForwardplanSHAchanged')
    for name,digest in forward['data_artifact_sha256'].items():
        if sha(out/name)!=digest:raise ValueError('Forwardartifactchanged:'+name)
    verify_pins(plan['code_and_source_pins'])
    pilot=ROOT/'submissions/r9_fold0_all5_cpu_r1';sp=json.loads((pilot/'prequery_plan.json').read_text());policy=json.loads((pilot/'frozen_policy.json').read_text())
    if tuple(policy['weights'])!=WEIGHTS:raise ValueError('Frozenall5weightsdiffer')
    pins,modelpins,binding,tree=inputs(sp['nn_dirs'],local(sp['tree_job']),local(sp['bundle']),local(sp['config']))
    pins[str((pilot/'frozen_policy.json').relative_to(ROOT))]=sha(pilot/'frozen_policy.json')
    if tree.settings['threads']!=4:raise ValueError('Preserveoriginaltree4threads')
    write(out/'MODEL_EVALUATION_STARTED.json',{'created_utc':utc(),'forward_file_sha256':sha(out/'FORWARD_BEFORE_MODEL.json'),
          'source_pins':pins,'model_pins':modelpins,'binding':binding,'weights':WEIGHTS,'threadsNN':2,'threadsTree':4})
    names=list(forward['panels']);panels={}
    for name in names:
        with np.load(out/(name+'_samples.npz'),allow_pickle=False) as z:panels[name]={k:z[k].copy() for k in z.files}
    db=np.concatenate([panels[n]['db'] for n in names]);x,w=relative_grid_v2(db)
    class Query(Dataset):
        def __len__(self):return len(x)
        def __getitem__(self,i):return {'x':x[i],'w':w[i]}
    pred_by_panel={n:{} for n in names};timing={};torch.set_num_threads(2)
    for kind in MEMBERS[:-1]:
        model,summary=restore_neural(local(sp['nn_dirs'][kind]),device='cpu',fixture=False,expected_identity=binding['selection_identity'])
        t=time.perf_counter();pred=predict_loader(model,DataLoader(Query(),batch_size=128,shuffle=False,num_workers=0),device='cpu',bf16=False,collect_targets=False)
        gap=float(np.max(abs(pred['share'].astype(float)-(w*pred['cell_mask']).sum((1,2)))))
        if gap>3e-6:raise AssertionError('NNphysicalpoolfailed')
        timing[kind]={'seconds':time.perf_counter()-t,'physical_pool_max_abs':gap,'CPU_FP32_threads':2,'epoch':summary['best_window_epoch']}
        for j,name in enumerate(names):pred_by_panel[name][kind]={k:pred[k][j*1000:(j+1)*1000].copy() for k in ('share','score','cell_mask')}
        del model,pred;gc.collect();print(json.dumps({'model_completed':kind,**timing[kind],'elapsed_seconds':time.perf_counter()-start}),flush=True)
    t=time.perf_counter();tp=tree.predict(db,batch_size=256);timing['lgbm_physics']={'seconds':time.perf_counter()-t,'threads':4}
    results={};contrasts={}
    for j,name in enumerate(names):
        preds=pred_by_panel[name];preds['lgbm_physics']={k:v[j*1000:(j+1)*1000] for k,v in tp.items()};preds['frozen_all5']=combine(preds,WEIGHTS)
        metrics={k:metric(panels[name],v) for k,v in preds.items()}
        for kind in MEMBERS[:-1]:
            regions={};ww=w[j*1000:(j+1)*1000];q=preds[kind]['cell_mask'];qt=panels[name]['cell_share']
            for region,inds in plan['region_indices'].items():
                r=np.asarray(inds);den=ww[:,:,r].sum((1,2));a=(ww[:,:,r]*q[:,:,r]).sum((1,2))/den;b=(ww[:,:,r]*qt[:,:,r]).sum((1,2))/den
                regions[region]={'pred_weighted_fraction_mean':float(a.mean()),'true_rounded_weighted_fraction_mean':float(b.mean()),
                    'bias':float((a-b).mean()),'rmse':float(np.sqrt(np.mean((a-b)**2)))}
            metrics[kind]['region_cell_mask_diagnostics']=regions
        results[name]=metrics
        values={f'{kind}_{k}':v for kind,p in preds.items() for k,v in p.items()}
        np.savez_compressed(out/(name+'_predictions.npz'),**values)
    for n,_,_ in CONDITIONS:
        a=n+'_asym';b=n+'_matched_R9eq6'
        if not np.allclose(panels[a]['share'],panels[b]['share'],atol=3e-8,rtol=0):raise AssertionError('Pairedtargetsmismatch')
        contrasts[n]={k:{'asym_minus_matched_RMSE':results[a][k]['rmse']-results[b][k]['rmse'],
            'asym_minus_matched_AUC':results[a][k]['controlled_addition_AUC']-results[b][k]['controlled_addition_AUC'],
            'paired_prediction_difference_RMS':float(np.sqrt(np.mean((pred_by_panel[a][k]['share'].astype(float)-pred_by_panel[b][k]['share'])**2)))} for k in results[a]}
    verify_pins(pins);verify_pins(plan['code_and_source_pins'])
    report={'format':'r11_asymmetric_eq_results_v1','status':'completed','created_utc':utc(),
        'predeclared_plan_sha256':sha(out/'predeclared_plan.json'),'forward_before_model_sha256':sha(out/'FORWARD_BEFORE_MODEL.json'),
        'model_free_forward_summary':forward['panels'],'evaluations':results,'paired_target_matched_contrasts':contrasts,
        'timings':timing,'elapsed_seconds':time.perf_counter()-start,'weights':WEIGHTS,
        'source_models_policy_code_unchanged':True,'no_Test_report_GPU_SSH_fitting_or_tuning':True,
        'interpretation':'User-specifiedasymmetricEQchallenge,notestimatedTestlaw. Target-matchedcontrolsisolate spectralresponseundermatchinggeometry/energytargets,notnaturalvesselgroundtruth.',
        'all_source_pins':pins,'all_model_pins':modelpins}
    write(out/'RESULTS.json',report)
    lines=['# R11 비대칭 EQ 진단','',
        '동결된 Fold0 4NN + LightGBM에 고정 가중치(각 NN 0.1, tree 0.6)를 적용했습니다. 학습·가중치 변경·Test 사용은 없습니다.',
        '저역 31.6–199.5 Hz는 추가 참조에만 −10/−15 dB EQ, 251.2 Hz는 절반 감쇠, 316.2–7943.3 Hz는 EQ 0 dB입니다. α는 이 중고역의 전체 native peak를 배경 중앙값 대비 +10/+20 dB로 맞췄습니다.',
        'bump_high는 3162.3–7943.3 Hz의 분당 선형 합 에너지를 dB로 바꾼 뒤 15분 max−median입니다. safe 범위는 1000–5011.9 Hz, mid는 316.2–2511.9 Hz입니다.',
        '각 비대칭 패널에는 같은 실제 y·배경·참조·시간 배치를 갖는 smooth-EQ6 대조를 붙였습니다. 대조의 α는 목표 y로 풀었으므로 표준 R9 SNR 분포를 그대로 재현한 baseline은 아닙니다.',
        '', '|패널|실제 y RMS|고역 bump 중앙값|목표 동시충족|앙상블 RMSE|예측 RMS|bias|AUC|','|---|---:|---:|---|---:|---:|---:|---:|']
    for name in names:
        f=forward['panels'][name];m=results[name]['frozen_all5'];lines.append(f"|{name}|{f['target_rms']:.5f}|{f['regions']['high3kto8k']['bump_all']['median']:.3f}|{f['both_diagnostic_targets_met']}|{m['rmse']:.5f}|{m['prediction_rms']:.5f}|{m['bias']:+.5f}|{m['controlled_addition_AUC']:.5f}|")
    lines+=['', '이 조건이 실제 Test의 생성 법칙 또는 분포라는 증거는 아닙니다. 목표 bump·y 범위를 못 맞춘 조건도 그대로 보존했습니다. 모든 모델별 수치와 저역/중역/고역 q 진단은 JSON에 있습니다.',
            '같은 selection 원천과 공유 참조를 사용하는 개발용 stress test입니다. q와 y는 실제로 추가한 전체 녹음의 에너지 비율이며 자연 선박만의 분리 정답이 아닙니다.',
            '', '[전체 모델별 결과](RESULTS.json) / [모델 예측 전 물리 요약](FORWARD_BEFORE_MODEL.json) / [사전 계획](predeclared_plan.json)']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'stage':'completed','elapsed_seconds':report['elapsed_seconds'],'output':str(out)}),flush=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','infer','all'])
    p.add_argument('--output',type=Path,default=ROOT/'reports/r11_diagnostics/asymmetric_eq_r1');a=p.parse_args()
    sys.addaudithook(guard)
    try:
        if a.phase in ('prepare','all'):prepare(a)
        if a.phase in ('infer','all'):infer(a)
    except BaseException as e:
        if a.output.exists():write(a.output/'FAILED.json',{'type':type(e).__name__,'error':str(e),'automatic_retry':False})
        raise

if __name__=='__main__':main()
