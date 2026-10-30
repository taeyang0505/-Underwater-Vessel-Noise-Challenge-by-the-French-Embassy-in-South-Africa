#!/usr/bin/env python3
"""Separate, replayable postprocessing of frozen R16 predictions. No training.

The original CUDA run/finalizer is never changed. A fresh protocol is written
before reading gate predictions. Report/query reads are blocked until the new
decision is frozen. Legacy report-source overlap is explicitly disclosed.
"""
from pathlib import Path
import argparse, csv, datetime, gc, hashlib, json, os, sys, time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pipeline.r16_multisupport_data import PARENT, REGIMES, read, write, sha, save, load, role

NAME='r16_postprocess_r1'
OUT=ROOT/'models'/NAME
BASE=ROOT/'models/r16_pretrained_cuda_bf16_r1/both/final'
EPS=(0.,.005,.01,.02)
FAMILIES=('dasheng_base','dinov2_small','timemoe')
MEMBERS=('lightgbm5','catboost5',*FAMILIES)
PANELS=('local_gate','domain_holdout','native12_VS_hf_high_negative','native12_VS_st_francis_dense_negative')
CODE_FILES=('ops/run_r16_postprocess.py','ops/r16_postprocess_math.py',
            'ops/r16_postprocess_select.py','ops/render_r16_postprocess_report.py')

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def protocol():
    return dict(format='r16_postprocess_protocol_v1',created_at=now(),no_retraining=True,
      original_run_immutable=True,original_finalizer_immutable=True,
      fit_rows=1206,checkpoint_selection_rows=1194,fit_only_parameters=True,
      epsilon_candidates=list(EPS),D0_H0_candidates=[[.5,.5],[.6,.4],[.4,.6]],
      score_transforms=['probability','standardized_logit','ecdf'],
      score_logit_clip=1e-6,score_share_weight=0.,query_cohort_statistics=False,
      tree_fit_cat_mass_candidates=[0.,.25,.5],
      comparison='each single intervention against immutable baseline; no cross-product search',
      regression_selection='Among gate-eligible independent candidates, lowest fit RMSE. Identity if none.',
      regression_gate=dict(strict_improvement=1e-6,nondegradation_tolerance=1e-6,
        axes=list(PANELS),subgroups='local_gate positive support<=3, positive support>=12, 0<y<=.03; native12 positive support>=12; n>=20',
        criteria='overall local_gate RMSE improves; no axis RMSE or absolute bias worsens; stretch also cannot worsen underprediction in weak/long positives'),
      safety_gate='Fit-only Tree reference. Fall back when any full axis RMSE is worse than Tree by >1e-6; no weights retuned on panels.',
      classification_gate='fit-fixed finite weighting/transform candidates; local_gate AUC strict improvement and no full-axis AUC degradation >1e-6',
      report_policy='No report performance read until DECISION_FREEZE. Report cannot change selected policy.',
      report_limitations=['Legacy panels repeatedly used in earlier development.',
        'Immutable baseline finalizer reads local_report before this separate comparison.',
        'native12/HF/SF selection panels share report background groups: report is fixed-policy confirmation, not an unbiased independent test.',
        'Additional exact-background-disjoint report subset is descriptive only; it is not group/24h independent.'],
      legacy_tree_ES='Preserved LightGBM used the whole legacy local_fit for early stopping; fit scores are development metrics.',
      stretched_numerics='Apply paired same-forward epsilon-minus-identity delta to stored FP32 shares; preserve identity exactly; report reconstruction drift; reject drift>0.002.',
      code_sha256={k:sha(ROOT/k) for k in CODE_FILES},official_upload=False,
      query_labels_or_LB_used=False,Test_used_for_choice=False)

def access_guard(state):
    base=str(BASE.resolve()).lower()
    def guard(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=str(Path(os.fsdecode(args[0])).resolve()).lower().replace('\\','/')
        if '/eda_deep/' in p or '/r15_band_comparison_' in p:raise PermissionError('No query-derived EDA for postprocessing selection')
        if (p.endswith('/test.csv') or p.endswith('/query_predictions.npz') or p.endswith('/test_prediction_audit.json')) and state['phase']!='query':
            raise PermissionError('Query locked until policy freeze and final report confirmation')
        if ('/local_report/' in p or p.endswith('/evaluation/local_report.npz') or p in [base+'/'+n for n in ('evaluation.json','results.md','progress.json','index.html')]) and state['phase'] not in ('report','query'):
            raise PermissionError('Report performance locked until policy freeze')
    return guard

def join(parts):
    keys=set.intersection(*(set(d) for d in parts))
    return {k:np.concatenate([d[k] for d in parts]) for k in keys}

def panel_data(name,freeze):
    from ops.r16_support_temporal_data import load_panel,load_long_panel
    from ops.r16_forward_data import OPERATORS
    if name=='selection':return role(0,'D0','selection'),None
    if name in ('local_gate','local_report','domain_holdout'):
        return join([load_panel(PARENT,name,op,freeze_receipt=freeze) for op in OPERATORS]),None
    if name.startswith('native12_VS_'):
        pos=join([load_long_panel(PARENT,'native12_positive',operator=op,freeze_receipt=freeze) for op in OPERATORS])
        neg=load_long_panel(PARENT,name.split('_VS_',1)[1],freeze_receipt=freeze)
        w=np.r_[np.full(len(pos['share']),.5/len(pos['share'])),np.full(len(neg['share']),.5/len(neg['share']))]
        return join([pos,neg]),w
    raise ValueError(name)

def attach(pred,d,weights=None):
    ids=d['sample_id'].astype(str)
    if not np.array_equal(pred['sample_id'].astype(str),ids):raise ValueError('Panel ID/order mismatch')
    if not np.array_equal(pred['target_share'],d['share']):raise ValueError('Panel labels differ')
    out={**pred,'support_minutes':(d['cell_share']>0).any(2).sum(1)}
    for key in ('base_window_index','site','background_group','recorder','operator'):
        if key in d:out[key]=d[key]
    if weights is not None:out['weights']=weights
    return out

def collect_stretch(datasets,predictions,device,batch,progress):
    """One frozen regression forward per model/panel; no optimizer/backprop."""
    import torch
    import ops.finalize_r16_pretrained_cuda_bf16 as original
    from pipeline.training_r16_pretrained_cuda_bf16 import branch,base_fingerprint,OPS
    from pipeline.training_r16_multisupport import features
    from pipeline.r16_cuda_precision import autocast_context
    original.configure_variant('both');plan=read(OPS/'RUN_PLAN.json');torch.set_num_threads(2)
    # Selection and gate only here; query/report have separate calls after freeze.
    inputs={};result={};drifts=[]
    for name,d in datasets.items():
        x,w=features(d['db']);inputs[name]=(x,w)
        result[name]={**predictions[name]}
        for r in REGIMES:result[name]['stretch_'+r]=np.repeat(predictions[name]['share_'+r][:,None,:],len(EPS),axis=1).astype(float)
    for r in REGIMES:
        for family in FAMILIES:
            folder=original.RUN/family/r/'regression';cp=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
            summary=read(folder/'summary.json');original.check_checkpoint(cp,summary,family,r,'regression',plan)
            model,prov=branch(family,'regression',device)
            for key in ('model_id','revision','weight_sha256'):
                if prov[key]!=cp['provenance'][key]:raise ValueError('Public weight identity changed')
            if base_fingerprint(model)!=cp['permanent_base_fingerprint']:raise ValueError('Frozen public base changed')
            model.load_adapter_state(cp['adapter_state']);model.eval();j=MEMBERS.index(family)
            for name,(x,w) in inputs.items():
                accum=[]
                with torch.inference_mode():
                    for off in range(0,len(x),batch):
                        xx=torch.as_tensor(x[off:off+batch].copy(),device=device)
                        ww=torch.as_tensor(w[off:off+batch].copy(),device=device).float()
                        with autocast_context(device):logits=model(xx)
                        qq=logits.float().sigmoid()
                        pooled=torch.stack([((qq-e).div(1-e).clamp(0,1)*ww).sum((1,2)) for e in EPS],1)
                        if not torch.isfinite(pooled).all():raise FloatingPointError('Nonfinite stretched inference')
                        accum.append(pooled.cpu().numpy())
                a=np.concatenate(accum).astype(float);saved=predictions[name]['share_'+r][:,j].astype(float)
                drift=float(abs(a[:,0]-saved).max());drifts.append(dict(panel=name,regime=r,family=family,max_identity_reconstruction_drift=drift))
                if drift>.002:raise ValueError('Stored prediction reconstruction drift exceeds fixed tolerance')
                corrected=np.clip(saved[:,None]+a-a[:,[0]],0,1);corrected[:,0]=saved
                if (np.diff(corrected,axis=1)>1e-7).any():raise ValueError('Stretch is not monotone')
                result[name]['stretch_'+r][:,:,j]=corrected
                write(progress,dict(phase='collecting_stretch',panel=name,family=family,regime=r,updated_at=now()))
            del model,cp;gc.collect()
            if device=='cuda':torch.cuda.empty_cache()
            elif device=='mps':torch.mps.empty_cache()
    return result,drifts

def stats(a):
    a=np.asarray(a,float);q=np.quantile(a,[0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1])
    return dict(mean=float(a.mean()),std=float(a.std()),rms=float(np.sqrt(np.mean(a*a))),median=float(q[5]),
      **{k:float(v) for k,v in zip(('min','p01','p05','p10','p25','p50','p75','p90','p95','p99','max'),q)})

def subset(panel,ix):
    n=len(panel['target_share']);return{k:v[ix] if isinstance(v,np.ndarray) and v.ndim and len(v)==n else v for k,v in panel.items()}

def family_metrics(name,panel):
    from ops.r16_postprocess_math import summary_metrics
    rows=[];y=panel['target_share'];w=panel.get('weights')
    for r in REGIMES:
        for j,f in enumerate(MEMBERS):
            rows.append(dict(dataset=name,regime=r,family=f,**summary_metrics(y,panel['share_'+r][:,j],panel['classifier_'+r][:,j],w)))
    # Regime 50:50 comparison for each family, not an independently optimized policy.
    for j,f in enumerate(MEMBERS):
        rows.append(dict(dataset=name,regime='D0/H0 50:50',family=f,**summary_metrics(y,.5*(panel['share_D0'][:,j]+panel['share_H0'][:,j]),.5*(panel['classifier_D0'][:,j]+panel['classifier_H0'][:,j]),w)))
    return rows

def run(device='cuda',batch=32):
    from ops.r16_postprocess_select import select,apply_policy
    from ops.r16_postprocess_math import summary_metrics
    from ops.render_r16_postprocess_report import render
    from ops.finalize_r16_production import write_submission
    # Completion is required before creating the experiment output. No partial reuse.
    if not (BASE/'COMPLETED.json').is_file():raise RuntimeError('Immutable baseline finalization not completed')
    if OUT.exists():raise FileExistsError('Fresh version required; no partial experiment reuse')
    OUT.mkdir(parents=True);state={'phase':'selection'};sys.addaudithook(access_guard(state))
    started=time.perf_counter();registered=ROOT/'ops/r16_postprocess_preregistration_r1.json'
    pro=read(registered)
    if pro.get('format')!='r16_postprocess_protocol_v1':raise ValueError('Registered protocol required')
    write(OUT/'PROTOCOL.json',pro);prosha=sha(OUT/'PROTOCOL.json')
    if prosha!=sha(registered):raise ValueError('Preregistration bytes changed during copy')
    original_done=read(BASE/'COMPLETED.json');baseline=read(BASE/'policy.json')
    if baseline['member_order']!=list(MEMBERS) or baseline['D0_H0_weights']!=[.5,.5]:raise ValueError('Wrong baseline family/DH contract')
    pins=read(BASE/'GLOBAL_FREEZE.json')['artifact_sha256']
    from pipeline.training_r16_multisupport import verify
    verify(pins);verify(pro['code_sha256'])
    original_bytes={str(p.relative_to(ROOT)):sha(p) for p in (BASE/'policy.json',BASE/'COMPLETED.json')}
    # Receipt metadata only. Original query predictions/CSV bytes stay unread now.
    original_bytes[original_done['submission']['path']]=original_done['submission']['sha256']
    freeze=BASE/'GLOBAL_FREEZE.json';datasets={};preds={};prediction_inputs={}
    for name in ('selection',*PANELS):
        d,w=panel_data(name,freeze);datasets[name]=d
        path=BASE/('selection_predictions.npz' if name=='selection' else 'evaluation/'+name+'.npz')
        prediction_inputs[str(path.relative_to(ROOT))]=sha(path)
        preds[name]=attach(load(path),d,w)
    full,drifts=collect_stretch(datasets,preds,device,batch,OUT/'progress.json')
    sel=full['selection'];ei=sel['ensemble_fit_row_indices'];ci=sel['checkpoint_selection_row_indices']
    if (len(ei)!=1206 or len(ci)!=1194 or len(np.unique(ei))!=len(ei) or len(np.unique(ci))!=len(ci)
        or set(ei)&set(ci) or set(ei)|set(ci)!=set(range(len(sel['target_share'])))):
        raise ValueError('Wrong fit/checkpoint split')
    bundle={'fit':subset(sel,ei),**{k:full[k] for k in PANELS}}
    verify(pro['code_sha256']);chosen=select(bundle,baseline)
    for name,d in full.items():save(OUT/'predictions'/f'{name}.npz',d)
    write(OUT/'SELECTION_RESULTS.json',chosen)
    decision=dict(format='r16_postprocess_decision_v1',frozen_at=now(),protocol_sha256=prosha,
      policy=chosen['policies']['selected'],baseline_policy_sha256=sha(BASE/'policy.json'),
      selection_results_sha256=sha(OUT/'SELECTION_RESULTS.json'),
      source_sha256=pro['code_sha256'],input_prediction_sha256=prediction_inputs.copy(),Test_used=False,local_report_performance_used=False)
    write(OUT/'DECISION_FREEZE.json',decision);decisionsha=sha(OUT/'DECISION_FREEZE.json')
    # Final confirmation: same frozen policy regardless of results.
    verify(pro['code_sha256']);verify(prediction_inputs);state['phase']='report';d,w=panel_data('local_report',freeze)
    report_path=BASE/'evaluation/local_report.npz';prediction_inputs[str(report_path.relative_to(ROOT))]=sha(report_path)
    raw=attach(load(report_path),d,w)
    report,reportdrift=collect_stretch({'local_report':d},{'local_report':raw},device,batch,OUT/'progress.json')
    rp=report['local_report'];save(OUT/'predictions/local_report.npz',rp);drifts+=reportdrift
    rows=[]
    for label,policy in chosen['policies'].items():
        out=apply_policy(rp,policy)
        rows.append(dict(panel='local_report',policy=label,**summary_metrics(rp['target_share'],out['share'],out['score'])))
    selected_bg=set(np.concatenate([full[k]['base_window_index'] for k in PANELS if 'base_window_index' in full[k]]).tolist())
    if 'base_window_index' in rp:
        keep=~np.isin(rp['base_window_index'],list(selected_bg))
        if keep.any():
            rr=subset(rp,keep)
            for label,policy in chosen['policies'].items():
                out=apply_policy(rr,policy);rows.append(dict(panel='local_report_exact_background_disjoint_only',policy=label,
                  not_group_or_24h_independent=True,**summary_metrics(rr['target_share'],out['share'],out['score'])))
    write(OUT/'REPORT_CONFIRMATION.json',dict(rows=rows,policy_changed=False,decision_sha256=decisionsha))
    if sha(OUT/'DECISION_FREEZE.json')!=decisionsha:raise ValueError('Policy changed during report')
    # Query inference-only after policy and report confirmation. No cohort transforms.
    verify(pro['code_sha256']);state['phase']='query'
    from ops.predict_r16_production_nn import read_raw_csv
    ids,db,qreceipt=read_raw_csv(ROOT/'data/raw/Test.csv')
    query_path=BASE/'query_predictions.npz';prediction_inputs[str(query_path.relative_to(ROOT))]=sha(query_path)
    qp=load(query_path)
    if not np.array_equal(ids.astype(str),qp['sample_id'].astype(str)):raise ValueError('Query IDs differ')
    # Collector needs only db; no query target values ever enter selection or scoring.
    qm,qd=collect_stretch({'query':{'db':db}},{'query':qp},device,batch,OUT/'progress.json');drifts+=qd;query=qm['query']
    save(OUT/'query_predictions.npz',query)
    submissions={};subroot=ROOT/'submissions'/NAME
    import shutil
    original_csv=ROOT/original_done['submission']['path'];copy=subroot/'submission_r16_original_frozen.csv'
    copy.parent.mkdir(parents=True,exist_ok=True)
    if copy.exists():raise FileExistsError(copy)
    shutil.copyfile(original_csv,copy)
    if sha(copy)!=sha(original_csv):raise ValueError('Original CSV copy differs')
    baseout=apply_policy(query,chosen['policies']['baseline'])
    if max(abs(baseout[k]-qp[k]).max() for k in ('share','score'))>1e-7:raise ValueError('Baseline reconstruction differs')
    submissions['original']=dict(path=str(copy.relative_to(ROOT)),sha256=sha(copy),rows=len(ids),
      original_bytes_identical=True,statistics={k:stats(baseout[k]) for k in baseout})
    selected=apply_policy(query,decision['policy'])
    changed=any(float(abs(selected[k]-baseout[k]).max())>1e-12 for k in baseout)
    if changed:
        receipt=write_submission(subroot/'submission_r16_selected_policy.csv',ids,selected)
        submissions['selected']={**receipt,'statistics':{k:stats(selected[k]) for k in selected}}
    else:submissions['selected']=dict(status='identity retained; original CSV is selected candidate')
    # Re-read every emitted CSV, independently check schema, ordering, finite/bounds.
    for name,rec in submissions.items():
        if 'path' not in rec:continue
        with (ROOT/rec['path']).open(newline='') as f:
            reader=csv.DictReader(f);table=list(reader)
            if reader.fieldnames!=['ID','share_added_vessel','score_added_vessel']:raise ValueError('Submission columns differ')
        if len(table)!=1500 or [r['ID'] for r in table]!=ids.astype(str).tolist() or len({r['ID'] for r in table})!=1500:raise ValueError('Submission IDs/rows differ')
        vv=np.asarray([[float(r[k]) for k in ('share_added_vessel','score_added_vessel')] for r in table])
        if not np.isfinite(vv).all() or (vv<0).any() or (vv>1).any():raise ValueError('Invalid submission values')
        rec['integrity']=dict(rows=1500,unique_IDs=1500,ID_order_preserved=True,finite=True,in_unit_interval=True,schema_verified=True)
    family=[]
    for name,pp in [('checkpoint_selection',subset(sel,ci)),('ensemble_fit',subset(sel,ei)),('local_gate',full['local_gate'])]:family+=family_metrics(name,pp)
    result=dict(status='completed',completed_at=now(),baseline_family_metrics=family,**chosen,final_report=rows,
      submissions=submissions,numerical_reconstruction=drifts,
      provenance=dict(protocol_sha256=prosha,decision_sha256=decisionsha,baseline_files=original_bytes,
        local_report_unbiased_independent=False,report_limitations=pro['report_limitations'],
        original_baseline_policy_sha256=sha(BASE/'policy.json'),input_prediction_sha256=prediction_inputs,no_retraining=True,official_upload=False,
        query_source=qreceipt,policy_changed_after_report_or_query=False,elapsed_seconds=time.perf_counter()-started))
    for path,h in original_bytes.items():
        if sha(ROOT/path)!=h:raise ValueError('Original baseline artifact modified')
    verify(pro['code_sha256']);verify(prediction_inputs)
    if sha(OUT/'DECISION_FREEZE.json')!=decisionsha:raise ValueError('Frozen policy modified')
    write(OUT/'RESULT.json',result);render(result,ROOT/'reports'/NAME)
    artifacts={str(p.relative_to(ROOT)):sha(p) for p in OUT.rglob('*') if p.is_file()}
    for rec in submissions.values():
        if 'path' in rec:artifacts[rec['path']]=rec['sha256']
    write(OUT/'COMPLETED.json',dict(status='completed',completed_at=now(),artifact_sha256=artifacts,
       submissions=submissions,official_upload=False,elapsed_seconds=time.perf_counter()-started))
    print(json.dumps(dict(status='completed',submissions=submissions),ensure_ascii=False),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['run','protocol','wait-and-run']);ap.add_argument('--device',default='cuda');ap.add_argument('--batch',type=int,default=32);ap.add_argument('--max-wait-hours',type=float,default=6.)
    args=ap.parse_args()
    if args.action=='protocol':print(json.dumps(protocol(),ensure_ascii=False,indent=2));return
    if args.action=='wait-and-run':
        deadline=time.monotonic()+args.max_wait_hours*3600
        while not (BASE/'COMPLETED.json').exists():
            if (BASE/'FAILED.json').exists():raise RuntimeError('Original baseline finalization failed; no restart')
            if time.monotonic()>deadline:raise TimeoutError('Baseline not complete within bounded wait')
            time.sleep(30)
    try:run(args.device,args.batch)
    except BaseException as e:
        if OUT.exists():write(OUT/'FAILED.json',dict(status='failed',error=repr(e),at=now(),automatic_restart=False))
        raise

if __name__=='__main__':main()
