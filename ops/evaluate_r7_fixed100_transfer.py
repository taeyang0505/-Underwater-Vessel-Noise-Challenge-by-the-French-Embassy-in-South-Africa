"""Predeclare then evaluate frozen fixed100 INNER-selected masks on old panels.

No fitting, checkpoint selection, new data, Test/old OUTER, GPU, or SSH.
The existing R6 cross panels are already seen development comparisons.
Model transforms accept db only; targets enter metrics only after inference.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np


ROOT=Path(__file__).resolve().parents[1]
PROTOCOL='configs/r7_transfer_audit_v1.json'
RUN='models/r7_fixed100_transfer_v1'
DATA='data/synthetic/r6_cpu_guardrails_v3_schemafix'
FIXED='models/r6_fixed100_v1'
RESULT='reports/R7_FIXED100_TRANSFER_RESULTS.json'
REPORT='reports/R7_FIXED100_TRANSFER_REPORT.md'
ROLES=('cross_generator','cross_sensor','cross_sensor_generator')
NAMES=('r4law_legacy8_global','r4law_relative8_global','r6law_legacy8_global',
       'r6law_relative8_global','r6law_relative8_local')


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def array_sha256(a):
    a=np.asarray(a)
    return hashlib.sha256(canonical({'shape':list(a.shape),'dtype':str(a.dtype)})
                          +np.ascontiguousarray(a).tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def assert_inputs(p):
    for name,digest in p['input_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Immutable input changed: '+name)
    payload={k:v for k,v in p.items() if k!='protocol_sha256'}
    if hashlib.sha256(canonical(payload)).hexdigest()!=p['protocol_sha256']:
        raise ValueError('Protocol payload changed')


def predeclare():
    if (ROOT/PROTOCOL).exists() or (ROOT/RUN).exists():
        raise FileExistsError('Fresh protocol/run required; no repeat or partial reuse')
    config=read(ROOT/'configs/r6_fixed100_v1.json')
    old_config=read(ROOT/'configs/r6_cpu_guardrails_v2.json')
    gate=read(ROOT/FIXED/'all_models_frozen.json')
    final=read(ROOT/FIXED/'results.json')
    if (gate['status']!='frozen' or final['status']!='completed'
            or gate['heldout_evaluated'] is not False
            or gate['config_sha256']!=sha(ROOT/'configs/r6_fixed100_v1.json')
            or [s['arm']['name'] for s in final['summaries']]!=list(NAMES)):
        raise ValueError('Fixed100 completion/config identity failed')
    files=['configs/r6_fixed100_v1.json','configs/r6_cpu_guardrails_v2.json',
           FIXED+'/freeze.json',FIXED+'/all_models_frozen.json',FIXED+'/results.json',
           'reports/R6V2_DATASET_AUDIT.json','reports/R6V2_CPU_ABLATION_RESULTS.json',
           'ops/evaluate_r7_fixed100_transfer.py']
    freeze=read(ROOT/FIXED/'freeze.json')
    original_freeze=read(ROOT/'models/r6_cpu_guardrails_v2/freeze.json')
    for name,digest in {**original_freeze['code_sha256'],**freeze['new_code_sha256'],
                        **config['original_frozen_core17_sha256']}.items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen code changed: '+name)
        files.append(name)
    # Extract only immutable filename/hash metadata, not embedded panel descriptors.
    audit_bytes=(ROOT/'reports/R6V2_DATASET_AUDIT.json').read_bytes()
    if hashlib.sha256(audit_bytes).hexdigest()!=config['data_manifest_sha256']:
        raise ValueError('Original data manifest changed')
    match=re.search(rb'"arrays"\s*:\s*(\{[^}]*\})',audit_bytes)
    if match is None:raise ValueError('Dataset array SHA map missing')
    array_map=json.loads(match.group(1))
    for filename in [r+'.npz' for r in ROLES]+['train_r4.npz','train_r6.npz']:
        name=DATA+'/'+filename
        if sha(ROOT/name)!=array_map[filename]:raise ValueError('Dataset bytes changed: '+name)
        files.append(name)
    policies=[]
    for summary in final['summaries']:
        name=summary['arm']['name'];folder=ROOT/FIXED/name
        if sha(folder/'summary.json')!=gate['model_summary_sha256'][name] or read(folder/'summary.json')!=summary:
            raise ValueError('Frozen model summary mismatch: '+name)
        selected=summary['best_inner_window'];epoch=selected['epoch']
        if (summary['epochs_run']!=100 or summary['status']!='completed'
                or selected['checkpoint']!='best_window.pt'
                or summary['artifact_sha256']['best_window.pt']!=summary['epoch_checkpoint_sha256'][f'epoch_{epoch:03d}.pt']):
            raise ValueError('INNER-only best-window policy binding failed')
        for filename in ['best_window.pt',f'checkpoints/epoch_{epoch:03d}.pt']:
            expected=(summary['artifact_sha256']['best_window.pt'] if filename=='best_window.pt'
                      else summary['epoch_checkpoint_sha256'][Path(filename).name])
            if sha(folder/filename)!=expected:raise ValueError('Selected checkpoint changed')
            files.append(FIXED+'/'+name+'/'+filename)
        files.append(FIXED+'/'+name+'/summary.json')
        prior=None
        if summary['prior_sha256'] is not None:
            prior=FIXED+'/'+name+'/legacy_train_prior.json'
            if sha(ROOT/prior)!=summary['prior_sha256']:raise ValueError('TRAIN prior changed')
            files.append(prior)
        if (summary['arm']['channels']=='legacy8')!=(prior is not None):
            raise ValueError('Prior/relative arm policy mismatch')
        policies.append({'name':name,'arm':summary['arm'],'model_config':summary['model_config'],
                         'parameters':summary['parameters'],'epoch':epoch,
                         'checkpoint':FIXED+'/'+name+'/best_window.pt',
                         'checkpoint_sha256':summary['artifact_sha256']['best_window.pt'],
                         'prior':prior,'selection':'already frozen strict own-INNER best-window; no cross selection'})
    controls={'constant0':0.,'constant0_1':.1}
    train_counts={}
    for law in ['r4','r6']:
        # TRAIN targets are the only numeric arrays read during predeclaration.
        with np.load(ROOT/DATA/f'train_{law}.npz',allow_pickle=False) as z:y=z['share'].astype('float64')
        if len(y)!=old_config['samples']['train'] or not np.isfinite(y).all() or ((y<0)|(y>1)).any():
            raise ValueError('Invalid original TRAIN target')
        controls[f'TRAIN_{law}_mean']=float(y.mean());train_counts[law]=len(y)
    p={'format':'r7_fixed100_transfer_protocol_v1','status':'predeclared_before_cross_array_or_prediction_read',
       'created_utc':datetime.now(timezone.utc).isoformat(),'models':policies,'roles':list(ROLES),
       'examples_per_role':{r:old_config['samples'][r] for r in ROLES},'constants':controls,
       'TRAIN_mean_counts':train_counts,'CPU_threads':2,'batch_size':128,'device':'CPU FP32',
       'fit':False,'checkpoint_or_weight_selection_using_cross':False,'Test_or_OUTER':False,
       'interpretation':'Already-seen R6 development transfer comparison; not blind, OOF or official-Test evidence',
       'score_policy':'unchanged model derived score; constants use presence score0.5',
       'metrics':'common fixed metrics plus slice AUC/share-AUC/site-macro; single-class/empty AUC null',
       'old_ES_comparison':'all five old frozen own-INNER macro-selected checkpoints; selection/budget/threads confounded',
       'all_bad_results_retained':True,'no_retraining_or_tuning_from_results':True,
       'input_sha256':{n:sha(ROOT/n) for n in sorted(set(files))}}
    p['protocol_sha256']=hashlib.sha256(canonical(p)).hexdigest()
    assert_inputs(p);write(ROOT/PROTOCOL,p)
    print(json.dumps({'status':'predeclared','protocol':PROTOCOL,'sha256':sha(ROOT/PROTOCOL),
                      'numeric_cross_arrays_read':False,'models':5,'roles':3}))


def extended_metrics(data,pred):
    from pipeline.cpu_ablation_r6_v2 import metrics
    from sklearn.metrics import roc_auc_score
    out=metrics(data,pred)
    y=data['share'];site=data['site'].astype(str);family=data['family'].astype(str)
    selectors={'site_slices':{s:site==s for s in sorted(set(site))},
               'family_slices':{s:family==s for s in sorted(set(family))},
               'target_slices':{'zero':y==0,'positive_low_to_.1':(y>0)&(y<=.1),
                                '.1_to_.4':(y>.1)&(y<=.4),'.4_to_1':y>.4}}
    for group,by in selectors.items():
        for name,ix in by.items():
            z=data['presence'][ix];p=pred['share'][ix];score=pred['score'][ix]
            m=out[group][name];m['presence_auc']=float(roc_auc_score(z,score)) if len(set(z))==2 else None
            m['presence_auc_share_only']=float(roc_auc_score(z,p)) if len(set(z))==2 else None
            errors=p.astype(float)-y[ix].astype(float);sites=site[ix]
            m['site_macro_rmse']=(float(np.mean([np.sqrt(np.mean(errors[sites==s]**2)) for s in sorted(set(sites))]))
                                  if ix.any() else None)
    return out


def data_identity(data):
    return {'query_identity':{'examples':len(data['db']),'db_sha256':array_sha256(data['db']),
                             'base_window_index_order_sha256':array_sha256(data['base_window_index']),
                             'site_order_sha256':array_sha256(data['site'])},
            'target_identity':{n+'_sha256':array_sha256(data[n]) for n in ['share','presence']}}


def run():
    p=read(ROOT/PROTOCOL);assert_inputs(p)
    if p['roles']!=list(ROLES) or [m['name'] for m in p['models']]!=list(NAMES):
        raise ValueError('Fixed arm/panel policy changed')
    output=ROOT/RUN
    if output.exists() or (ROOT/RESULT).exists() or (ROOT/REPORT).exists():
        raise FileExistsError('Fresh transfer output required; no repeated evaluation or partial reuse')
    output.mkdir(parents=True);write(output/'protocol_copy.json',p);start=time.monotonic()
    try:
        import torch
        from pipeline.relative_models_r6_v2 import TinyMaskCNNv2
        from pipeline.cpu_ablation_r6_v2 import grid,predict_cnn
        torch.set_num_threads(p['CPU_threads'])
        panels={};bindings={}
        wanted=['db','share','presence','site','family','base_window_index']
        for role in ROLES:
            with np.load(ROOT/DATA/(role+'.npz'),allow_pickle=False) as z:panels[role]={k:z[k] for k in wanted}
            d=panels[role];n=p['examples_per_role'][role]
            if (d['db'].shape!=(n,15,25) or d['share'].shape!=(n,) or d['presence'].shape!=(n,)
                    or not np.isfinite(d['share']).all() or not np.isin(d['presence'],[0,1]).all()
                    or not np.array_equal(d['presence'],(d['share']>0).astype(d['presence'].dtype))):
                raise ValueError('Original panel shape/target identity invalid: '+role)
            bindings[role]={'input_npz_sha256':p['input_sha256'][DATA+'/'+role+'.npz'],**data_identity(d)}
        write(output/'row_bindings.json',bindings)
        evaluation={role:{} for role in ROLES}
        old=read(ROOT/'reports/R6V2_CPU_ABLATION_RESULTS.json')
        for policy in p['models']:
            name=policy['name'];folder=output/name;folder.mkdir()
            checkpoint=torch.load(ROOT/policy['checkpoint'],map_location='cpu',weights_only=True)
            if (checkpoint['model_config']!=policy['model_config'] or checkpoint['arm']!=policy['arm']
                    or checkpoint['epoch']!=policy['epoch'] or checkpoint['seed']!=20261026):
                raise ValueError('Exact checkpoint metadata mismatch: '+name)
            model=TinyMaskCNNv2(**policy['model_config'])
            model.load_state_dict(checkpoint['state_dict'],strict=True)
            if sum(v.numel() for v in model.parameters())!=policy['parameters']:
                raise ValueError('Parameter count changed: '+name)
            model.requires_grad_(False);model.eval()
            prior=None if policy['prior'] is None else read(ROOT/policy['prior'])
            for role in ROLES:
                begun=time.monotonic();d=panels[role]
                x,w=grid(d['db'],policy['arm']['channels'],prior)
                pred=predict_cnn(model,x,w,p['batch_size'])
                if (any(not np.isfinite(v).all() for v in pred.values())
                        or pred['cell_mask'].shape!=d['db'].shape
                        or (pred['cell_mask'][w==0]!=0).any()
                        or max(float(np.max(abs(pred[k]-np.clip(pred[k],0,1)))) for k in ['share','score','cell_mask'])>1e-6):
                    raise ValueError('Nonfinite or invalid model predictions: '+name+'/'+role)
                pooling=float(np.max(abs(pred['share']-(w*pred['cell_mask']).sum((1,2)))))
                if pooling>3e-7:raise ValueError('Physical pooling failed')
                path=folder/(role+'_predictions.npz')
                np.savez_compressed(path,**pred,base_window_index=d['base_window_index'])
                m=extended_metrics(d,pred)
                record={'metrics':m,'predictions_path':path.relative_to(ROOT).as_posix(),
                        'predictions_sha256':sha(path),'checkpoint_sha256':policy['checkpoint_sha256'],
                        'epoch':policy['epoch'],**bindings[role],
                        'maximum_predicted_physical_pooling_error':pooling,
                        'elapsed_seconds':time.monotonic()-begun,
                        'old_ES_checkpoint_metrics':old['heldout_results'][role][name],
                        'RMSE_difference_from_old_ES':m['rmse']-old['heldout_results'][role][name]['rmse']}
                evaluation[role][name]=record;write(folder/(role+'_metrics.json'),record)
                print(json.dumps({'evaluated':name,'role':role,'rmse':m['rmse'],'macro':m['site_macro_rmse'],
                                  'auc':m['presence_auc'],'seconds':record['elapsed_seconds']}),flush=True)
            del model,checkpoint
        for role in ROLES:
            d=panels[role]
            for name,value in p['constants'].items():
                pred={'share':np.full(len(d['share']),value),'score':np.full(len(d['share']),.5)}
                evaluation[role][name]={'constant':value,'metrics':extended_metrics(d,pred),**bindings[role]}
        assert_inputs(p)
        result={'format':'r7_fixed100_transfer_result_v1','status':'completed','protocol':PROTOCOL,
                'protocol_file_sha256':sha(ROOT/PROTOCOL),'protocol_sha256':p['protocol_sha256'],
                'models':p['models'],'evaluations':evaluation,'row_bindings':bindings,
                'elapsed_seconds':time.monotonic()-start,
                'environment':{'torch':torch.__version__,'numpy':np.__version__,'CPU_threads':torch.get_num_threads()},
                'interpretation':p['interpretation'],'checkpoint_selection_changed':False,
                'all_inputs_unchanged_after_evaluation':True,'fit_Test_OUTER_GPU_SSH_submission':False,
                'all_bad_results_retained':True,'old_comparison_confound':'checkpoint metric,epoch budget andthreads differ; no isolated ES-off causal attribution'}
        write(output/'results.json',result);write(ROOT/RESULT,result)
        render(p,result)
        print(json.dumps({'status':'completed','result':RESULT,'sha256':sha(ROOT/RESULT),
                          'elapsed_seconds':result['elapsed_seconds']}),flush=True)
    except BaseException as error:
        write(output/'FAILED.json',{'type':type(error).__name__,'error':str(error),'partial_reusable':False})
        raise


def render(p,result):
    def f(v):return '—' if v is None else f'{v:.6f}'
    blocks=[];changes=[]
    for role in ROLES:
        lines=[f'## {role}','','| Arm/control | RMSE | Macro RMSE | MAE | Bias | Score AUC | Share AUC |',
               '|---|---:|---:|---:|---:|---:|---:|']
        for name,record in result['evaluations'][role].items():
            m=record['metrics'];lines.append('| '+' | '.join([name,f(m['rmse']),f(m['site_macro_rmse']),f(m['mae']),
                                                             f(m['bias']),f(m['presence_auc']),f(m['presence_auc_share_only'])])+' |')
            if 'RMSE_difference_from_old_ES' in record:
                changes.append(f"| {role} | {name} | {record['epoch']} | {record['RMSE_difference_from_old_ES']:+.6f} |")
        blocks.append('\n'.join(lines))
    text=f'''# R7 fixed100 체크포인트 전이 평가

5개 CNN의 **이미 own-INNER에서 동결한 best-window** checkpoint를 기존 동일 cross 패널3개에서 CPU FP32/threads2로 평가했다. Protocol은 패널 array/새 예측을 읽기 전에 고정했고 checkpoint·계수·loss·seed·데이터를 결과에 따라 바꾸지 않았다. 이 패널은 이전 R6에서 이미 관찰한 **개발용 전이 비교**이며 새 blind holdout, OOF, Test 성능이 아니다.

선택 epoch: {', '.join(m['name']+'='+str(m['epoch']) for m in p['models'])}. 정답·site·family는 metric에만 사용하고 신경망 입력은 db-only transform 및 해당 db의 물리 w다. legacy8의 prior는 기존 자기 TRAIN에서 fit한 동결 값만 사용했다. 모든 model state는 exact schema/strict=True로 복구했고 resize/partial load는 하지 않았다.

{chr(10).join(blocks)}

## 이전 ES checkpoint와의 기술적 차이

아래 Δ는 fixed100 best-window−이전 ES macro-selected다. 중단 여부 외에 checkpoint 기준·학습 예산·threads도 달라서 ES 제거 단독의 인과 효과가 아니다. 모든 악화도 보고하며 이 결과를 근거로 새 모델/epoch/ensemble을 선택하지 않았다.

| Panel | Arm | fixed100 epoch | Δ RMSE |
|---|---|---:|---:|
{chr(10).join(changes)}

site/target-bin/family별 n·RMSE·site-macro·MAE·bias·score/share AUC는 **모든5모델과 상수4개**에 대해 [전체 JSON](R7_FIXED100_TRANSFER_RESULTS.json)의 evaluations.metrics에 보존했다. 단일 class/빈 slice AUC는 null이다. 상수 .0/.1과 두 law의 TRAINmean은 protocol에 미리 고정했고 상수 presence score는 .5다.

전체 경과 {result['elapsed_seconds']:.2f}초. 원본 NPZ·모델·core source SHA는 전후 불변이며, 원천 data·모델 weight를 수정하거나 fit하지 않았다. 15개 prediction NPZ와 원래 row 순서의 SHA/target SHA를 새 run 폴더에 저장했다. cross-sensor와 joint는 같은 배경 draw를 공유하고, cross-generator는 원천 참조와 generator가 함께 바뀌므로 독립 패널 세트나 단일 원인 대조로 과장하지 않는다. 실제 대회 일반화·공식 LB 점수를 보장하지 않으며, 이 결과에 맞춘 재학습·calibration·패널 반복 재선택은 하지 않았다.

[사전 protocol](../{PROTOCOL}) · [row bindings](../{RUN}/row_bindings.json) · [모든 예측/metric](../{RUN}/results.json)
'''
    write_path=ROOT/REPORT
    with write_path.open('x') as stream:stream.write(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['predeclare','evaluate'])
    args=parser.parse_args()
    if args.action=='predeclare':predeclare()
    else:run()


if __name__=='__main__':main()
