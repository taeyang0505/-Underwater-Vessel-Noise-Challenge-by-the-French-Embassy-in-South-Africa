"""R10 temporal-order and CPU/GPU precision audit on fixed Fold0 selection.

No training, Test, report arrays, GPU, SSH or external transmission. All
models/policy/data remain immutable. Three fixed seeds are declared before
prediction. An unchanged shuffle score does not prove device memorization.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
SEEDS=(2026102601,2026102602,2026102603)
MEMBERS=('bg_cnn','tfgrid_fdy_conformer','fdy','itransformer','lgbm_physics')
WEIGHTS=(.1,.1,.1,.1,.6)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def atomic(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def utc():return datetime.now(timezone.utc).isoformat()
def array_sha(a):
    import numpy as np
    a=np.ascontiguousarray(a);h=hashlib.sha256(json.dumps([str(a.dtype),list(a.shape)]).encode());h.update(a.tobytes());return h.hexdigest()


def permute(raw,seed):
    """Shuffle observed minute rows within identical missing-mask patterns.

    With R9 complete/empty minutes this is exactly one permutation of all
    observed indices. Partial rows, if present, only swap identical masks.
    Completely missing rows stay in place. No value or label is imputed.
    """
    import numpy as np
    from pipeline.relative_grid_r6_v2 import relative_grid_v2
    required=['db','unrounded_db','cell_share']
    if any(raw[k].shape!=raw['db'].shape for k in required):raise ValueError('Aligneddb/raw/q required')
    n=len(raw['db']);permutations=np.tile(np.arange(15,dtype=np.int16),(n,1))
    partial=0
    for i in range(n):
        observed=np.isfinite(raw['db'][i]);patterns={}
        rng=np.random.default_rng(np.random.SeedSequence([seed,i]))
        for t,row in enumerate(observed):
            if not row.any():continue
            partial+=int(not row.all());patterns.setdefault(row.tobytes(),[]).append(t)
        for indices in patterns.values():
            indices=np.asarray(indices,dtype=np.int16);permutations[i,indices]=rng.permutation(indices)
    out={k:np.take_along_axis(raw[k],permutations[:,:,None],axis=1) for k in required}
    mask=np.isfinite(raw['db']);newmask=np.isfinite(out['db'])
    if not np.array_equal(mask,newmask):raise AssertionError('NaN positions changed')
    for k in ['db','unrounded_db']:
        if not np.array_equal(np.sort(raw[k],axis=1),np.sort(out[k],axis=1),equal_nan=True):raise AssertionError('Band marginal changed')
    if not np.array_equal(np.sort(raw['cell_share'],axis=1),np.sort(out['cell_share'],axis=1)):raise AssertionError('q marginal changed')
    peak=np.nanmax(raw['unrounded_db'],axis=(1,2),keepdims=True)
    e=np.where(mask,10.**((raw['unrounded_db']-peak)/10.),0.)
    pe=np.where(newmask,10.**((out['unrounded_db']-peak)/10.),0.)
    olde=e.sum((1,2));newe=pe.sum((1,2))
    olda=(e*raw['cell_share']).sum((1,2));newa=(pe*out['cell_share']).sum((1,2))
    oldy=olda/olde;newy=newa/newe
    energy_relative=float(np.max(abs(newe-olde)/olde))
    addition_relative=float(np.max(abs(newa-olda)/np.maximum(olda,1e-300)))
    delta=float(np.max(abs(newy-oldy)))
    original_saved_gap=float(np.max(abs(oldy-raw['share'])))
    if energy_relative>5e-14 or addition_relative>5e-14 or delta>5e-14 or original_saved_gap>1e-6:raise AssertionError('Known component energy/share changed')
    _,w=relative_grid_v2(raw['db']);_,pw=relative_grid_v2(out['db'])
    expected_w=np.take_along_axis(w,permutations[:,:,None],axis=1)
    weight_gap=float(np.max(abs(expected_w-pw)))
    if weight_gap>3e-7:raise AssertionError('Rounded weights do not follow row permutation')
    audit={'seed':seed,'examples':n,'partial_mask_minutes':partial,
        'NaN_positions_exact':True,'band_marginals_exact':True,'q_marginals_exact':True,
        'energy_sum_max_relative_error':energy_relative,'added_energy_sum_max_relative_error':addition_relative,
        'raw_recomputed_share_max_change':delta,'saved_float32_target_original_gap':original_saved_gap,
        'rounded_weight_permutation_max_abs':weight_gap,
        'sample_ids_unchanged':True,'targets_rewritten':False,
        'permutation_sha256':array_sha(permutations),
        'shuffled_db_sha256':array_sha(out['db']),'shuffled_q_sha256':array_sha(out['cell_share']),
        'changed_windows':int((permutations!=np.arange(15)).any(1).sum())}
    return out,permutations,audit


def measures(y,z,p,s):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    y=np.asarray(y,float);p=np.asarray(p,float);z=np.asarray(z);s=np.asarray(s,float)
    e=p-y
    return {'n':len(y),'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(abs(e).mean()),
            'bias':float(e.mean()),'addition_presence_auc':float(roc_auc_score(z,s)) if len(np.unique(z))==2 else None}


def differences(a,b):
    import numpy as np
    d=np.asarray(a,float)-np.asarray(b,float)
    return {'mean':float(d.mean()),'MAE':float(abs(d).mean()),'RMS':float(np.sqrt(np.mean(d*d))),
            'max_absolute':float(abs(d).max())}


def numeric_fixture():
    import numpy as np
    rng=np.random.default_rng(4);b=rng.uniform(.5,2,(4,15,25));a=rng.uniform(0,.4,(4,15,25));mask=np.ones_like(b,bool)
    mask[0,:5]=False;mask[1,1,4:]=False;mask[1,3,4:]=False;mask[2,8:]=False
    e=b+a;db=np.where(mask,80+10*np.log10(e),np.nan);q=np.where(mask,a/e,0.)
    y=(np.where(mask,a,0).sum((1,2))/np.where(mask,e,0).sum((1,2))).astype(np.float32)
    raw={'db':np.round(db,1).astype(np.float32),'unrounded_db':db,'cell_share':q.astype(np.float32),'share':y}
    before={k:array_sha(v) for k,v in raw.items()};audits=[]
    for seed in SEEDS:
        out,p,audit=permute(raw,seed);out2,p2,_=permute(raw,seed)
        assert np.array_equal(p,p2) and np.array_equal(out['db'],out2['db'],equal_nan=True)
        audits.append(audit)
    assert before=={k:array_sha(v) for k,v in raw.items()}
    assert len(set(a['permutation_sha256'] for a in audits))==3
    return {'status':'passed','synthetic_only':True,'source_arrays_unchanged':True,
            'same_seed_replay_exact':True,'three_distinct_seeds':True,'partial_mask_pattern_guard_tested':True,'audits':audits}


def run(args):
    # Fresh-process DYLD/OMP variables are supplied by the shell. This call
    # enforces them before importing either Torch or LightGBM.
    from ops.infer_r9_fold0_all5_pilot import environment,inputs,predict_neural_db,local,verify_pins,combine
    environment(4)
    import numpy as np
    from pipeline.dataset_r9 import FixedR9Dataset
    pilot=local(args.pilot);sourceplan=read(pilot/'prequery_plan.json');policy=read(pilot/'frozen_policy.json')
    ready=read(pilot/'PREQUERY_READY.json')
    if (policy['member_order']!=list(MEMBERS) or tuple(policy['weights'])!=WEIGHTS
            or ready['frozen_policy_file_sha256']!=sha(pilot/'frozen_policy.json')
            or ready['prequery_plan_sha256']!=sha(pilot/'prequery_plan.json')
            or canonical({k:v for k,v in policy.items() if k!='policy_sha256'})!=policy['policy_sha256']):
        raise ValueError('Exact already-frozen .1NN/.6tree policy required; no new selection')
    out=local(args.output)
    if out.exists():raise FileExistsError('Fresh R10 audit directory required; no adaptive retry')
    out.mkdir(parents=True);start=time.perf_counter()
    pins,_,binding,tree=inputs(sourceplan['nn_dirs'],local(sourceplan['tree_job']),local(sourceplan['bundle']),local(sourceplan['config']))
    if pins!=sourceplan['source_pins']:raise ValueError('Original model/pilot source pins differ')
    dataset=FixedR9Dataset(local(sourceplan['bundle']),0,'selection');raw=dataset.raw_arrays()
    if len(dataset)!=1000 or dataset.identity!=binding['selection_identity']:raise ValueError('Exact Fold0selection1000 required')
    hashes={k:array_sha(v) for k,v in raw.items()};metadata={}
    for kind in MEMBERS[:-1]:
        path=local(sourceplan['nn_dirs'][kind])/'best_window_validation_predictions.npz'
        metadata[kind]={'saved_GPU_prediction_sha256':sha(path),'summary_sha256':sha(path.with_name('summary.json'))}
    sourcepins=dict(pins,**{str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))})
    sourcepins[str((pilot/'frozen_policy.json').relative_to(ROOT))]=sha(pilot/'frozen_policy.json')
    plan={'format':'r10_temporal_permutation_plan_v1','created_utc':utc(),'selection_identity':dataset.identity,
          'seeds':list(SEEDS),'weights':list(WEIGHTS),'threads':4,'batch_size':args.batch_size,
          'source_pins':sourcepins,'original_array_sha256':hashes,'GPU_reference_predictions':metadata,
          'permutation':'whole observed25-band minute rows, samepermutation db/raw/q; NaN positions unchanged',
          'partial_missing_policy':'only permute rows within identical mask patterns','time_reversal_included':False,
          'Test_report_TRAIN_fitting_GPU_SSH_external_transmission':False,
          'interpretation':'No drop supports temporal-order insensitivity, not proof of static recorder memorization.'}
    atomic(out/'predeclared_plan.json',plan)
    milestones=[]
    def milestone(stage,**extra):
        row={'stage':stage,'elapsed_seconds':time.perf_counter()-start,'utc':utc(),**extra};milestones.append(row)
        atomic(out/'progress.json',row);print(json.dumps(row),flush=True)
    def predict(db,label):
        predictions={};timings={}
        for kind in MEMBERS[:-1]:
            predictions[kind],timings[kind]=predict_neural_db(local(sourceplan['nn_dirs'][kind]),db,
                binding['selection_identity'],batch_size=args.batch_size,threads=4)
            milestone(label+'/'+kind,model_prediction_seconds=timings[kind]['seconds'])
        t=time.perf_counter();predictions['lgbm_physics']=tree.predict(db,batch_size=args.batch_size)
        timings['lgbm_physics']={'seconds':time.perf_counter()-t}
        predictions['frozen_all5']=combine(predictions,WEIGHTS)
        return predictions,timings
    def save_predictions(path,preds):
        np.savez_compressed(path,sample_id=raw['sample_id'],member_order=np.asarray(MEMBERS+('frozen_all5',)),
            share=np.stack([preds[k]['share'] for k in MEMBERS+('frozen_all5',)],axis=1),
            score=np.stack([preds[k]['score'] for k in MEMBERS+('frozen_all5',)],axis=1))
    try:
        original,timings=predict(raw['db'],'baseline_CPU_FP32');save_predictions(out/'original_CPU_predictions.npz',original)
        base_metrics={k:measures(raw['share'],raw['presence'],p['share'],p['score']) for k,p in original.items()}
        precision={};reference={}
        for kind in MEMBERS[:-1]:
            p=local(sourceplan['nn_dirs'][kind])/'best_window_validation_predictions.npz'
            with np.load(p,allow_pickle=False) as z:
                ids=z['sample_id'].astype(str);ids0=raw['sample_id'].astype(str)
                if len(set(ids))!=1000 or set(ids)!=set(ids0):raise ValueError('GPU saved IDs differ:'+kind)
                lookup={value:i for i,value in enumerate(ids)};order=np.array([lookup[v] for v in ids0])
                if not np.array_equal(z['target_share'][order],raw['share']) or not np.array_equal(z['target_presence'][order],raw['presence']):raise ValueError('GPU saved target identity differs:'+kind)
                reference[kind]={'share':z['share'][order].copy(),'score':z['score'][order].copy()}
            precision[kind]={'CPU_FP32':base_metrics[kind],
                'saved_GPU_BF16':measures(raw['share'],raw['presence'],reference[kind]['share'],reference[kind]['score']),
                'CPU_minus_GPU_share':differences(original[kind]['share'],reference[kind]['share']),
                'CPU_minus_GPU_score':differences(original[kind]['score'],reference[kind]['score']),
                'ID_keyed_alignment':True,'exact_float_equality_required':False,
                'interpretation':'Different device/precision execution; a nonzero delta alone is not CSV or feature mapping failure.'}
        reference['lgbm_physics']=original['lgbm_physics'];reference['frozen_all5']=combine(reference,WEIGHTS)
        precision['frozen_all5']={'CPU_FP32':base_metrics['frozen_all5'],
            'GPU_NN_plus_same_CPU_tree_reference':measures(raw['share'],raw['presence'],reference['frozen_all5']['share'],reference['frozen_all5']['score']),
            'CPU_minus_reference_share':differences(original['frozen_all5']['share'],reference['frozen_all5']['share']),
            'reference_is_not_a_separately_trained_GPU_ensemble':True}
        atomic(out/'precision_parity.json',precision);milestone('baseline_and_precision_saved')
        permutations=[]
        for seed in SEEDS:
            transformed,indices,proof=permute(raw,seed)
            np.savez_compressed(out/f'permutation_{seed}.npz',sample_id=raw['sample_id'],permutation=indices)
            preds,pt=predict(transformed['db'],f'shuffle_{seed}');save_predictions(out/f'shuffled_predictions_{seed}.npz',preds)
            result={}
            for kind,p in preds.items():
                current=measures(raw['share'],raw['presence'],p['share'],p['score']);base=base_metrics[kind]
                result[kind]={'metrics':current,'delta_rmse':current['rmse']-base['rmse'],
                    'delta_mae':current['mae']-base['mae'],
                    'delta_AUC':None if current['addition_presence_auc'] is None else current['addition_presence_auc']-base['addition_presence_auc'],
                    'paired_prediction_share_difference':differences(p['share'],original[kind]['share']),
                    'paired_prediction_score_difference':differences(p['score'],original[kind]['score']),
                    'paired_mean_squared_error_change':float(np.mean((p['share'].astype(float)-raw['share'])**2-(original[kind]['share'].astype(float)-raw['share'])**2))}
            record={'seed':seed,'invariant_checks':proof,'models':result,'prediction_timings':pt}
            permutations.append(record);atomic(out/f'results_seed_{seed}.json',record);milestone(f'shuffle_{seed}_saved')
        verify_pins(sourcepins)
        if hashes!={k:array_sha(v) for k,v in raw.items()}:raise AssertionError('Original dataset changed in memory')
        summary={k:{'original_rmse':base_metrics[k]['rmse'],
                    'shuffle_rmse_mean':float(np.mean([v['models'][k]['metrics']['rmse'] for v in permutations])),
                    'shuffle_delta_rmse_min':float(min(v['models'][k]['delta_rmse'] for v in permutations)),
                    'shuffle_delta_rmse_max':float(max(v['models'][k]['delta_rmse'] for v in permutations)),
                    'original_AUC':base_metrics[k]['addition_presence_auc'],
                    'shuffle_AUC_mean':float(np.mean([v['models'][k]['metrics']['addition_presence_auc'] for v in permutations]))}
                 for k in original}
        report={'format':'r10_temporal_permutation_results_v1','status':'completed','created_utc':utc(),
            'predeclared_plan_sha256':sha(out/'predeclared_plan.json'),'baseline_CPU_metrics':base_metrics,
            'precision_parity':precision,'permutations':permutations,'summary':summary,
            'original_prediction_timings':timings,'milestones':milestones,'elapsed_seconds':time.perf_counter()-start,
            'original_arrays_policy_models_code_unchanged':True,'weights_refit':False,
            'Test_unused_report_GPU_SSH_external_transmission_or_training':False,
            'scope':'Same already-used selection1000, sharedreference background-source development data; not unseen generalization.',
            'interpretation':'No shuffle degradation supports temporal-order insensitivity in this test, not proof of static-device memorization. Shuffle creates artificial abrupt transitions; degradation alone does not prove vessel-specific temporal reasoning.',
            'output_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='progress.json'}}
        atomic(out/'RESULTS.json',report)
        lines=['# R10 시간 순서·정밀도 진단','',f"실행 완료: {report['elapsed_seconds']:.2f}초. 동일한 Fold0 selection1000, 세 사전 고정 seed, NN각.1/트리.6 가중치를 유지했습니다.",'',
               '|모델|원본 CPU RMSE|shuffle 평균 RMSE|RMSE 변화 범위|원본 AUC|shuffle 평균 AUC|',
               '|---|---:|---:|---:|---:|---:|']
        for k,v in summary.items():lines.append(f"|{k}|{v['original_rmse']:.6f}|{v['shuffle_rmse_mean']:.6f}|{v['shuffle_delta_rmse_min']:+.6f} .. {v['shuffle_delta_rmse_max']:+.6f}|{v['original_AUC']:.6f}|{v['shuffle_AUC_mean']:.6f}|")
        lines+=['','원본 CPU FP32와 저장 GPU BF16은 별도로 비교했습니다. ID·target 순서는 일치하며 수치 차이를 CSV 오류로 단정하지 않습니다.',
                '세 permutation 모두 원 관측 NaN 위치·band marginal·에너지 합·raw 정답 비율을 유지합니다. db/rawdb/q에 같은 행 순서를 적용하고 모델 feature와w를 다시 계산했습니다. 원본은 변경하지 않았습니다.',
                '점수가 유지돼도 이 설정의 시간 순서 변화에 민감하지 않다는 의미에 한정합니다. 장치 암기를 입증하지 않습니다. 반대로 악화는 비현실적인 abrupt transition에 대한 민감도일 수 있어 선박의 시간 문맥을 이해했다는 증명도 아닙니다.',
                'Selection은 이미 checkpoint 선택에 사용됐고509 reference를 공유합니다. Test·미사용report·GPU·SSH·외부 전송·학습·가중치 재조정은 없었습니다.','', '[전체 수치·정밀도·각seed·SHA](RESULTS.json)']
        (out/'RESULTS.md').write_text('\n'.join(lines)+'\n');milestone('completed')
        print(json.dumps({'status':'completed','output':str(out),'elapsed_seconds':report['elapsed_seconds'],'summary':summary}),flush=True)
        return report
    except BaseException as error:
        atomic(out/'FAILED.json',{'status':'failed','error_type':type(error).__name__,'error':str(error),'elapsed_seconds':time.perf_counter()-start,'automatic_retry':False});raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pilot',default='submissions/r9_fold0_all5_cpu_r1')
    p.add_argument('--output',default='reports/r10_diagnostics/temporal_permutation_r1');p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:print(json.dumps(numeric_fixture()));return
    if a.batch_size<1:raise ValueError('Positive batch required')
    run(a)


if __name__=='__main__':main()
