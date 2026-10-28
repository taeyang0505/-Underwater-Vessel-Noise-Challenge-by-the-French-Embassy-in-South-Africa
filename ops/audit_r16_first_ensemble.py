"""One-shot CPU audit AFTER the first R16 ensemble CSV is complete.

Original source, models, policies, panels and CSVs remain immutable. Never
reads Test observations or saved query prediction arrays. No model training.
"""
from pathlib import Path
import argparse,datetime,fcntl,gc,html,json,os,subprocess,sys,time,traceback
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OPS=ROOT/'ops/r16_first_ensemble_audit_r1'
REPORT=ROOT/'reports/r16_first_ensemble_audit_r1'
SEED=2026102815
MEMBERS=('tree5','bg_cnn','fdy','itransformer','tfgrid_fdy_conformer')
REGIMES=('D0','H0')
CPU='/Users/hongtaeyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    import hashlib
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
    t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def now():return datetime.datetime.now().astimezone().isoformat()
def code_pins():
    from pipeline.training_r16_multisupport import codes
    pins=codes()
    for n in ('ops/audit_r16_first_ensemble.py','ops/r16_postaudit_metrics.py','ops/verify_r16_postcsv_audit.py',
              'ops/finalize_r16_multisupport_full.py','ops/r16_support_temporal_data.py',
              'pipeline/trees_ensemble_r9.py'):
        pins[n]=sha(ROOT/n)
    return pins
def unchanged(pins):
    for n,h in pins.items():
        if sha(ROOT/n)!=h:raise ValueError('Pinned input/source changed: '+n)
def plan():
    from pipeline.r16_multisupport_data import OUT,PARENT
    OPS.mkdir(parents=True,exist_ok=True);p=OPS/'PROTOCOL.json'
    if p.exists():raise FileExistsError('Audit protocol already frozen; do not replace')
    protocol={'version':'r16_first_ensemble_postcsv_audit_v1','created_kst':now(),
      'trigger':'parent completed + final receipt and candidate CSV verified; no interruption or duplicate training',
      'bootstrap_reps':1000,'seed':SEED,'primary_resampling_unit':'background_group',
      'reference_cluster_bootstrap':'additional sensitivity if >=3 nonempty reference components; not treated as independent proof',
      'probability_logit_clip':1e-12,'ECE_bins':10,'CSV_precision':'float64 .17g',
      'panels':['local_fit','local_gate','local_report','domain_holdout','native12_VS_hf_high_negative','native12_VS_st_francis_dense_negative'],
      'score_methods':['prob_mean','logit_mean','logit_mean_prior80_to50','rank_mean_panel_diagnostic_only','rank_mean_fixed_local_fit_CDF'],
      'top_quartile_slice':'SAME prob-mean top25percent mask for all methods; disclose one-class undefined AUC',
      'balanced_calibration':'each class total0.5; NOT an estimate of Test prevalence or calibrated Test posterior',
      'regression_comparators':['R9 Fold0 Tree','R9 Tree5 mean','R16 D0/H0 Tree10 mean'],
      'orthogonal_projection_fit':'local_fit only, descriptive new~old OLS; fixed coefficients on all evaluation panels',
      'interventions':['other-site same-mask negative donor','same-zero static linear tilt0..12dB'],
      'NN_intervention_paths':['input Ch4 only','input Ch7 only','input Ch4+Ch7 with ORIGINAL pooling w','pool-only with original q','joint swapped input and w'],
      'feature_only_intervention_warning':'OOD feature sensitivity probe bypasses SAME-sqrtw forward validation ONLY in isolated audit; no production code changed',
      'maximum_intervention_unique_zero_windows_per_panel':400,
      'intervention_panel':'local_report only, unique background negatives; no Test',
      'one_SE_rule':'group OOB center-minus-optimized RMSE gain <= bootstrap SD of paired gain; NOT SD/sqrt(B)',
      'one_SE_action':'record recommendation and alternative policy only; NEVER replace frozen first CSV or original policy',
      'no_Test_observations_or_query_prediction_access':True,'no_training':True,'official_upload':False,
      'code_sha256':code_pins(),'source_artifact_sha256':{
          str((OUT/'manifest.json').relative_to(ROOT)):sha(OUT/'manifest.json'),str(PARENT.relative_to(ROOT)):sha(PARENT)}}
    write(p,protocol);return protocol

def guard(event,args):
    if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
    p=Path(os.fsdecode(args[0])).resolve()
    if p.name.lower() in ('test.csv','sample_submission.csv','query_predictions.npz'):
        raise PermissionError('Post-CSV audit forbids query observations and query arrays')
    mode=args[1] if len(args)>1 else 'r';flags=args[2] if len(args)>2 else 0
    writing=isinstance(mode,str) and any(k in mode for k in ('w','a','x','+'))
    writing|=isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
    if writing and not any(p.is_relative_to(a) for a in (OPS,REPORT)):
        raise PermissionError('Audit may write only its own versioned output: '+str(p))

def verify_parent():
    from ops.run_r16_multisupport_fivefold import verify_csv
    from pipeline.training_r16_multisupport import MODELS
    if read(ROOT/'ops/r16_multisupport_full_r1/state.json')['phase']!='completed':raise ValueError('First complete CSV required')
    final=MODELS/'final';done=read(final/'COMPLETED.json')
    if done['status']!='completed' or done['official_upload'] is not False:raise ValueError('Wrong completion receipt')
    path=ROOT/'submissions/r16_multisupport_full_r1/submission_r16.csv'
    verify_csv(path,done['submission']['sha256'])
    from ops.finalize_r16_multisupport_full import prerequisites
    pins=prerequisites();pins[str(path.relative_to(ROOT))]=sha(path)
    for name in ('policy.json','EVALUATION.json','COMPLETED.json','TEST_PREDICTION_AUDIT.json'):
        pins[str((final/name).relative_to(ROOT))]=sha(final/name)
    return final,pins

def join(parts):
    import numpy as np
    keys=set.intersection(*(set(d) for d in parts))
    return {k:np.concatenate([d[k] for d in parts]) for k in keys}

def panel_data(name,freeze):
    import numpy as np
    from pipeline.r16_multisupport_data import PARENT,role
    from ops.r16_support_temporal_data import load_panel,load_long_panel
    from ops.r16_forward_data import OPERATORS
    if name=='local_fit':return role(0,'D0','selection'),None
    if name in ('local_gate','local_report','domain_holdout'):
        return join([load_panel(PARENT,name,op,freeze_receipt=freeze) for op in OPERATORS]),None
    pos=join([load_long_panel(PARENT,'native12_positive',operator=op,freeze_receipt=freeze) for op in OPERATORS])
    neg=load_long_panel(PARENT,name.removeprefix('native12_VS_'),freeze_receipt=freeze)
    return join([pos,neg]),np.r_[np.full(len(pos['share']),.5/len(pos['share'])),np.full(len(neg['share']),.5/len(neg['share']))]

class Trees:
    def __init__(self):
        import lightgbm as lgb
        from pipeline.training_r16_multisupport import MODELS
        self.new={}
        for r in REGIMES:
            self.new[r]=[(lgb.Booster(model_file=str(MODELS/'trees'/r/f'fold_{f:02d}'/'regression.txt')),
                          lgb.Booster(model_file=str(MODELS/'trees'/r/f'fold_{f:02d}'/'classification.txt'))) for f in range(5)]
        from pipeline.trees_ensemble_r9 import PhysicsLGBM
        self.old=[];self.old_pins={}
        for f in range(5):
            folder=ROOT/f'models/r9_mac_trees_r2/fold_{f:02d}/models/lgbm_physics';s=read(folder/'summary.json')
            self.old.append(PhysicsLGBM.load(folder,expected_binding=s['training_binding']))
            for n,h in s['artifact_sha256'].items():self.old_pins[str((folder/n).relative_to(ROOT))]=h
            self.old_pins[str((folder/'summary.json').relative_to(ROOT))]=sha(folder/'summary.json')
    def predict(self,db):
        import numpy as np
        from pipeline.training_r16_multisupport import tree_features
        xr=tree_features(db,'regression');xc=tree_features(db,'classification')
        shares=[];probs=[]
        for r in REGIMES:
            for a,b in self.new[r]:shares.append(np.clip(a.predict(xr,num_threads=1),0,1));probs.append(b.predict(xc,num_threads=1))
        del xr,xc
        old=[]
        for tree in self.old:
            tree.settings['threads']=1 # inference object only; immutable boosters/receipts untouched
            old.append(tree.predict(db,batch_size=256)['share'])
        return np.column_stack(shares),np.column_stack(probs),np.column_stack(old)

def stored_members(final,name,d):
    import numpy as np
    from pipeline.r16_multisupport_data import load
    from pipeline.training_r16_multisupport import MODELS
    if name=='local_fit':
        s={};c={}
        for r in REGIMES:
            t=[load(MODELS/'trees'/r/f'fold_{f:02d}'/'selection.npz') for f in range(5)]
            if any(not np.array_equal(v['sample_id'],d['sample_id']) or not np.array_equal(v['target_share'],d['share']) for v in t):raise ValueError('Fit Tree IDs/targets differ')
            a=[np.mean([x['share'] for x in t],0)];b=[np.mean([x['classifier_prob'] for x in t],0)]
            for kind in MEMBERS[1:]:
                folder=MODELS/'neural'/r/kind;rp=load(folder/'selection_regression.npz');cp=load(folder/'selection_classification.npz')
                if any(not np.array_equal(v['sample_id'],d['sample_id']) or not np.array_equal(v['target_share'],d['share']) for v in (rp,cp)):raise ValueError('Fit neural IDs/targets differ')
                a.append(rp['share']);b.append(cp['classifier_prob'])
            s[r]=np.column_stack(a);c[r]=np.column_stack(b)
        return s,c
    raw=load(final/'evaluation'/f'{name}.npz')
    if not np.array_equal(raw['sample_id'],d['sample_id']) or not np.array_equal(raw['target_share'],d['share']):raise ValueError('Frozen panel ID/label differs')
    return {r:raw['share_'+r] for r in REGIMES},{r:raw['independent_classifier_'+r] for r in REGIMES}

def class_matrix(treeprobs,nnprobs,policy):
    import numpy as np
    cols=[];weights=[];labels=[]
    for i,r in enumerate(REGIMES):
        w=policy['experts'][r]['independent_classifier_weights']
        for f in range(5):cols.append(treeprobs[:,i*5+f]);weights.append(.5*w[0]/5);labels.append(r+'/tree'+str(f))
        for j,k in enumerate(MEMBERS[1:],1):cols.append(nnprobs[r][:,j]);weights.append(.5*w[j]);labels.append(r+'/'+k)
    return np.column_stack(cols),np.asarray(weights),labels

def static_tree_features(db,x):
    """Only grid/profile/sqrtw and their four aggregates are overridden.

    Other physics terms intentionally retain original db; this is a partial
    OOD feature intervention, not a realizable acoustic-generating process.
    """
    import numpy as np
    from pipeline.physics_features_r6_v2 import physics_features_r6_v2
    from ops.r16_temporal_features import temporal_statistics
    a=x[:,:8];stats=np.concatenate([fn(a,axis=2).reshape(len(a),-1) for fn in (np.mean,np.std,np.max,np.min)],1)
    extra,_=temporal_statistics(db)
    return np.concatenate([a.reshape(len(a),-1),stats,physics_features_r6_v2(db),extra],1).astype(np.float32)

def donors(db,sites):
    import numpy as np
    sites=np.asarray(sites).astype(str);mask=np.isfinite(db);pairs=[]
    for i in range(len(db)):
        candidates=np.flatnonzero((sites!=sites[i])&np.all(mask==mask[i],axis=(1,2)))
        if len(candidates):
            preferred=[int(j) for j in candidates if 'St_Francis' in sites[j]]
            pairs.append((i,preferred[0] if preferred else int(candidates[0])))
    return pairs

def interventions(d,trees,final,protocol):
    import numpy as np,torch
    from pipeline.training_r16_multisupport import features,MODELS,codes
    from pipeline.models_r16_multisupport import MultiSupportModel
    zero=np.flatnonzero(d['share']==0);_,ix=np.unique(d['base_window_index'][zero],return_index=True)
    zero=zero[np.sort(ix)][:protocol['maximum_intervention_unique_zero_windows_per_panel']]
    db=d['db'][zero];sites=d['site'][zero];x,w=features(db);rows=[]
    base=static_tree_features(db,x)
    from pipeline.training_r16_multisupport import tree_features
    if not np.array_equal(base,tree_features(db,'regression')):raise ValueError('Intervention feature reconstruction differs at baseline')
    pairs=donors(db,sites);cases=[]
    if pairs:
        target=np.asarray([i for i,j in pairs]);other=np.asarray([j for i,j in pairs]);cases.append(('other_site_exact_mask',target,x[other],w[other]))
    tilted=db+np.linspace(0.,12.,25)[None,None,:];tx,tw=features(tilted)
    cases.append(('same_zero_static_tilt0to12dB',np.arange(len(db)),tx,tw))
    for reg in REGIMES:
        for kind in MEMBERS:
            model=None
            if kind!='tree5':
                folder=MODELS/'neural'/reg/kind;model=MultiSupportModel(kind).eval();ck=torch.load(folder/'best_regression.pt',map_location='cpu',weights_only=True)
                if ck['code_sha256']!=codes() or ck['architecture']!=model.architecture():raise ValueError('NN identity differs')
                model.load_state_dict(ck['state_dict']);del ck
            for label,target,dx,dw in cases:
                xx=x[target];ww=w[target];raw=db[target]
                variants={'original':xx}
                for name,ch in [('input_profile_only',(4,)),('input_sqrtw_only',(7,)),('input_both_original_pool',(4,7))]:
                    a=xx.copy();a[:,ch]=dx[:,ch]
                    if not np.array_equal(a[:,[j for j in range(17) if j not in ch]],xx[:,[j for j in range(17) if j not in ch]]):raise AssertionError('Unrequested channels changed')
                    variants[name]=a
                predictions={};masks={}
                for name,a in variants.items():
                    if model is None:
                        tf=static_tree_features(raw,a);predictions[name]=np.mean([np.clip(tree.predict(tf,num_threads=1),0,1) for tree,_ in trees.new[reg]],0)
                    else:
                        out=[]
                        with torch.no_grad():
                            for start in range(0,len(a),32):
                                q=model.regressor._cell_mask(torch.from_numpy(a[start:start+32])).float().numpy()
                                out.append(np.where(ww[start:start+32]>0,q,0.))
                        q=np.concatenate(out);masks[name]=q;predictions[name]=(q*ww).sum((1,2))
                if model is not None:
                    predictions['pool_only_original_q']=(masks['original']*dw).sum((1,2))
                    predictions['joint_swapped_input_and_pool']=(masks['input_both_original_pool']*dw).sum((1,2))
                    with torch.no_grad():
                        p,_,_=model(torch.from_numpy(xx[:8]),torch.from_numpy(ww[:8]))
                    if not np.allclose(p.numpy(),predictions['original'][:8],atol=2e-6,rtol=0):raise ValueError('Audit baseline differs from production forward')
                original=predictions['original']
                for name,p in predictions.items():
                    delta=p-original
                    rows.append({'model':reg+'/'+kind,'donor_case':label,'intervention':name,'n_unique_zero_backgrounds':len(p),
                      'mean_prediction':float(p.mean()),'RMS_zero_error':float(np.sqrt(np.mean(p*p))),
                      'mean_delta':float(delta.mean()),'p95_delta':float(np.quantile(delta,.95)),
                      'max_absolute_delta':float(abs(delta).max()),'share_ge010':float(np.mean(p>=.1)),
                      'zero_label_retained':True,'feature_only_partial_OOD_not_causal_proof':name.startswith('input_')})
            del model;gc.collect()
    return {'rows':rows,'unique_negatives_available':len(zero),'exact_mask_other_site_pairs':len(pairs),
      'unmatched_other_site_count':len(zero)-len(pairs),'labels_not_modified':True,
      'production_forward_sqrtw_contract_not_changed':True,
      'retrained_15_channel_model_comparison_performed':False,
      'conclusion_limit':'Input ablation of a frozen model is NOT a trained purely-temporal comparator, nor proof of device memorization.'}

def render(result,protocol):
    rows=['# R16 첫 5종 앙상블 사후 감사','',
      '첫 CSV 생성·검증 이후 수행한 별도 CPU 감사다. 원본 모델·정책·CSV·진행 중 후속 학습을 수정하지 않았다. Test 관측값·query prediction 배열을 읽지 않았고, 모든 수치 선택/기술 분석은 local_fit 및 고정 패널에 한정했다.','',
      '## 첫 후보 성적표','', '| 패널 | RMSE | 독립 score AUC | Bias |','|---|---:|---:|---:|']
    for r in result['original_evaluation']['rows']:
        if r['model']=='blend' and '__' not in r['panel']:rows.append(f"| {r['panel']} | {r['rmse']:.6f} | {str(r['classifier_auc'])} | {r['bias']:.6f} |")
    rows+=['','## 1. 독립 분류 집계·보정 감사','',
      'Brier/Logloss는 관측 양성률과 클래스50:50 가중치를 별도로 보고한다. 50:50 가중은 평가 정의이며 Test prior 추정이 아니다. ln4 공통 차감은 AUC를 바꾸지 않는다. panel-rank는 코호트 의존 진단만 수행했고 Test에 적용하지 않았다. fixed-local_fit CDF도 별도 비교했다. top25% 조건부 AUC는 모든 방법에 같은 prob-mean 부분집합을 사용한다.','',
      '| 패널 | 집계 대상 | 방식 | AUC | Balanced Brier | Balanced Logloss | ECE10 | ≥.90 | Float64 ties |','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in result['score_audit']:
        m=r['metrics'];fmt=lambda v:'미정의' if v is None else f'{v:.6f}'
        rows.append(f"| {r['panel']} | {r['ensemble']} | {r['method']} | {fmt(m['auc'])} | {fmt(m['brier_class_balanced'])} | {fmt(m['logloss_class_balanced'])} | {m['ece10_observed_prior']:.6f} | {m['ge090']:.4f} | {m['float64_duplicate_count']} |")
    rows+=['','## 2. 연산자별 회귀·직교 잔차','',
      '새/옛 예측의 affine projection은 local_fit에서만 적합하고 다른 패널에는 고정 적용했다. 회귀 기울기는 기술 통계이며 보정에 쓰지 않았다. R9는 전체509 참조를 보았으므로 새 모델과 동일한 참조 독립성을 갖지 않는다. 모델·데이터가 함께 달라진 비교만으로 한 원인의 인과효과를 확정하지 않는다.','',
      '| 패널/연산자 | R9 비교기 | Old RMSE | New RMSE | Old Bias | New Bias | 직교잔차-옛오차 상관 | Discordant win |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in result['regression_audit']:
        a=r['metrics'];rows.append(f"| {r['panel']} / {r['operator']} | {r['old_model']} | {a['old']['rmse']:.6f} | {a['new']['rmse']:.6f} | {a['old']['bias']:.6f} | {a['new']['bias']:.6f} | {str(a['new_residual_vs_old_error_correlation'])} | {str(a['discordant_new_win_rate'])} |")
    rows+=['','## 4. 정적 채널 개입','',
      'partial-feature intervention은 production의 Ch7=sqrt(w) 동시성 계약을 의도적으로 벗어나는 OOD 민감도 검사다. 별도 mask-only 계산으로 입력효과와 풀링효과를 분리했으며 원본 forward 검사를 약화시키지 않았다. 실제 mask가 일치하는 다른 사이트 donor가 없으면 제외 수를 공개했다. 15채널 재학습 대조는 이번 감사에 포함하지 않았다.','',
      '| 모델 | donor | 개입 | n | 예측 평균 | 평균 변화 | p95 변화 | 최대 절댓값 변화 |','|---|---|---|---:|---:|---:|---:|---:|']
    for r in result['static_intervention']['rows']:rows.append(f"| {r['model']} | {r['donor_case']} | {r['intervention']} | {r['n_unique_zero_backgrounds']} | {r['mean_prediction']:.6f} | {r['mean_delta']:.6f} | {r['p95_delta']:.6f} | {r['max_absolute_delta']:.6f} |")
    rows+=['','## 5. 그룹 블록 부트스트랩 / 1-SE','',
      '행을 독립 resampling하지 않았다. background_group를 뽑고 해당 그룹의 모든 연산자를 유지했다. 최적 가중치는 in-bag local_fit에만 적합하고 위험 비교는 out-of-bag 그룹으로 했다. bootstrap SD를 √1000으로 나누지 않는다. 1-SE 중심점 추천은 별도 감사 권고이며 이미 동결된 첫 CSV를 덮어쓰지 않는다.','',
      '```json',json.dumps(result['bootstrap'],ensure_ascii=False,indent=2),'```','',
      '## 수학적·물리적 해석의 한계','',
      '- 양성 prior80%가 포화의 원인 중 하나일 수 있으나 실제 학습 init_score와 class_balance/boost_from_average를 확인하기 전 모델별 원인으로 확정하지 않는다.',
      '- 확률 평균과 로짓 평균은 순위가 달라질 수 있지만 어느 쪽이 항상 더 좋은 것은 아니다. 무모수 방법도 사후 후보 선택·평가 반복에 따른 선택 편향은 남는다.',
      '- 실제 자연 HF 과도음은 1분짜리만 있다고 가정하지 않는다. proxy hard-negative와 자연 선박 진실을 혼동하지 않는다.',
      '- 75→400+ 참조 확대는 locked B/C 및 bridge와 hull/day/UTC/bytes 분리를 재검증해야 한다. 전체509 연결그래프가 연결돼 있다는 기존 제약을 단순 바이트 중복제거로 우회하지 않는다. 이번에는 원음 풀·학습을 변경하지 않았다.',
      '- E=B+A의 일부 분해가 비식별인 예는 존재하지만 전체 Bayes error의 수치 하한이나 모든 정적 특징 사용시 오탐 폭발을 증명한 것은 아니다. 시간 특징만이 유일하게 옳은 설계라고도 단정하지 않는다.','',
      '[전체 수치](RESULTS.json) · [정책 및 검증 프로토콜](PROTOCOL.json) · [신뢰도 곡선](reliability.png)','',
      '공식 LB는 미측정이다. 이번 감사의 좋은 패널 수치를 Test 점수로 환산하지 않는다.']
    (REPORT/'R16_FIRST_ENSEMBLE_POSTCSV_AUDIT.md').write_text('\n'.join(rows)+'\n')
    (REPORT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>R16 첫 앙상블 감사</title><style>body{font:15px/1.7 system-ui;max-width:1400px;margin:30px auto;padding:20px}pre{white-space:pre-wrap}</style><h1>R16 첫 앙상블 사후 감사</h1><p><a href="R16_FIRST_ENSEMBLE_POSTCSV_AUDIT.md">전체 MD</a> · <a href="RESULTS.json">수치 JSON</a></p><img style="max-width:100%" src="reliability.png"><pre>'+html.escape('\n'.join(rows))+'</pre>')

def run():
    import numpy as np,torch
    from pipeline.r16_multisupport_data import load,save
    from pipeline.training_r16_multisupport import codes
    from ops.r16_postaudit_metrics import score_aggregates,classification_metrics,block_bootstrap,residual_audit
    protocol=read(OPS/'PROTOCOL.json');unchanged(protocol['code_sha256']);unchanged(protocol['source_artifact_sha256'])
    final,pins=verify_parent();pins.update(protocol['code_sha256']);pins.update(protocol['source_artifact_sha256'])
    if (OPS/'STARTED.json').exists() or REPORT.exists():raise FileExistsError('Fresh audit only, never reuse partial analysis')
    REPORT.mkdir(parents=True);os.environ['MPLCONFIGDIR']=str(REPORT/'.mplconfig')
    write(OPS/'STARTED.json',{'started_kst':now(),'PID':os.getpid(),'policy_sha256':sha(final/'policy.json'),'official_upload':False})
    write(REPORT/'PROTOCOL.json',protocol);sys.addaudithook(guard);torch.set_num_threads(1)
    try:os.nice(10)
    except OSError:pass
    started=time.perf_counter();policy=read(final/'policy.json');trees=Trees();pins.update(trees.old_pins)
    result={'status':'running','protocol_sha256':sha(OPS/'PROTOCOL.json'),'original_policy':policy,
            'original_evaluation':read(final/'EVALUATION.json'),'first_candidate_descriptive_statistics_only':read(final/'TEST_PREDICTION_AUDIT.json')['ensemble'],
            'score_audit':[],'individual_classifier_audit':[],'regression_audit':[],'per_operator_bias_variance':[],'no_Test_observations_or_query_arrays_read':True,
            'original_policy_and_CSV_modified':False,'no_model_training':True,'official_upload':False}
    fits=None;projection={}
    for name in protocol['panels']:
        d,weights=panel_data(name,final/'GLOBAL_FREEZE.json');s,c=stored_members(final,name,d)
        newshares,newprobs,oldshares=trees.predict(d['db'])
        for j,r in enumerate(REGIMES):
            if not np.allclose(newshares[:,5*j:5*j+5].mean(1),s[r][:,0],atol=2e-8,rtol=1e-7) or not np.allclose(newprobs[:,5*j:5*j+5].mean(1),c[r][:,0],atol=2e-8,rtol=1e-7):
                raise ValueError('Re-read complete Tree outputs differ from saved panel')
        cp,cw,labels=class_matrix(newprobs,c,policy)
        for j,label in enumerate(labels):result['individual_classifier_audit'].append({'panel':name,'model':label,'metrics':classification_metrics(d['share']>0,cp[:,j],weights)})
        if name=='local_fit':
            fits={'tree10':newprobs,'original_frozen_hybrid':cp};fit=d;fitshares=np.stack([s[r] for r in REGIMES])
            for oldname,old in [('R9_Fold0_Tree',oldshares[:,0]),('R9_Tree5',oldshares.mean(1))]:
                projection[oldname]=np.linalg.lstsq(np.column_stack([np.ones(len(old)),old]),newshares.mean(1),rcond=None)[0]
        for label,p,a in [('tree10',newprobs,None),('original_frozen_hybrid',cp,cw)]:
            methods=score_aggregates(p,a,fits[label]);top=methods['prob_mean']>=np.quantile(methods['prob_mean'],.75)
            for method,values in methods.items():result['score_audit'].append({'panel':name,'ensemble':label,'method':method,
                 'metrics':classification_metrics(d['share']>0,values,weights,top)})
            from ops.r16_postaudit_metrics import auc
            if abs(auc(d['share']>0,methods['logit_mean'],weights)-auc(d['share']>0,methods['logit_mean_prior80_to50'],weights))>1e-10:
                raise ValueError('Unexpected numerical rank change from constant logit offset')
        slices=[('ALL',np.ones(len(d['share']),bool))]
        if 'operator' in d:slices += [(str(op),d['operator']==op) for op in np.unique(d['operator'])]
        for oldname,old in [('R9_Fold0_Tree',oldshares[:,0]),('R9_Tree5',oldshares.mean(1))]:
            biases=[];newbiases=[]
            for op,mask in slices:
                m=residual_audit(d['share'][mask],old[mask],newshares.mean(1)[mask],projection[oldname],None if weights is None else weights[mask])
                result['regression_audit'].append({'panel':name,'operator':op,'old_model':oldname,'metrics':m})
                if op!='ALL':biases.append(m['old']['bias']);newbiases.append(m['new']['bias'])
            if biases:result['per_operator_bias_variance'].append({'panel':name,'old_model':oldname,'old_variance':float(np.var(biases)),
                'new_variance':float(np.var(newbiases)),'operator_count':len(biases),'includes_no_source_control_if_present':True})
        write(OPS/'state.json',{'phase':'panel_completed','panel':name,'PID':os.getpid(),'elapsed_seconds':time.perf_counter()-started,'training_modified':False})
        print(json.dumps({'stage':'panel_completed','panel':name,'n':len(d['share'])}),flush=True)
        if name=='local_report':report_data=d
        del d,s,c,cp,newshares,newprobs,oldshares;gc.collect()
    result['bootstrap']=block_bootstrap(fitshares,fit['share'],fit['background_group'],reps=protocol['bootstrap_reps'],seed=SEED)
    ref=fit['reference_component'].astype(str)
    if len(np.unique(ref))>=3:result['bootstrap_reference_cluster_sensitivity']=block_bootstrap(fitshares,fit['share'],ref,reps=protocol['bootstrap_reps'],seed=SEED+1,unit='reference_component (empty no-source component grouped together)')
    result['static_intervention']=interventions(report_data,trees,final,protocol)
    # Diagnostics only. Original policy never replaced by a post-hoc candidate.
    write(REPORT/'ONE_SE_RECOMMENDATION.json',{'original_policy_sha256':sha(final/'policy.json'),
       'recommend_center':result['bootstrap']['one_SE_center_recommended'],'center':[.5,.125,.125,.125,.125],
       'production_policy_changed':False,'no_Test_or_new_CSV':True})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    for ax,name in zip(axes.ravel(),protocol['panels']):
        ax.plot([0,1],[0,1],color='gray',linestyle='--',linewidth=1)
        for row in result['score_audit']:
            if row['panel']==name and row['ensemble']=='original_frozen_hybrid':
                rr=row['metrics']['reliability10'];ax.plot([r['mean_score'] for r in rr],[r['positive_fraction'] for r in rr],marker='o',markersize=3,label=row['method'])
        ax.set(title=name,xlim=(0,1),ylim=(0,1),xlabel='Mean predicted score',ylabel='Observed positive fraction')
        ax.grid(alpha=.2)
    axes.ravel()[0].legend(fontsize=7);fig.savefig(REPORT/'reliability.png',dpi=160);plt.close(fig)
    unchanged(pins);result.update(status='completed',elapsed_seconds=time.perf_counter()-started,code_and_artifact_sha256=pins)
    write(REPORT/'RESULTS.json',result);render(result,protocol)
    write(OPS/'COMPLETED.json',{'status':'completed','completed_kst':now(),'results_sha256':sha(REPORT/'RESULTS.json'),
        'report':str((REPORT/'R16_FIRST_ENSEMBLE_POSTCSV_AUDIT.md').relative_to(ROOT)),
        'original_CSV_sha256':pins['submissions/r16_multisupport_full_r1/submission_r16.csv'],'official_upload':False})
    write(OPS/'state.json',{'phase':'completed','completed_kst':now(),'report':str(REPORT.relative_to(ROOT)),'training_modified':False})

def supervise():
    OPS.mkdir(parents=True,exist_ok=True);lock=(OPS/'writer.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=read(OPS/'PROTOCOL.json');unchanged(p['code_sha256']);unchanged(p['source_artifact_sha256'])
    if (OPS/'STARTED.json').exists():raise FileExistsError('Do not restart a partial audit')
    while True:
        source=read(ROOT/'ops/r16_multisupport_full_r1/state.json')
        if source['phase'] in ('failed','paused'):
            write(OPS/'state.json',{'phase':'blocked_parent_'+source['phase'],'PID':os.getpid(),'updated_kst':now(),'no_restart':True});return
        if source['phase']=='completed':break
        write(OPS/'state.json',{'phase':'waiting_for_first_ensemble_CSV','PID':os.getpid(),'updated_kst':now(),
            'parent_progress':source.get('progress'),'training_modified':False});time.sleep(30)
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1','PYTHONPATH':'.:/private/tmp/underwater-ml-deps:/private/tmp/underwater-eda-deps',
         'DYLD_LIBRARY_PATH':'/opt/homebrew/opt/libomp/lib','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    with (OPS/'audit.log').open('xb',buffering=0) as log:
        result=subprocess.run([CPU,str(Path(__file__).resolve()),'run'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        write(OPS/'FAILED.json',{'status':'failed','exit_code':result.returncode,'log':str((OPS/'audit.log').relative_to(ROOT)),
            'no_retry':True,'original_training_not_modified':True});raise RuntimeError('Audit failed; no restart or model changes')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('plan','run','supervise'));a=parser.parse_args()
    if a.action=='plan':print(json.dumps(plan(),ensure_ascii=False))
    elif a.action=='run':run()
    else:supervise()

if __name__=='__main__':main()
