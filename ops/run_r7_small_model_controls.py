"""One predeclared small Ridge fit and frozen-model controls, CPU<=2 threads.

All panels are already-seen R6 development panels. No Test/old OUTER, GPU,
new CNN fit or duplicate CNN inference. Transfer-agent predictions are joined
only for the preselected fixed100 global CNN best by its own INNER.
"""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
CONFIG=ROOT/'configs/r7_small_model_controls_v1.json'
RUN=ROOT/'models/r7_small_model_controls_v1'
DATA=ROOT/'data/synthetic/r6_cpu_guardrails_v3_schemafix'
CNN=ROOT/'models/r6_fixed100_v1/r6law_relative8_global'
TREE=ROOT/'models/r6_cpu_guardrails_v2/r6law_relative_tree'
TRANSFER=ROOT/'reports/R7_FIXED100_TRANSFER_RESULTS.json'
ROLES={'inner':'inner_r6.npz','cross_generator':'cross_generator.npz','cross_sensor':'cross_sensor.npz','cross_sensor_generator':'cross_sensor_generator.npz'}
CODE=['ops/run_r7_small_model_controls.py','pipeline/physics_features_r6_v2.py','pipeline/cpu_ablation_r6_v2.py','pipeline/relative_grid_r6_v2.py','pipeline/relative_grid_r6.py']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def load(path):
    path=Path(path)
    if 'Test' in path.name or 'outer_validation' in path.name:raise ValueError('Test/old OUTER forbidden')
    import numpy as np
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in ['db','share','presence','site','family','base_window_index']}
def spec():
    m=json.loads((DATA/'manifest.json').read_text());cs=json.loads((CNN/'summary.json').read_text());ts=json.loads((TREE/'summary.json').read_text())
    gate=json.loads((ROOT/'models/r6_fixed100_v1/all_models_frozen.json').read_text())
    if gate['status']!='frozen' or cs['status']!='completed' or cs['epochs_run']!=100:raise ValueError('Completed fixed100 models required')
    if gate['model_summary_sha256']['r6law_relative8_global']!=sha(CNN/'summary.json'):raise ValueError('CNN summary freeze mismatch')
    for n,h in cs['artifact_sha256'].items():
        if n in ['best_window.pt','window_inner_predictions.npz'] and sha(CNN/n)!=h:raise ValueError('CNN artifact changed')
    for n,h in ts['model_sha256'].items():
        if sha(TREE/n)!=h:raise ValueError('Tree changed')
    selected=['train_r6.npz',*ROLES.values()]
    if any(sha(DATA/n)!=m['arrays'][n] for n in selected):raise ValueError('Input array changed')
    fixed=json.loads((ROOT/'configs/r6_fixed100_v1.json').read_text())
    if fixed['data_manifest_sha256']!=sha(DATA/'manifest.json'):raise ValueError('CNN data identity mismatch')
    old=json.loads((ROOT/'models/r6_cpu_guardrails_v2/results.json').read_text())
    if old['data_manifest_sha256']!=sha(DATA/'manifest.json'):raise ValueError('Tree data identity mismatch')
    files={str((DATA/n).relative_to(ROOT)):sha(DATA/n) for n in selected}
    for p in [DATA/'manifest.json',CNN/'summary.json',CNN/'best_window.pt',CNN/'window_inner_predictions.npz',TREE/'summary.json',TREE/'share.txt',TREE/'presence.txt',TREE/'inner_predictions.npz',ROOT/'configs/r6_fixed100_v1.json',ROOT/'models/r6_fixed100_v1/all_models_frozen.json',TRANSFER,
              ROOT/'ops/server_backups/r5_gpu_r1/restored_first_cnn_v1/fold_00/preprocessing_pilot/raw/bg_cnn/summary.json',ROOT/'configs/r5_training_gpu_r1.json']:
        files[str(p.relative_to(ROOT))]=sha(p)
    return {'version':'r7_small_model_controls_v1','status':'predeclared_before_new_Ridge_fit','threads':2,'seed':20261026,
        'data_version':str(DATA.relative_to(ROOT)),'panels':ROLES,'fit_scope':'TRAIN only; no search/early stopping/validation fit',
        'controls':['constant0','constant0_1','TRAIN_mean','Ridge27_alpha10','frozen_R6_relative_LightGBM','frozen_fixed100_R6_relative_global_CNN'],
        'Ridge':{'alpha':10.,'solver':'svd','fit_intercept':True,'features':'existing physics_features_r6_v2(db),27values/masks/counts',
            'imputer':'mean fitted TRAIN only, keep_empty_features=True','scaler':'StandardScaler fitted TRAIN only','prediction_clip':[0.,1.],
            'score':'predicted share diagnostic proxy; no calibrated presence classifier'},
        'CNN_policy':'preselected r6law_relative8_global/best_window.pt by own INNER window RMSE; no cross-panel selection',
        'CNN_transfer_report':str(TRANSFER.relative_to(ROOT)),'already_seen_development_panels_not_blind':True,
        'code_sha256':{n:sha(ROOT/n) for n in CODE},'input_sha256':files,
        'Test_old_OUTER_GPU_SSH_or_submission':False,'generator_or_existing_model_mutation':False}
def verify():
    cfg=json.loads(CONFIG.read_text())
    for n,h in {**cfg['code_sha256'],**cfg['input_sha256']}.items():
        current=sha(ROOT/n)
        if current!=h:
            ledger=RUN/'serialization_replay_patch.json'
            if n!='ops/run_r7_small_model_controls.py' or not ledger.exists():raise ValueError('Frozen R7 input/code changed: '+n)
            patch=json.loads(ledger.read_text())
            if patch['before_script_sha256']!=h or patch['after_script_sha256']!=current or patch['saved_ridge_sha256']!=sha(RUN/'ridge27_model.json') or patch['additional_fit_allowed'] is not False:raise ValueError('Unapproved serialization replay change')
    return cfg
def metric(d,p,score):
    from pipeline.cpu_ablation_r6_v2 import metrics
    return metrics(d,{'share':p,'score':score})
def prepare():
    cfg=spec();save(CONFIG,cfg);print(json.dumps({'status':'plan_saved_no_fit','config':str(CONFIG),'sha256':sha(CONFIG)},indent=2))
def restored_ridge_prediction(v,state):
    import numpy as np
    # Original features/StandardScaler output and Ridge coefficients are
    # float32. Match sklearn's in-place centering then scaling order exactly.
    z=np.array(v,dtype=np.float32,copy=True)
    if not np.isfinite(z).all():raise ValueError('Finite27-feature schema changed')
    z-=np.asarray(state['scaler_mean'],np.float32);z/=np.asarray(state['scaler_scale'],np.float32)
    return np.clip(z@np.asarray(state['coef'],np.float32)+np.float32(state['intercept']),0,1)
def fit_controls(*,replay_saved=False):
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from lightgbm import Booster
    from threadpoolctl import threadpool_limits
    from pipeline.physics_features_r6_v2 import physics_features_r6_v2,FEATURE_NAMES
    from pipeline.cpu_ablation_r6_v2 import tree_features
    cfg=verify()
    if RUN.exists() and not replay_saved:raise FileExistsError('No repeated fit reuse')
    if replay_saved:
        if not (RUN/'serialization_replay_patch.json').exists() or (RUN/'controls_results.json').exists():raise ValueError('Only one declared saved-model serialization replay allowed')
    else:
        RUN.mkdir(parents=True);save(RUN/'prefit_binding.json',{'config_sha256':sha(CONFIG),'declared_config':cfg})
    start=time.monotonic()
    train=load(DATA/'train_r6.npz');mean=float(train['share'].mean())
    with threadpool_limits(limits=2):
        if replay_saved:
            state=json.loads((RUN/'ridge27_model.json').read_text())
            scaler=StandardScaler();scaler.mean_=np.asarray(state['scaler_mean']);scaler.scale_=np.asarray(state['scaler_scale']);scaler.var_=np.asarray(state['scaler_variance']);scaler.n_features_in_=27;scaler.n_samples_seen_=state['TRAIN_examples']
            model=Ridge(alpha=10.,solver='svd',fit_intercept=True);model.coef_=np.asarray(state['coef'],np.float32);model.intercept_=np.float32(state['intercept']);model.n_features_in_=27
        else:
            f=physics_features_r6_v2(train['db']);imputer=SimpleImputer(strategy='mean',keep_empty_features=True);scaler=StandardScaler()
            x=scaler.fit_transform(imputer.fit_transform(f));model=Ridge(alpha=10.,solver='svd',fit_intercept=True);model.fit(x,train['share'])
            state={'feature_names':list(FEATURE_NAMES),'imputer_statistics':imputer.statistics_.tolist(),'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),
                   'scaler_variance':scaler.var_.tolist(),'coef':model.coef_.tolist(),'intercept':float(model.intercept_),'alpha':10.,'TRAIN_examples':len(f),'new_fit_count':1,'fit_scope':'TRAIN only'}
            save(RUN/'ridge27_model.json',state)
        reg=Booster(model_file=str(TREE/'share.txt'),params={'num_threads':2});clf=Booster(model_file=str(TREE/'presence.txt'),params={'num_threads':2})
        results={};bindings={}
        for role,name in ROLES.items():
            d=load(DATA/name);v=physics_features_r6_v2(d['db']);raw=model.predict(scaler.transform(v if replay_saved else imputer.transform(v)));ridge=np.clip(raw,0,1)
            # Explicit JSON-state reload and chunk replay, no second fit.
            replay=restored_ridge_prediction(v,state)
            if not np.allclose(replay,ridge,atol=1e-12,rtol=0):raise ValueError('Ridge JSON restore failed')
            tf=tree_features(d['db'],'relative8',None);p=np.clip(reg.predict(tf,num_threads=2),0,1).astype(np.float32);s=.5*p+.5*clf.predict(tf,num_threads=2).astype(np.float32)
            predictions={'constant0':np.zeros(len(v)),'constant0_1':np.full(len(v),.1),'TRAIN_mean':np.full(len(v),mean),'Ridge27_alpha10':ridge,'frozen_R6_relative_LightGBM':p}
            results[role]={};arrays={'base_window_index':d['base_window_index']}
            for kind,pred in predictions.items():
                score=s if kind=='frozen_R6_relative_LightGBM' else pred if kind=='Ridge27_alpha10' else np.full(len(v),.5)
                results[role][kind]=metric(d,pred,score);arrays[kind+'_share']=pred;arrays[kind+'_score']=score
            if role=='inner':
                with np.load(TREE/'inner_predictions.npz',allow_pickle=False) as z:delta=float(np.max(abs(p-z['share'])))
                if delta>1e-6:raise ValueError('Frozen tree INNER replay diverged')
                with np.load(CNN/'window_inner_predictions.npz',allow_pickle=False) as z:cp=z['share'];cs=z['score']
                cm=metric(d,cp,cs);expected=json.loads((CNN/'summary.json').read_text())['best_inner_window']['metrics']['rmse']
                if abs(cm['rmse']-expected)>1e-10:raise ValueError('Frozen CNN INNER identity diverged')
                results[role]['frozen_fixed100_R6_relative_global_CNN']=cm
            out=RUN/(role+'_control_predictions.npz');np.savez_compressed(out,**arrays)
            bindings[role]={'input_npz':str((DATA/name).relative_to(ROOT)),'input_npz_sha256':sha(DATA/name),'predictions':str(out.relative_to(ROOT)),'predictions_sha256':sha(out),
                'rows':len(v),'primary_index_order_sha256':hashlib.sha256(np.ascontiguousarray(d['base_window_index']).tobytes()).hexdigest(),
                'Ridge_json_restore_max_abs_error':float(np.max(abs(replay-ridge)))}
            print(json.dumps({'controls_completed':role,'Ridge_RMSE':results[role]['Ridge27_alpha10']['rmse']}),flush=True)
    verify();report={'status':'controls_completed_CNN_transfer_pending','config_sha256':sha(CONFIG),'results':results,'panel_bindings':bindings,
        'ridge_model_sha256':sha(RUN/'ridge27_model.json'),'TRAIN_mean':mean,'new_fit_count':1,'elapsed_seconds':time.monotonic()-start,'threads':2,
        'CNN_inference_executed_here':False,'new_CNN_tree_or_foundation_fit':False,'already_seen_development_panels_not_blind':True,
        'serialization_replay_patch':None if not replay_saved else json.loads((RUN/'serialization_replay_patch.json').read_text()),'additional_fit_during_replay':False}
    save(RUN/'controls_results.json',report);print(json.dumps({'status':report['status'],'results':str(RUN/'controls_results.json')},indent=2))
def finalize():
    import numpy as np
    verify();report=json.loads((RUN/'controls_results.json').read_text());transfer=json.loads(TRANSFER.read_text())
    # The transfer producer schema is checked explicitly before consumption.
    if transfer.get('status') not in ['completed','PASS']:raise ValueError('Completed transfer report required')
    save(RUN/'transfer_report_binding.json',{'report':str(TRANSFER.relative_to(ROOT)),'sha256':sha(TRANSFER),'preselected_CNN_checkpoint_sha256':sha(CNN/'best_window.pt')})
    report['transfer_report_sha256']=sha(TRANSFER)
    # Producer-dependent extraction is deliberately fail-closed until schema is frozen.
    evaluations=transfer.get('evaluations',{})
    if set(evaluations)!=set(['cross_generator','cross_sensor','cross_sensor_generator']):raise ValueError('Expected fixed transfer role schema; no heuristic metric matching')
    for role in ['cross_generator','cross_sensor','cross_sensor_generator']:
        e=evaluations[role]['r6law_relative8_global'];bound=report['panel_bindings'][role]
        if e['input_npz_sha256']!=bound['input_npz_sha256'] or e['checkpoint_sha256']!=sha(CNN/'best_window.pt'):raise ValueError('Transfer input/checkpoint mismatch')
        path=ROOT/e['predictions_path']
        if sha(path)!=e['predictions_sha256']:raise ValueError('Transfer predictions changed')
        d=load(DATA/ROLES[role])
        with np.load(path,allow_pickle=False) as z:
            if not np.array_equal(z['base_window_index'],d['base_window_index']):raise ValueError('Transfer primary row order mismatch')
            own=metric(d,z['share'],z['score'])
        if abs(own['rmse']-e['metrics']['rmse'])>1e-10:raise ValueError('Transfer metric recomputation mismatch')
        report['results'][role]['frozen_fixed100_R6_relative_global_CNN']=own
    report['status']='completed';report['causal_model_size_claim']=False;report['official_LB_forecast']=False
    r5file=ROOT/'ops/server_backups/r5_gpu_r1/restored_first_cnn_v1/fold_00/preprocessing_pilot/raw/bg_cnn/summary.json'
    r5=json.loads(r5file.read_text());r6=json.loads((CNN/'summary.json').read_text())
    report['R5_R6_comparison']={'R5_checkpoint_summary':str(r5file.relative_to(ROOT)),'R5_parameters':r5['receipt']['total_parameters'],
        'R5_epochs_run':r5['epochs_run'],'R5_best_epoch':r5['best_epoch'],'R5_own_INNER_RMSE':r5['best_inner_rmse'],
        'R6_parameters':r6['parameters'],'R6_epochs_run':r6['epochs_run'],'R6_best_window_epoch':r6['best_inner_window']['epoch'],
        'R6_own_INNER_RMSE':r6['best_inner_window']['metrics']['rmse'],
        'same_label_population_or_controlled_size_ablation':False,
        'loss_weights_already_same_in_R5':{'share':1.,'cell':.25,'presence':.05},
        'R5_had_direct_q_head_no_forced_prior_logit_anchor':True,'free_presence_head_R5_vs_mask_derived_score_R6':True,
        'meaning':'Different source/target generator/representation/split/training protocol; no isolated causal model-size claim or LB forecast'}
    save(ROOT/'reports/R7_SMALL_MODEL_CONTROLS_RESULTS.json',report)
    lines=['# R7 소형 모델·단순 대조군 공통 패널 평가','',
        '새 학습은 TRAIN-only27개 물리 특징 Ridge(alpha=10)1회입니다. 상수, 기존 LightGBM, fixed100CNN을 동일 INNER와3개 cross panel에서 비교했습니다. 모든 패널은 이미 평가한 개발 자료이며 독립 blind 평가가 아닙니다. Test·기존 OUTER·GPU·추가 CNN학습은 사용하지 않았습니다.','',
        '| 고정 대조군 | INNER RMSE | CROSS_GENERATOR | CROSS_SENSOR | CROSS_SENSOR_GENERATOR |','|---|---:|---:|---:|---:|']
    for name in json.loads(CONFIG.read_text())['controls']:
        lines.append('| '+name+' | '+' | '.join(f"{report['results'][role][name]['rmse']:.6f}" for role in ROLES)+' |')
    lines+=['','Ridge imputer/scaler는 TRAIN에서만 fit했고,예측은사전고정[0,1]clip입니다. 정답·site·reference ID는입력특징이아닙니다. 기존CNN은ownINNER로선택된epoch31 best_window를그대로사용했고,다른agent의고정추론을재검증하여중복CNN추론을하지않았습니다.','',
        '사이트·target 구간별 MAE/bias/RMSE·presence AUC와 모든 파일 바인딩은 [JSON](R7_SMALL_MODEL_CONTROLS_RESULTS.json)에 포함됩니다. 이 결과를 공식 LB 개선량이나 모델 크기 변화의 단독 인과효과로 해석하지 않습니다.','',
        '## R5 실패와 R6 0.059377 사이에서 실제로 달라진 것','',
        '아래 R5 수치는 보관된 Fold0 raw checkpoint summary의 자체 INNER 값입니다. 실제 제출 경로·후처리 재현은 별도 감사 대상이며, 이 표는 공개 LB와 같은 평가분포를 가정하지 않습니다. R5도 이미 direct mask를 사용했습니다. R6에서 새로 prior-logit anchor를 제거했다거나 cell/presence loss 가중치를 처음 추가했다는 설명은 맞지 않습니다.','',
        '| 항목 | R5 raw Wide CNN | R6 relative global tinyCNN | 근거 |','|---|---|---|---|',
        f"| 파라미터/구조 | {r5['receipt']['total_parameters']:,}, width128/5blocks | {r6['parameters']:,}, width32/3blocks+주파수 global context | [Wide](../pipeline/hybrid_models_r5.py#L70), [tiny](../pipeline/relative_models_r6_v2.py#L57) |",
        '| 원천·분할 | 기존 Fold0 TRAIN69 reference clips/12groups, INNER25/5; TRAIN primary53,244 | 새 sensor/source role 분리, TRAIN51/9·INNER18/3; fixed8192draws(unique5639)/INNER1024 | [R6 split](../pipeline/source_split_r6.py#L127), [감사](R6_GUARDRAILS_INDEPENDENT_AUDIT.json) |',
        '| 생성·정답 분포 | 목표 share를 .005–.95로 선추출, observed energy에서 alpha 역산; power timewarp/fade; 매epoch 새합성 | peak-SNR 기반 alpha, native minute/short passage; mask와 독립 geometry; observed y=0도 허용; 고정 합성배열 반복 | [R4](../pipeline/passage_sampler_r4.py#L198), [R6](../pipeline/robust_sampler_r6.py#L115), [고정배열 루프](../pipeline/fixed100_r6.py#L27) |',
        '| 입력·pool | raw legacy8, TRAIN-fitted background prior q0/ratio가 추가입력; raw energy weights | 0.1dB relative8, q05/sqrt(w) 포함; prior 입력 없음, rounded energy weights·unrounded label | [R5 grid](../pipeline/grid_inputs_r5.py#L117), [R6 grid](../pipeline/relative_grid_r6_v2.py#L1) |',
        '| 출력·loss | direct sigmoid q→에너지 pool; 별도 presence head; shareMSE+.25cellMSE+.05BCE | direct sigmoid q→에너지 pool 유지; mask로 계산한 fixed score에 BCE; 가중치는 동일 | [R5 loss](../pipeline/neural_r5.py#L153), [R6 score](../pipeline/relative_models_r6_v2.py#L20), [R6 loss](../pipeline/fixed100_r6.py#L31) |',
        '| optimizer/dropout | Adam lr.001, wd.0003, dropout.1 | 동일 Adam lr.001, wd.0003, dropout.1 | [R5 config](../configs/r5_training_gpu_r1.json), [R6 config](../configs/r6_fixed100_v1.json) |',
        f"| epoch·선택 | {r5['epochs_run']}epochs, ES7; best{r5['best_epoch']} | 실제100epochs/noES, ownINNER window best{r6['best_inner_window']['epoch']} | [R5 summary](../{r5file.relative_to(ROOT)}), [R6 summary](../models/r6_fixed100_v1/r6law_relative8_global/summary.json) |",
        f"| 자체 INNER RMSE | {r5['best_inner_rmse']:.9f}, 기존합성label | {r6['best_inner_window']['metrics']['rmse']:.9f}, 새합성label | 서로다른문제분포, 수치만직접우열판정불가 |",'',
        '**해석:** 원천·SNR/target law·representation·간섭control·평가split·학습기간이 함께 바뀌었습니다. 소형모델이 더 맞는 표현에서 유용한 패턴을 학습했는지는 위 공통패널 대조군으로 평가할 수 있지만, 모델 축소 하나가 R5의 공식 제출 실패를 해결했다고 증명한 실험은 아닙니다. 두 CNN 모두 from-scratch task model이며 대형 pretrained backbone의 성능을 검증한 결과도 아닙니다.']
    with (ROOT/'reports/R7_SMALL_MODEL_CONTROLS_RESULTS.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({'status':'completed','report':'reports/R7_SMALL_MODEL_CONTROLS_RESULTS.md'},indent=2))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','fit','replay-saved','finalize']);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='fit':fit_controls()
    elif a.action=='replay-saved':fit_controls(replay_saved=True)
    else:finalize()
if __name__=='__main__':main()
