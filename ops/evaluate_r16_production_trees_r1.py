"""Read-only R16 Tree5 audit; only selection, local_gate and own-fold CV.

Report/domain panels stay closed until the hybrid global model/policy freeze.
No fitting, Test access, or model/policy changes are performed here.
"""
from __future__ import annotations
import hashlib,json,os,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/r16_production_tree_evaluation_r1'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def write(p,v):
    p=Path(p);t=p.with_name(p.name+'.tmp')
    t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)

def guard(event,args):
    if event=='open'and args and isinstance(args[0],(str,bytes,os.PathLike)):
        n=os.fsdecode(args[0]).replace('\\','/')
        if n.endswith('/Test.csv')or '/data/local_report/'in n or '/data/domain_holdout/'in n:
            raise PermissionError('Query and globally locked report/domain inputs forbidden')

def main():
    sys.addaudithook(guard)
    from pipeline.trees_r16_production import PhysicsLGBM,measure,transform
    from pipeline.r16_production_data import validate_production_manifest,load_production_role
    from ops.r16_forward_data import load_panel
    regpath=ROOT/'models/r16_production_trees_r1/ALL_FIVEFOLD_TREES_FROZEN.json'
    registry=json.loads(regpath.read_text())
    if registry.get('status')!='completed'or registry.get('all_models_frozen')is not True or set(registry['models'])!=set(map(str,range(5))):
        raise ValueError('Completed five-fold model freeze required')
    pins=registry['artifact_sha256'].copy();pins[str(regpath.relative_to(ROOT))]=sha(regpath)
    for n,h in pins.items():
        if sha(ROOT/n)!=h:raise ValueError('Registry artifact mismatch '+n)
    if OUT.exists():raise FileExistsError('Fresh audit output required')
    OUT.mkdir(parents=True);(OUT/'predictions').mkdir()
    began=time.time();models=[];folds=[]
    for f in range(5):
        folder=ROOT/registry['models'][str(f)]['model_dir'];s=json.loads((folder/'summary.json').read_text())
        mp=ROOT/f'data/synthetic/r16_production_v1/fold_{f:02d}/manifest.json'
        validate_production_manifest(mp,f,verify_sources=True)
        binding={'fold':f,'manifest_sha256':sha(mp),'recipe':'R16_improved_mixed6_unchanged','source_refs':'A only; B/C protected'}
        models.append(PhysicsLGBM.load(folder,expected_binding=binding))
        folds.append({'fold':f,'train_rows':s['train_rows'],'selection_rows':s['selection_rows'],
          'elapsed_seconds':s['elapsed_seconds'],'iterations':s['iterations'],'selection_metrics':s['selection_metrics']})
    plan={'status':'frozen_before_evaluation','members':[str(f)for f in range(5)],'family_weights':[.2]*5,
      'score':'mean of five independent classifier probabilities; share weight=0',
      'roles':['saved_selection','local_gate','own_fold_cv_validation'],
      'local_report_domain_deferred_until_full_hybrid_global_freeze':True,
      'source_artifact_sha256':pins,'Test_used':False,'training_performed':False,'policy_selection':False}
    write(OUT/'PLAN.json',plan)
    metrics=[];selection=[]
    for f,m in enumerate(models):
        with np.load(ROOT/registry['models'][str(f)]['model_dir']/'selection_predictions.npz',allow_pickle=False)as z:
            d={k:z[k]for k in z.files}
        if selection and (not np.array_equal(d['sample_id'],selection[0]['sample_id'])or not np.array_equal(d['target_share'],selection[0]['target_share'])):
            raise ValueError('Shared selection identity differs')
        selection.append(d)
        metrics.append({'panel':'shared_selection','model':f'tree_fold{f}',**measure(d['target_share'],d['share'],d['classifier_prob'])})
    sp=np.mean([d['share']for d in selection],0);sc=np.mean([d['classifier_prob']for d in selection],0)
    metrics.append({'panel':'shared_selection','model':'Tree5_equal',**measure(selection[0]['target_share'],sp,sc)})
    np.savez_compressed(OUT/'predictions/shared_selection.npz',share=sp,classifier_prob=sc,
       target_share=selection[0]['target_share'],sample_id=selection[0]['sample_id'])
    manifest=ROOT/'reports/r16_cross_operator_r1/data/manifest.json';dm=json.loads(manifest.read_text())
    gate=[]
    for op in dm['panels']['local_gate']:
        d=load_panel(manifest,'local_gate',op)
        xreg=transform(d['db'],'regression');xclf=transform(d['db'],'classification')
        ps=[];cs=[]
        for f,m in enumerate(models):
            p=np.clip(m.regressor.predict(xreg,num_threads=2),0,1);c=m.classifier.predict(xclf,num_threads=2)
            ps.append(p);cs.append(c)
            metrics.append({'panel':'local_gate/'+op,'model':f'tree_fold{f}',**measure(d['share'],p,c)})
        p=np.mean(ps,0);c=np.mean(cs,0);gate.append((d['share'],p,c))
        metrics.append({'panel':'local_gate/'+op,'model':'Tree5_equal',**measure(d['share'],p,c)})
        np.savez_compressed(OUT/f'predictions/local_gate__{op}.npz',share=p,classifier_prob=c,
            target_share=d['share'],sample_id=d['sample_id'],site=d['site'],
            model_share=np.asarray(ps),model_classifier_prob=np.asarray(cs))
        print(json.dumps({'completed':'local_gate/'+op,'seconds':time.time()-began}),flush=True)
    metrics.append({'panel':'local_gate/pooled6','model':'Tree5_equal',**measure(*[np.concatenate([g[j]for g in gate])for j in range(3)])})
    cv=[]
    for f,m in enumerate(models):
        mp=ROOT/f'data/synthetic/r16_production_v1/fold_{f:02d}/manifest.json'
        d=load_production_role(mp,f,'cv_validation');pr=m.predict(d['db']);cv.append((d['share'],pr['share'],pr['classifier_prob']))
        metrics.append({'panel':f'own_fold_cv/fold{f}','model':f'tree_fold{f}',**measure(d['share'],pr['share'],pr['classifier_prob'])})
        np.savez_compressed(OUT/f'predictions/own_fold_cv__{f}.npz',**pr,target_share=d['share'],sample_id=d['sample_id'],site=d['site'])
        print(json.dumps({'completed':f'own_fold_cv/{f}','seconds':time.time()-began}),flush=True)
    metrics.append({'panel':'own_fold_cv/pooled5000','model':'each_own_fold_only',**measure(*[np.concatenate([g[j]for g in cv])for j in range(3)])})
    for n,h in pins.items():
        if sha(ROOT/n)!=h:raise ValueError('Frozen source or artifact changed during evaluation')
    result={'status':'completed','folds':folds,'metrics':metrics,'registry_entries_verified':len(pins),
      'all_registry_source_weight_SHA_passed':True,'selection_is_shared_development_not_OOF':True,
      'own_fold_CV_is_restricted_development_pool_only':True,
      'own_fold_CV_never_averages_foreign_fold_models':True,
      'local_report_domain_deferred_until_full_hybrid_global_freeze':True,
      'new_training':False,'policy_tuning':False,'Test_used':False,'elapsed_seconds':time.time()-began,
      'source_artifact_sha256':pins,'script_sha256':sha(__file__)}
    write(OUT/'RESULTS.json',result)
    lines=['# R16 본체 PhysicsLightGBM 5-Fold 결과','',
      '회귀 3,827특징·독립 분류 700특징. 모든 가중치와 원천 SHA를 확인한 뒤 동결 평가했다. 새 학습·Test 접근·정책 튜닝은 없다.','',
      '## 학습 완료 현황','',
      '| Fold | TRAIN | selection | 회귀/분류 round | 학습 분 | selection RMSE | 독립 분류 AUC |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for s in folds:
        q=s['selection_metrics'];it=s['iterations']
        lines.append(f"| {s['fold']} | {s['train_rows']:,} | {s['selection_rows']:,} | {it['regression']}/{it['classification']} | {s['elapsed_seconds']/60:.2f} | {q['rmse']:.6f} | {q['classifier_auc']:.6f} |")
    lines+=['','selection은 모든 Fold가 공유한 개발용 validation이다. 이 수치를 전체 OOF 또는 공식 LB로 부르지 않는다.','',
       '## 동결 평가','', '| 패널 | 모델 | N | RMSE | share AUC | 독립 분류 AUC | bias |', '|---|---|---:|---:|---:|---:|---:|']
    for q in metrics:
        if q['model'] not in ('Tree5_equal','each_own_fold_only')and not q['panel'].startswith('own_fold_cv/'):continue
        lines.append(f"| {q['panel']} | {q['model']} | {q['n']:,} | {q['rmse']:.6f} | {q['share_auc']:.6f} | {q['classifier_auc']:.6f} | {q['bias']:+.6f} |")
    lines+=['','각 Fold의 cv_validation에는 해당 Fold 모델만 적용했다. 다른 Fold 모델은 이 원천을 학습했을 수 있으므로 이 패널에서 Tree5 평균을 내지 않았다.','',
       'local_report 및 domain_holdout의 새 본체 평가는 신경망 4종과 최종 앙상블 정책이 모두 고정된 뒤 수행한다. 기존 접근 보호를 우회하지 않았다.','',
       '전체 수치·검산 근거: [RESULTS.json](RESULTS.json).']
    (OUT/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    write(OUT/'FINAL_VERIFICATION.json',{'status':'passed','output_sha256':{str(p.relative_to(OUT)):sha(p)for p in OUT.rglob('*')if p.is_file()},'Test_used':False})
    print(json.dumps({'status':'completed','metrics':len(metrics),'seconds':time.time()-began}),flush=True)

if __name__=='__main__':main()
