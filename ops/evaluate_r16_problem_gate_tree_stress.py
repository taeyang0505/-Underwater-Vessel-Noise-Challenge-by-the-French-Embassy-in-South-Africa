"""Frozen Tree5 on four literal R11 long-passage/HF-negative combinations."""
from pathlib import Path
import hashlib,json,os,sys,time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/r16_problem_gate_r1/tree_stress'
NAMES=('native_long15','full_smooth15','hf_high','st_francis_dense')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def guard(event,args):
    if event=='open'and args and isinstance(args[0],(str,bytes,os.PathLike)):
        n=os.fsdecode(args[0]).replace('\\','/').lower()
        if n.endswith('/test.csv')or '/submissions/'in n or '/saved_query_audit' in n:
            raise PermissionError('No Test or query-derived audit access')

def metrics(y,p,c):
    from sklearn.metrics import roc_auc_score
    z=y>0;e=p-y
    return {'n':len(y),'positive_n':int(z.sum()),'zero_n':int((~z).sum()),
      'RMSE':float(np.sqrt(np.mean(e*e))),'bias':float(e.mean()),'target_mean':float(y.mean()),
      'share_AUC':float(roc_auc_score(z,p))if z.any()and (~z).any()else None,
      'classifier_AUC':float(roc_auc_score(z,c))if z.any()and (~z).any()else None,
      'share_mean_positive':float(p[z].mean())if z.any()else None,'share_mean_negative':float(p[~z].mean())if (~z).any()else None,
      'classifier_mean_positive':float(c[z].mean())if z.any()else None,'classifier_mean_negative':float(c[~z].mean())if (~z).any()else None,
      'classifier_ge099':float(np.mean(c>=.99)),'classifier_ge090':float(np.mean(c>=.9)),
      'classifier_negative_ge050':float(np.mean(c[~z]>=.5))if (~z).any()else None,
      'classifier_positive_below050':float(np.mean(c[z]<.5))if z.any()else None,
      'positive_negative_rank_inversion_fraction':float(1-roc_auc_score(z,c))if z.any()and (~z).any()else None}

def main():
    sys.addaudithook(guard)
    from pipeline.trees_r16_production import PhysicsLGBM,transform
    start=time.time();registry_path=ROOT/'models/r16_production_trees_r1/ALL_FIVEFOLD_TREES_FROZEN.json'
    registry=json.loads(registry_path.read_text());pins=registry['artifact_sha256'].copy();pins[str(registry_path.relative_to(ROOT))]=sha(registry_path)
    if registry['status']!='completed'or registry['all_models_frozen']is not True or set(registry['models'])!=set(map(str,range(5))):raise ValueError('Tree5 freeze missing')
    oldroot=ROOT/'reports/r16_r11_stress_r1';oldplan=json.loads((oldroot/'PLAN.json').read_text());oldreceipt=json.loads((oldroot/'FINAL_VERIFICATION.json').read_text())
    for n in ('PLAN.json','RESULTS.json','FINAL_VERIFICATION.json'):pins[str((oldroot/n).relative_to(ROOT))]=sha(oldroot/n)
    oldresults=json.loads((oldroot/'RESULTS.json').read_text())
    for n,h in pins.items():
        if sha(ROOT/n)!=h:raise ValueError('Frozen pin changed '+n)
    if OUT.exists():raise FileExistsError('Fresh report directory required')
    OUT.mkdir(parents=True);(OUT/'predictions').mkdir()
    models=[];training=[]
    for f in range(5):
        folder=ROOT/registry['models'][str(f)]['model_dir'];s=json.loads((folder/'summary.json').read_text());models.append(PhysicsLGBM.load(folder,expected_binding=s['binding']))
        mp=ROOT/f'data/synthetic/r16_production_v1/fold_{f:02d}/manifest.json';dm=json.loads(mp.read_text());e=dm['roles']['train'];tp=ROOT/e['path']
        if sha(tp)!=e['sha256']:raise ValueError('Frozen train provenance changed')
        pins[e['path']]=e['sha256']
        with np.load(tp,allow_pickle=False)as z:training.append({'backgrounds':set(z['base_window_index'].tolist()),'groups':set(z['background_group'].tolist()),'references':set(z['reference_index'].tolist())})
    plan={'status':'frozen_before_inference','literal_base_panels':list(NAMES),'pairing':'existing arrays concatenated; no synthesis',
      'family_weights':[.2]*5,'score':'mean independent classifier probability; share weight0',
      'new_training':False,'Test_used':False,'new_recipe_or_policy_selection':False,'source_artifact_sha256':pins}
    dump(OUT/'PLAN.json',plan)
    panels={};source=[];rows=[]
    for name in NAMES:
        npzpath=ROOT/f'reports/r11_diagnostics/full_support_hf_r1/{name}_samples.npz'
        rel=str(npzpath.relative_to(ROOT));h=oldplan['input_file_sha256'][rel]
        if sha(npzpath)!=h:raise ValueError('Original literal R11 bytes changed')
        pins[rel]=h
        with np.load(npzpath,allow_pickle=False)as z:db=z['db'];y=z['share']
        pp=oldroot/f'predictions_numeric/R11_{name}.npz';ph=oldreceipt['numeric_prediction_file_sha256'][str(pp.relative_to(oldroot))]
        if sha(pp)!=ph:raise ValueError('Prior numeric prediction receipt differs')
        pins[str(pp.relative_to(ROOT))]=ph
        with np.load(pp,allow_pickle=False)as z:old={k:z[k]for k in z.files}
        if not np.array_equal(y,old['target_share']):raise ValueError('Prior comparison targets differ')
        xr=transform(db,'regression');xc=transform(db,'classification');ps=[];cs=[]
        for f,m in enumerate(models):
            p=np.clip(m.regressor.predict(xr,num_threads=2),0,1);c=m.classifier.predict(xc,num_threads=2);ps.append(p);cs.append(c)
        p=np.mean(ps,0);c=np.mean(cs,0)
        panels[name]={'y':y,'p':p,'c':c,'member_p':np.asarray(ps),'member_c':np.asarray(cs),'old':old}
        for f,t in enumerate(training):
            source.append({'panel':name,'fold':f,'rows':len(y),'literal_sample_path':rel,'literal_sample_sha256':h,
              'TRAIN_background_ID_overlap':len(t['backgrounds']&set(old['base_window_index'].tolist())),
              'TRAIN_reference_clip_overlap':len(t['references']&set(old['reference_index'][old['reference_index']>=0].tolist())),
              'source_is_original_R11_exposed_stress_not_new_unseen_reference':True})
        rows.append({'panel':'R11_'+name,'model':'R16_Tree5',**metrics(y,p,c)})
        np.savez_compressed(OUT/f'predictions/R11_{name}.npz',sample_id=old['sample_id'],target_share=y,share=p,classifier_prob=c,member_share=np.asarray(ps),member_classifier_prob=np.asarray(cs))
    for positive in ('native_long15','full_smooth15'):
        for negative in ('hf_high','st_francis_dense'):
            a=panels[positive];b=panels[negative];y=np.r_[a['y'],b['y']];p=np.r_[a['p'],b['p']];c=np.r_[a['c'],b['c']];name=f'R11_{positive}_VS_{negative}'
            rows.append({'panel':name,'model':'R16_Tree5',**metrics(y,p,c)})
            for f in range(5):rows.append({'panel':name,'model':f'R16_tree_fold{f}',**metrics(y,np.r_[a['member_p'][f],b['member_p'][f]],np.r_[a['member_c'][f],b['member_c'][f]])})
            for model in ('control_NN','control_Tree','control_60T40N','mixed_NN','mixed_Tree','mixed_60T40N'):
                idx=a['old']['model_order'].tolist().index(model);pold=np.r_[a['old']['share'][:,idx],b['old']['share'][:,idx]];cold=np.r_[a['old']['independent_classifier'][:,idx],b['old']['independent_classifier'][:,idx]]
                rows.append({'panel':name,'model':'R16small_'+model,**metrics(y,pold,cold)})
            np.savez_compressed(OUT/f'predictions/{name}.npz',target_share=y,share=p,classifier_prob=c)
    for n,h in pins.items():
        if sha(ROOT/n)!=h:raise ValueError('Source artifact changed during evaluation')
    result={'status':'completed','metrics':rows,'source':source,'registry_source_weights_unchanged':True,
      'new_training':False,'Test_used':False,'new_data_synthesized':False,'policy_or_recipe_selection':False,
      'physical_target_meaning':'R11 known added energy share; no hidden challenge labels',
      'source_artifact_sha256':pins,'script_sha256':sha(__file__),'elapsed_seconds':time.time()-start}
    dump(OUT/'RESULTS.json',result)
    lines=['# R16 본체 Tree5 — 고정 R11 장시간/HF 실패 조건 감사','',
      '기존 R11 원본 NPZ SHA를 확인하고, 완료된 Tree5(동일 가중 평균)를 그대로 평가했다. 새 합성·학습·Test 접근·가중치 조정은 없다.','',
      '| 고정 패널 | 모델 | RMSE | bias | share AUC | 독립 분류 AUC | 분류 양성 평균 | 분류 음성 평균 | p≥.99 |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for q in rows:
        if '_VS_'not in q['panel']or q['model'] not in ('R16_Tree5','R16small_mixed_Tree','R16small_mixed_60T40N'):continue
        lines.append(f"| {q['panel']} | {q['model']} | {q['RMSE']:.6f} | {q['bias']:+.6f} | {q['share_AUC']:.6f} | {q['classifier_AUC']:.6f} | {q['classifier_mean_positive']:.6f} | {q['classifier_mean_negative']:.6f} | {q['classifier_ge099']:.3%} |")
    lines+=['','확인된 사실: 위 고정 조건에서 출력 역전·오류·분류 확률 포화를 직접 측정했다. 이 결과만으로 비공개 Test의 동일 조건 존재나 Public 붕괴의 단일 원인을 확정하지 않는다.','',
      '가설: 시간 상대 특징이 긴 통과를 기준선으로 차감하는 현상과 자연 HF 음성에 대한 오탐이 남았는지는 별도 구조 검증이 필요하다. 이번 감사는 모델을 수정하지 않는다.','',
      '이 패널은 이미 반복 노출된 개발 스트레스다. 배경 및 참조의 실제 TRAIN 중복은 RESULTS.json source에 Fold별로 표기했으며, 미관측 참조 성능으로 주장하지 않는다.']
    (OUT/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    dump(OUT/'VERIFICATION.json',{'status':'passed','output_sha256':{str(p.relative_to(OUT)):sha(p)for p in OUT.rglob('*')if p.is_file()},'Test_used':False})
    print(json.dumps({'status':'completed','seconds':time.time()-start,'rows':len(rows)}),flush=True)

if __name__=='__main__':main()
