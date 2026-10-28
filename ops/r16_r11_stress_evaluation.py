"""Frozen R16 small models on all saved, non-query R11 acoustic stresses.

No fitting, source regeneration, policy selection, challenge query IO or
large old-model inference. Existing R9/R14 predictions are reused only.
"""
from pathlib import Path
import csv,datetime,hashlib,json,os,sys,time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'reports/r16_r11_stress_r1';R11=ROOT/'reports/r11_diagnostics'
MODELROOT=ROOT/'models/r16_clean_pilot_r1';POLICYROOT=ROOT/'reports/r16_locked_comparison_r1'
POS=('full_smooth12','full_smooth13','full_smooth15','native_long15')
NEG=('st_francis_dense','hf_high')
OLDKINDS=('bg_cnn','tfgrid_fdy_conformer','fdy','itransformer','lgbm_physics')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def csvwrite(p,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(p).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        p=os.fsdecode(args[0]).replace('\\','/').lower()
        if (Path(p).name in ('test.csv','sample_submission.csv') or '/saved_query_audit' in p
            or '/submissions/' in p or '/eda_deep/' in p or '/r15_band_comparison_' in p
            or 'query_distribution' in p):raise PermissionError('R11 stress prohibits query or query-derived files:'+p)
    if event in ('socket.connect','subprocess.Popen','os.system'):raise PermissionError('No network/process action in stress evaluation')

def inventory():
    records=[]
    for group in ('full_support_hf_r1','asymmetric_eq_r1','asymmetric_feasible_models_r1'):
        for path in sorted((R11/group).glob('*_samples.npz')):
            records.append({'group':group,'mode':path.name.removesuffix('_samples.npz'),'path':path,
              'predictions':path.with_name(path.name.replace('_samples.npz','_predictions.npz'))})
    for mode in ('sparse','dense','dense_mask5'):
        records.append({'group':'st_francis_density_r2','mode':mode,'path':R11/'st_francis_density_r2'/(mode+'_known_components.npz'),
                        'predictions':R11/'st_francis_density_r2'/(mode+'_predictions.npz')})
    if len(records)!=31:raise ValueError('Expected31 fixed non-query R11 acoustic panels')
    return records

def metrics(y,p,c=None,score=None):
    from scipy.stats import rankdata
    y=np.asarray(y,float);p=np.asarray(p,float);e=p-y;z=y>0;n=int(z.sum());m=len(z)-n
    def auc(a):
        if a is None or not n or not m:return None
        a=np.asarray(a,float);return float((rankdata(a)[z].sum()-n*(n+1)/2)/(n*m))
    return {'n':len(y),'positive_n':n,'zero_n':m,'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(abs(e).mean()),'bias':float(e.mean()),
       'prediction_mean':float(p.mean()),'prediction_rms':float(np.sqrt(np.mean(p*p))),'target_mean':float(y.mean()),
       'auc_share':auc(p),'auc_independent_classifier':auc(c),'auc_frozen_score':auc(score),
       'classifier_ge099_fraction':float(np.mean(np.asarray(c)>=.99)) if c is not None else None}

def source_info(rec,data,pool,train,reference_full,reference_A):
    import pandas as pd
    group,mode=rec['group'],rec['mode'];folder=rec['path'].parent;n=len(data['share'])
    if 'base_window_index' in data:base=np.asarray(data['base_window_index'],int)
    elif group=='st_francis_density_r2':
        rr=[json.loads(line) for line in (folder/(mode+'_recipes.jsonl')).read_text().splitlines()]
        if len(rr)!=n:raise ValueError('Density recipes alignment failed')
        base=np.asarray([r['base_window_index'] for r in rr],int)
    else:
        meta=folder/('positive_background_manifest.parquet' if mode in ('r9_original','stationary15_control','full_smooth12','full_smooth13','full_smooth15','native_long15','full_smooth15_share030') else mode+'_background_manifest.parquet')
        frame=pd.read_parquet(meta)
        if len(frame)!=n:raise ValueError('R11 background metadata count differs')
        base=frame.base_window_index.to_numpy(int)
    if 'reference_index' in data:refs=np.asarray(data['reference_index'],int)
    elif group=='st_francis_density_r2':
        bank=read(folder/'treatment_bank.json');lookup={int(v['treatment_id']):int(v['reference_index']) for v in bank}
        refs=np.asarray([lookup.get(int(r['treatment_id']),-1) for r in rr],int)
    else:
        recipe=folder/(mode+'_recipes.json')
        refs=np.asarray([int(r.get('reference_index',-1)) for r in read(recipe)],int) if recipe.exists() else np.full(n,-1,int)
    if len(refs)!=n:raise ValueError('R11 reference recipe count differs')
    frame=pool.set_index('base_window_index').loc[base].reset_index()
    index_overlap=len(set(base)&set(train.base_window_index));group_overlap=len(set(frame.background_group)&set(train.background_group))
    gaps=[]
    for sensor,g in frame.groupby('sensor_group'):
        other=train[train.sensor_group.eq(sensor)]
        if not len(other):continue
        a=pd.to_datetime(g.start_export_clock).to_numpy(dtype='datetime64[ns]').astype('int64')/1e9
        b=np.sort(pd.to_datetime(other.start_export_clock).to_numpy(dtype='datetime64[ns]').astype('int64')/1e9)
        pos=np.searchsorted(b,a);distance=np.minimum(abs(a-b[np.clip(pos-1,0,len(b)-1)]),abs(a-b[np.clip(pos,0,len(b)-1)]))
        gaps.extend((np.maximum(distance-900.,0)/3600).tolist())
    seen_refs=set(refs[refs>=0]);selected=reference_full[reference_full.reference_row_index.isin(seen_refs)]
    overlap_refs=set(reference_A.reference_row_index)&seen_refs
    report={'panel':rec['name'],'rows':n,'unique_backgrounds':len(set(base)),'background_source_groups':frame.background_group.nunique(),
       'actual_R16_TRAIN_background_ID_overlap':index_overlap,'actual_R16_TRAIN_background_group_overlap':group_overlap,
       'minimum_same_sensor900s_support_gap_hours':min(gaps) if gaps else None,
       'background_unseen_and24h_separated':index_overlap==0 and group_overlap==0 and (not gaps or min(gaps)>=24-1e-8),
       'references_unique':len(seen_refs),'reference_exact_clip_overlap_with_R16_A':len(overlap_refs),
       'reference_hulls_overlapping_R16_A':len(set(selected.hull_id)&set(reference_A.hull_id)),
       'reference_recording_days_overlapping_R16_A':len(set(selected.recording_day)&set(reference_A.recording_day)),
       'reference_independence_not_assumed':True,'exposed_development_stress_only':True}
    return base,refs,frame.site.astype(str).to_numpy(),report

def create_small_backends():
    import torch,lightgbm as lgb
    from ops.r16_clean_models import SmallDecoupledCNN
    states={}
    for arm in ('control','mixed'):
        p=MODELROOT/arm;s=read(p/'COMPLETED.json')
        if s['status']!='completed' or s['epochs']!=100 or s['early_stopping'] is not False:raise ValueError('Frozen100epoch small model required')
        for name,h in {**s['checkpoint_sha256'],**s['tree_sha256']}.items():
            if sha(p/name)!=h:raise ValueError('R16 frozen model changed')
        reg=torch.load(p/'best_regression.pt',map_location='cpu',weights_only=True)
        clf=torch.load(p/'best_classifier.pt',map_location='cpu',weights_only=True)
        model=SmallDecoupledCNN().to(memory_format=torch.channels_last).eval()
        states[arm]={'model':model,'reg_state':reg['state_dict'],'clf_state':clf['state_dict'],
            'same_checkpoint':s['checkpoint_sha256']['best_regression.pt']==s['checkpoint_sha256']['best_classifier.pt'],
            'tree':lgb.Booster(model_file=str(p/'tree_regressor.txt')),'classifier':lgb.Booster(model_file=str(p/'tree_classifier.txt'))}
    return states

def predict_small(db,backends,policies):
    import torch
    from pipeline.relative_grid_r6_v2 import relative_grid_v2
    from pipeline.features_r12 import compact_relative_features
    from ops.r16_clean_models import temporal_features
    x,w=relative_grid_v2(db);regfeatures=compact_relative_features(db,include_shape=False);clffeatures=temporal_features(db)
    pred={};max_pool=0.;max_missing=0.
    for arm,state in backends.items():
        model=state['model'];shares=[];scores=[];model.load_state_dict(state['reg_state'],strict=True)
        with torch.no_grad():
            for start in range(0,len(db),256):
                xx=torch.from_numpy(x[start:start+256]).contiguous(memory_format=torch.channels_last);ww=torch.from_numpy(w[start:start+256])
                p,q,l=model(xx,ww);max_pool=max(max_pool,float(abs(p-(q*ww).sum((1,2))).max()))
                if (ww==0).any():max_missing=max(max_missing,float(abs(q[ww==0]).max()))
                shares.append(p.numpy());scores.append(l.sigmoid().numpy())
        nn_p=np.concatenate(shares)
        if not state['same_checkpoint']:
            scores=[];model.load_state_dict(state['clf_state'],strict=True)
            with torch.no_grad():
                for start in range(0,len(db),256):
                    _,_,l=model(torch.from_numpy(x[start:start+256]).contiguous(memory_format=torch.channels_last),torch.from_numpy(w[start:start+256]));scores.append(l.sigmoid().numpy())
        nn_c=np.concatenate(scores);tree_p=np.clip(state['tree'].predict(regfeatures,num_threads=2),0,1);tree_c=state['classifier'].predict(clffeatures,num_threads=2)
        for label,key,p,c in [('NN','nn',nn_p,nn_c),('Tree','tree',tree_p,tree_c),('60T40N','fixed60tree40nn',.6*tree_p+.4*nn_p,.5*tree_c+.5*nn_c)]:
            lam=policies[arm+'__'+key]['score_share_lambda'];score=lam*p+(1-lam)*c
            if not np.isfinite(np.r_[p,c,score]).all() or ((np.r_[p,c,score]<0)|(np.r_[p,c,score]>1)).any():raise ValueError('Invalid frozen small prediction')
            pred[arm+'_'+label]={'share':p,'classifier':c,'score':score,'score_share_lambda':lam}
    if max_pool>3e-6 or max_missing!=0:raise ValueError('Physical pooling/missing mask failed')
    return pred,{'physical_pool_max_abs_error':max_pool,'missing_mask_q_max':max_missing}

def load_legacy(rec,n):
    saved=npz(rec['predictions']);result={};group=rec['group']
    for i,kind in enumerate(OLDKINDS):
        name='R9_'+kind if kind!='lgbm_physics' else 'R9_tree_f0'
        p=saved['share'][:,i] if group=='st_francis_density_r2' else saved[kind+'_share']
        s=saved['score'][:,i] if group=='st_francis_density_r2' else saved[kind+'_score']
        c=saved.get(kind+'_score_classifier')
        if len(p)!=n or len(s)!=n:raise ValueError('Saved legacy prediction row count differs')
        result[name]={'share':p,'score':s,'classifier':c,'identity':'hash-pinned same original sample file; positional rows; no saved per-row IDs'}
    key='all5_frozen' if group=='full_support_hf_r1' else 'frozen_all5'
    p=saved['share'][:,-1] if group=='st_francis_density_r2' else saved[key+'_share']
    s=saved['score'][:,-1] if group=='st_francis_density_r2' else saved[key+'_score']
    result['R9_all5_frozen_Fold0']={'share':p,'score':s,'classifier':None,'identity':'hash-pinned original sample row order'}
    return result

def main():
    if OUT.exists():raise FileExistsError('Fresh stress report required; no partial reuse')
    sys.addaudithook(guard);OUT.mkdir(parents=True);(OUT/'predictions').mkdir();started=time.perf_counter()
    records=inventory();sourcepins={};modelpins={};codepins={str(Path(__file__).relative_to(ROOT)):sha(__file__)}
    for p in [POLICYROOT/'GLOBAL_FREEZE.json',POLICYROOT/'POLICIES.json',MODELROOT/'PLAN.json',MODELROOT/'ALL_COMPLETED.json',ROOT/'ops/r16_clean_models.py']:
        modelpins[str(p.relative_to(ROOT))]=sha(p)
    policies=read(POLICYROOT/'POLICIES.json')['policies']
    for arm in ('control','mixed'):
        p=MODELROOT/arm;s=read(p/'COMPLETED.json');modelpins[str((p/'COMPLETED.json').relative_to(ROOT))]=sha(p/'COMPLETED.json')
        for name,h in {**s['checkpoint_sha256'],**s['tree_sha256']}.items():modelpins[str((p/name).relative_to(ROOT))]=h
    for rec in records:
        rec['name']='R11_'+rec['mode'] if rec['group']=='full_support_hf_r1' else 'R11_'+rec['group']+'__'+rec['mode']
        for p in [rec['path'],rec['predictions']]:
            if not p.is_file():raise ValueError('Missing frozen R11 artifact')
            sourcepins[str(p.relative_to(ROOT))]=sha(p)
    for group in ('full_support_hf_r1','asymmetric_eq_r1','asymmetric_feasible_models_r1','st_francis_density_r2'):
        for p in (R11/group).iterdir():
            if p.suffix in ('.json','.jsonl','.parquet'):sourcepins[str(p.relative_to(ROOT))]=sha(p)
    pairnames=['R11_'+p+'_VS_'+n for p in POS for n in NEG]
    for name in pairnames:
        for parent in ('reports/r14_production_r2/hybrid_fast3','reports/r14_tree_full_r1/predictions'):
            p=ROOT/parent/(name+'.npz')
            if p.exists():sourcepins[str(p.relative_to(ROOT))]=sha(p)
    metadata_paths=['data/versions/background_cv_r9/background_pool_manifest.parquet','reports/r16_cross_operator_r1/data/train_backgrounds.parquet',
                    'reports/r16_cross_operator_r1/data/reference_partitions.parquet','data/versions/vtuad_full_r9/reference_manifest.parquet']
    for p in metadata_paths:sourcepins[p]=sha(ROOT/p)
    plan={'scope':'frozen31nonqueryR11base acoustic panels and8standard pair combinations; no source regeneration',
       'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'base_panels':[{'name':r['name'],'path':str(r['path'].relative_to(ROOT))} for r in records],
       'standard_pairs':pairnames,'primary_benchmark':'R16 original crossoperator/source panels remain primary; R11 is stress diagnosis only',
       'excluded':['saved_query_audit_r1:queryderived','failed st_francis_density_r1','model-free-only arithmetic/feasibility folders without stored acoustic panels'],
       'new_models':['control_NN','control_Tree','control_60T40N','mixed_NN','mixed_Tree','mixed_60T40N'],
       'old_models':'saved R9 available individual4NN+Treef0+originalaggregate; R14 only exact cached8standardpairs; other23-catalogue entries unmeasured',
       'score':'original local_fit-only score lambdas frozen, plus independent classifier and share-only reported separately',
       'no_Test_or_Testderived_read':True,'no_fit_or_threshold_weight_selection':True,'old_heavy_model_inference':False,
       'new_small_CPU_threads':2,'input_file_sha256':sourcepins,'R16_frozen_file_sha256':modelpins,'code_sha256':codepins}
    dump(OUT/'PLAN.json',plan)
    import torch,pandas as pd
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    backends=create_small_backends();pool=pd.read_parquet(ROOT/metadata_paths[0]);train=pd.read_parquet(ROOT/metadata_paths[1])
    reference_A=pd.read_parquet(ROOT/metadata_paths[2]);reference_A=reference_A[reference_A.r16_reference_partition.eq('A')]
    reference_full=pd.read_parquet(ROOT/metadata_paths[3]);rows=[];overlap=[];panels={};qa=[];availability=[];outputs={}
    catalog=['R9_'+k for k in OLDKINDS[:-1]]+['R9_tree_f'+str(i) for i in range(5)]+['R14_'+k for k in OLDKINDS[:-1]]+['R14_tree_'+v+'_f'+str(i) for i in range(5) for v in ('baseline','highshare')]
    for rec in records:
        data=npz(rec['path']);db=data['db'];y=data['share'];n=len(y)
        if db.shape!=(n,15,25) or not np.isfinite(y).all() or ((y<0)|(y>1)).any():raise ValueError('R11 sample schema differs')
        base,refs,site,proof=source_info(rec,data,pool,train,reference_full,reference_A);overlap.append(proof)
        ids=data.get('sample_id',np.asarray([rec['name']+'_'+str(i) for i in range(n)])).astype(str)
        pr,pq=predict_small(db,backends,policies);qa.append({'panel':rec['name'],**pq})
        old=load_legacy(rec,n);allpred={**pr,**old}
        for model,p in allpred.items():
            rows.append({'panel':rec['name'],'group':rec['group'],'model':model,'source':'new frozen R16 inference' if model in pr else 'exact saved R9 reuse',
                **metrics(y,p['share'],p.get('classifier'),p['score'])})
        for model in catalog:availability.append({'panel':rec['name'],'model':model,'measured':model in old,'method':'saved same-panel reuse' if model in old else 'not measured; no new old-model inference'})
        names=list(pr);path=OUT/'predictions'/(rec['name']+'.npz')
        np.savez_compressed(path,sample_id=ids,target_share=y,presence=(y>0).astype(np.int64),base_window_index=base,reference_index=refs,
          site=site,model_order=np.asarray(names),share=np.column_stack([pr[k]['share'] for k in names]),
          independent_classifier=np.column_stack([pr[k]['classifier'] for k in names]),frozen_score=np.column_stack([pr[k]['score'] for k in names]))
        outputs[str(path.relative_to(OUT))]=sha(path)
        panels[rec['name']]={'ids':ids,'y':y,'new':pr,'old':old,'proof':proof}
        print(json.dumps({'panel':rec['name'],'n':n,'seconds':time.perf_counter()-started}),flush=True)
    # Pairing existing fixed sets is evaluation bookkeeping, not synthesis.
    cache_checks=[]
    for positive in POS:
        for negative in NEG:
            name='R11_'+positive+'_VS_'+negative;a=panels['R11_'+positive];b=panels['R11_'+negative]
            ids=np.r_[a['ids'],b['ids']];y=np.r_[a['y'],b['y']];pr={}
            for model in a['new']:
                pr[model]={k:np.r_[a['new'][model][k],b['new'][model][k]] for k in ('share','classifier','score')}
            for model,p in pr.items():rows.append({'panel':name,'group':'standard_R11_pair','model':model,'source':'fixed base predictions concatenated',**metrics(y,p['share'],p['classifier'],p['score'])})
            for model in a['old']:
                p=np.r_[a['old'][model]['share'],b['old'][model]['share']];s=np.r_[a['old'][model]['score'],b['old'][model]['score']]
                ca=a['old'][model].get('classifier');cb=b['old'][model].get('classifier');c=np.r_[ca,cb] if ca is not None and cb is not None else None
                rows.append({'panel':name,'group':'standard_R11_pair','model':model,'source':'saved fixed R9 base predictions concatenated',**metrics(y,p,c,s)})
            measured=set(a['old'])
            cache=ROOT/'reports/r14_production_r2/hybrid_fast3'/(name+'.npz')
            if cache.exists():
                d=npz(cache)
                if not np.array_equal(d['sample_id'].astype(str),ids) or not np.array_equal(d['target_share'],y):raise ValueError('R14 common panel IDs/targets differ')
                members=d['member_order'].tolist();mapping={'tree_baseline':'R14_tree_baseline_f0','tree_highshare':'R14_tree_highshare_f0',
                   'nn_itransformer':'R14_itransformer','nn_fdy':'R14_fdy','nn_bg_cnn':'R14_bg_cnn'}
                for member,model in mapping.items():
                    p=d['member_share'][:,members.index(member)];measured.add(model)
                    rows.append({'panel':name,'group':'standard_R11_pair','model':model,'source':'exact R14 cache ID/target match; own score=share','rmse':metrics(y,p)['rmse'],**metrics(y,p,None,p)})
                rows.append({'panel':name,'group':'standard_R11_pair','model':'R14_fast3_exactFold0','source':'exact R14 cache ID/target match',**metrics(y,d['share'],None,d['score'])})
                cache_checks.append({'panel':name,'path':str(cache.relative_to(ROOT)),'IDs_targets_exact':True})
            treecache=ROOT/'reports/r14_tree_full_r1/predictions'/(name+'.npz')
            if treecache.exists():
                d=npz(treecache)
                if not np.array_equal(d['sample_id'].astype(str),ids) or not np.array_equal(d['target_share'],y):raise ValueError('R14 Tree cache IDs/targets differ')
                for variant in ('baseline','highshare'):
                    i=d['model_order'].tolist().index('FullTree_'+variant+'P1');p=d['share'][:,i];s=d['score'][:,i];c=d[variant+'P1_classifier']
                    rows.append({'panel':name,'group':'standard_R11_pair','model':'R14_tree_'+variant+'_f0_rawclf','source':'exact pinned FullTree Fold0 cache',**metrics(y,p,c,s)})
                cache_checks.append({'panel':name,'path':str(treecache.relative_to(ROOT)),'IDs_targets_exact':True})
            for model in catalog:availability.append({'panel':name,'model':model,'measured':model in measured,'method':'saved same-panel reuse' if model in measured else 'not measured; no new old-model inference'})
            path=OUT/'predictions'/(name+'.npz');names=list(pr)
            np.savez_compressed(path,sample_id=ids,target_share=y,model_order=np.asarray(names),share=np.column_stack([pr[k]['share'] for k in names]),
                independent_classifier=np.column_stack([pr[k]['classifier'] for k in names]),frozen_score=np.column_stack([pr[k]['score'] for k in names]))
            outputs[str(path.relative_to(OUT))]=sha(path)
    csvwrite(OUT/'metrics.csv',rows);csvwrite(OUT/'source_overlap.csv',overlap);csvwrite(OUT/'old_model_comparison_availability.csv',availability)
    for p,h in {**sourcepins,**modelpins,**codepins}.items():
        if sha(ROOT/p)!=h:raise ValueError('Frozen stress/model/policy artifact changed:'+p)
    lookup={(r['panel'],r['model']):r for r in rows};lines=['# R16 동결 소형 모델의 고정 R11 스트레스 평가','',
      'R16 원래 교차 연산자 검증을 제1기준으로 유지하고, 이 평가는 이미 노출된 고정 R11 스트레스의 사후 진단으로만 수행했다. Test·Test 파생 파일·기존모델 재학습·생성기 수정·정책 재선택은 없다. 원본 패널을 새로 합성하지 않았다.','',
      f'저장 음향31개(총{sum(len(p["y"]) for p in panels.values()):,}행)에 control/mixed NN·Tree·고정60T40N을 추론했다. 표준 R11 양성4×음성2 조합8개는 기존 예측을 이어 붙여 평가했다. 조합/반복을 독립 원천으로 세지 않는다.','',
      '## 표준 장시간·SF/HF 조합','',
      '| 패널 | control NN RMSE | mixed NN RMSE | mixed NN 독립분류 AUC | mixed NN 동결score AUC | R9 기존all5 AUC | R14 기존Fast3 AUC |','|---|---:|---:|---:|---:|---:|---:|']
    for name in pairnames:
        a=lookup[name,'control_NN'];b=lookup[name,'mixed_NN'];old=lookup[name,'R9_all5_frozen_Fold0'];r14=lookup.get((name,'R14_fast3_exactFold0'),{})
        lines.append(f"| {name} | {a['rmse']:.6f} | {b['rmse']:.6f} | {b['auc_independent_classifier']:.6f} | {b['auc_frozen_score']:.6f} | {old['auc_frozen_score']:.6f} | {r14.get('auc_frozen_score',float('nan')):.6f} |")
    lines +=['','## 정확한 비교와 미측정','',
      'R9 과거4NN·Tree Fold0·동결 all5 예측은 같은 원본 samples 파일 SHA와 원래 저장 순서로 재사용했다. 일부 옛 NPZ에는 per-row ID가 없어 SHA-결박 positional comparison이라는 한계를 기록한다. R14는 표준8개 조합에만 정확한 sample_id와 정답이 일치한 캐시를 사용했다. 비교 파일이 없는 기존23모델의 항목은 미측정으로 표기했고 새로 추론하지 않았다. 독립 classifier가 저장되지 않은 옛 출력은 AUC를 만들거나 복원해 채우지 않았다. R9 derived score와 R14 share-only NN score를 독립분류 확률이라고 부르지 않는다.','',
      '## 원천·정답·해석 한계','',
      'source_overlap.csv는 실제 R16 TRAIN4,000개 배경 ID/group/900초 지원 간격과 R11 각 패널을 비교한다. 참조는 정확한 clip index뿐 아니라 R16 A와 hull·날짜 중복도 보고한다. 배경이 분리돼도 참조가 겹치면 미관측 참조 일반화라고 부르지 않는다. 전체509 참조 그래프가 연결돼 있다는 기존 한계도 유지한다.','',
      'R11 full_smooth 계열은 통계적으로 늘린/평탄한 장시간 성분 등 사전 고정된 가정 밖 challenge를 포함한다. 이것이 대회 생성기라는 증거는 없다. edge 복원과 QC대조, 비대칭 EQ의 원래 불가능/확장 조건, 자연SF sparse/dense의2016–18/2015 시기 차이 및 5view 반복도 그대로 유지한다. density의paired dense→5분 삭제는 다른 실제 sparse 녹음과 구분한다. 추가0 정답은 자연 선박 부재가 아니다.','',
      '단일 class만 있는 원시 패널은 AUC 미정의로 저장한다. λ는 R16 local_fit에서 동결된0/.25 그대로 사용하며, 독립분류 AUC와 share-only AUC도 별도로 보고한다. 스트레스 결과로 체크포인트·가중치·λ·threshold를 다시 선택하지 않았다. 새 full5fold OOF, 공식 LB 향상 또는 모든 스트레스 해결을 주장하지 않는다.','',
      '[모든 지표](metrics.csv) · [원천 중복](source_overlap.csv) · [기존모델 미측정표](old_model_comparison_availability.csv) · [사전 계획](PLAN.json)']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    result={'status':'completed_frozen_stress_diagnostic','base_panels':31,'standard_pair_panels':8,
      'base_rows':sum(len(p['y']) for p in panels.values()),'new_small_models':6,'metric_rows':len(rows),'metrics':rows,
      'source_overlap':overlap,'physical_pool_checks':qa,'exact_R14_cached_panel_matches':cache_checks,
      'R16_original_benchmark_remains_primary':True,'new_fit_or_policy_selection_or_Test_access':False,
      'old_heavy_models_reinferred':False,'frozen_sources_and_models_unchanged':True,'elapsed_seconds':time.perf_counter()-started,
      'input_source_sha256':sourcepins,'model_policy_sha256':modelpins,'code_sha256':codepins,
      'output_sha256':{**outputs,**{n:sha(OUT/n) for n in ['metrics.csv','source_overlap.csv','old_model_comparison_availability.csv','README.md','PLAN.json']}}}
    dump(OUT/'RESULTS.json',result);print(json.dumps({'status':result['status'],'path':str(OUT),'metric_rows':len(rows),'seconds':result['elapsed_seconds']}))

if __name__=='__main__':main()
