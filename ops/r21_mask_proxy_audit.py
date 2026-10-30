"""R21 frozen classifier-input ablation: no training, Test, or policy fitting.

The original feature width and model weights stay fixed. Only 25 pure
observation fractions may be attenuated toward TRAIN medians. The remaining
375 broadly 'observed'-named statistics contain actual audio changes and are
preserved. Neural O hiding is an explicit, optional out-of-distribution probe.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import datetime as dt, hashlib, json, sys
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'reports/r21_clean_val_zero_training_r1/mask_audit'
STRENGTHS=(0.,.5,1.)

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def guard(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
  n=Path(os.fsdecode(args[0])).name.lower()
  if n in ('test.csv','query.npz','query_predictions.npz') or 'submission' in n:
   raise PermissionError('R21 mask audit excludes Test and submissions')

@lru_cache(maxsize=1)
def feature_contract():
 from ops.r17_clean_classifier_features import original_names
 from pipeline.trees_r18_production import schemas
 names=original_names();ix=np.array([i for i,n in enumerate(names) if n.startswith('observed_fraction/')],int)
 audio=np.array([i for i,n in enumerate(names) if 'observed' in n and not n.startswith('observed_fraction/')],int)
 if len(names)!=2033 or len(ix)!=25 or len(audio)!=375:raise ValueError('Audited schema differs')
 new=schemas()
 if len(new['classification_profile_energy'])!=2408 or len(new['classification_removed_explicit_mask'])!=25:raise ValueError('R18 schema differs')
 return names,ix,audio,new

@lru_cache(maxsize=1)
def neutral_contract():
 p=OUT/'TRAIN_NEUTRAL_VALUES.json';d=json.loads(p.read_text())
 if d['script_sha256']!=sha(__file__):raise ValueError('Neutralization code changed since freeze')
 return d

def neutralize_old_classifier_features(features,strength=1.,fold=0,regime='D0'):
 """Preserve N×2033 shape; x'=(1-s)x+s TRAIN_median on PURE25 only.

 Returned array is a copy; raw db/O/w/q/y and original features are untouched.
 Fold/regime-specific contract prevents using evaluation-population statistics.
 """
 if strength not in STRENGTHS:raise ValueError('Use the predeclared strengths 0/.5/1')
 names,ix,_audio,_new=feature_contract();x=np.asarray(features,np.float32)
 if x.ndim!=2 or x.shape[1]!=len(names) or not np.isfinite(x).all():raise ValueError('Expected finite N×2033')
 med=np.array(neutral_contract()['per_job'][f'{regime}/fold_{int(fold):02d}']['neutral_pure25'],np.float32)
 out=x.copy();out[:,ix]=(1-float(strength))*out[:,ix]+float(strength)*med
 return out

def transformed_old_classifier_features(db,strength=1.,fold=0,regime='D0'):
 from pipeline.training_r16_multisupport import tree_features
 return neutralize_old_classifier_features(tree_features(db,'classification'),strength,fold,regime)

def mask_nn_classifier_input(x,hide_observation=False):
 """OPTIONAL classifier-only O-channel hiding probe, never default production.

 Accept old 15-channel classifier tensors after CLASSIFIER_CHANNELS selection.
 Source Ch5 observed_mask occupies classifier position5. Other14 channels are
 unchanged. This deliberately shifts the trained input distribution and cannot
 eliminate missingness encoded indirectly by audio validity/zeros. Pooling
 weights, raw O, regression input, truth, and all model parameters stay intact.
 """
 if x.ndim!=4 or tuple(x.shape[1:])!=(15,15,25):raise ValueError('Expected old classifier N×15×15×25')
 out=x.clone() if hasattr(x,'clone') else np.array(x,copy=True)
 if hide_observation:out[:,5]=0
 return out

def main():
 sys.addaudithook(guard);OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'COMPLETED.json').exists():raise FileExistsError('Immutable completed audit exists')
 protocol={'registered_KST':dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(timespec='seconds'),
 'model_training':False,'GPU_MPS':False,'Test_read':False,'new_LGB_feature_columns':False,
 'strengths':list(STRENGTHS),'operation':'constant TRAIN-median attenuation of 25 pure observation fractions, feature width remains2033',
 'production_policy_changed':False,'NN_O_hide':'optional diagnostic only; source Ch5=classifier position5 set0; regression/rawO/w/q/y unchanged',
 'no_claim':'Cannot guarantee all mask proxies disappear; audio validity and sqrt(w) still encode missingness',
 'source_script_sha256':sha(__file__)}
 write(OUT/'PROTOCOL.json',protocol)
 names,ix,audio,new=feature_contract();pin={};jobs={};compositions=[];checks=[]
 for fold in range(5):
  p=ROOT/f'data/synthetic/r16_multisupport_v1/fold_{fold:02d}/train_backgrounds.parquet';f=pd.read_parquet(p)
  obs=f.qc_observed_minutes.to_numpy(float)/15;med=float(np.median(obs));pin[str(p.relative_to(ROOT))]=sha(p)
  for regime in ('D0','H0'):
   jobs[f'{regime}/fold_{fold:02d}']={'n':len(f),'neutral_pure25':[med]*25,'source':'actual TRAIN metadata qc_observed_minutes/15; all25bands share minute mask',
    'source_path':str(p.relative_to(ROOT)),'source_sha256':sha(p),'per_band_missingness':'full25band minute masks; no partial-band rows in source contract'}
   compositions.extend({'fold':fold,'regime':regime,'nobs':int(v),'n':int(c),'fraction':float(c/len(f))}for v,c in f.qc_observed_minutes.value_counts().sort_index().items())
  if fold==0:
   for regime in ('D0','H0'):
    wp=ROOT/f'data/cache/r16_multisupport_full_r1/{regime}/train/w.npy';mp=wp.with_name('manifest.json');m=json.loads(mp.read_text())
    if sha(wp)!=m['artifact_sha256']['w.npy']:raise ValueError('Actual TRAIN w cache SHA differs')
    w=np.load(wp,mmap_mode='r');cnt=[];partial=0
    for off in range(0,len(w),2048):
     o=w[off:off+2048]>0;cnt.append(o.any(2).sum(1));partial+=int((o.any(2)!=o.all(2)).sum())
    nobs=np.concatenate(cnt)
    if not np.array_equal(nobs,f.qc_observed_minutes.to_numpy()) or partial:raise ValueError('TRAIN metadata vs actual cache mask differs')
    checks.append({'regime':regime,'fold':0,'actual_cache_n':len(w),'metadata_Nobs_exactly_matches_all_rows':True,'partial_band_minutes':partial,'median_observed_fraction':float(np.median(nobs/15))})
    pin[str(wp.relative_to(ROOT))]=sha(wp);pin[str(mp.relative_to(ROOT))]=sha(mp)
 write(OUT/'TRAIN_NEUTRAL_VALUES.json',{'script_sha256':sha(__file__),'per_job':jobs,'statistics_from_TRAIN_only':True,'evaluation_labels_or_Test_used':False,'neural_mask_channel_index':5,'old_classifier_feature_indices':ix.tolist(),'all_neutral_values_are_one':all(all(v==1 for v in j['neutral_pure25'])for j in jobs.values())})
 pd.DataFrame(compositions).to_csv(OUT/'TRAIN_MASK_COMPOSITION.csv',index=False)
 pd.DataFrame([{'index':i,'name':n,'pure25':bool(i in ix),'audio_observed375':bool(i in audio),'operation':'TRAIN_median_attenuation' if i in ix else 'unchanged'}for i,n in enumerate(names)]).to_csv(OUT/'FEATURE_DECISIONS_2033.csv',index=False)
 # Recompute actual LightGBM importances; no CatBoost import/training required.
 import lightgbm as lgb
 importance=[];gainrows=[]
 for regime in ('D0','H0'):
  for fold in range(5):
   path=ROOT/f'models/r16_multisupport_full_r1/trees/{regime}/fold_{fold:02d}/classification.txt';before=sha(path);booster=lgb.Booster(model_file=str(path))
   if booster.num_feature()!=2033:raise ValueError('Frozen old model width mismatch')
   gain=booster.feature_importance(importance_type='gain').astype(float);splits=booster.feature_importance(importance_type='split').astype(float);total=max(gain.sum(),1e-30)
   for category,indices in [('pure_observation_fraction',ix),('audio_observed375',audio),('broad_name_observed400',np.r_[ix,audio])]:
    importance.append({'regime':regime,'fold':fold,'category':category,'features':len(indices),'gain_fraction':float(gain[indices].sum()/total),'split_count':int(splits[indices].sum())})
   for i in ix:gainrows.append({'regime':regime,'fold':fold,'feature_index':int(i),'feature_name':names[i],'gain':float(gain[i]),'gain_fraction':float(gain[i]/total),'split_count':int(splits[i])})
   if sha(path)!=before:raise ValueError('Model changed')
   pin[str(path.relative_to(ROOT))]=before
 for regime in ('D0','H0'):
  path=ROOT/f'models/r18_production_r1/trees/{regime}/classification.txt';booster=lgb.Booster(model_file=str(path))
  if booster.num_feature()!=2408:raise ValueError('New model width mismatch')
  pin[str(path.relative_to(ROOT))]=sha(path)
 pd.DataFrame(importance).to_csv(OUT/'FROZEN_LGB_IMPORTANCE_DECOMPOSITION.csv',index=False)
 pd.DataFrame(gainrows).to_csv(OUT/'PURE25_IMPORTANCE_ALL_JOBS.csv',index=False)
 oldimp=ROOT/'reports/r17_clean_pipeline_r1/features/classifier_importance_decomposition_mean.csv'
 im=pd.read_csv(oldimp);pin[str(oldimp.relative_to(ROOT))]=sha(oldimp);im.to_csv(OUT/'HISTORICAL_LGB_CAT_IMPORTANCE_DECOMPOSITION.csv',index=False)
 # Deterministic schema fixtures: complete observations must be unchanged;
 # only pure25 may change at missing minutes, and neural hiding changes only O.
 from pipeline.training_r16_multisupport import tree_features
 rng=np.random.default_rng(20261030);db=np.round(rng.normal(70,5,(8,15,25)),1).astype(np.float32);db[1,0]=np.nan;db[2,:10]=np.nan
 x=tree_features(db,'classification');z=neutralize_old_classifier_features(x,1.)
 changed=np.flatnonzero(np.any(x!=z,axis=0));assert set(changed)<=set(ix) and np.array_equal(x[0],z[0])
 assert np.array_equal(x,neutralize_old_classifier_features(x,0.))
 nt=rng.normal(size=(8,15,15,25)).astype(np.float32);nh=mask_nn_classifier_input(nt,True)
 assert np.all(nh[:,5]==0) and np.array_equal(nt[:,np.arange(15)!=5],nh[:,np.arange(15)!=5])
 checks.append({'fixture_rows':8,'changed_columns_at_strength1':changed.tolist(),'changed_columns_subset_of_pure25':True,'fully_observed_row_exactly_unchanged':True,'identity_strength0_exact':True,'NN_only_position5_changes':True,'regression_or_rawO_w_q_y_modified':False})
 write(OUT/'NUMERIC_QA.json',{'status':'passed','checks':checks})
 summary=pd.DataFrame(importance).groupby(['category','features'],as_index=False).gain_fraction.mean()
 text='# R21 동결 분류 입력의 마스크 대리변수 감사\n\n'
 text+='재학습·GPU/MPS·Test 접근·가중치 수정 없음. LightGBM의 기존2,033차원 입력을 유지하고 **순수 관측비율25개만** TRAIN 중앙값 쪽으로 약화하는 API를 구현했다. 새 feature를 추가한 LightGBM이나 새 학습 모델이 아니다.\n\n'
 text+='## 실제 중요도 구분\n\n'+summary.to_markdown(index=False,floatfmt='.8f')+'\n\n'
 text+='R17 이름 기반 observation_mask_proxy400개 중375개는 실제 음향의 adjacent delta75개 및 contiguous lag300개다. 기존15.34%/21.35% 전체를 순수 마스크로 해석하면 안 된다. 역사적 LGB 순수25 Gain은0.8280%, CatBoost PredictionValuesChange는3.0977%였다. 두 중요도의 정의도 다르다. 실제 모델10개별 수치와25개 칼럼별 수치는 CSV에 보존했다.\n\n'
 text+='## 고정 입력 대조 API\n\n'
 text+='`transformed_old_classifier_features(db, strength=0/.5/1, fold, regime)`는 N×2033을 반환한다. 순수25 칼럼만 x′=(1−s)x+s TRAIN_median으로 변경한다. 모든5fold·D0/H0 TRAIN의 중앙 관측비율은1.0이었다. 따라서 **완전 관측창은 어떤 strength에서도 원본과 완전히 동일**하여 그 창의 오탐을 이 방법만으로 해결할 수 없다.\n\n'
 text+='`neutralize_old_classifier_features(features, ...)`는 이미 계산된 특징에 같은 연산을 한다. 실제TRAIN mask metadata와fold0 D0/H0의 w cache 전체행이 정확히 일치함을 검산했다. 타깃 라벨은 보지 않았다. R18의2,408특징 분류기는 이미순수25를제거했지만 시간validity·zero·sqrt(w)의0위치로 결측정보가간접표현될수있다. 이를모두제거하면실제관측의미와음향정보를함께파괴한다.\n\n'
 text+='## 신경망\n\n'
 text+='old DINO/Dasheng의 분류15채널에는 원본Ch5 observed_mask가분류입력position5로남는다. 선택적 `mask_nn_classifier_input(x, hide_observation=True)`는분류입력position5만0으로숨긴다. 이는학습분포밖입력에대한진단이며자동생산채택하지않는다. raw db·실제O·회귀x17·물리w·정답q/y와모델가중치는보존한다. 나머지14채널의음향통계에도결측validity가반영되어완전한마스크정보제거를보장하지않는다.\n\n'
 text+='## 불가능한 것과 검증 범위\n\n'
 text+='동결 LightGBM에 quiet_B_share 열을단순추가해도기존split은그열을사용하지않으며 feature폭검사가실패한다. 새열을학습에반영하거나기존트리split가중치를낮추는일은재학습또는모델변형이다. 이번에는그일을하지않았다. 입력중립화가AUC/오탐을실제로개선하는지는새Clean검증셋의동결예측으로평가해야하며본모듈자체는성과를주장하지않는다.\n'
 (OUT/'R21_MASK_PROXY_AUDIT.md').write_text(text)
 write(OUT/'INPUT_AND_MODEL_SHA256.json',pin)
 manifest={str(p.relative_to(OUT)):sha(p)for p in sorted(OUT.rglob('*'))if p.is_file()and p.name not in ('ARTIFACT_SHA256.json','COMPLETED.json')}
 write(OUT/'ARTIFACT_SHA256.json',manifest)
 write(OUT/'COMPLETED.json',{'status':'completed','files':len(manifest),'script_sha256':sha(__file__),'pure_observation_features':len(ix),'audio_observed_features_preserved':len(audio),'new_model_training':False,'model_optimizer_steps':0,'Test_read':False,'production_policy_changed':False})
 print(json.dumps({'status':'completed','path':str(OUT),'pure25_indices':ix.tolist(),'all_TRAIN_neutral_values':1.0},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
