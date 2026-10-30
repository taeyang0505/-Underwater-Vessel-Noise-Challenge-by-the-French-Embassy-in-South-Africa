"""Saved-tree importance and conditional-error audit, no retraining/inference."""
import sys,json,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ops.r17_audit_common import *
from pipeline.trees_r16_production import schemas
from ops.r16_temporal_features import temporal_statistics

def category(n):
 if 'observed' in n:return 'observation_mask_proxy'
 if 'sqrt_rounded_energy' in n:return 'energy_weight_sqrtw'
 if 'profile' in n or 'band_median_minus_window_median' in n:return 'relative_static_profile'
 if 'lag' in n or 'autocorr' in n or 'persistence' in n or 'rolling' in n or 'arch' in n:return 'time_structure'
 if 'corr' in n or 'eigen' in n:return 'cross_band_temporal'
 if 'q05' in n or 'quantile' in n or 'q20' in n:return 'within_window_quantile'
 if n.startswith('grid/') and ('t00/'in n or 't14/'in n):return 'first_last_minute_proxy'
 return 'other_relative_or_physics'
def main():
 import lightgbm as lgb
 from catboost import CatBoostRegressor,CatBoostClassifier
 out=OUT/'shortcut';out.mkdir(parents=True,exist_ok=True);extra=temporal_statistics(np.ones((1,15,25),np.float32))[1];schemas0=schemas();names={k:schemas0[k]+['F1/'+n for n in extra] for k in ['regression','classification']};tops=[];cats=[];pins={}
 for reg in ['D0','H0']:
  for fold in range(5):
   for kind in names:
    p=ROOT/f'models/r16_multisupport_full_r1/trees/{reg}/fold_{fold:02d}/{kind}.txt';m=lgb.Booster(model_file=str(p));assert m.num_feature()==len(names[kind]);pins[str(p.relative_to(ROOT))]=sha(p)
    for typ in ['gain','split']:
     imp=m.feature_importance(importance_type=typ);den=max(float(imp.sum()),1e-30)
     for rank,j in enumerate(np.argsort(-imp)[:50],1):tops.append(dict(model='LightGBM',regime=reg,fold=fold,task=kind,type=typ,rank=rank,index=int(j),feature=names[kind][j],importance=float(imp[j]),fraction=float(imp[j]/den),category=category(names[kind][j])))
     c={}
     for n,v in zip(names[kind],imp):c[category(n)]=c.get(category(n),0)+v
     cats.extend(dict(model='LightGBM',regime=reg,fold=fold,task=kind,type=typ,category=k,fraction=float(v/den)) for k,v in c.items())
    p=ROOT/f'models/r16_catboost_peft_cpu8_r1/{reg}/fold_{fold:02d}/{kind}.cbm';m=(CatBoostRegressor()if kind=='regression'else CatBoostClassifier());m.load_model(str(p));pins[str(p.relative_to(ROOT))]=sha(p);imp=m.get_feature_importance(type='PredictionValuesChange',thread_count=2);assert len(imp)==len(names[kind]);den=max(float(imp.sum()),1e-30)
    for rank,j in enumerate(np.argsort(-imp)[:50],1):tops.append(dict(model='CatBoost',regime=reg,fold=fold,task=kind,type='PredictionValuesChange',rank=rank,index=int(j),feature=names[kind][j],importance=float(imp[j]),fraction=float(imp[j]/den),category=category(names[kind][j])))
    c={}
    for n,v in zip(names[kind],imp):c[category(n)]=c.get(category(n),0)+v
    cats.extend(dict(model='CatBoost',regime=reg,fold=fold,task=kind,type='PredictionValuesChange',category=k,fraction=float(v/den))for k,v in c.items())
    print('IMPORTANCE',reg,fold,kind,flush=True)
 csv(out/'tree_top50_all.csv',tops);csv(out/'tree_importance_category_mass.csv',cats);jsave(out/'tree_sources_sha256.json',pins);jsave(out/'tree_feature_names.json',names)
 tables={p.stem:pd.read_parquet(p)for p in (OUT/'eda').glob('*.parquet')};conditional=[];correlations=[];overlap=[]
 for panel in ['ensemble_fit','local_gate','local_report','domain_holdout','native12_VS_hf_high_negative','native12_VS_st_francis_dense_negative']:
  f=tables[panel];f=f.assign(ybin=pd.cut(f.share,[-1e-12,.0,.03,.05,.15,.30,1.],include_lowest=True).astype(str))
  f['HFbin']=pd.cut(f.bump_high_db,[-.01,3,10,20,1000]).astype(str);f['LFbin']=pd.cut(f.LF_energy_fraction,[0,.5,.8,.95,1.00001],include_lowest=True).astype(str);f['supportbin']=pd.cut(f.support_minutes,[-1,0,3,7,11,15]).astype(str)
  for dim in ['HFbin','LFbin','supportbin','site','n_obs','operator']:
   if dim not in f:continue
   for (yb,key),z in f.groupby(['ybin',dim],observed=True):
    for model in ['lightgbm5','catboost5','dinov2_small','dasheng_base','timemoe']:
     error=z['pred_'+model]-z.share;conditional.append(dict(panel=panel,y_bin=yb,condition=dim,value=str(key),n=len(z),model=model,y_mean=z.share.mean(),prediction_mean=z['pred_'+model].mean(),rmse=np.sqrt((error**2).mean()),bias=error.mean(),HF_bump_median=z.bump_high_db.median(),LF_fraction_median=z.LF_energy_fraction.median()))
  z=f[f.share.between(.05,.15)]
  for model in ['lightgbm5','catboost5','dinov2_small','dasheng_base']:
   for col in ['bump_high_db','LF_energy_fraction','support_minutes','n_obs','q05_contaminated_band_fraction','q20_contaminated_band_fraction']:
    correlations.append(dict(panel=panel,model=model,feature=col,n=len(z),pearson_error=z['error_'+model].corr(z[col]),spearman_error=z['error_'+model].corr(z[col],method='spearman')))
  overlap.append(dict(panel=panel,n=len(f),zero_fraction=(f.share==0).mean(),support0_fraction=(f.support_minutes==0).mean(),quantization_error_max=f.quantization_error_max.max(),positive_fullsupport_fraction=(f.loc[f.share>0,'support_minutes']>=12).mean()))
 csv(out/'conditional_error_all.csv',conditional);csv(out/'same_y_005_015_error_correlations.csv',correlations);csv(out/'mask_support_quantization.csv',overlap)
 cat=pd.DataFrame(cats);agg=cat.groupby(['model','task','type','category']).fraction.mean().reset_index();csv(out/'importance_category_average.csv',agg)
 (out/'SHORTCUT_AUDIT.md').write_text('# R17 편법 단서 및 정답-특징 전달성 감사\n\n학습·새 모델 추론0회. 20개 LightGBM(2레시피×5fold×2task)의 gain/split과 20개CatBoost의 PredictionValuesChange를 전수 추출했다. 서로 다른 중요도 정의를 같은 gain/split으로 부르지 않았다.\n\n'+agg.to_markdown(index=False)+'\n\n중요도는 연관된 모델 의존도이며 인과적 편법 증거가 아니다. 양자화는 모든입력이0.1dB이므로 단일 격자오차만으로 양성/음성을 가르는 신호는 이번자료에서 확인할 수 없다. 참조 삽입 경계·ambient·지원분과O겹침은 관련특징 대리변수와 조건부오차로만 평가했으며 중요도만으로 원인을 확정하지 않았다. ambient원음의 분리정답은 없으므로 주변음 성분에 대한 확정비율은 미측정이다.\n\nR16에는 PCA/절대db/LFanchor/q0입력이 없지만 회귀에는 상대정적profile과sqrt(w)가 남아있고, 분류에는 두 채널이 제외된다. 이를 모두 정적특징 제거라고 부르면 부정확하다.\n\n동일y구간[0.05,0.15]의 HF·LF분모·support·site·n_obs 오차는conditional_error_all.csv와same_y_005_015_error_correlations.csv에 있다. 숨겨진Test정답과 비교한 조건부오차는 계산하지 않았다.\n')
 print('SHORTCUT_DONE',flush=True)
if __name__=='__main__':main()
