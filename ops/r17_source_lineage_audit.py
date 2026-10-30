"""Full numeric-bank lineage and reuse audit. No synthesis/training/Test."""
import sys,json,warnings
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ops.r17_audit_common import *
from pipeline.dataset_r9 import R9Sources,make_schedule
from pipeline.r16_multisupport_data import role
from scipy.stats import ks_2samp
import xarray as xr

def reuse_rows(v,label,kind):
 counts=pd.Series(v).value_counts();p=counts/counts.sum()
 return dict(dataset=label,unit=kind,uses=int(counts.sum()),unique=int(len(counts)),minimum=int(counts.min()),median=float(counts.median()),maximum=int(counts.max()),kish_concentration_ess=float(1/(p*p).sum()),largest_fraction=float(p.max()))
def main():
 out=OUT/'source';out.mkdir(parents=True,exist_ok=True);s=R9Sources(verify_banks=False);r=s.references.copy()
 a=pd.read_parquet(ROOT/'data/synthetic/r16_multisupport_v1/reference_A_folds.parquet');tr=pd.read_parquet(ROOT/'data/synthetic/r16_multisupport_v1/fold_00/train_references.parquet')
 selected=set(a.clip_id);train=set(tr.clip_id);rows=[]
 for i,row in r.iterrows():
  x=s.reference_minutes[i];p=x.mean(0);den=p.sum();cv=x.std(0)/np.maximum(p,1e-30)
  rows.append(dict(reference_index=i,clip_id=row.clip_id,weak_class=row.weak_class,duration_seconds=float(row.duration_seconds),native_minutes=len(x),A_prime75=row.clip_id in selected,Fold0_TRAIN60=row.clip_id in train,excluded_from_Aprime=row.clip_id not in selected,LF_fraction=float(p[:9].sum()/den),MF_fraction=float(p[9:20].sum()/den),HF_fraction=float(p[20:].sum()/den),broadband_temporal_cv=float(x.sum(1).std()/x.sum(1).mean()),band_cv_median=float(np.median(cv)),raw_power_sum=float(den),sha256=row.sha256))
 f=pd.DataFrame(rows);csv(out/'reference_509_all.csv',f)
 ds=xr.Dataset({'relative_band_power':(('clip','frequency_hz'),np.stack([m.mean(0)/m.mean(0).sum() for m in s.reference_minutes])),'native_minutes':('clip',f.native_minutes.values),'A_prime75':('clip',f.A_prime75.astype(int).values),'Fold0_TRAIN60':('clip',f.Fold0_TRAIN60.astype(int).values)},coords={'clip':f.clip_id.astype(str).values,'frequency_hz':FREQ},attrs={'power_units':'uncalibrated digital power; only within-clip normalized spectra compared','source':'VTUAD v1.3.2 admitted vessel archive, 509 byte-unique files'})
 ds.to_netcdf(out/'reference_509.xarray.nc',engine='scipy')
 summary=[];ks=[]
 for name,ix in [('all509',np.ones(len(f),bool)),('A_prime75',f.A_prime75),('Fold0_TRAIN60',f.Fold0_TRAIN60),('excluded434',~f.A_prime75),('not_in_fold0_449',~f.Fold0_TRAIN60)]:
  z=f[ix];summary.append({'group':name,'n':len(z),'total_hours':z.duration_seconds.sum()/3600,**{c+'_median':z[c].median() for c in ['native_minutes','LF_fraction','MF_fraction','HF_fraction','broadband_temporal_cv']}})
 for c in ['native_minutes','LF_fraction','MF_fraction','HF_fraction','broadband_temporal_cv']:
  for name,ix in [('A75_vs_excluded434',f.A_prime75),('TRAIN60_vs_other449',f.Fold0_TRAIN60)]:
   z=ks_2samp(f.loc[ix,c],f.loc[~ix,c]);ks.append(dict(comparison=name,feature=c,ks=float(z.statistic),pvalue=float(z.pvalue)))
 csv(out/'reference_group_summary.csv',summary);csv(out/'reference_group_KS.csv',ks)
 uses=[];detail=[]
 for regime in ['D0','H0']:
  d=role(0,regime,'train')
  for kind,key in [('background_window','base_window_index'),('background_group','background_group'),('positive_reference','reference_index')]:
   v=d[key];v=v[v>=0] if kind=='positive_reference' else v;uses.append(reuse_rows(v,regime,kind));cnt=pd.Series(v).value_counts()
   detail.extend(dict(dataset=regime,unit=kind,identity=str(k),uses=int(vv)) for k,vv in cnt.items())
  csv(out/(regime+'_site_counts.csv'),pd.Series(d['site']).value_counts().rename_axis('site').reset_index(name='rows'))
  csv(out/(regime+'_support_counts.csv'),pd.Series(d['active_minutes'][d['share']>0]).value_counts().sort_index().rename_axis('active_minutes').reset_index(name='rows'))
 csv(out/'reuse_summary.csv',uses);csv(out/'reuse_full_counts.csv',detail)
 # Exact R9 per-epoch scheduler is replayed without generating acoustic data.
 frame=s.role_metadata(0,'train');sch=make_schedule(len(frame),1,0,s.config['seed']);counts=np.bincount(sch['reference_index'][sch['reference_index']>=0],minlength=509)
 csv(out/'R9_epoch1_scheduled_reference_use.csv',[dict(reference_index=i,uses=int(n))for i,n in enumerate(counts)])
 recorded=[]
 for p in (ROOT/'ops/server_backups/r9_gpu_r1').glob('*/restored/fold_00/models/*/summary.json'):
  x=json.loads(p.read_text());recorded.append({'file':str(p.relative_to(ROOT)),'sha256':sha(p),'keys':list(x),'reference_509_mentions':'509' in json.dumps(x)})
 jsave(out/'R9_run_evidence.json',{'configured_shared_pool':509,'scheduler_epoch1_positive_requests':int(counts.sum()),'scheduled_unique_references':int((counts>0).sum()),'use_min':int(counts.min()),'use_max':int(counts.max()),'backup_summaries':recorded,'qualification':'scheduler replay and archived run code/summary evidence; no per-example R9 training trace was saved for independent forensic confirmation of every consumed batch'})
 epoch=[]
 for p in (ROOT/'models/r16_pretrained_cuda_bf16_r1/both').glob('*/*/*/summary.json'):
  x=json.loads(p.read_text());logs=[json.loads(z)for z in (p.parent/'epochs.jsonl').read_text().splitlines() if z.strip()]
  epoch.append({'family':p.parts[-4],'regime':p.parts[-3],'task':p.parts[-2],'best_epoch':x.get('best_epoch'),'epochs_run':x.get('epochs_run',len(logs)),'stage2_epochs_run':max(0,len(logs)-3),'best_after_warmup':max(0,x.get('best_epoch',0)-3),'summary_sha256':sha(p),'log_first':logs[0],'log_best':logs[x['best_epoch']-1],'log_last':logs[-1]})
 jsave(out/'epochs_full_evidence.json',epoch);csv(out/'epochs_summary.csv',[{k:v for k,v in z.items()if not k.startswith('log_')} for z in epoch])
 (out/'SOURCE_LINEAGE.md').write_text('# R17 원천 계보·재사용 전수 감사\n\n'+pd.DataFrame(summary).to_markdown(index=False)+'\n\n'+pd.DataFrame(uses).to_markdown(index=False)+'\n\nKish ESS는 재사용 빈도의 집중도 유효개수다. 시간 중첩·동일 선체·녹음일 의존성을 제거한 독립 표본 수가 아니다. D0/H0는 같은 배경·참조 스케줄을 공유하므로 2배 독립 자료로 세면 안 된다.\n\n509→119는 시간/선체/녹음일 연결 격리,119→75는 B19/C25 예약,75→60은 Fold0 참조15개 제외 결과다. 전체509의 전이적 연결 그래프는1개 성분이므로 strict독립성을 요구하면 전체를 동시에5개독립참조Fold로 나눌 수 없다. 이 선택은 다양성 감소와 독립평가 사이의 tradeoff이며, 필요성이나 원인 영향량이 이번 집계만으로 입증되지는 않는다.\n\nH0의 best epoch가3~7이라고 실제학습이3~7에서 중단된 것은 아니다. epochs_summary.csv의 실제10~22epoch와 구분한다. 초기최적점·train/validation 손실격차는 조기과적합과 양립하지만 lr=1e-3 또는 원천반복이 주원인이라는 인과결론은 LR/자료량 대조실험 없이 확정할 수 없다.\n\nVTUAD 파워는 비보정 디지털값이다. 장비를 넘는 절대 파워 비교 대신 clip내LF/MF/HF비율을 사용했다. 509개는선박archive이며 weak background22개도 포함한다. 별도ambient16개는 이집계에 포함되지 않는다.\n')
 print('SOURCE_DONE',len(f),flush=True)
if __name__=='__main__':main()
