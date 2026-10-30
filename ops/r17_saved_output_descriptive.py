"""No fit: descriptive statistics of saved outputs on unlabeled Test IDs."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ops.r17_audit_common import *
def main():
 q=pd.read_csv(OUT/'task1/rows_Test_descriptive_only.csv');models=['R9','lightgbm5','A_original','catboost5','dinov2_small','dasheng_base','timemoe'];keys=['R9_share']+['share_'+m for m in models[1:]];rows=[]
 for n,k in zip(models,keys):
  v=q[k].values;delta=v-q.R9_share.values;rows.append(dict(model=n,mean=v.mean(),rms=np.sqrt((v*v).mean()),median=np.median(v),p10=np.quantile(v,.1),p90=np.quantile(v,.9),delta_R9_mean=delta.mean(),delta_R9_RMS=np.sqrt((delta*delta).mean())))
 f=pd.DataFrame(rows);csv(OUT/'eda/SAVED_OUTPUT_DESCRIPTIVE.csv',f);c=q[keys].corr();c.to_csv(OUT/'eda/SAVED_OUTPUT_PEARSON.csv');s=q[keys].corr(method='spearman');s.to_csv(OUT/'eda/SAVED_OUTPUT_SPEARMAN.csv')
 page='<meta charset="utf-8"><title>저장 출력 기술 통계</title><style>body{font:16px system-ui;max-width:1300px;margin:30px auto}td,th{padding:9px;border-bottom:1px solid #ddd}table{border-collapse:collapse}h2{margin-top:30px}</style><h1>R9 / R16 저장 출력 비교</h1><p>Test 1,500행에 대한 비학습 기술 통계입니다. 정답·LB예측·자료집단 보정·가중치선택에 사용하지 않았습니다.</p>'+f.to_html(index=False,float_format=lambda x:f'{x:.6f}')+'<h2>Pearson</h2>'+c.to_html(float_format=lambda x:f'{x:.4f}')+'<h2>Spearman</h2>'+s.to_html(float_format=lambda x:f'{x:.4f}')
 (OUT/'eda/SAVED_OUTPUT_DESCRIPTIVE.html').write_text(page)
if __name__=='__main__':main()
