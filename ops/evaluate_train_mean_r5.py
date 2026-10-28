"""Our TRAIN-mean constant comparator, explicitly NOT the official starter.

One scalar is fitted on each fold TRAIN only. db/mask row-count prediction is
scored on the same TRAIN and INNER. No OUTER/Test, model selection, GPU or SSH.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,os,sys,time
import numpy as np
import pandas as pd
from pipeline.fold_data_r5 import FoldDataR5
from pipeline.grid_inputs_r5 import canonical_sha256,file_sha256
from ops.evaluate_official_starter_r5 import array_sha
ROOT=Path(__file__).resolve().parents[1]

def score(y,p,c):
    e=p-y;mse=float(np.mean(e*e));check=float(np.var(y)+(y.mean()-c)**2)
    if abs(mse-check)>1e-12:raise AssertionError('Constant moment identity failed')
    return {'n':len(y),'rmse':float(np.sqrt(mse)),'mae':float(np.mean(np.abs(e))),
        'bias_pred_minus_truth':float(e.mean()),'target_mean':float(y.mean()),'prediction_mean':float(p.mean()),
        'moment_MSE_residual':abs(mse-check)}

def predict(db,mask,fitted_train_mean):
    if db.shape!=mask.shape or not np.array_equal(mask,np.isfinite(db)):raise ValueError('db/mask alignment')
    return np.full(len(db),fitted_train_mean,dtype='float64')

def run(output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Fresh diagnostic required')
    data=ROOT/'data/synthetic/group_cv_passage_r4_v1';allowed=set();source={}
    for f in range(5):
        for n in ['readiness_manifest.json','sources/manifest.json','train.npz','validation.npz','train_sample_metadata.parquet','inner_validation_metadata.parquet']:
            p=data/f'fold_{f:02d}'/n;allowed.add(p.resolve());source[p.relative_to(ROOT).as_posix()]=file_sha256(p)
    data_root=(ROOT/'data').resolve()
    def check(p):
        q=Path(p).resolve()
        if q.is_relative_to(data_root) and q not in allowed:raise ValueError('Only TRAIN/INNER files allowed')
        return q
    def audit(event,a):
        if event in {'socket.connect','socket.getaddrinfo','subprocess.Popen','os.system'}:raise RuntimeError('External action forbidden')
        if event=='open' and isinstance(a[0],(str,bytes,os.PathLike)):
            q=check(os.fsdecode(a[0]))
            if q.is_relative_to(data_root) and a[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):raise ValueError('No data writes')
    sys.addaudithook(audit);nl,pr=np.load,pd.read_parquet
    def load(p,*a,**k):check(p);return nl(p,*a,**k)
    def parquet(p,*a,**k):check(p);return pr(p,*a,**k)
    np.load,pd.read_parquet=load,parquet
    output.mkdir(parents=True);rows=[];pooled={r:{'y':[],'p':[]} for r in ['train','inner']};begin=time.perf_counter()
    for f in range(5):
        fold=FoldDataR5(data/f'fold_{f:02d}',42);train,tm=fold.load('train')
        fitted=float(train['share'].astype(float).mean())
        for role,a,md in [('train',train,tm),('inner',*fold.load('inner'))]:
            y=a['share'].astype(float);p=predict(a['db'],np.isfinite(a['db']),fitted)
            m=score(y,p,fitted);ids=md.sample_id.astype(str).to_numpy(dtype=str)
            filename=f'fold_{f:02d}_{role}_predictions.npz';np.savez_compressed(output/filename,share=p,score=np.full(len(p),.5),sample_id=ids)
            row={'fold':f,'split':role,'fitted_constant':fitted,'fit_split':'train','training_binding':fold.binding,
                'metrics':m,'sample_id_order_sha256':canonical_sha256(ids.tolist()),'target_share_sha256':array_sha(a['share']),
                'prediction_file':filename,'prediction_file_sha256':file_sha256(output/filename),
                'score_column_note':'Uninformative fixed .5; only share scalar was fitted.',
                'by_group':{col:{str(k):score(y[g.index.to_numpy()],p[g.index.to_numpy()],fitted) for k,g in md.groupby(col,dropna=False)} for col in ['site','reference_group']}}
            rows.append(row);pooled[role]['y'].append(y);pooled[role]['p'].append(p)
        print(json.dumps({'fold':f,'TRAIN_fitted_mean':fitted,'INNER_RMSE':rows[-1]['metrics']['rmse']}),flush=True)
        if fold.opened_roles!=['train','inner']:raise AssertionError('Only TRAIN/INNER opened')
    for n,h in source.items():
        if file_sha256(ROOT/n)!=h:raise ValueError('Frozen input changed')
    # Per-fold constants differ: pooled metrics use explicit predictions, not a single global fit.
    totals={}
    for role,v in pooled.items():
        y=np.concatenate(v['y']);p=np.concatenate(v['p']);e=p-y
        totals[role]={'n':len(y),'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(np.mean(np.abs(e))),
            'bias_pred_minus_truth':float(e.mean()),'target_mean':float(y.mean()),'prediction_mean':float(p.mean())}
    result={'status':'completed our TRAIN-mean comparator','official_baseline':False,'diagnostic_only':True,
        'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'folds':rows,'pooled':totals,
        'fit_scope':'One scalar per fold, fitted only on that fold TRAIN. No INNER fitting or global mean fit.',
        'prediction_clipping_applied':False,'real_R4_R5_data_or_core_modified':False,
        'OUTER_Test_audio_SSH_GPU_external_transmission_used':False,'cross_fold_sources_are_not_independent_rows':True,
        'source_files_sha256':source,'script_sha256':file_sha256(__file__),
        'official_comparator_metrics_sha256':file_sha256(ROOT/'reports/R5_OFFICIAL_STARTER_TRAIN_INNER/metrics.json'),
        'environment':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'system':os.uname().sysname},
        'elapsed_seconds':time.perf_counter()-begin}
    with (output/'metrics.json').open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    lines=['# 우리 TRAIN 평균 상수 기준선 — 공식 starter 아님','',
        '각 fold의 TRAIN 정답 평균 하나만 구해 고정했습니다. INNER는 평균 추정에 사용하지 않았습니다. 예측 clipping은 없습니다.','',
        '| Fold | TRAIN에서 fit한 평균 | TRAIN RMSE | INNER RMSE | INNER MAE | INNER bias |',
        '|---|---:|---:|---:|---:|---:|']
    for f in range(5):
        t=rows[2*f];v=rows[2*f+1];m=v['metrics']
        lines.append(f'| {f} | {t["fitted_constant"]:.9f} | {t["metrics"]["rmse"]:.9f} | {m["rmse"]:.9f} | {m["mae"]:.9f} | {m["bias_pred_minus_truth"]:.9f} |')
    lines+=['',f'전체 INNER pooled RMSE: **{totals["inner"]["rmse"]:.9f}**.',
        '', '[전체 지표·source SHA](metrics.json) · [공식 starter 별도 진단](../R5_OFFICIAL_STARTER_TRAIN_INNER/README.md)',
        '', '이 평균 기준선과 공식 starter의 0.1 고정값은 서로 다른 알고리즘입니다. 두 수치는 내부 검증 결과이며 공식 점수나 OUTER OOF가 아닙니다.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'report':str(output/'metrics.json'),'pooled':totals,'elapsed_seconds':result['elapsed_seconds']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=ROOT/'reports/R5_TRAIN_MEAN_BASELINE_TRAIN_INNER');run(p.parse_args().output_dir)
