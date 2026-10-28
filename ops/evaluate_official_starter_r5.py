"""Official fixed starter constants on frozen R5 TRAIN/INNER, CPU diagnostic.

The notebook is inspected, never executed (other cells perform Test IO).
Only its exact cell6 assignments, share=.1 and score=.5, are applied. No fit,
input rounding, clipping, algorithm repair, OUTER, Test, audio, SSH or GPU.
"""
from pathlib import Path
import argparse,ast,datetime,hashlib,importlib.metadata,json,os,sys,time
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK_SHA='08242f4771c38de43ab9022d460b537b5eaa90a687bb1ddb532061b512dd6b83'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def array_sha(a):
    import numpy as np
    h=hashlib.sha256(json.dumps({'shape':list(a.shape),'dtype':str(a.dtype)},sort_keys=True,separators=(',',':')).encode());h.update(np.ascontiguousarray(a).tobytes());return h.hexdigest()
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')

def predict_official_starter(db,mask):
    """db/mask-only row-count adapter; no labels or source identity accepted."""
    import numpy as np
    if db.ndim!=3 or db.shape[1:]!=(15,25) or mask.shape!=db.shape or not np.array_equal(mask,np.isfinite(db)):
        raise ValueError('Expected aligned R4 [N,15,25] db/mask')
    return {'share':np.full(len(db),.1,dtype='float64'),'score':np.full(len(db),.5,dtype='float64')}

def metrics(y,p):
    import numpy as np
    if not len(y):return {'n':0,'rmse':None,'mae':None,'bias_pred_minus_truth':None,'target_mean':None,'prediction_mean':None}
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float);e=p-y
    mse=float(np.mean(e*e));closed=float(np.var(y)+(float(y.mean())-.1)**2)
    if abs(mse-closed)>1e-12:raise AssertionError('Constant-baseline independent moment identity failed')
    return {'n':len(y),'rmse':float(np.sqrt(mse)),'mae':float(np.mean(np.abs(e))),
        'bias_pred_minus_truth':float(e.mean()),'target_mean':float(y.mean()),'prediction_mean':float(p.mean()),
        'target_min':float(y.min()),'target_max':float(y.max()),'independent_moment_MSE_residual':abs(mse-closed)}

def run(output):
    import numpy as np,pandas as pd
    from pipeline.fold_data_r5 import FoldDataR5
    from pipeline.runner_r5 import CODE_NAMES
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Fresh diagnostic output directory required')
    notebook=ROOT/'data/raw/maritime_starter_notebook.ipynb'
    if sha(notebook)!=NOTEBOOK_SHA:raise ValueError('Official notebook source SHA differs')
    nb=json.loads(notebook.read_text());source=''.join(nb['cells'][6]['source'])
    expected='submission = sample.copy()\nsubmission["share_added_vessel"] = 0.1\nsubmission["score_added_vessel"] = 0.5\n'
    if ast.dump(ast.parse(source))!=ast.dump(ast.parse(expected)):raise ValueError('Official starter cell6 algorithm differs')
    data=ROOT/'data/synthetic/group_cv_passage_r4_v1'
    allowed={notebook.resolve()};inputs={};code={p:sha(ROOT/p) for p in CODE_NAMES+['pipeline/train_lgbm.py','pipeline/prototypes.py']}
    code['ops/evaluate_official_starter_r5.py']=sha(__file__)
    for f in range(5):
        for n in ['readiness_manifest.json','sources/manifest.json','train.npz','validation.npz','train_sample_metadata.parquet','inner_validation_metadata.parquet']:
            p=data/f'fold_{f:02d}'/n;allowed.add(p.resolve());inputs[p.relative_to(ROOT).as_posix()]=sha(p)
    root_data=(ROOT/'data').resolve()
    def check(p):
        q=Path(p).resolve()
        if q.is_relative_to(root_data) and q not in allowed:raise ValueError('Non-TRAIN/INNER data IO blocked: '+str(q))
        if q.suffix.lower() in {'.wav','.flac','.mp3'}:raise ValueError('Audio not used by official constant starter')
        return q
    def audit(event,a):
        if event in {'socket.connect','socket.getaddrinfo','subprocess.Popen','os.system'}:raise RuntimeError('External actions forbidden in CPU baseline diagnostic')
        if event=='open' and isinstance(a[0],(str,bytes,os.PathLike)):
            q=check(os.fsdecode(a[0]))
            if q.is_relative_to(root_data) and a[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):raise ValueError('Source data writes forbidden')
    sys.addaudithook(audit)
    old_load,old_parquet=np.load,pd.read_parquet
    def load(p,*a,**k):check(p);return old_load(p,*a,**k)
    def parquet(p,*a,**k):check(p);return old_parquet(p,*a,**k)
    np.load,pd.read_parquet=load,parquet
    output.mkdir(parents=True);begin=time.perf_counter();rows=[];pooled={'train':[],'inner':[]};pred_files={}
    for f in range(5):
        fold=FoldDataR5(data/f'fold_{f:02d}',42)
        for role in ['train','inner']:
            a,md=fold.load(role);pred=predict_official_starter(a['db'],np.isfinite(a['db']))
            y=a['share'].astype('float64');p=pred['share'];overall=metrics(y,p)
            by_group={}
            for field in ['site','reference_group','background_group']:
                g=md[field].fillna('<missing>').astype(str).to_numpy()
                by_group[field]={key:metrics(y[g==key],p[g==key]) for key in sorted(set(g))}
            bins={'y=0':y==0,'0<y<=.1':(y>0)&(y<=.1),'.1<y<=.25':(y>.1)&(y<=.25),
                '.25<y<=.5':(y>.25)&(y<=.5),'.5<y<=.75':(y>.5)&(y<=.75),
                '.75<y<=.95':(y>.75)&(y<=.95),'.95<y<=1':(y>.95)&(y<=1)}
            if sum(int(b.sum()) for b in bins.values())!=len(y):raise ValueError('Target bins do not cover valid target range')
            ids=md.sample_id.astype(str).to_numpy(dtype=str);name=f'fold_{f:02d}_{role}_predictions.npz'
            np.savez_compressed(output/name,share=pred['share'],score=pred['score'],sample_id=ids)
            pred_files[name]=sha(output/name)
            row={'fold':f,'split':role,'metrics':overall,'by_target_bin':{k:metrics(y[b],p[b]) for k,b in bins.items()},
                'by_group':by_group,'training_binding':fold.binding,'query_db_shape':list(a['db'].shape),'db_dtype':str(a['db'].dtype),
                'observed_cells':int(np.isfinite(a['db']).sum()),'sample_id_order_sha256':canonical(ids.tolist()),
                'target_share_sha256':array_sha(a['share']),'raw_db_sha256':array_sha(a['db']),
                'prediction_file':name,'prediction_file_sha256':pred_files[name],
                'prediction_dtype':str(p.dtype),'prediction_clipping_applied':False,'fitting_performed':False}
            rows.append(row);pooled[role].append(y)
            write(output/f'fold_{f:02d}_{role}_metrics.json',row)
            print(json.dumps({'fold':f,'split':role,**overall}),flush=True)
            del a,md,pred,y,p
        if fold.opened_roles!=['train','inner']:raise AssertionError('Unexpected role opened')
    for p,h in inputs.items():
        if sha(ROOT/p)!=h:raise ValueError('Frozen data changed during diagnostic')
    for p,h in code.items():
        if sha(ROOT/p)!=h:raise ValueError('Code changed during diagnostic')
    if sha(notebook)!=NOTEBOOK_SHA:raise ValueError('Official source changed during diagnostic')
    source_manifest={'official_notebook':'data/raw/maritime_starter_notebook.ipynb','official_notebook_sha256':NOTEBOOK_SHA,
        'official_source_page':'https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge/data',
        'cell_index_zero_based':6,'exact_cell_source':source,'whole_notebook_executed':False,
        'schema_adapter':'R4 db/mask validates query shape and supplies N. Original constants are repeated in metadata row order; original SampleSubmission/Test IO is not executed.',
        'algorithm_modified_or_repaired':False,'fit_required_or_performed':False,
        'share_constant':.1,'score_constant':.5,'prediction_clipping_applied':False,
        'code_sha256':code,'input_files_sha256':inputs,'prediction_files_sha256':pred_files,
        'all_inputs_and_core19_unchanged_after_evaluation':True}
    write(output/'source_manifest.json',source_manifest)
    summary={'status':'completed official starter TRAIN/INNER diagnostic','diagnostic_only':True,
        'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'folds':rows,
        'pooled':{role:metrics(np.concatenate(ys),np.full(sum(len(v) for v in ys),.1)) for role,ys in pooled.items()},
        'train_crossfold_rows_are_repeated_sources':True,'metrics_are_INNER_not_OUTER_OOF_or_official_scores':True,
        'official_public_Benchmark_identical_to_starter_algorithm_confirmed':False,
        'Test_OUTER_calibration_sealed_audio_SSH_GPU_or_external_transmission_used':False,
        'original_R4_R5_pipeline_or_GPU_run_modified':False,'source_manifest_sha256':sha(output/'source_manifest.json'),
        'environment':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,
            'pyarrow':importlib.metadata.version('pyarrow'),'OS':os.uname().sysname,'machine':os.uname().machine,
            'logical_CPUs':os.cpu_count()},'elapsed_seconds':time.perf_counter()-begin}
    write(output/'metrics.json',summary)
    lines=['# 官方 starter 상수 방법: 현재 TRAIN / INNER 진단','',
        '공식 notebook cell 6의 share=0.1, score=0.5를 수정 없이 평가했습니다. 학습·예측 clipping은 없습니다. 전체 notebook의 Test IO는 실행하지 않았습니다.','',
        '| Fold | TRAIN 행 | TRAIN RMSE | INNER 행 | INNER RMSE | INNER MAE | INNER bias(pred−truth) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for f in range(5):
        t=rows[2*f]['metrics'];v=rows[2*f+1]['metrics']
        lines.append(f'| {f} | {t["n"]:,} | {t["rmse"]:.9f} | {v["n"]:,} | {v["rmse"]:.9f} | {v["mae"]:.9f} | {v["bias_pred_minus_truth"]:.9f} |')
    lines+=['',f'전체 INNER {summary["pooled"]["inner"]["n"]:,}행의 pooled RMSE는 **{summary["pooled"]["inner"]["rmse"]:.9f}**입니다.',
        '', 'INNER는 신경망 early stopping·전처리 선택에 쓰이는 내부 검증입니다. 이 값은 OUTER OOF나 공식 점수가 아닙니다. 공개 Benchmark가 이 상수 starter와 동일 알고리즘이라는 공식 매핑은 확인되지 않았습니다.',
        '', 'TRAIN 5fold 합계는 동일 원천이 반복된 합성 행을 포함합니다. 독립 표본 수나 신뢰구간으로 해석하지 않습니다.',
        '', '[전체 지표·target bin·site/reference/background group](metrics.json) · [공식 source SHA·코드·입력·예측 manifest](source_manifest.json)',
        '', '공식 starter에는 별도 학습 모델이 없습니다. waveform→band-power helper는 이번 상수 예측의 regression 모델이 아니므로 실행하거나 교체하지 않았습니다.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':summary['status'],'pooled':summary['pooled'],'report':str(output/'metrics.json'),'elapsed_seconds':summary['elapsed_seconds']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=ROOT/'reports/R5_OFFICIAL_STARTER_TRAIN_INNER')
    run(p.parse_args().output_dir)
