"""Staged, frozen R16 NN prediction; never fits or selects a policy.

selection/local_fit can be inspected during development. local_gate,
local_report, domain_holdout and the official raw query require a global
completed-model and ensemble-policy receipt. Query labels/site metadata never
enter predict_completed_model. This module does not open query data on import.
"""
from pathlib import Path
import argparse,csv,gc,hashlib,json,os,sys,time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
KINDS=('bg_cnn','fdy','itransformer','tfgrid_fdy_conformer')
OPERATORS=('control_eq6','mfhf_shelf6','mfhf_shelf12','mfhf_shelf12_lfminus6','alpha_preEQ_inserted','independent_subband_snr')
FREQUENCIES=('31.6','39.8','50.1','63.1','79.4','100','125.9','158.5','199.5','251.2','316.2','398.1','501.2','631','794.3','1000','1258.9','1584.9','1995.3','2511.9','3162.3','3981.1','5011.9','6309.6','7943.3')
FEATURE_COLUMNS=tuple(f'm{m:02d}_TOL_{f}'for m in range(1,16)for f in FREQUENCIES)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def safe(p):
    p=Path(p);p=(ROOT/p).resolve()if not p.is_absolute()else p.resolve()
    if not p.is_relative_to(ROOT):raise ValueError('Project-contained path required')
    return p
def relative(p):return str(safe(p).relative_to(ROOT))
def raw_db(db):
    if isinstance(db,dict):raise TypeError('Raw db-only array required')
    a=np.asarray(db,np.float32)
    if a.ndim!=3 or a.shape[1:]!=(15,25)or not len(a)or np.isinf(a).any()or(np.isfinite(a).any(2).sum(1)<3).any():raise ValueError('Finite/NaN [N,15,25] with >=3 observed minutes required')
    return a
def read_raw_csv(path,expected_rows=1500):
    """Same exact minute-major public schema as inference_r5.read_query_csv."""
    before=sha(path)
    with Path(path).open(newline='',encoding='utf-8-sig')as f:
        r=csv.reader(f)
        if next(r,None)!=['ID',*FEATURE_COLUMNS]:raise ValueError('Exact ID+375 public feature columns required')
        ids=[];values=[]
        for row in r:
            if len(row)!=376 or not row[0]or row[0].strip()!=row[0]:raise ValueError('Malformed query ID/row')
            ids.append(row[0]);values.append([float(v)if v.strip()else np.nan for v in row[1:]])
    if len(ids)!=expected_rows or len(set(ids))!=len(ids):raise ValueError('Exact unique query row count required')
    db=raw_db(np.asarray(values,np.float32).reshape(-1,15,25))
    if sha(path)!=before:raise ValueError('Raw query changed during read')
    return np.asarray(ids,dtype=str),db,{'raw_query_sha256':before,'ID_order_preserved':True}
def guard_state():
    state={'allowed_query':None}
    def guard(event,args):
        if event=='open'and args and isinstance(args[0],(str,bytes,os.PathLike)):
            n=os.fsdecode(args[0]).lower().replace('\\','/')
            if '/eda_deep/'in n or '/r15_band_comparison_'in n:raise PermissionError('Query-derived EDA cannot be used by frozen prediction')
            if n.endswith('/test.csv'):
                allowed=state['allowed_query']
                if allowed is None or Path(os.fsdecode(args[0])).resolve()!=allowed:raise PermissionError('Raw query is locked until global model/policy freeze verification')
    return state,guard
def global_freeze(path,manifest,model_root):
    """Verify every declared model/policy byte before any locked input open."""
    if not path:raise PermissionError('Global model+policy freeze receipt required')
    path=safe(path);r=read(path)
    if (r.get('status')!='completed'or r.get('all_models_frozen')is not True
        or r.get('ensemble_policy_frozen')is not True or r.get('data_manifest_sha256')!=sha(manifest)
        or not r.get('artifact_sha256')):raise PermissionError('Global freeze certificate is incomplete')
    pins=r['artifact_sha256'];own=relative(__file__)
    if pins.get(own)!=sha(__file__):raise PermissionError('Frozen prediction script SHA is absent/stale')
    expected={k:relative(Path(model_root)/k)for k in KINDS}
    if r.get('nn_models')!=expected:raise PermissionError('Global receipt must bind all four complete R16 NN model folders')
    policy=r.get('ensemble_policy_path')
    if not policy or policy not in pins:raise PermissionError('Frozen ensemble policy artifact is absent')
    tree_path=r.get('tree_registry_path')
    if not tree_path or tree_path not in pins:raise PermissionError('Completed five-fold Tree registry is absent')
    for name,h in pins.items():
        if str(name).lower().endswith('/test.csv'):raise PermissionError('Raw query belongs in query_source_allowlist, not model artifact closure')
        if sha(safe(name))!=h:raise ValueError('Frozen artifact changed: '+str(name))
    trees=read(safe(tree_path))
    if (trees.get('status')!='completed'or trees.get('all_models_frozen')is not True
        or set(trees.get('models',{}))!=set(map(str,range(5)))):raise PermissionError('Five completed CPU Tree folds required')
    for name,h in trees.get('artifact_sha256',{}).items():
        if pins.get(name)!=h:raise PermissionError('Tree artifact omitted from global closure: '+str(name))
    from pipeline.training_r16_production import restore_completed_model
    for kind in KINDS:
        folder=safe(Path(model_root)/kind)
        for name in ('summary.json','best_regression.pt','best_classifier.pt'):
            if relative(folder/name)not in pins:raise PermissionError('Global model artifact closure is incomplete')
        m,_=restore_completed_model(folder,'cpu');del m;gc.collect()
    return r
def predict(stage,*,model_root,manifest,output,fold=0,kind=None,operator=None,freeze_receipt=None,query=None,batch=128,device='cpu'):
    from pipeline.training_r16_production import predict_completed_model,atomic_json,atomic_npz,metrics
    import torch
    if not 1<=batch<=128 or device not in ('cpu','mps'):raise ValueError('Bounded CPU/MPS inference required')
    if device=='mps'and(not torch.backends.mps.is_available()or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0'):raise RuntimeError('Explicit available MPS with fallback0 required')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    state,audit=guard_state();sys.addaudithook(audit)
    out=safe(output);mp=safe(manifest);model_root=safe(model_root)
    if out.exists():raise FileExistsError('Fresh frozen prediction folder required')
    selected=KINDS if kind is None else(kind,)
    if any(k not in KINDS for k in selected):raise ValueError('Unknown architecture')
    receipt=None
    if stage not in ('selection','local_fit'):
        receipt=global_freeze(freeze_receipt,mp,model_root)
    input_binding={'manifest':relative(mp),'manifest_sha256':sha(mp),'stage':stage,'fold':fold}
    if stage=='selection':
        if operator or query:raise ValueError('selection uses immutable production selection only')
        from pipeline.r16_production_data import load_production_role,validate_production_manifest
        input_binding['data_source_validation']=validate_production_manifest(mp,fold,verify_sources=True)
        panels=[('selection',load_production_role(mp,fold,'selection'))]
    elif stage in ('local_fit','local_gate','local_report','domain_holdout'):
        if query:raise ValueError('Fixed local panels cannot receive a query path')
        from ops.r16_forward_data import load_panel
        operators=OPERATORS if operator is None else(operator,)
        if any(x not in OPERATORS for x in operators):raise ValueError('Unknown fixed generator panel')
        panels=[(op,load_panel(mp,stage,op,freeze_receipt=safe(freeze_receipt)if freeze_receipt else None))for op in operators]
    elif stage=='query':
        if kind or operator or not query:raise ValueError('Final query requires all four frozen architectures and an approved raw CSV')
        qp=safe(query);allowed=receipt.get('query_source_allowlist',{})
        if relative(qp)not in allowed:raise PermissionError('This query source was not predeclared by the frozen policy receipt')
        state['allowed_query']=qp
        if sha(qp)!=allowed[relative(qp)]:raise ValueError('Predeclared raw query bytes differ')
        ids,db,qreceipt=read_raw_csv(qp);panels=[('query',{'sample_id':ids,'db':db})];input_binding.update(qreceipt)
    else:raise ValueError('Unknown explicit frozen prediction stage')
    out.mkdir(parents=True);began=time.perf_counter();pins={};results=[]
    for k in selected:
        folder=model_root/k
        for name in ('summary.json','best_regression.pt','best_classifier.pt'):pins[relative(folder/name)]=sha(folder/name)
    for panel,d in panels:
        db=raw_db(d['db']);ids=np.asarray(d['sample_id']).astype(str)
        if len(ids)!=len(db)or len(set(ids))!=len(ids):raise ValueError('Prediction IDs not uniquely aligned')
        for k in selected:
            pr=predict_completed_model(model_root/k,db,batch_size=batch,device=device)
            finite=all(np.isfinite(a).all()for a in pr.values())
            pool_gap=float(np.max(abs(pr['share']-(pr['cell_mask']*pr['weights']).sum((1,2)))))
            if (not finite or pool_gap>3e-6
                or any(((pr[n]<0)|(pr[n]>1)).any()for n in ('share','score','cell_mask','weights'))):raise ValueError('Frozen probability/pooling check failed')
            payload={'sample_id':ids,**pr};record={'panel':panel,'kind':k,'n':len(ids),'classification_score_independent':True}
            if stage!='query':
                payload.update(target_share=np.asarray(d['share']),target_presence=np.asarray(d['presence']))
                record['metrics']=metrics(d['share'],pr['share'],pr['score'])
            path=out/f'{panel}__{k}.npz';atomic_npz(path,**payload);record.update(path=path.name,sha256=sha(path));results.append(record)
            print(json.dumps({'event':'frozen_prediction_complete',**record}),flush=True)
    for name,h in pins.items():
        if sha(safe(name))!=h:raise ValueError('Checkpoint changed during frozen prediction')
    if receipt:global_freeze(freeze_receipt,mp,model_root)
    result={'status':'completed','stage':stage,'input_binding':input_binding,'model_artifact_sha256':pins,
        'inference_script_sha256':sha(__file__),'freeze_receipt_sha256':sha(safe(freeze_receipt))if freeze_receipt else None,
        'results':results,'elapsed_seconds':time.perf_counter()-began,'fit_or_calibration_or_policy_selection':False,
        'Test_read':stage=='query','official_upload':False,'raw_db_only_to_model':True,'device':device}
    atomic_json(out/'RESULTS.json',result);return result

def fixture(output):
    """Raw-array/schema and lock tests only; no completed full-model claim."""
    import tempfile
    from pipeline.models_r16_production import make_r16_model
    from pipeline.relative_grid_r6_v2 import relative_grid_v2
    from pipeline.training_r16_production import predict_features,atomic_json
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    rng=np.random.default_rng(2026102716);db=np.round(rng.normal(80,4,(8,15,25)),1).astype(np.float32);db[:2,0]=np.nan
    x,w=relative_grid_v2(raw_db(db));pr=predict_features(make_r16_model('bg_cnn').eval(),x,w,'cpu',4)
    pool=float(abs(pr[0]-(pr[2]*w).sum((1,2))).max())
    if pool>3e-6:raise ValueError('Synthetic array pooling failed')
    path=Path(tempfile.mkdtemp(prefix='r16-query-schema-',dir='/private/tmp'))/'synthetic.csv'
    with path.open('x',newline='')as f:
        writer=csv.writer(f);writer.writerow(['ID',*FEATURE_COLUMNS])
        for i,v in enumerate(db):writer.writerow([f'fixture{i}',*[''if np.isnan(a)else str(float(a))for a in v.ravel()]])
    ids,replay,_=read_raw_csv(path,expected_rows=8)
    if not np.array_equal(replay,db,equal_nan=True):raise ValueError('Synthetic schema roundtrip differs')
    try:global_freeze(None,ROOT/'unused.json',ROOT/'models/unused')
    except PermissionError:missing_gate_rejected=True
    else:raise AssertionError('Missing global freeze gate passed')
    result={'status':'passed','synthetic_only':True,'physical_pool_max_abs':pool,'CSV_schema_roundtrip_exact':True,
        'missing_global_receipt_rejected_before_input_open':missing_gate_rejected,'completed_R16_model_checkpoint_inference':'not_measured_until_training_completed',
        'official_raw_Test_opened':False,'production_weights_or_policy_selected':False,'script_sha256':sha(__file__)}
    output=safe(output)
    if output.exists():raise FileExistsError('Fresh fixture report required')
    atomic_json(output,result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['fixture','selection','local_fit','local_gate','local_report','domain_holdout','query'])
    p.add_argument('--model-root');p.add_argument('--manifest');p.add_argument('--fold',type=int,default=0);p.add_argument('--output',required=True)
    p.add_argument('--kind',choices=KINDS);p.add_argument('--operator',choices=OPERATORS);p.add_argument('--freeze-receipt');p.add_argument('--query');p.add_argument('--batch',type=int,default=128);p.add_argument('--device',choices=['cpu','mps'],default='cpu');a=p.parse_args()
    if a.stage=='fixture':r=fixture(a.output)
    else:
        if not a.model_root or not a.manifest:p.error('--model-root and --manifest required')
        r=predict(a.stage,model_root=a.model_root,manifest=a.manifest,output=a.output,fold=a.fold,kind=a.kind,operator=a.operator,freeze_receipt=a.freeze_receipt,query=a.query,batch=a.batch,device=a.device)
    print(json.dumps({'status':'completed','result':r}))
if __name__=='__main__':main()
