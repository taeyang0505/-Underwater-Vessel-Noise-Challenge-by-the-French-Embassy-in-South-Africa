"""R19 law-transfer diagnostics: frozen CPU inference, never fitting/training.

Torch and LightGBM phases intentionally run in different processes. All seven
input panels come from the independent paired forward-panel builder. This
script never reads challenge Test/query predictions or original Test.csv.
"""
from __future__ import annotations
import argparse, datetime as dt, gc, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT = ROOT / 'reports/r19_multislice_law_transfer_r1/inference'
PANELS = ROOT / 'reports/r19_multislice_law_transfer_r1/law_panels'
LAWS = ('control_eq6','mfhf_shelf6','mfhf_shelf12','mfhf_shelf12_lfminus6',
        'alpha_preEQ_inserted','independent_subband_snr','vanilla_addition')
FAMILIES = ('lightgbm','dinov2_small','dasheng_base')
REGIMES = ('D0','H0')
TASKS = ('regression','classification')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()

def now(): return dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(timespec='seconds')
def read(p): return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
    t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
    with t.open('wb') as f: np.savez_compressed(f,**v)
    t.replace(p)
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def forbid_test(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes)):
        p=str(args[0]).lower()
        if Path(p).name in ('test.csv','sample_submission.csv','query.npz','query_predictions.npz'):
            raise PermissionError('Challenge query/Test is outside R19 local diagnostic scope')

def source_pins():
    names=('ops/r19_frozen_law_inference.py','pipeline/r18_production_finalization.py',
           'pipeline/training_r18_production_neural.py','pipeline/training_r16_pretrained_cuda_bf16.py',
           'pipeline/pretrained_audio2d_r16.py','pipeline/relative_grid_r6_v2.py',
           'pipeline/relative_grid_r6.py','ops/r16_temporal_features.py',
           'pipeline/trees_r18_production.py','pipeline/training_r16_multisupport.py',
           'pipeline/trees_r16_production.py')
    return {n:sha(ROOT/n) for n in names}

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    pins={}; parts=[]; metadata=[]
    for law in LAWS:
        p=PANELS/(law+'.npz')
        if not p.exists(): raise FileNotFoundError(p)
        pins[str(p.relative_to(ROOT))]=sha(p);d=load(p)
        n=len(d['db'])
        if d['db'].shape!=(n,15,25) or np.isinf(d['db']).any():raise ValueError('Bad geometry')
        if not np.array_equal(d['share']>0,d['presence']>0):raise ValueError('Bad presence')
        if len(set(d['sample_id'].astype(str)))!=n:raise ValueError('Duplicate panel IDs')
        parts.append(d['db'].astype(np.float32));metadata.append({'law':law,'n':n,'path':str(p.relative_to(ROOT))})
    db=np.concatenate(parts); unique=[];inverse=[];seen={}
    for row in db:
        # Only exact observation repeats are reused (e.g. identical no-addition
        # negatives across laws). No nearest-neighbour or tolerance matching.
        key=np.nan_to_num(row,nan=np.float32(-1e20)).tobytes()
        j=seen.get(key)
        if j is None:j=len(unique);seen[key]=j;unique.append(row)
        inverse.append(j)
    unique=np.stack(unique);inverse=np.asarray(inverse,np.int64)
    assert np.array_equal(unique[inverse],db,equal_nan=True)
    protocol={'status':'registered_before_law_model_predictions','created_KST':now(),
        'laws':metadata,'input_sha256':pins,'source_sha256':source_pins(),
        'rows_total':len(db),'exact_unique_db_rows':len(unique),'exact_repeats_reused':len(db)-len(unique),
        'device':'cpu','precision':'FP32; no autocast','torch_threads':10,'microbatch':32,
        'new_training':False,'optimizer_steps':0,'Test_or_query_read':False,
        'model_selection':False,'calibration_or_policy_fit':False,
        'old_B1_weights':{'lightgbm':.60,'dinov2_small':.25,'dasheng_base':.15},
        'new_B2_weights':{'lightgbm':.50,'dinov2_small':.30,'dasheng_base':.20},
        'regime_weights':{'D0':.5,'H0':.5},'score':'Independent classifier probability, same fixed weights; no share mixing',
        'slices':['all','y=0','0<y<=0.05','0.05<y<=0.15','y>0.15','0<y<=0.10','active_minutes>=12 & y>0'],
        'metrics':'RMSE/bias and class AUC; positive-only slices add zero negatives only for diagnostic AUC',
        'limitations':['Custom/vanilla panels have paired geometry but different realized share distributions.',
                      'CPU FP32 may differ slightly from historical CUDA BF16 predictions; parity probe recorded.',
                      'Panels are diagnostics on local source observations, not unseen official evaluation.']}
    p=OUT/'PROTOCOL.json'
    if p.exists():
        old=read(p)
        for k in ('input_sha256','source_sha256','rows_total','exact_unique_db_rows'):
            if old[k]!=protocol[k]:raise ValueError('Immutable protocol changed: '+k)
    else:write(p,protocol)
    save(OUT/'input_index.npz',{'inverse':inverse})
    return unique,inverse,metadata

def verify_protocol():
    p=read(OUT/'PROTOCOL.json')
    for n,h in {**p['input_sha256'],**p['source_sha256']}.items():
        if sha(ROOT/n)!=h:raise ValueError('Pinned source changed: '+n)
    return p

def branch_path(generation,family,regime,task):
    base='models/r16_pretrained_cuda_bf16_r1/both' if generation=='old' else 'models/r18_production_r1/neural'
    return ROOT/base/family/regime/task

def neural():
    import torch
    from pipeline import training_r18_production_neural as newapi
    from pipeline import training_r16_pretrained_cuda_bf16 as oldapi
    torch.set_num_threads(10);torch.set_num_interop_threads(1)
    d,inverse,metadata=prepare();verify_protocol()
    feature_start=time.perf_counter();x,w=newapi.feature17(d)
    x=torch.from_numpy(x.copy());w=torch.from_numpy(w.copy())
    xold=x[:,oldapi.CLASSIFIER_CHANNELS].contiguous()
    write(OUT/'FEATURES_RAM.json',{'rows':len(d),'seconds':time.perf_counter()-feature_start,
          'RAM_bytes':x.numel()*4+w.numel()*4+xold.numel()*4,'disk_feature_cache':False})
    with np.load(ROOT/'reports/r18_production_r1/finalization/inputs/local_gate.npz') as z:
        probe_db=z['db'][:32];probe_ids=z['sample_id'][:32]
    probe_x,probe_w=newapi.feature17(probe_db)
    probe_x=torch.from_numpy(probe_x.copy());probe_w=torch.from_numpy(probe_w.copy())
    completed=[];start_all=time.perf_counter()
    for family in FAMILIES[1:]:
      for generation in ('old','new'):
        result={};pins={};parity=[]
        target=OUT/f'{generation}_{family}.npz'
        if target.exists():
            receipt=read(target.with_suffix('.json'))
            if receipt['prediction_sha256']!=sha(target):raise ValueError('Saved output hash changed')
            completed.append(target.name);continue
        for regime in REGIMES:
          for task in TASKS:
            folder=branch_path(generation,family,regime,task);cp=folder/'best.pt';summary=read(folder/'summary.json')
            expected=summary['artifact_sha256']['best.pt'] if generation=='old' else summary['artifacts']['best.pt']
            if sha(cp)!=expected:raise ValueError('Model pin differs')
            pins[str(cp.relative_to(ROOT))]=expected;pins[str((folder/'summary.json').relative_to(ROOT))]=sha(folder/'summary.json')
            ck=torch.load(cp,map_location='cpu',weights_only=True)
            if (ck['family'],ck['regime'],ck['task'])!=(family,regime,task):raise ValueError('Model job differs')
            model,_=(oldapi.branch if generation=='old' else newapi.branch)(family,task,'cpu')
            fingerprint=oldapi.base_fingerprint if generation=='old' else newapi.fingerprint
            expected_base=ck['permanent_base_fingerprint'] if generation=='old' else ck['frozen_base_sha256']
            if fingerprint(model)!=expected_base:raise ValueError('Frozen public weights differ')
            model.load_adapter_state(ck['adapter_state']);model.eval()
            if any(p.device.type!='cpu' for p in model.parameters()):raise ValueError('Non-CPU model')
            xx=xold if generation=='old' and task=='classification' else x
            probe=probe_x[:,oldapi.CLASSIFIER_CHANNELS] if generation=='old' and task=='classification' else probe_x
            with torch.inference_mode():
                probe_p=model(probe).float().sigmoid()
                if task=='regression':probe_p=(probe_p*probe_w).sum((1,2))
                reference=load(ROOT/f'reports/r18_production_r1/finalization/predictions/{generation}/{family}/local_gate.npz')
                if not np.array_equal(reference['sample_id'][:32].astype(str),probe_ids.astype(str)):raise ValueError('Probe order')
                key=('share_'if task=='regression'else'prob_')+regime
                err=probe_p.numpy()-reference[key][:32]
                parity.append({'regime':regime,'task':task,'n':32,'max_abs':float(abs(err).max()),'mean_abs':float(abs(err).mean()),
                               'reason':'Historical CUDA BF16 vs diagnostic CPU FP32; exact weights/features unchanged'})
                predicted=[];start=time.perf_counter()
                for off in range(0,len(d),32):
                    logits=model(xx[off:off+32]);p=logits.float().sigmoid()
                    if task=='regression':p=(p*w[off:off+32]).sum((1,2))
                    predicted.append(p.numpy())
                    if off%512==0:
                        elapsed=time.perf_counter()-start
                        progress={'phase':'frozen_CPU_neural_inference','family':family,'generation':generation,'regime':regime,'task':task,
                            'unique_rows_completed':min(off+32,len(d)),'unique_rows_total':len(d),'branch_elapsed_sec':elapsed,
                            'all_elapsed_sec':time.perf_counter()-start_all,'completed_outputs':completed,'updated_KST':now()}
                        write(OUT/'progress.json',progress);print(json.dumps(progress,ensure_ascii=False),flush=True)
                pred=np.concatenate(predicted)
            if not np.isfinite(pred).all() or np.any((pred<0)|(pred>1)):raise ValueError('Prediction range')
            result[key]=pred[inverse]
            if fingerprint(model)!=expected_base:raise ValueError('Frozen base changed during inference')
            del model,ck;gc.collect()
        save(target,result)
        write(target.with_suffix('.json'),{'status':'completed','device':'cpu','precision':'float32','n':len(inverse),
            'model_sha256':pins,'prediction_sha256':sha(target),'CPU_vs_historical_precision_probe':parity,
            'optimizer_steps':0,'feature_masking_at_inference':False,'created_KST':now()})
        completed.append(target.name)
    verify_protocol();write(OUT/'NEURAL_COMPLETED.json',{'status':'completed','outputs':completed,'elapsed_sec':time.perf_counter()-start_all,'created_KST':now()})

def trees():
    import lightgbm as lgb
    from pipeline.training_r16_multisupport import tree_features
    from pipeline.trees_r18_production import transform
    d,inverse,metadata=prepare();verify_protocol();t=time.perf_counter()
    if 'torch' in sys.modules:raise RuntimeError('Torch and LightGBM must use separate processes')
    old={};new={};oldpins={};newpins={}
    legacy_expected=read(ROOT/'reports/r18_production_r1/finalization/PLAN.json')['legacy_source_sha256']
    for task in TASKS:
        pieces=[]
        for off in range(0,len(d),256):pieces.append(tree_features(d[off:off+256],task))
        x=np.concatenate(pieces);del pieces
        for regime in REGIMES:
            foldpred=[]
            for fold in range(5):
                p=ROOT/f'models/r16_multisupport_full_r1/trees/{regime}/fold_{fold:02d}/{task}.txt'
                expected=legacy_expected[str(p.relative_to(ROOT))]
                if sha(p)!=expected:raise ValueError('Old tree model changed')
                oldpins[str(p.relative_to(ROOT))]=expected
                model=lgb.Booster(model_file=str(p));foldpred.append(np.clip(model.predict(x,num_threads=2),0,1));del model
            key=('share_'if task=='regression'else'prob_')+regime
            old[key]=np.mean(foldpred,axis=0)[inverse]
        del x;gc.collect()
    for task in TASKS:
        branch='relative' if task=='regression' else 'classification'
        pieces=[transform(d[off:off+256],branch,batch_size=256) for off in range(0,len(d),256)]
        x=np.concatenate(pieces);del pieces
        for regime in REGIMES:
            folder=ROOT/f'models/r18_production_r1/trees/{regime}';summary=read(folder/'summary.json')
            if summary['selected_regression']!='relative':raise ValueError('Expected immutable relative regressor')
            p=folder/(branch+'.txt');expected=summary['artifact_sha256'][p.name]
            if sha(p)!=expected:raise ValueError('New tree model changed')
            newpins[str(p.relative_to(ROOT))]=expected
            model=lgb.Booster(model_file=str(p));pred=np.clip(model.predict(x,num_threads=2),0,1).astype(np.float32)
            new[('share_'if task=='regression'else'prob_')+regime]=pred[inverse]
            del model
        del x;gc.collect()
    for generation,values,pins in [('old',old,oldpins),('new',new,newpins)]:
        target=OUT/f'{generation}_lightgbm.npz';save(target,values)
        write(target.with_suffix('.json'),{'status':'completed','device':'cpu','n':len(inverse),'model_sha256':pins,
            'prediction_sha256':sha(target),'optimizer_steps':0,'created_KST':now()})
    verify_protocol();write(OUT/'TREE_COMPLETED.json',{'status':'completed','elapsed_sec':time.perf_counter()-t,'created_KST':now()})

def finalize():
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    d,inverse,metadata=prepare();verify_protocol();pred={};modelpins={}
    for generation in ('old','new'):
      for family in FAMILIES:
        p=OUT/f'{generation}_{family}.npz';r=read(p.with_suffix('.json'))
        if sha(p)!=r['prediction_sha256']:raise ValueError('Prediction changed')
        pred[generation,family]=load(p);modelpins.update(r['model_sha256'])
    combined={}
    for name,gen,weights in [('old_B1','old',(.60,.25,.15)),('new_B2','new',(.50,.30,.20))]:
      for target in ('share','prob'):
        combined[name+'_'+target]=sum(fw*.5*(pred[gen,family][target+'_D0']+pred[gen,family][target+'_H0']) for family,fw in zip(FAMILIES,weights))
    arrays={'law':np.concatenate([np.full(m['n'],m['law']) for m in metadata]),
            'sample_id':np.concatenate([load(ROOT/m['path'])['sample_id'] for m in metadata]),**combined}
    save(OUT/'fixed_blend_predictions.npz',arrays)
    metrics=[];rowparts=[];offset=0
    for m in metadata:
      raw=load(ROOT/m['path']);n=m['n'];y=raw['share'];z=y>0;active=np.any(raw['cell_share']>0,2).sum(1)
      row=pd.DataFrame({'law':m['law'],'sample_id':raw['sample_id'],'target_share':y,'presence':z.astype(int),'active_minutes':active})
      for name in ('old_B1','new_B2'):
        s=combined[name+'_share'][offset:offset+n];c=combined[name+'_prob'][offset:offset+n]
        row[name+'_share']=s;row[name+'_score']=c
        slices={'all':np.ones(n,bool),'y=0':y==0,'0<y<=0.05':z&(y<=.05),
                '0.05<y<=0.15':(y>.05)&(y<=.15),'y>0.15':y>.15,'0<y<=0.10':z&(y<=.10),
                'active_minutes>=12 & y>0':z&(active>=12)}
        for label,ix in slices.items():
          if not ix.any():continue
          e=s[ix]-y[ix]; aucix=ix if label=='all' else ix|(y==0)
          auc=float(roc_auc_score(z[aucix],c[aucix])) if np.unique(z[aucix]).size==2 else None
          share_auc=float(roc_auc_score(z[aucix],s[aucix])) if np.unique(z[aucix]).size==2 else None
          metrics.append({'law':m['law'],'candidate':name,'slice':label,'n':int(ix.sum()),'target_mean':float(y[ix].mean()),
              'prediction_mean':float(s[ix].mean()),'rmse':float(np.sqrt(np.mean(e*e))),'bias':float(e.mean()),
              'score_mean':float(c[ix].mean()),'score_ge090':float((c[ix]>=.9).mean()),
              'score_AUC_zero_vs_slice':auc,'share_AUC_zero_vs_slice':share_auc})
      offset+=n;rowparts.append(row)
    df=pd.DataFrame(metrics);df.to_csv(OUT/'LAW_TRANSFER_MATRIX.csv',index=False)
    df[df.slice.eq('all')].to_csv(OUT/'LAW_TRANSFER_OVERALL.csv',index=False)
    rows=pd.concat(rowparts,ignore_index=True);rows.to_csv(OUT/'LAW_TRANSFER_ROW_PREDICTIONS.csv',index=False)
    paired=[];vanilla=rows[rows.law.eq('vanilla_addition')].set_index('sample_id')
    for law in LAWS[:-1]:
        part=rows[rows.law.eq(law)].set_index('sample_id');v=vanilla.loc[part.index]
        for candidate in ('old_B1','new_B2'):
            for source,a in [(law,part),('vanilla_paired_subset',v)]:
                y=a.target_share.to_numpy();s=a[candidate+'_share'].to_numpy();c=a[candidate+'_score'].to_numpy();z=y>0
                paired.append({'custom_law':law,'source':source,'candidate':candidate,'n':len(y),'target_mean':float(y.mean()),
                    'rmse':float(np.sqrt(np.mean((s-y)**2))),'bias':float(np.mean(s-y)),
                    'AUC':float(roc_auc_score(z,c)) if np.unique(z).size==2 else None})
    pd.DataFrame(paired).to_csv(OUT/'PAIRED_LAW_VS_VANILLA.csv',index=False)
    files=[p for p in OUT.glob('*') if p.is_file() and p.name not in ('OUTPUT_SHA256.json','COMPLETED.json','progress.json')]
    write(OUT/'OUTPUT_SHA256.json',{p.name:{'sha256':sha(p),'bytes':p.stat().st_size}for p in sorted(files)})
    write(OUT/'COMPLETED.json',{'status':'completed','created_KST':now(),'rows_total':len(rows),'law_count':len(LAWS),
        'model_sha256':modelpins,'new_training':False,'optimizer_steps':0,'device':'cpu','raw_Test_or_query_read':False,
        'matrix_sha256':sha(OUT/'LAW_TRANSFER_MATRIX.csv')})
    print(df[df.slice.eq('all')].to_string(index=False),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('prepare','neural','trees','finalize','all'),default='all');args=parser.parse_args()
    sys.addaudithook(forbid_test)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['TOKENIZERS_PARALLELISM']='false'
    if args.phase=='prepare':prepare()
    elif args.phase=='neural':neural()
    elif args.phase=='trees':trees()
    elif args.phase=='finalize':finalize()
    else:
        prepare()
        for phase in ('neural','trees','finalize'):
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--phase',phase],cwd=ROOT,env=os.environ.copy(),check=True)

if __name__=='__main__': main()
