"""Predeclared INNER-only scalar calibration of a fixed FP32 CPU CNN.

The base CNN already used INNER for early stopping. Calibrator crossfitting
does not make its predictions independent neural OOF estimates. Importing this
module performs no data IO/fitting; query application accepts a share vector.
"""
from pathlib import Path
import argparse, datetime, hashlib, json, os, sys, time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/R5_CNN_PILOT_CALIBRATION'
SPEC_SHA='580728c8531382b04148c492e13483e7acd06db7a1cf96642ac1ca914dbeafaa'
CPU_SHA='6a44209a9c160b158ac01ac681a69f8fcfd4fcba2ac56a8fecc9fccc52f7278e'
PROFILE_SHA='9f91a8bf85c2b7ba33ea85368269073a2f6d5c53a499c87dc23f0f3b67ccce80'
CHECKPOINT_SHA='aa222697275306cf93e5778a707f7ee5fd75610ae0a0bf643d654854ee96d46f'
POLICY_FORMAT='r5_single_cnn_frozen_calibration_v1'
CANDIDATES=('identity','offset','constrained_affine')
RIDGE=.01

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def load_json(path):
    def bad(v):raise ValueError('Nonfinite JSON constant: '+v)
    return json.loads(Path(path).read_text(),parse_constant=bad)
def write_new(path,value):
    with Path(path).open('x') as f:f.write(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def array_sha(a):
    a=np.asarray(a);h=hashlib.sha256(json.dumps({'shape':list(a.shape),'dtype':str(a.dtype)},sort_keys=True,separators=(',',':')).encode());h.update(np.ascontiguousarray(a).tobytes());return h.hexdigest()

def validate_frozen_policy(policy):
    if not isinstance(policy,dict) or policy.get('format')!=POLICY_FORMAT or policy.get('policy_sha256')!=canonical({k:v for k,v in policy.items() if k!='policy_sha256'}):raise ValueError('Frozen calibration policy format/SHA mismatch')
    if policy.get('checkpoint_sha256')!=CHECKPOINT_SHA or policy.get('CPU_predictions_sha256')!=CPU_SHA or policy.get('specification_sha256')!=SPEC_SHA or policy.get('calibration_code_sha256')!=sha(__file__):raise ValueError('Calibration model/input/specification/code identity mismatch')
    if policy.get('fit_scope')!='INNER_only' or policy.get('frozen_before_Test') is not True or policy.get('no_OUTER_or_Test_fitting') is not True or policy.get('application')!='clip(a*p+b,0,1)':raise ValueError('Unsupported frozen calibration scope')
    name=policy.get('candidate');c=policy.get('coefficients',{});a,b=c.get('a'),c.get('b')
    if name not in CANDIDATES or any(type(v) not in {int,float} or not np.isfinite(v) for v in [a,b]):raise ValueError('Finite declared scalar coefficients required')
    if not .75<=a<=1.25 or not -.05<=b<=.05 or name in {'identity','offset'} and a!=1 or name=='identity' and b!=0:raise ValueError('Frozen coefficients violate predeclared bounds')
    if name!='identity' and policy.get('all_admission_gates_passed') is not True:raise ValueError('Unadmitted scalar calibration forbidden')
    return policy

def apply_frozen_policy(policy,raw_share):
    validate_frozen_policy(policy)
    if isinstance(raw_share,dict):raise TypeError('Pass raw share only; no labels or cohort metadata')
    p=np.asarray(raw_share)
    if p.ndim!=1 or not len(p) or not np.issubdtype(p.dtype,np.floating) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Nonempty finite raw share vector in [0,1] required')
    c=policy['coefficients'];return np.clip(float(c['a'])*p.astype('float64')+float(c['b']),0,1)

def fit_candidate(name,p,y):
    if name=='identity':return {'a':1.,'b':0.}
    p=np.asarray(p,dtype='float64');y=np.asarray(y,dtype='float64')
    if p.shape!=y.shape or p.ndim!=1 or not len(y) or not np.isfinite(p).all() or not np.isfinite(y).all():raise ValueError('Finite aligned meta-fit rows required')
    lower,upper=(1.,1.) if name=='offset' else (.75,1.25)
    if name not in {'offset','constrained_affine'}:raise ValueError('Unknown predeclared candidate')
    H=np.array([[np.mean(p*p)+RIDGE,np.mean(p)],[np.mean(p),1+RIDGE]],dtype='float64');t=np.array([np.mean(p*y)+RIDGE,np.mean(y)])
    choices=[];interior=np.linalg.solve(H,t)
    if lower<=interior[0]<=upper and -.05<=interior[1]<=.05:choices.append(interior)
    for a in sorted({lower,upper}):choices.append(np.array([a,np.clip((t[1]-H[1,0]*a)/H[1,1],-.05,.05)]))
    for b in [-.05,.05]:choices.append(np.array([np.clip((t[0]-H[0,1]*b)/H[0,0],lower,upper),b]))
    def loss(v):return float(np.mean((v[0]*p+v[1]-y)**2)+RIDGE*((v[0]-1)**2+v[1]**2))
    best=min(choices,key=loss);return {'a':float(best[0]),'b':float(best[1])}

def transform(p,c):return np.clip(c['a']*p+c['b'],0,1)
def metric(y,p):
    e=p-y;return {'n':len(y),'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(np.mean(abs(e))),'bias_pred_minus_truth':float(np.mean(e))}
def group_metrics(y,p,group):return {str(g):metric(y[group==g],p[group==g]) for g in sorted(set(group.tolist()))}

def crossfit(p,y,refs,bg):
    zero=y==0;names=sorted(set(refs[~zero].tolist()));bgnames=sorted(set(bg.tolist()))
    if len(names)!=5 or len(bgnames)!=50 or np.count_nonzero(zero)!=600 or len(y)!=3000:raise ValueError('Predeclared 5-reference/50-background/600-zero/3000-row population differs; no partition retry')
    ri=np.full(len(y),-1,dtype='int64')
    for i,name in enumerate(names):ri[(~zero)&(refs==name)]=i
    shard=np.array([int(hashlib.sha256(str(g).encode()).hexdigest(),16)%5 for g in bg],dtype='int64')
    if set(shard.tolist())!=set(range(5)):raise ValueError('Predeclared background hash produced missing shard; no retry')
    predictions={name:np.full(len(y),np.nan) for name in CANDIDATES};coverage=np.zeros(len(y),'int64');fold_id=np.full(len(y),-1,'int64');records=[]
    for i in range(5):
        for j in range(5):
            held=((~zero)&(ri==i)&(shard==j)) | (zero&(i==0)&(shard==j))
            fit=(shard!=j)&(zero|(ri!=i))
            if np.any(held&fit) or set(bg[held]).intersection(bg[fit]) or set(refs[held&~zero]).intersection(refs[fit&~zero]):raise ValueError('Meta-fit/heldout source overlap')
            if not np.any(fit&zero) or not np.any(fit&~zero):raise ValueError('Predeclared meta-fit has missing target class; no retry')
            coverage[held]+=1;fold_id[held]=i*5+j;params={}
            for name in CANDIDATES:
                c=fit_candidate(name,p[fit],y[fit]);params[name]=c;predictions[name][held]=transform(p[held],c)
            records.append({'reference_index':i,'reference_key':names[i],'background_shard':j,'fit_rows':int(fit.sum()),'fit_zero_rows':int(np.count_nonzero(fit&zero)),'heldout_rows':int(held.sum()),'heldout_zero_rows':int(np.count_nonzero(held&zero)),'fit_background_groups':sorted(set(bg[fit])),'heldout_background_groups':sorted(set(bg[held])),'positive_reference_and_background_disjoint':True,'coefficients':params})
    if not np.all(coverage==1) or any(not np.isfinite(v).all() for v in predictions.values()):raise ValueError('Crossed coverage must be exactly once for every row')
    return predictions,records,fold_id,shard

def admission(y,predictions,refs,bg):
    identity=predictions['identity'];base=metric(y,identity);zero=y==0;reference=np.where(zero,'__zero_control__',refs);base_ref=group_metrics(y,identity,reference);rows={}
    for name,p in predictions.items():
        m=metric(y,p);delta=(p-y)**2-(identity-y)**2;bg_delta={str(g):float(delta[bg==g].mean()) for g in sorted(set(bg.tolist()))};ref_m=group_metrics(y,p,reference)
        improvement=base['rmse']-m['rmse'];fraction=float(np.mean([v<0 for v in bg_delta.values()]));worsening={g:ref_m[g]['rmse']-base_ref[g]['rmse'] for g in base_ref}
        gates={'RMSE_improvement_at_least_0.0005':improvement>=.0005,'background_group_improvement_fraction_at_least_0.60':fraction>=.60,'each_positive_reference_and_zero_RMSE_worsening_at_most_0.003':all(v<=.003 for v in worsening.values())}
        rows[name]={'crossfitted_diagnostic':m,'RMSE_improvement_vs_identity':improvement,'background_group_improvement_fraction':fraction,'background_group_paired_mean_delta_MSE':bg_delta,'by_positive_reference_and_zero':ref_m,'reference_RMSE_worsening_vs_identity':worsening,'gates':gates,'admitted':name!='identity' and all(gates.values())}
    admitted=[name for name in CANDIDATES if rows[name]['admitted']];selected=min(admitted,key=lambda name:rows[name]['crossfitted_diagnostic']['rmse']) if admitted else 'identity'
    return selected,rows

def install_guard(folder):
    data=(ROOT/'data').resolve();allowed={(folder/n).resolve() for n in ['readiness_manifest.json','sources/manifest.json','train.npz','validation.npz','inner_validation_metadata.parquet','sources/background_manifest.parquet']}
    def audit(event,args):
        if event in {'socket.connect','socket.getaddrinfo','subprocess.Popen','os.system'}:raise RuntimeError('No network/subprocess in scalar calibration')
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0])).resolve()
            if path.name.lower() in {'test.csv','outer_validation.npz','outer_validation_metadata.parquet'} or path.suffix.lower() in {'.wav','.flac','.mp3'}:raise ValueError('Test/OUTER/audio forbidden')
            writing=args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)
            if path.is_relative_to(data) and (path not in allowed or writing):raise ValueError('Undeclared source access/write forbidden')
            if writing and not path.is_relative_to(OUT.resolve()):raise ValueError('Calibration may only write its separate report directory')
    sys.addaudithook(audit)

def run():
    started=time.perf_counter();sys.dont_write_bytecode=True
    spec=OUT/'specification.json'
    if sha(spec)!=SPEC_SHA:raise ValueError('Predeclared specification changed')
    for name in ['metrics.json','crossed_predictions.npz','frozen_policy.json','README.md']:
        if (OUT/name).exists():raise FileExistsError('No calibration rerun/overwrite')
    declaration=load_json(spec);folder=ROOT/declaration['dataset_fold'];install_guard(folder)
    profile_path=ROOT/declaration['cpu_profile_report'];cpu=ROOT/declaration['cpu_predictions']
    if sha(profile_path)!=PROFILE_SHA or sha(cpu)!=CPU_SHA:raise ValueError('Predeclared CPU prediction/profile SHA differs')
    profile=load_json(profile_path)
    if profile['checkpoint_sha256']!=CHECKPOINT_SHA or profile['CPU_full_FP32'] is not True or profile['prediction_file_sha256']!=CPU_SHA or profile['model_state_unchanged'] is not True:raise ValueError('Fixed CPU checkpoint provenance differs')
    snapshots={str(spec):SPEC_SHA,str(profile_path):PROFILE_SHA,str(cpu):CPU_SHA,str(Path(__file__)):sha(__file__)}
    for field in ['source_files_sha256','input_files_sha256','code_sha256']:
        for name,h in profile[field].items():
            path=ROOT/name
            if sha(path)!=h:raise ValueError('CPU source/input/code changed: '+name)
            snapshots[str(path)]=h
    gate=load_json(folder/'readiness_manifest.json');binding=profile['training_binding']
    if binding['fold']!=0 or binding['seed']!=42 or binding['data_package_hash']!=gate['data_package_hash'] or binding['source_hash']!=gate['source_hash']:raise ValueError('INNER package/source binding differs')
    import pandas as pd
    md=pd.read_parquet(folder/'inner_validation_metadata.parquet').reset_index(drop=True)
    with np.load(folder/'validation.npz',allow_pickle=False) as z:data={k:z[k] for k in ['db','share','presence','cell_share']}
    with np.load(cpu,allow_pickle=False) as z:cp={k:z[k] for k in z.files}
    if set(cp)!={'share','score','cell_mask','sample_id'}:raise ValueError('Expected exact CPU prediction schema')
    ids=md.sample_id.astype(str).to_numpy(dtype=str);y=data['share'].astype('float64');p=cp['share'].astype('float64')
    if not np.array_equal(ids,cp['sample_id']) or len(set(ids.tolist()))!=3000 or p.shape!=(3000,) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('INNER CPU prediction row/shape/range mismatch')
    raw=data['db'].astype('float64');observed=np.isfinite(raw);energy=np.where(observed,10**((raw-np.nanmax(raw,(1,2),keepdims=True))/10),0);w=(energy/energy.sum((1,2),keepdims=True)).astype('float32')
    expected={'sample_id_order_sha256':canonical(ids.tolist()),'raw_db_sha256':array_sha(data['db']),'raw_energy_weights_sha256':array_sha(w),'target_share_sha256':array_sha(data['share'])}
    if expected!=profile['inner_identity']:raise ValueError('CPU raw INNER/weight/target/order fingerprint mismatch')
    q=cp['cell_mask']
    if q.shape!=(3000,15,25) or not np.isfinite(q).all() or ((q<0)|(q>1)).any() or not np.allclose((w.astype('float64')*q).sum((1,2)),p,atol=3e-6,rtol=0):raise ValueError('CPU physical pooling mismatch')
    refs=md.reference_group.astype(str).to_numpy();bg=md.background_group.astype(str).to_numpy();zero=y==0
    if not np.array_equal(data['presence'],(~zero).astype('float32')):raise ValueError('Known zero/addition label mismatch')
    for iszero,ref in zip(zero,refs):
        parsed=json.loads(ref)
        if not isinstance(parsed,list) or len(parsed)!=(0 if iszero else 1):raise ValueError('Single-ref/zero metadata convention differs')
    predictions,folds,fold_id,shard=crossfit(p,y,refs,bg);selected,rows=admission(y,predictions,refs,bg)
    # One full-INNER coefficient refit only for an admitted selected candidate.
    coefficients=fit_candidate(selected,p,y);full=transform(p,coefficients)
    report={'format':'r5_single_cnn_calibration_diagnostic_v1','created_at_utc':now(),'specification_sha256':SPEC_SHA,'calibration_code_sha256':sha(__file__),'CPU_prediction_sha256':CPU_SHA,'CPU_profile_sha256':PROFILE_SHA,'checkpoint_sha256':CHECKPOINT_SHA,'inner_identity':expected,'training_binding':binding,'selected_candidate':selected,'candidates':rows,'crossed_folds':folds,'source_checks':{'rows':3000,'positive_reference_groups':5,'zero_rows':600,'background_groups':50,'each_row_heldout_exactly_once':True,'per_fit_reference_and_background_disjoint':True,'extra_24h_purge_applied':False,'original_TRAIN_INNER_purge_modified':False,'temporal_support_metadata':'INNER base_background_index can join sources.background_manifest.base_window_index to site/sensor_group/recorder/start_export_clock; exact observed-minute timestamps/UTC confirmation are absent here. No additional temporal-independence claim.'},'selected_full_INNER_refit_coefficients':coefficients,'selected_full_INNER_resubstitution_diagnostic_only':metric(y,full),'full_INNER_coefficient_refits':int(selected!='identity'),'independent_neural_OOF_estimate':False,'base_model_already_used_all_INNER_for_early_stopping':True,'new_GPU_training':False,'OUTER_Test_or_external_access':False,'LB_score_forecast':None,'elapsed_seconds':time.perf_counter()-started}
    np.savez_compressed(OUT/'crossed_predictions.npz',sample_id=ids,truth=y,raw_CPU_share=p,reference_group=refs,background_group=bg,background_shard=shard,crossed_fold_id=fold_id,**predictions)
    write_new(OUT/'metrics.json',report)
    policy={'format':POLICY_FORMAT,'created_at_utc':now(),'candidate':selected,'coefficients':coefficients,'application':'clip(a*p+b,0,1)','checkpoint_sha256':CHECKPOINT_SHA,'checkpoint_epoch':profile['checkpoint_epoch'],'CPU_predictions_sha256':CPU_SHA,'CPU_profile_sha256':PROFILE_SHA,'specification_sha256':SPEC_SHA,'calibration_code_sha256':sha(__file__),'training_binding':binding,'fit_scope':'INNER_only','frozen_before_Test':True,'no_OUTER_or_Test_fitting':True,'all_admission_gates_passed':bool(selected!='identity' and rows[selected]['admitted']),'admission_diagnostics_sha256':sha(OUT/'metrics.json'),'source_core_code_sha256':profile['code_sha256'],'raw_inference_policy':{'device':'cpu','dtype':'float32','batch_size':128,'torch_threads':4,'torch_interop_threads':1,'observation_policy':profile['observation_policy']},'limitations':declaration['limitations'],'score_presence_modified':False}
    policy['policy_sha256']=canonical(policy);validate_frozen_policy(policy);write_new(OUT/'frozen_policy.json',policy)
    for name,h in snapshots.items():
        if sha(name)!=h:raise ValueError('Source/input/spec/code changed during calibration')
    lines=['# 고정 CNN의 INNER 전용 보정 진단','',f'선택: **{selected}**, a={coefficients["a"]:.12g}, b={coefficients["b"]:.12g}.','', '| 후보 | calibrator-crossfit RMSE | identity 대비 개선 | BG 개선 비율 | gate 통과 |','|---|---:|---:|---:|---|']
    for name in CANDIDATES:
        v=rows[name];lines.append(f'| {name} | {v["crossfitted_diagnostic"]["rmse"]:.9f} | {v["RMSE_improvement_vs_identity"]:+.9f} | {v["background_group_improvement_fraction"]:.1%} | {v["admitted"]} |')
    lines+=['','정해진 25개 reference×background 분할과 세 후보를 한 번만 계산했다. 모든 행은 검증에 정확히 한 번 포함됐고 각 fit의 양성 reference와 background group이 검증과 겹치지 않는다. zero는 사전 선언대로 i=0에서만 검증한다. 별도 24시간 purge는 추가하지 않았다.','', '기본 CNN이 전체 INNER로 early stopping됐으므로 이 수치는 독립 신경망 OOF 성능이 아니다. 합성 원천의 Test 대표성이나 LB 점수를 추정하지 않는다. 전 기준을 통과한 후보가 없으면 identity를 유지하며 재분할·gate 완화·재시도를 하지 않는다.','', '비identity 최종 계수는 선택 후 전체 INNER에서 한 번만 다시 fitting한다. 그 재대입 점수는 진단용이며 선택 근거가 아니다. scalar 보정 이후 원본 cell mask는 원본 share의 물리적 가중합을 설명하며, 보정된 scalar를 그대로 설명한다고 주장하지 않는다. score/presence는 수정하지 않는다.','', '[고정 사양](specification.json) · [진단 수치](metrics.json) · [동결 정책](frozen_policy.json)','']
    with (OUT/'README.md').open('x') as f:f.write('\n'.join(lines))
    return {'selected_candidate':selected,'coefficients':coefficients,'candidates':{k:{'rmse':v['crossfitted_diagnostic']['rmse'],'improvement':v['RMSE_improvement_vs_identity'],'BG_improved_fraction':v['background_group_improvement_fraction'],'gates':v['gates'],'admitted':v['admitted']} for k,v in rows.items()},'policy':str(OUT/'frozen_policy.json'),'policy_file_sha256':sha(OUT/'frozen_policy.json'),'policy_sha256':policy['policy_sha256'],'OUTER_Test_GPU_accessed':False}

def self_test():
    rng=np.random.default_rng(42);p=rng.uniform(0,1,100);y=np.clip(.88*p+.025+rng.normal(0,.01,len(p)),0,1)
    for name in ['offset','constrained_affine']:
        c=fit_candidate(name,p,y);cost=lambda a,b:float(np.mean((a*p+b-y)**2)+RIDGE*((a-1)**2+b*b));best=cost(c['a'],c['b'])
        for a in ([1.] if name=='offset' else np.linspace(.75,1.25,81)):
            for b in np.linspace(-.05,.05,81):assert best<=cost(a,b)+1e-12
    return {'passed':True,'scope':'tiny numeric box-quadratic fixture only','source_data_or_GPU_opened':False,'calibration_code_sha256':sha(__file__)}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-test',action='store_true');args=parser.parse_args();print(json.dumps(self_test() if args.self_test else run(),indent=2,allow_nan=False))
if __name__=='__main__':main()
