"""One-shot R2 completed-artifact collection. No scheduler or training launch.

Invoke manually at a bounded interval. Only completed ledger artifacts and
already-computed result reports are read. Every inbound ZIP is CRC/SHA checked.
"""
from pathlib import Path, PurePosixPath
import argparse,base64,hashlib,json,os,re,subprocess,time,uuid,zipfile

ROOT=Path(__file__).resolve().parents[1]
BASE='/workspace/underwater_vessel_noise_r2'
RUN=BASE+'/project/models/group_cv_physics_v2_r2_adam'
PY='/workspace/underwater_vessel_noise_r1/venv/bin/python'
BACKUP_SCRIPT='/workspace/underwater_vessel_noise_r1/ops/backup_completed_training.py'
HOST='root@103.196.86.97'
OPTIONS=['-oBatchMode=yes','-oConnectTimeout=15','-oUserKnownHostsFile=/private/tmp/underwater_runpod_known_hosts','-oStrictHostKeyChecking=accept-new']
BACKUPS=ROOT/'ops/server_backups'
PROGRESS=ROOT/'ops/r2_progress'
STATE=PROGRESS/'collection_state.json'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def safe_name(name):
    p=PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or p.as_posix()!=name or '\\' in name:
        raise ValueError('Unsafe collection path')
    return name

def atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.partial_'+uuid.uuid4().hex[:8])
    temporary.write_bytes(data);os.replace(temporary,path)

def remote(code):
    r=subprocess.run(['ssh',*OPTIONS,'-p','13475',HOST,PY,'-'],input=code,text=True,capture_output=True)
    if r.returncode:raise RuntimeError('Read-only/backup SSH failed: '+r.stderr[-2000:]+r.stdout[-2000:])
    return json.loads(r.stdout)

def probe(known):
    return remote("""
from pathlib import Path
import base64,hashlib,json
run=Path(%r);known=%r
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
result={'run_exists':run.exists(),'folds':[],'final':None}
for fold in sorted(run.glob('fold_*')):
 p=fold/'artifact_ledger.json';m=fold/'outer_metrics.json';policy=fold/'ensemble_policy.json'
 if not p.exists():continue
 ledger=json.loads(p.read_text())
 required={'task_inner_seed_42','tabicl_inner_seed_42','kumo_large_inner_seed_42','frozen_inner_policy','task_outer_seed_42','tabicl_outer_seed_42','kumo_large_outer_seed_42'}
 row={'fold':fold.name,'completed_steps':sorted(ledger),'ledger_sha256':sha(p),'outer_complete':False}
 if required.issubset(ledger) and m.exists() and policy.exists():
  if sha(policy)!=ledger['frozen_inner_policy']['ensemble_policy.json']:raise ValueError('Frozen policy SHA changed')
  metric=json.loads(m.read_text())
  if metric.get('fold')!=int(fold.name.split('_')[1]):raise ValueError('Fold metrics binding differs')
  row.update(outer_complete=True,outer_metrics_sha256=sha(m),policy_sha256=sha(policy),ledger_sha256=sha(p))
  if known.get(fold.name)!=sha(m):row['outer_metrics_base64']=base64.b64encode(m.read_bytes()).decode()
 result['folds'].append(row)
names=['execution_identity.json','cv_report.json','oof_predictions.npz','oof_metadata.parquet']
if all((run/n).is_file() for n in names):
 report=json.loads((run/'cv_report.json').read_text())
 if report.get('status','').startswith('completed nested Group CV'):
  result['final']={'file_sha256':{n:sha(run/n) for n in names},'n_splits':report.get('n_splits'),'oof_examples':report.get('oof_examples')}
print(json.dumps(result))
"""%(RUN,known))

def snapshot():
    expected=sha(ROOT/'ops/backup_completed_training.py')
    return remote("""
from pathlib import Path
import hashlib,json,subprocess
p=Path(%r)
if hashlib.sha256(p.read_bytes()).hexdigest()!=%r:raise ValueError('Backup helper SHA differs from audited local code')
r=subprocess.run([%r,str(p),'--run-dir',%r,'--backup-dir',%r],capture_output=True,text=True)
if r.returncode:raise RuntimeError(r.stdout+r.stderr)
print(r.stdout)
"""%(BACKUP_SCRIPT,expected,PY,RUN,BASE+'/ops/backups'))

def verify_archive(path,manifest_name):
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)) or archive.testzip() is not None:raise ValueError('ZIP duplicate/CRC error')
        manifest=json.loads(archive.read(manifest_name))
        entries=manifest['files']
        if set(names)!={e['archive_path'] for e in entries}|{manifest_name}:raise ValueError('Unmanifested ZIP entries')
        entry_map={}
        for entry in entries:
            name=safe_name(entry['archive_path'])
            if entry.get('relative_path')!=name:raise ValueError('ZIP restoration path differs')
            h=hashlib.sha256();size=0
            with archive.open(name) as source:
                for b in iter(lambda:source.read(2**20),b''):h.update(b);size+=len(b)
            if h.hexdigest()!=entry['sha256'] or size!=entry['bytes']:raise ValueError('ZIP entry hash/size differs: '+name)
            entry_map[name]=entry
        for row in manifest.get('source_ledgers',[]):
            ledger=json.loads(archive.read(row['path']))
            if sorted(ledger)!=sorted(row['included_steps']):raise ValueError('Filtered ledger step set differs')
            fold=PurePosixPath(row['path']).parent.as_posix()
            for items in ledger.values():
                for name,digest in items.items():
                    if entry_map[fold+'/'+name]['sha256']!=digest:raise ValueError('Completed ledger/ZIP SHA disagreement')
        if manifest_name=='__result_collection_manifest.json':
            with archive.open('oof_predictions.npz') as source:
                with zipfile.ZipFile(source) as predictions:
                    if predictions.testzip() is not None:raise ValueError('Final OOF NPZ internal CRC differs')
            parquet=archive.read('oof_metadata.parquet')
            if len(parquet)<12 or parquet[:4]!=b'PAR1' or parquet[-4:]!=b'PAR1':raise ValueError('Final OOF Parquet footer is incomplete')
    return manifest

def download(result,manifest_name,allowed_folds=None):
    path=result['path']
    if not path.startswith(BASE+'/ops/backups/') or not re.fullmatch(r'[A-Za-z0-9_.-]+\.zip',Path(path).name):
        raise ValueError('Remote archive path leaves approved backup directory')
    BACKUPS.mkdir(parents=True,exist_ok=True);local=BACKUPS/('r2_'+Path(path).name)
    if not local.exists():
        partial=local.with_name(local.name+'.partial_'+uuid.uuid4().hex[:8])
        r=subprocess.run(['scp',*OPTIONS,'-P','13475',HOST+':'+path,str(partial)],capture_output=True,text=True)
        if r.returncode:raise RuntimeError('Inbound SCP failed; preserve remote archive and ask root to download: '+r.stderr[-2000:])
        if sha(partial)!=result['sha256']:raise ValueError('Inbound archive SHA differs')
        os.link(partial,local);partial.unlink()
    if sha(local)!=result['sha256']:raise ValueError('Existing local backup SHA differs')
    manifest=verify_archive(local,manifest_name)
    with zipfile.ZipFile(local) as archive:
        for entry in manifest['files']:
            name=entry['archive_path']
            if allowed_folds is not None and name.startswith('fold_') and name.split('/')[0] not in allowed_folds:continue
            if manifest_name=='__result_collection_manifest.json' or name.endswith(('_history.json','_training_summary.json','run_manifest.json','ensemble_policy.json','artifact_ledger.json','outer_metrics.json')) or name=='cv_report.json':
                destination=PROGRESS/safe_name(name)
                raw=archive.read(name)
                if destination.exists() and destination.read_bytes()!=raw:
                    # A filtered ledger may acquire more completed steps. All
                    # actual training history/summary artifacts are immutable.
                    if not name.endswith('artifact_ledger.json'):raise ValueError('Previously collected completed artifact changed: '+name)
                    previous=json.loads(destination.read_bytes());current=json.loads(raw)
                    if any(current.get(label)!=items for label,items in previous.items()):raise ValueError('Previously completed ledger step changed: '+name)
                atomic(destination,raw)
    receipt={'archive':str(local),'sha256':result['sha256'],'bytes':local.stat().st_size,
        'ZIP_CRC_passed':True,'entry_SHA_and_completed_ledger_passed':True,'entries':len(manifest['files']),
        'remote':result,'verified_unix':time.time(),
        'field_scope':{'remote.training_started':'Whether this backup helper launched training; false does not mean the independent CV job is stopped.'}}
    atomic(local.with_suffix('.verification.json'),json.dumps(receipt,indent=2).encode())
    return receipt

def final_snapshot(expected):
    return remote("""
from pathlib import Path
import datetime,hashlib,json,os,uuid,zipfile
run=Path(%r);out=Path(%r);expected=%r
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
names=list(expected)
for f in sorted(run.glob('fold_*')):names.extend(str((f/n).relative_to(run)) for n in ['outer_metrics.json','ensemble_policy.json','artifact_ledger.json'])
entries=[]
for name in names:
 p=run/name
 if p.is_symlink() or not p.is_file():raise ValueError('Missing/linked final result')
 digest=sha(p)
 if name in expected and digest!=expected[name]:raise ValueError('Final result changed since probe')
 entries.append({'relative_path':name,'archive_path':name,'sha256':digest,'bytes':p.stat().st_size})
out.mkdir(parents=True,exist_ok=True)
name='results_r2_'+datetime.datetime.now(datetime.timezone.utc).strftime('%%Y%%m%%dT%%H%%M%%S_%%fZ')+'_'+uuid.uuid4().hex[:8]+'.zip'
path=out/name;partial=path.with_suffix('.zip.partial')
with zipfile.ZipFile(partial,'x',zipfile.ZIP_DEFLATED,compresslevel=4) as archive:
 for row in entries:archive.write(run/row['archive_path'],row['archive_path'])
 archive.writestr('__result_collection_manifest.json',json.dumps({'format':'completed_r2_results_v1','files':entries,'source_data_or_vendor_weights_included':False},indent=2))
with zipfile.ZipFile(partial) as archive:
 if archive.testzip() is not None:raise ValueError('Final result ZIP CRC failure')
for row in entries:
 if sha(run/row['archive_path'])!=row['sha256']:raise ValueError('Final result changed while archiving')
os.link(partial,path);partial.unlink()
print(json.dumps({'status':'completed','path':str(path),'sha256':sha(path),'bytes':path.stat().st_size,'files':len(entries)}))
"""%(RUN,BASE+'/ops/backups',expected))

def once():
    PROGRESS.mkdir(parents=True,exist_ok=True)
    state=json.loads(STATE.read_text()) if STATE.exists() else {'folds':{},'final_complete':False}
    state.setdefault('ledgers',{})
    status=probe({name:row['outer_metrics_sha256'] for name,row in state['folds'].items()})
    for row in status['folds']:
        old=state['folds'].get(row['fold'])
        if old and any(row.get(key)!=old.get(key) for key in ['outer_metrics_sha256','policy_sha256','ledger_sha256']):
            raise ValueError('Previously collected completed fold changed: '+row['fold'])
    changed=[r for r in status['folds'] if r['outer_complete'] and r['fold'] not in state['folds']]
    new_ledgers=[r for r in status['folds'] if r['completed_steps'] and state['ledgers'].get(r['fold'])!=r['ledger_sha256']]
    events=[]
    if changed or new_ledgers:
        result=snapshot()
        if result['status']!='completed' or result.get('skipped'):raise ValueError('Backup did not fully verify completed ledger: '+str(result))
        receipt=download(result,'__snapshot_manifest.json',allowed_folds={r['fold'] for r in status['folds'] if r['completed_steps']})
        for row in changed:
            raw=base64.b64decode(row['outer_metrics_base64'],validate=True)
            if hashlib.sha256(raw).hexdigest()!=row['outer_metrics_sha256']:raise ValueError('Outer report transfer hash differs')
            json.loads(raw);atomic(PROGRESS/row['fold']/'outer_metrics.json',raw)
            row.pop('outer_metrics_base64',None);state['folds'][row['fold']]={**row,'backup':receipt['archive']}
        for row in {r['fold']:r for r in [*new_ledgers,*changed]}.values():
            state['ledgers'][row['fold']]=row['ledger_sha256']
            atomic(PROGRESS/row['fold']/'collection_receipt.json',json.dumps({
                'source_run':RUN,'fold':{k:v for k,v in row.items() if k!='outer_metrics_base64'},'verified_archive':receipt,
                'stage':'outer_complete' if row['outer_complete'] else 'completed_ledger_steps; fold still in progress',
                'history_summary_source':'Exact files from SHA-verified completed ledger ZIP; original RUN-relative paths retained.'
            },indent=2).encode())
        events.append({'new_completed_folds':[r['fold'] for r in changed],
            'new_completed_ledgers':[{k:r[k] for k in ['fold','completed_steps','ledger_sha256']} for r in new_ledgers],
            'backup_receipt':receipt})
    final=status['final']
    if final and len(state['folds'])==5 and not state['final_complete']:
        signature=hashlib.sha256(json.dumps(final['file_sha256'],sort_keys=True).encode()).hexdigest()
        candidate=state.get('final_candidate',{})
        if candidate.get('signature')==signature and time.time()-candidate.get('seen_unix',0)>=30:
            final_backup=snapshot()
            if final_backup['status']!='completed' or final_backup.get('skipped'):raise ValueError('Final model backup did not fully verify')
            receipt=download(final_backup,'__snapshot_manifest.json')
            results=download(final_snapshot(final['file_sha256']),'__result_collection_manifest.json')
            state.update(final_complete=True,final_backups=[receipt,results],final_file_sha256=final['file_sha256'])
            events.append({'final_complete':True,'model_backup':receipt['archive'],'OOF_results_backup':results['archive']})
        else:state['final_candidate']={'signature':signature,'seen_unix':time.time()}
    for row in status['folds']:row.pop('outer_metrics_base64',None)
    state['last_probe']={'unix':time.time(),'status':status}
    atomic(STATE,json.dumps(state,indent=2).encode())
    print(json.dumps({'events':events,'completed_folds':sorted(state['folds']),'final_complete':state['final_complete'],'status':status},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args();once()
