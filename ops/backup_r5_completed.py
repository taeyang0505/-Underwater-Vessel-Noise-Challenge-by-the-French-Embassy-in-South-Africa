"""Local snapshots of completed R5 models/folds, with SHA/CRC verification.

This script never launches training, mutates RUN, reads source data or uploads
anything. A snapshot is a recovery copy, not permission to resume a partial run.
"""
from pathlib import Path, PurePosixPath
import argparse, datetime, fcntl, hashlib, json, os, re, tempfile, uuid, zipfile

MANIFEST='__snapshot_manifest.json'
KINDS={'bg_cnn','tfgrid_fdy_conformer','lgbm_physics','timemoe_public','dasheng','fdy','itransformer'}
FORBIDDEN={'data','raw','external','pretrained','pretrained_r5','sealed_holdout','calibration','__pycache__'}

class BackupError(ValueError):pass

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def encode(value):return (json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
def read_json(path):
    def bad(v):raise BackupError('Nonfinite JSON: '+v)
    value=json.loads(Path(path).read_text(),parse_constant=bad)
    if not isinstance(value,dict):raise BackupError('JSON object required')
    return value

def name_ok(name):
    if not isinstance(name,str) or not name or '\\' in name or '\x00' in name:raise BackupError('Invalid snapshot path')
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or p.as_posix()!=name or re.match(r'^[A-Za-z]:',name):raise BackupError('Escaping/noncanonical snapshot path')
    if FORBIDDEN.intersection(s.lower() for s in p.parts) or p.name.lower() in {'test.csv','sample_submission.csv','domain_holdout.npz'}:raise BackupError('Source/vendor/Test payload forbidden')
    if p.suffix not in {'.json','.jsonl','.npz','.pt','.txt'}:raise BackupError('Unsupported snapshot artifact')
    return name

def regular(root,name):
    name_ok(name);path=Path(root)
    for part in PurePosixPath(name).parts:
        path=path/part
        if path.is_symlink():raise BackupError('Symlink artifact rejected')
    if not path.is_file():raise FileNotFoundError(name)
    return path

def _add(files,run,path,origin,expected=None):
    rel=path.relative_to(run).as_posix();path=regular(run,rel);actual=sha(path)
    if expected is not None and actual!=expected:raise BackupError('Recorded artifact SHA mismatch: '+rel)
    old=files.get(rel)
    if old and old['sha256']!=actual:raise BackupError('Conflicting artifact identity: '+rel)
    files[rel]={'path':path,'sha256':actual,'bytes':path.stat().st_size,'origin':origin}

def collect(run):
    files={};models=[];folds=[];skipped=[]
    for folder in sorted(run.glob('fold_*')):
        if not re.fullmatch(r'fold_0[0-4]',folder.name):continue
        if folder.is_symlink():raise BackupError('Symlink fold rejected')
        context_path=folder/'context.json'
        if not context_path.exists():continue
        context_path=regular(folder,'context.json')
        try:context=read_json(context_path)
        except (OSError,ValueError) as e:skipped.append({'fold':folder.name,'reason':'context incomplete: '+type(e).__name__});continue
        if canonical(context.get('config'))!=context.get('config_sha256'):raise BackupError('Context config hash mismatch')
        binding=context.get('training_binding');prior=context['prior'];prior_path=regular(folder,prior['filename'])
        if sha(prior_path)!=prior['file_sha256']:raise BackupError('Frozen prior SHA mismatch')
        obj=read_json(prior_path)
        if obj.get('fit_split')!='train' or obj.get('training_binding')!=binding or obj.get('shared_prior_payload_sha256')!=prior['payload_sha256'] or canonical({k:v for k,v in obj.items() if k!='shared_prior_payload_sha256'})!=prior['payload_sha256']:raise BackupError('Frozen prior payload/binding mismatch')
        fold_files={};_add(fold_files,run,context_path,'fold_context');_add(fold_files,run,prior_path,'TRAIN_shared_prior',prior['file_sha256'])
        for summary_path in sorted(folder.rglob('summary.json')):
            if summary_path.parent==folder:continue
            rel=summary_path.relative_to(folder).parts
            if not (len(rel)==3 and rel[0]=='models' and rel[1] in KINDS or len(rel)==4 and rel[0]=='preprocessing_pilot' and rel[2]=='bg_cnn'):continue
            summary_path=regular(run,summary_path.relative_to(run).as_posix())
            if summary_path.is_symlink():raise BackupError('Symlink model summary rejected')
            try:summary=read_json(summary_path)
            except (OSError,ValueError):skipped.append({'path':str(summary_path.relative_to(run)),'reason':'model summary incomplete'});continue
            kind=summary.get('kind')
            if kind not in KINDS or summary.get('status')!='completed':continue
            if summary.get('fixture_only') is not False or summary.get('training_binding')!=binding:raise BackupError('Nonproduction or cross-fold model summary')
            model_dir=summary_path.parent;selected={}
            try:
                _add(selected,run,summary_path,'completed_model_summary')
                if kind=='lgbm_physics':
                    required={'share.txt','presence.txt','prototypes.npz','physics_pca.json','inner_predictions.npz'}
                    if set(summary.get('artifact_sha256',{}))!=required:raise BackupError('Incomplete tree artifact ledger')
                    for name,expected in summary['artifact_sha256'].items():_add(selected,run,regular(model_dir,name),'completed_tree',expected)
                else:
                    for name,expected in [('best.pt',summary['checkpoint_sha256']),('inner_predictions.npz',summary['prediction_sha256'])]:_add(selected,run,regular(model_dir,name),'completed_neural',expected)
                    _add(selected,run,regular(model_dir,'epochs.jsonl'),'completed_epoch_history')
                    receipt=regular(model_dir,summary['dynamic_source_receipts_file']);record=read_json(receipt)
                    identity=summary.get('dynamic_source_identity');best=record.get('completed_epochs',{}).get(str(summary['best_epoch']))
                    if record.get('source_identity')!=identity or not isinstance(identity,dict) or identity.get('binding')!=binding or not isinstance(best,dict) or best.get('epoch')!=summary['best_epoch'] or best.get('clean_primary_once') is not True or best.get('examples')!=binding.get('train_examples') or best.get('source_identity_sha256')!=canonical(identity):raise BackupError('Best epoch source receipt mismatch')
                    _add(selected,run,receipt,'completed_dynamic_source_receipts')
            except (FileNotFoundError,json.JSONDecodeError) as e:
                skipped.append({'path':str(model_dir.relative_to(run)),'reason':'completed model still committing: '+type(e).__name__});continue
            # A SHA mismatch is explicit failure, never a partial model backup.
            files.update(selected);files.update(fold_files);models.append({'path':str(model_dir.relative_to(run)),'kind':kind,'summary_sha256':sha(summary_path)})
        summary_path=folder/'summary.json'
        if summary_path.exists():
            try:summary=read_json(summary_path)
            except (OSError,ValueError):skipped.append({'fold':folder.name,'reason':'fold summary incomplete'});continue
            if summary.get('status')=='completed':
                if summary.get('training_binding')!=binding or summary.get('outer_changed_policy') is not False:raise BackupError('Completed fold binding/policy mismatch')
                ledger=summary.get('artifact_sha256')
                if not isinstance(ledger,dict) or not ledger:raise BackupError('Complete fold artifact ledger required')
                selected={}
                for name,expected in ledger.items():_add(selected,run,regular(folder,name),'completed_fold_ledger',expected)
                _add(selected,run,summary_path,'completed_fold_summary');files.update(selected);folds.append(folder.name)
    # Running run_state.json is mutable and is deliberately not snapshotted.
    state_path=run/'run_state.json'
    if state_path.exists():
        state=read_json(state_path)
        if state.get('status') in {'completed','completed_requested_folds'}:
            _add(files,run,state_path,'completed_run_state')
    return files,models,folds,skipped

def verify_archive(path):
    if Path(path).is_symlink():raise BackupError('Symlink ZIP rejected')
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:raise BackupError('ZIP CRC failure')
        manifest=json.loads(z.read(MANIFEST));entries=manifest['files']
        if manifest.get('format')!='r5_completed_snapshot_v1':raise BackupError('Unknown snapshot format')
        names=[v['archive_path'] for v in entries]
        if len(set(names))!=len(names) or set(z.namelist())!=set(names+[MANIFEST]):raise BackupError('Unexpected/duplicate ZIP entry')
        for item in entries:
            name=name_ok(item['archive_path']);info=z.getinfo(name)
            if (info.external_attr>>16)&0o170000==0o120000:raise BackupError('ZIP symlink forbidden')
            h=hashlib.sha256();count=0
            with z.open(name) as src:
                for block in iter(lambda:src.read(2**20),b''):h.update(block);count+=len(block)
            if h.hexdigest()!=item['sha256'] or count!=item['bytes']:raise BackupError('ZIP entry SHA/size mismatch')
        expected=canonical([{k:v for k,v in e.items() if k!='origin'} for e in entries])
        if manifest.get('content_fingerprint')!=expected:raise BackupError('Snapshot content fingerprint mismatch')
    return {'verified':True,'sha256':sha(path),'bytes':Path(path).stat().st_size,'files':len(entries),'content_fingerprint':expected}

def snapshot(run,backup_dir,ops_dir=None):
    run=Path(run);backup_dir=Path(backup_dir)
    if run.is_symlink() or backup_dir.is_symlink():raise BackupError('Symlink RUN/backup root rejected')
    if not run.exists():return {'status':'noop','reason':'RUN absent','training_start_action_executed':False}
    run=run.resolve();backup_dir=backup_dir.resolve()
    if backup_dir.is_relative_to(run):raise BackupError('Backup output must remain outside RUN')
    backup_dir.mkdir(parents=True,exist_ok=True)
    with (backup_dir/'.backup.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return {'status':'noop','reason':'backup already running'}
        files,models,folds,skipped=collect(run)
        if not files:return {'status':'noop','reason':'No verified completed models/folds yet','skipped':skipped}
        if ops_dir is not None:
            ops_dir=Path(ops_dir)
            if ops_dir.is_symlink():raise BackupError('Symlink OPS root rejected')
            for name in ['startup_receipt.json','config.json','environment_receipt.json','preflight_report.json']:
                p=ops_dir/name
                if p.exists():
                    p=regular(ops_dir,name);read_json(p)
                    files['__ops__/'+name]={'path':p,'sha256':sha(p),'bytes':p.stat().st_size,'origin':'frozen_launch_receipt'}
        entries=[{'relative_path':name,'archive_path':name,'sha256':v['sha256'],'bytes':v['bytes'],'origin':v['origin']} for name,v in sorted(files.items())]
        fingerprint=canonical([{k:v for k,v in e.items() if k!='origin'} for e in entries])
        index=backup_dir/'latest.json'
        if index.exists():
            if index.is_symlink():raise BackupError('Symlink backup index rejected')
            old=read_json(index)
            if old.get('content_fingerprint')==fingerprint:
                if Path(old['archive_name']).name!=old['archive_name']:raise BackupError('Escaping archive index')
                p=backup_dir/old['archive_name'];checked=verify_archive(p)
                if checked['sha256']!=old['sha256']:raise BackupError('Previously verified archive changed')
                return {'status':'deduplicated','path':str(p),**checked,'completed_models':len(models),'completed_folds':folds,'skipped':skipped}
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_'+uuid.uuid4().hex[:8]
        final=backup_dir/('r5_snapshot_'+stamp+'.zip');partial=final.with_suffix('.zip.partial')
        manifest={'format':'r5_completed_snapshot_v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'run_directory_hint':str(run),'paths_relative_to':'RUN; __ops__ entries belong to the launch OPS receipt directory',
            'content_fingerprint':fingerprint,'files':entries,'completed_models':models,'completed_folds':folds,'skipped':skipped,
            'raw_data_vendor_weights_or_code_included':False,'in_progress_artifacts_excluded':True,
            'training_start_action_executed':False,'restoration':'Verify this ZIP first. Extract into a fresh RUN and a separate OPS directory; a model-only snapshot is not an automatically resumable production run.'}
        try:
            with partial.open('xb') as handle:
                with zipfile.ZipFile(handle,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
                    for item in entries:
                        source=files[item['archive_path']]['path'];h=hashlib.sha256();count=0
                        with source.open('rb') as src,z.open(item['archive_path'],'w',force_zip64=True) as dst:
                            for block in iter(lambda:src.read(2**20),b''):dst.write(block);h.update(block);count+=len(block)
                        if h.hexdigest()!=item['sha256'] or count!=item['bytes']:raise BackupError('Artifact changed during backup')
                    z.writestr(MANIFEST,encode(manifest))
                handle.flush();os.fsync(handle.fileno())
            receipt=verify_archive(partial)
            for value in files.values():
                if value['path'].is_symlink() or sha(value['path'])!=value['sha256']:raise BackupError('Artifact changed before snapshot commit')
            os.link(partial,final);partial.unlink()
            record={'archive_name':final.name,'sha256':receipt['sha256'],'content_fingerprint':fingerprint}
            temp=index.with_suffix('.json.tmp');temp.write_bytes(encode(record));temp.replace(index)
        except BaseException:partial.unlink(missing_ok=True);raise
        return {'status':'completed','path':str(final),**receipt,'completed_models':len(models),'completed_folds':folds,'skipped':skipped,'training_start_action_executed':False}

def self_test():
    with tempfile.TemporaryDirectory(prefix='r5_backup_fixture_') as tmp:
        base=Path(tmp);run=base/'run';out=base/'backups'
        assert snapshot(run,out)['status']=='noop'
        folder=run/'fold_00';model=folder/'models/bg_cnn';model.mkdir(parents=True)
        binding={'fold':0,'seed':42,'train_examples':8};config={'seed':42}
        prior={'training_binding':binding,'fit_split':'train'};prior['shared_prior_payload_sha256']=canonical(prior)
        (folder/'shared_background_prior.json').write_bytes(encode(prior))
        context={'config':config,'config_sha256':canonical(config),'training_binding':binding,
            'prior':{'filename':'shared_background_prior.json','file_sha256':sha(folder/'shared_background_prior.json'),'payload_sha256':prior['shared_prior_payload_sha256']}}
        (folder/'context.json').write_bytes(encode(context))
        (model/'best.pt').write_bytes(b'synthetic completed adapter')
        (model/'inner_predictions.npz').write_bytes(b'synthetic prediction fixture')
        (model/'epochs.jsonl').write_text('{"epoch":0,"inner_rmse":0.3}\n')
        identity={'binding':binding};receipt={'epoch':0,'examples':8,'clean_primary_once':True,'source_identity_sha256':canonical(identity)}
        (model/'dynamic_source_receipts.json').write_bytes(encode({'source_identity':identity,'completed_epochs':{'0':receipt}}))
        summary={'status':'completed','kind':'bg_cnn','fixture_only':False,'training_binding':binding,'checkpoint_sha256':sha(model/'best.pt'),
            'prediction_sha256':sha(model/'inner_predictions.npz'),'best_epoch':0,'dynamic_source_identity':identity,'dynamic_source_receipts_file':'dynamic_source_receipts.json'}
        (model/'summary.json').write_bytes(encode(summary));(folder/'unrecorded_in_progress.pt').write_bytes(b'partial')
        first=snapshot(run,out);assert first['status']=='completed' and first['completed_models']==1
        checked=verify_archive(first['path']);assert checked['verified']
        with zipfile.ZipFile(first['path']) as z:
            assert 'fold_00/unrecorded_in_progress.pt' not in z.namelist()
            restore=base/'restore'
            for item in json.loads(z.read(MANIFEST))['files']:
                p=restore/item['relative_path'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(item['archive_path']));assert sha(p)==item['sha256']
        assert snapshot(run,out)['status']=='deduplicated'
        # A completed fold adds its entire committed artifact ledger.
        fold_summary={'status':'completed','training_binding':binding,'outer_changed_policy':False,
            'artifact_sha256':{path.relative_to(folder).as_posix():sha(path) for path in folder.rglob('*') if path.is_file() and path.name!='unrecorded_in_progress.pt'}}
        (folder/'summary.json').write_bytes(encode(fold_summary))
        full=snapshot(run,out);assert full['completed_folds']==['fold_00'];verify_archive(full['path'])
        changed=json.loads(json.dumps(fold_summary));changed['artifact_sha256']['context.json']='0'*64
        (folder/'summary.json').write_bytes(encode(changed))
        try:snapshot(run,out)
        except BackupError:pass
        else:raise AssertionError('Invalid completed-fold ledger accepted')
        (folder/'summary.json').write_bytes(encode(fold_summary))
        (model/'best.pt').write_bytes(b'changed')
        try:snapshot(run,out)
        except BackupError:pass
        else:raise AssertionError('Changed checkpoint accepted')
        for bad in ['../bad.pt','/absolute.pt','pretrained_r5/model.pt','Test.csv','raw/a.json']:
            try:name_ok(bad)
            except BackupError:pass
            else:raise AssertionError('Unsafe path accepted')
    return {'passed':True,'synthetic_fixture_only':True,'checks':['absent noop','completed-model roundtrip','completed-fold artifact ledger and changed-ledger rejection','SHA and ZIP CRC','deduplication','unrecorded partial exclusion','changed checkpoint rejection','path/source/vendor/Test exclusion'],'server_or_training_access':False}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path);p.add_argument('--backup-dir',type=Path);p.add_argument('--ops-dir',type=Path)
    p.add_argument('--verify',type=Path);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    try:
        if a.self_test:r=self_test()
        elif a.verify:r=verify_archive(a.verify)
        elif a.run_dir and a.backup_dir:r=snapshot(a.run_dir,a.backup_dir,a.ops_dir)
        else:p.error('--run-dir/--backup-dir, --verify, or --self-test required')
        print(json.dumps(r,indent=2,allow_nan=False))
    except (BackupError,OSError,ValueError,KeyError,zipfile.BadZipFile) as e:
        print(json.dumps({'status':'failed','error':type(e).__name__,'message':str(e),'training_start_action_executed':False}));raise SystemExit(2)

if __name__=='__main__':main()
