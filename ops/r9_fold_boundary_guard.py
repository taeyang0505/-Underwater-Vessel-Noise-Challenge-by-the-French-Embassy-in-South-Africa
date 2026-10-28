"""External R9 operator guard: SIGSTOP only after Fold0's last model summary.

No runner/code/ledger mutation, SIGCONT, SIGKILL, restart or pod operation.
summary completion is a MODEL boundary, not proof the fold ledger finalized.
--self-test uses a fake process/signal backend and never accesses /proc.
"""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import struct
import sys
import tempfile
import time

KINDS=('bg_cnn','tfgrid_fdy_conformer','fdy','itransformer')
IN_CLOSE_WRITE=0x8
IN_MOVED_TO=0x80
IN_Q_OVERFLOW=0x4000
IN_IGNORED=0x8000
IN_DELETE_SELF=0x400
IN_MOVE_SELF=0x800


def utc():return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    tmp.replace(path)
def read(path):return json.loads(Path(path).read_text())


def parse_stat(text):
    right=text.rfind(')')
    if right<0:raise ValueError('Invalid proc stat')
    pid=int(text[:text.find('(')].strip());tail=text[right+2:].split()
    return {'pid':pid,'state':tail[0],'ppid':int(tail[1]),'pgid':int(tail[2]),'startticks':int(tail[19])}


class LinuxProcesses:
    def __init__(self):
        if not sys.platform.startswith('linux') or not hasattr(os,'pidfd_open') or not hasattr(signal,'pidfd_send_signal'):
            raise RuntimeError('Linux pidfd APIs required; no unsafe PID fallback')
        self.actions=[]
    def stat(self,pid):return parse_stat(Path(f'/proc/{pid}/stat').read_text())
    def owner(self):return {'pid':os.getpid(),'pgid':os.getpgrp(),'startticks':self.stat(os.getpid())['startticks']}
    def group(self,pgid):
        found=[]
        for path in Path('/proc').glob('[0-9]*/stat'):
            try:value=parse_stat(path.read_text())
            except (FileNotFoundError,ProcessLookupError):continue
            if value['pgid']==pgid:found.append(value)
        return found
    def threads(self,pid):
        return [parse_stat(path.read_text()) for path in Path(f'/proc/{pid}/task').glob('[0-9]*/stat')]
    def _pidfd_stop(self,pid,startticks,pgid=None):
        fd=os.pidfd_open(pid,0)
        try:
            current=self.stat(pid)
            if current['startticks']!=startticks or (pgid is not None and current['pgid']!=pgid):
                raise ValueError('PID identity changed immediately before signal')
            signal.pidfd_send_signal(fd,signal.SIGSTOP,None,0)
            self.actions.append({'signal':'SIGSTOP','scope':'pidfd','pid':pid,'startticks':startticks,'utc':utc()})
        finally:os.close(fd)
    def stop_group(self,pid,startticks,pgid):
        # First pin and stop the leader; its PID/PGID cannot normally be reused
        # between the identity check and group STOP while that leader is alive.
        self._pidfd_stop(pid,startticks,pgid)
        current=self.stat(pid)
        if current['startticks']!=startticks or current['pgid']!=pgid:raise ValueError('Stopped group leader identity differs')
        os.killpg(pgid,signal.SIGSTOP)
        self.actions.append({'signal':'SIGSTOP','scope':'process_group','pgid':pgid,'utc':utc()})
    def stop_supervisor(self,pid,startticks):self._pidfd_stop(pid,startticks)


class Inotify:
    def __init__(self,folder):
        if not sys.platform.startswith('linux'):raise RuntimeError('Production inotify requires Linux')
        libc=ctypes.CDLL(None,use_errno=True)
        libc.inotify_init1.argtypes=[ctypes.c_int];libc.inotify_init1.restype=ctypes.c_int
        libc.inotify_add_watch.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_uint32]
        libc.inotify_add_watch.restype=ctypes.c_int
        self.fd=libc.inotify_init1(os.O_NONBLOCK|os.O_CLOEXEC)
        if self.fd<0:raise OSError(ctypes.get_errno(),'inotify_init1 failed')
        mask=IN_MOVED_TO|IN_CLOSE_WRITE|IN_DELETE_SELF|IN_MOVE_SELF
        self.wd=libc.inotify_add_watch(self.fd,os.fsencode(folder),mask)
        if self.wd<0:
            os.close(self.fd);raise OSError(ctypes.get_errno(),'inotify_add_watch failed')
    def events(self,timeout=.1):
        if not select.select([self.fd],[],[],timeout)[0]:return []
        data=os.read(self.fd,65536);offset=0;result=[]
        while offset<len(data):
            wd,mask,cookie,size=struct.unpack_from('iIII',data,offset);offset+=16
            name=data[offset:offset+size].split(b'\0',1)[0].decode();offset+=size
            result.append({'wd':wd,'mask':mask,'cookie':cookie,'name':name})
        return result
    def close(self):os.close(self.fd)


@dataclass
class Settings:
    base:Path
    project:Path
    run:Path
    config:Path
    output:Path
    pid:int
    startticks:int
    pgid:int
    supervisor:int
    supervisor_startticks:int
    config_sha256:str
    data_hash:str
    stop_timeout:float=10.
    maximum_wait_seconds:float=7200.


class Guard:
    def __init__(self,settings,processes):
        self.s=settings;self.proc=processes;self.identity_sha=None;self.initial_group=[];self.claimed=False
        self.watch_dir=self.s.run/'fold_00/models/itransformer'
        self.summary=self.watch_dir/'summary.json'
        self.receipt={'format':'r9_fold_boundary_operator_guard_v1','status':'initializing',
            'started_utc':utc(),'operator_owner':self.proc.owner(),
            'expected':{'pid':self.s.pid,'startticks':self.s.startticks,'pgid':self.s.pgid,
                'supervisor':self.s.supervisor,'supervisor_startticks':self.s.supervisor_startticks,
                'config_sha256':self.s.config_sha256,'data_hash':self.s.data_hash},
            'watch_directory':str(self.watch_dir),'trigger':'summary.json atomic completion',
            'signals_allowed':['SIGSTOP'],'automatic_resume_allowed':False,
            'original_code_or_ledger_modified':False,'guard_source_sha256':sha(Path(__file__))}
    def save(self,status,**extra):
        self.receipt.update(status=status,updated_utc=utc(),signals_sent=self.proc.actions,**extra)
        atomic(self.s.output/'status.json',self.receipt)
        print(json.dumps({'status':status,'status_file':str(self.s.output/'status.json'),
                          'notification':extra.get('notification')}),flush=True)
    def future_models(self):return [str(self.s.run/f'fold_{fold:02d}/models') for fold in range(1,5) if (self.s.run/f'fold_{fold:02d}/models').exists()]
    def identities(self,*,running=False):
        a=self.proc.stat(self.s.pid);b=self.proc.stat(self.s.supervisor)
        if a['startticks']!=self.s.startticks or a['pgid']!=self.s.pgid or a['state'] in 'ZX':raise ValueError('Training PID/startticks/PGID mismatch')
        if b['startticks']!=self.s.supervisor_startticks or b['state'] in 'ZX':raise ValueError('Supervisor PID/startticks mismatch')
        if running and (a['state'] in 'Tt' or b['state'] in 'Tt'):raise ValueError('Already stopped process requires manual review, not another guard')
        owner=self.proc.owner()
        if owner['pid'] in [self.s.pid,self.s.supervisor] or owner['pgid']==self.s.pgid:raise ValueError('Watcher would stop itself')
        return a,b
    def verify_files(self,*,full_code):
        if sha(self.s.config)!=self.s.config_sha256:raise ValueError('Frozen config bytes changed')
        identity_file=self.s.run/'execution_identity.json';identity=read(identity_file)
        if (identity.get('format')!='r9_execution_identity_v1' or identity.get('pid')!=self.s.pid
                or identity.get('fixture_only') is not False or identity.get('resume_allowed') is not False
                or identity.get('config_file_sha256')!=self.s.config_sha256
                or identity.get('data_bundle_hash')!=self.s.data_hash
                or identity.get('config_sha256')!=canonical(read(self.s.config))):
            raise ValueError('Execution identity/config/data binding differs')
        if self.identity_sha is not None and sha(identity_file)!=self.identity_sha:raise ValueError('Execution identity changed after arm')
        config=read(self.s.config)
        if self.s.run.resolve()!=(self.s.project/config['run_output']).resolve():raise ValueError('Guard run path differs from the frozen config')
        bundle=self.s.project/config['data_output'];ready=read(bundle/'readiness.json')
        if ready.get('ready_to_train') is not True or ready.get('source_hash')!=self.s.data_hash or sha(bundle/'manifest.json')!=ready.get('manifest_sha256'):
            raise ValueError('Prepared bundle readiness/source identity differs')
        closure=identity.get('code_sha256')
        if not isinstance(closure,dict) or not closure:raise ValueError('Source closure absent')
        if full_code:
            for name,digest in closure.items():
                relative=Path(name);path=self.s.project/relative
                if relative.is_absolute() or '..' in relative.parts or path.is_symlink() or sha(path)!=digest:raise ValueError('Frozen source closure changed:'+name)
        return identity
    def arm(self):
        created_claim=False
        try:
            self.s.output.mkdir(parents=True,exist_ok=True)
            with (self.s.output/'claim.json').open('x') as stream:
                self.claimed=True;created_claim=True;json.dump(self.receipt,stream,indent=2)
            self.identities(running=True);self.verify_files(full_code=True)
            if not self.watch_dir.is_dir() or self.watch_dir.is_symlink():raise ValueError('Active model directory missing/unsafe')
            if self.summary.exists():raise ValueError('Summary already present at initial arm; no late automatic signal')
            if self.future_models():raise ValueError('A later fold model directory already exists')
            self.identity_sha=sha(self.s.run/'execution_identity.json')
            self.initial_group=self.proc.group(self.s.pgid)
            self.save('preflight_passed',identity_sha256=self.identity_sha,initial_group=self.initial_group)
        except BaseException as error:
            if created_claim:self.save('failed_before_pause',error_type=type(error).__name__,error=str(error),notification='Arming failed; no signals sent. Manual operator review required.')
            raise
    def health_check(self,elapsed):
        try:
            if elapsed>self.s.maximum_wait_seconds:raise TimeoutError('Bounded boundary wait expired')
            self.identities(running=True)
            if (self.watch_dir/'FAILED.json').exists() or (self.s.run/'FAILED.json').exists():raise RuntimeError('Training failure receipt detected while armed')
            if self.future_models():raise RuntimeError('A later-fold model appeared before a verified boundary event')
            self.save('armed',last_health_utc=utc(),wait_elapsed_seconds=elapsed)
        except BaseException as error:
            self.save('failed_before_pause',error_type=type(error).__name__,error=str(error),notification='Armed guard health check failed; no automatic signal/restart. Operator must inspect immediately.')
            raise
    def check_stopped(self):
        last=None;deadline=time.monotonic()+self.s.stop_timeout
        while True:
            a,b=self.identities();members=self.proc.group(self.s.pgid)
            if not any(v['pid']==self.s.pid for v in members):raise ValueError('Training group leader disappeared')
            checks=[]
            for member in members+[b]:
                if member['state'] in 'ZX':continue
                threads=self.proc.threads(member['pid'])
                if not threads:raise ValueError('No task status for live process')
                checks.extend(threads)
            last={'group_processes':members,'supervisor':b,'tasks':checks}
            if checks and all(v['state']=='T' for v in checks):return last
            if time.monotonic()>deadline:raise TimeoutError('Not every live process/task reached SIGSTOP T state:'+str(last))
            time.sleep(.05)
    def completed_models(self,identity):
        result={}
        for kind in KINDS:
            folder=self.s.run/'fold_00/models'/kind;summary=read(folder/'summary.json')
            if (summary.get('status')!='completed' or summary.get('kind')!=kind
                    or summary.get('epochs_run')!=100 or summary.get('fixture_only') is not False
                    or summary.get('stop_reason')!='completed_fixed_epochs'):
                raise ValueError('All four complete nonfixture100epoch model summaries required:'+kind)
            source=summary.get('code_sha256')
            if not isinstance(source,dict) or not source or any(identity['code_sha256'].get(n)!=h for n,h in source.items()):raise ValueError('Model source closure binding differs:'+kind)
            artifacts=summary.get('artifact_sha256')
            if not isinstance(artifacts,dict) or not artifacts:raise ValueError('Model artifacts absent:'+kind)
            for name,digest in artifacts.items():
                relative=Path(name);path=folder/relative
                if relative.is_absolute() or '..' in relative.parts or path.is_symlink() or sha(path)!=digest:raise ValueError('Completed model artifact changed:'+kind+'/'+name)
            result[kind]={'summary_sha256':sha(folder/'summary.json'),'epochs_run':100,
                          'artifact_count':len(artifacts),'all_artifact_hashes_verified':True}
        return result
    def trigger(self,reason):
        try:
            # No summary is read before the pause. Fast identity/file checks
            # precede signals; full summary/artifact verification follows STOP.
            self.identities();self.verify_files(full_code=False)
            self.proc.stop_group(self.s.pid,self.s.startticks,self.s.pgid)
            self.identities();self.proc.stop_supervisor(self.s.supervisor,self.s.supervisor_startticks)
            stopped=self.check_stopped()
            identity=self.verify_files(full_code=True);models=self.completed_models(identity)
            if self.future_models():raise ValueError('Later-fold model directory appeared before pause; boundary race detected')
            ledger_path=self.s.run/'fold_00/artifact_ledger.json'
            ledger=read(ledger_path) if ledger_path.exists() else {}
            required=['train/'+kind for kind in KINDS]+['neural_completed']
            ledger_complete=all(ledger.get('steps',{}).get(k,{}).get('status')=='completed' for k in required)
            self.save('paused_after_last_model_summary',trigger_reason=reason,stopped_processes=stopped,
                models=models,Fold0_all_four_model_training_complete=True,
                Fold0_original_ledger_complete=ledger_complete,
                Fold0_neural_completed_receipt_exists=(self.s.run/'fold_00/neural_completed.json').exists(),
                later_fold_model_directories=[],notification='Training group and supervisor SIGSTOP verified; do not auto-resume. Fold ledger may be incomplete; originals preserved.')
            return self.receipt
        except BaseException as error:
            status='failed_after_pause' if self.proc.actions else 'failed_before_pause'
            self.save(status,error_type=type(error).__name__,error=str(error),
                notification='Operator attention required. Never automatically CONT/KILL/retry; any stopped processes remain stopped.')
            raise


def watch(guard):
    guard.arm()
    watcher=None
    try:
        watcher=Inotify(guard.watch_dir)
        guard.save('armed',watch_registered=True)
        started=time.monotonic();last_health=started
        # Register first, then check existence: closes the rename-before-watch
        # race after initial preflight. Poll existence also covers lost events.
        while True:
            if guard.summary.exists():return guard.trigger('summary_exists_after_watch_registration')
            for event in watcher.events(.1):
                if event['mask']&(IN_Q_OVERFLOW|IN_IGNORED|IN_DELETE_SELF|IN_MOVE_SELF):raise RuntimeError('Inotify watch lost/overflow; no silent continuation')
                if event['name']=='summary.json' and event['mask']&(IN_MOVED_TO|IN_CLOSE_WRITE):return guard.trigger('inotify_summary_completion')
            now=time.monotonic()
            if now-started>guard.s.maximum_wait_seconds:raise TimeoutError('Bounded boundary wait expired; no automatic restart or signal')
            if now-last_health>=5.:
                guard.health_check(now-started)
                last_health=now
    except BaseException as error:
        if not guard.receipt['status'].startswith('failed'):
            guard.save('failed_before_pause',error_type=type(error).__name__,error=str(error),notification='Guard failed; no automatic signals/restart. Operator must inspect running process immediately.')
        raise
    finally:
        if watcher is not None:watcher.close()


def self_test():
    class FakeProcesses:
        def __init__(self):
            self.actions=[];self.rows={2257:{'pid':2257,'startticks':86168117,'pgid':2257,'state':'R'},2256:{'pid':2256,'startticks':86168115,'pgid':2200,'state':'S'},3001:{'pid':3001,'startticks':86168122,'pgid':2257,'state':'S'}}
        def owner(self):return {'pid':99999,'pgid':99999,'startticks':1,'fixture_only':True}
        def stat(self,pid):return dict(self.rows[pid])
        def group(self,pgid):return [dict(v) for v in self.rows.values() if v['pgid']==pgid]
        def threads(self,pid):return [dict(self.rows[pid])]
        def stop_group(self,pid,startticks,pgid):
            assert self.rows[pid]['startticks']==startticks
            self.actions.append({'signal':'SIGSTOP','scope':'fake_group','pgid':pgid})
            for v in self.rows.values():
                if v['pgid']==pgid:v['state']='T'
        def stop_supervisor(self,pid,startticks):
            assert self.rows[pid]['startticks']==startticks
            self.actions.append({'signal':'SIGSTOP','scope':'fake_supervisor','pid':pid});self.rows[pid]['state']='T'
    outcomes={}
    def setup(base):
        project=base/'project';run=project/'models/run';watchdir=run/'fold_00/models/itransformer';watchdir.mkdir(parents=True)
        config=project/'configs/r9.json';config.parent.mkdir();atomic(config,{'data_output':'data/bundle','run_output':'models/run'})
        bundle=project/'data/bundle';bundle.mkdir(parents=True);atomic(bundle/'manifest.json',{'source_hash':'dataset'})
        atomic(bundle/'readiness.json',{'ready_to_train':True,'source_hash':'dataset','manifest_sha256':sha(bundle/'manifest.json')})
        code=project/'pipeline/frozen.py';code.parent.mkdir();code.write_text('frozen = True\n');closure={'pipeline/frozen.py':sha(code)}
        atomic(run/'execution_identity.json',{'format':'r9_execution_identity_v1','pid':2257,'fixture_only':False,'resume_allowed':False,'config_file_sha256':sha(config),'config_sha256':canonical(read(config)),'data_bundle_hash':'dataset','code_sha256':closure})
        for kind in KINDS:
            folder=run/'fold_00/models'/kind;folder.mkdir(parents=True,exist_ok=True);(folder/'weight.bin').write_bytes(b'fixture-not-model-weights')
            if kind!='itransformer':atomic(folder/'summary.json',{'kind':kind,'status':'completed','epochs_run':100,'fixture_only':False,'stop_reason':'completed_fixed_epochs','code_sha256':closure,'artifact_sha256':{'weight.bin':sha(folder/'weight.bin')}})
        s=Settings(base,project,run,config,base/'ops/guard',2257,86168117,2257,2256,86168115,sha(config),'dataset',.1)
        return Guard(s,FakeProcesses()),closure
    with tempfile.TemporaryDirectory(prefix='r9_boundary_guard_FAKE_') as temp:
        root=Path(temp)
        for case in ['success','pid_reused_arm','summary_exists_arm','later_fold_exists_arm','config_tamper_trigger','code_tamper_after_pause','later_fold_race','incomplete_summary_after_pause','duplicate_claim','driver_exit_health','FAILED_health','timeout_health']:
            guard,closure=setup(root/case)
            def finish():atomic(guard.summary,{'kind':'itransformer','status':'completed','epochs_run':100,'fixture_only':False,'stop_reason':'completed_fixed_epochs','code_sha256':closure,'artifact_sha256':{'weight.bin':sha(guard.watch_dir/'weight.bin')}})
            try:
                if case=='pid_reused_arm':guard.proc.rows[2257]['startticks']+=1
                if case=='summary_exists_arm':finish()
                if case=='later_fold_exists_arm':(guard.s.run/'fold_01/models').mkdir(parents=True)
                guard.arm()
                if case=='duplicate_claim':guard.arm()
                if case=='driver_exit_health':guard.proc.rows[2257]['state']='Z';guard.health_check(5.)
                if case=='FAILED_health':atomic(guard.watch_dir/'FAILED.json',{'error':'fixture'});guard.health_check(5.)
                if case=='timeout_health':guard.health_check(guard.s.maximum_wait_seconds+1.)
                finish()
                if case=='config_tamper_trigger':guard.s.config.write_text('{}')
                if case=='code_tamper_after_pause':(guard.s.project/'pipeline/frozen.py').write_text('changed')
                if case=='later_fold_race':(guard.s.run/'fold_01/models').mkdir(parents=True)
                if case=='incomplete_summary_after_pause':value=read(guard.summary);value['epochs_run']=99;atomic(guard.summary,value)
                guard.trigger('fake_event')
                assert case=='success' and guard.receipt['Fold0_original_ledger_complete'] is False
                assert all(a['signal']=='SIGSTOP' for a in guard.proc.actions)
                outcomes[case]={'passed':True,'signals':len(guard.proc.actions),'ledger_incomplete_honestly_recorded':True}
            except (ValueError,FileExistsError,RuntimeError,TimeoutError):
                assert case!='success'
                if case in ['code_tamper_after_pause','later_fold_race','incomplete_summary_after_pause']:
                    assert guard.receipt['status']=='failed_after_pause' and len(guard.proc.actions)==2
                    assert guard.proc.rows[2257]['state']=='T' and guard.proc.rows[2256]['state']=='T'
                else:assert not guard.proc.actions
                outcomes[case]={'passed':True,'signals':len(guard.proc.actions),'status':guard.receipt['status']}
    return {'status':'passed','fixture_only':True,'real_proc_or_PID_accessed':False,'real_signals_sent':False,'cases':outcomes,'guard_source_sha256':sha(Path(__file__))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--self-test',action='store_true');p.add_argument('--test-output',type=Path)
    for name in ['base','project','run','config','output']:p.add_argument('--'+name,type=Path)
    for name in ['pid','startticks','pgid','supervisor','supervisor-startticks']:p.add_argument('--'+name,type=int)
    p.add_argument('--config-sha256');p.add_argument('--data-hash');p.add_argument('--stop-timeout',type=float,default=10.)
    p.add_argument('--maximum-wait-seconds',type=float,default=7200.)
    a=p.parse_args()
    if a.self_test:
        result=self_test()
        if a.test_output:atomic(a.test_output,result)
        print(json.dumps(result));return
    required=['base','project','run','config','output','pid','startticks','pgid','supervisor','supervisor_startticks','config_sha256','data_hash']
    if any(getattr(a,k) is None for k in required):p.error('All explicit production paths/process/config/data identities are required')
    if a.pid!=a.pgid or min(a.pid,a.startticks,a.supervisor,a.supervisor_startticks)<1 or not 0<a.stop_timeout<=60 or not 0<a.maximum_wait_seconds<=7200:
        p.error('Positive pinned group leader identities and bounded stop/wait timeouts required')
    s=Settings(a.base.resolve(),a.project.resolve(),a.run.resolve(),a.config.resolve(),a.output.resolve(),a.pid,a.startticks,a.pgid,a.supervisor,a.supervisor_startticks,a.config_sha256,a.data_hash,a.stop_timeout,a.maximum_wait_seconds)
    if s.project!=s.base/'project' or s.project not in s.run.parents or s.project not in s.config.parents or s.base/'ops' not in s.output.parents:
        p.error('Run/config must be inside approved project and guard output inside base/ops')
    guard=Guard(s,LinuxProcesses())
    try:watch(guard)
    except BaseException as error:
        if not guard.receipt['status'].startswith('failed'):
            # A duplicate claim belongs to another guard; never overwrite it.
            if not guard.claimed:raise
            guard.save('failed_before_pause',error_type=type(error).__name__,error=str(error),notification='Arming failed: no automatic signals')
        raise


if __name__=='__main__':main()
