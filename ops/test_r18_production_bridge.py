#!/usr/bin/env python3
"""Bridge-only fixtures; no model, report or Test data is opened."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

SPEC=importlib.util.spec_from_file_location('r18_bridge',Path(__file__).with_name('finish_r18_production_bridge.py'))
bridge_module=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge_module)


class BridgeSnapshotTests(unittest.TestCase):
    def test_previous_stat_get_race_is_reproduced(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'progress.json'
            p.write_text(json.dumps({'revision':0}))
            stat_sha=hashlib.sha256(p.read_bytes()).hexdigest()
            q=p.with_suffix('.tmp');q.write_text(json.dumps({'revision':1}));q.replace(p)
            self.assertNotEqual(stat_sha,hashlib.sha256(p.read_bytes()).hexdigest())

    def test_single_rpc_snapshot_under_continuous_atomic_updates(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'progress.json'
            p.write_text(json.dumps({'revision':0,'repeated_revision':[0]*20}))
            stop=threading.Event();updates=[0]
            def writer():
                while not stop.is_set():
                    updates[0]+=1;r=updates[0]
                    q=p.with_suffix('.tmp')
                    q.write_text(json.dumps({'revision':r,'repeated_revision':[r]*20}))
                    q.replace(p);time.sleep(0.0005)
            thread=threading.Thread(target=writer);thread.start()
            obj=object.__new__(bridge_module.Bridge);obj.remote=tmp
            calls=[]
            def rpc(code,*args,**kwargs):
                calls.append((code,args))
                result=subprocess.run([sys.executable,'-c',code,*map(str,args)],check=True,capture_output=True)
                return json.loads(result.stdout)
            obj.rpc=rpc
            try:
                for _ in range(40):
                    before=len(calls);result=obj.remote_json('progress.json')
                    self.assertEqual(len(calls)-before,1)
                    self.assertEqual(result['repeated_revision'],[result['revision']]*20)
                self.assertGreater(updates[0],40)
            finally:stop.set();thread.join()

    def test_missing_snapshot_and_unsafe_path(self):
        with tempfile.TemporaryDirectory()as tmp:
            result=subprocess.run([sys.executable,'-c',bridge_module.REMOTE_JSON_SNAPSHOT,tmp,'missing.json'],check=True,capture_output=True)
            self.assertEqual(json.loads(result.stdout),{'exists':False})
            result=subprocess.run([sys.executable,'-c',bridge_module.REMOTE_JSON_SNAPSHOT,tmp,'../outside.json'],capture_output=True)
            self.assertNotEqual(result.returncode,0)

    def test_immutable_npz_overwrite_still_rejected(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'predictions.npz';p.write_bytes(b'local immutable prediction')
            other=b'different remote prediction'
            digest=hashlib.sha256(other).hexdigest()
            obj=object.__new__(bridge_module.Bridge);obj.root=Path(tmp);obj.remote='/remote'
            def rpc(code,*args,**kwargs):
                return {'exists':True,'sha256':digest,'bytes':len(other)}if args[1]=='stat'else other
            obj.rpc=rpc
            with self.assertRaises(FileExistsError):obj.download('predictions.npz')
            self.assertEqual(p.read_bytes(),b'local immutable prediction')


if __name__=='__main__':unittest.main(verbosity=2)
