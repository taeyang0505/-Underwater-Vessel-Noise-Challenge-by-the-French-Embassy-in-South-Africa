"""Requeue a DEAD, never-started Fold1..4 waiter after verified parent recovery.

Preserves old waiter receipts in the recovery ledger. Does not restart any
partial model; original frozen continuation code and its PLAN remain unchanged.
"""
from pathlib import Path
import fcntl,json,os,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ops import run_r16_multisupport_fivefold as controller
from pipeline.r16_multisupport_data import read,write,sha

def main():
 ops=controller.OPS;recovery=ROOT/'ops/r16_multisupport_resume_20261028_r1'
 receipt=read(recovery/'VERIFIED.json')
 if receipt['status']!='verified' or read(ROOT/'ops/r16_multisupport_full_r1/state.json')['phase']!='neural':raise ValueError('Verified parent recovery required')
 if controller.MODELS.exists() or (ops/'FAILED.json').exists() or (ops/'FOLD0_HANDOFF.json').exists():raise ValueError('Only a never-started continuation waiter can be requeued')
 with (ops/'writer.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  old=read(ops/'STARTED.json')
  try:os.kill(old['PID'],0)
  except ProcessLookupError:pass
  else:raise RuntimeError('Old waiter is alive; do not duplicate')
  archive=recovery/'old_fivefold_waiter';archive.mkdir()
  for name in ('STARTED.json','state.json'):
   source=ops/name;target=archive/name
   if source.exists():source.rename(target)
  write(recovery/'FIVEFOLD_REQUEUE.json',{'old_waiter':old,'archive':str(archive.relative_to(ROOT)),
   'original_controller_sha256':sha(ROOT/'ops/run_r16_multisupport_fivefold.py'),
   'PLAN_sha256':sha(ops/'PLAN.json'),'runtime':receipt['runtime'],
   'training_started_before_first_CSV':False,'model_or_recipe_changed':False,'automatic_retry':False})
 # Interpreter relocation only; run_stage(...mps=True) still dispatches
 # models whose frozen device settings explicitly select MPS (or CPU iTrans).
 controller.MPS=controller.CPU
 controller.supervise()

if __name__=='__main__':main()
