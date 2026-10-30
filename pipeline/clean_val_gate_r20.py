"""Fail-closed, zero-training Clean-Val-R20 diagnostic adoption gate.

This is a stress-cohort comparison, not proof of public-test representativeness.
No score/label fitting or feature transformation is performed by this module.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib,json
import numpy as np

@dataclass(frozen=True)
class Policy:
 version:str='Clean-Val-R20_gate_v1'
 baseline:str='old_B1'
 axes:tuple=(('all','rmse'),('zero','rmse'),('zero','positive_bias'),
             ('weak','rmse'),('weak','positive_bias'),('HF_zero','rmse'))
 strict_improvement_slices:tuple=('all','zero','weak','HF_zero')
 tolerance:float=1e-6
 HF_threshold_db:float=12.
 weak_positive_upper:float=.10
 negative_selection:str='HF3..8k bump >=15dB AND LF<=200Hz energy fraction <=.65; source-admitted SA only'
 negative_fraction:float=.5
 missing_empty_or_nonfinite:str='reject'
 calibration_or_public_score_fitting:bool=False
 scope:str='New Clean-Val-R20 primary diagnostic gate; no automatic alteration of frozen historical pipelines'
 DEFAULT_NONNEGATIVE_BIAS_FLOOR:str='max(mean(pred-y),0); zero floor cannot be strictly reduced'

DEFAULT=Policy()
def policy_dict():return json.loads(json.dumps(asdict(DEFAULT)))
def policy_sha():return hashlib.sha256(json.dumps(policy_dict(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
def _vector(v,n=None,bounded=False):
 a=np.asarray(v,float)
 if a.ndim!=1 or not len(a) or (n is not None and len(a)!=n) or not np.isfinite(a).all():raise ValueError('Finite nonempty aligned vectors required')
 if bounded and ((a<0)|(a>1)).any():raise ValueError('Share outside [0,1]')
 return a
def evaluate(y,baseline,candidates,hf_db,weights=None):
 """Return all six checks and four strict-RMSE comparisons without fitting.

 An identity/reference is noninferior but is never called an improvement.
 Every required slice must exist, even if a subset duplicates another axis.
 """
 results={}
 for name,p in candidates.items():
  try:
   yy=_vector(y,bounded=True);bb=_vector(baseline,len(yy),True);pp=_vector(p,len(yy),True);hf=_vector(hf_db,len(yy))
   w=_vector(np.ones(len(yy)) if weights is None else weights,len(yy))
   if (w<0).any() or w.sum()<=0:raise ValueError('Invalid sample weights')
   masks={'all':np.ones(len(yy),bool),'zero':yy==0,'weak':(yy>0)&(yy<=DEFAULT.weak_positive_upper),'HF_zero':(yy==0)&(hf>=DEFAULT.HF_threshold_db)}
   checks=[]
   for sl,metric in DEFAULT.axes:
    ix=masks[sl]
    if not ix.any() or w[ix].sum()<=0:raise ValueError('Missing required slice '+sl)
    ww=w[ix]/w[ix].sum()
    def stat(v):
     e=v[ix]-yy[ix]
     return float(np.sqrt(ww@(e*e))) if metric=='rmse' else max(float(ww@e),0.)
    b=stat(bb);c=stat(pp);delta=c-b
    checks.append({'slice':sl,'metric':metric,'n':int(ix.sum()),'baseline':b,'candidate':c,'delta':delta,'noninferior':delta<=DEFAULT.tolerance,'strictly_improved':delta < -DEFAULT.tolerance})
   noninferior=all(x['noninferior'] for x in checks)
   improved=all(next(c['strictly_improved'] for c in checks if c['slice']==sl and c['metric']=='rmse') for sl in DEFAULT.strict_improvement_slices)
   results[name]={'valid':True,'error':None,'noninferior_all_axes':noninferior,'strict_four_slice_RMSE_improvement':improved,'adoption_pass':noninferior and improved,'failed_axes':sum(not c['noninferior'] for c in checks),'checks':checks,
     'HF_zero_equals_zero':bool(np.array_equal(masks['zero'],masks['HF_zero']))}
  except (ValueError,TypeError,KeyError) as e:results[name]={'valid':False,'error':str(e),'adoption_pass':False,'checks':[]}
 return {'policy':policy_dict(),'policy_sha256':policy_sha(),'results':results}
