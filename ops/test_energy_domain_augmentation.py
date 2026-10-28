"""Numeric physical cases; no model training or heldout inputs."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from ops.energy_domain_augmentation import augment_components

def make(b,a,mask=None,dtype=np.float64):
    b=np.asarray(b,float);a=np.asarray(a,float);e=b+a
    o=np.ones_like(e,bool) if mask is None else np.asarray(mask,bool)
    db=np.where(o,10*np.log10(e)+87.25,np.nan).astype(dtype)
    q=np.where(o,a/e,0).astype(dtype)
    return db,q

def share(db,q):
    o=np.isfinite(db);e=np.where(o,10**((db-np.nanmax(db,axis=(-2,-1),keepdims=True))/10),0)
    return (q*e).sum((-2,-1))/e.sum((-2,-1))

def check(data):
    assert np.allclose(share(data['db'],data['cell_share']),data['share'],atol=3e-6)
    assert np.array_equal(np.asarray(data['presence']),np.asarray(data['share']>0,dtype=np.float32))

def run():
    checks=[];db,q=make([[9.,1.]],[[1.,9.]])
    # With one minute the common-filter target has a closed-form answer.
    result,a=augment_components(db,q,11,mode='common_response',probability=1,strength_db=4)
    rng=np.random.default_rng(11);rng.random(1);tilt=rng.uniform(-4,4,(1,2))[0,0]
    f=10**(np.array([-tilt,tilt])/10);expected=(f[0]+9*f[1])/(10*f.sum())
    assert np.isclose(result['share'],expected,atol=1e-7) and not np.isclose(expected,.5)
    assert np.array_equal(result['cell_share'],q);check(result)
    assert np.allclose(result['db']-db,[-tilt,tilt],atol=1e-12)
    checks.append('common q invariant, weighted share changes by closed-form value, absolute dB level restored')
    result,a=augment_components(db,q,11,mode='reference_smooth',probability=1,strength_db=12)
    assert np.isclose(result['share'],.5,atol=1e-7) and not np.allclose(result['cell_share'],q)
    assert a['max_reference_share_restore_error']<1e-12;check(result)
    checks.append('reference-only response changes q and restores original global share before caps')
    b=np.ones((15,25));added=np.zeros_like(b);added[0]=60
    db,q=make(b,added);result,a=augment_components(db,q,7,probability=1,strength_db=0)
    assert np.isclose(result['share'],.25+.73/15,atol=1e-7) and a['support_cap_count']==1;check(result)
    checks.append('one-minute short-support target ceiling applied from known components')
    added=np.ones_like(b);added[0]=1e6;db,q=make(b,added)
    result,a=augment_components(db,q,7,probability=1,strength_db=0)
    assert a['minute_snr_cap_count']==1 and a['max_minute_snr_db_after']<=20+1e-8 and result['share']>0;check(result)
    checks.append('60 dB native-support spike limited to 20 dB and positive target recomputed')
    # q32 at a true 20 dB boundary has a small legitimate reconstruction drift.
    db,q=make(b,b*100,dtype=np.float32);result,a=augment_components(db,q,3,mode='common_response',probability=1,strength_db=4)
    assert np.array_equal(result['cell_share'],q) and a['max_minute_snr_db_after']<20.001;check(result)
    checks.append('half-ULP allowance accepts a legitimate float32 20 dB input without changing q')
    # A common response may need weakening to avoid a new support violation.
    b=np.ones((15,25));added=np.zeros_like(b);added[0,:4]=28
    db,q=make(b,added);result,a=augment_components(db,q,10,mode='common_response',probability=1,strength_db=4)
    assert a['common_backtrack_count']==1
    assert result['share']<=.25+.73/15+1e-7 and np.array_equal(result['cell_share'],q);check(result)
    checks.append('common smooth response preserves q and does not introduce a short-support ceiling breach')
    # Existing r1 multi-short exceptions must not be silently relabelled.
    added=np.zeros_like(b);added[0]=10;added[1]=2
    db,q=make(b,added);assert share(db,q)>.25+.73*2/15
    result,a=augment_components(db,q,4,mode='common_response',probability=1,strength_db=4)
    assert a['preexisting_support_cap_violation_count']==1 and a['preexisting_support_cap_identity_count']==1
    assert np.array_equal(result['db'],db,equal_nan=True) and np.array_equal(result['cell_share'],q)
    checks.append('preexisting multi-short support exception stays identity with explicit count')
    mask=np.ones((15,25),bool);mask[2:4]=False;mask[6,3]=False
    x0,q0=make(b,np.zeros_like(b),mask,dtype=np.float32);x1,q1=make(b,b*.4,mask,dtype=np.float32)
    x=np.stack([x0,x1]);qq=np.stack([q0,q1]);original_x=x.copy();original_q=qq.copy()
    for mode in ['reference_smooth','common_response','off']:
        first,a=augment_components(x,qq,22,mode=mode,probability=1,strength_db=4)
        second,c=augment_components(x,qq,22,mode=mode,probability=1,strength_db=4)
        for k in first:assert np.array_equal(first[k],second[k],equal_nan=True)
        assert np.array_equal(np.isnan(first['db']),np.isnan(x)) and first['share'][0]==0 and first['presence'][0]==0
        assert a['curve_abs_max_db']<=4+1e-10;check(first)
        if mode=='off':assert np.array_equal(first['db'],x,equal_nan=True) and np.array_equal(first['cell_share'],qq)
    assert np.array_equal(x,original_x,equal_nan=True) and np.array_equal(qq,original_q)
    untouched,a=augment_components(x,qq,9,probability=0)
    assert np.array_equal(untouched['db'],x,equal_nan=True) and np.array_equal(untouched['cell_share'],qq)
    checks.append('deterministic replay, exact masks, zero preservation, off/probability-zero identity, no source mutation')
    db,q=make([[1,1]],[[1e6,1e6]])
    try:augment_components(db,q,3,mode='common_response',probability=1)
    except ValueError:checks.append('impossible q-preserving repair of a genuinely invalid 60 dB input is rejected')
    else:raise AssertionError('Invalid common-response input accepted')
    return {'passed':True,'checks':checks,'actual_training_executed':False,'outer_or_Test_accessed':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path);args=parser.parse_args();r=run()
    r['module_sha256']=hashlib.sha256(Path(__file__).with_name('energy_domain_augmentation.py').read_bytes()).hexdigest()
    if args.report:args.report.write_text(json.dumps(r,indent=2))
    print(json.dumps(r,indent=2))
