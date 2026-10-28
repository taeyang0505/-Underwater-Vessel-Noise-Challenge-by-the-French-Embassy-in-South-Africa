"""Independent numeric fixtures for R6 peak-SNR synthesis. No real data IO."""
from pathlib import Path
import hashlib,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from pipeline.robust_sampler_r6 import synthesize,energy_weights,rounding_share_gap_bound,FAMILIES

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rejects(fn):
    try:fn()
    except (ValueError,TypeError):return
    raise AssertionError('Invalid/unsupported numeric request accepted')

def check_identity(result,original_mask):
    x=result['unrounded_db'];rounded=result['db'];q=result['cell_share'];y=result['share'];recipe=result['recipe']
    assert x.shape==rounded.shape==q.shape==(15,25) and rounded.dtype==np.float32 and x.dtype==q.dtype==np.float64
    assert np.array_equal(np.isnan(x),original_mask) and np.array_equal(np.isnan(rounded),original_mask) and (q[original_mask]==0).all()
    # Reconstruct energy independently, without the generator's relative peak.
    e=np.where(~original_mask,10**(x/10),0);recovered=float((e*q).sum()/e.sum());assert abs(recovered-y)<1e-11
    er=np.where(~original_mask,10**(rounded.astype(float)/10),0);rounded_y=float((er*q).sum()/er.sum());gap=abs(rounded_y-y)
    assert gap<=recipe['rounding_share_gap_theory_bound']+1e-11 and abs(gap-recipe['rounded_weight_vs_raw_target_gap'])<1e-11
    assert result['presence']==int(y>0) and 0<=y<1 and ((q>=0)&(q<=1)).all()
    assert not recipe['share_cap_or_high_share_deletion'] and recipe['requested_share'] is None
    assert not recipe['trajectory']['waveform_repeated'] and not recipe['trajectory']['timewarp_applied'] and not recipe['trajectory']['window_boundary_fade_forced']
    if not recipe['zero']:
        expected=10**(recipe['snr_db_requested']/10)*recipe['background_observed_broadband_median_peak_relative']/recipe['reference_full_trajectory_minute_broadband_peak']
        assert np.isclose(recipe['alpha_peak_relative_background_units'],expected,atol=1e-12,rtol=1e-12)
        assert abs(recipe['snr_db_actual_full_reference_peak']-recipe['snr_db_requested'])<1e-10
    return gap,recipe['rounding_share_gap_theory_bound']

def run():
    base=np.full((15,25),40,dtype='float64');reference=np.ones((600,25),dtype='float64');frozen_base=base.copy();frozen_reference=reference.copy();checks=[];maximum_gap=0.;maximum_bound=0.;cases=0
    # Complete full-window constant powers give analytic, independently known y.
    sensitivity=[]
    for snr in [-20.,-10.,0.,10.,20.]:
        out=synthesize(base,np.ones((450,25)),42,'minute_first',0,snr,False,lowpass_probability=0);factor=10**(snr/10);assert abs(out['share']-factor/(1+factor))<1e-12;check_identity(out,np.isnan(base))
        sensitivity.append({'snr_db':snr,'share':out['share'],'scope':'constant-power whole15min numeric sensitivity only, not real-data representative'})
        if snr==20:assert out['share']>.95
    checks.append('known full-window CPA-SNR y; high-share >.95 retained')
    short_native=synthesize(base,np.ones((60,25)),9,'minute_first',0,0.,False,lowpass_probability=0)
    assert abs(short_native['share']-2/17)<1e-12
    checks.append('same peak-SNR produces duration-dependent share, not requested-u inversion')
    # A complete minute followed by a giant incomplete fragment excludes that
    # fragment in minute_first; it is not padded, repeated or called a minute.
    tail=np.ones((31,25));tail[-1]=1e6;out=synthesize(base,tail,4,'minute_first',0,0.,False,lowpass_probability=0)
    assert out['recipe']['trajectory']['native_incomplete_tail_segments_excluded']==1 and out['recipe']['reference_full_trajectory_minute_broadband_peak']==25
    checks.append('minute-first uses complete native60s only')
    for family in FAMILIES:
        for seed in range(12):
            rng=np.random.default_rng(seed);b=rng.uniform(60,110,(15,25));r=np.exp(rng.normal(-2,1,(600,25)));b[2:4]=np.nan;b[:,7]=np.nan
            result=synthesize(b,r,seed,family,12,None,False);gap,bound=check_identity(result,np.isnan(b));maximum_gap=max(maximum_gap,gap);maximum_bound=max(maximum_bound,bound);cases+=1
            assert -10<=result['recipe']['snr_db_requested']<=10 and np.max(abs(np.asarray(result['recipe']['reference_eq_db'])))<=12+1e-10
            again=synthesize(b,r,seed,family,12,None,False)
            for name in ['db','unrounded_db','cell_share']:assert np.array_equal(result[name],again[name],equal_nan=True)
            assert result['recipe']==again['recipe']
    checks.append('all4 families: masked energy/q/y, EQ bound, default SNR, deterministic replay')
    # Root's correction: CPA outside the query must never be rescaled to make
    # its visible tail reach the requested SNR. Alpha is placement-independent.
    cpa=np.ones((600,25));cpa[:30]*=1000;alphas=[];outside=[]
    for seed in range(8):
        result=synthesize(base,cpa,seed,'edge_straddling_left',0,0.,False,lowpass_probability=0);check_identity(result,np.isnan(base));rec=result['recipe'];alphas.append(rec['alpha_peak_relative_background_units'])
        if rec['trajectory']['source_visible_begin_2s']>=30:
            assert result['share']<.01 and rec['snr_db_actual_observed_window_peak']<-20;outside.append(seed)
    assert outside and np.allclose(alphas,.001,atol=1e-15,rtol=0)
    for side in ['left','right','both']:
        result=synthesize(base,reference,12,'edge_straddling_'+side,0,0.,False,lowpass_probability=0);rec=result['recipe']['trajectory'];assert rec['left_straddles'] or rec['right_straddles']
        if side=='both':assert rec['left_straddles'] and rec['right_straddles'] and rec['left_window_edge_reference_power']>0 and rec['right_window_edge_reference_power']>0
    checks.append('edge left/right/both; outside CPA has no visible-tail renormalization or forced-zero boundary')
    result=synthesize(base,reference,2,'short_passage',12,0.,False,lowpass_probability=1)
    assert result['recipe']['channel']['lowpass_applied'] and max(abs(np.array(result['recipe']['reference_eq_db'])))<=12+1e-10
    result=synthesize(base,reference,2,'short_passage',0,0.,False,lowpass_probability=1)
    assert not result['recipe']['channel']['enabled'] and not np.any(result['recipe']['reference_eq_db'])
    checks.append('lowpass probability override; combined EQ bound; EQ0 ablation off')
    # The caller alone determines exact zero count; zero accepts no reference.
    results=[synthesize(base,None,i,'minute_first',0,None,True) if i<20 else synthesize(base,np.ones((450,25)),i,'minute_first',0,0.,False) for i in range(100)]
    assert sum(v['presence']==0 for v in results)==20
    for result in results[:20]:assert result['share']==0 and not result['cell_share'].any();check_identity(result,np.isnan(base))
    checks.append('caller-owned exact20/100 zero controls; no reference/label surrogate required')
    # Distribution-free rounding theorem, tested on arbitrary bounded q, not
    # just masks produced by this generator. No fit/real inputs are involved.
    rng=np.random.default_rng(902)
    for _ in range(100):
        db=rng.uniform(30,160,(15,25));q=rng.random((15,25));rounded=np.round(db,1).astype('float32');delta=float(np.max(abs(rounded.astype(float)-db)));gap=abs(float(np.sum(energy_weights(db)*q)-np.sum(energy_weights(rounded)*q)))
        assert gap<=rounding_share_gap_bound(delta)+1e-12
    assert abs(rounding_share_gap_bound(.05)-.005756399149621913)<1e-12
    checks.append('independent arbitrary-q rounding bound')
    rejects(lambda:synthesize(base,np.ones((20,25)),0,'minute_first'))
    rejects(lambda:synthesize(base,np.ones((30,25)),0,'two_second_first_no_warp'))
    rejects(lambda:synthesize(base,np.ones((60,25)),0,'edge_straddling_both'))
    rejects(lambda:synthesize(base,-reference,0))
    rejects(lambda:synthesize(np.full((15,25),np.nan),reference,0))
    rejects(lambda:synthesize(base,reference,0,eq_max_db=13))
    rejects(lambda:synthesize(base,reference,0,snr_db=float('nan')))
    rejects(lambda:synthesize(base,reference,0,family='unknown'))
    checks.append('unsupported/incomplete/invalid requests fail explicitly; no padding/retry')
    assert np.array_equal(base,frozen_base) and np.array_equal(reference,frozen_reference)
    return {'passed':True,'scope':'numeric fixtures only; no challenge data/Test/OUTER/GPU/model fit/SSH/network','families':list(FAMILIES),'random_masked_cases':cases,'checks':checks,'predeclared_lower_higher_SNR_numeric_sensitivity':sensitivity,'maximum_observed_rounding_pool_gap':maximum_gap,'maximum_proven_gap_bound':maximum_bound,'nominal_float64_0_1dB_gap_bound':rounding_share_gap_bound(.05),'full_reference_CPA_peak_not_window_tail':True,'input_arrays_unchanged':True,'source_code_sha256':{'pipeline/robust_sampler_r6.py':digest(ROOT/'pipeline/robust_sampler_r6.py'),'ops/verify_robust_sampler_r6.py':digest(__file__)}}

if __name__=='__main__':print(json.dumps(run(),indent=2,allow_nan=False))
