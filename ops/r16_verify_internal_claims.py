"""R16 pre-fit mathematics and source checks using numeric toys only.

No Test file or Test-derived statistics are read; no model is loaded, fit or
inferred. Existing R15 functions are exercised only on invented arrays.
"""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/r16_independent_qa_r1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def auc_oracle(y,score):
    p=np.asarray(score)[np.asarray(y)>0];n=np.asarray(score)[np.asarray(y)==0]
    return float(((p[:,None]>n[None,:])+.5*(p[:,None]==n[None,:])).mean())


def main():
    OUT.mkdir(parents=True,exist_ok=True);target=OUT/'PRE_FIT_MATH_VERIFICATION.json'
    if target.exists():raise FileExistsError('Preserve existing verification')
    from ops.r15_forward_prototype import choose_trajectory,synthesize_pair,shelf_curve
    source=ROOT/'ops/r15_forward_prototype.py';before=sha(source)
    # Separable time x spectrum: fixed broadband peak SNR fixes the summed
    # added-energy trajectory while retaining the changed spectral ratios.
    time=np.asarray([.2,.4,1.,.7,.3])
    spectrum=np.ones(25);spectrum[:9]=10
    native=time[:,None]*spectrum[None,:]
    bg=np.full((15,25),80.)
    selection={'crop_begin_minute':0,'selected_minutes':5,'window_start_minute':3}
    control,ca=synthesize_pair(bg,native,selection,np.zeros(25),0.)
    eq=shelf_curve(12.,-6.)
    shifted,sa=synthesize_pair(bg,native,selection,eq,0.)
    totals0=control['added_full_power'].sum(1);totals1=shifted['added_full_power'].sum(1)
    assert np.allclose(totals0,totals1,atol=1e-14,rtol=0)
    assert abs(float(control['share']-shifted['share']))<1e-14
    t=5
    ratio0=control['added_full_power'][t,20]/control['added_full_power'][t,0]
    ratio1=shifted['added_full_power'][t,20]/shifted['added_full_power'][t,0]
    expected=10**(18/10)
    assert abs(ratio1/ratio0-expected)<1e-11
    for r in [ca,sa]:
        snr=r['alpha']*r['full_native_peak_after_all_channels']/r['background_median_observed_minute_power']
        assert abs(snr-1)<1e-14
    # Exhaust all native peak indices and supported crop lengths: every
    # chosen crop includes its original pre-EQ broadband maximum.
    crop_checks=0
    for peak in range(9):
        reference=np.ones((9,25));reference[peak]*=20
        for seed in range(100):
            choice=choose_trajectory(bg,reference,np.random.default_rng(seed))
            begin,length=choice['crop_begin_minute'],choice['selected_minutes']
            assert begin<=peak<begin+length
            crop_checks+=1
    # EQ can move the peak. The original peak is still in the crop; the
    # post-EQ full-native peak can be outside without breaking its definition.
    moving=np.ones((3,25))*.01
    moving[0,0]=10;moving[0,20]=1
    moving[1,0]=8;moving[1,20]=8
    moving[2,0]=1;moving[2,20]=1
    gain=np.ones(25);gain[0]=100
    old_peak=int(moving.sum(1).argmax());new_peak=int((moving*gain).sum(1).argmax())
    assert old_peak==1 and new_peak==0
    crop={'crop_begin_minute':1,'selected_minutes':1,'window_start_minute':4}
    moved,receipt=synthesize_pair(bg,moving,crop,10*np.log10(gain),0.)
    inserted_peak=float(moved['added_full_power'].sum(1).max())
    requested_linear=receipt['alpha']*receipt['full_native_peak_after_all_channels']/receipt['background_median_observed_minute_power']
    achieved_inserted_linear=inserted_peak/receipt['background_median_observed_minute_power']
    assert abs(requested_linear-1)<1e-14 and 0<achieved_inserted_linear<1
    # Perfect share is a perfect presence ranking even if positive shares
    # span many orders of magnitude because true no-addition shares equal0.
    truth=np.asarray([0,0,0,0,1e-8,1e-4,.01,.8])
    perfect_auc=auc_oracle(truth,truth);assert perfect_auc==1
    # Spearman is a rank correlation, not a mixture weight or pair agreement.
    a=np.asarray([1.,2.,3.,4.]);b=np.asarray([1.,2.,4.,3.])
    rho=float(np.corrcoef(a,b)[0,1]);agreements=[]
    for i in range(4):
        for j in range(i+1,4):agreements.append(np.sign(a[i]-a[j])==np.sign(b[i]-b[j]))
    pair_fraction=float(np.mean(agreements));assert abs(rho-.8)<1e-14 and abs(pair_fraction-5/6)<1e-14 and rho!=pair_fraction
    fallback={'tree_family':.5,'bg_cnn':.125,'fdy':.125,'itransformer':.125,'tfgrid_fdy_conformer':.125}
    mini={'tree_family':.5,'cnn_A':.25,'cnn_B':.25}
    def feasible(weights):
        return abs(sum(weights.values())-1)<1e-12 and weights['tree_family']>=.4 and all(0<=v<=.4 for k,v in weights.items() if k!='tree_family')
    assert feasible(fallback) and feasible(mini) and not feasible({'tree_family':.5,'cnn':.5})
    assert sha(source)==before
    result={'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Purely numeric pre-fit checks; no Test or Test-derived input files, models, training, inference or policy fitting',
        'R15_source_sha256':before,'source_unchanged':True,
        'fixed_full_native_post_EQ_peak_SNR_is_valid_conditioning':True,
        'separable_reference_fixed_total_added_trajectory_max_error':float(np.max(np.abs(totals0-totals1))),
        'separable_reference_share_control':float(control['share']),'separable_reference_share_EQ':float(shifted['share']),
        'relative_HF_LF_added_power_ratio_multiplier':float(ratio1/ratio0),'expected_18dB_relative_multiplier':expected,
        'original_peak_in_crop_checks':crop_checks,
        'post_EQ_peak_shift_toy':{'pre_EQ_peak':old_peak,'post_EQ_peak':new_peak,'crop_begin':1,'crop_minutes':1,
            'requested_full_native_SNR_dB':0.,'actual_inserted_peak_SNR_dB':float(10*np.log10(achieved_inserted_linear)),
            'interpretation':'Full-native reference level and inserted-crop level differ; neither an absent original peak nor a label identity error.'},
        'perfect_true_share_presence_AUC':perfect_auc,'perfect_true_share_values':truth.tolist(),
        'rank_correlation_not_pair_agreement_toy':{'Spearman':rho,'pairwise_order_agreement':pair_fraction},
        'feasible_cap_examples':{'old_fourNN':fallback,'new_twoNN':mini},
        'infeasible_cap_example':{'tree_family':.5,'single_CNN':.5},
        'classifier_feature_QA_required':'Independent temporal branch only; no profile, normalized energy weight, PCA/anchor or shared regressor encoder if claiming per-band static-offset invariance; test rounding-commuting offsets and state pre-rounding limits.',
        'fit_gate_report_contract':'Fit/checkpoint/ensemble selection may use only predeclared fit roles; gate evaluates a frozen candidate, report remains locked until all policies frozen.',
        'limits':['New scaling/channel priors are admissible experimental alternatives, not a uniquely correct reconstruction of hidden official generation.',
            'Fixed broadband SNR does not require fixed HF bump; mixture max-minus-median is nonlinear and depends on background and temporal support.',
            'Independent presence classifier is a useful ablation, but no AUC improvement follows from architecture alone.',
            'No causal conclusion about earlier public performance is established by these mathematical fixtures.']}
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'PASS','path':str(target.relative_to(ROOT)),'sha256':sha(target),'ratio_multiplier':ratio1/ratio0,'original_peak_checks':crop_checks}))


if __name__=='__main__':main()
