"""Predeclared18-condition physics-only extension; no model outputs read.

This is a forward feasibility calculation for a user-specified challenge,
not Test fitting, not a new model, and not a source/production mutation.
The plan is frozen before inspecting the original audit's model results.
"""
from __future__ import annotations
import argparse,json,hashlib,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def stats(y,bump,floor):
    import numpy as np
    rms=float(np.sqrt(np.mean(y*y)));med=float(np.median(bump))
    return {'n':len(y),'y_mean':float(y.mean()),'y_rms':rms,'high_bump_median':med,
            'y_quantiles':np.quantile(y,[0,.1,.5,.9,1]).tolist(),
            'high_bump_quantiles':np.quantile(bump,[0,.1,.5,.9,1]).tolist(),
            'high_only_energy_floor_RMS':float(np.sqrt(np.mean(floor*floor))),
            'fraction_with_high_only_floor_above020':float(np.mean(floor>.20)),
            'both_panel_targets_met':bool(.15<=rms<=.20 and 14<=med<=18)}
def run(out):
    import numpy as np,pandas as pd
    from ops.r11_asymmetric_eq_audit import guard
    from pipeline.dataset_r9 import R9Sources
    from pipeline.sampler_r9 import place_minutes
    sys.addaudithook(guard);out=out.resolve();plan=json.loads((out/'predeclared_plan.json').read_text())
    if (out/'RESULTS.json').exists():raise FileExistsError('No repeated/adaptive physics search')
    old=ROOT/plan['source_run']
    for name,digest in plan['source_SHA'].items():
        if sha(old/name)!=digest:raise AssertionError('Originalforwardinputchanged:'+name)
    # Explicitly prohibit even reading saved model metrics/prediction files.
    def no_model(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes)):
            name=Path(args[0]).name
            if name.endswith('_predictions.npz') or name in ('RESULTS.json','metrics.json'):
                # Reading existing model results is forbidden; writing our own is allowed.
                flags=args[2] if len(args)>2 else 0
                if not flags & 1 and not flags & 2:raise PermissionError('Forward-only excludes model results')
    sys.addaudithook(no_model)
    started=time.perf_counter();sources=R9Sources();meta=pd.read_parquet(old/'background_source_manifest.parquet')
    specs=json.loads((old/'shared_specs.json').read_text());n=len(specs)
    B=[];P=[];fullHF=[];site=[];zeros=[]
    for s in specs:
        row=meta.iloc[s['row']];db=sources.background(row);peak=float(db.max());B.append(10.**((db-peak)/10))
        native=sources.reference_minutes[max(s['ref'],0)]
        placed,_=place_minutes(native,np.random.default_rng(s['placement_seed']),s['family']);P.append(placed)
        fullHF.append(float(native[:,20:25].sum(1).max()));site.append(str(row.site));zeros.append(s['no_source'])
    B=np.asarray(B);P=np.asarray(P);fullHF=np.asarray(fullHF);site=np.asarray(site);zeros=np.asarray(zeros,bool)
    Btotal=B.sum((1,2));anchor=np.median(B[:,:,20:25].sum(2),axis=1)
    records=[];vectors={};floor_by_snr={};bounderr=0.
    for snr in plan['high3162to7943_fullnative_peakSNR_db']:
        alpha=10.**(snr/10)*anchor/fullHF;alpha[zeros]=0
        Ahigh=alpha[:,None,None]*P[:,:,20:25];highsum=Ahigh.sum((1,2));floor=highsum/(Btotal+highsum)
        floor_by_snr[str(snr)]={'all':stats(floor,np.zeros(n),floor),
            'by_site':{str(s):{'n':int((site==s).sum()),'high_only_floor_RMS':float(np.sqrt(np.mean(floor[site==s]**2))),
                'fraction_floor_above020':float(np.mean(floor[site==s]>.2))} for s in sorted(set(site))}}
        for lf in plan['LF_db']:
            for mid in plan['mid316to1995_db']:
                eq=np.zeros(25);eq[:9]=lf;eq[9]=(lf+mid)/2;eq[10:19]=mid;eq[19]=mid/2
                A=alpha[:,None,None]*P*10**(eq[None,None,:]/10);E=B+A;q=A/E;y=A.sum((1,2))/E.sum((1,2))
                identity=abs((q*E).sum((1,2))/E.sum((1,2))-y);bounderr=max(bounderr,float(identity.max()))
                if identity.max()>1e-12 or (y+1e-12<floor).any():raise AssertionError('Energyidentity/lowerboundfailed')
                high=10*np.log10(E[:,:,20:25].sum(2));bump=high.max(1)-np.median(high,axis=1)
                name=f'lf{abs(lf)}_mid{abs(mid)}_hf{snr}'
                records.append({'name':name,'LF_EQ_db':lf,'mid_EQ_db':mid,'strict_HF_peak_SNR_db':snr,
                    'all':stats(y,bump,floor),'by_site':{str(s):stats(y[site==s],bump[site==s],floor[site==s]) for s in sorted(set(site))}})
                vectors[name+'_share']=y;vectors[name+'_high_bump']=bump;vectors[name+'_high_only_y_floor']=floor
    np.savez_compressed(out/'forward_vectors.npz',**vectors,site=site)
    result={'format':'r11_asymmetric_forward_feasibility_v1','status':'completed',
        'predeclared_plan_sha256':sha(out/'predeclared_plan.json'),'script_sha256':sha(__file__),
        'conditions':records,'feasible_both_targets':[v['name'] for v in records if v['all']['both_panel_targets_met']],
        'arithmetic_lower_bound':{'derivation':'ForfixedHFaddedprofile/scale/support, R_total>=R_HF. Thus y=R_total/(B_total+R_total)>=R_HF/(B_total+R_HF). EqualityrequireszeroadditionoutsideHF.',
            'RMS_consequence':'ElementwiseboundimpliesRMS(y)>=RMS(floor). IfRMSfloor>.20, noLF/midattenuationatthatfixedHFprofilecanreachglobalRMS<=.20.',
            'by_HF_SNR':floor_by_snr},'raw_identity_max_error':bounderr,
        'all18_conditions_preserved':True,'model_predictions_or_errors_used':False,
        'new_models_or_weight_tuning_executed':False,'Test_or_unused_report_opened':False,
        'not_Test_distribution_estimate':True,'production_unchanged':True,
        'model_evaluation_selected_from_this_grid':False,'elapsed_seconds':time.perf_counter()-started,
        'limitations':['ThisgridchangesmiddleresponseandusesSTRICTHFanchor,notthesameanchorastheinitial4conditions.',
            'FeasibleaggregatemomentsdonotprovephysicalplausibilityoractualTestgeneratoragreement.',
            'Samealready-usedselectionbackgrounds/shared509referencecorpus;notnewblinddataset.']}
    write(out/'RESULTS.json',result)
    lines=['# R11 추가 물리 feasibility: 모델을 사용하지 않은 18조건','',
        '원래 네 조건·여덟 모델 비교 패널을 변경하지 않고, 사전에 고정한 별도 18조건만 정방향 계산했습니다. 모델 성능이나 Test 수치는 읽지 않았습니다.',
        '이 추가 표의 SNR 기준은 strictHF 3162–7943 Hz입니다. 원래 네 조건의 316–7943 Hz 기준과 다릅니다. LF와 mid를 더 감쇠시키는 강한 반사실적 가설이며 실제 Test 법칙이라고 주장하지 않습니다.',
        '', '|조건|전체 y RMS|전체 고역 bump 중앙값|두 목표 충족|','|---|---:|---:|---|']
    for v in records:lines.append(f"|{v['name']}|{v['all']['y_rms']:.6f}|{v['all']['high_bump_median']:.3f}|{v['all']['both_panel_targets_met']}|")
    lines+=['', '산술 하한: 고역의 실제 추가 에너지가 A_H로 고정되면 y ≥ A_H/(B_total+A_H)입니다. 따라서 LF와 mid를 0으로 줄여도 이 하한 아래로 내릴 수 없습니다. site별 하한과 가능 조건은 JSON에 있습니다.',
        '두 집계 목표를 만족해도 실제 생성 과정이 맞거나 성능 문제가 설명됐다는 증거는 아닙니다. 이 추가 grid는 모델 예측·학습·조건 채택에 사용하지 않았습니다.',
        '', '[전체 조건·site별 물리통계·하한](RESULTS.json) / [사전 계획](predeclared_plan.json)']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':'completed','feasible':result['feasible_both_targets'],'elapsed_seconds':result['elapsed_seconds']}),flush=True)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'reports/r11_diagnostics/asymmetric_forward_feasibility_r1');run(p.parse_args().output)
