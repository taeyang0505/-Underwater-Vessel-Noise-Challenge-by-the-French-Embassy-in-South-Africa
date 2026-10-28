"""R12 frozen matched-control score-only follow-up for R13 design.

Four predeclared classifier weights; select on R12 gate only, then report
all saved development panels. No model inference/training/Test reads and
no transfer of this chosen weight into the future R13 experiment.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
WEIGHTS=(0.,.25,.5,1.)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def token(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def score(share,clf,w):
    import numpy as np
    a=np.asarray(share,float);b=np.asarray(clf,float)
    if w not in WEIGHTS or a.shape!=b.shape or a.ndim!=1 or not np.isfinite(np.r_[a,b]).all():raise ValueError('Fourfixedweightsandalignedfiniteprobabilitiesrequired')
    if ((a<0)|(a>1)|(b<0)|(b>1)).any():raise ValueError('Boundedshare/classifierrequired')
    return ((1-w)*a+w*b).astype(np.float32)
def metrics(y,s,physical_share):
    import numpy as np
    from ops.verify_r12_evaluation import pair_auc
    y=np.asarray(y,float);z=y>0;s=np.asarray(s,float);p=np.asarray(physical_share,float);bins=[];ece=0.
    for k in range(10):
        m=(s>=k/10)&((s<(k+1)/10) if k<9 else s<=1)
        if m.any():
            mean=float(s[m].mean());freq=float(z[m].mean());ece+=float(m.mean())*abs(mean-freq)
            bins.append({'lower':k/10,'upper':(k+1)/10,'n':int(m.sum()),'mean_score':mean,'presence_frequency':freq})
    return {'n':len(y),'positive_n':int(z.sum()),'auc_presence':pair_auc(y,s),
        'brier_presence':float(np.mean((s-z)**2)),'ECE10_presence':ece,'score_mean':float(s.mean()),'score_RMS':float(np.sqrt(np.mean(s*s))),
        'score_le001_fraction':float(np.mean(s<=.01)),'score_ge099_fraction':float(np.mean(s>=.99)),
        'zero_score_mean':float(s[~z].mean()) if (~z).any() else None,'positive_score_mean':float(s[z].mean()) if z.any() else None,
        'physical_share_RMSE_unchanged':float(np.sqrt(np.mean((p-y)**2))),'physical_share_mean':float(p.mean()),
        'reliability_bins':bins}
def run(a):
    import numpy as np
    from ops.evaluate_r12_ablation import guard
    sys.addaudithook(guard)
    source=(ROOT/a.source).resolve();out=(ROOT/a.output).resolve()
    if not source.is_relative_to(ROOT) or not out.is_relative_to(ROOT) or out.exists():raise ValueError('Freshproject-containedoutputsrequired')
    r=read(source/'RESULTS.json')
    if r.get('status')!='completed' or r.get('format')!='r12_mask_matched_followup_evaluation_v2':raise ValueError('Completedstrictmatchedv2snapshotrequired')
    out.mkdir(parents=True);started=time.perf_counter()
    manifest={n:sha(source/n) for n in r['output_sha256']}
    for n,h in r['output_sha256'].items():
        if manifest[n]!=h:raise AssertionError('ChangedR12snapshot:'+n)
    plan={'format':'r13_R12snapshot_score_plan_v1','created_unix':time.time(),'source_results_sha256':sha(source/'RESULTS.json'),
        'source_output_sha256':manifest,'weights_classifier':WEIGHTS,'primary_share':'Blend50_L1HM1P1',
        'formula':'(1-lambda)*fixedphysicalshare+lambda*independentTreeclassifier',
        'primary_selection':'maximumR12_validation_gatepresenceAUC; firstcanonicalweightontie; Brier/ECE report-only',
        'secondary_share':'Tree_L1HM1P1; allfourweightstabulatedonly,notalternatepolicyselection',
        'gate_is_already_exposed_development':True,'other_panels':'post-selectiondescriptivereportonly',
        'future_R13_weight_selection':'freshselectionafterallR13fitsfreeze; doNOTinheritthissnapshotweight',
        'regression_share_unchanged':True,'no_Test_new_model_inference_fit_or_production_edits':True,
        'Brier_target':'binarypresenceofknownwholeaddedrecording; notenergyfraction',
        'AUC_vs_calibration':'StrictmonotonicsharetransformcannotimproveAUCapartfromquantization/ties;classifiermixturecanchangeordering. LowerBrierdoesnotprovebetterenergyRMSE.',
        'script_sha256':sha(__file__)}
    write(out/'PREDECLARED_PLAN.json',plan)
    def arrays(name):
        with np.load(source/'predictions'/(name+'.npz'),allow_pickle=False) as z:
            order=z['model_order'].astype(str).tolist();return {'ids':z['sample_id'].copy(),'y':z['target_share'].copy(),
                'blend':z['share'][:,order.index('Blend50_L1HM1P1')].copy(),'tree':z['share'][:,order.index('Tree_L1HM1P1')].copy(),
                'classifier':z['extra_tree_classifier'].copy(),'cnn_contrast':z['extra_cnn_contrast'].copy()}
    gate=arrays('R12_validation_gate');selection={str(w):metrics(gate['y'],score(gate['blend'],gate['classifier'],w),gate['blend']) for w in WEIGHTS}
    chosen=max(WEIGHTS,key=lambda w:selection[str(w)]['auc_presence'])
    policy={'format':'r13_R12snapshot_score_policy_v1','chosen_classifier_weight':chosen,
        'candidate_gate_metrics':selection,'primary_share':'Blend50_L1HM1P1','selection_target':'presenceAUC',
        'canonical_first_tie':True,'plan_file_sha256':sha(out/'PREDECLARED_PLAN.json'),
        'not_future_R13_policy':True,'not_original8recipe_change':True}
    policy['policy_sha256']=token(policy);write(out/'SCORE_POLICY_FROZEN_BEFORE_OTHER_PANELS.json',policy)
    policysha=sha(out/'SCORE_POLICY_FROZEN_BEFORE_OTHER_PANELS.json');rows=[];details={}
    for path in sorted((source/'predictions').glob('*.npz')):
        name=path.stem;data=gate if name=='R12_validation_gate' else arrays(name);detail={}
        saved={'sample_id':data['ids'],'target_share':data['y'],'classifier':data['classifier'],
            'physical_blend_share':data['blend'],'physical_tree_share':data['tree'],'classifier_weight_order':np.asarray(WEIGHTS)}
        for context in ('blend','tree'):
            candidates=[];base=metrics(data['y'],score(data[context],data['classifier'],0.),data[context]);contextmetrics={}
            for w in WEIGHTS:
                candidate=score(data[context],data['classifier'],w);m=metrics(data['y'],candidate,data[context]);contextmetrics[str(w)]=m;candidates.append(candidate)
                rows.append({'panel':name,'share_context':context,'classifier_weight':w,
                    'primary_snapshot_policy_selected':context=='blend' and w==chosen,
                    **{k:v for k,v in m.items() if k!='reliability_bins'},
                    'AUC_delta_vs_share':m['auc_presence']-base['auc_presence'] if m['auc_presence'] is not None else None,
                    'Brier_delta_vs_share':m['brier_presence']-base['brier_presence']})
            saved[context+'_score_candidates']=np.column_stack(candidates);detail[context]=contextmetrics
            if len({m['physical_share_RMSE_unchanged'] for m in contextmetrics.values()})!=1:raise AssertionError('Scoreonlychangealteredregression')
        detail['raw_CNN_contrast_ge099_fraction']=float(np.mean(data['cnn_contrast']>=.99));details[name]=detail
        np.savez_compressed(out/(name+'_scores.npz'),**saved)
    with (out/'score_results.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    if sha(out/'SCORE_POLICY_FROZEN_BEFORE_OTHER_PANELS.json')!=policysha or sha(source/'RESULTS.json')!=plan['source_results_sha256']:raise AssertionError('Policy/sourcechanged')
    for n,h in manifest.items():
        if sha(source/n)!=h:raise AssertionError('R12snapshotmodified')
    report={'format':'r13_R12snapshot_score_results_v1','status':'completed','chosen_classifier_weight':chosen,
        'policy_file_sha256':policysha,'selection_uses_only_gate':True,'gate_is_reused_development_not_blind':True,
        'R12_source_unchanged':True,'physical_regression_predictions_unchanged':True,'future_R13_score_not_selected_here':True,
        'panels':details,'rows':len(rows),'elapsed_seconds':time.perf_counter()-started,
        'no_training_inference_Test_external_actions':True,'source_results_sha256':plan['source_results_sha256'],
        'plan_file_sha256':sha(out/'PREDECLARED_PLAN.json'),'script_sha256':sha(__file__)}
    write(out/'RESULTS.json',report)
    lines=['# R13 설계용 R12 snapshot: share와 독립 classifier 점수','',
        '기존 strict matched control의 저장 예측만 사용했습니다. 모델 추론·학습·Test 접근·원 R12 정책 수정은 없습니다.',
        f"기존 R12 gate에서 네 사전 고정 classifier 가중치 중 presence AUC 최대 λ={chosen}를 선택한 뒤 다른 패널을 보고했습니다. 이 gate는 이미 노출된 개발 검증이며 새 blind 성능 추정이 아닙니다. R13 새 모델은 전체 동결 후 별도 gate 선택을 해야 하고 이 λ를 상속하지 않습니다.",
        'share는 실제 추가 에너지 비율이고 classifier는 추가 여부 확률을 목표로 합니다. 따라서 presence Brier와 에너지 RMSE는 다른 목표입니다. 점수만 바꿨으므로 회귀 share와 RMSE는 그대로입니다. 단조 share 재보정만으로 순위 AUC가 개선되는 것은 아닙니다.',
        '', '|패널|classifier λ|presence AUC|Brier|ECE10|score≥.99|에너지 RMSE(불변)|','|---|---:|---:|---:|---:|---:|---:|']
    for name in details:
        for w in [0.,chosen,1.]:
            if w in [0.,chosen,1.][: [0.,chosen,1.].index(w)]:continue
            m=details[name]['blend'][str(w)];fmt=lambda v:'undefined' if v is None else f'{v:.8f}'
            lines.append('|'+ '|'.join([name,str(w)]+[fmt(m[k]) for k in ['auc_presence','brier_presence','ECE10_presence','score_ge099_fraction','physical_share_RMSE_unchanged']])+'|')
    lines+=['','전체 네 λ와 tree-share secondary 표, reliability bin과 raw contrast 포화는 JSON/CSV에 보존했습니다. 점수 정의가 유일한 실패 원인이라고 단정할 수 없으며 원래 회귀 오차·시간 지지·source/generator 차이는 별도로 남습니다.',
        '', '[전체 수치](RESULTS.json) / [정책 동결](SCORE_POLICY_FROZEN_BEFORE_OTHER_PANELS.json) / [모든 후보 표](score_results.csv)']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'status':'completed','lambda':chosen,'rows':len(rows),'elapsed_seconds':report['elapsed_seconds']}));return report
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',default='reports/r12_mask_matched_evaluation_r2');p.add_argument('--output',default='reports/r13_score_snapshot_r1');run(p.parse_args())
if __name__=='__main__':main()
