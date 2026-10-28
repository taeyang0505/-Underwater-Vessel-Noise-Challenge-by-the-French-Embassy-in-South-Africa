"""Render the R2 report from downloaded ACTUAL histories and an explicit final report.

No server access, fitting, dataset loading, interpolation, synthetic curves or
automatic outer-metric discovery. Missing histories produce no placeholder plot.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse,hashlib,json,math,re,shutil,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
START=1792869909.5714338
EXPECTED_CONFIG='92bcf84892839fdf9e1963d4448229923cf34a80a4dbe9ecd9eb6ae27a5452f4'
EXPECTED_SOURCE='f69509648c416cc8e8338a209ca10e728ae9430670fd3922696af40d646f9481'
NEURAL=['tabm','cnn','fdy','itransformer']
LABELS={'tabm':'TabM','cnn':'BG residual CNN (8ch)','fdy':'FDY-inspired CRNN (6ch)',
    'itransformer':'iTransformer-inspired (6ch)','lgbm':'LightGBM','lgbm_physics':'LightGBM + physics',
    'lgbm_retrieval':'LightGBM + retrieval','tabicl':'TabICLv2 FP32','kumo_large':'Kumo Large FP32','ensemble':'Inner-weight ensemble'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()

def link(label,path,report):
    import os
    return f'[{label}]({Path(os.path.relpath(path,report.parent)).as_posix()})'

def finite(value):return isinstance(value,(int,float)) and math.isfinite(value)

def histories(progress):
    found={};errors=[]
    if not progress.exists():return found,errors
    for file in sorted(progress.rglob('*_history.json')):
        kind=file.name.removesuffix('_history.json')
        if kind not in NEURAL:continue
        text=str(file);fold=re.search(r'fold_(\d+)',text);seed=re.search(r'seed_(\d+)',text)
        if not fold or not seed:continue
        try:
            rows=json.loads(file.read_text())
            if not isinstance(rows,list) or not rows:raise ValueError('Empty/non-list history')
            epochs=[row['epoch'] for row in rows]
            if epochs!=sorted(set(epochs)) or any(type(e)is not int or e<1 for e in epochs):raise ValueError('Invalid actual epoch order')
            for row in rows:
                if not finite(row.get('validation_rmse')) or not finite(row.get('base_train_rmse',row.get('train_rmse'))):raise ValueError('Missing/nonfinite recorded RMSE')
            summaryfile=file.with_name(kind+'_training_summary.json')
            summary=json.loads(summaryfile.read_text()) if summaryfile.exists() else {}
            item={'file':file,'rows':rows,'summary':summary,'summary_file':summaryfile if summaryfile.exists() else None,
                  'fold':int(fold.group(1)),'seed':int(seed.group(1)),'kind':kind}
            key=(item['fold'],item['seed'],kind)
            if key not in found or len(rows)>len(found[key]['rows']):found[key]=item
        except (ValueError,KeyError,TypeError,json.JSONDecodeError) as exc:
            errors.append({'path':str(file),'error':str(exc)})
    return found,errors

def save_plot(fig,base):
    fig.savefig(base.with_suffix('.png'),dpi=190,bbox_inches='tight',facecolor='white')
    fig.savefig(base.with_suffix('.svg'),bbox_inches='tight',facecolor='white');plt.close(fig)
    return [base.with_suffix('.png'),base.with_suffix('.svg')]

def plot_histories(found,assets):
    outputs=[];proof=[]
    for fold,seed in sorted({(f,s) for f,s,_ in found}):
        items=[found[(fold,seed,k)] for k in NEURAL if (fold,seed,k) in found]
        fig,axes=plt.subplots(1,len(items),figsize=(4.1*len(items),3.8),squeeze=False)
        for axis,item in zip(axes[0],items):
            rows=item['rows'];epochs=[r['epoch'] for r in rows]
            train=[r.get('base_train_rmse',r.get('train_rmse')) for r in rows];val=[r['validation_rmse'] for r in rows]
            axis.plot(epochs,train,color='#607e9b',lw=1.6,label='Recorded train RMSE',marker='.' if len(rows)<3 else None)
            axis.plot(epochs,val,color='#c0874d',lw=1.6,label='Recorded inner RMSE',marker='.' if len(rows)<3 else None)
            summary=item['summary'];best=summary.get('best_epoch');initial=summary.get('initial_validation_rmse')
            if finite(initial):axis.scatter([0],[initial],marker='o',facecolors='none',edgecolors='#c0874d',label='Recorded epoch-0 inner')
            if type(best)is int:axis.axvline(best,color='#74866b',ls=':',lw=1.2,label=f'Selected epoch {best}')
            axis.set_title(LABELS[item['kind']]);axis.set_xlabel('Actual epoch');axis.set_ylabel('RMSE')
            axis.set_ylim(0,max(max(train),max(val),initial if finite(initial) else 0)*1.09+.001)
            axis.grid(axis='y',color='#e1e6e0',lw=.6);axis.legend(frameon=False,fontsize=7)
            proof.append({'kind':item['kind'],'fold':fold,'seed':seed,'actual_rows':len(rows),
                'history_source':str(item['file'].relative_to(ROOT)),'history_sha256':sha(item['file']),
                'summary_sha256':sha(item['summary_file']) if item['summary_file'] else None,
                'interpolation_or_smoothing':False})
        fig.suptitle(f'R2 — Fold {fold+1}, seed {seed} — actual training records',fontsize=12)
        fig.tight_layout();outputs.extend(save_plot(fig,assets/f'fold_{fold:02d}_seed_{seed}_learning_curves'))
    return outputs,proof

def final_result(file,names):
    if file is None:return None
    data=json.loads(file.read_text())
    if not str(data.get('status','')).startswith('completed') or data.get('n_splits')!=5 or len(data.get('folds',[]))!=5 or data.get('oof_examples')!=15000:
        raise ValueError('Only an explicit completed 5-fold/15,000-row final report is accepted')
    if not set(names+['ensemble']).issubset(data.get('pooled_oof',{})):raise ValueError('Final report lacks configured candidates')
    for kind in names+['ensemble']:
        pooled=data['pooled_oof'][kind]['rmse']
        fold=[f['ensemble']['rmse'] if kind=='ensemble' else f['models'][kind]['rmse'] for f in data['folds']]
        if not finite(pooled) or not all(finite(v) for v in fold):raise ValueError('Nonfinite final metric')
        if not np.isclose(np.sqrt(np.mean(np.square(fold))),pooled,atol=1e-6,rtol=1e-6):raise ValueError('Pooled RMSE differs from equal-size fold SSE aggregation')
    if data.get('outer_used_for_early_stopping') is not False or data.get('outer_used_for_ensemble_weights') is not False:
        raise ValueError('Final report must retain the outer-use restrictions')
    for flag in ['challenge_test_accessed','calibration_accessed','sealed_accessed']:
        if data.get(flag) is not False:raise ValueError('Final report violates the frozen assessment scope: '+flag)
    return data

def plot_final(data,names,assets):
    allnames=names+['ensemble'];fig,axis=plt.subplots(figsize=(10.2,6.5))
    for i,fold in enumerate(data['folds']):
        values=[fold['ensemble']['rmse'] if k=='ensemble' else fold['models'][k]['rmse'] for k in allnames]
        axis.scatter(values,np.arange(len(allnames))+(i-2)*.1,s=27,alpha=.8,label=f'Fold {fold["fold"]+1}')
    pooled=[data['pooled_oof'][k]['rmse'] for k in allnames]
    axis.scatter(pooled,range(len(allnames)),marker='D',color='#202b26',s=35,label='Pooled OOF')
    for i,value in enumerate(pooled):axis.text(1.02,i,f'{value:.4f}',transform=axis.get_yaxis_transform(),va='center',fontsize=9,color='#202b26')
    axis.text(1.02,1.02,'Pooled RMSE',transform=axis.transAxes,ha='left',fontsize=9,color='#202b26')
    axis.set_yticks(range(len(allnames)),[LABELS[k] for k in allnames]);axis.set_xlabel('Actual held-out RMSE')
    axis.invert_yaxis();axis.set_xlim(left=0);axis.grid(axis='x',color='#e1e6e0');axis.legend(ncol=3,frameon=False,fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.10))
    axis.set_title('Five-fold source-group OOF')
    fig.tight_layout();return save_plot(fig,assets/'completed_oof_rmse')

def final_policies(progress,data,names):
    """Read the frozen inner weights only after an explicit completed report."""
    rows=[]
    for fold in sorted(f['fold'] for f in data['folds']):
        path=progress/f'fold_{fold:02d}'/'ensemble_policy.json'
        receipt=path.with_name('collection_receipt.json')
        policy=json.loads(path.read_text());collection=json.loads(receipt.read_text())
        if sha(path)!=collection['fold']['policy_sha256']:raise ValueError('Collected frozen policy SHA mismatch')
        weights=np.asarray(policy['weights'],dtype=float)
        if policy['models']!=names or weights.shape!=(len(names),) or not np.isfinite(weights).all() or weights.min() < -1e-8 or not np.isclose(weights.sum(),1.,atol=1e-6):
            raise ValueError('Invalid frozen inner weights')
        if policy.get('affine')!=[1.,0.]:raise ValueError('Unexpected affine calibration')
        if policy.get('outer_used_for_fit') is not False or policy.get('fit_split')!='inner validation only' or policy.get('fold')!=fold:
            raise ValueError('Unexpected policy fitting scope')
        rows.append({'fold':fold,'file':path,'sha256':sha(path),'weights':weights})
    return rows

def write_bundle(report,bundle):
    """Copy only report-linked evidence/figures, rewrite local links, verify ZIP."""
    stage=bundle.with_suffix('');stage.mkdir(parents=True,exist_ok=True);text=report.read_text();copied={}
    def replace(match):
        prefix,target=match.group(1),match.group(2)
        if target.startswith(('http://','https://','#')):return match.group(0)
        source=(report.parent/target).resolve()
        if not source.is_file() or not source.is_relative_to(ROOT):raise ValueError('Missing/non-project portable evidence: '+target)
        suffix='figures' if source.suffix in ['.png','.svg'] else 'evidence'
        destination=Path(suffix)/source.name
        if str(destination) in copied and copied[str(destination)]!=source:destination=Path(suffix)/(sha(source)[:8]+'_'+source.name)
        copied[str(destination)]=source;(stage/destination).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,stage/destination)
        return prefix+destination.as_posix()+')'
    rewritten=re.sub(r'(!?\[[^\]]*\]\()([^)]*)\)',replace,text)
    (stage/report.name).write_text(rewritten)
    paths=[stage/report.name]+[stage/k for k in copied]
    manifest={'purpose':'Portable R2 report/evidence only; no training data or model weights',
        'files':[{'path':f.relative_to(stage).as_posix(),'sha256':sha(f),'bytes':f.stat().st_size} for f in paths]}
    manifestpath=stage/'report_bundle_manifest.json';manifestpath.write_text(json.dumps(manifest,indent=2))
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for file in paths+[manifestpath]:z.write(file,file.relative_to(stage).as_posix())
    with zipfile.ZipFile(bundle) as z:
        if z.testzip() is not None:raise RuntimeError('Report bundle CRC failed')
        for f in manifest['files']:
            if hashlib.sha256(z.read(f['path'])).hexdigest()!=f['sha256']:raise RuntimeError('Report bundle SHA failed')
    return manifest

def main(args):
    report=args.output;report.parent.mkdir(parents=True,exist_ok=True);assets=report.parent/'r2_actual_curves';assets.mkdir(exist_ok=True)
    previous=assets/'figure_provenance.json'
    if previous.exists() and json.loads(previous.read_text()).get('final_report_explicitly_supplied') and args.final_report is None:
        raise ValueError('A completed report cannot be downgraded; supply its explicit final report again')
    configpath=ROOT/'configs/group_cv_physics_training_r2.json';auditpath=ROOT/'ops/final_r2_integration_audit.json'
    config=json.loads(configpath.read_text());audit=json.loads(auditpath.read_text())
    if sha(configpath)!=EXPECTED_CONFIG or audit['configuration_sha256']!=EXPECTED_CONFIG or audit['source_hash']!=EXPECTED_SOURCE:
        raise ValueError('The report must describe the frozen canonical R2 configuration/data')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    found,errors=histories(args.progress_root);figures,proof=plot_histories(found,assets)
    receipts=sorted(args.progress_root.glob('fold_*/collection_receipt.json')) if args.progress_root.exists() else []
    completed_folds=sum(json.loads(p.read_text()).get('fold',{}).get('outer_complete') is True for p in receipts)
    names=config['models']+list(config['foundation_models']);final=final_result(args.final_report,names)
    policies=final_policies(args.progress_root,final,names) if final else []
    verification_path=ROOT/'reports/R2_OOF_INDEPENDENT_VERIFICATION.json'
    verification=json.loads(verification_path.read_text()) if final else None
    if final and (not verification['status'].startswith('passed') or verification['examples']!=15000 or verification['input_sha256']['cv_report.json']!=sha(args.final_report)):
        raise ValueError('Independent final OOF verification does not bind this report')
    runtime=None;runtime_path=assets/'execution_receipt.json'
    if final:
        state_path=ROOT/'ops/runpod_run_state.json';state=json.loads(state_path.read_text())
        duration=state['finished_server_unix']-state['started_server_unix']
        if state['r2_config_sha256']!=EXPECTED_CONFIG or state['started_server_unix']!=START or state['full_training_status']!='completed' or not np.isclose(duration,state['elapsed_training_seconds'],atol=1e-6):
            raise ValueError('Runtime receipt does not bind the completed R2 run')
        runtime={'source_status_path':str(state_path.relative_to(ROOT)),'source_status_sha256_at_snapshot':sha(state_path),
            'scope':'Immutable R2 report snapshot; excludes server connection and other experiment metadata',
            'started_server_unix':START,'finished_server_unix':state['finished_server_unix'],
            'started_KST':datetime.fromtimestamp(START,ZoneInfo('Asia/Seoul')).isoformat(),
            'finished_KST':datetime.fromtimestamp(state['finished_server_unix'],ZoneInfo('Asia/Seoul')).isoformat(),
            'elapsed_training_seconds':duration,'configuration_sha256':EXPECTED_CONFIG,
            'source_hash':EXPECTED_SOURCE,'actual_neural_histories':len(found),'actual_neural_epochs':sum(len(v['rows']) for v in found.values()),
            'independent_oof_verification_sha256':sha(verification_path),'cv_report_sha256':sha(args.final_report)}
        runtime_path.write_text(json.dumps(runtime,indent=2))
    if final:figures+=plot_final(final,names,assets)
    provenance={'configuration_sha256':EXPECTED_CONFIG,'source_hash':EXPECTED_SOURCE,'launch_unix':START,
        'report_builder_sha256':sha(Path(__file__)),
        'launch_KST':datetime.fromtimestamp(START,ZoneInfo('Asia/Seoul')).isoformat(),
        'build_KST':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),'server_polled':False,
        'actual_histories':proof,'unreadable_history_files':errors,'final_report_explicitly_supplied':bool(final),
        'collection_receipts':[{'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in receipts],
        'downloaded_completed_fold_ledgers':completed_folds,
        'final_report_sha256':sha(args.final_report) if final else None,
        'independent_oof_verification_sha256':sha(verification_path) if final else None,
        'execution_receipt_sha256':sha(runtime_path) if final else None,
        'frozen_inner_policy_sources':[{'fold':p['fold'],'path':str(p['file'].relative_to(ROOT)),'sha256':p['sha256']} for p in policies],
        'figures':{f.name:sha(f) for f in figures},'fabricated_curves_or_metrics':False}
    provpath=assets/'figure_provenance.json';provpath.write_text(json.dumps(provenance,indent=2))
    L=lambda label,path:link(label,ROOT/path,report)
    lines=['# R2 전체 모델 학습·추론 파이프라인 보고서','',
        f'실제 본 실행 시작: **2026-10-25 04:25:09 KST** (`{START}`).',
        '이 문서는 고정된 한 파이프라인과 내려받은 실제 실행 기록만 정리합니다. 새 실험·서버 조회·데이터 재생성을 수행하지 않습니다.','',
        '## 현재 결과 상태','']
    if not final:
        lines+=['**최종 5-fold OOF는 아직 이 보고서에 반영되지 않았습니다.** 아래 구성 수치는 고정 설정·검증 결과이며, 미측정 모델 성능을 대신하지 않습니다.',
            f'현재 로컬에서 확인한 실제 신경망 history는 {len(found)}개이고, 다운로드 ledger에서 완료된 fold는 {completed_folds}/5개입니다. 없는 이력에는 곡선이나 성능 수치를 생성하지 않습니다.','']
    else:
        lines += [f'**전체 5-fold / OOF 15,000개 집계 완료.** 최종 inner-weight ensemble의 pooled RMSE는 **{final["pooled_oof"]["ensemble"]["rmse"]:.6f}**입니다.',
            '이 값은 합성 데이터의 원천 분리 OOF 결과이며 Zindi 공식 점수가 아닙니다.',
            f'단일 후보 중 pooled RMSE가 가장 낮은 모델은 **{LABELS[min(names,key=lambda k:final["pooled_oof"][k]["rmse"])]}** ({min(final["pooled_oof"][k]["rmse"] for k in names):.6f})입니다. 사전에 inner로 고정한 ensemble보다 낮으며, 이 결과를 보고 OOF 가중치를 다시 맞추지 않았습니다.',
            f'실제 실행은 **{int(runtime["elapsed_training_seconds"]//60)}분 {runtime["elapsed_training_seconds"]%60:.1f}초**였습니다. 시작·완료 timestamp와 실행 설정을 '+link('실행 receipt',runtime_path,report)+'에 보존했습니다.',
            f'독립 검증은 15,000개 고유 ID 정합, 모든 모델의 pooled·경계 제외 RMSE 재계산, 다섯 동결 inner policy의 예측 재구성을 통과했습니다. 경계 후보 {verification["boundary_flagged_examples"]:,}개를 제외한 진단 표본은 {verification["clean_examples"]:,}개입니다. '+link('독립 검증 원본',verification_path,report),'',
            '| 후보 | Pooled OOF RMSE | Fold RMSE 평균 ± 표준편차 | 경계 후보 제외 RMSE |','|---|---:|---:|---:|']
        for kind in names+['ensemble']:
            m=final['pooled_oof'][kind];s=final['fold_rmse_mean_std'][kind];clean=m.get('oof_rmse_excluding_boundary')
            lines.append(f'| {LABELS[kind]} | {m["rmse"]:.6f} | {s["mean"]:.6f} ± {s["std"]:.6f} | {clean:.6f} |' if finite(clean) else f'| {LABELS[kind]} | {m["rmse"]:.6f} | {s["mean"]:.6f} ± {s["std"]:.6f} | 기록 없음 |')
        lines+=['','Pooled RMSE는 전체 SSE를 합쳐 계산합니다. Fold RMSE의 단순 산술평균과 다르며, 표준편차는 다섯 원천 fold의 산포입니다. 독립 녹음 15,000개의 불확실성 구간을 뜻하지 않습니다.','',link('최종 cv_report 원본',args.final_report,report),'']
    lines += ['## 고정한 단일 파이프라인','',
        '| 단계·모델 | 현재 실행 구성 |','|---|---|',
        '| 데이터 | 물리 r1: train 합계 267,837 + inner 15,000 + outer 15,000. 원천 분할·평가 NPZ 고정 |',
        '| LightGBM 계열 | 기본 / physics / training-only retrieval, 3후보. 최대 800 trees, 회귀 inner early stopping 40 |',
        '| TabM | feature version 3, 1,258차원 |',
        '| CNN 대체 모델 | 8채널 background-prior residual CNN, architecture 4. 다른 후보를 추가한 것이 아니라 CNN을 대체 |',
        '| FDY / iTransformer | 기존 feature 3의 6채널, 각각 독립 후보 |',
        '| TabICLv2 / Kumo Large | vendor 원본 가중치 고정, fold TRAIN context만 새 구성. FP32, 각 4 members, context 16,000 / 4,096 |',
        '| 신경망 최적화 | Adam, 최대 100 epochs, patience 7, dropout 0.1, batch 128, seed 42 |',
        '| 추가 domain augmentation | none. 실패한 추가 증강 pilot을 본 실행에 자동 적용하지 않음 |',
        '| 앙상블·후처리 | 각 fold inner에서 nonnegative·합 1 가중치를 학습하고 고정한 뒤 outer에 적용. Outer-fitted affine 보정 없음 |','',
        '**7개 모델 계열, 9개 후보**이며 마지막에 inner-weight ensemble을 평가합니다. Adam/epoch 정책은 신경망에 적용됩니다. LightGBM과 frozen-context foundation 모델을 100 epochs 학습했다고 표현하지 않습니다.','',
        '회귀 주 손실은 정답 share에 대한 MSE로, RMSE를 최소화하는 것과 같은 예측값을 목표로 합니다. TabM은 share MSE + 0.03 presence BCE를 사용합니다. 셀 mask를 출력하는 CNN·FDY·iTransformer에는 0.03 × 관측 에너지 가중 cell MSE가 추가되고, BG CNN만 0.001 × 에너지 가중 residual-logit 제곱 페널티를 더합니다. 보조 손실이 평가 지표 자체라는 뜻은 아니며 best checkpoint 선택은 고정 inner RMSE로 합니다.',
        '신경망 초기 learning rate는 0.001, weight decay는 0.0003입니다. Plateau scheduler와 gradient norm clipping 1.0을 쓰고, 종료 후 마지막 epoch가 아닌 실제 best checkpoint를 복원합니다.','',
        '## 데이터와 라벨의 범위','',
        '남아공 원본 1,715,779행은 보존했습니다. 이 원본에 검증된 자연 선박 에너지 정답을 완성했다는 뜻은 아닙니다. 저장 정답은 알려진 추가 참조의 관측 셀 에너지 비중이며, 참조에 포함된 주변음도 함께 셉니다.',
        '보조 presence 정답과 score 역시 controlled addition의 유무를 나타냅니다. 배경 안에 원래 존재하던 자연 선박을 모두 검증한 정답으로 해석하지 않습니다.',
        'r1은 각 fold의 깨끗한 학습 배경을 한 번씩 primary로 사용합니다. 학습 수는 53,244 / 53,315 / 53,522 / 52,998 / 54,758개입니다. 외부 참조 1,171클립은 26개 연결 component이며 native 장시간 참조의 독립 component는 5개입니다.',
        '267,837은 다섯 fold의 학습 역할을 더한 수입니다. 원천이 다른 fold의 학습에 다시 등장하므로 독립 배경 267,837개라는 뜻은 아닙니다. 교차 fold의 고유 primary 배경 창은 94,711개이고, 정규 train·inner·outer NPZ 15개는 합계 461,390,519 bytes입니다.',
        '공유 센서-주간 블록과 reference hull/day/overlap/byte 연결 단위로 합성 전에 분리했고, 활성 분할 간 15분 support를 고려한 24시간 purge를 유지했습니다. Calibration·sealed·Test를 이 CV에 넣지 않았습니다.',
        '고정 r1 train은 합성 시 reference-only smooth EQ 최대 ±4dB와 깨끗한 같은 분할 배경 blending 확률 0.25 등을 적용한 데이터입니다. 이번 R2 실행은 이를 그대로 쓰며 추가적인 epoch별 domain augmentation을 넣지 않습니다.',
        '원본 r1 train의 support·20dB minute SNR 제한은 그대로이며, 기존 inner/outer NPZ는 byte로 보존했습니다. 해당 제한이 옛 평가 분포까지 적용됐다고 주장하지 않습니다.','',
        L('데이터 원천·분할 감사','reports/group_cv_physics_data_audit.json')+' · '+L('독립 데이터 검증','reports/group_cv_physics_data_verification.json')+' · '+L('실제 크기·분포','reports/group_cv_physics_data_facts.json'),'','## Background prior를 CNN에 쓰는 방법','',
        '각 fold에서 해당 TRAIN의 `db`와 알려진 `cell_share`로 B=(1−q)E를 복원하고, 배경 스펙트럼의 집계 cluster 32개를 새로 만듭니다. Fold 0 prior나 이전 task 가중치를 다른 fold에 재사용하지 않습니다.',
        'Inner/outer에서는 query `db`와 이미 고정된 prior만 사용합니다. Query 정답·source tag·query 집단 통계로 prior를 다시 맞추지 않습니다. Checkpoint는 prior 파일 SHA, embedded payload, TRAIN NPZ·package·seed·source·fold binding을 검사하며 변조·누락을 거부합니다.',
        '6개 기존 입력 채널에 q0와 log-normalized raw Bhat/E를 더합니다. q0를 정답으로 쓰지 않고 softened logit prior에 학습된 residual을 더합니다. 잔차 정규화 가중치는 0.001이고 CNN 출력은 원래 관측 에너지 가중치로 pooling합니다.',
        '**개발 근거 구분:** raw prior 자체는 fold 0 inner RMSE 약 0.5547, zero 예제 RMSE 약 0.8390으로 실패했습니다. 반면 학습된 residual CNN pilot은 같은 inner에서 약 0.28835로 기존 CNN 약 0.32115보다 개선됐습니다. 두 수치는 최종 OOF가 아니며, raw q0의 정확성을 입증하지 않습니다.',
        'CNN 구조 선택에 fold 0 inner가 사용됐고 이 원천은 다른 fold에서는 outer 역할을 합니다. 따라서 전체 OOF는 모델 개발 증거이며, 모델 설계 선택까지 포함해 완전히 손대지 않은 최종 현장 평가라고 해석하지 않습니다.','',
        L('Raw prior 진단','ops/background_prior_raw_diagnostic.json')+' · '+L('기존 12모델·hybrid parity','reports/background_hybrid_parity_verification.json')+' · '+L('훈련 연결 검증','reports/cnn_background_prior_training_verification.json')+' · '+L('prior binding·guard 검증','reports/cnn_background_prior_outer_verification.json'),'']
    if policies:
        lines += ['## 실제 동결 앙상블','',
            '최종 inner ensemble의 실제 동결 가중치는 아래와 같습니다. Outer 값으로 다시 맞추지 않았고 affine은 모든 fold에서 [1, 0]입니다. 아래 표는 표시 자릿수만 반올림했으며 전체 정밀도는 원본 JSON에 보존했습니다.','',
            '| 후보 | '+' | '.join(f'Fold {p["fold"]+1}' for p in policies)+' |',
            '|---|'+'---:|'*len(policies)]
        for i,kind in enumerate(names):lines.append('| '+LABELS[kind]+' | '+' | '.join(f'{p["weights"][i]:.4f}' for p in policies)+' |')
        lines += ['','정책 원본: '+' · '.join(link(f'Fold {p["fold"]+1}',p['file'],report) for p in policies),'']
    lines += ['## 실제 학습 곡선과 종료 기록','']
    if not found:lines+=['다운로드된 실제 history가 아직 없어 이 절에는 그림을 넣지 않았습니다. 기록이 도착하면 같은 스크립트가 실제 값만 그립니다.','']
    else:
        lines += [f'현재 수집된 신경망 학습 {len(found)}건의 실제 epoch 합계는 {sum(len(v["rows"]) for v in found.values())}입니다. 100은 각 학습의 최대 예산이며 실제 수행 epoch 수로 바꾸어 적지 않았습니다.','']
        for fold,seed in sorted({(f,s) for f,s,_ in found}):
            image=assets/f'fold_{fold:02d}_seed_{seed}_learning_curves.png'
            lines += [f'![Fold {fold+1} 실제 학습 곡선]({image.relative_to(report.parent).as_posix()})',
                f'*Fold {fold+1}, seed {seed}. Train과 inner RMSE의 실제 epoch 기록이며 보간·평활화·예측 곡선을 추가하지 않았습니다. 점선은 실제 저장된 선택 epoch입니다. 패널별 y 범위가 다르므로 모델 간 수치는 축과 표로 비교합니다.* '+link('SVG 원본',image.with_suffix('.svg'),report),'']
        lines+=['| Fold | 모델 | 실제 epoch 수 | 선택 epoch | 최선 inner RMSE | 종료 사유 | 실제 원본 |','|---:|---|---:|---:|---:|---|---|']
        for key in sorted(found,key=lambda key:(key[0],key[1],NEURAL.index(key[2]))):
            item=found[key]
            s=item['summary'];best=s.get('best_epoch');metric=s.get('best_validation_rmse')
            sources=link('history',item['file'],report)+((' · '+link('summary',item['summary_file'],report)) if item['summary_file'] else '')
            reason=s.get('stop_reason','history만 도착')
            if reason=='early_stopping_patience_reached':reason=f'patience {s.get("early_stopping_patience",7)} 조기 종료'
            elif reason=='max_epochs_reached':reason=f'{s.get("max_epochs",100)} epochs 도달'
            lines.append(f'| {item["fold"]+1} | {LABELS[item["kind"]]} | {len(item["rows"])} | {best if best is not None else "미기록"} | {f"{metric:.6f}" if finite(metric) else "미기록"} | {reason} | {sources} |')
        lines+=['','Epoch 0이 선택된 hybrid가 있다면 실제 측정된 초기 prior 상태가 inner에서 더 좋았다는 뜻입니다. 수렴·global minimum 또는 충분한 일반화를 증명하는 표현으로 바꾸지 않습니다.','']
        tree_summaries=[]
        for path in sorted(args.progress_root.glob('fold_*/seed_*/lgbm*_training_summary.json')):
            item=json.loads(path.read_text())
            tree_summaries.append((path,item))
        if tree_summaries:
            lines += ['| Fold | 트리 후보 | 최대 trees | 실제 선택 iteration | 실제 원본 |','|---:|---|---:|---:|---|']
            for path,item in tree_summaries:
                fold=int(re.search(r'fold_(\d+)',str(path)).group(1));kind=item['kind']
                lines.append(f'| {fold+1} | {LABELS[kind]} | {item["max_trees"]} | {item["best_iteration"]} | {link("summary",path,report)} |')
            lines += ['','트리의 선택 iteration은 실제 summary 값이며 신경망 epoch 수와 단위가 다릅니다.','']
        if receipts:
            lines += ['실제 내려받은 기록의 ZIP·entry·ledger 검증 출처: '+ ' · '.join(link(p.parent.name+' receipt',p,report) for p in receipts),
                'Receipt의 `verified_archive.remote.training_started=false`는 백업 도구가 새 학습을 시작하지 않았다는 뜻입니다. 진행 중인 CV 학습이 실행되지 않았다는 상태값은 아닙니다.','']
    if final:
        image=assets/'completed_oof_rmse.png';lines += [f'![완료된 실제 OOF 비교]({image.relative_to(report.parent).as_posix()})','*색 점은 실제 다섯 fold 결과, 검정 표시는 pooled OOF입니다. 임의의 신뢰구간이나 가상 지표를 그리지 않았습니다.* '+link('SVG 원본',image.with_suffix('.svg'),report),'']
    lines += ['트리의 epoch 곡선이나 foundation의 gradient 학습 곡선은 만들지 않습니다. 해당 모델에 실제 기록이 없는 숫자를 신경망 곡선처럼 표시하지 않습니다.','',
        '## 재현·해시·해석 제한','',
        f'- 고정 config SHA256: `{EXPECTED_CONFIG}`',f'- 데이터 source hash: `{EXPECTED_SOURCE}`',
        '- 데이터·코드·설정 해시를 고정하고 완성된 artifact ledger만 재개합니다. 중간 파일이나 다른 설정의 결과를 이어 붙이지 않습니다.',
        '- 독립 native 참조가 적고 합성 예제들이 원천을 재사용합니다. 큰 합성 행 수가 큰 독립 현장 표본 수를 뜻하지 않습니다.',
        '- 자연 배경에도 선박 소리가 있을 수 있고, sensor response·absolute pressure calibration·run-edge/URN 경고가 모두 해결된 것은 아닙니다.',
        '- OOF가 완료돼도 실제 Zindi 점수·새 해역 일반화·리더보드 우위를 자동으로 보장하지 않습니다.','',
        L('고정 config','configs/group_cv_physics_training_r2.json')+' · '+L('최종 통합 감사','ops/final_r2_integration_audit.json')+' · '+L('r1 보존 데이터 gate','ops/final_r1_data_gate_for_r2.json')+' · '+L('배포 manifest','delivery/r2_full_pipeline_code_v2_manifest.json')+' · '+link('그림·history provenance',provpath,report),'',
        '보고서 재생성: `python -m ops.build_r2_full_pipeline_report`. 최종 결과는 조율자가 제공한 완료 파일을 `--final-report <cv_report.json>`으로 명시한 경우에만 읽습니다. 로컬 progress 폴더의 outer metric 파일을 자동 탐색하지 않습니다.']
    report.write_text('\n'.join(lines)+'\n');bundlemanifest=write_bundle(report,args.bundle)
    print(json.dumps({'report':str(report),'actual_histories':len(found),'actual_figures':len(figures)//2,
        'final_OOF_included':bool(final),'history_parse_errors':len(errors),'portable_bundle':str(args.bundle),
        'portable_files':len(bundlemanifest['files']),'bundle_bytes':args.bundle.stat().st_size,'bundle_sha256':sha(args.bundle)},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--progress-root',type=Path,default=ROOT/'ops/r2_progress')
    parser.add_argument('--final-report',type=Path);parser.add_argument('--output',type=Path,default=ROOT/'reports/R2_FULL_PIPELINE_REPORT.md')
    parser.add_argument('--bundle',type=Path,default=ROOT/'reports/R2_FULL_PIPELINE_REPORT_bundle.zip');main(parser.parse_args())
