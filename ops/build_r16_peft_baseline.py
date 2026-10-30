"""Inventory and read-only evaluation of completed R16 scratch checkpoints.

No training, checkpoint reselection, policy fitting, Test reads, or submission.
The evaluation population is frozen explicitly; missing models stay missing.
"""
from pathlib import Path
import argparse, datetime, gc, hashlib, json, sys, time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = ROOT / 'models/r16_multisupport_full_r1'
OUT = ROOT / 'reports/r16_pretrained_ft_r1'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()


def read(p): return json.loads(Path(p).read_text())


def write(p, v):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def rel(p): return str(Path(p).relative_to(ROOT))


def inventory():
    manifest = ROOT / 'data/synthetic/r16_multisupport_v1/manifest.json'
    data = read(manifest); dh = sha(manifest); completed = []; pending = []; artifacts = {}
    for family, name, regime, fold in [
        *[('neural', kind, r, 0) for r in ('D0', 'H0')
          for kind in ('bg_cnn', 'fdy', 'itransformer', 'tfgrid_fdy_conformer')],
        *[('trees', f'fold_{f:02d}', r, f) for r in ('D0', 'H0') for f in range(5)]]:
        folder = RUN / family / regime / name; sp = folder / 'summary.json'
        row = dict(family=family, kind=name, regime=regime, fold=fold, folder=rel(folder))
        if not sp.is_file():
            row['status'] = 'paused_partial' if (folder / 'progress.json').is_file() else 'not_started'
            if (folder / 'progress.json').is_file(): row['last_progress'] = read(folder / 'progress.json')
            pending.append(row); continue
        s = read(sp)
        if s.get('status') != 'completed': raise ValueError('Unexpected model completion status')
        if s['data_manifest_sha256'] != dh: raise ValueError('Data identity mismatch')
        for name2, expected in s['code_sha256'].items():
            if sha(ROOT / name2) != expected: raise ValueError('Model source changed: ' + name2)
        checks = {}
        for name2, expected in s['artifact_sha256'].items():
            p = folder / name2
            if sha(p) != expected: raise ValueError('Model artifact changed: ' + rel(p))
            checks[rel(p)] = expected
        artifacts.update(checks); artifacts[rel(sp)] = sha(sp)
        row.update(status='completed', source_sha256=s['code_sha256'], artifacts=checks)
        if family == 'neural':
            row.update(epochs=s['epochs'], early_stopping=s['early_stopping'],
                       best_regression_epoch=s['best_regression_epoch'], fit_rmse=s['best_rmse'],
                       best_classifier_epoch=s['best_classifier_epoch'], fit_auc=s['best_classifier_auc'])
        else:
            row.update(iterations=s['iterations'], train_n=s['train_n'],
                       feature_counts=s['feature_counts'], selection_metrics=s['selection_metrics'],
                       fit_rmse=s['selection_metrics']['rmse'], fit_auc=s['selection_metrics']['classifier_auc'])
        completed.append(row)
    cache = {}
    for mp in (ROOT/'data/cache/r16_multisupport_full_r1').glob('*/*/manifest.json'):
        c = read(mp)
        checks = {n: sha(mp.parent/n) == h for n, h in c['artifact_sha256'].items()}
        if not all(checks.values()) or c['data_manifest_sha256'] != dh: raise ValueError('Cache verification failed')
        cache[rel(mp)] = dict(n=c['n'], manifest_sha256=sha(mp), all_artifacts_verified=True)
    result = dict(status='completed', generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scratch_run=rel(RUN), data_manifest_sha256=dh, source_identity=data['source_identity'],
        reference_partition=data['reference_partition'], fivefold_scope=data['fivefold_scope'],
        selection={**data['selection'], 'unique_backgrounds':400, 'operators_per_background':6,
                   'purpose':'development checkpoint selection; not independent OOF'},
        training_rows={f:{r:data['folds'][f]['roles']['train'][r]['n'] for r in ('D0','H0')} for f in data['folds']},
        completed=completed, pending=pending, caches=cache,
        frozen_artifact_sha256=artifacts, Test_used=False, training_started_by_this_script=False,
        evaluated_fixed_panels=False, all_scratch_neural_folds_completed=False,
        caveats=['local_fit uses 400 unique backgrounds repeated under six operators; its 2400 rows are not 2400 independent sources.',
                 'Best regression epoch and independent classifier epoch differ and are loaded separately.',
                 'Completed scratch CNN is Fold0 only. Partial FDY is excluded from frozen completed baseline.',
                 'Tree fivefold CV is nested inside original R9 Fold0 TRAIN and A-prime reference pool, not full509 independent CV.',
                 'Tree5 average is never evaluated as OOF on its constituent ownfold CV rows.',
                 'No all-five architecture scratch ensemble was completed; no such baseline score is claimed.'])
    write(OUT/'SCRATCH_BASELINE_STATUS.json',result); render(result)
    return result


def render(d):
    lines=['# R16 사전학습 전환 시점의 Scratch 기준점', '',
           '완료된 체크포인트와 원천 해시를 확인한 읽기 전용 기록입니다. `local_fit`은 개발용 선택 점수이며, 독립 OOF나 공식 리더보드 점수가 아닙니다.', '',
           '| 모델 | D/H | Fold | 최적 local_fit RMSE | 최적 독립 분류 AUC | 학습 상태 |',
           '|---|---|---:|---:|---:|---|']
    for a in d['completed']:
        state=(f"100 epoch, RMSE ep{a['best_regression_epoch']}, AUC ep{a['best_classifier_epoch']}" if a['family']=='neural'
               else f"reg {a['iterations']['regression']} / clf {a['iterations']['classification']} trees")
        lines.append(f"| {a['family']}/{a['kind']} | {a['regime']} | {a['fold']} | {a['fit_rmse']:.6f} | {a['fit_auc']:.6f} | {state} |")
    lines += ['', '## 미완료·제외', '']
    for a in d['pending']:
        ep=a.get('last_progress',{}).get('epoch')
        lines.append(f"- {a['regime']} {a['kind']}: {a['status']}"+(f', epoch {ep} 저장 후 정지' if ep else ''))
    lines += ['', '## 데이터와 검증 범위', '',
              f"- 데이터 manifest SHA-256: `{d['data_manifest_sha256']}`",
              f"- `local_fit.npz` SHA-256: `{d['selection']['sha256']}`",
              '- local_fit: 고유 배경 400개 × 생성 연산자 6종 = 2,400행. 체크포인트 선택에 사용됐습니다.',
              '- D0/H0 TRAIN: Fold당 약 6.1만 행. 동일 원천 분리·900초 support·24시간 purge 계약을 유지합니다.',
              '- 캐시 `.npy`와 완료 모델의 모든 ledger entry SHA-256을 재검증했습니다.',
              '- 초기 상태에서는 현재 완료 모델의 고정 패널 추론이 미측정입니다. 과거 소형·다른 레시피 점수를 현재 모델 점수로 옮겨 쓰지 않았습니다.', '']
    if d.get('evaluated_fixed_panels'):
        lines += ['## 동결 후 고정 패널 평가', '', '| 패널 | 모델 | RMSE | 독립 분류 AUC | share AUC |', '|---|---|---:|---:|---:|']
        for a in d.get('evaluation_rows',[]):
            auc=lambda x:'단일 클래스' if x is None else f'{x:.6f}'
            lines.append(f"| {a['panel']} | {a['model']} | {a['rmse']:.6f} | {auc(a['classifier_auc'])} | {auc(a['share_auc'])} |")
    lines += ['', '## 해석상 제한', '']+['- '+v for v in d['caveats']]
    lines += ['', '정확한 원천·체크포인트·캐시 해시는 [JSON](SCRATCH_BASELINE_STATUS.json)에 보관합니다. 공식 Test 입력과 제출 파일을 읽거나 만들지 않았습니다.', '']
    (OUT/'SCRATCH_BASELINE_STATUS.md').write_text('\n'.join(lines))


def evaluate(d, owncv=False):
    import numpy as np
    import torch
    import lightgbm as lgb
    from pipeline.training_r16_multisupport import codes, features, tree_features, metric, verify
    from pipeline.r16_multisupport_data import PARENT, role, guard
    from pipeline.models_r16_multisupport import MultiSupportModel
    from ops.r16_support_temporal_data import load_panel, load_long_panel
    from ops.r16_forward_data import OPERATORS
    if not (RUN/'PAUSED.json').is_file(): raise RuntimeError('Scratch process must pause before baseline evaluation')
    sys.addaudithook(guard); torch.set_num_threads(2)
    # Scope is explicitly only completed models; an absent model is never silently initialized.
    policy=OUT/'SCRATCH_BASELINE_EVALUATION_POLICY.json'
    write(policy,dict(status='fixed_before_locked_inputs', models=['CNN_D0','CNN_H0','CNN_DH50','Tree5_D0','Tree5_H0','Tree5_DH50'],
        D0_H0_weights=[.5,.5], checkpoint_selection='existing completed summary best epoch only',
        any_new_policy_fit=False, partial_models_excluded=True))
    artifacts={**d['frozen_artifact_sha256'],rel(policy):sha(policy),rel(Path(__file__)):sha(__file__)}
    freeze=OUT/'SCRATCH_BASELINE_GLOBAL_FREEZE.json'
    write(freeze,dict(status='completed', all_models_frozen=True, ensemble_policy_frozen=True,
         scope='Only completed scratch CNN D0/H0 and Tree10, all listed before locked inputs',
         data_manifest_sha256=sha(PARENT),artifact_sha256=artifacts))
    trees={r:[(lgb.Booster(model_file=str(RUN/'trees'/r/f'fold_{f:02d}'/'regression.txt')),
               lgb.Booster(model_file=str(RUN/'trees'/r/f'fold_{f:02d}'/'classification.txt'))) for f in range(5)] for r in ('D0','H0')}
    neural={}
    for r in ('D0','H0'):
        folder=RUN/'neural'/r/'bg_cnn'; m=MultiSupportModel('bg_cnn').eval()
        reg=torch.load(folder/'best_regression.pt',map_location='cpu',weights_only=True)
        if reg['code_sha256']!=codes() or reg['architecture']!=m.architecture(): raise ValueError('CNN identity mismatch')
        m.load_state_dict(reg['state_dict']);cl=torch.load(folder/'best_classifier.pt',map_location='cpu',weights_only=True)
        m.classifier.load_state_dict({k.removeprefix('classifier.'):v for k,v in cl['state_dict'].items() if k.startswith('classifier.')})
        neural[r]=m;del reg,cl
    def predict(db):
        result={};xp,wp=features(db)
        for r,m in neural.items():
            pp=[];cc=[]
            with torch.no_grad():
                for off in range(0,len(db),64):
                    x=torch.from_numpy(xp[off:off+64]);w=torch.from_numpy(wp[off:off+64]);p,c,q=m(x,w)
                    if not torch.allclose(p,(w*q).sum((1,2)),atol=1e-7,rtol=0):raise ValueError('Physics pooling differs')
                    pp.append(p.numpy());cc.append(c.sigmoid().numpy())
            result['CNN_'+r]=(np.concatenate(pp),np.concatenate(cc))
        tr=tree_features(db,'regression');tc=tree_features(db,'classification')
        for r,pairs in trees.items(): result['Tree5_'+r]=(np.mean([np.clip(a.predict(tr,num_threads=2),0,1) for a,b in pairs],0),np.mean([b.predict(tc,num_threads=2) for a,b in pairs],0))
        for family in ('CNN','Tree5'): result[family+'_DH50']=tuple(.5*(result[family+'_D0'][j]+result[family+'_H0'][j]) for j in (0,1))
        return result
    rows=[];began=time.perf_counter()
    def one(name, data, weights=None):
        predictions=predict(data['db'])
        for model,(p,c) in predictions.items():rows.append(dict(panel=name,model=model,**metric(data['share'],p,c,weights)))
        write(OUT/'SCRATCH_EVALUATION_PROGRESS.json',dict(status='running',last_panel=name,rows=rows,seconds=time.perf_counter()-began))
        print(json.dumps(dict(panel=name,seconds=time.perf_counter()-began,rows=rows[-6:])),flush=True)
    for name in ('local_gate','local_report','domain_holdout'):
        parts=[load_panel(PARENT,name,op,freeze_receipt=freeze) for op in OPERATORS]
        keys=set.intersection(*(set(p) for p in parts));one(name,{k:np.concatenate([p[k] for p in parts]) for k in keys})
        gc.collect()
    parts=[load_long_panel(PARENT,'native12_positive',operator=op,freeze_receipt=freeze) for op in OPERATORS]
    keys=set.intersection(*(set(p) for p in parts));pos={k:np.concatenate([p[k] for p in parts]) for k in keys}
    for name in ('hf_high_negative','st_francis_dense_negative'):
        neg=load_long_panel(PARENT,name,freeze_receipt=freeze);keys=set(pos)&set(neg)
        combined={k:np.concatenate([pos[k],neg[k]]) for k in keys}
        weights=np.r_[np.full(len(pos['share']),.5/len(pos['share'])),np.full(len(neg['share']),.5/len(neg['share']))]
        one('native12_VS_'+name,combined,weights)
    cv=[]
    if owncv:
        for r,pairs in trees.items():
            ys=[];pp=[];cc=[]
            for f,(a,b) in enumerate(pairs):
                v=role(f,r,'cv');p=np.clip(a.predict(tree_features(v['db'],'regression'),num_threads=2),0,1);c=b.predict(tree_features(v['db'],'classification'),num_threads=2)
                cv.append(dict(regime=r,fold=f,**metric(v['share'],p,c)));ys.append(v['share']);pp.append(p);cc.append(c)
            cv.append(dict(regime=r,fold='pooled_ownfold_only',**metric(np.concatenate(ys),np.concatenate(pp),np.concatenate(cc))))
    verify(codes());d.update(evaluated_fixed_panels=True,evaluation_rows=rows,tree_ownfold_CV=cv,
                            evaluation_seconds=time.perf_counter()-began, fixed_panel_freeze_sha256=sha(freeze))
    write(OUT/'SCRATCH_BASELINE_STATUS.json',d);render(d)
    return d


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evaluate',action='store_true');p.add_argument('--owncv',action='store_true');a=p.parse_args()
    d=inventory()
    print(json.dumps(dict(completed=len(d['completed']),pending=len(d['pending']),inventory=rel(OUT/'SCRATCH_BASELINE_STATUS.json'))),flush=True)
    if a.evaluate:evaluate(d,a.owncv)
