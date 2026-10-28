"""R2 stdlib-only status rendering. No subprocess/model/query access.

R1 outputs and failure evidence remain immutable; hybrid outputs come only
from the independently versioned R2 output supervisor/driver.

Supervisor API: render(snapshot=None) -> pathlib.Path. All writes are to
the new status HTML only. No old report, training source or state is edited.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse,csv,hashlib,html,json,math,os,statistics,time
from urllib.parse import quote

ROOT=Path(__file__).resolve().parents[1]
KINDS=('itransformer','fdy','bg_cnn','tfgrid_fdy_conformer')
LABELS={'itransformer':'iTransformer','fdy':'FDY-CRNN','bg_cnn':'Wide BG CNN','tfgrid_fdy_conformer':'TFGrid-FDY-Conformer'}
KST=ZoneInfo('Asia/Seoul')

def maybe(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError,TypeError):return default
def history(path):
    out=[]
    try:
        for line in Path(path).read_text().splitlines():
            try:
                row=json.loads(line)
                if isinstance(row,dict):out.append(row)
            except ValueError:continue  # trailing incomplete append is retried next refresh
    except OSError:pass
    return out
def numeric(value):
    try:
        x=float(value);return x if math.isfinite(x) else None
    except (TypeError,ValueError):return None
def fmt(value,digits=6):
    x=numeric(value);return '—' if x is None else f'{x:,.{digits}f}'
def duration(value):
    x=numeric(value)
    if x is None:return '미측정'
    if x<60:return f'{x:.1f}초'
    if x<3600:return f'{x/60:.1f}분'
    return f'{x/3600:.1f}시간'
def esc(value):return html.escape(str(value),quote=True)
def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def link(path,out,label):return f'<a href="{esc(quote(os.path.relpath(path,out.parent),safe="/"))}">{esc(label)}</a>'
def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def validate_submission(path,root):
    """Only generated local output CSVs; never reads raw Test or labels."""
    p=Path(path);manifest=p.with_suffix('.manifest.json');m=maybe(manifest)
    if not isinstance(m,dict) or not p.is_file() or p.is_symlink():return None
    if (root.resolve() not in p.resolve().parents or m.get('rows')!=1500 or m.get('ID_order_preserved') is not True
        or m.get('numeric_roundtrip_exact') is not True or m.get('official_upload') is not False):return None
    if m.get('path')!=p.relative_to(root).as_posix():return None
    if isinstance(m.get('checks'),dict) and not all(v is True for v in m['checks'].values()):return None
    try:
        if file_sha(p)!=m.get('output_sha256'):return None
        with p.open(newline='') as f:
            reader=csv.reader(f)
            if next(reader)!=['ID','share_added_vessel','score_added_vessel']:return None
            rows=list(reader)
        if len(rows)!=1500 or any(len(r)!=3 for r in rows) or len({r[0] for r in rows})!=1500:return None
        if any(not r[0].strip() or any(numeric(v) is None or not 0<=float(v)<=1 for v in r[1:]) for r in rows):return None
    except (OSError,ValueError,StopIteration):return None
    return m

def render(snapshot=None,*,root=ROOT,output=None):
    root=Path(root).resolve();out=Path(output or root/'reports/R14_PRODUCTION_START.html')
    cfg=maybe(root/'configs/r14_production_mac_v1.json',{}) or {}
    nn=root/'models/r14_production_nn_fold0_r1';tree=root/'models/r14_production_trees_r1'
    origin='감독 snapshot'
    if snapshot is None:snapshot=maybe(root/'ops/r14_production_state_r2.json')
    if not isinstance(snapshot,dict):
        origin='실제 학습 파일 fallback · 감독 snapshot 준비 중'
        snapshot={'status':'running','production_recipe_approved':cfg.get('production_recipe_approved') is True,
            'tree_progress':maybe(tree/'progress.json',{}),'neural_run_state':maybe(nn/'run_state.json',{}),
            'neural_current_batch':maybe(nn/'progress.json',{}),'neural_models':{},'stages':{}}
    state=dict(snapshot);current=state.get('neural_current_batch') or maybe(nn/'progress.json',{}) or {}
    model_snap=state.get('neural_models') or {};models={};rows=[];conditional_seconds={}
    for kind in KINDS:
        hist=history(nn/kind/'epochs.jsonl');last=(model_snap.get(kind) or (hist[-1] if hist else None) or maybe(nn/kind/'progress.json',{}) or {})
        summary=maybe(nn/kind/'summary.json',{}) or {};spec=(cfg.get('models') or {}).get(kind,{})
        budget=int(spec.get('epochs',50 if kind==KINDS[-1] else 100));done=max([int(v.get('epoch',0)) for v in hist]+[int(last.get('epoch',0))])
        completed=summary.get('status')=='completed' and summary.get('epochs_run')==budget
        if completed:done=budget
        times=[numeric(v.get('actual_epoch_seconds_including_validation_checkpoint')) for v in hist]
        times=[v for v in times if v is not None and v>=0];first=times[0] if times else None;recent=statistics.median(times[-5:]) if times else None
        inflight=0.;active=current.get('kind')==kind and numeric(current.get('epoch')) is not None and int(current['epoch'])>done
        if active:
            den=numeric(current.get('total_rows'));num=numeric(current.get('rows_seen'))
            inflight=max(0.,min(1.,num/den)) if den and num is not None else 0.
        remaining=None if recent is None else max(0,budget-done-inflight)*recent
        conditional_seconds[kind]=remaining
        status='완료' if completed else ('학습 중' if active else ('epoch 대기' if done else '시작 대기'))
        frac=min(1.,max(0.,(done+inflight)/budget));epochs=f'<strong>{done} / {budget}</strong><div class="bar"><span style="width:{frac*100:.2f}%"></span></div>'
        if active:epochs+=f'<small>현재 epoch {int(current["epoch"])} · {inflight:.1%} 전수 처리</small>'
        latest=last.get('selection') or {};device=spec.get('device','CPU' if kind=='itransformer' else 'MPS').upper()
        best=summary.get('best_selection_rmse',last.get('best_selection_rmse'))
        rows.append([esc(LABELS[kind])+f'<small>{esc(device)} · {esc(status)}</small>',epochs,
            fmt(best),fmt(last.get('train_online_rmse')),fmt(latest.get('rmse')),duration(first)+'<small>최근 '+duration(recent)+'</small>',
            ('완료' if completed else '미측정' if remaining is None else '~'+duration(remaining))])
        models[kind]={'history':hist,'done':done,'budget':budget,'recent':recent,'completed':completed,'last':last}
    failure=state.get('error') or (maybe(nn/'FAILED.json',{}) or {}).get('error') or (maybe(tree/'FAILED.json',{}) or {}).get('error')
    phase=(state.get('neural_run_state') or {}).get('phase') or current.get('phase')
    overall='오류 확인 필요' if failure or state.get('status')=='failed' else '로컬 산출물 생성 완료' if state.get('status')=='completed' else '본체 학습 진행 중'
    tp=state.get('tree_progress') or maybe(tree/'progress.json',{}) or {};full_registry=maybe(tree/'ALL_FIVEFOLD_TREES_FROZEN.json',{}) or {}
    if full_registry.get('status')=='completed' and len(full_registry.get('models',{}))==5:trees_done=10
    else:
        legacy=maybe(root/'models/r14_tree_full_r1/ALL_TREES_FROZEN.json',{}) or {}
        reused=len(legacy.get('models',{})) if legacy.get('status')=='completed' else 0
        new={(v.get('fold'),v.get('variant')) for v in tp.get('completed_new_learners',[]) if isinstance(v,dict)}
        trees_done=min(10,reused+len(new))
    fastready=all(models[k]['completed'] for k in KINDS[:3]);fast_eta=None
    if all(conditional_seconds[k] is not None for k in KINDS[:3]):
        fast_eta=sum(conditional_seconds[k] for k in KINDS[:3])
        cache_times=[numeric(v.get('raw_cache_materialization_seconds')) for v in models['itransformer']['history'][-5:]];cache_times=[v for v in cache_times if v is not None]
        current_epoch=max([int(current.get('epoch',0))]+[models[k]['done'] for k in KINDS[:3]])
        if cache_times:fast_eta+=statistics.median(cache_times)*max(0,100-current_epoch)
    updated=state.get('updated_KST') or datetime.now(KST).isoformat(timespec='seconds')
    runtime=maybe(root/'ops/r14_production_nn_launch_r1.json',{}) or {};pid=runtime.get('PID','—')
    approved=state.get('production_recipe_approved') is True or cfg.get('production_recipe_approved') is True
    if not runtime and not any(v['history'] for v in models.values()) and not failure and state.get('status')!='completed':overall='시작 상태 확인 대기'
    stages=dict(state.get('stages') or {})
    # The completed Tree-only artifact is intentionally reused from R1. R2
    # does not regenerate it, and may have no tree-only stage entry itself.
    old_tree=validate_submission(root/'submissions/R14/submission_r14_fulltree_5fold.csv',root)
    if old_tree and (maybe(root/'reports/r14_production/TREE_FIVEFOLD_REPORT.json',{}) or {}).get('status')=='completed':
        stages['tree-only']={'status':'completed','reused_R1':True}
    stage_labels={'tree-only':'Tree5 단독 CSV · R1 보존','hybrid-fast3':'빠른3종 + Tree5 CSV · R2','hybrid-full4':'TF 포함 별도 CSV · R2'}
    stage_cards=''.join(f'<div class="stage"><b>{esc(label)}</b><span>{"완료" if (stages.get(k) or {}).get("status")=="completed" else "생성 중" if state.get("active_output")==k else "완료 barrier 대기"}</span></div>' for k,label in stage_labels.items())
    report_dir=root/'reports/r14_production_r2';tree_report_dir=root/'reports/r14_production';sections=[]
    tr=maybe(tree_report_dir/'TREE_FIVEFOLD_REPORT.json')
    if isinstance(tr,dict) and tr.get('status')=='completed':
        metrics=tr.get('pooled_OOF') or {};detail=[]
        for v in tr.get('folds',[]):detail.append([esc(v.get('fold','—')),esc(v.get('n','—')),fmt(v.get('rmse')),fmt(v.get('auc_score'))])
        sections.append('<section><h2>완료된 Tree5 배경-CV report · R1 보존</h2><p>pool RMSE '+fmt(metrics.get('rmse'))+' · score AUC '+fmt(metrics.get('auc_score'))+' · 참조 음원 공유, 전체 NN5-fold OOF 아님</p>'+table(['Fold','행 수','RMSE','score AUC'],detail)+link(tree_report_dir/'TREE_FIVEFOLD_REPORT.json',out,'R1 보존 report JSON')+'</section>')
    for scope,label in [('fast3','빠른3종 혼합'),('full4','TF 포함 혼합')]:
        audit_path=report_dir/f'HYBRID_{scope}_QUERY_DISTRIBUTION_AUDIT.json';audit=maybe(audit_path)
        if isinstance(audit,dict):
            dist=[]
            for audit_origin,title in [('R9','이전 R9'),('new','새 R14')]:
                for key,caption in [('share','share'),('score','score')]:
                    v=(audit.get(audit_origin) or {}).get(key) or {};q=v.get('quantiles') or {}
                    dist.append([esc(title+' '+caption),fmt(v.get('mean')),fmt(v.get('rms')),fmt(q.get('p05')),fmt(q.get('median')),fmt(q.get('p95')),fmt(v.get('ge099_fraction'))])
            sections.append('<section><h2>'+esc(label)+' · 추론 분포</h2><p>분포 검산이며 정답 기반 Test 성능이 아닙니다. Hybrid 검증은 NN Fold0+Tree Fold0, 최종 Tree5 bagging은 추론 확장입니다.</p>'+table(['출력','평균','RMS','p05','중앙값','p95','≥.99 비율'],dist)+link(audit_path,out,'분포 audit JSON')+'</section>')
        cp=report_dir/f'HYBRID_{scope}_17_PANEL_METRICS.csv'
        if cp.is_file():
            try:
                with cp.open(newline='') as f:panel_rows=list(csv.DictReader(f))
                pt=[[esc(v.get('panel','')),fmt(v.get('rmse')),fmt(v.get('auc_score')),fmt(v.get('rmse_minus_previous')),fmt(v.get('R14_smallCNN_fullTree_previous_rmse'))] for v in panel_rows]
                sections.append('<section><h2>'+esc(label)+' · 고정17 개발 패널</h2><p>모델·정책 동결 후의 평가. 기존 노출 패널이며 공식 Test/새 맹검이 아닙니다. ΔRMSE는 이전 소형CNN 결합 대비입니다.</p>'+table(['패널','RMSE','score AUC','ΔRMSE','이전 RMSE'],pt)+'</section>')
            except (OSError,csv.Error):pass
    files=[('Tree5 단독','submissions/R14/submission_r14_fulltree_5fold.csv'),
        ('빠른3종 · 정확 Fold0 · R2','submissions/R14_r2/submission_r14_fold0.csv'),('빠른3종 · Tree5 bagging · R2','submissions/r14_main_r2/submission_r14_step1.csv'),
        ('TF 포함 · 정확 Fold0 · R2','submissions/R14_r2/submission_r14_fold0_with_tfgrid.csv'),('TF 포함 · Tree5 bagging · R2','submissions/r14_main_r2/submission_r14_step1_with_tfgrid.csv')]
    downloads=[]
    for label,name in files:
        p=root/name;m=validate_submission(p,root)
        if m:downloads.append('<div class="download">'+link(p,out,label+' CSV 다운로드')+'<small>1,500행 · ID/schema/범위/SHA 검산 · 업로드 안 함<br><code>'+esc(m['output_sha256'])+'</code></small></div>')
    result_section='<section><h2>로컬 제출 파일</h2><p>아래 링크는 실제 CSV와 manifest 검산을 모두 통과한 경우에만 표시됩니다. 파일을 생성하는 것과 공식 대회에 업로드하는 것은 별도입니다.</p>'+(''.join(downloads) if downloads else '<div class="empty">아직 검증된 제출 CSV가 없습니다. 학습 및 동결 barrier가 완료되면 자동 표시됩니다.</div>')+'</section>'
    notes='<p class="note">최근 최대5개 완료 epoch의 중앙값을 사용한 조건부 시간입니다. 모델별 잔여는 해당 모델 연산량 환산이며 round-robin 대기·Tree·후처리를 포함하지 않습니다. fast3 잔여는 공유 합성을 한 번씩 더하지만, 향후 장치·동시 부하 변화나 실패를 보장하지 않습니다. TF는 첫 실제 epoch가 없으면 미측정으로 표시합니다. 메모리는 공유24GB이며 순간 peak를 보장하지 않습니다.</p>'
    alert='' if not failure else '<div class="alert"><strong>실패를 보존하고 자동 재시작하지 않습니다.</strong><br>'+esc(failure)+'</div>'
    old_state=maybe(root/'ops/r14_production_state.json',{}) or {}
    archive='<section><h2>출력 감독 R2 · R1 기록 보존</h2><p>신경망 학습과 완료된 Tree5 CSV·배경-CV report를 유지하면서, 새 hybrid 산출물은 R2 폴더에 기록합니다. 이 페이지는 최신 R2 상태를 표시합니다.</p>'
    if old_state.get('status')=='failed':
        archive+='<p><strong>이전 R1 출력 감독 종료</strong> · '+esc(old_state.get('updated_KST','시각 미기록'))+'</p><p class="note">'+esc(old_state.get('error','원 실패 기록 보존'))+'<br>출력 감독의 종료 기록이며, 현재 학습 실패로 합산하지 않습니다.</p>'
    state_links=[]
    for name,label in [('r14_production_state.json','원 R1 상태 기록'),('r14_production_state_r2.json','최신 R2 감독 상태')]:
        state_path=root/'ops'/name
        state_links.append(link(state_path,out,label) if state_path.is_file() else esc(label+' 생성 대기'))
    archive+='<p class="note">'+' · '.join(state_links)+'</p></section>'
    head=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="30"><title>R14 production 진행 상황 · 출력 감독 R2</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f4f5f1;color:#182521;font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Noto Sans KR",sans-serif;font-size:14px;line-height:1.6}}main{{max-width:1280px;margin:auto;padding:32px 24px 70px}}header{{border-bottom:1px solid #c9d2c8;padding-bottom:20px;margin-bottom:24px}}h1{{font-size:30px;letter-spacing:-1px;margin:4px 0}}h2{{font-size:20px;margin:0 0 14px}}p{{margin:8px 0}}.eyebrow{{color:#496353;font-size:12px;font-weight:700;letter-spacing:2px}}.muted,.note,small{{color:#66756c}}small{{display:block;font-size:11px;font-weight:400}}.badge{{display:inline-block;border:1px solid #adc5ae;border-radius:20px;padding:3px 10px;background:#e4efe1;color:#315d37;font-size:12px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:22px 0}}.card,section{{background:#fff;border:1px solid #d8dfd4;border-radius:12px;padding:20px}}.card strong{{display:block;font-size:25px;margin-top:6px}}section{{margin-top:18px}}.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;min-width:710px}}th{{color:#64766b;font-size:12px;text-align:left;font-weight:600;background:#f3f6f0}}td,th{{padding:12px 10px;border-bottom:1px solid #e7ece3;vertical-align:top}}tbody tr:last-child td{{border-bottom:0}}td{{font-variant-numeric:tabular-nums}}.bar{{height:5px;background:#e7ece3;border-radius:4px;max-width:140px;margin:7px 0}}.bar span{{display:block;height:100%;background:#66856a;border-radius:4px}}.stages{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.stage{{background:#f2f5ef;padding:14px;border-radius:8px}}.stage span{{display:block;color:#65776b;font-size:12px}}.alert{{padding:18px;background:#fff0e9;border:1px solid #da9c82;border-radius:10px;color:#844629;margin:20px 0}}.note{{font-size:12px;margin:16px 0 0}}a{{color:#356441;text-decoration:none;border-bottom:1px solid #afc4ac}}a:hover{{color:#142c19}}code{{font-size:10px;word-break:break-all}}.download{{padding:14px 0;border-bottom:1px solid #e6ece1}}.empty{{padding:18px;background:#f6f7f2;color:#748071;border-radius:8px}}footer{{font-size:11px;color:#718176;margin-top:24px}}@media(max-width:800px){{main{{padding:22px 12px}}.cards{{grid-template-columns:repeat(2,1fr)}}.stages{{grid-template-columns:1fr}}h1{{font-size:24px}}}}
</style><main><header><div class="eyebrow">R14 · MAC CPU + MPS · OUTPUT SUPERVISOR R2</div><h1>{esc(overall)}</h1><p>14채널 본체 Fold0 + FullTree 5-Fold · 로컬 파일 생성까지의 실제 상태</p><span class="badge">{'신규 사용자 production 승인 TRUE' if approved else 'production 승인 상태 확인 대기'}</span><p class="muted">갱신 {esc(updated)} · 30초 자동 새로고침 · {esc(origin)}</p></header>'''
    body=alert+'<div class="cards"><div class="card">Tree 학습기 완료<strong>'+str(trees_done)+' / 10</strong><small>5fold×2변형 · 각 회귀+분류</small></div><div class="card">빠른3종 완료<strong>'+str(sum(models[k]['completed'] for k in KINDS[:3]))+' / 3</strong><small>각100epoch · round-robin 공유 합성</small></div><div class="card">fast3 조건부 잔여<strong>'+('완료' if fastready else duration(fast_eta))+'</strong><small>Tree·TF50·추론/검산 제외</small></div><div class="card">현재 단계<strong>'+esc('TF50' if phase=='TF' else 'fast3' if phase=='fast3' else state.get('active_output') or '준비/감독')+'</strong><small>학습 PID '+esc(pid)+'</small></div></div>'
    body+='<section><h2>신경망 실제 진행</h2>'+table(['모델 / 장치','완료 epoch / 예산','최저 selection RMSE','최근 TRAIN online RMSE','최근 selection RMSE','첫 epoch / 최근 중앙값','조건부 개별 잔여'],rows)+notes+'<p class="note">Adam · early stopping 없음 · fast3 100epoch. TF50은 사용자 시간예산 예외를 사전에 고정한 별도 모델 예산입니다. TRAIN online RMSE는 학습 중 배치 통계이며 고정 TRAIN 재평가 점수가 아닙니다.</p></section>'
    body+='<section><h2>동결 이후 산출물 단계</h2><div class="stages">'+stage_cards+'</div><p class="note">Neural은 먼저 Fold0만 학습합니다. NN Fold0 + Tree5 bagging을 전체 Neural5-fold OOF라고 부르지 않습니다. 나머지 Neural Fold1~4는 첫 요청 파일 이후 별도 단계입니다.</p></section>'
    body+=archive+result_section+''.join(sections)
    body+='<section><h2>공식 성능과 승인 범위</h2><p><strong>이전 R9 공식 결과:</strong> RMSE 0.217535806 · AUC 0.555066666.</p><p><strong>새 R14 Test:</strong> 정답이 없어 공식 RMSE/AUC는 아직 미측정입니다. local report 점수와 출력 분포를 공식 성능으로 환산하지 않습니다.</p><p class="note">이전 진단의 production FALSE와 당시 4/5 판정·원지표는 그대로 보존했습니다. 현재 TRUE는 사용자 신규 승인으로 생성된 별도 production config입니다. 자동 공식 업로드는 하지 않습니다. 참조 음원을 공유하므로 reference-independent 검증이라고 주장하지 않습니다.</p></section>'
    body+='<footer>이 페이지는 JSON/JSONL·생성된 결과 CSV만 읽습니다. 모델 실행·추론·재학습·설정 변경·네트워크·원 Test 읽기를 하지 않습니다. '+esc(datetime.now(KST).isoformat(timespec='seconds'))+'</footer></main></html>'
    out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_name(out.name+'.tmp');tmp.write_text(head+body);tmp.replace(out)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot');p.add_argument('--output');a=p.parse_args()
    print(render(maybe(a.snapshot) if a.snapshot else None,output=a.output))
