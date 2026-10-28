"""Local three-figure entry page and Desktop delivery, preserving existing edits."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import html
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def build():
    cards = [
        ('preprocessing', '01 · Data Preprocessing',
         '공식 남아공 자료와 VTUAD의 출처·시간·용량, 품질 처리, 배경 원천 분할, 동적 합성, 고정 검증 자료를 연결합니다.'),
        ('training', '02 · Training',
         '상대 8채널, 네 신경망의 셀 마스크·에너지 풀링, Mac CPU LightGBM, 100 epoch와 검증 역할을 설명합니다.'),
        ('inference', '03 · Inference',
         '학습 완료 후의 고정 전처리·체크포인트·fold별 정책과 5-fold 평균, CSV 검사 계약을 설명합니다.'),
    ]
    body = []
    for key, title, description in cards:
        relative = '../figures/r9_pipeline_' + key + '.png'
        if not (ROOT/'figures'/('r9_pipeline_'+key+'.png')).is_file():
            raise FileNotFoundError(relative)
        body.append(f'<section id="{key}"><h2>{title}</h2><p>{description}</p>'
                    f'<p><a href="{relative}" target="_blank">원본 이미지 열기·확대</a></p>'
                    f'<a href="{relative}" target="_blank"><img src="{relative}" alt="{title}"></a></section>')
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>R9 데이터 보고서와 구조도 3장</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f5f1;color:#18231f;font:16px/1.75 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif}main{max-width:1500px;margin:auto;padding:38px 24px}header,section{background:white;border:1px solid #d9ded9;border-radius:12px;padding:26px;margin-bottom:24px}h1{font-size:30px;line-height:1.35}h2{font-family:Georgia,serif;font-size:27px}a{color:#245d69}nav{display:flex;flex-wrap:wrap;gap:12px;margin:22px 0}nav a,.button{display:inline-block;padding:9px 15px;border:1px solid #bed0c7;border-radius:7px;text-decoration:none;background:#f1f6f1}img{display:block;width:100%;height:auto;border:1px solid #eee}strong{color:#1f4936}.note{border-left:4px solid #c2a660;padding-left:15px;background:#fcfaf3}.small{font-size:13px;color:#536059}li{margin:8px 0}@media print{nav{display:none}section{break-before:page;border:0}main{padding:0}}
</style><main><header><p class="small">R9 v1 · 2026-10-26 · 데이터·코드 검증 결과와 실행 계약</p><h1>데이터 설명·EDA 보고서와<br>전처리 / 학습 / 추론 구조도</h1>
<p><a class="button" href="DATA_EDA_PREPROCESSING_R9_REPORT.pdf">전체 PDF 열기</a> <a class="button" href="DATA_EDA_PREPROCESSING_R9_REPORT.html">상세 보고서 HTML</a> <a class="button" href="DATA_EDA_PREPROCESSING_R9_REPORT.md">Markdown 원문</a></p>
<nav><a href="#preprocessing">전처리</a><a href="#training">학습</a><a href="#inference">추론</a></nav>
<p class="note"><strong>현재 상태:</strong> 새 데이터 15,000개 고정 평가 예제와 동적 TRAIN 준비·검증 완료. 기본 크기 네 모델의 실제 자료 Adam 1-step·체크포인트 재로드 통과. R9 GPU 100-epoch 본학습과 공식 평가는 아직 실행 전입니다.</p>
<ul><li>SA 배경: 고유 허용 원천 100,964개. TRAIN은 fold별 78,069–78,869개를 매 epoch 한 번씩 사용합니다.</li><li>VTUAD: 참조 509개(원 WAV header 합 33.1633시간, 완전 minute 합 32.9667시간) + 별도 ambient controls 16개. 다섯 fold가 참조 풀을 공유하므로 <strong>배경 원천 CV</strong>입니다.</li><li>고정 평가 예제: fold당 selection 1,000 / ensemble-fit 500 / ensemble-gate 500 / report 1,000. 원천 창 수와 합성 예제 수를 구별해야 합니다.</li><li>새 데이터 bundle 71.245 MB(기존 source bank 제외), 압축 NPZ 합 24.708 MB. TRAIN은 동적으로 생성하므로 고정 학습 NPZ 개수로 세지 않습니다.</li><li>구조도 heatmap은 구조 설명용 예시이며 실측 성능 그래프가 아닙니다. R6 파일럿 CSV는 로컬 진단 전용으로 업로드하지 않습니다.</li></ul>
<p><a href="R9_DATA_READINESS.md">데이터 검증 설명</a> · <a href="R9_REAL_DATA_CONNECTION_VERIFICATION.json">실제 데이터·본체 모델 연결 검사</a> · <a href="R9_BACKGROUND_CV_AUDIT.md">5-fold·purge 감사</a> · <a href="R9_REFERENCE_POOL_AUDIT.md">509개 참조 검증</a> · <a href="R9_R6_PILOT_SUBMISSION.md">R5 / R6 예측 분포 진단</a></p>
</header>'''+''.join(body)+'''<footer class="small">수치 정의·근거 파일·분모는 상세 보고서를 참조하세요. 모델 크기나 로컬 RMSE만으로 공식 리더보드 성능을 보장하지 않습니다.</footer></main></html>'''
    state_file = ROOT/'ops/r9_remote_state.json'
    if state_file.is_file():
        state = json.loads(state_file.read_text())
        if state.get('production_training_started'):
            page = page.replace('R9 GPU 100-epoch 본학습과 공식 평가는 아직 실행 전입니다.',
                                'R9 GPU 신경망 본학습은 '+html.escape(state.get('started_KST',''))+
                                '에 시작했습니다. 공식 평가 결과는 아직 없습니다. PDF와 그림의 pending은 결과 미확정을 뜻하며, 정적 보고서 작성 시점과 현재 운영 상태를 구별하세요.')
            page = page.replace('<nav>', '<p><a href="R9_GPU_DEPLOYMENT_LAUNCH_RECEIPT.md">GPU 배포·학습 시작 기록</a> · '
                                '<a href="../ops/r9_remote_state.json">서버 실행 상태 기록</a></p><nav>')
    path = ROOT/'reports/R9_VISUAL_REVIEW_START.html'
    path.write_text(page)
    return path


def local_links(path):
    source = path.read_text()
    values = re.findall(r'(?:href|src)=["\']([^"\']+)["\']',source) if path.suffix=='.html' else re.findall(r'\]\(([^)]+)\)', source)
    for value in values:
        if '://' in value or value.startswith(('#','mailto:','data:')):
            continue
        value = re.sub(r':\d+$', '', value.split('#')[0]).strip('<>')
        if value:
            yield (path.parent/value).resolve()


def deliver(target):
    target = Path(target).resolve()
    if not target.is_dir() or target==ROOT:
        raise ValueError('Existing independent Desktop project required')
    names = set()
    for pattern in ('reports/R8*','reports/R9*','reports/DATA_EDA_PREPROCESSING_R9_REPORT*',
                    'figures/r9_pipeline*','pipeline/*r9*.py','ops/*r9*.py',
                    'ops/build_r8_data_report_pdf.py','configs/r9*.json',
                    'data/versions/background_cv_r9','data/versions/vtuad_full_r9',
                    'data/synthetic/group_cv_minute_full_r9_v1','submissions/r6_pilot_r1'):
        for path in ROOT.glob(pattern):
            candidates = [path] if path.is_file() else [v for v in path.rglob('*') if v.is_file()]
            for item in candidates:
                if '__pycache__' not in item.parts:
                    names.add(str(item.relative_to(ROOT)))
    # Portable report dependencies, copied only when absent or already identical.
    for name in list(names):
        path = ROOT/name
        if path.suffix in {'.md','.html'}:
            for dep in local_links(path):
                if not dep.is_relative_to(ROOT) or not dep.is_file():
                    raise ValueError('Invalid source link '+str(dep))
                if not (target/dep.relative_to(ROOT)).exists():
                    names.add(str(dep.relative_to(ROOT)))
    previous = ROOT/'reports/R9_VISUAL_DESKTOP_DELIVERY.json'
    known = json.loads(previous.read_text()).get('files',{}) if previous.exists() else {}
    for name in sorted(names):
        source,dest = ROOT/name,target/name
        if not dest.resolve().is_relative_to(target) or dest.is_symlink():
            raise ValueError('Unsafe or user-owned destination link '+name)
        if dest.exists() and sha(dest)!=sha(source) and known.get(name,{}).get('sha256')!=sha(dest):
            raise ValueError('Preserving changed Desktop file: '+name)
    records={}
    copied=0
    for name in sorted(names):
        source,dest = ROOT/name,target/name
        digest=sha(source)
        if not dest.exists() or sha(dest)!=digest:
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,dest);copied+=1
        assert sha(dest)==digest
        records[name]={'sha256':digest,'bytes':source.stat().st_size}
    checked=0
    for name in names:
        path=target/name
        if path.suffix in {'.md','.html'}:
            for dep in local_links(path):
                if not dep.is_file():
                    raise ValueError('Broken delivered report link: '+str(dep))
                checked+=1
    receipt={'status':'passed','created_utc':datetime.now(timezone.utc).isoformat(),
             'destination':str(target),'files_verified':len(records),'copied':copied,
             'bytes_verified':sum(v['bytes'] for v in records.values()),
             'local_links_checked':checked,'existing_IDE_edits_preserved':True,'files':records}
    previous.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    shutil.copyfile(previous,target/'reports'/previous.name)
    return {k:v for k,v in receipt.items() if k!='files'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--destination',type=Path)
    a=p.parse_args();print(build())
    if a.destination:print(json.dumps(deliver(a.destination),ensure_ascii=False,indent=2))
