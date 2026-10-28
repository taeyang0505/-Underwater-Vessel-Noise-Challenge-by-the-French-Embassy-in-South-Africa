"""Build a portable, offline entry point from existing audit and EDA artifacts.

Reads R4 TRAIN/INNER arrays for descriptive plots; never fits a model or reads
Test or OUTER acoustic arrays. Historical EDA remains clearly labeled.
"""
from pathlib import Path
import csv
import hashlib
import html
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def make_plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    out = ROOT / 'reports' / 'r4_visual_review'
    out.mkdir(exist_ok=True)
    cv = read_json('data/synthetic/group_cv_passage_r4_v1/cv_manifest.json')
    verification = read_json('reports/R4_DATA_FINAL_VERIFICATION.json')
    reference = read_json('reports/r4_reference_audit.json')
    colors = ['#31796b', '#5083b2', '#bc8249']
    x = np.arange(5)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout='constrained')
    for k, (label, key) in enumerate([('TRAIN', 'train_examples'), ('INNER', 'inner_validation_examples')]):
        values = [f[key] for f in cv['folds']]
        axes[0].bar(x + (k-1)*.23, values, .23, label=label, color=colors[k])
    axes[0].bar(x+.23, [3000]*5, .23, label='OUTER', color=colors[2])
    axes[0].set(xticks=x, xticklabels=[f'Fold {i}' for i in x], ylabel='Synthetic examples',
                title='R4: examples per source-separated role')
    axes[0].legend(frameon=False)
    for k, role in enumerate(['train', 'validation', 'outer_validation']):
        values = [f['partitions'][role]['components'] for f in reference['folds']]
        axes[1].bar(x + (k-1)*.23, values, .23,
                    label=['TRAIN', 'INNER', 'OUTER'][k], color=colors[k])
    axes[1].set(xticks=x, xticklabels=[f'Fold {i}' for i in x], ylabel='Selected reference components',
                title='22 components, split BEFORE synthesis')
    axes[1].legend(frameon=False)
    fig.savefig(out / '01_fold_sizes_and_sources.png', dpi=170); plt.close(fig)

    fold = ROOT / cv['folds'][0]['data_dir']
    with np.load(fold / 'train.npz') as z:
        train_y = z['share'].copy()
        # Choose a fixed explanatory TRAIN example. This is not model selection.
        index = int(np.flatnonzero((train_y > .55) & (train_y < .65))[0])
        db, q = z['db'][index].copy(), z['cell_share'][index].copy()
        y = float(train_y[index])
    with np.load(fold / 'validation.npz') as z:
        inner_y = z['share'].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout='constrained')
    for values, color, label in [(train_y, colors[0], 'TRAIN'), (inner_y, colors[1], 'INNER')]:
        positive = values[values > 0]
        axes[0].hist(positive, bins=np.linspace(0, 1, 21), density=True,
                     histtype='step', lw=2, color=color, label=label)
    axes[0].set(xlabel='Known added-reference share (positive only)', ylabel='Density',
                title='Fold 0: same share-generation policy')
    axes[0].legend(frameon=False)
    for k, role in enumerate(['train', 'validation', 'outer_validation']):
        rows = [p for p in verification['partitions'] if p['split'] == role]
        axes[1].plot([p['fold'] for p in rows], [p['share_mean'] for p in rows],
                     'o-', label=['TRAIN', 'INNER', 'OUTER'][k], color=colors[k])
    axes[1].set(xlabel='Fold', ylabel='Mean known share', ylim=(.38198, .38202),
                title='All roles: audit summaries, not predictions')
    axes[1].ticklabel_format(axis='y', useOffset=False)
    axes[1].legend(frameon=False)
    fig.savefig(out / '02_matched_share_distributions.png', dpi=170); plt.close(fig)

    mask = np.isfinite(db)
    relative = np.where(mask, np.power(10., (db.astype('float64')-np.nanmax(db))/10), 0.)
    w = relative / relative.sum()
    reconstructed = float((w*np.nan_to_num(q)).sum())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for ax, array, name, cmap, lo, hi in [
        (axes[0], db, 'Observed mixture (dB)', 'viridis', None, None),
        (axes[1], np.where(mask, q, np.nan), 'Known cell share q (TRAIN target)', 'magma', 0, 1),
        (axes[2], w, 'Observed energy weights w', 'Blues', 0, None)]:
        im = ax.imshow(array, aspect='auto', origin='lower', cmap=cmap, vmin=lo, vmax=hi)
        ax.set(xlabel='Frequency band index', ylabel='Minute index', title=name)
        fig.colorbar(im, ax=ax, shrink=.78)
    fig.suptitle(f'R4 Fold 0 TRAIN #{index}: target y={y:.6f}; sum(w*q)={reconstructed:.6f}', fontsize=12)
    fig.savefig(out / '03_actual_training_example.png', dpi=170); plt.close(fig)
    receipt = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'scope': 'Descriptive artifact review; no model fitting or Test/OUTER acoustic reads',
               'inputs': {str(fold.relative_to(ROOT) / f): sha(fold/f)
                          for f in ['train.npz', 'validation.npz']},
               'outer_usage': 'Previously saved R4 structural audit summaries only',
               'training_example_index': index, 'target': y, 'pooled_known_mask': reconstructed,
               'absolute_reconstruction_error': abs(y-reconstructed),
               'figures': {p.name: sha(p) for p in sorted(out.glob('*.png'))}}
    (out/'figure_provenance.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2))


CAPTIONS = {
 '01_training_missingness':'원본 주파수 대역별 결측', '02_sites_and_quality':'관측 지점별 데이터 수와 품질',
 '03_test_missingness':'제출 입력의 결측 구조 · 과거 기술 통계',
 '04_spectral_distributions':'주파수별 관측 분포', '05_daily_coverage':'날짜별 관측 범위',
 '06_training_windows':'실제 원본 관측 창', '07_duration_and_sentinel':'관측 길이와 이상치',
 '00_site_map':'남아공 관측 지점', '01_timeline':'관측 기간',
 '02_spectra_by_site_recorder':'지점·수신기별 스펙트럼', '03_band_energy_share':'주파수 대역별 에너지 비중',
 '04_diurnal_by_site':'지점별 일중 변화', '05_monthly_broadband':'월별 광대역 에너지',
 '06_band_correlation_pca':'주파수 상관과 PCA', '07_temporal_dynamics':'시간 변화',
 '08_missing_patterns':'결측 패턴', '09_window_features_train_vs_test':'과거 원본·Test 기술 통계 비교',
 '10_transient_spectra':'일시적 에너지 증가의 스펙트럼', '11_test_window_gallery':'과거 Test 관측 예시',
 '12_train_window_gallery':'원본 학습 배경 관측 예시', '13_domain_probe':'과거 도메인 진단 · 현재 모델 선택에 미사용',
 '14_natural_transients':'자연 관측의 일시적 변화', '15_natural_transient_examples':'자연 관측 변화의 예시',
 '16_target_mechanics':'추가 에너지 비율의 의미', '17_site_matched_comparison':'과거 지점 비교 진단',
 '01_fold_sizes_and_sources':'최신 R4 · fold별 예제 수와 독립 원천 구성',
 '02_matched_share_distributions':'최신 R4 · TRAIN/INNER 생성 분포와 전체 역할 평균',
 '03_actual_training_example':'최신 R4 · 실제 합성 예제와 에너지 가중합 정답',
 'completed_oof_rmse':'구버전 R2 · 실측 OOF RMSE',
}


def link(path, title):
    return f'<a href="{html.escape(str(path), quote=True)}">{html.escape(title)}</a>'


def gallery(directory):
    result = []
    for p in sorted((ROOT/directory).glob('*.png')):
        rel = p.relative_to(ROOT).as_posix()
        title = CAPTIONS.get(p.stem, p.stem.replace('_', ' '))
        result.append(f'<figure><a href="{rel}" target="_blank"><img loading="lazy" src="{rel}" alt="{html.escape(title)}"></a><figcaption>{html.escape(title)}<small>{rel}</small></figcaption></figure>')
    return '<div class="gallery">'+''.join(result)+'</div>'


def table_previews():
    result=[]
    for d in ['eda/tables','eda_deep/tables']:
        for p in sorted((ROOT/d).glob('*.csv')):
            with p.open(newline='', encoding='utf-8-sig') as f:
                rows=[]
                for i,row in enumerate(csv.reader(f)):
                    if i == 9: break
                    rows.append(row[:14])
            if not rows: continue
            rel=p.relative_to(ROOT).as_posix()
            table=''.join('<tr>'+''.join(f'<{"th" if i==0 else "td"}>{html.escape(v)}</{"th" if i==0 else "td"}>' for v in row)+'</tr>' for i,row in enumerate(rows))
            result.append(f'<details><summary>{html.escape(rel)}</summary><p>{link(rel,"전체 CSV 열기 / 저장")}</p><div class="scroll"><table>{table}</table></div><p class="muted">앞 8개 데이터 행·최대 14열 미리보기. 전체 내용은 CSV에 있습니다.</p></details>')
    return ''.join(result)


def build_html():
    eda=read_json('eda/summary.json');cv=read_json('data/synthetic/group_cv_passage_r4_v1/cv_manifest.json')
    sections=[
      ('current','최신 R4 데이터', '<p>VTUAD 119클립·22개 선택 그룹, 동일 생성 레시피의 5-fold 데이터입니다. 원천 분리와 남아공 배경의 24시간 purge를 유지했습니다. R5 CPU 통합 검증을 마쳤으며 실제 GPU 학습 점수는 아직 없습니다.</p>'+gallery('reports/r4_visual_review')),
      ('eda','원본 EDA', '<p>처음 내려받은 대회 데이터의 구조·결측·품질을 실제 파일로 점검한 결과입니다. 그림을 누르면 원본 크기로 열립니다.</p>'+gallery('eda/figures')),
      ('deep','심화 EDA · 역사 기록', '<p>지점·수신기·기간·스펙트럼·결측·관측 예시를 담았습니다. 일부 그림은 과거 Test 기술 통계/도메인 진단입니다. 현재 R4/R5 생성 정책·모델·가중치 튜닝의 근거로 사용하지 않습니다.</p>'+gallery('eda_deep/figures')),
      ('curves','R2 학습 결과 · 구버전', '<p>아래는 이전 R2 데이터·모델로 완료한 실측 결과입니다. 새 R4 데이터의 성능으로 읽으면 안 됩니다. CNN OOF RMSE 0.238545, 당시 앙상블 0.245585이며 공식 제출 점수는 아닙니다.</p>'+gallery('reports/r2_actual_curves')),
      ('tables','EDA 표 미리보기', table_previews()),
    ]
    docs=[('reports/R4_DATA_SUMMARY.md','최신 데이터 구축·검증 완료 보고서'),
          ('reports/FINAL_DATASET_SCALE.md','최종 데이터 규모 · 시간/GB/샘플 수'),
          ('reports/R4_DATA_FINAL_VERIFICATION.json','297,837건 전수 검증 근거'),
          ('reports/R5_MODEL_REFERENCE_FACTCHECK.md','대형 모델 규격·공개 가중치·출처 검증'),
          ('MODEL_ROSTER_AND_STATUS.md','현재 모델 후보·크기·논문·진행 상태'),
          ('reports/R5_INTEGRATION_READINESS.md','R5 통합 러너·전처리·GPU 준비 상태'),
          ('reports/R5_OUTLIER_AUDIT.md','실제 이상치/QC 후보 비율과 TRAIN 제외 현황'),
          ('TASK_EDA_REPORT.md','원본·심화 EDA 상세 보고서'),
          ('DATASET_CONSTRUCTION_PREPROCESSING_REPORT.md','이전 데이터 구축/전처리 상세 보고서 · R2 포함'),
          ('reports/R2_FULL_PIPELINE_REPORT.md','이전 R2 본학습 전체 보고서'),
          ('reports/eda_pipeline.ipynb','이전 EDA 노트북 · IDE/Jupyter에서 열기'),
          ('reports/report.html','이전 종합 HTML 보고서 · 역사 기록'),
          ('README_ANTIGRAVITY.md','Antigravity에서 여는 법과 폴더 설명')]
    sections.append(('docs','보고서와 코드', '<ul>'+''.join('<li>'+link(p,t)+'</li>' for p,t in docs)+'</ul><p>코드: <code>pipeline/</code> · 설정: <code>configs/</code> · 원본: <code>data/raw/</code> · 최신 합성: <code>data/synthetic/group_cv_passage_r4_v1/</code> · 외부 음원: <code>external/</code> · 모델/백업: <code>models/</code>, <code>ops/server_backups/</code></p>'))
    sections.append(('pipeline','현재 R5 파이프라인 · GPU 학습 대기', '<p>새 R5 러너와 실제 모델의 합성 CPU 연결 검증을 마쳤습니다. 아래는 현재 후보 구조이며, 최종 포함 모델·가중치는 GPU 본학습 후 INNER에서 결정합니다. 새 OOF 성능을 측정한 그림이 아닙니다.</p><figure>'+link('figures/pipeline_r5_candidate_v1.png','현재 R5 도식 원본 열기')+'<img loading="lazy" src="figures/pipeline_r5_candidate_v1.png" alt="R5 candidate architecture; GPU CV pending"></figure><details><summary>이전 R2 완료 도식 보기 · 현재 구조와 구분</summary><figure>'+link('figures/pipeline_r2_completed_v1.png','이전 R2 도식 원본 열기')+'<img loading="lazy" src="figures/pipeline_r2_completed_v1.png" alt="Historical R2 pipeline"></figure></details>'))
    nav=''.join(link('#'+sid,title) for sid,title,_ in sections)
    content=''.join(f'<section id="{sid}"><h2>{title}</h2>{body}</section>' for sid,title,body in sections)
    timestamp=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')
    css='''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f3f6f6;color:#1c3037;font:16px/1.7 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Noto Sans KR",sans-serif}header{background:#173b43;color:white;padding:45px max(28px,calc((100vw - 1300px)/2));}header .tag{color:#a8d5c9;font-size:12px;letter-spacing:2px}h1{font-size:34px;line-height:1.3;margin:10px 0 15px}header p{max-width:930px;color:#d4e4e7}nav{display:flex;gap:10px;flex-wrap:wrap;padding:18px 28px;background:white;border-bottom:1px solid #dce4e7}a{color:#226d85;text-decoration:none}a:hover{text-decoration:underline}nav a{font-size:14px;background:#edf4f4;border-radius:20px;padding:5px 12px}main{max-width:1360px;margin:auto;padding:25px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.stat{background:white;border:1px solid #dde7e8;padding:18px;border-radius:12px}.stat b{display:block;font-size:27px;color:#246b60}.stat span{font-size:13px;color:#65757b}section{background:white;border:1px solid #dde7e8;padding:28px;margin:22px 0;border-radius:14px;scroll-margin-top:18px}h2{margin:0 0 12px;font-size:24px}p{margin:10px 0 18px}.gallery{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}figure{margin:0;border:1px solid #e0e7e9;border-radius:10px;overflow:hidden;background:#fafcfc}figure img{display:block;width:100%;height:auto}figcaption{padding:12px;font-weight:600}small{display:block;font-size:11px;color:#77868c;font-weight:400;overflow-wrap:anywhere}.note{padding:15px 20px;background:#e8f3ef;border-left:4px solid #398170;border-radius:7px;margin:20px 0}.muted{color:#76868c;font-size:13px}.scroll{overflow:auto}table{border-collapse:collapse;font-size:12px;white-space:nowrap}td,th{padding:8px;border:1px solid #e1e8eb;text-align:left}th{background:#eef4f5}details{border:1px solid #dce6e8;border-radius:8px;margin:10px 0;padding:12px}summary{cursor:pointer;font-size:14px}code{font-size:13px;background:#edf3f4;padding:3px 6px;border-radius:4px}footer{padding:25px;text-align:center;color:#7b8b90;font-size:12px}@media(max-width:800px){.gallery{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,1fr)}h1{font-size:27px}main{padding:12px}section{padding:18px}}'''
    result=f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Underwater Vessel Noise · EDA와 실험 자료</title><style>{css}</style></head><body><header><div class="tag">FRENCH EMBASSY · UNDERWATER VESSEL NOISE</div><h1>데이터와 실험을 직접 보는 곳</h1><p>원본 EDA, 최신 R4 데이터 검증, R5 후보 도식·CPU 통합 검증, 이전 R2 학습 곡선을 모았습니다. 이 HTML은 인터넷·서버 실행 없이 열 수 있습니다.</p><small style="color:#afc7cc">검토 화면 생성: {timestamp}</small></header><nav>{nav}</nav><main><div class="stats"><div class="stat"><b>{eda['training']['rows']:,}</b><span>원본 남아공 관측 행 · 실제 자연 선박 정답 없음</span></div><div class="stat"><b>{cv['training_examples_total']:,}</b><span>5-fold TRAIN 역할 합계 · 고유 원천 수와 구분</span></div><div class="stat"><b>119 / 22</b><span>선택 VTUAD 클립 / 참조 연결 그룹</span></div><div class="stat"><b>15,000</b><span>INNER 합계 / OUTER 합계 각각</span></div></div><div class="note"><strong>현재 상태:</strong> R4 데이터 전수 검증·R5 CPU 통합 검증 완료. GPU 첫 실측·본학습 대기. 새 OOF 및 최종 앙상블 가중치는 아직 확정되지 않았습니다.</div>{content}</main><footer>원본 데이터·역사적 보고서·검증 근거를 보존한 로컬 검토 페이지 · 그림을 누르면 확대</footer></body></html>'''
    (ROOT/'START_HERE.html').write_text(result)


if __name__ == '__main__':
    make_plots()
    build_html()
    print(json.dumps({'entry':'START_HERE.html','latest_plots':3,'offline':True,'model_training':False}))
