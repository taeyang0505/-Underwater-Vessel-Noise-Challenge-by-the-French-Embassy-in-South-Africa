"""Build a portable Korean evidence report from existing aggregate artifacts.

No acoustic arrays, Test CSV, training, fitting, GPU or network are used.
All305 R8 catalog entries and all26 archived aggregate tables are preserved.
The tables' original Test-derived historical entries remain clearly archival;
they are not recomputed or used to set any generator/model parameters.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import textwrap
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
STEM = 'DATA_EDA_PREPROCESSING_R9_REPORT'
FONT_PATH = '/Library/Fonts/Arial Unicode.ttf'
INPUTS = [
    'reports/R8_SA_EDA_EVIDENCE.json', 'reports/R8_SA_EDA_EVIDENCE.md',
    'reports/R8_EXTERNAL_DATA_EVIDENCE.json', 'reports/R8_EXTERNAL_DATA_EVIDENCE.md',
    'reports/R8_FEATURE_PIPELINE_EVIDENCE.json', 'reports/R8_FEATURE_PIPELINE_EVIDENCE.md',
    'reports/R9_DIAGRAM_FACTS.json', 'reports/R9_DIAGRAM_FACTS.md',
    'reports/R9_BACKGROUND_CV_AUDIT.json', 'reports/R9_BACKGROUND_CV_AUDIT.md',
    'reports/R9_REFERENCE_POOL_AUDIT.json', 'reports/R9_REFERENCE_POOL_AUDIT.md',
    'reports/R9_MODEL_ADAPTER_VERIFICATION.json', 'reports/R9_MODEL_ADAPTER_VERIFICATION.md',
    'reports/R9_TRAINING_ENGINE_VERIFICATION.json', 'reports/R9_TRAINING_ENGINE_VERIFICATION.md',
    'reports/R9_TREES_ENSEMBLE_VERIFICATION.json', 'reports/R9_TREES_ENSEMBLE_VERIFICATION.md',
    'reports/R7_R5_SUBMISSION_PATH_AUDIT_RESULTS.json', 'reports/R7_R5_SUBMISSION_PATH_AUDIT.md',
    'reports/R9_R6_PILOT_SUBMISSION.json', 'reports/R9_R6_PILOT_SUBMISSION.md',
    'configs/r9_full_training_v1.json',
    'reports/R9_DATASET_READY_VERIFICATION.json',
    'data/synthetic/group_cv_minute_full_r9_v1/manifest.json',
    'data/synthetic/group_cv_minute_full_r9_v1/readiness.json',
    'reports/R9_REAL_DATA_CONNECTION_VERIFICATION.json',
    'reports/R9_REAL_DATA_CONNECTION_VERIFICATION.md',
    'reports/R9_TRAINING_ENGINE_VERIFICATION_GLOBAL_FREQUENCY.json',
    'reports/R9_TRAINING_ENGINE_VERIFICATION_GLOBAL_FREQUENCY.md',
    'reports/R9_RUNNER_VERIFICATION.json',
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def load(name):
    return json.loads((ROOT/name).read_text())


def number(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    if isinstance(value, float):
        return f'{value:.12g}'
    return str(value)


MD_LINK = re.compile(r'(!?\[[^\]\n]*\]\()(<[^>\n]+>|[^)\n]+)(\))')


def local_link_target(target):
    """Split a local Markdown/HTML destination from fragment/line locator."""
    raw = html.unescape(target.strip())
    angled = raw.startswith('<') and raw.endswith('>')
    if angled:
        raw = raw[1:-1]
    if not raw or raw.startswith('#') or urlsplit(raw).scheme or raw.startswith('//'):
        return None
    path = raw.split('#', 1)[0].split('?', 1)[0]
    suffix = raw[len(path):]
    line = re.search(r':\d+(?::\d+)?$', path)
    if line:
        suffix = path[line.start():]+suffix
        path = path[:line.start()]
    return unquote(path), suffix, angled


def portable_markdown(source, destination):
    """Rebase copies only, using original logical paths (do not resolve data symlinks)."""
    changes=[]
    def replace(match):
        parsed=local_link_target(match.group(2))
        if parsed is None:
            return match.group(0)
        old,suffix,angled=parsed
        original_target=Path(os.path.abspath(source.parent/old))
        rebound=os.path.relpath(original_target,destination.parent).replace(os.sep,'/')+suffix
        changes.append({'original':match.group(2),'portable':rebound,
                        'original_target':str(original_target),'target_exists':original_target.exists()})
        return match.group(1)+('<'+rebound+'>' if angled else rebound)+match.group(3)
    return MD_LINK.sub(replace,source.read_text()),changes


def verify_portable_links(output_dir, assets):
    """Check every generated descendant Markdown/HTML link, not source files."""
    files=[output_dir/(STEM+'.md'),output_dir/(STEM+'.html')]
    files+=sorted(assets.rglob('*.md'))+sorted(assets.rglob('*.html'))
    checked=[];broken=[]
    for path in files:
        text=path.read_text()
        if path.suffix=='.md':
            links=[m.group(2) for m in MD_LINK.finditer(text)]
        else:
            links=re.findall(r'(?:href|src)=[\"\']([^\"\']+)[\"\']',text,re.I)
        for link in links:
            parsed=local_link_target(link)
            if parsed is None:continue
            target,suffix,_=parsed
            candidate=Path(os.path.abspath(path.parent/target))
            row={'file':str(path.relative_to(output_dir)),'link':link,
                 'target':os.path.relpath(candidate,output_dir).replace(os.sep,'/'),
                 'exists':candidate.exists()}
            checked.append(row)
            if not row['exists']:broken.append(row)
    return {'status':'passed' if not broken else 'failed','files_checked':len(files),
            'relative_links_checked':len(checked),'broken_count':len(broken),
            'broken':broken,'checked':checked}


def plot_aggregates(assets, facts, source):
    os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/r9_report_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    import numpy as np
    import pandas as pd
    fp = FontProperties(fname=FONT_PATH)
    plt.rcParams.update({'font.family': fp.get_name(), 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'normal', 'savefig.facecolor': 'white'})
    folder = assets/'figures'
    folder.mkdir(exist_ok=True)
    result = {}
    def save(key, fig):
        fig.tight_layout(pad=1.3)
        fig.savefig(folder/(key+'.png'), dpi=190)
        plt.close(fig)
        result[key] = f'{STEM}_assets/figures/{key}.png'
    site = pd.read_csv(ROOT/'eda/tables/site_summary.csv').sort_values('rows')
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(site.site, site.rows, color='#427886')
    for i, n in enumerate(site.rows):
        ax.text(n+8000, i, f'{n:,} ({n/1715779:.1%})', va='center', fontsize=8)
    ax.set_xlim(0, 1240000)
    ax.set_xlabel('Original source-export records (rows)')
    ax.set_title('SA source imbalance | 1,715,779 rows / 10 sites', loc='left')
    save('01_site_rows', fig)
    # Existing original-SA aggregate only; no Test observation/statistic use.
    miss = pd.read_csv(ROOT/'eda/tables/column_missingness.csv')
    common = ['TOL_'+str(x) for x in ['31.6','39.8','50.1','63.1','79.4','100','125.9','158.5','199.5','251.2','316.2','398.1','501.2','631','794.3','1000','1258.9','1584.9','1995.3','2511.9','3162.3','3981.1','5011.9','6309.6','7943.3']]
    all_bands=sorted([v for v in miss.column.unique() if v.startswith('TOL_')],key=lambda v:float(v[4:]))
    subset = miss[miss.column.isin(all_bands)]
    pivot = subset.pivot(index='site', columns='column', values='null_fraction').reindex(columns=all_bands)
    fig, ax = plt.subplots(figsize=(10, 5.1))
    image = ax.imshow(pivot.to_numpy()*100, aspect='auto', cmap='Blues', vmin=0,vmax=100)
    ax.set_yticks(range(len(pivot)), pivot.index, fontsize=8)
    tick_positions=range(0,len(all_bands),5)
    ax.set_xticks(list(tick_positions), [all_bands[i][4:] for i in tick_positions], rotation=45, ha='right', fontsize=8)
    ax.axvline(all_bands.index(common[0])-.5,color='#32725d',ls='--',lw=1.3)
    ax.axvline(all_bands.index(common[-1])+.5,color='#32725d',ls='--',lw=1.3)
    ax.set_xlabel('All62 source TOL centres (Hz); dashed lines delimit the common25 bands')
    ax.set_title('Original-SA band availability | source columns, not temporal gaps', loc='left')
    fig.colorbar(image, ax=ax, label='Null cells / site rows (%)', pad=.02)
    save('02_missingness', fig)
    band = pd.read_csv(ROOT/'eda/tables/common_band_global_stats.csv')
    band = band[band.dataset == 'training_all'].sort_values('frequency_hz')
    if len(band) != 25:
        raise ValueError('Expected original-SA training_all25 aggregate bands')
    fig, ax = plt.subplots(figsize=(10, 4.9))
    x = band.frequency_hz.to_numpy()
    ax.fill_between(x, band.p10.to_numpy(), band.p90.to_numpy(), color='#c6dcd8', alpha=.9, label='p10-p90')
    ax.plot(x, band.p50.to_numpy(), color='#24586b', lw=2, label='Median')
    ax.set_xscale('log')
    ax.set_xlabel('Band centre (Hz; logarithmic axis)')
    ax.set_ylabel('Source-export dB (uncalibrated convention)')
    ax.set_title('Original-SA band profile | descriptive aggregate, not vessel-only power', loc='left')
    ax.legend(frameon=False)
    save('03_band_profile', fig)
    fig, ax = plt.subplots(figsize=(10, 4.3))
    labels = ['VT refs: header', 'VT refs: complete minutes', 'VT ambient controls', 'HearMyShip unique', 'NOAA primary extension']
    hours = [119388/3600, 118680/3600, 31980/3600, 6.382676898148148, 3.]
    colors = ['#427886','#71a298','#b0bdb4','#d6c6ab','#d6c6ab']
    ax.barh(labels[::-1], hours[::-1], color=colors[::-1])
    for i, h in enumerate(hours[::-1]):
        ax.text(h+.3, i, f'{h:.3f}h', va='center', fontsize=9)
    ax.set_xlim(0,39)
    ax.set_xlabel('Sum of clip durations (not independent field hours)')
    ax.set_title('External source inventory | R9 positives use VT reference pool only', loc='left')
    save('04_source_hours', fig)
    folds = facts['background_split']['folds']
    roles = ['train','selection','ensemble_fit','ensemble_gate','report']
    fig, ax = plt.subplots(figsize=(10,4.8))
    left = np.zeros(5)
    for role, color in zip(roles,['#427886','#a3c6b6','#dac69c','#c99573','#756a91']):
        values = np.array([f['window_counts'][role] for f in folds])
        ax.barh(np.arange(5), values, left=left, label=role, color=color)
        left += values
    ax.set_yticks(np.arange(5),[f'Fold {f}' for f in range(5)])
    ax.set_xlim(0,103000)
    ax.set_xlabel('Unique allowed background windows after minute QC and24h purge')
    ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(.5,-.15), frameon=False, fontsize=8)
    ax.set_title('R9 actual source-window roles | generated-example counts differ', loc='left')
    save('05_fold_roles', fig)
    fig, ax = plt.subplots(figsize=(10,4.3))
    names = ['Wide CNN','TF-grid + FDY + Conformer','FDY-CRNN adaptation','iTransformer adaptation']
    params = [facts['neural_models'][k]['total_parameters'] for k in ['bg_cnn','tfgrid_fdy_conformer','fdy','itransformer']]
    ax.barh(names[::-1], np.array(params[::-1])/1e6, color=['#a3c6b6','#a3c6b6','#427886','#427886'])
    for i,n in enumerate(params[::-1]):
        ax.text(n/1e6+.025,i,f'{n:,}',va='center',fontsize=9)
    ax.set_xlim(0,3.1)
    ax.set_xlabel('Trainable parameters (millions); all fresh initialization')
    ax.set_title('R9 actual adapter sizes | size is not a performance ranking',loc='left')
    save('06_model_parameters',fig)
    return result


def sections(f, charts, pilot):
    table = lambda head, rows: {'head':head,'rows':rows}
    s=[]
    def add(title, lead, paragraphs=(), **kwargs):
        s.append({'title':title,'lead':lead,'paragraphs':list(paragraphs),**kwargs})
    add('데이터·EDA·전처리 근거 보고서', 'French Embassy Underwater Vessel Noise Challenge | R4-R9 version audit', [
        '무엇을 예측하는지, 어떤 실측 자료를 사용했는지, 결측과 단위가 어떻게 처리되는지, 합성 정답이 어디서 생기는지를 실제 파일·코드·보존된 집계표와 연결한 보고서입니다.',
        '현재 R9는 원천 풀·분할·샘플러·모델·학습 엔진의 구현 및 작은 CPU 숫자 검증을 마친 단계입니다. R9 전체5fold100epoch GPU 본학습과 새 production 추론은 아직 실행하지 않았습니다.',
        '본문 뒤에 기존 EDA 지표305개 전부를 수록합니다. 원 집계표26개·4,137행은 portable companion CSV/HTML에 원본 그대로 포함하며 파일별 분모·SHA를 함께 보존합니다.',
        '작성 기준:2026-10-26. 공식 평가·자체 합성 검증·예측값의 서술 통계를 엄격하게 구분합니다. Test 재스캔·재학습·분포 맞추기와 신규 외부 전송은 하지 않았습니다.',
    ],kind='cover')
    add('읽는 순서와 증거의 의미','수치보다 먼저 분모와 버전을 확인합니다.',[
        '1단계는 TASK와 source inventory,2단계는 원 SA EDA와 waveform 변환,3단계는 버전·분할·합성 정답,4단계는 특징·학습·앙상블·추론입니다. 마지막은 실행 상태와 감사 결과입니다.',
        '원 관측 행 수, 고유 source window 수, 합성 draw 수, 여러fold의 역할 합은 서로 다릅니다. clip duration의 합도 독립적인 비중복 field-recording hours가 아닙니다.',
        'verified는 해당 기록·코드·정합성 근거가 확인됐다는 뜻입니다. unknown은 원인·권한·정답 등 미확인 사항이고, datasetdependent는 분모·처리·버전에 따라 달라집니다. 내부 수식 정합성이 공식 분포에서의 성능을 증명하지 않습니다.',
        '부록의 과거 Test-derived 집계는 보존된 연구 이력입니다. 이 보고서에서 다시 계산하거나 생성기·특징·가중치 설정의 목표로 사용하지 않았습니다.',
    ])
    add('TASK: 정답은 어떤 에너지인가','자연 상태 선박의 존재 라벨이 아니라, 인위적으로 추가한 측정 recording의 알려진 에너지 비율입니다.',[
        '공개 SA 데이터는 실제 수중 관측값이지만 vessel-only 에너지 정답, 원 오디오, AIS 정답이 포함되지 않습니다. 따라서 원171만행을 모델로 추정해 만든 score를 검증된 선박 정답으로 바꿀 수 없습니다.',
        '합성 예제에서는 배경 B와 추가 recording A를 선형 에너지로 알고 있습니다. 같은 관측 mask O에서 E=B+A, cell target q=A/E, window target y=합(A)/합(E)를 계산합니다. 원 배경에도 자연 선박음이 포함될 수 있으며 y는 그것을 모두 분리한 비율이 아닙니다.',
        '추가 reference도 주변 소음이 포함된 whole recording입니다. AIS weak class나 Ship annotation만으로 그 에너지를 순수 vessel-only로 확정하지 않습니다. presence는 관측 위치에 실제 추가된 성분이0보다 큰지를 나타냅니다.',
        '공식 주 평가는 share RMSE이며 leaderboard 점수는1-RMSE입니다. diagnostic score의 AUC와 주 순위 손실은 구분합니다. 점수0.90의 필요조건은 공식 RMSE0.10이지 개별 모델별 임의 목표 구간이 아닙니다.',
    ])
    add('원천·라이선스·사용 범위','수집돼 있는 데이터와 현재 R9의 학습 입력을 구분합니다.',[
        'SA는 대회 제공 source-export band summaries입니다. organiser-review README는 새로운 재배포 권한을 부여하지 않으며 압력 보정·UTC·명목 sample-rate 토큰의 해석도 미확인 사항을 보존합니다.',
        'VTUAD rebuilt v1.3.2는 ONC hydrophone/AIS 시나리오 구간의 strict view입니다. CC BY4.0의 attribution·버전·수정 고지를 유지합니다. strict와 과거12개 class archive는 같은525시나리오의 다른 cut이며 독립 자료로 이중 집계하지 않습니다.',
        'HearMyShip·NOAA는 수집·검토 기록이 있지만 R9의509-positive reference 풀에 섞이지 않습니다. 원천 라이선스가 SA나 다른 기관 데이터에 자동으로 적용되지 않습니다.',
        '대회 외부 데이터 사용은 공개 이용 가능성·라이선스·대회 고지 조건을 함께 확인해야 합니다. 이 보고서 작성 과정에서 계정 게시나 외부 연락은 하지 않았습니다.',
    ],table=table(['원천','현재 확보/역할','시간 의미'],[
        ['SA','1,715,779행 /10sites','dur 합28,358.3308h; 독립획득시간 아님'],
        ['VTUAD','525=509reference+16controls','header42.0467h; active native-minute32.9667h'],
        ['HearMyShip','1,145 unique WAV; R9참조0','6.382677h,10-30s; 완전1분0'],
        ['NOAA extension','12primary clips/6deployments; R9참조0','3h; 별도reserved/sealed관리'],
    ]))
    add('원본 SA 스키마와25대역','source ID와 provenance는 분할·감사에 쓰며 모델 predictor에 넣지 않습니다.',[
        'Parquet10개는 각74열로 동일한 schema를 가지며 TOL62열과 metadata/QC12열을 포함합니다. 모델에는 공통25band만 사용합니다. 다른37개 TOL열을 값이 없다는 이유로 합성하거나 모델의 현재25band와 혼합하지 않습니다.',
        '공통 중심은 c_i=10^(1.5+0.1i),i=0..24입니다. 적분 지지 구간은[c_i/10^0.05,c_i*10^0.05)이며 전체 약28.18-8912.51Hz입니다. 명목 centre의 최대7943.3Hz와 upper edge를 혼동하지 않습니다.',
        '모델 격자는 시간15 x 주파수25이며 source site/recorder/time/ID와의 join은 별도 metadata에 남깁니다. 파일명 sample_rate_hz는 측정 WAV header가 아닙니다.',
    ],table=table(['index','centre Hz','index','centre Hz'],[
        [i,f'{10**(1.5+.1*i):.1f}',i+13 if i+13<25 else '',f'{10**(1.5+.1*(i+13)):.1f}' if i+13<25 else ''] for i in range(13)
    ]))
    add('SA 결측과 초기 QC','결측은 사용할 수 없는 관측값이며 silence가 아닙니다.',[
        '초기 minute 적격 조건은 공통25band finite, sentinel -156.5 미포함,55<=dur<=65, timestamp초<=5입니다. 적격1,698,224행, 실제 창 포함1,698,184행으로 단계별 분모가 다릅니다. 원1,715,779행은 보존됩니다.',
        '같은 minute의 중복은 duration이60초에 가까운 행, 이른 timestamp, localindex 순으로 하나를 선택합니다. 실제 export-clock15분 빈의 원 위치에 넣고 최소3observed분을 요구합니다. 공백을 압축·보간·반복하지 않습니다.',
        '후속 boundary QC는 원row_id와 실제window/minute를 join해 해당 분만 추가 NaN으로 가립니다. 창의 첫/끝 분을 무조건 지우지 않습니다. flag의 물리적 원인이 확인됐다는 뜻도 아닙니다.',
    ],image=charts['02_missingness'],caption='원본 SA 집계표의 site x common-band null fraction. recorder별 지원·운영 차이를 모두 물리적 선박 차이로 해석하지 않습니다.')
    add('SA 사이트 불균형','모든 source-export 행을 같은 독립 사례로 볼 수 없습니다.',[
        'Richards Bay가 원 행 수의 큰 비중을 차지합니다. site별 관측 기간·recorder duty cycle·결측·sample/export 설정이 다르므로 단순 전체 평균은 가장 많은 원천에 영향을 많이 받습니다.',
        'source grouping과 support purge는 같은 장치·가까운 시간대의 유사 창이 train/validation에 섞이는 것을 줄이기 위한 조치입니다. 다만 이것으로 알려지지 않은 reference·generator·새 해역으로의 일반화를 보장하지는 않습니다.',
        'R9는 배경 원창을 복제하는 oversampling 대신 각 epoch 원창을 한 번 쓰는 설계입니다. site-balanced objective는 별도 loss_weight를 공급해 구현하며, sampling 분포 변경과 구분합니다.',
    ],image=charts['01_site_rows'],caption='기존 site_summary.csv의 원 관측 행수. 행수는 독립 녹음수·합성 예제수·고유 hull수가 아닙니다.')
    add('대역 분포·단위·희귀 이벤트','높은 에너지를 자동 outlier로 제거하면 목표 회귀를 악화시킬 수 있습니다.',[
        'TOL은 source-export dB convention입니다. 시간 평균·주파수 합산·mixture는 dB를 그대로 더하지 않고10^(dB/10)의 선형 power에서 수행합니다. 장치별 gain·교정이 확인되지 않아 같은 수치가 보정된 pressure-unit이라고 주장하지 않습니다.',
        '높은 LF broadband bump, 큰 share, 드문 선박 통과 같은 자연적 희귀 현상은 값이 크다는 이유만으로 오류가 아닙니다. measurement invalid와 natural rare event를 구분하고 filtering은 TRAIN-only 기준 및 고정 validation ablation으로 판단해야 합니다.',
        '그림은 보존된 training_all 분위수입니다. 음수 sentinel과 recorder 차이가 포함된 기술 통계이며 vessel-only spectrum이나 물리적 수중원인 분해가 아닙니다.',
    ],image=charts['03_band_profile'],caption='기존 common_band_global_stats.csv에서 training_all25행만 사용했습니다. p10-p90 envelope와 median은 모델의 실제 성능 곡선이 아닙니다.')
    add('기록 시간·duty cycle·장치 caveat','export clock과 시간 합은 획득 시각의 물리적 검증을 대신하지 않습니다.',[
        '기록은2015-04-21부터2025-12-06까지의 export clock을 포함하지만, 모든 장치가 그 기간을 연속 녹음한 것이 아닙니다. 원 metadata는17site-recorder coverage entries와12recorder identifiers를 보여주며 이 수를17독립물리장치로 바꾸지 않습니다.',
        'dur의 범위는1-360초이고36,234행은60초와 다릅니다. dur 합102,089,991초=28,358.3308h는 원export행의 합계이며, 녹음 중복·불연속 duty·획득 기간을 독립 검증한 시간 수치가 아닙니다.',
        '명목12k sample token을 가진 recorder에서136,518행의6.3/7.9k bands가 finite인 미해결 사례가 있습니다. finite QC flag만으로 Nyquist support나 실제 sample rate를 인증할 수 없습니다. 원 값을 보존하고 nominal-band warning을 diagnostics에 남깁니다.',
        '일주기·월별·ACF는 관측된 구간과 recorder 운영의 영향을 받습니다. 대회 Test의 숨은 site나 timestamp를 이러한 패턴으로 역추정하는 근거로 쓰지 않습니다.',
    ])
    add('VTUAD 전체 풀과 시간의 분모','R9의 확장509개는 새로운 독립 해역509개가 아닙니다.',[
        '엄격한 vessel archive509개는 전체33recording days·178original hull IDs의 시나리오입니다. 공유 receiver/시간/원천을 연결한 전체 graph는1component입니다. 모든fold가 이509개를 공유하므로 reference-independent CV가 아닙니다.',
        'header119,388초 중 완전2초 bins는119,376초입니다. 다시 완전60초로 묶은1,978분만 minute-first sampler에 사용할 수 있어 실제 native-minute 합은118,680초=32.9667h입니다. 단편 tail을 padding·반복해 늘리지 않습니다.',
        '509개 중 weak_class=background22개도 사용자 all509 정책에 따라 포함합니다. 이때 target은 실제 추가된 whole recording의 에너지이며 weak class에서 q나presence를 가져오지 않습니다. dedicated ambient16개는 별도controls이고 자동 y0가 아닙니다.',
    ],image=charts['04_source_hours'],caption='획득된 source inventory의 시간 분모. R9 주 참조는 VT509이며 HearMyShip와NOAA는 현재 active R9 positives에 포함되지 않습니다.')
    add('외부 자료의 확보와 현재 미사용 범위','다운로드·검토·현재 학습 사용을 별도 상태로 기록합니다.',[
        'HearMyShip는1,146verifiedentries에서 중복MD5를 제외한1,145uniqueWAV입니다. total6.382677h, 파일10-30초여서 measuredcomplete-minute0입니다. 이를 반복해15분 관측으로 부르지 않습니다. 과거R2에서는 일부를 사용했지만 현재R4/R5/R6/R9참조에는0개입니다.',
        'NOAA SanctSound는 annotation·Good frequency support·generation-pinnedrange·cropframes·SHA 근거를 가진 primary12clips,6deployments,12days,3h를 확보했습니다. PM NoShip1개는 quarantine입니다. annotationShip는 isolatedvesselenergy정답이 아닙니다.',
        '기존 sealed 자료는 예측·학습·튜닝에 다시 열지 않았습니다. 검색만한PORTIA/VLIZ 등 공개후보의 metadata와 실제 획득 음향값을 혼동하지 않습니다. 현재 새509확장은 이미 확보된 strictVTpool을 새production에서 허용하는 변경입니다.',
        '각 원천의 native support·duration·calibration·annotation은 다릅니다. 공통 선형power계산을 적용해도 waveform 길이, crop, provenance, source isolation이 모두 동일해지는 것은 아닙니다.',
    ])
    add('공통 waveform-to-power 변환','audio branch에만 waveform 처리가 있고 SA는 이미 band summary입니다.',[
        'VTWAV는 header32kHz,channel0을 선택하고 완전2초 frame별 Hannperiodogram을 계산합니다. constant detrend,one-sidedPSDdensity를 사용합니다. targetband내PSD의합에frequencyspacing을 곱해power를 적분합니다.',
        'targetband upperedge가Nyquist를 넘으면 그 band를 사용 가능하다고 채우지 않습니다. native sample rate를 유지하며 upsampling으로 지원 대역을 만들어내지 않습니다. 완전2초미만 tail은 버리고,30bins의 선형power평균을1분으로 집계합니다.',
        'SA의 source-exportdB는 원오디오를 복원하지 않고 선형energy로 변환합니다. native reference의 digitalmean-square와 SA exportpower의 절대 단위가 교정돼 같다는 주장은 하지 않으며, 합성 scale은 사전 고정된 상대 SNR로 정합니다.',
        '로그전후의 순서가 중요합니다. 평균(logpower)와log(평균power)는 다르며, R9 minute-first는 후자를 기반으로 knowncomponent를 구성합니다.',
    ])
    add('R4·R5·R6·R9 버전별 규모','저장 행 합계와 실제 한 모델이 본 데이터가 다릅니다.',[
        'R4의297,837은5foldTRAIN267,837+INNER15,000+OUTER15,000의 저장 역할 합계입니다. 원천이fold사이순환하므로297,837개의 독립 fieldrecording이 아닙니다. canonical15NPZ414.21MiB, 내부NPYmember854.40MiB입니다.',
        'R5CNN은Fold0TRAIN53,244개 배경을 매epoch동적합성하고 fixedINNER3,000으로 checkpoint를 선택했습니다. 전체297,837개를 한꺼번에 학습한 것이 아닙니다.',
        'R6는 TRAIN8,192/INNER1,024와cross3패널각1,024를 고정 생성했습니다. R4/R6대조법의 중복 저장까지 합한21,504rows와 sourceunique는 구분합니다. NPZ40.90MiB/내부NPY128.69MiB입니다.',
        'R9source는100,965허용배경/509refs로 확장했지만 새production합성NPZ·실제학습결과는 아직 없습니다. plannedfixedroleexamples와source원창수,100epoch의 examplevisits를 새로운 독립 데이터로 더하지 않습니다.',
    ],table=table(['버전','배경/예제·참조','현재 상태'],[
        ['R4','297,837stored; VT119/22components','구축·전수정합성검사완료'],
        ['R5Fold0','53,244dynamic/epoch; INNER3,000','기존CNN학습·제출경로감사'],
        ['R6','8,192fixedTRAIN; VT94/17groups','작은CPU모델·fixed100완료'],
        ['R9','78k대TRAIN/fold; VT509shared','원천/코드/숫자fixture; fullrun미실행'],
    ]))
    add('R6-relative8: 네 NN의 공통 입력','원 관측 창 하나만으로 계산하고 label·sourceID·TRAINprior를 받지 않습니다.',[
        'Q=round(db.astype(float64),1),O는finite mask입니다. med_f/q20_f/q05_f는 각band의 관측 시간 통계이고 med_win은전체관측cell의median입니다. 결측feature와w는0, 별도mask채널로결측을알립니다.',
        '공통gain은 반올림 후 수학적으로 상쇄됩니다. 반올림 전 임의gain은 경계값을 넘을 수 있어 완전히 동일하다고 주장하지 않습니다. staticbandoffset도profile채널4와energyweights에는 남습니다.',
    ],table=table(['채널','정의'],[
        ['0','O*(Q-med_f)/10'],['1','O*(Q-q20_f)/15'],['2','관측인접pair의delta_db/10; 첫분0'],
        ['3','f10..24:relu(channel2)*tanh(low0..8positive-delta평균/3)'],['4','O*(med_f-med_win)/20'],
        ['5','O:observed-cellmask'],['6','O*(Q-q05_f)/15'],['7','sqrt(w_rounded)'],
    ]))
    add('기존 R5 8채널과 달라진 점','모양이 같아도 입력 의미는 다릅니다.',[
        'R5legacy8은 centeredlevel/20,absolutelevel/100,sqrt(rawenergyweight),observedmask,temporalq20excess/20,adjacentdelta/20,backgroundpriorq0,log1p(Bhat/E)/log11이었습니다.',
        'TRAIN의 knownB로 만든 aggregatebackgroundprior는 inference에 label을 직접 주는 것이 아니지만, 실제 background truth도 아닙니다. R6-relative8은 그prior와absolutelevel을 제거하고q05/sqrt(roundedw)를 동일window에서 계산합니다.',
        'R9어댑터는 채널5가binarymask이고 채널7이동일w의sqrt인지 검사하므로, 기존R5의8채널 텐서를 같은모양이라는이유로받지않습니다. R5checkpoint를 R9initialweights로 재사용하지 않습니다.',
        'R5와R6 모두 이미 directsigmoidmask/physicalpool 및share+.25cell+.05BCE를 사용했습니다. R6개선이lossweight변경이나출력prioranchor제거에서새로생겼다고주장하면틀립니다. generator/source/input/size/split등이함께변경됐습니다.',
    ])
    add('물리27개와 LightGBM3,827열','서로 다른 버전의1,324·3,827·27을 혼동하지 않습니다.',[
        'R6physics27은9values+9definedflags+9valid-minutecounts입니다. group의모든band가있는분만사용하고regionallevel은10log10(sumlinearenergy)입니다. 미정의값0을정의flag와함께둡니다.',
        '값9개는low/mid/high/safebump4개,low-mid/high/safePearson3개,broadbandpopulationstd,median(high-low)tilt입니다. bump는max-median>=2validminutes, correlation은>=3pairedminutes와양쪽nonconstant를요구합니다.',
        'LightGBM은relative8flatten3,000+timefeaturestats800+physics27=3,827입니다. 800통계는0으로채운grid의일반mean/std/max/min이므로observed-onlynan통계가아닙니다.',
        '기존R5tree1,324는fixed1,258+physics45+TRAINprototype8+추가physics13의다른분기입니다. R7Ridgecontrol27은physics만사용한간단한대조군입니다.',
    ],table=table(['그룹','0-basedbandindices','실제centre범위Hz'],[
        ['low','0..8','31.6-199.5'],['mid','10..19','316.2-2511.9'],['high','20..24','3162.3-7943.3'],['safe','15..22','1000-5011.9'],
    ]))
    add('R9 5fold 배경 분할과24h purge','원천 단위로 나누고, 겹친 support를 제거합니다.',[
        '원래허용100,965창과258sensor-weekgroups의bucket을유지합니다. heldout1bucket을뺀4개bucket이TRAIN으로purge전약80/20이며random-rowKFold가아닙니다.',
        'heldoutwholegroups를SHA순서/사전정한largest-remainderquota로selection50%,ensemblefit20%,gate15%,report15%로나눕니다. 행비율·site비율까지일치시키지않으며작은site가일부역할에없을수있습니다.',
        '같은sensor의900ssupport사이24hgap을요구합니다. report>gate>fit>selection우선순위를모델결과없이적용하고TRAIN은heldout전체에대해purge합니다. 모든rolepair의24h미달/index/groupoverlap은0입니다.',
    ],image=charts['05_fold_roles'],caption='실제source-window 역할 수. 별도고정합성예제는fold마다selection1000/fit500/gate500/report1000계획입니다. reference509개는모든fold공유합니다.')
    add('R9 minute-first 합성 데이터','전체 native trajectory peak로 상대세기를 정하고 crop·mask는 별도로 적용합니다.',[
        '2초선형power를먼저60초완전minute로집계합니다. 그다음smoothfrequencyEQ<=6dB,선택·crop·배치를적용합니다. timewarp,강제boundaryfade,waveform반복과gapfill은없습니다.',
        'native_inside는최대15분연속crop을창안에배치하고native_short는실제1-3분부분을사용합니다. native_edge는전체trajectory를창경계에걸치게배치합니다. 1분source는정수minute경계를걸칠수없어기록된fallback을사용합니다.',
        'alpha=10^(SNR/10)*median_observed_minute(sumB)/full_native_EQ_peak입니다. SNR은Uniform[-10,+10]dB이며fullnativepeak는crop/mask전값입니다. 그러므로최종관측창의peakSNR나globalshare가요청SNR와같을필요는없습니다.',
        '설정은inside/short/edge=.5/.25/.25,explicitno-source20%입니다. referenceindex는epoch사이고정permutation을이어cycling하며N>=509면한cycle에서모두등장합니다. 이는한acoustic event내source반복이아닙니다.',
    ])
    add('정답·mask·반올림의 정확성','예측 풀링 제약과 raw 정답 풀링 오차를 별도로 검사합니다.',[
        '관측O밖B/A는0입니다. E=B+A,cellq=A/E,sharey=sumA/sumE를float64원성분에서계산합니다. no-source는A=0이고,양성요청이어도실제추가구간이전부mask되면관측presence0/y0가가능합니다. 이를기피하는재시도는하지않습니다.',
        '모델은Q=round(rawmixturedb,1)의energyweights를사용하고q/y는rawcomponent기준을유지합니다. 따라서sum(w_raw*q)=y는정의상성립하지만sum(w_round*q)는일반적으로y와다릅니다.',
        'q∈[0,1]에서nominal0.05dBround오차가만드는최대poolinggap은tanh(ln10*0.05/20)≈0.0057564입니다. sampler는float32halfULP를추가한상계를기록합니다. 저장q/y의float32오차도별도수치검사대상입니다.',
        '모델예측은항상yhat=sum(w_round*qhat)로계산하며FP32오차허용3e-6내에같아야합니다. knownrawlabel을억지로roundedqpool값으로다시정의하지않습니다.',
    ])
    add('R9 모델 크기와 실제 연결','네 신경망 모두 fresh weights이며 대형일수록 낫다는 보장은 없습니다.',[
        'WideCNN은width128,5residualblocks입니다. R9에서각channel/minute의observedfrequencymean을hidden에더하는parameter-freeglobalresidual을추가했습니다. 전체25band문맥이qhead에전달되며전결측분context는0입니다.',
        'TF-grid+FDY+Conformer는E64,H128,grid5blocks,Conformer1block입니다. fullbandfrequencyBLSTM,subbandtimeBLSTM,cross-frameattention,frequencydynamicconvolution과Conformer를15x25energygrid에맞게조합한taskadaptation입니다.',
        'FDY는width32/3dynamicblocks/BiGRU, iTransformer는width96/3layers/4heads와25bandtokens입니다. 원논문의waveform/complex-STFT출력과다른모델이므로공식fullSOTA완전재현이라부르지않습니다.',
    ],image=charts['06_model_parameters'],caption='별도 learned presence head는제거했습니다. Wideglobalcontext추가params0개이며counts는현재R9어댑터의실제전체=학습가능값입니다.')
    add('학습 루프·loss·checkpoint 정책','실제100epochs를 끝까지 실행하되 선택 기준과 last 상태를 함께 남깁니다.',[
        'NN정책은Adam lr0.001/weightdecay0.0003/dropout0.1/gradientclip1,exact100epochs/noearlystopping입니다. ReduceLROnPlateau는selectionwindowRMSE를보고factor0.5/patience2/minlr1e-6으로학습률을줄이지만학습을중단하지않습니다.',
        'perrowloss=shareMSE+0.25*sum(w*(qhat-q)^2)+0.05*BCE(derivedscorelogit,presence)입니다. source/sitebalance는양의loss_weight를곱해objective를바꾸며샘플을복제하는것과다릅니다.',
        'onlineTRAINRMSE는각minibatch갱신도중의unweighted통계이고fixedTRAINevaluation과같지않습니다. selection은고정값/순서/hash를매epoch대조합니다. bestwindow/bestmacro/last/every10epoch를기록합니다.',
        '코드·입력·architecture·seed·precision·sourceidentity를checkpoint에묶습니다. last에는optimizer/scheduler/RNGstate도남기지만기존partialrun의자동resume는거부합니다. CPU2epoch숫자fixture는production100epoch실행증거가아닙니다.',
    ])
    add('LightGBM와 안전한 앙상블','미리 정한 후보 하나만 gate에서 확인합니다.',[
        'CPUtree는regressor+presenceclassifier,최대1500boostingrounds/ES40/threads4입니다. selectionclippedshareRMSE와classifierlogloss를각각사용합니다. TRAIN100kcap를넘으면자동subsample하지않고거부합니다.',
        'ensemblefit에서bestNNsingle을fallback으로고정합니다. 전체5모델equalaverage,NN4equalaverage,LightGBMsingle중fit에서가장좋은후보하나를정하고source-disjointgate에서fallback보다RMSE0.0001이상나을때만채택합니다.',
        '실패하면이미fit에서고른fallback으로돌아갑니다. gate를보고다른후보를반복시험하거나가중치를다시최적화하지않습니다. Tree가fit에서만좋다는이유로fallback경로로gate를우회하지않습니다.',
        '최종weights는nonnegative/simplex,affine[1,0]입니다. gate는모델선택의일부이므로blindtest가아닙니다. report는모든fold의checkpoint와policy를고정한후열어최종개발CV를보고하며설정을다시선택하지않습니다.',
    ])
    add('R5 INNER와 LB 차이: 확인한 것과 모르는 것','재현 성공은 데이터 대표성이나 숨은 generator 정합의 증명이 아닙니다.',[
        'R5의epoch19원checkpoint/TRAINprior/정책으로CPU2재현한INNER RMSE는0.090808147081입니다. 저장CPU4와bitwise일치했고GPU BF16기록은0.090849261020입니다. parser375열/시간·주파수/ID/maskroundtrip과동일입력예측이검증됐습니다.',
        '원handoffCSV의SHAe32b04...c83fe는로컬저장rawpredictions와동결정책에서정확재현됐습니다. 다만당시실제Zindi업로드파일bytes의SHA는독립확인되지않았습니다.',
        '사용자가보고한R5공식점수0.65199132는RMSE0.34800868에해당해자체INNER보다훨씬큽니다. 이차이를추론구현누락,작은모델,source/generator조건차이중하나만의원인으로확정하지못했습니다.',
        'R6fixed100globalCNN의bestepoch31자체INNER0.0593773/crossgenerator0.0606567/crosssensor0.0950611/joint0.1104068은같은개발프로젝트의합성패널결과입니다. officialLB와동일분포또는독립새holdout성능이라고부르지않습니다.',
    ])
    add('최신 로컬 R6 예측 요약의 한계','RMS는 RMSE가 아닙니다. 정답 없는 출력 통계로 성능을 판단하지 않습니다.',[
        '기존에사전동결한R6epoch31localpilot에서1,500행추론을실행한보존receipt가있습니다. 이보고서는그receipt만인용하고Test배열을새로열지않았습니다. 이pilot은R9신규5fold모델결과가아닙니다.',
        f"기존receipt의R6share mean={pilot['statistics']['all']['output_summary']['R6_share']['mean']:.6f}, RMS={pilot['statistics']['all']['output_summary']['R6_share']['RMS']:.6f}입니다. R5share RMS={pilot['statistics']['all']['output_summary']['R5_share']['RMS']:.6f}보다작다는것은출력크기의서술이지정답오차개선의증거가아닙니다.",
        '출력RMS를0.22-0.27같은임의범위로맞추거나Benchmark를정답의2차moment로해석할수없습니다. 상수0.25도RMS0.25가되지만실제label을설명하지못할수있습니다.',
        '관측누락형태는site정답이아닙니다. StFrancis같은숨은Test원천을ID/누락/cohort패턴으로지정하거나그원천에맞춰generator를튜닝하지않습니다. localpilot는업로드하지않는diagnostic이며새공식점수는없습니다.',
    ])
    add('추론·후처리의 새 계약','고정된 모델과 per-window 변환만 사용합니다.',[
        'publicschema는ID와m01_TOL_31.6부터m15_TOL_7943.3까지의명시적375열순서입니다. minute-major15x25로파싱하고ID순서를유지합니다. rawquerylabel이나sourceID는요구하지않습니다.',
        'perwindowQ/relative8/w를계산해각fold의frozen4NN/CPUtree를실행하고그fold의동결convexpolicy로share와diagnosticscore를합칩니다. 모델·입력policy·코드SHA가다르거나fixture/partialcheckpoint면거부해야합니다.',
        '5fold평균같은foldaggregation은production실행전에정해고정해야합니다. 현재R9config에는최종foldaggregation이아직pin되지않았으므로이보고서에서완료된구현이라주장하지않습니다.',
        '마지막은범위[0,1],finite,ID/행/열/순서와CSVroundtrip을검사하고해시receipt를남깁니다. Test에서fit/calibration/pseudo-label/generator조정/LB계수튜닝은없습니다. outputscore는diagnostic이며별도정답확률보정이아닙니다.',
    ])
    add('현재 실행 상태와 다음 gate','source readiness, operator fixture, full training을 분리합니다.',[
        '완료:SA허용pool분할·실제minuteQC·24hsupport감사,VT509+16manifest·cycleschedule,minute-firstsampler구현,NN4개CPU1Adamupdate·저장복원fixture,trainingengineCPU2epochfixture,tree10roundfixture·ensemblegate검증입니다.',
        '미실행:R9productiondynamicdataset/fixedroleNPZ전수검증,full5fold×4NN×100epochsCUDA본학습,production5foldtree/ensemblefit·gate·report와새R9inference입니다. builder/report시점의상태snapshot이며미래실행결과를앞당겨기록하지않습니다.',
        'WideCNN의parameter-freefrequencyglobalresidual은본학습전추가했고4모델oracle/backpropfixture를다시통과했습니다. root가기존R9sourcefreeze를명시적으로열어수정했으며R5/R6은불변입니다.',
        '실제production전에는새dataset/no-source/family비율·every-background-once·siteweight·mask·라벨·sourcecodebinding을통합검증해야합니다. GPU BF16와worker4실행도별도현장검증대상입니다.',
    ])
    for key,title in [('preprocessing','상세 이미지 1: Data Preprocessing'),('training','상세 이미지 2: Training'),('inference','상세 이미지 3: Inference')]:
        source_image = ROOT/f'figures/r9_pipeline_{key}.png'
        add(title,'원천·수식·단계·검증 gate를 분리한 상세 파이프라인.',[
            '이 도식은 현재 설계/구현과 예정된 production 실행을 정리합니다. 모형 내부 차트가 있다면 실제 학습 곡선·공식 성능을 나타내지 않는 schematic으로 읽어야 합니다.',
            '도식의 source count와 버전은 본문의 수치표·JSON을 기준으로 확인합니다. R9 shared-reference caveat와 fullrun미실행 상태는 그림의 간결한 표현보다 우선하는 해석 조건입니다.',
        ],pipeline_key=key,pipeline_source=str(source_image),kind='pipeline')
    add('전체 근거와 재현·보존','305개 지표와26개 표를 빠짐없이 확인할 수 있습니다.',[
        '부록A는기존R8metric_catalog305entries를모두표시합니다. 정확한value/unit/denominator/source/status/version/caveat/evidence_mode는companionJSON/CSV에도그대로포함합니다. 긴값을줄여새수치로바꾸지않습니다.',
        '부록B는26개aggregate table의분모·행수·SHA와portableCSV/HTML경로입니다.4,137행전체는companion에보존됩니다. 과거Test-derivedtable은archival로명시하고새학습설정근거로사용하지않습니다.',
        '원R8SA/external/features보고서·JSON과R9분할/모델/engine/ensemble/replayreceipt를같은assets/evidence에복사해문맥을보존합니다. 원파일은수정하지않으며사본SHA가일치해야합니다.',
        '외부metadata는공식대회data페이지,VTUADDOI10.5281/zenodo.23037895,각원천기록의URL을확인할수있습니다. 이PDF의성능논의는독립공식점수예측이나보장으로해석하지않습니다.',
    ])
    # Late source-ready receipts supersede the earlier implementation snapshot.
    # Keep report wording synchronized without changing any source artifact.
    by_title={p['title']:p for p in s}
    by_title['데이터·EDA·전처리 근거 보고서']['paragraphs'][1] = '현재 R9는 전체509 reference 풀, 배경5fold 분할, 실제 고정예제15,000개와 동적 TRAIN loader의 준비·검증을 마쳤습니다. 모델·학습 엔진의 작은 CPU 숫자 검증도 완료했습니다. R9 전체5fold100epoch GPU 본학습과 새 production 추론은 아직 실행하지 않았습니다.'
    by_title['SA 결측과 초기 QC']['paragraphs'][2] = '후속 boundary QC는 원 row_id와 실제 window/minute를 join해 해당 분만 추가 NaN으로 가립니다. 창의 첫/끝 분을 무조건 지우지 않습니다. 원 허용 pool의6,254후보 중1개는 min3 미달로 제외됐으며, retained pool에서 실제 추가 masking은6,253분입니다. flag의 물리적 원인을 확인했다는 뜻은 아닙니다.'
    by_title['SA 결측과 초기 QC']['caption'] = '원본62개 TOL열의 site별 null fraction. 점선 안 공통25대역의 원행 null은0입니다. 이후15분 격자의 NaN은 시간 gap·초기 screening·실제 QC 분 masking에서 생기며 raw column 결측과 구분합니다.'
    by_title['R4·R5·R6·R9 버전별 규모']['paragraphs'][3] = 'R9는 허용 배경100,965창과 공유 reference509개를 사용합니다. 현재20개 고정 NPZ에15,000예제를 생성했고,5fold 동적 TRAIN은 epoch당392,177개 원창을 한 번씩 방문하는 준비 상태입니다. 서로 다른 epoch와 fold의 방문 횟수를 독립 녹음수로 더하지 않습니다.'
    by_title['R4·R5·R6·R9 버전별 규모']['table']['rows'][-1][-1] = '실제 data ready / full NN GPU 미실행'
    by_title['R9 5fold 배경 분할과24h purge']['paragraphs'] = [
        '원래 허용한100,965창과258개 sensor-week group의 bucket을 유지합니다. held-out 1개 bucket을 뺀4개 bucket이 TRAIN이므로 purge 전 약80/20이며, 무작위 행 단위 KFold가 아닙니다.',
        'held-out source group 전체를 SHA 순서와 사전 quota로 selection50%, ensemble fit20%, gate15%, report15%에 배정합니다. 행수나 site 비율을 정확히 맞추지는 않으며 작은 site는 일부 역할에 없을 수 있습니다.',
        '같은 sensor의900초 support 사이에24시간 gap을 요구합니다. report > gate > fit > selection 우선순위를 모델 결과 없이 적용하고 TRAIN은 held-out 전체에 대해 purge합니다. 모든 role pair에서24h 미달과 index/group 교차 중복은0입니다.',
    ]
    by_title['R9 5fold 배경 분할과24h purge']['caption'] = '실제 source-window 역할 수. 고정 합성예제는 fold마다 selection1000 / fit500 / gate500 / report1000개가 준비됐습니다. reference509개는 모든fold가 공유합니다.'
    by_title['R9 minute-first 합성 데이터']['paragraphs'] = [
        '완전2초 선형 power를 먼저60초 native minute로 집계합니다. 이후 smooth frequency EQ 최대6dB, 실제 구간 선택·crop·배치를 적용합니다. time warp, 강제 boundary fade, waveform 반복과 gap fill은 없습니다.',
        'native_inside는 최대15분의 연속 crop을 창 안에 배치합니다. native_short는 실제1-3분 구간을 사용하고, native_edge는 전체 trajectory가 창 경계를 걸치도록 배치합니다. 1분 source는 정수 minute 경계를 걸칠 수 없어 기록된 fallback을 사용합니다.',
        'alpha는 10^(SNR/10) x 관측 배경의 분별 broadband power 중앙값 / 전체 native EQ trajectory peak 입니다. SNR은 Uniform[-10,+10]dB입니다. crop/mask 이전 peak로 scale을 정하므로 최종 창의 peak나 global share가 요청 SNR과 같을 필요는 없습니다.',
        'inside/short/edge 비율은0.5/0.25/0.25, explicit no-source는20%입니다. reference index는 양성 요청에 cycling하며 모든509개가500개 fold-epoch 일정에 등장함을 검사했습니다. 이는 하나의 event 안에서 waveform을 반복한 처리가 아닙니다.',
    ]
    by_title['학습 루프·loss·checkpoint 정책']['paragraphs'][1] = '행별 loss는 share MSE +0.25 x sum(w x (qhat-q)^2) +0.05 x BCE(derived score logit, presence)입니다. site balance는 N/(site 수 x 해당 site 행수)의 양의 loss_weight를 곱하는 objective이며, 배경을 복제하는 sampling과 다릅니다. 실제 dataset에서 weight 공급을 확인했습니다.'
    by_title['추론·후처리의 새 계약']['paragraphs'][2] = '현재 config는 완료한5개 fold policy의 산술평균을 명시적으로 고정했습니다. member share와 diagnostic score에 같은 frozen 가중치를 적용한 뒤 동일하게 fold 평균을 냅니다. 추가 후처리는 수치적인 [0,1] clipping만 허용하며, R9 production inference의 실제 실행은 아직 남아 있습니다.'
    by_title['현재 실행 상태와 다음 gate']['paragraphs'] = [
        '완료: 실제 source/QC/24h 분할, VT509+16 manifest, minute-first 합성예제15,000개,20개 고정 NPZ,5fold 동적 TRAIN loader와 모든509 참조의500 fold-epoch 일정 검증입니다. cached-minute와 직접2초 경로의1,527 case가 bitwise 일치했고4개 spawned CPU worker의 순서·출력도 재현됐습니다.',
        '미실행: R9 production NN4종 x5fold x100epochs CUDA 본학습, 최종 Mac CPU tree fitting과 ensemble fit/gate/report 평가, 새 R9 production inference입니다. 데이터 ready와 모델 training completed를 구별합니다.',
        'WideCNN은 본학습 전 parameter-free masked frequency-global residual을 추가했고4모델 oracle·역전파 검증을 다시 통과했습니다. 기존 R5/R6 모델과 원천 파일은 변경하지 않았습니다.',
        '고정 데이터의 raw q/y energy identity 최대 오차는3.2465156e-8, rounded fixed-q pooling 최대 gap은0.0016588222로 사전 상계 이내입니다. 이는 생성 정합성 QA이며 공식 평가의 대표성이나 성능 개선을 증명하지 않습니다.',
    ]
    by_title['현재 실행 상태와 다음 gate']['paragraphs'].insert(1,
        '실제 TRAIN4개(결측 포함)와 selection8개로 기본크기4모델의 CPU Adam1step·유한 gradient·동일w·엄격 checkpoint복원을 연결 검증했고2.80초에 통과했습니다. 이는 실제 데이터 연결 smoke이며 본학습·성능평가·GPU 실행은 아닙니다. 새 runner는 단계 분리와 global report gate를 숫자fixture로 확인했습니다.')
    storage={
        'title':'파일 크기·실제 준비 데이터의 분모',
        'lead':'압축 파일, 압축 내부 member, 배열 payload, 동적 epoch 방문을 따로 표시합니다.',
        'paragraphs':[
            '현재 SA ZIP은94,916,719bytes이고 추출 member 합은96,002,006bytes입니다. member가 Parquet이므로 이 값도 완전히 decode한 table RAM 크기는 아닙니다. 과거97.8MB 추정치 대신 실제 파일 stat을 사용합니다.',
            'VT vessel3parts와 ambient ZIP의 합은4,638,375,330bytes, 내부 WAV 합은9,687,575,100bytes입니다. 원파형을 새로 열지 않고 ZIP directory만 확인했습니다. README1,152bytes까지 더한 과거 download inventory 합과 구별합니다.',
            'R9는 원 배경 array와 reference minute cache를 공유하며 TRAIN 전체를100배 저장하지 않습니다.20개 고정 NPZ와 recipes·metadata·readiness 등으로 구성된 bundle은71,244,935bytes입니다. 모델 checkpoint와 원시 audio는 이 bundle 합에 포함하지 않습니다.',
        ],
        'table':table(['항목','실제 bytes / 개수','의미'],[
            ['R9 fixed NPZ','24,707,614B /20files','15,000 fixed examples'],
            ['NPZ 내부 NPY members','94,601,160B','header포함, 압축해제 member합'],
            ['R9 전체 prepared bundle','71,244,935B','recipes/metadata/cache포함'],
            ['R9 dynamic TRAIN','392,177 examples/epoch','5fold합; 원창 once/epoch'],
            ['SA base bank','200,461,628B','133641x15x25 float32+header'],
            ['HearMyShip unique WAV','3,315,250,540B','수집 inventory; R9 positive미사용'],
        ])}
    s.insert(5,storage)
    return s


def build(output_dir):
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame, NextPageTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image, KeepTogether
    from reportlab.pdfgen import canvas
    from pypdf import PdfReader
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assets = output_dir/(STEM+'_assets')
    for name in ['figures','tables','evidence']:
        (assets/name).mkdir(parents=True, exist_ok=True)
    input_hashes = {name:sha(ROOT/name) for name in INPUTS}
    sa, facts, pilot = load(INPUTS[0]),load('reports/R9_DIAGRAM_FACTS.json'),load('reports/R9_R6_PILOT_SUBMISSION.json')
    if len(sa['metric_catalog']) != 305 or len(sa['historical_aggregate_tables']) != 26:
        raise ValueError('Complete R8 catalog/table inventory required')
    companions=[];portable_evidence=[]
    for name in INPUTS:
        destination=assets/'evidence'/Path(name).name
        source=ROOT/name
        if source.suffix.lower()=='.md':
            text,changes=portable_markdown(source,destination)
            destination.write_text(text)
        else:
            shutil.copyfile(source,destination)
            assert sha(destination)==input_hashes[name]
            changes=[]
        portable_evidence.append({'original_source':name,'original_sha256':input_hashes[name],
            'portable_path':str(destination.relative_to(output_dir)),
            'portable_sha256':sha(destination),'relative_link_rebindings':changes,
            'original_modified':False,'byte_exact_copy':not changes})
    table_records=[]
    for i,t in enumerate(sa['historical_aggregate_tables'],1):
        source=ROOT/t['source_file']
        if sha(source)!=t['sha256']:
            raise ValueError('Frozen aggregate table changed:'+t['source_file'])
        destination=assets/'tables'/f'{i:02d}_{source.name}'
        shutil.copyfile(source,destination)
        with source.open(newline='') as stream:
            reader=csv.reader(stream);columns=next(reader);rows=list(reader)
        if len(rows)!=t['published_row_count']:
            raise ValueError('Published table row count differs')
        relative=f'{STEM}_assets/tables/{destination.name}'
        archival=any(word in t['source_file'].lower() for word in ['test','domain_probe','site_matched'])
        table_records.append({**t,'portable_csv':relative,'archival_Test_related':archival,
                              'portable_html':relative[:-4]+'.html'})
        table_html='<table><thead><tr>'+''.join('<th>'+html.escape(v)+'</th>' for v in columns)+'</tr></thead><tbody>'
        table_html+=''.join('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'
        destination.with_suffix('.html').write_text('<!doctype html><meta charset="utf-8"><title>'+html.escape(source.name)+'</title><style>body{font:13px system-ui;margin:30px}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #ddd;white-space:nowrap}th{position:sticky;top:0;background:#e8efec}</style><h1>'+html.escape(source.name)+'</h1><p>'+html.escape(t['denominator'])+'</p><p>SHA256: '+t['sha256']+'</p>'+('<p>과거 Test-derived archival 표. 새 추론·학습·튜닝에는 사용하지 않음.</p>' if archival else '')+table_html)
        companions.extend([relative,relative[:-4]+'.html'])
    catalog_csv=assets/'evidence/metric_catalog_305.csv'
    columns=list(sa['metric_catalog'][0])
    with catalog_csv.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for row in sa['metric_catalog']:writer.writerow({k:number(v) for k,v in row.items()})
    charts=plot_aggregates(assets,facts,sa)
    pages=sections(facts,charts,pilot)
    image_status={}
    for page in pages:
        if page.get('pipeline_key'):
            source=Path(page['pipeline_source']);key=page['pipeline_key']
            image_status[key]=source.exists()
            if source.exists():
                dest=assets/'figures'/source.name;shutil.copyfile(source,dest)
                page['image']=f'{STEM}_assets/figures/{source.name}'
                page['caption']='상세 도식. 수식·분모·실행 상태와 shared-reference 한계는 본문을 함께 확인하십시오.'
            else:
                page['paragraphs'].append('상세 이미지 생성 중: 이미지 파일이 준비되면 같은 builder를 다시 실행해 PDF/HTML/MD에 삽입합니다. 현재 판은 이 안내를 표시하며 가짜 결과나 임시 유사 도식을 만들지 않았습니다.')
    pdfmetrics.registerFont(TTFont('Korean',FONT_PATH))
    pdfmetrics.registerFontFamily('Korean',normal='Korean',bold='Korean',italic='Korean',boldItalic='Korean')
    styles=getSampleStyleSheet()
    body=ParagraphStyle('KBody',fontName='Korean',fontSize=10.1,leading=17.1,spaceAfter=13,wordWrap='CJK',textColor=colors.HexColor('#24302f'))
    small=ParagraphStyle('KSmall',parent=body,fontSize=7.3,leading=10.5,spaceAfter=4)
    title=ParagraphStyle('KTitle',parent=body,fontSize=20,leading=28,spaceAfter=14,textColor=colors.HexColor('#183e48'))
    lead=ParagraphStyle('KLead',parent=body,fontSize=10.8,leading=17,spaceAfter=18,textColor=colors.HexColor('#57746f'))
    cap=ParagraphStyle('KCaption',parent=body,fontSize=7.8,leading=12,spaceBefore=8,spaceAfter=10,textColor=colors.HexColor('#65716c'))
    heading=ParagraphStyle('KHeading',parent=body,fontSize=13,leading=19,spaceAfter=12,textColor=colors.HexColor('#183e48'))
    story=[];width=511
    esc=lambda v:html.escape(str(v)).replace('\n','<br/>')
    def para(v,style=body):return Paragraph(esc(v),style)
    def table(headers,rows,widths=None,style=small):
        widths=widths or [width/len(headers)]*len(headers)
        values=[[para(v,style) for v in headers]]+[[para(v,style) for v in row] for row in rows]
        t=Table(values,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dfeae5')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#8fa89c')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.whitesmoke,colors.white]),
            ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
            ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
        return t
    from PIL import Image as PILImage
    def picture(path,max_height=285,max_width=511):
        a=output_dir/path
        with PILImage.open(a) as im:w,h=im.size
        scale=min(max_width/w,max_height/h)
        return Image(str(a),width=w*scale,height=h*scale,hAlign='LEFT')
    for i,page in enumerate(pages):
        landscape=page.get('kind')=='pipeline'
        if i:
            story.append(NextPageTemplate('landscape' if landscape else 'portrait'))
            story.append(PageBreak())
        if landscape:
            story.append(para(page['title'],heading))
            if 'image' in page:
                story.append(picture(page['image'],420,757.89))
                story.append(para('설계·구현과 예정된 실행을 구분한 상세 도식. full R9 GPU 학습은 미실행이며 shared-reference CV caveat를 본문과 함께 확인하십시오.',cap))
            else:
                story.append(para('상세 도식 생성 중입니다. 최종 이미지가 준비되면 같은 builder로 다시 삽입합니다. 현재 PDF에 실제 학습 결과처럼 보이는 대체 도식을 만들지 않았습니다.',body))
            continue
        if i==0:
            story.append(Spacer(1,48));story.append(para('EVIDENCE REPORT / 2026.10.26',lead))
        else:story.append(para(f'{i:02d} / DATA & MODEL PIPELINE',small))
        story.append(para(page['title'],title));story.append(para(page['lead'],lead))
        if not landscape:
            for p in page['paragraphs']:story.append(para(p))
        if 'table' in page:story.append(table(page['table']['head'],page['table']['rows']))
        if 'image' in page:
            story.append(Spacer(1,6));story.append(picture(page['image'],255,511))
            story.append(para(page.get('caption',''),cap))
        if i==0:
            story.append(Spacer(1,24));story.append(table(['완료한 것','아직 실행하지 않은 것'],[
                ['Source/QC/분할·코드·작은CPU숫자검증','R9production5fold100epochGPU·새최종추론'],
                ['305metrics /26tables /6aggregateplots','공식성능개선·LB예측보장'],
            ]))
    story.append(PageBreak());story.append(para('부록 A / 전체305개 지표',title))
    story.append(para('기존 R8 metric_catalog 전체입니다. source·분모·버전·상태·주의사항을 보존하며, 이 표는 새 원시 데이터 분석이 아닙니다. 모든 필드는 companion evidence JSON/metric_catalog_305.csv에도 들어 있습니다. 긴 값의 표시는 원 precision을 유지합니다.',body))
    rows=[]
    for i,m in enumerate(sa['metric_catalog'],1):
        rows.append([f"{i:03d} {m['metric']}",number(m['value'])+'\n'+number(m['unit']),
                     number(m['status'])+' / '+number(m['dataset_version'])+'\n분모: '+number(m['denominator'])+'\n출처: '+number(m['source_file'])+'\n'+number(m.get('source_section',''))+('\n주의: '+number(m['caveat']) if m.get('caveat') else '')])
    story.append(table(['지표','보존된 값 / 단위','상태·버전·분모·출처·주의'],rows,[155,140,216]))
    story.append(PageBreak());story.append(para('부록 B / 원 집계표26개 전체',title))
    story.append(para('전체4,137행의값을 portable companion CSV/HTML에 원본 그대로 포함했습니다. PDF에는26개 전부의분모·행수·상태·SHA·경로를 수록합니다. Test-derived archival 표는 보존 이력이며새학습설정에사용하지않습니다.',body))
    for i,t in enumerate(table_records,1):
        story.append(para(f"{i:02d}. {t['source_file']} ({t['published_row_count']}행)",heading))
        story.append(para('분모: '+t['denominator'],small))
        story.append(para('상태: '+t['status']+(' / 과거Test-derived archival' if t['archival_Test_related'] else ''),small))
        story.append(para('SHA256: '+t['sha256'],small))
        story.append(Paragraph('CSV: <link href="'+esc(t['portable_csv'])+'">'+esc(t['portable_csv'])+'</link>',small))
        story.append(Paragraph('HTML: <link href="'+esc(t['portable_html'])+'">'+esc(t['portable_html'])+'</link>',small))
        story.append(Spacer(1,10))
    class NumberedCanvas(canvas.Canvas):
        def __init__(self,*a,**kw):super().__init__(*a,**kw);self.states=[]
        def showPage(self):self.states.append(dict(self.__dict__));self._startPage()
        def save(self):
            total=len(self.states)
            for state in self.states:
                self.__dict__.update(state);self.setFont('Korean',7)
                self.setFillColor(colors.HexColor('#748780'))
                self.drawString(42,23,'UNDERWATER VESSEL NOISE / DATA EVIDENCE / 2026-10-26')
                right=self._pagesize[0]-42
                self.drawRightString(right,23,f'{self._pageNumber} / {total}')
                self.setStrokeColor(colors.HexColor('#c5d3cc'));self.line(42,34,right,34)
                canvas.Canvas.showPage(self)
            canvas.Canvas.save(self)
    pdf=output_dir/(STEM+'.pdf')
    doc=BaseDocTemplate(str(pdf),pagesize=(595.276,841.89),rightMargin=42,leftMargin=42,
                       topMargin=42,bottomMargin=48,title='데이터·EDA·전처리 근거 보고서 R9',author='Competition research workspace')
    doc.addPageTemplates([
        PageTemplate(id='portrait',frames=[Frame(42,48,511.276,751.89,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],pagesize=(595.276,841.89)),
        PageTemplate(id='landscape',frames=[Frame(42,48,757.89,505.276,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],pagesize=(841.89,595.276)),
    ])
    doc.build(story,canvasmaker=NumberedCanvas)
    css='''body{margin:0;background:#edf1ee;color:#24302f;font-family:"Arial Unicode MS","Apple SD Gothic Neo",sans-serif;font-size:15px;line-height:1.9}.page,article{max-width:970px;margin:26px auto;background:white;padding:48px 58px;box-shadow:0 4px 25px #173c3810}h1,h2{color:#183e48;line-height:1.5}h1{font-size:30px}.lead{color:#57746f;font-size:18px}table{width:100%;border-collapse:collapse;font-size:12px;word-break:break-word}th,td{padding:9px;border-bottom:1px solid #d9e3dc;text-align:left;vertical-align:top}th{background:#dfeae5}img{width:100%;height:auto}.caption{font-size:12px;color:#65716c}a{color:#225e79}details{margin:14px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px}@media print{body{background:white}.page{box-shadow:none;break-after:page;margin:0;padding:20px}}'''
    h=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>데이터·EDA·전처리 근거 보고서 R9</title><style>'+css+'</style><body>']
    md=['# 데이터·EDA·전처리 근거 보고서 R9','', '2026-10-26. 원시 Test 재분석·fit·GPU 실행 없이 기존 근거를 통합했습니다.','']
    for page in pages:
        h+=['<section class="page"><h1>'+esc(page['title'])+'</h1><p class="lead">'+esc(page['lead'])+'</p>']
        md+=['## '+page['title'],'',page['lead'],'']
        for p in page['paragraphs']:h+=['<p>'+esc(p)+'</p>'];md += [p,'']
        if 'table' in page:
            t=page['table'];h+=['<table><tr>'+''.join('<th>'+esc(v)+'</th>' for v in t['head'])+'</tr>']
            h+=['<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>' for row in t['rows']];h+=['</table>']
            md+=['|'+'|'.join(t['head'])+'|','|'+'|'.join(['---']*len(t['head']))+'|']
            md+=['|'+'|'.join(str(v).replace('|','/') for v in row)+'|' for row in t['rows']];md+=['']
        if 'image' in page:
            h+=['<img src="'+esc(page['image'])+'"><p class="caption">'+esc(page.get('caption',''))+'</p>']
            md+=['!['+page['title']+']('+page['image']+')','',page.get('caption',''),'']
        h+=['</section>']
    h+=['<article><h1>부록 A / 전체305개 지표</h1><p>원래값/분모/precision을보존한metriccatalog전체입니다.</p>']
    md+=['## 부록 A / 전체305개 지표','','정확한 전체필드 CSV: ['+catalog_csv.name+']('+STEM+'_assets/evidence/'+catalog_csv.name+')','']
    for i,m in enumerate(sa['metric_catalog'],1):
        h+=['<details><summary>'+esc(f"{i:03d} {m['metric']} = {number(m['value'])}")+'</summary><pre>'+esc(json.dumps(m,ensure_ascii=False,indent=2))+'</pre></details>']
        md+=[f"### {i:03d} {m['metric']}",'','```json',json.dumps(m,ensure_ascii=False,indent=2),'```','']
    h+=['</article><article><h1>부록 B / 집계표26개·전체4,137행</h1>']
    md+=['## 부록 B / 집계표26개·전체4,137행','']
    for t in table_records:
        text=f"{t['source_file']} / {t['published_row_count']}행 / "+('historicalTest-related' if t['archival_Test_related'] else 'sourceaggregate')
        h+=['<p><a href="'+esc(t['portable_html'])+'">'+esc(text)+'</a> / <a href="'+esc(t['portable_csv'])+'">CSV</a><br>'+esc(t['denominator'])+'<br><small>'+t['sha256']+'</small></p>']
        md+=['- ['+text+']('+t['portable_html']+') / [CSV]('+t['portable_csv']+')','  - 분모: '+t['denominator'],'  - SHA256: `'+t['sha256']+'`','']
    h+=['</article></body></html>']
    (output_dir/(STEM+'.html')).write_text('\n'.join(h))
    (output_dir/(STEM+'.md')).write_text('\n'.join(md)+'\n')
    (assets/'tables/index.json').write_text(json.dumps(table_records,ensure_ascii=False,indent=2)+'\n')
    (assets/'evidence/portable_copy_manifest.json').write_text(json.dumps(portable_evidence,ensure_ascii=False,indent=2)+'\n')
    link_verification=verify_portable_links(output_dir,assets)
    if link_verification['broken_count']:
        (output_dir/(STEM+'_LINK_VERIFICATION.json')).write_text(json.dumps(link_verification,ensure_ascii=False,indent=2)+'\n')
        raise ValueError('Broken portable evidence links: '+str(link_verification['broken']))
    after={name:sha(ROOT/name) for name in INPUTS}
    if input_hashes!=after:raise ValueError('Source evidence changed during rendering; rerun from a stable snapshot')
    reader=PdfReader(pdf)
    texts=[p.extract_text() or '' for p in reader.pages]
    missing_metrics=[m['metric'] for m in sa['metric_catalog'] if not any(m['metric'] in text.replace('\n','') for text in texts)]
    # CJK font extraction can insert line breaks; companion rows remain exact.
    report={'format':'r9_pdf_report_verification_v1','status':'built_pending_visual_QA',
            'created_utc':datetime.now(timezone.utc).isoformat(),'pdf_pages':len(reader.pages),
            'designed_main_sections':len(pages),'catalog_entries':len(sa['metric_catalog']),
            'aggregate_tables':len(table_records),'aggregate_table_data_rows':sum(t['published_row_count'] for t in table_records),
            'all_companion_table_SHAs_match':True,'input_source_SHA_unchanged':True,
            'source_sha256':input_hashes,'pipeline_images_embedded':image_status,
            'new_aggregate_plot_count':len(charts),'raw_Test_or_acoustic_arrays_read':False,
            'new_model_training_or_GPU':False,'font':FONT_PATH,'font_embedded':True,
            'text_metric_name_extraction_missing':missing_metrics,
            'minimum_page_text_characters':min(map(len,texts)),
            'file_sha256':{suffix:sha(output_dir/(STEM+'.'+suffix)) for suffix in ['pdf','md','html']},
            'asset_files':{str(p.relative_to(output_dir)):sha(p) for p in assets.rglob('*') if p.is_file()},
            'builder_sha256':sha(Path(__file__))}
    report['portable_evidence_copy_manifest']=portable_evidence
    report['portable_link_verification']=link_verification
    (output_dir/(STEM+'_LINK_VERIFICATION.json')).write_text(json.dumps(link_verification,ensure_ascii=False,indent=2)+'\n')
    (output_dir/(STEM+'_VERIFICATION.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','pdf_pages','designed_main_sections','catalog_entries','aggregate_tables','pipeline_images_embedded']}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'reports')
    args=parser.parse_args();build(args.output_dir)


if __name__=='__main__':main()
