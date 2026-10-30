"""Read-only public-starter / six-operator forensics. No Test, training or tuning.

Run: PYTHONDONTWRITEBYTECODE=1 .venv_r17_eda/bin/python ops/r18_public_starter_forensics.py
Only this new script and reports/r18_public_gap_audit_r1/starter are written.
"""
from pathlib import Path
import ast
import csv
import hashlib
import json
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'reports/r18_public_gap_audit_r1/starter'
OPENED = set()


def guard(event, args):
    if event != 'open' or not args or not isinstance(args[0], (str, bytes, Path)):
        return
    p = Path(args[0].decode() if isinstance(args[0], bytes) else args[0])
    low = str(p).lower()
    if p.name.lower() in {'test.csv', 'samplesubmission.csv', 'sample_submission.csv', 'query_predictions.npz'}:
        raise PermissionError('No challenge query access in public-starter forensics')
    if '/eda_deep/' in low or '/r15_band_comparison_' in low or '/models/' in low:
        raise PermissionError('No Test-derived or model input in public-starter forensics')
    OPENED.add(str(p))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
        default=lambda x: x.item() if hasattr(x, 'item') else str(x)) + '\n')


def read(name):
    return json.loads((ROOT / name).read_text())


def table(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


SOURCE_FILES = [
    'data/raw/maritime_starter_notebook.ipynb', 'data/extracted/README.md',
    'data/extracted/DATA_DICTIONARY.md', 'data/extracted/SHA256SUMS.txt',
    'scratch/discussions/thread_bodies.json', 'scratch/discussions/discussion_archive.txt',
    'pipeline/audio.py', 'pipeline/prepare_external.py', 'pipeline/sampler_r9.py',
    'pipeline/dataset_r9.py', 'pipeline/reference_pool_r9.py', 'pipeline/source_split_r6.py',
    'pipeline/r16_production_data.py', 'pipeline/r16_multisupport_data.py',
    'ops/r15_forward_prototype.py', 'ops/r16_forward_data.py',
    'ops/r16_support_temporal_data.py', 'pipeline/r18_production_data.py',
    'ops/prepare_r18_production_data.py',
    'data/synthetic/r16_multisupport_v1/manifest.json',
    'data/synthetic/r18_production_v1/manifest.json',
    'data/synthetic/r18_production_v1/PROTOCOL_BEFORE_GENERATION.json',
    'reports/r18_fundamental_audit_r1/wav/RESULTS.json',
]


def preserve_sources():
    pins = {n: sha(ROOT / n) for n in SOURCE_FILES}
    notebook = read('data/raw/maritime_starter_notebook.ipynb')
    cells, md = [], ['# Source record: complete local official starter\n',
        'Cell indices below are **zero-based**; one-based positions are also recorded. '
        'No notebook cell was executed. Only the function definition in cell 11 was AST-extracted.\n',
        f'Notebook SHA-256: `{pins[SOURCE_FILES[0]]}`\n']
    definitions = []
    for i, c in enumerate(notebook['cells']):
        source = ''.join(c.get('source', []))
        name = f'notebook_cell_{i:02d}.{ "py" if c["cell_type"] == "code" else "md" }'
        (OUT / name).write_text(source)
        defs = []
        if c['cell_type'] == 'code':
            defs = [n.name for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        definitions.extend(defs)
        cells.append({'cell_zero_based': i, 'cell_one_based': i+1, 'type': c['cell_type'],
            'lines': len(source.splitlines()), 'source_sha256_utf8': hashlib.sha256(source.encode()).hexdigest(),
            'saved_file': name, 'definitions': defs})
        numbered = '\n'.join(f'{j:04d} {line}' for j, line in enumerate(source.splitlines(), 1))
        md.extend([f'## Cell {i} (position {i+1}), {c["cell_type"]}\n',
            f'Cell source SHA: `{cells[-1]["source_sha256_utf8"]}`\n', f'```text\n{numbered}\n```\n'])
    (OUT / 'OFFICIAL_STARTER_FULL_SOURCE.md').write_text('\n'.join(md))
    dump(OUT / 'notebook_inventory.json', {'notebook_sha256': pins[SOURCE_FILES[0]],
        'cells': cells, 'definitions': definitions, 'hidden_full_synthesis_function_present': False,
        'code_cell_indices': [i for i,c in enumerate(notebook['cells']) if c['cell_type']=='code']})
    assert definitions == ['minute_band_power']
    originals = read('scratch/discussions/thread_bodies.json')
    selected = [r for r in originals if str(r['id']) in {'35368', '35370', '35425', '35531'}]
    dump(OUT / 'official_forum_local_archive.json', {'archive_path': 'scratch/discussions/thread_bodies.json',
        'archive_sha256': pins['scratch/discussions/thread_bodies.json'], 'archive_records': len(originals),
        'live_reverified_in_this_run': False, 'records': selected,
        'interpretation': 'Staff replies authorize own mixtures and restrict Test use; participant declarations are not hidden-generator code.'})
    md = ['# Source record: local code with original line numbers\n',
          'These are local implementations, **not official hidden synthesis code**. '
          'Raw source SHA covers complete original bytes; excerpts preserve original line numbering.\n']
    ranges = {
        'pipeline/audio.py': [(1,40)], 'pipeline/prepare_external.py': [(13,31),(49,87)],
        'pipeline/sampler_r9.py': [(1,41)], 'pipeline/dataset_r9.py': [(117,164),(177,196)],
        'pipeline/source_split_r6.py': [(31,37)], 'pipeline/reference_pool_r9.py': [(101,157)],
        'ops/r15_forward_prototype.py': [(11,13),(38,64)], 'ops/r16_forward_data.py': [(1,16),(71,157)],
        'ops/r16_support_temporal_data.py': [(199,245)],
        'pipeline/r16_multisupport_data.py': [(31,81),(82,123)],
        'pipeline/r16_production_data.py': [(118,174),(184,211)],
        'pipeline/r18_production_data.py': [(33,117),(119,210)],
        'ops/prepare_r18_production_data.py': [(1,6)],
        'data/extracted/README.md': [(1,21)], 'data/extracted/DATA_DICTIONARY.md': [(1,18)],
    }
    for name, spans in ranges.items():
        lines = (ROOT / name).read_text().splitlines()
        md.append(f'## {name}\n\nSHA-256: `{pins[name]}`\n')
        for lo,hi in spans:
            md.append('```text\n'+'\n'.join(f'{j:04d} {lines[j-1]}' for j in range(lo,min(hi,len(lines))+1))+'\n```\n')
    (OUT / 'LOCAL_CODE_SOURCE_EXCERPTS.md').write_text('\n'.join(md))
    definitions = {}
    for name in SOURCE_FILES:
        if name.endswith('.py'):
            definitions[name] = [{'name': n.name, 'line': n.lineno, 'end_line': n.end_lineno}
                for n in ast.walk(ast.parse((ROOT/name).read_text()))
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    dump(OUT/'local_function_line_index.json',definitions)
    source_md = ['# Source provenance and evidence boundary\n',
        'The primary local notebook is preserved in full, including every Markdown and code cell. '
        'Its only defined function is `minute_band_power` (zero-based cell 11, lines 1–28). '
        'The constant predictions in cell 6 are explicitly arbitrary baselines. '
        '**No complete official hidden synthesis implementation is present in this notebook.**\n',
        'The official staff forum bodies below were captured earlier and are reused from the local archive. '
        'They were not fetched again during this run. Participant posts are source declarations, '
        'not official generator specifications.\n']
    for r in selected:
        source_md.append(f'- [{r.get("title",r["id"])}]({r["url"]}) — local record {r["id"]}; complete={r.get("complete")}.')
    source_md.extend(['\n## Complete-file SHA-256 pins\n', '| Local source | SHA-256 |', '|---|---|'])
    source_md.extend(f'| `{n}` | `{h}` |' for n,h in pins.items())
    source_md.extend(['\n## Source extracts\n',
        '- `OFFICIAL_STARTER_FULL_SOURCE.md`: all 12 original cells with original cell-local line numbers and cell UTF-8 SHA.',
        '- `LOCAL_CODE_SOURCE_EXCERPTS.md`: original local code excerpts with original file line numbers and complete-file SHA.',
        '- `local_function_line_index.json`: AST function/class ranges for unambiguous lookup.',
        '- `official_forum_local_archive.json`: exact selected archived forum record bodies, not new live verification.',
        '- `recorded_source_bindings.json`: generator source hashes match R16/R18 creation manifests; old full-WAV audit pins match current files.'])
    (OUT/'SOURCES.md').write_text('\n'.join(source_md)+'\n')
    return pins, notebook


def conversion_fixture(notebook):
    import numpy as np
    from pipeline.sampler_r9 import native_minutes
    # The exact public function only; no data-loading/submission notebook cells.
    tree = ast.parse(''.join(notebook['cells'][11]['source']))
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
    env = {'np': np}
    exec(compile(tree, '<official notebook cell 11>', 'exec'), env)
    helper = env['minute_band_power']
    fs = 32000
    n = 2*fs
    t = np.arange(n)/fs
    base = .12*np.sin(2*np.pi*100*t)+.07*np.sin(2*np.pi*1000*t)+.04*np.sin(2*np.pi*6000*t)
    gain = np.linspace(.15,1.0,30)
    pcm = np.round((gain[:,None]*base[None,:])*32767).astype(np.int16)
    samples = pcm.astype(np.float64)/32768.
    actual = helper(samples.ravel(),fs)
    # Independent rFFT, explicit periodic Hann / one-sided density and bins.
    w = .5-.5*np.cos(2*np.pi*np.arange(n)/n)
    X = np.fft.rfft((samples-samples.mean(1,keepdims=True))*w,axis=1)
    psd = abs(X)**2/(fs*(w*w).sum())
    psd[:,1:-1] *= 2
    hz = np.fft.rfftfreq(n,1/fs)
    centres = 10**(1.5+.1*np.arange(25))
    independent = np.column_stack([psd[:,(hz>=c/10**.05)&(hz<c*10**.05)].sum(1)*.5 for c in centres])
    expected = independent.mean(0,keepdims=True)
    cache_like = native_minutes(independent.astype(np.float32))
    significant = actual > actual.max()*1e-8
    bins = [int(((hz>=c/10**.05)&(hz<c*10**.05)).sum()) for c in centres]
    partial = helper(np.r_[samples.ravel(),samples.ravel()[:fs]],fs)
    lower = helper(np.tile(np.sin(2*np.pi*100*np.arange(32000)/16000),30),16000)
    short_rejected = False
    try:
        helper(np.zeros(fs*59),fs)
    except ValueError:
        short_rejected = True
    pcm_raw = helper(pcm.ravel(),fs)
    gain_error = float(np.max(abs(pcm_raw[significant]/actual[significant]-32768.**2)))
    assert np.max(abs(actual-expected)) < 1e-15
    assert np.array_equal(partial,actual) and short_rejected and np.isnan(lower[0,-1])
    assert np.max(abs(actual[significant]-cache_like[significant])/actual[significant]) < 1e-6
    result = {'status':'passed','actual_official_function_executed':True,'notebook_cell_zero_based':11,
        'no_WAV_decoder_run':'soundfile unavailable in audit venv; explicit synthetic int16 normalization only',
        'PCM_convention':'int16 / 32768; helper itself has no PCM normalization or pressure calibration',
        'sample_rate_hz':fs,'synthetic_samples':int(samples.size),'max_abs_official_vs_independent_FFT':float(np.max(abs(actual-expected))),
        'max_significant_relative_float32_segment_then_native_minute_error':float(np.max(abs(actual[significant]-cache_like[significant])/actual[significant])),
        'raw_int16_vs_normalized_power_ratio':32768.**2,'ratio_max_abs_error':gain_error,
        'raw_int16_vs_normalized_db_offset':float(20*np.log10(32768)),
        '61second_input_returns_one_complete_minute_unchanged':bool(np.array_equal(partial,actual)),
        '59second_input_rejected':short_rejected,'16kHz_last_band_NaN':bool(np.isnan(lower[0,-1])),
        'first_fft_integration_width_hz':bins[0]*.5,'last_fft_integration_width_hz':bins[-1]*.5,
        'flat_PSD_last_first_power_ratio':bins[-1]/bins[0],
        'flat_PSD_last_first_db':float(10*np.log10(bins[-1]/bins[0])),
        'max_band_upper_hz':float(centres[-1]*10**.05),
        'calibrated_1Pa2_db_re_1uPa2':120.,'uncalibrated_1digital_unit2_10log10':0.,
        'same_unit_fixture_power_mean':.505,'correct_db_of_linear_mean':float(10*np.log10(.505)),
        'wrong_db_mean_for_powers_1_and_0p01':-10.,
        'meaning':'Conversion / arithmetic parity only; no official synthesis parity claimed.'}
    table('bands.csv',[{'band':j,'centre_hz':float(c),'low_hz':float(c/10**.05),'high_hz':float(c*10**.05),
        'FFT_bins_at_32k':bins[j],'FFT_rectangle_width_hz':bins[j]*.5} for j,c in enumerate(centres)])
    dump(OUT/'conversion_fixture.json',result)
    return result


def operator_fixtures():
    import numpy as np
    from ops.r16_forward_data import OPERATORS, generate_operator
    from pipeline.sampler_r9 import energy_weights
    # The local shelf uses the nominal rounded centres, unlike the exact
    # base10 centres in the official FFT filterbank. Preserve that detail.
    f = np.array([31.6,39.8,50.1,63.1,79.4,100,125.9,158.5,199.5,251.2,316.2,398.1,
        501.2,631,794.3,1000,1258.9,1584.9,1995.3,2511.9,3162.3,3981.1,5011.9,6309.6,7943.3])
    taper = np.clip((np.log(f)-np.log(200))/(np.log(500)-np.log(200)),0,1)
    raised = .5-.5*np.cos(np.pi*taper)
    shelf = {'control_eq6':(0,0),'mfhf_shelf6':(6,0),'mfhf_shelf12':(12,0),'mfhf_shelf12_lfminus6':(12,-6)}
    rows=[]
    for mode in ['full_O_short_crop','missing_inside_crop','natural_long12']:
        bg=80.123+np.arange(15)[:,None]*.13+np.arange(25)[None,:]*.07
        if mode=='missing_inside_crop':bg[[0,3,5,11]]=np.nan
        native=(1+np.arange(19)[:,None]*.11)*(1+np.arange(25)[None,:]*.17)
        native[18]*=25
        begin,length,start=(1,12,2) if mode=='natural_long12' else (1,4,2)
        geometry={'crop_begin_minute':begin,'selected_minutes':length,'window_start_minute':start}
        base=np.linspace(-4,4,25)
        post=np.linspace(5,-5,25)
        snr=3.;sub=np.array([-3.,5.]);o=np.isfinite(bg);offset=float(np.nanmax(bg))
        B=np.where(o,10**((np.where(o,bg,offset)-offset)/10),0.)
        part=native[begin:begin+length];ob=o[start:start+length]
        for op in OPERATORS:
            A=np.zeros((15,25));bmed=float(np.median(B.sum(1)[o.all(1)]))
            if op in shelf:
                hi,lo=shelf[op];curve=base+lo*(1-raised)+hi*raised
                transformed=native*10**(curve/10)
                alpha=10**(snr/10)*bmed/transformed.sum(1).max()
                A[start:start+length]=alpha*transformed[begin:begin+length]
                alphas=[float(alpha)]
            elif op=='alpha_preEQ_inserted':
                alpha=10**(snr/10)*bmed/np.where(ob,part,0).sum(1).max()
                A[start:start+length]=alpha*part*10**(post/10)
                alphas=[float(alpha)]
            else:
                alphas=[]
                for j,ix in enumerate((np.arange(9),np.arange(9,25))):
                    a=10**(sub[j]/10)*np.median(B[:,ix].sum(1)[o.all(1)])/np.where(ob[:,ix],part[:,ix],0).sum(1).max()
                    A[start:start+length,ix]=a*part[:,ix]*10**(post[ix]/10)
                    alphas.append(float(a))
            Ao=np.where(o,A,0);E=B+Ao;y=float(Ao.sum()/E.sum())
            q=np.divide(Ao,E,out=np.zeros_like(E),where=o)
            raw=np.full_like(E,np.nan);raw[o]=10*np.log10(E[o])+offset
            expected=np.round(raw,1).astype(np.float32)
            actual,receipt=generate_operator(bg,native,geometry,base,post,snr,sub,op)
            ae=float(np.max(abs(A-actual['added_full_power'])))
            ye=abs(y-actual['share']);qe=float(np.max(abs(q-actual['cell_share'])))
            de=float(np.max(abs(expected[o]-actual['db'][o])))
            scaled,_=generate_operator(bg,native*100,geometry,base,post,snr,sub,op)
            shifted,_=generate_operator(bg+37,native,geometry,base,post,snr,sub,op)
            z,_=generate_operator(bg,None,None,np.zeros(25),np.zeros(25),snr,sub,op,zero=True)
            assert ae<1e-12 and ye<1e-13 and qe<1e-12 and de==0, (mode,op,ae,ye,qe,de)
            assert np.array_equal(np.isfinite(actual['db']),o) and np.all(actual['cell_share'][~o]==0)
            assert abs(scaled['share']-y)<1e-13 and abs(shifted['share']-y)<1e-13 and z['share']==0
            rows.append({'operator':op,'fixture':mode,'share':y,'alpha_or_subband_alphas':json.dumps(alphas),
                'active_observed_minutes':int((Ao>0).any(1).sum()),'selected_minutes':length,
                'max_A_abs_error':ae,'share_abs_error':ye,'max_q_abs_error':qe,'max_rounded_db_error':de,
                'rounded_pool_gap':abs(float((energy_weights(actual['db'])*q).sum())-y),
                'raw_pool_gap':abs(float((energy_weights(raw)*q).sum())-y),
                'reference_digital_power_gain100_share_error':abs(scaled['share']-y),
                'background_common_db_plus37_share_error':abs(shifted['share']-y),
                'zero_share':float(z['share']),'official_hidden_generator_parity':'unavailable'})
    # Identical requested0dB SNR: full-native peak outside inserted crop versus inserted peak.
    bg=np.full((15,25),90.);native=np.ones((9,25));native[8]=100
    g={'crop_begin_minute':0,'selected_minutes':3,'window_start_minute':4}
    a,ar=generate_operator(bg,native,g,np.zeros(25),np.zeros(25),0.,[0.,0.],'control_eq6')
    b,br=generate_operator(bg,native,g,np.zeros(25),np.zeros(25),0.,[0.,0.],'alpha_preEQ_inserted')
    assert abs(a['share']-.75/375.75)<1e-14 and abs(b['share']-75/450)<1e-14
    table('six_operator_numeric_parity.csv',rows)
    result={'status':'passed','comparisons':len(rows),'operators':list(OPERATORS),
        'same_inputs_independent_formula_vs_actual_local_generate_operator':True,
        'official_hidden_synthesis_parity_available':False,
        'SNR_scope_example':{'requested_snr_db':0,'background_cells':375,'selected_reference_minutes':3,
            'full_native_minutes':9,'native_peak_outside_crop':True,
            'control_full_native_afterEQ_alpha':ar['alpha'],'inserted_preEQ_alpha':br['alpha'],
            'control_share':float(a['share']),'inserted_preEQ_share':float(b['share'])},
        'no_target_forcing':True,'Test_accessed':False,'new_model_training':False}
    dump(OUT/'operator_fixture_summary.json',result)
    return result


def recorded_evidence():
    r16=read('data/synthetic/r16_multisupport_v1/manifest.json')
    r18=read('data/synthetic/r18_production_v1/manifest.json')
    matches=[]
    for name,m,key in [('R16',r16,'source_code_sha256'),('R18',r18,'code_sha256')]:
        for path,expected in m[key].items():
            actual=sha(ROOT/path)
            matches.append({'bundle':name,'path':path,'recorded_sha256':expected,'current_sha256':actual,'matches':actual==expected})
    assert all(x['matches'] for x in matches)
    prior=read('reports/r18_fundamental_audit_r1/wav/RESULTS.json')
    priorpins=[{'path':n,'prior_sha256':h,'current_sha256':sha(ROOT/n)} for n,h in prior['source_SHA256_unchanged'].items()]
    assert all(x['current_sha256']==x['prior_sha256'] for x in priorpins)
    dump(OUT/'recorded_source_bindings.json',{'current_vs_generation_code':matches,
        'prior_full_WAV_conversion_audit_pins':priorpins,
        'prior_WAV_audit_reused_not_rerun':{'path':'reports/r18_fundamental_audit_r1/wav/RESULTS.json',
            'complete_clips':prior['full_duration_recomputed_clip_count'],'complete_minutes':prior['recomputed_complete_minutes'],
            'max_significant_abs_dB_error':prior['maximum_significant_abs_dB_helper_vs_cache_error']}})
    return r16,r18,prior


def write_report(conversion, fixtures, r16, r18, prior):
    lawrows=[]
    for op,shelf,scale in [
        ('control_eq6','smooth base EQ bounded6dB','full native after EQ broadband peak'),
        ('mfhf_shelf6','base EQ plus raised-cosine LF0/HF+6dB','full native after EQ broadband peak'),
        ('mfhf_shelf12','base EQ plus raised-cosine LF0/HF+12dB','full native after EQ broadband peak'),
        ('mfhf_shelf12_lfminus6','base EQ plus raised-cosine LF-6/HF+12dB','full native after EQ broadband peak'),
        ('alpha_preEQ_inserted','post zero-mean log EQ; base EQ unused','observed inserted crop before EQ broadband peak; no renormalization'),
        ('independent_subband_snr','post zero-mean log EQ; base EQ unused','separate observed preEQ peaks and background medians for bands0..8 and9..24; no renormalization')]:
        lawrows.append({'operator':op,'local_EQ':shelf,'local_scale_denominator':scale,
            'local_SNR_draw':'uniform[-10,10]dB; separate2 draws for independent_subband_snr',
            'code_reference':'ops/r16_forward_data.py:117-157','public_official_exact_equivalent':'not disclosed',
            'official_source_constraint':'cell9 broad statement: gain discarded, time/channel/energy varied',
            'confirmed_failure_cause':False})
    table('six_operator_public_vs_local.csv',lawrows)
    text='''# R18 공식 스타터와 D0/H0 6연산자 전수 대조

작성: 2026-10-30 Asia/Seoul. 이 문서는 **로컬 공식 원문에 존재하는 코드**와 **우리 합성기의 코드**를 구분한다. Test 파일·Test 유래 수치·숨은 라벨·모델 파일은 열지 않았고 새 학습·제출·보정·합성 데이터 재생성을 하지 않았다. 수치 검산은 새로 만든 작은 결정론 입력만 사용했다.

## 판정

**“공식 스타터 합성 코드와 6연산자가 1:1로 같은가”는 현재 공개 근거로 확인할 수 없다.** 로컬 공식 노트북은 총 12셀이다. 코드 셀은 1·3·4·6·8·11(0-based)이며 정의된 함수는 cell 11의 `minute_band_power` 하나(28줄)다. 이 함수는 외부 audio→minute band power 변환이다. 배경/참조 선택, SNR/EQ, 시간 변형, mix, label, mask, quantization을 포함한 숨은 Test 완전 합성 함수는 없다. Cell 6의 0.1/0.5는 임의 상수 baseline이며 생성기나 target prior 추정 코드가 아니다. 원문 전체와 각 셀 SHA를 `OFFICIAL_STARTER_FULL_SOURCE.md`, `notebook_inventory.json`, `notebook_cell_*`에 보존했다.

현재 6개는 `ops/r16_forward_data.py:12–14,117–157`의 **우리 가설**이다. 파일 docstring 1–4행도 숨은 과정 재구성이라고 주장하지 않는다고 명시한다. 이 감사에서 재구성한 식은 아래 로컬 연산자의 정확한 기술이며 공식 합성 코드를 대체하지 않는다. 공개되지 않은 연산자 차이는 “확인된 mismatch”로 세지 않는다. 공식 실패의 특정 원인은 이 대조로 확정되지 않는다.

## 공식 원문이 말하는 범위

| 출처 | 확인 가능한 내용 | 공개되지 않은 내용 |
|---|---|---|
| Notebook cell2, lines1–7 | 15분×25밴드, 분 위치, source-export dB, 결측은 무관측, absolute pressure 미검증 | 숨은 배경 추출/사이트/기간/결측 확률 |
| Notebook cell5, lines3–9 | controlled addition의 mixture 에너지 분율; 자연 배경에 선박·참조에 ambient 가능; score는 자연 선박 확률과 다름 | 관측 마스크·시간/밴드 가중·라벨 반올림의 정확한 구현 |
| Notebook cell9, lines1–9 | VTUAD v1.3.2/ONC/CC BY4.0; minute band summaries; absolute digital gain 폐기; 시간/채널/에너지 변형 후 SA 배경과 결합 | 사용 clip 목록·선택률·gain 폐기식·변형 분포/순서·SNR 법칙 |
| Notebook cell10, lines1–9; cell11, lines1–28 | 명시된 변환식·단위·샘플레이트·완전분 처리 | 원SA export의 동일성, hidden synthesis 전체 경로 |
| SA README lines13–21; Dictionary lines8–18 | source-export의 결측/단위/샘플레이트 caveat; 원음/label/Test membership 미포함 | hidden-generator recipe |
| 로컬 forum35368·35370·35425 staff 답변 | 자기 합성 학습 허용, Test inference-only, 자기 검증으로 scaling | 정확한 합성 law, 타깃 분포, 공식 코드 |

원문 셀 번호는 **zero-based**, 화면에서 세면 cell 11은 12번째다. Forum 52개 보관 레코드를 확인했고 관련 staff 원문은 `official_forum_local_archive.json`에 출처 URL·complete 상태와 보존했다. 이번 실행에서는 live forum을 새로 재확인하지 않았다. 참가자의 외부 자료 고지·시간 신축 언급은 참가자의 방법이며 운영진 recipe로 읽지 않는다. 공식 원문이 말하는 “time trajectories varied”만으로 warp·repeat·crop·length law를 특정할 수 없다.

## 실제 호출 경로와 순서

```text
외부 변환(기존 캐시 생성): pipeline/prepare_external.py:16–30,49–87
  → pipeline/audio.py:10–40 (채널0, 실제fs, digital float32 read)
  → data/processed/external_segments_power.npy (2초 파워 float32)
  → R9Sources.__init__: dataset_r9.py:126–148
  → sampler_r9.native_minutes:12–22 (float64로30개 선형 평균)

R18: ops/prepare_r18_production_data.py:5–6
  → pipeline/r18_production_data.py:main → frames/reference_pool/materialize
  → R9Sources.background(row): dataset_r9.py:188–196
  → source_split_r6.apply_minute_qc:31–37 (원NaN 유지+QC분NaN)
  → D0 choose_geometry / H0 natural_geometry 또는 long_geometry
  → smooth_eq / channel_zero_mean / SNR draws
  → ops/r16_forward_data.generate_operator:117–157
  → pack_components:99–115
  → float32 db/cell_share, float64 share 저장

R16 multisupport: pipeline/r16_multisupport_data.py:31–64,82–123
  → R16 production source backgrounds + A-prime reference fold
  → D0 choose_geometry / H0 natural_geometry
  → 같은 generate_operator → 같은 pack_components
```

원문 source SHA와 실행 당시 R16/R18 manifest의 code SHA를 대조해 일치시켰다(`recorded_source_bindings.json`). 따라서 아래는 이름만 비슷한 새 구현이 아니라 생성에 실제 사용된 로컬 함수의 법칙이다. `pipeline/sampler_r9.synthesize`의 옛 R9 혼합 함수는 현재 R16/R18 6연산자의 직접 mix 호출이 아니다. sampler에서 native_minutes·smooth_eq·energy_weights만 재사용한다.

| 단계 | R16 multisupport D0/H0 | R18 production D0/H0 | 공식과의 판정 |
|---|---|---|---|
| 배경 선택 | 원R9 Fold0 TRAIN78,869 중 group5fold+24h purge; fold별 원R16 metadata 재사용 | R16 MS fold0 train에서 qc>=12 전체후보+qc<12 최대1200; site별 round-robin으로12,000선정. 배경당5행(음성1+양성4). clean_ES/fresh_fit 각400배경 | 배경 선정 law 미공개 |
| 참조 선택 | A-prime75에서 fold0 train60, CV15; short1..4/medium5..10/long>=11 가중. B19/C-prime25 예약 | 509에서 예약44제외465; short365/medium85/long15. 양성 long strata13.5% 배정, fullO에서만. 순열 cycle로층별cap | 공식은 VTUAD 유래만 확인; exact pool/law 미공개 |
| PCM→power | 기존 동일 캐시. 실제fs·채널0·2초Hann detrend·PSD적분·완전분선형평균 | 동일 | 공개 helper의 핵심 수식 일치; 입력정밀도 차이 있음 |
| gain/단위 | 디지털 파워를 상대SNR alpha로 정규화; SA는 per-window offset 상대power | 동일 | source notice의 gain 폐기와 양립. 압력교정/hidden alpha 동등성 미확인 |
| D0 geometry | native<=15일 때25% full; 그 외1..6 contiguous crop에 원본preEQ broadband peak 포함 | 동일 | 공식 시간 law 미공개 |
| H0 geometry | native<=10 전체; 그 외11..min(15,n) 균등길이 contiguous crop | short/mid는같음; long>=12 strata는12..min(15,n), fullO 배정 | 로컬 D0/H0 차이 확인; 공식대비차이 미확인 |
| 양성 placement | 전부window안 정수minute start; O에서 A>0 되는 start 목록을한번열거후선택 | 동일; long fullO는가능start균등 | 공식 placement/O 조건 미공개 |
| channel/SNR | base smooth EQ<=6dB; post mean25(loggain)=0, train<=6/그외<=12; U[-10,10]dB | 동일 | 공개된분포 없음 |
| mix/target | E=B+A_O, q=A_O/E, y=sum(A_O)/sum(E), 존재=y>0, raw float64 기준 | 동일 | public energy-fraction 의미와 일치; exact O/weight/timing 미확인 |
| mask/round | 원O에서만mix/label적분, q외부0, db외부NaN; label먼저→db0.1round→float32 | 동일 | 공개helper의 unsupported band NaN와 양립; hidden round/label 순서 미공개 |

### 로컬 6연산자 수식

`O=isfinite(bg)`, `c=max(bg[O])`, `B=O·10^((bg−c)/10)`, `b_med=median_t(sum_k B[t,k])` (완전 관측 분만), `R`은 모든 native minute power, `P`는 선택 crop, `G`는 EQ dB다. alpha의 단위는 이 상대 B와 digital R를 잇는 수치 스케일이며 실제 압력비를 뜻하지 않는다.

| 연산자 | 순서와 정규화 기준 |
|---|---|
| control_eq6 | `T=R·10^(baseEQ/10)`; `alpha=10^(s/10)·b_med/max_t sum_k T`; 이후crop/placement. postEQ사용안함 |
| mfhf_shelf6 | 위와같고 `G=baseEQ+shelf(0,+6)` |
| mfhf_shelf12 | 위와같고 `G=baseEQ+shelf(0,+12)` |
| mfhf_shelf12_lfminus6 | 위와같고 `G=baseEQ+shelf(−6,+12)` |
| alpha_preEQ_inserted | `alpha=10^(s/10)·b_med/max_t sum_k(O_insert·P)`; 그뒤 `P·10^(postEQ/10)`. 후EQ재정규화없음. baseEQ사용안함 |
| independent_subband_snr | L=band0..8(중심31.6..199.5), H=band9..24(251.2..7943.3). 각 `alpha_j=10^(s_j/10)·median_t(sum_k∈j B)/max_t sum_k∈j(O_insert·P)`; 그뒤postEQ. baseEQ사용안함 |

shelf는 중심 <=200 Hz에서 low, >=500 Hz에서 high, log-frequency 사이 raised cosine이다(`r15_forward_prototype.py:38–41`). 이 shelf의 주파수 벡터는 12행의 소수 1자리 명목 중심이며 공식 FFT의 정확한 10진 중심과 구분한다. 앞 4개는 **전체 원음/EQ 후/관측 마스크 전 peak**를 쓴다. 뒤 2개는 **삽입 crop/O 반영/EQ 전 peak**를 쓴다. 뒤 2개는 post EQ를 바꾸면 requested SNR가 동일해도 최종 총파워가 달라진다. log EQ 평균 0은 선형 파워 gain 평균 1을 의미하지 않는다. 독립 subband는 두 SNR을 각자 추첨하며 broadband SNR draw는 이 연산자에서 사용하지 않는다. 이 차이는 코드에서 확인되지만 어느 쪽이 공식 law인지는 공개되지 않았다.

native complete minute을 만든 뒤 EQ/crop/placement를 한다. 분 경계 정렬·고정 선형 EQ/gain만 있으면 선형 평균과 합산은 교환된다. 부분 분 삽입·timewarp·시간 변화 EQ·비선형 처리는 교환되지 않을 수 있다. 공개 notice는 그 세부를 확정하지 않으므로 minute-first만으로 공식 동등성/실패 원인을 결론 내리지 않는다.

### PCM·PSD·dB를 혼동하지 않기

공식 cell 11은 1차원 finite samples를 float64로 cast할 뿐 **정수 PCM의 스케일이나 Pa 교정을 수행하지 않는다**. 현재 audio.py는 libsndfile의 float32 read로 PCM을 해석하고 채널 0을 선택한다. float32 입력/segment 저장이 공식 helper의 float64 경로와 수치 정밀도상 차이는 만들지만 PSD 평균/적분을 바꾸지는 않는다.

공식 변환은 `Sxx=|rFFT((x−mean(x))·h)|²/(fs·sum(h²))` (h는 Hann 창, 내부 단측 빈은 2배), `P_band=sum(Sxx[low<=f<high])·df`, `df=.5 Hz`이며, 완전한 2초 30개를 선형 평균한다. **PSD bin의 mean이 아니다.** 중심 `10^(1.5+.1k)`, edge `centre·10^(±.05)`이며 high>Nyquist이면 NaN이다. 완전 분 뒤 잔여 시간은 버린다. 디지털 파워에 임의로 `1e−12` 기준을 붙이지 않는다. Pa² 입력일 때만 `10log10(P/1e−12)`이고 현재 SA source-export 절대 교정은 미검증이다. 타깃 y는 공통 gain 기준에 불변이지만 주파수별 응답에는 불변이 아니다.

로컬 pack의 정확한 순서는 (1) 원배경 O와 상대 B 정의 (2) 기하/EQ/alpha로 A 만들기 (3) `A_O=where(O,A,0)` (4) `E=B+A_O` (5) raw q/y/presence 계산 (6) `raw_db=10log10(E)+c` (7) `round(raw_db,1).astype(float32)`이다. 저장 q는 float32, y는 float64이며 round 후 db에서 y를 다시 정의하지 않는다. O 밖 B=0은 관측 에너지 적분의 규약이지 실제 결측 배경이 무음이라는 주장이 아니다.

## 새 수치검산과 재사용한 전수검산

새 fixture는 공식 cell 11 함수만 AST로 추출해 실행한다. 노트북의 Test 로드·상수 제출 셀은 실행하지 않았다. synthetic int16/32 kHz/60초 배열을 명시적으로 32768로 나눈 뒤 공식 helper와 독립 rFFT 식을 비교했다. 이 환경에는 soundfile이 없어 이번 fixture에서 WAV 디코더를 호출했다고 주장하지 않는다. 현재 audio.py 원문과 기존 509 WAV 전수 감사 증거를 함께 확인했다.

CONVERSION_NUMBERS

실제 6연산자×3마스크/support fixture=18건을 독립 전사한 수식과 비교했다. A/q/y/rounded db, 원 O 유지, O 밖 q=0, 명시 zero, 원음 power×100·배경 공통 dB+37의 share 불변을 모두 통과했다(`six_operator_numeric_parity.csv`). 이는 **로컬 법칙 구현 parity**이며 공식 숨은 생성기 parity가 아니다.

SNR_SCOPE_EXAMPLE

기존 R18 WAV 감사는 509 WAV 전체/1,978 완전 분을 공식 helper로 재계산했다. 이번에는 그 작업을 반복하지 않고 기존 `RESULTS.json`의 11개 원본 pin을 현재 SHA와 대조하여 모두 일치를 확인했다. 기존 최대 유의 dB 차는 PRIOR_ERROR dB다. 따라서 PSD 평균/적분 혼동이나 현재 캐시의 대형 변환 버그라는 설명은 현재 증거로 지지되지 않는다. 기록기 물리 교정이나 숨은 합성 동등성까지 입증하지는 않는다.

## 차이와 실패원인의 구분

**확인된 차이/사실:** 공식 starter는 완전 합성기가 아니다. 로컬 6 law는 peak/EQ/O 범위가 서로 다르다. D0/H0의 support가 다르다. R16의 좁은 reference split와 R18의 넓은 465 pool/길이 할당/배경 반복이 서로 다르다. helper와 audio.py는 입력/캐시 float 정밀도·채널 선택 책임에서 차이가 있다. 이 사실들은 구체적 코드와 fixture로 검증했다.

**아직 미확정:** 숨은 Test의 source 선택·SNR 정의·channel·시간 support·관측 mask 라벨 적분·quantization 순서가 무엇인지, 어느 로컬 law가 그것과 다른지, 그 차이가 공식 실패의 원인인지. 공식 notice의 광범위한 조건을 충족하는 여러 생성법이 서로 다른 정답을 만들 수 있다. 현재 자료만으로 원인 한 개를 지목하거나 특정 법칙으로 교체하는 결론은 내릴 수 없다.

이 감사는 추가 학습/hidden label 회수/Test 분포 맞춤을 수행하지 않았다. 미래의 law 비교도 허용 원천·자체 source-purged validation에 기반해 가정과 분포를 먼저 고정해야 한다. 공식 운영진 recipe 추가 공개가 없으면 1:1 숨은 generator 대조의 빈칸은 빈칸으로 유지한다.

## 파일과 재현

- `OFFICIAL_STARTER_FULL_SOURCE.md`, `notebook_cell_*`, `notebook_inventory.json`: 모든공식셀·원문코드·SHA.
- `LOCAL_CODE_SOURCE_EXCERPTS.md`: 원래파일행번호와원문발췌·완전파일SHA.
- `official_forum_local_archive.json`: 관련운영진/참가자원문과구분·archive SHA.
- `six_operator_public_vs_local.csv`: 여섯법칙의공개근거coverage.
- `conversion_fixture.json`, `bands.csv`, `six_operator_numeric_parity.csv`, `operator_fixture_summary.json`: 새순수수치검산.
- `recorded_source_bindings.json`: 생성당시code와현재SHA비교, 기존WAV감사pin재확인.
- `SOURCE_MANIFEST.json`, `RESULTS.json`, `SHA256SUMS.json`: 출처·완료·무변경확인.

재현명령: `PYTHONDONTWRITEBYTECODE=1 .venv_r17_eda/bin/python ops/r18_public_starter_forensics.py`.
'''
    numbers=(f'- 독립 FFT와 공식 helper의 최대 절대차: `{conversion["max_abs_official_vs_independent_FFT"]:.6g}`.\n'
        f'- float32 2초 캐시 후 minute 평균 경로의 최대 유의 상대차: `{conversion["max_significant_relative_float32_segment_then_native_minute_error"]:.6g}`.\n'
        f'- 평탄 PSD일 때 최고/최저 대역 실효폭은 {conversion["last_fft_integration_width_hz"]}/{conversion["first_fft_integration_width_hz"]} Hz, 파워비 {conversion["flat_PSD_last_first_power_ratio"]:.6f}배({conversion["flat_PSD_last_first_db"]:.6f} dB). PSD mean으로 바꾸면 이 대역폭 효과를 없앤다.\n'
        f'- 정수 PCM을 정규화하지 않고 helper에 직접 넣으면 int16의 공통 offset이 {conversion["raw_int16_vs_normalized_db_offset"]:.6f} dB다. 현재 decoder 정규화 누락의 발견이 아니라 단위 예시다.\n'
        '- 61초의 잔여 1초 제외, 59초 reject, 16 kHz 마지막 대역 NaN을 확인했다.\n'
        f'- power 1과 .01의 올바른 선형 평균 후 dB는 {conversion["correct_db_of_linear_mean"]:.6f}, 잘못된 dB 평균은 −10이다.\n')
    e=fixtures['SNR_scope_example']
    example=(f'순수 예시에서 동일 requested SNR=0 dB·15분 일정 배경·9분 원음 중 조용한 3분 삽입을 사용했다. '
        f'원음 peak가 crop 밖에서 100배이면 control alpha={e["control_full_native_afterEQ_alpha"]}, '
        f'share={e["control_share"]:.9f}; inserted-preEQ는 alpha={e["inserted_preEQ_alpha"]}, '
        f'share={e["inserted_preEQ_share"]:.9f}다. 동일 SNR 숫자만으로 동일 실현 share를 가정할 수 없다. '
        '이것은 법칙 차이의 물리 예시이며 실제 Test의 SNR/정답을 추론한 것이 아니다.')
    text=text.replace('CONVERSION_NUMBERS',numbers).replace('SNR_SCOPE_EXAMPLE',example).replace('PRIOR_ERROR',f'{prior["maximum_significant_abs_dB_helper_vs_cache_error"]:.9g}')
    (OUT/'R18_PUBLIC_STARTER_FORENSICS.md').write_text(text)


def main():
    started=time.perf_counter()
    sys.addaudithook(guard)
    OUT.mkdir(parents=True,exist_ok=True)
    pins,notebook=preserve_sources()
    conversion=conversion_fixture(notebook)
    fixture=operator_fixtures()
    r16,r18,prior=recorded_evidence()
    write_report(conversion,fixture,r16,r18,prior)
    unchanged={n:sha(ROOT/n)==h for n,h in pins.items()}
    assert all(unchanged.values())
    dump(OUT/'SOURCE_MANIFEST.json',{'source_files_sha256':pins,'source_files_unchanged':unchanged,
        'script_sha256':sha(__file__),'raw_folder_script_inventory':[str(p.relative_to(ROOT)) for p in (ROOT/'data/raw').rglob('*') if p.is_file() and p.suffix.lower() in {'.py','.ipynb','.md','.txt','.html'}],
        'official_document_scope':'Local supplied notebook and SA docs; local primary forum archive. No live revalidation claimed.'})
    project_access = []
    for name in sorted(OPENED):
        p=Path(name)
        try:
            relative=p.resolve().relative_to(ROOT)
        except (ValueError,OSError):
            continue
        if '.venv' not in str(relative):
            project_access.append(str(relative))
    dump(OUT/'FILE_ACCESS_RECEIPT.json',{'audit_hook_active':True,'project_open_paths':project_access,
        'forbidden_query_and_model_open_paths':[],
        'boundary':'Test.csv, sample submission, query predictions, models and known Test-derived artifact folders denied before open.'})
    dump(OUT/'RESULTS.json',{'status':'completed','script_sha256':sha(__file__),
        'official_full_synthesis_code_present_in_local_starter':False,'official_defined_function':'minute_band_power',
        'official_notebook_cells':len(notebook['cells']),'six_operator_public_hidden_parity':'unavailable, not treated as mismatch',
        'local_independent_formula_comparisons':fixture['comparisons'],'local_formula_checks_passed':True,
        'official_conversion_fixture_passed':True,'all_original_sources_unchanged':all(unchanged.values()),
        'recorded_generation_code_SHA_verified':True,'prior509_WAV_audit_reused_with_current_SHA_binding':True,
        'Test_accessed':False,'new_training':False,'hidden_label_recovery':False,'Test_calibration':False,
        'large_intermediate_files_created':False,'new_synthetic_training_dataset':False,
        'confirmed_official_failure_cause':False,'elapsed_seconds':time.perf_counter()-started,
        'opened_path_count':len(OPENED),'report':'reports/r18_public_gap_audit_r1/starter/R18_PUBLIC_STARTER_FORENSICS.md'})
    dump(OUT/'SHA256SUMS.json',{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'})
    print(json.dumps({'status':'completed','report':str(OUT/'R18_PUBLIC_STARTER_FORENSICS.md'),
        'operator_comparisons':fixture['comparisons'],'source_files_unchanged':all(unchanged.values()),
        'elapsed_seconds':time.perf_counter()-started},ensure_ascii=False))


if __name__=='__main__':
    main()
