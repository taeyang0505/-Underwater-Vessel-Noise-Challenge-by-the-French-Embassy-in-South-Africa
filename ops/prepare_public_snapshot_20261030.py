"""Prepare a reviewed public code/report snapshot; never train or contact a Pod.

Publishing is a separate GitHub operation. Source code bytes are preserved.
Raw observations, waveform/derived arrays, checkpoints and runtimes are omitted.
"""
from __future__ import annotations
import argparse, ast, csv, datetime, hashlib, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'reports/repository_sync_20261030'
EXPORT = AUDIT / 'public_snapshot.nosync'
SECRET = re.compile(rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|\b(?:ghp_|github_pat_|hf_)[A-Za-z0-9_]{24,}|\bAKIA[A-Z0-9]{16}\b|\bsk-[A-Za-z0-9]{32,}')
forbidden_components = {'data', 'data.nosync', 'external', 'external.nosync', 'models', '.git', 'scratch', 'delivery', '__pycache__', 'server_backups'}
report_dirs = {'r16_v1_fourway_r2', 'r16_pretrained_cuda_bf16_r1', 'r17_eda_audit_20261030',
 'r17_clean_pipeline_r1', 'r18_fundamental_audit_r1', 'r18_production_r1', 'r18_public_gap_audit_r1',
 'r19_multislice_law_transfer_r1', 'r20_integrated_cpu_diagnostics_r1', 'r21_clean_val_zero_training_r1'}
records, exclusions = {}, []


def sha(b): return hashlib.sha256(b).hexdigest()
def blob(b): return hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest()
def out(rel, b, source=None, kind='documentation'):
    if isinstance(b, str): b = b.encode('utf-8')
    if SECRET.search(b):
        raise ValueError('Secret-like literal found; publication rejected: ' + rel)
    rel = str(rel)
    if Path(rel).is_absolute() or '..' in Path(rel).parts:
        raise ValueError('Unsafe export path')
    p = EXPORT / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b)
    records[rel] = {'bytes': len(b), 'sha256': sha(b), 'git_blob_sha1': blob(b), 'kind': kind,
                    'source_sha256': source, 'source_bytes_preserved': source == sha(b) if source else None}


def copy(rel, kind='source', sanitize=False):
    p = ROOT / rel
    if not p.is_file() or p.is_symlink(): return
    b = p.read_bytes()
    if SECRET.search(b):
        exclusions.append({'path': rel, 'reason': 'secret-like literal; content not emitted'}); return
    if p.suffix == '.py':
        try: ast.parse(b, filename=rel)
        except SyntaxError:
            exclusions.append({'path': rel, 'reason': 'syntax does not parse in observed Python3.12; frozen source not edited'}); return
    original = sha(b)
    if sanitize:
        # Only prose/receipts are path-normalized. Code bytes are never changed.
        text = b.decode('utf-8-sig')
        text = text.replace(str(ROOT) + '/', '').replace(str(ROOT), '.')
        text = re.sub(r'/Users/[^/\s"<>]+/', '<LOCAL_USER>/', text)
        text = re.sub(r'root@(?:\d{1,3}\.){3}\d{1,3}', '<PRIVATE_SSH_HOST>', text)
        b = text.encode('utf-8')
    out(rel, b, original, kind)


VERIFIER = '''"""Read-only public snapshot checks. No data download, training or model import."""
from pathlib import Path
import ast,csv,hashlib,json
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'handoff/PUBLIC_SNAPSHOT_20261030.json').read_text())
bad=[];syntax=[];rows=0
for rel,rec in manifest['files'].items():
 p=root/rel
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=rec['sha256']:bad.append(rel);continue
 if p.suffix=='.py':
  try:ast.parse(p.read_bytes(),filename=rel)
  except SyntaxError:syntax.append(rel)
 if rel.startswith('submissions/') and p.suffix=='.csv':
  with p.open(encoding='utf-8-sig',newline='') as f:a=list(csv.DictReader(f))
  if len(a)!=1500 or len({r['ID'] for r in a})!=1500:bad.append(rel+':rows/IDs')
  for r in a:
   for c in ('share_added_vessel','score_added_vessel'):
    if not 0<=float(r[c])<=1:bad.append(rel+':range')
  rows+=len(a)
print(json.dumps({'files':len(manifest['files']),'hash_or_CSV_errors':bad,'syntax_errors':syntax,'prediction_rows_checked':rows,'training':False},ensure_ascii=False))
raise SystemExit(bool(bad or syntax))
'''


def prepare():
    EXPORT.mkdir(parents=True, exist_ok=True)
    proposed = json.loads((AUDIT / 'safe_candidate_files.json').read_text())
    # Current first-party code only, not ops nested vendored runtimes/backups.
    sources = {str(p.relative_to(ROOT)) for p in ROOT.glob('*.py')}
    sources |= {str(p.relative_to(ROOT)) for base in ('pipeline', 'ops') for p in (ROOT / base).glob('*.py')}
    sources |= {str(p.relative_to(ROOT)) for p in ROOT.glob('*.sh')}
    sources |= {str(p.relative_to(ROOT)) for p in (ROOT / 'ops').glob('*.sh')}
    sources |= {str(p.relative_to(ROOT)) for p in (ROOT / 'configs').glob('*') if p.is_file() and p.suffix in ('.json', '.yaml', '.yml', '.toml')}
    sources |= {str(p.relative_to(ROOT)) for p in ROOT.glob('requirements*') if p.is_file()}
    for rel in sorted(sources): copy(rel)
    # Own prose reports and genuine aggregate evidence, not per-window matrices.
    aggregate = proposed['safe_aggregate_evidence_files'] + proposed['requires_sanitization_or_review']
    banned_names = re.compile(r'row_predictions|row_features|physical_rows|sample_metadata|eligible_identity|train_source_rows|all_sa_strict|source_rows|selected_negative|reference_509_all|reference_509_status|reference_training_exposure|reference_identity_audit|sample_.*catalog|physical\.csv$', re.I)
    for rec in aggregate:
        rel = rec['path']; p = Path(rel)
        if len(p.parts) < 3 or p.parts[0] != 'reports' or p.parts[1] not in report_dirs: continue
        if any(x in p.parts for x in ('data', 'data_multisite', 'inputs', 'features', 'refit', 'rejected_probe_direct_O_zero')): continue
        if p.suffix not in ('.md', '.csv', '.json'): continue
        if banned_names.search(p.name):
            exclusions.append({'path': rel, 'reason': 'per-window/row/source-level derived acoustic evidence; private artifact'}); continue
        if p.suffix == '.csv':
            with (ROOT / rel).open(encoding='utf-8-sig', newline='') as f:
                header = next(csv.reader(f), [])
            if set(header) & {'sample_id', 'source_clock_id', 'base_window_index', 'row_id', 'true_q', 'db'}:
                exclusions.append({'path': rel, 'reason': 'row-level derived data identifier/features'}); continue
        copy(rel, 'report_or_aggregate', sanitize=p.suffix in ('.md', '.json'))
    # Explicit latest materials whose filenames postdate the candidate scan.
    for rel in [
        'README.md', 'reports/r21_clean_val_zero_training_r1/START.html',
        'reports/r21_clean_val_zero_training_r1/QUIET_B_FEATURE_DICTIONARY.csv',
        'reports/r21_clean_val_zero_training_r1/FINAL_DECISION.json',
        'reports/r21_clean_val_zero_training_r1/CLEAN_VAL_GATE_RESULTS.json',
        'reports/r21_clean_val_zero_training_r1/VERIFICATION_QA.json',
        'reports/r21_clean_val_zero_training_r1/data_multisite/SOURCE_AUDIT.json',
        'reports/r21_clean_val_zero_training_r1/data_multisite/DATASET_NOTES.md',
        'reports/r21_clean_val_zero_training_r1/data_multisite/POSTFREEZE_QA.json',
        'reports/repository_sync_20261030/STORAGE_INVENTORY.md',
        'reports/repository_sync_20261030/storage_inventory.json',
        'reports/repository_sync_20261030/storage_inventory.csv']:
        copy(rel, 'readme' if rel == 'README.md' else 'report_or_aggregate', sanitize=Path(rel).suffix in ('.json', '.md') and rel != 'README.md')
    for folder in ('r17_fixed_blends_20261030', 'r18_production_r1', 'r16_v1_fourway_r2'):
        for p in sorted((ROOT / 'submissions' / folder).glob('*')):
            if p.suffix in ('.csv', '.md', '.json'):
                copy(str(p.relative_to(ROOT)), 'first_party_prediction' if p.suffix == '.csv' else 'prediction_provenance', sanitize=p.suffix in ('.md', '.json'))
    out('handoff/verify_snapshot_20261030.py', VERIFIER, kind='verification_utility')
    env = (AUDIT / 'audit_environment.json').read_bytes()
    out('handoff/audit_environment_20261030.json', env, kind='environment_record')
    out('CLAUDE.md', '''# 현재 인계 지침 — 2026-10-30 / R21

이 문서와 README가 과거10월28일학습/복구지시보다 우선한다. 한국어로 소통한다.

1. **추가 모델 재학습·GPU/MPS 학습·새 K-Fold 확대 금지.** RunPod는사용자가Stop확인. 오래된waiter/복구명령/PID를자동재실행하지않는다.
2. R17 old_B1이공식최고: .789471136 / RMSE .210528863 / AUC .572711111. R18 new_B2는 .778723110 / .221276889 / .573555555로악화. old_B2는같은old모델50/30/20이며new_B2와다르다.
3. R21의CPU무학습실험완료. Clean-Val100행은50배경의paired stress이며Test대표집합/새최종홀드아웃아님. old_B2가사전정책후보중엄격게이트통과했지만개선CI는0포함,Public개선미확정.
4. Quiet4종및5/10%결합은기각. 기존LightGBM에quiet_B_share를추가학습한상태아님. 순수마스크importance와음향차분/lag를구분한다.
5. 원본프로젝트의모델/데이터/동결source/PLAN/manifest는수정하지않는다. 검증코드는새버전으로작성하며원본과공개정제본SHA를구분한다.
6. 원자료/Test관측값/외부WAV/합성행렬/vendor가중치/비밀키·토큰을공개Git에넣지않는다. 공식대회제출·외부자문전송·Pod구매/시작은자동수행하지않는다.
7. README→handoff/CURRENT_STATE.md→R21/R20원문→실행코드/동결영수증순서로읽는다. 과거문서의가설을사실로승격하지않는다. 로컬RMSE/AUC/분포일치로실전성능을보장하지않는다.

공개clone은스냅샷이다. 최신원자료와가중치는포함되지않는다. 먼저 `python3 handoff/verify_snapshot_20261030.py`로공개SHA/구문/CSV를검증한다. 실제학습프로세스상태를확인하지않고과거PID가살아있다고보고하지않는다.
''', kind='current_handoff')
    out('AGENTS.md', '''# Instructions for this public snapshot — 2026-10-30

Read README.md, CLAUDE.md and handoff/CURRENT_STATE.md first. Their current no-training contract supersedes historical Fold1–4/waiter/recovery plans. This is a snapshot, not the live training directory. Do not train, resume, start GPU/MPS jobs or execute historical supervisors. Preserve canonical frozen code/data/model/policy bytes; new checks belong in separately versioned files. Never publish raw competition observations, waveforms, synthetic acoustic matrices, vendor weights or credentials. User-requested source/report publication is authorized; competition uploads, external consultation and Pod actions are not. Report evidence and uncertainty in Korean. Verify the public snapshot manifest before claiming completion; original experiment receipts may reference private excluded artifacts.
''', kind='current_handoff')
    out('handoff/CURRENT_STATE.md', '''# 현재 상태 — 2026-10-30 / R21 완료

최신상태의정식요약은 [README](../README.md)다. 이전10월28일학습중snapshot과Fold1~4후속계획은폐기/역사기록이다. **지금학습을시작하지않는다.**

- R16 version1 LGB/Cat각D0/H0×5fold×회귀/분류완료. Dasheng/DINO/TimeMoE각Fold0 D/H×독립회귀/분류완료. 최고R17 B1에는LGB/DINO/Dash60/25/15만사용.
- R18 LGB는D/H단일분할, DINO/Dash각4작업완료. TimeMoE/Cat추가학습제외. 공식new_B2는최고점대비악화.
- R19~R21은CPU저장예측·동결추론·원천/물리피처감사. R21새스트레스100행평가완료,기존생산모델/제출파일변경없음.
- 새Clean-Val정책후보old_B2만통과. Quiet와입력마스크가림은오탐해결실패. 검증대표성/실전AUC문제는미해결.
- 남은과제는독립배경/참조/법칙전이,자연HF음성과약한양성분리,구간별비열화없는검증체계다. 새학습은현재승인밖이다.
- version2 ConvNeXt-V2-Base는별도후보/step검토상태이며본학습/제출완료아님.

최신결과: [R21보고서](../reports/r21_clean_val_zero_training_r1/R21_CLEAN_VAL_ZERO_TRAINING_REPORT.md), [R20통합](../reports/r20_integrated_cpu_diagnostics_r1/R20_INTEGRATED_DIAGNOSTICS_REPORT.md).

Public수치는사용자제공공식기록이며실시간LB조회가아니다. 기존model Release는10월28일시점일뿐현재PEFT/R18복원본을보장하지않는다. 공개범위와파일해시는PUBLIC_SNAPSHOT_20261030.json, 용량은README §6를참고한다.
''', kind='current_handoff')
    out('handoff/PUBLICATION_SCOPE_20261030.md', '''# 2026-10-30 공개 범위와 복원 한계

현재 자체source·configs·최신보고서·집계CSV·첫번째당사자예측CSV를게시한다. sourcebytes는원본과같고, 일부prose/receipt의개인절대경로와SSH주소만정제했으며source/exportSHA를manifest로구분한다.

공개하지않은자료: 원본SA/Test관측값/외부WAV/음향합성db,q,y행렬/행별특징표/학습cache/최신모델checkpoint/vendor가중치/환경runtime/비밀인증자료/Pod운영로그·원격state. 과거자료와Release는삭제하지않았다. 따라서Gitclone만으로최신실험을재실행할수없다. 원본데이터정식취득및private모델복원은DATA_AND_RESTORE.md와해당동결입력manifest를따른다.

원본실험영수증은제외된private파일해시도포함할수있고, 공개용문서정제본SHA와다를수있다. clone검증에는PUBLIC_SNAPSHOT_20261030.json과verify_snapshot_20261030.py를사용한다. 옛EXPORT_MANIFEST/verify_checkout.py는과거스냅샷기준이며이번갱신검증도구가아니다.

지금업로드는공식대회제출도,모델재학습도,새로운모델배포도아니다. 타사자료의권리는원출처에있으며저장소전체에재라이선스를선언하지않는다. 원천코드·모델·실측값의인과관계를단정하거나Public점수를라벨역산에쓰지않는다.
''', kind='publication_scope')
    ignore = '''# Private licensed/raw/derived acoustic payloads and model/runtime artifacts
/data/
/data.nosync/
/external/
/external.nosync/
/models/
/scratch/
/outputs/
/delivery/
/ops/server_backups/
/ops/r14_production_runtime_py/
*.wav
*.flac
*.mp3
*.zip
*.tar.gz
*.pt
*.pth
*.safetensors
*.npy
*.npz
*.parquet
*.pkl
*.joblib
*.log
__pycache__/
*.pyc
.DS_Store
.venv*/
.env
.env.*
!.env.example
id_rsa*
id_ed25519*
credentials*
*.pem
'''
    out('.gitignore', ignore, kind='publication_scope')
    if (EXPORT / 'handoff/EXPORT_REVIEW_20261030.json').exists():
        (EXPORT / 'handoff/EXPORT_REVIEW_20261030.json').unlink()
    out('handoff/EXPORT_REVIEW_20261030.json', json.dumps({'rejected_candidates': exclusions, 'no_data_or_training': True,
        'secret_literals_emitted': False, 'code_bytes_preserved': True}, ensure_ascii=False, indent=2) + '\n', kind='export_review')
    current = {'timestamp_KST': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
       'scope': 'Current public source/docs/aggregate evidence/predictions only; excludes private acoustic arrays and current checkpoints',
       'files': records, 'payload_bytes_excluding_manifest': sum(x['bytes'] for x in records.values()),
       'files_excluding_manifest': len(records), 'new_training': False, 'official_submission': False,
       'old_release_is_current_checkpoint_backup': False}
    p = EXPORT / 'handoff/PUBLIC_SNAPSHOT_20261030.json'
    p.write_text(json.dumps(current, ensure_ascii=False, indent=2) + '\n')
    allfiles = dict(records)
    b=p.read_bytes();allfiles[str(p.relative_to(EXPORT))]={'bytes':len(b),'sha256':sha(b),'git_blob_sha1':blob(b),'kind':'public_manifest'}
    (AUDIT / 'PUBLIC_EXPORT_FILES.json').write_text(json.dumps(allfiles, ensure_ascii=False, indent=2) + '\n')
    cache_upload_batches(allfiles)
    print(json.dumps({'directory': str(EXPORT), 'files': len(allfiles), 'bytes': sum(x['bytes'] for x in allfiles.values()),
          'source_files': sum(x['kind']=='source' for x in allfiles.values()), 'excluded_count': len(exclusions),
          'manifest_SHA256': sha(p.read_bytes())}, ensure_ascii=False))


def cache_upload_batches(files):
    # Safe UTF-8 chunks for connector Git tree uploads. The caller reassembles
    # content per path, then atomically advances the Git ref after verification.
    chunks=[]
    for rel,rec in sorted(files.items()):
        text=(EXPORT/rel).read_bytes().decode('utf-8')
        # Character chunks preserve Unicode; budget leaves JSON overhead space.
        for n in range(0,len(text),16000):
            chunks.append({'path':rel,'offset':n,'last':n+16000>=len(text),'content':text[n:n+16000],
                           'git_blob_sha1':rec['git_blob_sha1'],'sha256':rec['sha256']})
        if not text:chunks.append({'path':rel,'offset':0,'last':True,'content':'','git_blob_sha1':rec['git_blob_sha1'],'sha256':rec['sha256']})
    batches=[];pending=[];size=0
    for item in chunks:
        length=len(json.dumps(item,ensure_ascii=False).encode())
        if pending and size+length>100000:batches.append(pending);pending=[];size=0
        pending.append(item);size+=length
    if pending:batches.append(pending)
    folder=AUDIT/'upload_batches';folder.mkdir(exist_ok=True)
    for n,batch in enumerate(batches):
        (folder/f'{n:04d}.json').write_text(json.dumps({'number':n,'total':len(batches),'chunks':batch},ensure_ascii=False))
    (folder/'INDEX.json').write_text(json.dumps({'count':len(batches),'private_transport_only':True}))


def emit_batch(number):
    print((AUDIT/'upload_batches'/f'{number:04d}.json').read_text())


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--emit-batch',type=int);args=a.parse_args()
    if args.emit_batch is None:prepare()
    else:emit_batch(args.emit_batch)
