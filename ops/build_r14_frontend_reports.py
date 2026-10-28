"""Render R14 diagnostic evidence; no model/data-array execution."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(n):return json.loads((ROOT/n).read_text())

def main():
    cpu_name='reports/R14_TORCH_FRONTEND/RESULTS.json';full_name='reports/R14_TORCH_FRONTEND/FULL_EPOCH_V2.json';mps_name='reports/R14_MPS_BENCHMARK_r2/RESULTS.json'
    cpu,full,mps=map(read,(cpu_name,full_name,mps_name))
    assert cpu['status']==mps['status']=='passed' and full['status']=='completed'
    for report in (cpu,full):
        for n,h in report['code_sha256'].items():assert sha(ROOT/n)==h,n
    for n,h in cpu['original_R13_code_sha256'].items():assert sha(ROOT/n)==h,n
    for n,h in mps['source_code_sha256'].items():assert sha(ROOT/n)==h,n
    source_names=['pipeline/features_r14_torch.py','pipeline/raw_frontend_r14.py','ops/verify_r14_torch_frontend.py',
        'ops/benchmark_r14_full_epoch.py','ops/benchmark_r14_mps.py','ops/build_r14_frontend_reports.py',
        'ops/r14_runtime_py/tabm.py','ops/r14_runtime_py/rtdl_num_embeddings.py','ops/r14_runtime_py/receipt.json']
    closure={n:sha(ROOT/n) for n in source_names}
    feature_error=max(max(v['channel_max_abs_error']) for v in cpu['feature_checks']);weight_error=max(v['weights_max_abs_error'] for v in cpu['feature_checks'])
    summary={'status':'CPU_frontend_verified_not_production_integrated','feature_count':11,'data_hash':cpu['data_hash'],
        'feature_max_abs_error':feature_error,'weights_max_abs_error':weight_error,
        'source_code_sha256':closure,'original_R13_code_unchanged':True,
        'evidence_sha256':{n:sha(ROOT/n) for n in (cpu_name,full_name,mps_name)},
        'CPU4096_benchmark':cpu['benchmark'],'CPU_full_epoch':full,'model_checks':cpu['model_checks'],
        'MPS_fp64_frontend_supported':False,'CUDA_frontend_parity_latency_VRAM_verified':False,
        'production_runner_or_approval_modified':False,'anchor14channels_integrated':False,
        'Test_report_newdatafit_or_network':False}
    out=ROOT/'reports/R14_TORCH_FRONTEND.json';md=out.with_suffix('.md')
    if out.exists() or md.exists():raise FileExistsError('Fresh diagnostic reports required')
    out.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    md.write_text(f'''# R14 배치 Torch 11채널 전처리 검증

**새 R14 경로의 수치·연결 검증을 통과했다.** 원형 R13 모델·자료·정책·생산 CUDA guard는 변경하지 않았다. 기존 생산 러너가 자동으로 이 경로로 바뀐 것은 아니며, 부모가 별도로 검토 중인 LF-anchor 14채널도 통합하지 않았다.

## 구현과 API

- `pipeline/features_r14_torch.py`: `transform(db, basis)->(x,w)`, 또는 반복 실행에 적합한 `TorchFeatureFrontend(basis)` 모듈. db만 받고 frozen TRAIN PCA를 buffers로 보관한다. 기본 출력은 float32 `[B,11,15,25]`와 `[B,15,25]`다.
- float64 round-to-even0.1dB, 관측값 linear quantile/median, 인접 상승·q05·sqrt(w), safe23 PCA 행렬곱·부분관측 최소제곱·곡률을 배치 Torch로 계산한다. median은 짝수 관측에서 중간 두 값을 평균해야 하므로 lower-median을 쓰지 않았다.
- `pipeline/raw_frontend_r14.py`: `RawR14Dataset(parent)`는 원 `raw_item()`으로 db/mask/q/y를 읽고 Dataset 안에서 특징을 계산하지 않는다. `RawFeatureMaskModel(backbone,basis)`는 forward에 db/mask만 받아 특징→기존 본체를 연결한다. `return_weights=True`이면 `(outputs,w)`를 반환해 원 물리 손실을 적용한다. q/y/presence/site는 forward 입력이 아니다.
- P0는 기존8채널을 그대로 두고 3채널을 zero-pad한다. PCA는 기존 fold-TRAIN 적합 결과를 읽을 뿐 새로 적합하지 않는다. 물리 정답은 재정의하지 않는다.

## 수치·학습 연결

프로토콜을 결과 전에 저장했다. 허용오차는 특징3e−6, w3e−8, 상대오차0이다. synthetic float32/64·rounding ties·NaN minute·없는 전체 band·partial safe-band와 실제 TRAIN4,096개를 비교했다. 최대 특징 차이는 **{feature_error:.4g}**, w 차이는 **{weight_error:.4g}**였다. 추가3개 shape채널은23/24번 band를 바꿔도 동일했다. 기존8채널과 full25 물리 w의 상단 band 의존은 유지한다. inf/전부결측/정답dict/불일치mask는 거부했다.

전체크기 WideBG/TFGrid-FDY-Conformer/FDY/iTransformer 모두 기존 NumPy 특징 입력과 새 raw-forward 출력 차이0, 실제 TRAIN으로 Adam1step finite gradient, frontend trainable parameter0, checkpoint strict round-trip bitwise를 확인했다. 이는 모델 성능 개선이나100epoch 수렴 증거가 아니다. [원수치·모델별 검사](R14_TORCH_FRONTEND/RESULTS.json)

## CPU 비용 분리

동일4,096개·CPU2threads에서 합성만 **{cpu['benchmark']['synthesis_only_seconds']:.3f}초**, NumPy 개별 특징 **{cpu['benchmark']['NumPy_per_item_seconds']:.3f}초**, NumPy batch128 **{cpu['benchmark']['NumPy_batched128']['median_seconds']:.3f}초**, Torch batch128 **{cpu['benchmark']['Torch_batched128']['median_seconds']:.4f}초**였다. 배치 특징 부분의 큰 감소와 전체 합성비용을 구분해야 한다.

별도로 실제 Fold0 TRAIN **{full['actual_primary_rows']:,}개 전부**를1회 읽었다. raw 합성·collation **{full['raw_synthesis_collation_seconds']:.3f}초**, 배치 Torch 특징 **{full['batched_Torch_features_seconds']:.3f}초**, stream SHA 포함 전체 **{full['total_seconds_including_stream_hash']:.3f}초**였다. known-zero15,774개와 입력·출력 stream을 기록했고 원천 SHA가 유지됐다. [전수 벤치마크](R14_TORCH_FRONTEND/FULL_EPOCH_V2.json)

이전1thread NumPy 전수 경로115.44초와 새2thread 경로는 thread/실행시점이 다르므로 엄격한 같은 조건 속도비로 부르지 않는다. 새 전수 측정에는 DataLoader multiprocessing·본체 학습·checkpoint·validation·GPU 전송이 없다. 첫 전수 측정15.817초도 이력으로 보존했고, 부하 존재를 단정하던 메타데이터를 '부하 통제하지 않음'으로 정확히 기록한 새 측정16.030초를 함께 남겼다. host 동시 부하는 통제하지 않았다. 남은 약13초의 합성비용이 사라졌거나 모든64 CPU시간이 제거됐다고 주장하지 않는다.

## GPU 경계와 원본 보존

CUDA에서 전처리까지 같은 forward 안에 두는 구현 구조는 준비했지만 **CUDA 수치·지연·VRAM은 미측정**이다. FP64 기준을 검증 없이 FP32로 바꾸지 않는다. MPS는 FP64를 지원하지 않아 별도 CPU-FP64 frontend→FP32 MPS 본체 경로를 실제 검증했다. 이 모듈 전체를 `.half()`/`.float()`로 바꿔 PCA buffer를 변형하면 거부한다. [MPS 실측](R14_MPS_BENCHMARK.md)

활성 원자료 hash는 `{cpu['data_hash']}`이고 원 R13 코드30개 SHA를 실행 전후 확인했다. 새 특징에 q0/정답/추정 site/공개 Test 분포를 넣지 않았다. 새 본학습·공식 제출은 하지 않았다. [소스 closure·기계 판독 보고서](R14_TORCH_FRONTEND.json)
''')
    batch32=[r for r in mps['results'] if r['batch_size']==32]
    rows='\n'.join(f"|{r['kind']}|{r['parameters']:,}|{r['backbone_timing']['CPU']['median_step_seconds']:.5f}|{r['backbone_timing']['MPS']['median_step_seconds']:.5f}|{float(np_median(r['MPS_CPUfrontend_transfer_Adam_seconds'])):.5f}|" for r in batch32)
    ms={'status':'actual_MPS_short_Adam_steps_passed','measured_conditions':12,'failures':0,
        'FP64_frontend_on_MPS':False,'explicit_CPU_FP64_frontend_MPS_FP32_backbone':True,
        'source_code_sha256':closure,'evidence_sha256':{mps_name:sha(ROOT/mps_name)},
        'raw_result':mps_name,'original_R13_frozen_sources_unchanged':True,'production_launch_authorized':False,
        'full_epoch_or_100epoch_elapsed_time_measured':False,'total_training_time_estimated':False}
    mout=ROOT/'reports/R14_MPS_BENCHMARK.json';mmd=mout.with_suffix('.md')
    if mout.exists() or mmd.exists():raise FileExistsError('Fresh MPS summary required')
    mout.write_text(json.dumps(ms,indent=2)+'\n')
    mmd.write_text(f'''# M4 Pro MPS 실제 본체 학습 가능성 진단

**M4 Pro20 GPU코어·24GB에서 실제4종 본체 모두 짧은 FP32 학습을 수행할 수 있었다.** batch8/16/32 총12조건에서 forward/backward/Adam이 finite였고 unsupported-operator CPU fallback은0으로 두었다. 이는 본학습 완료나 CUDA 대체 속도 보장이 아니다.

환경은 Python3.14.5/PyTorch{mps['torch']} 사용자 venv다. CPU는2threads, 각 조건 warmup1step 후3개 Adam step의 중앙값이다. host 동시 CPU 부하를 통제하지 않았고 GPU 최초 compile 비용을 전체 학습시간에 외삽하지 않았다. 매 조건 fresh weights와 별도 optimizer로 시작했으며 생산 checkpoint를 변경하지 않았다. CPU/MPS eval 출력 최대 차이는 {max(max(r['CPU_MPS_eval_max_abs_errors_share_score_mask']) for r in mps['results']):.4g}로 사전 tolerance 안이었다. CPU/MPS dropout RNG가 다르므로 optimizer 궤적 동일성은 주장하지 않는다.

|본체|파라미터|CPU2threads1step(초)|MPS1step(초)|CPU특징+전송+MPS1step(초)|
|---|---:|---:|---:|---:|
{rows}

모두batch32 결과다. Wide/TFGrid/FDY는 이 조건에서 빨랐고 **iTransformer는 CPU가 더 빨랐다.** MPS 사용 여부를 모델마다 무조건 동일하게 판단할 수 없다. [batch8/16 포함12조건 원수치](R14_MPS_BENCHMARK_r2/RESULTS.json)

동기화된 step 이후 관측 driver 메모리는 최대 **{max(r['observed_max_driver_bytes'] for r in mps['results'])/2**30:.2f}GiB**, process lifetime peak RSS는 **{max(r['process_lifetime_peak_RSS_bytes'] for r in mps['results'])/2**30:.2f}GiB**였다. 이는 activation 순간 peak가 아니고 CPU/Metal 공유 메모리의 다른 앱·Tree 작업까지 포함한 안전 headroom 보장이 아니다. batch128이나 더 큰 모델의 가용성을 추측하지 않는다. 실제 full epoch·100epoch 전체 시간이 없으므로 전체 소요시간은 추정하지 않았다.

FP64 MPS 할당이 실제로 거부됐다. 따라서 R14의 NumPy-parity FP64 특징은 **CPU에서 계산한 뒤 x/w와 정답 tensor만 FP32로 MPS에 전송**했다. FP64 frontend를 MPS에서 직접 실행했다고 표현하지 않는다. [`MPS 문서`](https://docs.pytorch.org/docs/stable/notes/mps.html), [`Apple PyTorch 안내`](https://developer.apple.com/metal/pytorch/)와 별도로 실제 dtype probe 오류를 원JSON에 남겼다.

첫 진단은 venv에 tabm이 없어 모델 초기화 전에 실패했고 그대로 보존했다. 원 모델의 공통 import를 수정하지 않고 기존 순수 Python tabm/rtdl 두 파일을 SHA 그대로 workspace overlay에 제공한 fresh r2에서12조건을 실행했다. 다운로드·venv 설치·숨은 모델 대체는 없었다. [overlay receipt](../ops/r14_runtime_py/receipt.json) / [처음 실패 이력](R14_MPS_BENCHMARK/RESULTS.json)

원 R13 CUDA-only 생산 guard와 승인false는 유지한다. 이 결과는 MPS 지원을 검증하는 신규 학습엔진을 별도로 만드는 근거이며 원 러너가 지금 MPS 본학습을 지원한다는 뜻이 아니다. LF-anchor14채널도 본 진단에 넣지 않았다. [소스 closure](R14_MPS_BENCHMARK.json)
''')
    print(json.dumps({'reports':{str(p.relative_to(ROOT)):sha(p) for p in (out,md,mout,mmd)}}))

def np_median(x):return sorted(x)[len(x)//2]
if __name__=='__main__':main()
