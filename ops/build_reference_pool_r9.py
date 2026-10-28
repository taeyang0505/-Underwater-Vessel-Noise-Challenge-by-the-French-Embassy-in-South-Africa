"""Build and independently verify the newly authorized full VT recording pool."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.reference_pool_r9 import (
    ROOT,DEFAULT_OUTPUT,reference_indices_for_epoch,audit_slice,build,
    validate_reference_pool,sha,write,
)


def self_test():
    for fold in range(5):
        a=reference_indices_for_epoch(509,600,1,fold)
        b=reference_indices_for_epoch(509,600,2,fold)
        assert set(a)==set(b)==set(range(509))
        assert np.array_equal(np.r_[a,b],reference_indices_for_epoch(509,1200,1,fold))
        counts=np.bincount(a,minlength=509);assert counts.max()-counts.min()==1
        short=np.concatenate([reference_indices_for_epoch(509,128,e,fold) for e in range(1,5)])
        assert set(short)==set(range(509))
    x=np.ones((59,25),dtype=np.float32);before=x.copy();q=audit_slice(x)
    assert q['complete_minute_count']==q['valid_complete_minute_count']==1
    assert q['trailing_complete2s_bins_without_full_minute']==29
    assert q['has_measured_complete_minute'] and np.array_equal(x,before)
    assert not audit_slice(x[:29])['has_measured_complete_minute']
    bad=x.copy();bad[0,0]=np.nan
    assert not audit_slice(bad)['finite_nonnegative_all25']
    assert audit_slice(bad)['invalid_2s_segments']==1
    empty=x.copy();empty[0]=0
    assert not audit_slice(empty)['strictly_positive_broadband_all_segments']
    negative=x.copy();negative[0,0]=-1
    assert not audit_slice(negative)['finite_nonnegative_all25']
    for args in [(509,600,0,0),(509,600,1,5),(0,600,1,0),(509,600.5,1,0)]:
        try:reference_indices_for_epoch(*args)
        except ValueError:pass
        else:raise AssertionError('Invalid scheduling request accepted')
    print('PASS numeric-only pool QC and continuous5fold reference-index cycling; no source arrays read')


def independent_audit(output):
    folder=ROOT/output;m=json.loads((folder/'manifest.json').read_text());g=validate_reference_pool(output)
    refs=pd.read_parquet(folder/'reference_manifest.parquet')
    ambient=pd.read_parquet(folder/'ambient_control_manifest.parquet')
    q=pd.read_parquet(folder/'clip_quality_manifest.parquet')
    schedule=pd.read_parquet(folder/'training_reference_schedule.parquet')
    order=np.load(folder/'reference_cycle_order.npy',allow_pickle=False)
    assert len(refs)==509 and len(ambient)==16 and len(q)==525
    assert len(schedule)==2545 and not refs.sha256.duplicated().any() and not ambient.sha256.duplicated().any()
    assert not(set(refs.sha256)&set(ambient.sha256))
    assert (refs.role=='vessel_passage').all() and (ambient.role=='ambient_control').all()
    assert refs.native_complete_minute_count.ge(1).all() and q.invalid_2s_segments.eq(0).all()
    assert int(refs.is_weak_background_label_in_vessel_archive.sum())==22
    assert not refs.automatic_presence_or_energy_label_from_original_class.any()
    assert not ambient.automatic_presence_or_energy_label_from_original_class.any()
    for fold,part in schedule.groupby('background_fold'):
        assert set(part.clip_id)==set(refs.clip_id)
        assert set(part.reference_row_index)==set(range(509))
        assert np.array_equal(part.reference_row_index.to_numpy(),order[int(fold)])
        assert (part.usage=='TRAIN').all()
    old_r4=pd.read_parquet(ROOT/'data/versions/vtuad_reference_r4_v1/reference_manifest.parquet')
    old_r6=pd.read_parquet(ROOT/'data/versions/source_split_r6_guardrails_v2/reference_manifest.parquet')
    assert set(old_r4.clip_id)<=set(refs.clip_id) and set(old_r6.clip_id)<=set(refs.clip_id)
    g.update({'full_source_counts':m['counts'],'509_refs_all_five_TRAIN_cycles_verified':True,
              '525_numeric_band_slices_passed':True,'no_padding_repeat_or_new_waveform':True,
              'legacy_R4_selected_refs_retained':len(old_r4),'legacy_R6_refs_retained':len(old_r6),
              'new_reference_records_vs_old_R4':len(refs)-len(old_r4),
              'new_reference_records_vs_old_R6':len(refs)-len(old_r6),
              'old_reference_holdouts_contaminated_for_new_trained_models':True,
              'numeric_fixture':'numeric-only schedule+missingband+zero+negative+singlecomplete-minute passed',
              'artifact_directory':output,'artifact_SHA':m['file_sha256'],
              'input_SHA':m['input_sha256'],'bank_SHA':m['bank']['sha256']})
    return m,g


def report(m,audit,output):
    js=ROOT/'reports/R9_REFERENCE_POOL_AUDIT.json';md=ROOT/'reports/R9_REFERENCE_POOL_AUDIT.md'
    if js.exists() or md.exists():raise FileExistsError('Fresh audit outputs required')
    write(js,audit);c=m['counts']
    rows='\n'.join(f'| {k} | {v} |' for k,v in c['reference_weak_class_counts'].items())
    splits='\n'.join(f'| {k} | {v} |' for k,v in c['original_split_counts_restored'].items())
    text=f'''# R9 全 strict VTUAD 참조 pool — 실제 구축·검증

사용자의 새 명시적 허가에 따라 **새 production 버전에서만** 기존 VT reference 예약/제외를 해제했다. 원본 R4/R5/R6 데이터·모델·패널·SA분할은 덮어쓰지 않았다. 새 pool은 **vessel-archive 측정509개 전체**를 원WAV bytes 기준으로만 중복 검사해509개를 유지하고, **dedicated ambient16개**를 별도 controls manifest로 보관한다. 추가 WAV 다운로드나 학습을 하지 않았다.

## 실제 산출물과 readiness

디렉터리: [vtuad_full_r9](../{output}/manifest.json). `readiness.json.ready=true`와 독립 validator PASS를 확인했다. 이는 **측정 참조 pool 사용 준비 완료**이며 새 합성 generator/배경CV/모델 학습 전체 준비를 대신하는 gate는 아니다.

- 참고 bank: `{m['bank']['path']}` / shape `{m['bank']['shape']}` / dtype `{m['bank']['dtype']}` / SHA `{m['bank']['sha256']}`.
- 새 source_hash: `{m['source_hash']}`.
- 509개 reference 모두 full25band finite·nonnegative·segment broadband>0,native최소1개 완전minute를 확인했다. no-minute0,원WAV byte중복0이다. 미관측 대역을 채우거나upsample/pad/repeat하지 않았다.
- reference header합 **{c['reference_header_seconds']:,.0f}s**, 실제완전2s사용 **{c['reference_complete2s_seconds']:,.0f}s**, ambient header합 **{c['ambient_header_seconds']:,.0f}s**다. 합산시간은독립fieldrecording시간이아니다.
- 과거 R4선택119개/R6선택94개는전부포함되며 새개수차는각각390/415개다. 이를새로운물리녹음의수집으로세지않는다.

## 역할·약한 라벨·품질

| 기존 weak_class | vessel-archive refs/509 |
|---|---:|
{rows}

archive_role=vessel_passage안의 **weak_class=background22개**도사용자all509지시로포함했다. 이를verified vessel-only파워로바꾸거나class명으로추가성분presence/energy label을정하지않는다. 실제추가된whole recording의알려진성분으로q/y를정해야한다. ambient16개도별도controls이지자동으로자연선박부재또는에너지0정답인것이아니다.

모든원래hull/day/UTC/providerWAVSHA/header길이·samplingfs·원래split을보존했다. 이전providerchecksum/원파형QC기록과이번convertedbank수치검증은구분한다. 525개 `large_DC_fraction_review` 경고를지우지않았다. detrend는압력calibration이나pure-vessel분리를보장하지않는다.

| 복원한기존split | vessel refs |
|---|---:|
{splits}

## 5개 background-CV fold에서 참조를 공유

새 user policy에서는모든509개가각fold TRAIN에들어갈수있다. `training_reference_schedule.parquet`는5×509=2545행이며각foldcycle에서509개모두정확히1회등장한다. `reference_cycle_order.npy[5,509]`와독립대조했다.

API: `reference_indices_for_epoch(reference_count,n_examples,epoch,fold,seed=20261026)`는fold별고정permutation을epoch마다중단없이이어가는**참조index cycling**이다. epoch당N을fold안에서고정하고전체epoch순서를한번생성한뒤minibatch로slice한다. N≥509면매epoch모든참조를쓴다. N<509면연속여러epoch를통해fullcycle을완료한다. 이는한합성event에서waveform을반복하는처리가아니다.

**reference 독립5fold가아니다.** 전체525corpus의연결graph는1이며모든fold가참조bank를공유한다. SAbackground분할과24hpurge는별도부모builder에서유지해야한다. 이pool로훈련한새모델에대해과거VTreferencecalibration/domain/outer기반평가는오염된개발비교이므로독립holdout/OOF라부르지않는다. 원래파일이변한것이아니라새모델의훈련접근범위가달라진것이다.

## schema와 무결성

- reference/ambient manifest는`power_cache=segment_file`로원bank를가리키고`segment_start/count`로slice한다. `duration_seconds`는원header길이,`measured_used_seconds`는완전2s사용길이다.
- `expected_segment_slice_sha256`은원float32배열의C-contiguousrawbytes SHA다. `segment_array_identity_sha256`은canonical(shape,dtype)JSON+같은rawbytes SHA다. 원WAV SHA와구분한다.
- `clip_quality_manifest.parquet`는525개각각의valid2s/minute/per-band관측count와sliceSHA를보존한다. class/hull/day는provenance이지모델target이나예측입력이아니다.
- `duplicate_WAV_bytes_manifest.parquet`는이번중복0행을명시한다.

[독립검증JSON](R9_REFERENCE_POOL_AUDIT.json) · [readiness](../{output}/readiness.json) · [새pool코드](../pipeline/reference_pool_r9.py) · [builder/검증](../ops/build_reference_pool_r9.py)
'''
    with md.open('x') as f:f.write(text.replace('R9 全 strict','R9 전체 strict'))
    return {'json_sha256':sha(js),'md_sha256':sha(md),'manifest_sha256':sha(ROOT/output/'manifest.json')}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default=DEFAULT_OUTPUT)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--self-test',action='store_true');g.add_argument('--build',action='store_true');g.add_argument('--verify',action='store_true')
    a=p.parse_args()
    if a.self_test:self_test();return
    if a.build:
        self_test();build(a.output)
        m,g=independent_audit(a.output);receipts=report(m,g,a.output)
        print(json.dumps({'status':g['status'],'output':a.output,'source_hash':m['source_hash'],**receipts}))
    else:
        _,g=independent_audit(a.output);print(json.dumps(g,ensure_ascii=False))


if __name__=='__main__':main()
