#!/usr/bin/env python3
"""One disposable CPU Adam step per full-size model: real R9 TRAIN inputs.

Connection fixture only. No production model, no performance comparison,
no report role, Test, GPU, SSH, waveform decoding or existing-file changes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from pipeline.dataset_r9 import R9Sources, DynamicR9Dataset, FixedR9Dataset, array_sha, sha, canonical
from pipeline.models_r9 import make_r9_model
from pipeline.training_r9 import physical_multitask_loss
MODEL_KINDS=('bg_cnn','tfgrid_fdy_conformer','fdy','itransformer')


def _batch(dataset,indices):
    rows=[dataset[i] for i in indices]
    return {key:torch.from_numpy(np.stack([row[key] for row in rows])).float()
            for key in ('x','w','share','cell_share','presence')},rows


def _tensor_hash(batch):
    return {k:array_sha(v.detach().cpu().numpy()) for k,v in batch.items()}


def run(config,bundle,output):
    start=time.monotonic();output=Path(output)
    if output.exists():raise FileExistsError('Fresh verification output required')
    torch.set_num_threads(2);torch.manual_seed(4926)
    sources=R9Sources(config)
    train=DynamicR9Dataset(config,0,sources=sources)
    selection=FixedR9Dataset(bundle,0,'selection')
    ordered_minutes=train.metadata.qc_observed_minutes.to_numpy()[train.schedule['primary_order']]
    full=np.flatnonzero(ordered_minutes==15)[:2].tolist()
    missing=np.flatnonzero(ordered_minutes<15)[:2].tolist()
    if len(full)!=2 or len(missing)!=2:raise AssertionError('Need actual present/missing TRAIN inputs')
    indices=full+missing
    batch,rows=_batch(train,indices)
    validation,validation_rows=_batch(selection,list(range(8)))
    original_hashes=_tensor_hash(batch);validation_hashes=_tensor_hash(validation)
    labels=(batch['share'],batch['presence'],batch['cell_share'])
    val_labels=(validation['share'],validation['presence'],validation['cell_share'])
    loss_weight=torch.tensor([float(row['loss_weight']) for row in rows])
    if not (torch.isfinite(loss_weight).all() and (loss_weight>0).all()):raise AssertionError('Invalid site weight')
    if not torch.equal(batch['x'][:,5]>0,batch['w']>0):raise AssertionError('Mask/w mismatch')
    if not torch.equal(batch['x'][:,7],batch['w'].sqrt()):raise AssertionError('Rounded w/sqrt(w) mismatch')
    pins={n:sha(ROOT/n) for n in ['pipeline/models_r9.py','pipeline/training_r9.py',
                                 'pipeline/dataset_r9.py','pipeline/sampler_r9.py',
                                 'pipeline/relative_grid_r6.py','pipeline/relative_grid_r6_v2.py',
                                 'pipeline/hybrid_models_r5.py','pipeline/models.py']}
    folder=Path(tempfile.mkdtemp(prefix='r9_REAL_DATA_connection_fixture_DO_NOT_REUSE_',dir='/private/tmp'))
    cases=[]
    for kind in MODEL_KINDS:
        case_start=time.monotonic();torch.manual_seed(4926)
        model=make_r9_model(kind,dropout=.1).float().cpu()
        architecture=model.architecture()
        parameters=sum(p.numel() for p in model.parameters())
        if parameters!=architecture['total_parameters']:raise AssertionError('Architecture count mismatch')
        optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0003)
        model.train();optimizer.zero_grad(set_to_none=True)
        outputs=model(batch['x'],batch['w'])
        per_row,terms=physical_multitask_loss(outputs,labels,batch['w'])
        loss=(per_row*loss_weight).mean()
        if not torch.isfinite(loss):raise AssertionError('Nonfinite actual-data connection objective')
        loss.backward()
        grads=[p.grad for p in model.parameters() if p.requires_grad]
        if any(g is None or not torch.isfinite(g).all() for g in grads):raise AssertionError('Nonfinite/missing gradient')
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise AssertionError('Nonfinite clipped gradient')
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):raise AssertionError('Nonfinite updated parameter')
        steps=[int(state['step'].item()) for state in optimizer.state.values() if 'step' in state]
        if not steps or set(steps)!={1}:raise AssertionError('Exactly one Adam update required')
        model.eval()
        with torch.no_grad():
            out=model(validation['x'],validation['w'])
            validation_loss,_=physical_multitask_loss(out,val_labels,validation['w'])
        if not torch.isfinite(validation_loss).all():raise AssertionError('Selection connection invalid')
        path=folder/(kind+'.pt')
        torch.save({'fixture_only':True,'never_reuse_for_production':True,'architecture':architecture,
                    'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),
                    'optimizer_steps':1,'training_dataset_identity':train.identity,
                    'selection_dataset_identity':selection.identity,'code_sha256':pins},path)
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        if checkpoint['architecture']!=architecture or checkpoint['code_sha256']!=pins:
            raise AssertionError('Saved architecture/code identity differs')
        restored=make_r9_model(checkpoint['architecture']['kind'],**checkpoint['architecture']['model_settings'])
        if restored.architecture()!=architecture:raise AssertionError('Model architecture reconstruction differs')
        restored.load_state_dict(checkpoint['model_state'],strict=True);restored.eval()
        with torch.no_grad():replayed=restored(validation['x'],validation['w'])
        if any(not torch.equal(a,b) for a,b in zip(out,replayed)):
            raise AssertionError('Reload predictions must be bitwise identical')
        if any(v.dtype!=torch.float32 for v in out):raise AssertionError('FP32 pool/output contract differs')
        q=out[2];w=validation['w']
        pooling_error=float((out[0]-(q*w).sum((1,2))).abs().max())
        if pooling_error>3e-6 or torch.any(q[w==0]!=0):raise AssertionError('Physical pool/masked q differs')
        cases.append({'kind':kind,'default_full_model':True,'total_parameters':parameters,
                      'architecture':architecture,'train_rows':len(indices),'selection_rows':8,
                      'optimizer':'Adam','learning_rate':.001,'weight_decay':.0003,
                      'dropout':.1,'clip_gradient':1.,'optimizer_steps':1,
                      'all_gradients_and_updated_parameters_finite':True,
                      'numeric_objective_not_performance':float(loss.detach()),
                      'raw_target_vs_rounded_weight_max_gap':float(terms['raw_label_vs_rounded_pool_gap'].max()),
                      'prediction_pool_max_error':pooling_error,'strict_reload_bitwise':True,
                      'checkpoint_sha256':sha(path),'temporary_checkpoint_file':str(path),
                      'elapsed_seconds':time.monotonic()-case_start})
        del model,restored,optimizer,checkpoint,outputs,out,replayed,grads
        gc.collect()
    if _tensor_hash(batch)!=original_hashes or _tensor_hash(validation)!=validation_hashes:
        raise AssertionError('Fixture modified input/label bytes')
    sources.assert_unchanged()
    if any(sha(ROOT/n)!=value for n,value in pins.items()):raise AssertionError('Code changed during fixture')
    report={'status':'passed','created_utc':datetime.now(timezone.utc).isoformat(),
            'fixture_only':True,'actual_source_connection_test':True,
            'production_training_or_performance_evaluation':False,
            'report_role_or_Test_or_GPU_or_SSH_or_waveform_accessed':False,
            'CPU_threads':2,'torch_version':torch.__version__,'numpy_version':np.__version__,
            'source_identity':sources.identity,'data_bundle_hash':selection.data_bundle_hash,
            'config_file_sha256':sha(config),'code_sha256':pins,'fixture_code_sha256':sha(__file__),
            'train_indices':indices,'TRAIN_observed_minutes':[int(ordered_minutes[i]) for i in indices],
            'train_sample_ids':[str(row['sample_id']) for row in rows],
            'selection_sample_ids':[str(row['sample_id']) for row in validation_rows],
            'input_tensor_sha256':original_hashes,'selection_tensor_sha256':validation_hashes,
            'source_q_y_unchanged':True,'same_rounded_w_for_channel7_and_pool':True,
            'cases':cases,'all_four_default_full_size_models_passed':len(cases)==4,
            'temporary_weights_never_reuse':str(folder),
            'CUDA_BF16_and_production_throughput_remain_pending':True,
            'elapsed_seconds':time.monotonic()-start}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    output.with_suffix('.md').write_text('# R9 실제 자료 연결 fixture\n\n'
       '**PASS — 기본 전체 크기 4개 모델, 각각 CPU Adam 1step만 수행한 폐기용 연결검증이다.** '
       'Fold0 TRAIN의 완전관측 2개·결측 포함 2개와 selection 고정 첫8개를 사용했다. '
       '실제 본학습이나 성능 비교가 아니며, report·Test·GPU·SSH·WAV 접근은 없다.\n\n'
       '채널5/7·동일 rounded energy weights·변경하지 않은 raw q/y·물리 pooling·finite loss/gradient/parameter·'
       'optimizer 1step·엄격 checkpoint reconstruction·bitwise reload·원본 입력/코드/원천 SHA 보존을 검증했다. '
       '폐기용 weight는 /private/tmp에만 저장했고 production 재사용을 금지한다. CUDA BF16 및 실제100epoch 속도는 이 검증에 포함되지 않는다.\n\n'
       f'[숫자·모델 크기·입력/체크포인트 SHA]({output.name})\n')
    print(json.dumps({'status':'passed','report':str(output),'report_sha256':sha(output),
                      'models':[{'kind':c['kind'],'parameters':c['total_parameters']} for c in cases],
                      'elapsed_seconds':report['elapsed_seconds']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/r9_full_training_v1.json')
    p.add_argument('--bundle',default='data/synthetic/group_cv_minute_full_r9_v1')
    p.add_argument('--output',default='reports/R9_REAL_DATA_CONNECTION_VERIFICATION.json')
    args=p.parse_args();run(args.config,args.bundle,args.output)
