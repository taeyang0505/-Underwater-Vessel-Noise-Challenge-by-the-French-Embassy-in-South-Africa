"""Explicit future CUDA1epoch measurement; no checkpoint is retained/reused."""
from pathlib import Path
import json,sys,time,gc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def benchmark(config,bundle,runtime_policy,output):
    import torch
    from torch.utils.data import DataLoader
    from pipeline.runner_r13 import ProductionData,NEURAL,CODE_NAMES,sha,_validate_config
    from pipeline.models_r13 import make_r13_model
    from pipeline.training_r13 import physical_multitask_loss,_seed_worker
    out=Path(output)
    if out.exists():raise FileExistsError('Fresh CUDA benchmark receipt required')
    if not torch.cuda.is_available():raise RuntimeError('Actual CUDA hardware required; no CPU substitution')
    c=_validate_config(json.loads(Path(config).read_text()));data=ProductionData(config,bundle,runtime_policy_path=runtime_policy)
    # This is an explicit benchmark, even if production recipe approval isfalse.
    rows=data.train(0);rows.set_epoch(1);torch.set_num_threads(c['training']['threads']);results=[]
    prop=torch.cuda.get_device_properties(0);hardware={'device_name':prop.name,'device_uuid':str(getattr(prop,'uuid','unavailable')),
        'total_memory_bytes':prop.total_memory,'torch':torch.__version__,'torch_cuda':torch.version.cuda}
    try:
        for kind in NEURAL:
            torch.manual_seed(c['seed']);model=make_r13_model(kind).to('cuda');model.train()
            opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0003)
            gen=torch.Generator().manual_seed(c['seed']+1000003)
            loader=DataLoader(rows,batch_size=c['training']['batch_size'],shuffle=True,num_workers=c['training']['num_workers'],
                worker_init_fn=_seed_worker,generator=gen,pin_memory=True,persistent_workers=False,
                **({'multiprocessing_context':'spawn','prefetch_factor':2} if c['training']['num_workers'] else {}))
            torch.cuda.reset_peak_memory_stats();begin=time.perf_counter();seen=0;steps=0
            for batch in loader:
                x=batch['x'].to('cuda',non_blocking=True);w=batch['w'].to('cuda',non_blocking=True)
                y,z,q=(batch[k].to(device='cuda',dtype=torch.float32,non_blocking=True) for k in ('share','presence','cell_share'))
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):p,score,mask=model(x,w)
                losses,_=physical_multitask_loss((p,score,mask),(y,z,q),w)
                weight=batch['loss_weight'].to('cuda') if c['training']['site_balanced_loss'] else torch.ones_like(losses)
                loss=(losses*weight).mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
                seen+=len(x);steps+=1
            torch.cuda.synchronize()
            if seen!=len(rows):raise ValueError('Incomplete actual TRAIN epoch')
            results.append({'kind':kind,'rows':seen,'actual_Adam_steps':steps,'seconds':time.perf_counter()-begin,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                'parameters':sum(p.numel() for p in model.parameters()),'batch_size':c['training']['batch_size']})
            del opt,model;gc.collect();torch.cuda.empty_cache()
        data.assert_unchanged()
        value={'format':'r13_actual_CUDA1epoch_benchmark_v1','status':'passed','benchmark_only':True,
            'config_sha256':sha(config),'runtime_policy_sha256':sha(runtime_policy),'data_hash':data.bundle_hash,
            'hardware':hardware,'model_results':results,'all_four_models_one_complete_epoch':True,
            'model_weights_or_optimizer_retained':False,'production100training_started':False,
            'Test_or_selection_gate_report_predictions':False,'code_sha256':{n:sha(ROOT/n) for n in CODE_NAMES}}
        out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(value,indent=2)+'\n');return value
    except BaseException as e:
        out.parent.mkdir(parents=True,exist_ok=True);out.with_suffix('.FAILED.json').write_text(json.dumps({'error':repr(e),'partial_reuse':False,'benchmark_only':True,'hardware':hardware},indent=2)+'\n');raise

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--bundle',required=True);p.add_argument('--runtime-policy',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(benchmark(a.config,a.bundle,a.runtime_policy,a.output)))
