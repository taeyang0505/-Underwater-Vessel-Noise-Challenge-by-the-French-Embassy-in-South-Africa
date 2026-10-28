"""Lock R14 anchor choices using validation_fit, then report all17 panels.

All17 are exposed development stress panels, never independent OOF or LB.
No Test I/O. CNN parameter checks and pool equality are required.
"""
from pathlib import Path
import argparse,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--models',default='models/r14_anchor_cpu_r1');ap.add_argument('--output',default='reports/r14_anchor_cpu_r1');a=ap.parse_args()
    from ops.train_r12_lightgbm import environment,guard
    environment(2);sys.addaudithook(guard)
    import numpy as np,torch,lightgbm as lgb
    from torch import nn
    from pipeline.training_r12 import sha,write,predict
    from pipeline.relative_models_r6_v2 import TinyMaskCNNv2
    from pipeline.features_r14_anchor import channels,tree_features
    from pipeline.features_r12 import load_spectral_basis
    from pipeline.dataset_r12 import load_r12_arrays,validate_r12_bundle
    from ops import evaluate_r12_ablation as parent
    torch.set_num_threads(2)
    folder=ROOT/a.models;f=json.loads((folder/'ALL_MODELS_FROZEN.json').read_text())
    if f['status']!='completed' or (folder/'FAILED.json').exists():raise ValueError('Completed fresh fits required')
    for p,h in f['plan']['source_pins'].items():
        if sha(ROOT/p)!=h:raise ValueError('Frozen source changed:'+p)
    out=ROOT/a.output;out.mkdir(parents=True,exist_ok=False);(out/'predictions').mkdir()
    original=ROOT/'data/synthetic/r12_cpu_ablation_v3';m=validate_r12_bundle(original)
    bp=ROOT/'models/r12_shared_basis_v1/basis.npz';basis=load_spectral_basis(bp,expected_training_binding=m['basis_training_binding'])
    class Model(TinyMaskCNNv2):
        def __init__(self):super().__init__(input_channels=14,width=24,blocks=2,dropout=.1,global_context=True)
        def add_global_context(self,h):
            v=torch.cat([h[:,:,:,s:e].mean(3) for s,e in ((10,20),(20,23),(15,23))],1)
            return h+(nn.functional.gelu(self.context_projection(v))*torch.sigmoid(self.context_gate(v))).unsqueeze(3)
        def forward(self,x,w):
            q=self.mask_head(self.add_global_context(self.body(x)))[:,0].sigmoid();q=torch.where(w>0,q,torch.zeros_like(q))
            return (w*q).sum((1,2)),q
    models={}
    for c in (0,2):
        cp=folder/('anchor_c'+str(c))/'best.pt'
        if sha(cp)!=f['models']['anchor_c'+str(c)]['checkpoint_sha256']:raise ValueError('Checkpoint hash changed')
        sp=cp.parent/'summary.json'
        if sha(sp)!=f['models']['anchor_c'+str(c)]['summary_sha256']:raise ValueError('CNN summary hash changed')
        summary=json.loads(sp.read_text());history=[json.loads(v) for v in (cp.parent/'epochs.jsonl').read_text().splitlines()]
        ck=torch.load(cp,map_location='cpu',weights_only=True)
        if (summary['status']!='completed' or summary['epochs_run']!=100 or summary['optimizer']!='Adam' or summary['early_stopping']
            or [v['epoch'] for v in history]!=list(range(1,101)) or ck['epoch']!=summary['best_epoch']
            or ck['epoch']!=int(np.argmin([v['validation_fit_rmse'] for v in history]))+1 or ck['row_c']!=c):raise ValueError('CNN fixed100/best checkpoint differs')
        mo=Model();mo.load_state_dict(ck['state_dict']);mo.eval();models[c]=mo
    trees={c:lgb.Booster(model_file=str(folder/'tree'/('share_c'+str(c)+'.txt'))) for c in (0,2)}
    clf=lgb.Booster(model_file=str(folder/'tree/presence.txt'))
    for c in (0,2):
        if sha(folder/'tree'/('share_c'+str(c)+'.txt'))!=f['models']['tree_c'+str(c)]['model_sha256']:raise ValueError('Tree hash changed')
    if sha(folder/'tree/presence.txt')!=f['models']['classifier']['model_sha256']:raise ValueError('Classifier hash changed')
    def infer(db):
        xx,ww=channels(db,basis);xt=torch.from_numpy(xx);wt=torch.from_numpy(ww);xf=tree_features(db,basis)
        cp=np.clip(clf.predict(xf,num_threads=2),0,1).astype(np.float32);pred={};poolmax=0.;raw={'observed_weights':ww}
        for c in (0,2):
            n=predict(models[c],xt,wt,128);poolmax=max(poolmax,float(np.max(abs((ww*n['cell_mask']).sum((1,2))-n['share']))))
            raw['cnn_c'+str(c)+'_cell_mask']=n['cell_mask']
            t=np.clip(trees[c].predict(xf,num_threads=2),0,1).astype(np.float32)
            pred['CNN_c'+str(c)]=n['share'];pred['Tree_c'+str(c)]=t
            pred['Blend50_c'+str(c)]=(.5*n['share'].astype(float)+.5*t.astype(float)).astype(np.float32)
        if poolmax>3e-6:raise ValueError('Physical pool mismatch')
        return pred,cp,poolmax,raw
    val=load_r12_arrays(original,'L0_H0','validation_fit');pv,cv,_,_=infer(val['db'])
    # Independent classifier does not alter masks or share. All choices use fit1000 only.
    rmse={k:float(np.sqrt(np.mean((v.astype(float)-val['share'])**2))) for k,v in pv.items()}
    choice=min(rmse,key=rmse.get)
    lambdas=(0.,.25,.5,.75,1.);aucs={str(l):parent.auc(val['share'],((1-l)*pv[choice]+l*cv)) for l in lambdas}
    lam=max(lambdas,key=lambda l:aucs[str(l)])
    policy={'regression':choice,'classifier_weight':lam,'fit_rmse':rmse,'fit_auc':aucs,
      'selection_data':'R12 validation_fit1000 only','models_freeze_sha256':sha(folder/'ALL_MODELS_FROZEN.json'),
      'source_sha256':sha(__file__),'Test_read':False,'development_exposed':True}
    write(out/'selected_policy.json',policy);ph=sha(out/'selected_policy.json')
    oldplan,oldf=parent.check_freeze('reports/R12_ALL17_GLOBAL_FREEZE.json')
    oldpol=ROOT/'reports/r12_cpu_ablation_r1/selected_policy.json'
    panels=[parent.load_R12(oldplan,r,oldpol,oldf['freeze_sha256']) for r in ('validation_gate','eval','high_snr_long')]
    dev,r11=parent.load_development();panels+=dev
    for pos in parent.DEV_R11_POS:
        for neg in parent.DEV_R11_NEG:
            x,z=r11[pos],r11[neg];panels.append({'name':x['name']+'_VS_'+neg,'y':np.r_[x['y'],z['y']],
              'db':np.concatenate((x['db'],z['db'])),'sample_id':np.r_[x['sample_id'],z['sample_id']],
              'site':np.r_[x['site'],z['site']],'stage':'exposed_development'})
    if len(panels)!=17:raise ValueError('Expected17')
    rows=[];pmax=0
    for p in panels:
        if sha(out/'selected_policy.json')!=ph:raise ValueError('Changed policy')
        pp,cp,pm,raw=infer(p['db']);pmax=max(pmax,pm);allpred={}
        for k,v in pp.items():allpred[k]={'share':v,'score':v}
        allpred['R14_fit_selected']={'share':pp[choice],'score':((1-lam)*pp[choice]+lam*cp).astype(np.float32),'score_classifier':cp}
        allpred['R14_classifier_only']={'share':pp[choice],'score':cp,'score_classifier':cp}
        # Same frozen references for direct numerical comparisons, no refit or selection.
        with np.load(ROOT/'reports/r12_mask_matched_evaluation_r2/predictions'/(p['name']+'.npz'),allow_pickle=False) as z:
            if not np.array_equal(p['sample_id'].astype(str),z['sample_id'].astype(str)) or not np.array_equal(p['y'],z['target_share']):raise ValueError('Panel alignment differs')
            order=z['model_order'].astype(str).tolist();ix=order.index('Blend50_L1HM1P1')
            allpred['R12_matched_share_only']={'share':z['share'][:,ix],'score':z['score'][:,ix]}
        for k,v in allpred.items():
            metric=parent.metrics(p,v);e=v['share'].astype(float)-p['y'];hi=p['y']>.2
            rows.append({'panel':p['name'],'model':k,**metric,'high_y_bias':float(e[hi].mean()) if hi.any() else None,
              'bias':float(e.mean()),'classifier_ge099':float((cp>=.99).mean())})
        np.savez_compressed(out/'predictions'/(p['name']+'.npz'),sample_id=p['sample_id'],target_share=p['y'],model_order=np.asarray(list(allpred)),
          share=np.column_stack([v['share'] for v in allpred.values()]),score=np.column_stack([v['score'] for v in allpred.values()]),classifier=cp,**raw)
        print(json.dumps({'panel':p['name'],**rows[-3]}),flush=True)
    parent.csv_rows(out/'results.csv',rows)
    results={'status':'completed','panels':17,'rows':len(rows),'policy':policy,'physical_pool_max_error':pmax,
      'Test_read':False,'GPU_training':False,'not_blind_OOF':True,'outputs':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}}
    write(out/'RESULTS.json',results)

if __name__=='__main__':main()
