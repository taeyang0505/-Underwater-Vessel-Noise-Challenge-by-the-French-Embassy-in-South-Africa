"""Feature-only R15 audit of the exact frozen Fold0 R14 frontend.

No backbone, checkpoint, labels, fitting, calibration or inference is used.
The official NumPy transform is checked against the official Torch CPU FP64
frontend. Stored predictions are joined only for descriptive correlations.
"""
from pathlib import Path
import argparse, csv, hashlib, json, time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT='reports/r15_static_channel_audit_r1'
TEST_SHA='3ef819192c05c4308b66564e6307e6f4f26fe806afe96ec93e2d15f25a3474d4'
BASIS_SHA='8cc263b3a43f0715daf755bed142b2e45b4373030a1e89fa7c499d77283d2d8d'
SEMANTIC=('PCA_signed_residual_RMS_db','PCA_positive_residual_RMS_db',
          'mid_curvature_RMS_db','MF_minus_LF_mean_db','MF_minus_LF_p90_db',
          'LF_anchor_residual_signed_mean_db','LF_anchor_residual_RMS_db',
          'LF_anchor_positive_fraction_mean','LF_anchor_positive_fraction_p95')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p,d): Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def corr(a,b,rank=False):
    a=np.asarray(a,float);b=np.asarray(b,float);ok=np.isfinite(a)&np.isfinite(b);a=a[ok];b=b[ok]
    if rank:a=pd.Series(a).rank(method='average').to_numpy();b=pd.Series(b).rank(method='average').to_numpy()
    if len(a)<3 or a.std()==0 or b.std()==0:return None,int(len(a))
    return float(np.corrcoef(a,b)[0,1]),int(len(a))
def summarize(v,mask):
    v=np.where(mask,v,np.nan)
    return {'mean':np.nanmean(v,(1,2)), 'RMS':np.sqrt(np.nanmean(v*v,(1,2))),
            'p95':np.nanquantile(v,.95,axis=(1,2))}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default=OUT);a=p.parse_args()
    out=(ROOT/a.output).resolve()
    if not out.is_relative_to(ROOT/'reports') or out.exists():raise FileExistsError('Fresh reports directory only')
    config=read(ROOT/'configs/r14_production_mac_v1.json');bundle=ROOT/config['bundle_path'];manifest=read(bundle/'manifest.json');entry=manifest['folds']['0'];basispath=bundle/entry['basis_path']
    if sha(basispath)!=BASIS_SHA or entry['basis_sha256']!=BASIS_SHA:raise ValueError('Exact Fold0 TRAIN PCA file required')
    paths=['ops/r15_static_channel_audit.py','configs/r14_production_mac_v1.json',
           str((bundle/'manifest.json').relative_to(ROOT)),str(basispath.relative_to(ROOT)),
           'pipeline/features_r14_production.py','pipeline/features_r14_torch.py','pipeline/features_r14_anchor.py',
           'pipeline/features_r12.py','pipeline/relative_grid_r6.py','pipeline/relative_grid_r6_v2.py',
           'data/raw/Test.csv','reports/r15_member_audit_r1/aligned_saved_predictions_1500.csv',
           'reports/r15_descriptive_audit_r1/RESULTS.json','reports/r15_descriptive_audit_r1/joined1500.csv']
    pins={n:sha(ROOT/n) for n in paths}
    if pins['data/raw/Test.csv']!=TEST_SHA:raise ValueError('Original query bytes differ')
    out.mkdir(parents=True)
    plan={'format':'r15_exact_static_channel_plan_v1','created_unix':time.time(),
          'source_file_sha256':pins,'PCA_basis':str(basispath.relative_to(ROOT)),
          'basis_file_sha256':BASIS_SHA,'basis_training_binding':entry['basis_training_binding'],
          'fit_or_backbone_or_checkpoint_load_or_model_prediction':False,
          'formula_sources':{'base11':'pipeline.features_r12.transform_channels / features_r14_torch.TorchFeatureFrontend',
                             'anchor3':'pipeline.features_r14_anchor.anchor_maps / features_r14_production.ProductionFrontend'},
          'summary_formula':{'ch8_signedPCA_RMS':'10*sqrt(mean_observed_safe0..22(channel8^2)); units dB',
                             'ch9_positivePCA_RMS':'10*sqrt(mean_observed_safe0..22(channel9^2)); units dB',
                             'ch10_curvature_RMS':'10*sqrt(mean_observed_band10..21(channel10^2)); units dB',
                             'ch11_MF_minus_LF':'20*mean_observed_safe0..22(channel11); MF251.2..3162.3,LF31.6..158.5Hz',
                             'ch12_anchor_residual':'20*mean or RMS observed_safe0..22(channel12); units dB',
                             'ch13_positive_fraction':'mean/p95 observed_safe0..22(channel13); bounded derived feature, NOT true ship fraction'},
          'summary_regions':'Channels0..7 full25 observed cells; PCA/anchor safe0..22; curvature10..21; mask also fullgrid fraction',
          'group_law':'Reuse previous declared low-temporal q25(bumpHigh) AND q25(stdBroadband); separately intersect with top-quartile actual PCApositiveRMS, MF-LFmean, anchorRMS. No threshold optimization.',
          'classification_semantics':'Input-only descriptive groups. No hidden Test labels or actual site GT.',
          'declared_semantic_features':list(SEMANTIC)}
    write(out/'plan.json',plan);started=time.time()
    from pipeline.features_r12 import load_spectral_basis
    from pipeline.features_r14_anchor import channels
    basis=load_spectral_basis(basispath,expected_sha256=BASIS_SHA,expected_training_binding=entry['basis_training_binding'])
    if basis['fit_split']!='train' or basis['source_kind']!='released_sa_background':raise ValueError('TRAIN basis provenance differs')
    # Raw parser representation matches the frozen query parser: float32 input.
    tags=('31.6','39.8','50.1','63.1','79.4','100','125.9','158.5','199.5','251.2','316.2','398.1','501.2','631','794.3','1000','1258.9','1584.9','1995.3','2511.9','3162.3','3981.1','5011.9','6309.6','7943.3')
    with (ROOT/'data/raw/Test.csv').open(newline='',encoding='utf-8-sig') as stream:
        rd=csv.reader(stream)
        if next(rd)!=['ID',*[f'm{m:02d}_TOL_{hz}' for m in range(1,16) for hz in tags]]:raise ValueError('Canonical public header differs')
        raw=list(rd)
    ids=[r[0] for r in raw];db=np.asarray([[float(v) if v.strip() else np.nan for v in r[1:]] for r in raw],np.float32).reshape(-1,15,25)
    joined=pd.read_csv(ROOT/'reports/r15_descriptive_audit_r1/joined1500.csv',dtype={'ID':str})
    wide=pd.read_csv(ROOT/'reports/r15_member_audit_r1/aligned_saved_predictions_1500.csv',dtype={'ID':str})
    if ids!=joined.ID.tolist() or ids!=wide.ID.tolist():raise ValueError('Exact stored ID order required')
    import torch
    from pipeline.features_r14_production import ProductionFrontend,FEATURE_NAMES
    torch.set_num_threads(1);frontend=ProductionFrontend(basis).cpu().eval()
    maxx=0.;maxw=0.;pieces=[];batch=128
    for start in range(0,len(db),batch):
        b=db[start:start+batch];x,w=channels(b,basis)
        with torch.no_grad():tx,tw=frontend(torch.from_numpy(b))
        xd=float(np.max(abs(x-tx.numpy())));wd=float(np.max(abs(w-tw.numpy())));maxx=max(maxx,xd);maxw=max(maxw,wd)
        if xd>3e-6 or wd>3e-7:raise ValueError('Official NumPy/Torch feature-only parity failed')
        obs=np.isfinite(b);safe=obs.copy();safe[:,:,23:]=False;curv=np.zeros_like(obs);curv[:,:,10:22]=obs[:,:,10:22]
        summaries={}
        for ch in range(14):
            mask=curv if ch==10 else safe if ch>=8 else obs
            for stat,val in summarize(x[:,ch].astype(float),mask).items():summaries[f'ch{ch:02d}_{stat}']=val
        summaries['ch05_fullgrid_mean']=x[:,5].mean((1,2))
        summaries.update({'PCA_signed_residual_RMS_db':10*summaries['ch08_RMS'],
                          'PCA_positive_residual_RMS_db':10*summaries['ch09_RMS'],
                          'mid_curvature_RMS_db':10*summaries['ch10_RMS'],
                          'MF_minus_LF_mean_db':20*summaries['ch11_mean'],
                          'MF_minus_LF_p90_db':20*np.nanquantile(np.where(safe,x[:,11],np.nan),.9,axis=(1,2)),
                          'LF_anchor_residual_signed_mean_db':20*summaries['ch12_mean'],
                          'LF_anchor_residual_RMS_db':20*summaries['ch12_RMS'],
                          'LF_anchor_positive_fraction_mean':summaries['ch13_mean'],
                          'LF_anchor_positive_fraction_p95':summaries['ch13_p95']})
        pieces.append(pd.DataFrame(summaries));del tx,tw,x,w
    statics=pd.concat(pieces,ignore_index=True);statics.insert(0,'ID',ids)
    if not np.isfinite(statics.drop(columns='ID')).all().all():raise ValueError('Nonfinite descriptive feature summaries')
    # Additional synthetic missing/partial-mask boundary cases compare exact formulas.
    fixture=np.linspace(60,120,4*15*25).reshape(4,15,25).astype(np.float32)
    fixture[0,0]=np.nan;fixture[1,14]=np.nan;fixture[2,4,24]=np.nan;fixture[3,6,3]=np.nan
    fx,fw=channels(fixture,basis)
    with torch.no_grad():ftx,ftw=frontend(torch.from_numpy(fixture))
    fixerr=float(np.max(abs(fx-ftx.numpy())));fixw=float(np.max(abs(fw-ftw.numpy())))
    if fixerr>3e-6 or fixw>3e-7:raise ValueError('Partial-mask feature formula parity failed')
    alljoined=joined.merge(statics,on='ID',validate='one_to_one',sort=False)
    definitions=pd.DataFrame({'index':range(14),'official_channel_name':FEATURE_NAMES})
    statics.to_csv(out/'static_channels1500.csv',index=False);definitions.to_csv(out/'channel_definitions.csv',index=False)
    predcols=list(wide.columns[1:]);numeric=list(statics.columns[1:]);relations=[];groups=[];site=[]
    for feature in numeric:
        for series in predcols:
            pe,n=corr(alljoined[feature],alljoined[series]);sp,_=corr(alljoined[feature],alljoined[series],True)
            relations.append({'feature':feature,'series':series,'n':n,'Pearson':pe,'Spearman':sp})
    previous=read(ROOT/'reports/r15_descriptive_audit_r1/RESULTS.json');th=previous['input_only_group_thresholds']
    low=(alljoined.bump_high_db<=th['bump_high_q25_db'])&(alljoined.std_broadband_db<=th['std_broadband_q25_db'])
    features={'PCA_positive_residual_RMS_db':None,'MF_minus_LF_mean_db':None,'LF_anchor_residual_RMS_db':None}
    masks={'all':np.ones(len(db),bool),'low_temporal_predeclared':low};thresholds={}
    for name in features:
        cut=float(alljoined[name].quantile(.75));high=alljoined[name]>=cut;thresholds[name]=cut
        masks['high_'+name]=high;masks['low_temporal_AND_high_'+name]=low&high
    for group,mask in masks.items():
        s=alljoined.loc[mask];alljoined['group_'+group]=np.asarray(mask,bool)
        for series in predcols:
            a=s[series].to_numpy(float);groups.append({'group':group,'series':series,'n':len(s),'mean':float(a.mean()) if len(a) else None,'median':float(np.median(a)) if len(a) else None,'rms':float(np.sqrt(np.mean(a*a))) if len(a) else None,'delta_R14minusR9_share_mean':float(s.delta_R14minusR9_share.mean()) if len(s) else None})
    for name,s in alljoined.groupby('probe_site'):
        for feature in SEMANTIC:
            a=s[feature].to_numpy(float);site.append({'probe_site_estimate_not_GT':name,'feature':feature,'n':len(s),'mean':float(a.mean()),'median':float(np.median(a)),'p90':float(np.quantile(a,.9))})
    alljoined.to_csv(out/'joined1500.csv',index=False)
    pd.DataFrame(relations).to_csv(out/'feature_prediction_correlations.csv',index=False)
    pd.DataFrame(groups).to_csv(out/'lowtemporal_x_actualstatic_groups.csv',index=False)
    pd.DataFrame(site).to_csv(out/'site_static_stats.csv',index=False)
    for p,h in pins.items():
        if sha(ROOT/p)!=h:raise ValueError('Frozen input/source changed:'+p)
    result={'status':'passed_feature_only_descriptive_audit','rows':len(db),'channels':14,'prediction_series':len(predcols),
            'elapsed_seconds':time.time()-started,'source_file_sha256':pins,'plan_sha256':sha(out/'plan.json'),
            'basis_file_sha256':BASIS_SHA,'basis_TRAIN_binding':basis['training_binding'],
            'official_numpy_vs_torch_all1500_max_channel_error':maxx,'official_numpy_vs_torch_all1500_max_weight_error':maxw,
            'synthetic_partial_mask_max_channel_error':fixerr,'synthetic_partial_mask_max_weight_error':fixw,
            'all_inputs_unchanged':True,'group_input_only_q75_thresholds':thresholds,'group_counts':{k:int(np.asarray(v).sum()) for k,v in masks.items()},
            'actual_backbone_checkpoint_model_load_or_fit_or_prediction_calls':0,
            'feature_frontend_only_executed':True,'selection_or_report_or_Test_label_arrays_read':False,
            'limits':['The same frozen TRAIN-only PCA was reused; no query population fit or basis change occurred.',
                      'These are deterministic re-expressions of existing acoustic inputs, not new measurements, real background estimates or known vessel energy.',
                      'MF/LF/anchor/PCA are not guaranteed background-pure or recorder-invariant; their correlation with stored predictions does not establish causal feature reliance or error.',
                      'Input-only quartile groups are descriptive and not selected to optimize any model/label metric.',
                      'probe_site remains a historical kNN estimate, not actual Test site GT. No hidden Test labels, score matching, weights, calibration or submission generation.'],
            'output_sha256':{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
    write(out/'RESULTS.json',result)
    print(json.dumps({'status':result['status'],'output':str(out),'max_channel_error':maxx,'max_weight_error':maxw,'RESULTS_sha256':sha(out/'RESULTS.json')}))

if __name__=='__main__':main()
