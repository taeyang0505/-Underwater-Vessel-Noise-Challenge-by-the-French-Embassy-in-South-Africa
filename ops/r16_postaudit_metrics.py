"""Pure statistical helpers for the separately versioned R16 post-CSV audit.

No models, Test access, policy mutation, training or query-population fitting.
"""
import csv
import io
import numpy as np
from scipy.special import expit
from scipy.stats import rankdata, pearsonr
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

EPS=1e-12
CENTER=np.asarray([.5,.125,.125,.125,.125],float)

def norm_weights(n, weights=None):
    w=np.ones(n,float) if weights is None else np.asarray(weights,float)
    if w.shape!=(n,) or not np.isfinite(w).all() or (w<0).any() or w.sum()<=0:
        raise ValueError('Invalid weights')
    return w/w.sum()

def balanced_weights(z,weights=None):
    z=np.asarray(z,bool);w=norm_weights(len(z),weights);out=np.zeros(len(z))
    if not z.any() or z.all():return None
    for k in (False,True):out[z==k]=.5*w[z==k]/w[z==k].sum()
    return out

def auc(z,s,w=None):
    z=np.asarray(z,bool)
    return float(roc_auc_score(z,s,sample_weight=w)) if z.any() and not z.all() else None

def correlation(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)<3 or np.std(x)<1e-14 or np.std(y)<1e-14:return None
    return float(pearsonr(x,y).statistic)

def score_aggregates(probabilities,weights=None,fit_probabilities=None):
    p=np.asarray(probabilities,float)
    if p.ndim!=2 or not len(p) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid probability matrix')
    a=norm_weights(p.shape[1],weights);clipped=np.clip(p,EPS,1-EPS)
    logits=np.log(clipped)-np.log1p(-clipped);mean=logits@a
    # Subtracting a shared constant changes calibration, NEVER ideal rank/AUC.
    out={'prob_mean':p@a,'logit_mean':expit(mean),'logit_mean_prior80_to50':expit(mean-np.log(4.)),
         'rank_mean_panel_diagnostic_only':np.column_stack([rankdata(p[:,j],method='average')/len(p) for j in range(p.shape[1])])@a}
    if fit_probabilities is not None:
        fit=np.asarray(fit_probabilities,float)
        if fit.ndim!=2 or fit.shape[1]!=p.shape[1] or not np.isfinite(fit).all():raise ValueError('CDF fit dimension differs')
        cols=[]
        for j in range(p.shape[1]):
            s=np.sort(fit[:,j]);cols.append((np.searchsorted(s,p[:,j],'left')+np.searchsorted(s,p[:,j],'right'))/(2.*len(s)))
        out['rank_mean_fixed_local_fit_CDF']=np.column_stack(cols)@a
    return out

def classification_metrics(z,p,weights=None,conditional_mask=None):
    z=np.asarray(z,bool);p=np.asarray(p,float);w=norm_weights(len(z),weights);bw=balanced_weights(z,w)
    if p.shape!=z.shape or not np.isfinite(p).all():raise ValueError('Invalid aligned scores')
    clipped=np.clip(p,EPS,1-EPS);loss=-(z*np.log(clipped)+(1-z)*np.log1p(-clipped))
    bins=np.minimum((np.clip(p,0,1)*10).astype(int),9);reliability=[];ece=0.
    for k in range(10):
        ix=bins==k
        if not ix.any():continue
        ww=w[ix];mass=float(ww.sum());mean=float(ww@p[ix]/mass);rate=float(ww@z[ix]/mass);ece+=mass*abs(mean-rate)
        reliability.append({'bin':k,'n':int(ix.sum()),'weight_mass':mass,'mean_score':mean,'positive_fraction':rate})
    bece=0.
    if bw is not None:
        for k in range(10):
            ix=bins==k
            if ix.any():bece+=abs(float(bw[ix]@(p[ix]-z[ix])))
    # Exact .17g float64 CSV round-trip, not decimal truncation/calibration.
    stream=io.StringIO();writer=csv.writer(stream);writer.writerows([[format(v,'.17g')] for v in p]);stream.seek(0)
    reread=np.asarray([float(row[0]) for row in csv.reader(stream)])
    if not np.array_equal(p,reread):raise AssertionError('Float64 CSV round-trip differs')
    m=np.ones(len(z),bool) if conditional_mask is None else np.asarray(conditional_mask,bool)
    result={'n':len(z),'n_positive':int(z.sum()),'auc':auc(z,p,w),
       'brier_observed_prior':float(w@((p-z)**2)),'logloss_observed_prior':float(w@loss),
       'brier_class_balanced':None if bw is None else float(bw@((p-z)**2)),
       'logloss_class_balanced':None if bw is None else float(bw@loss),
       'ece10_observed_prior':ece,'ece10_class_balanced':None if bw is None else bece,
       'reliability10':reliability,'score_mean':float(w@p),'p50':float(np.median(p)),
       'ge090':float(w@(p>=.9)),'gt098':float(w@(p>.98)),
       'float64_duplicate_count':len(p)-len(np.unique(p)),
       'float64_tied_value_count':int(np.sum(np.unique(p,return_counts=True)[1]>1)),
       'CSV17g_duplicate_count':len(reread)-len(np.unique(reread)),
       'raw_probability_exact_zero':int((p==0).sum()),'raw_probability_exact_one':int((p==1).sum()),
       'conditional_same_probmean_top_quartile':{'n':int(m.sum()),'positive':int(z[m].sum()),'auc':auc(z[m],p[m],w[m]) if m.any() else None}}
    return result

def regress_metrics(y,p,weights=None):
    y=np.asarray(y,float);p=np.asarray(p,float);e=p-y;w=norm_weights(len(y),weights)
    design=np.column_stack([np.ones(len(y)),p]);coef=np.linalg.lstsq(design*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0]
    return {'n':len(y),'rmse':float(np.sqrt(w@(e*e))),'bias':float(w@e),'mae':float(w@abs(e)),
            'prediction_mean':float(w@p),'target_mean':float(w@y),'prediction_rms':float(np.sqrt(w@(p*p))),
            'calibration_intercept_descriptive_only':float(coef[0]),'calibration_slope_descriptive_only':float(coef[1]),
            'calibration_slope_identifiable':bool(np.std(p)>1e-12)}

def capped_fit(x,y,row_weights=None):
    x=np.asarray(x,float);y=np.asarray(y,float);v=norm_weights(len(y),row_weights);ridge=1e-4
    result=minimize(lambda a:float(v@((x@a-y)**2)+ridge*np.sum((a-CENTER)**2)),CENTER.copy(),
      jac=lambda a:2*x.T@(v*(x@a-y))+2*ridge*(a-CENTER),method='SLSQP',
      bounds=[(.5,1.)]+[(0.,.25)]*4,constraints=[{'type':'eq','fun':lambda a:a.sum()-1.,'jac':lambda a:np.ones(5)}],
      options={'ftol':1e-12,'maxiter':400})
    if not result.success or result.x[0]<.5-1e-8 or np.any(result.x[1:]>.25+1e-8) or abs(result.x.sum()-1)>1e-8:
        raise ValueError('Bootstrap cap optimization failed: '+result.message)
    return result.x

def block_bootstrap(shares,y,groups,*,reps=1000,seed=2026102815,unit='background_group'):
    """Paired D0/H0 group bootstrap; compare risks on OUT-OF-BAG groups.

    SD of bootstrap risk differences is a bootstrap uncertainty estimate;
    dividing by sqrt(B) would incorrectly turn Monte Carlo precision into SE.
    """
    y=np.asarray(y,float);groups=np.asarray(groups).astype(str);unique,inv=np.unique(groups,return_inverse=True)
    if len(unique)<3:raise ValueError('At least three groups required for OOB bootstrap')
    a=np.asarray(shares,float)
    if a.shape!=(2,len(y),5):raise ValueError('D0/H0 aligned five-member shares required')
    rng=np.random.default_rng(seed);weight=[];delta=[];risks=[];omitted=0
    for _ in range(reps):
        counts=np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique));v=counts[inv].astype(float);oob=counts[inv]==0
        fitted=np.stack([capped_fit(x,y,v) for x in a]);weight.append(fitted)
        if not oob.any():omitted+=1;continue
        pred=.5*(a[0,oob]@fitted[0]+a[1,oob]@fitted[1]);center=.5*(a[0,oob]@CENTER+a[1,oob]@CENTER)
        ro=float(np.sqrt(np.mean((pred-y[oob])**2)));rc=float(np.sqrt(np.mean((center-y[oob])**2)))
        delta.append(rc-ro);risks.append([ro,rc,int(oob.sum())])
    weight=np.asarray(weight);delta=np.asarray(delta);std=float(delta.std(ddof=1)) if len(delta)>1 else None
    gain=float(delta.mean()) if len(delta) else None
    return {'groups':len(unique),'rows':len(y),'replications':reps,'valid_OOB_replications':len(delta),'no_OOB_replications':omitted,
            'weight_mean':weight.mean(0).tolist(),'weight_bootstrap_standard_error':weight.std(0,ddof=1).tolist(),
            'weight_percentile95_low':np.quantile(weight,.025,axis=0).tolist(),'weight_percentile95_high':np.quantile(weight,.975,axis=0).tolist(),
            'OOB_optimized_RMSE_mean':float(np.mean(np.asarray(risks)[:,0])) if risks else None,
            'OOB_center_RMSE_mean':float(np.mean(np.asarray(risks)[:,1])) if risks else None,
            'OOB_center_minus_optimized_RMSE_mean':gain,'OOB_difference_bootstrap_SE':std,
            'OOB_difference_percentile95':np.quantile(delta,[.025,.975]).tolist() if len(delta) else None,
            'one_SE_center_recommended':None if gain is None or std is None else bool(gain<=std),
            'one_SE_scope':'audit recommendation only; original frozen policy/CSV NEVER changed',
            'resampled_unit':unit+'; every row belonging to a sampled cluster retained',
            'bootstrap_SD_not_divided_by_sqrt_replications':True,
            'crossed_reference_dependence_not_fully_resolved':True}

def residual_audit(y,old,new,fit_ab,weights=None):
    y=np.asarray(y,float);old=np.asarray(old,float);new=np.asarray(new,float)
    residual=new-(fit_ab[0]+fit_ab[1]*old);truth_error=y-old
    discord=abs(new-old)>1e-8;oe=abs(old-y);ne=abs(new-y)
    ties=np.isclose(ne,oe,atol=1e-12,rtol=0)
    wins=(ne<oe)&~ties
    w=norm_weights(len(y),weights)
    return {'old':regress_metrics(y,old,w),'new':regress_metrics(y,new,w),'pearson_old_new':correlation(old,new),
            'new_residual_vs_old_error_correlation':correlation(residual,truth_error),
            'orthogonal_projection_coefficients_fitted_on_local_fit_only':list(map(float,fit_ab)),
            'discordant_rows':int(discord.sum()),
            'correlations_unweighted_descriptive_only':True,
            'discordant_new_win_rate':float(w[discord]@(wins[discord].astype(float)+.5*ties[discord])/w[discord].sum()) if discord.any() else None}
