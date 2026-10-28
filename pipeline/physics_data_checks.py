"""Saved float32 component diagnostics with explicit quantization bounds."""
import numpy as np

def audit_active_snr(db,cell_share,recorded_snr,batch=512):
    """Test true active SNR against float32 rounding intervals, not a loose tol.

    q close to one loses relative precision for the small background E*(1-q).
    Each q/db gets its IEEE float32 half-ULP interval. Ratios formed from those
    intervals must contain the original component SNR; no label is modified.
    """
    recorded=np.asarray(recorded_snr,dtype=float);errors=[];wide=0;count=0
    for start in range(0,len(recorded),batch):
        values=np.asarray(db[start:start+batch],dtype=np.float32)
        q=np.asarray(cell_share[start:start+batch],dtype=np.float32)
        use=np.isfinite(recorded[start:start+batch])
        if not use.any():continue
        values=values[use];q=q[use];true=recorded[start:start+batch][use]
        valid=np.isfinite(values);active=(q>0).any(2)
        mask=valid&active[:,:,None]
        x=values.astype(float);energy=np.where(valid,10**((x-np.nanmax(x,axis=(1,2),keepdims=True))/10),0)
        d=np.where(valid,np.abs(np.spacing(values)).astype(float)/2,0)
        low_e=energy*10**(-d/10);high_e=energy*10**(d/10)
        v=q.astype(float)
        low_q=np.clip((v+np.nextafter(q,np.float32(-np.inf)).astype(float))/2,0,1)
        high_q=np.clip((v+np.nextafter(q,np.float32(np.inf)).astype(float))/2,0,1)
        def mass(v):return np.where(mask,v,0).sum((1,2))
        alo=mass(low_q*low_e);ahi=mass(high_q*high_e)
        blo=mass((1-high_q)*low_e);bhi=mass((1-low_q)*high_e)
        with np.errstate(divide='ignore',invalid='ignore'):
            lower=10*np.log10(alo/bhi);upper=10*np.log10(ahi/blo)
            point=10*np.log10(mass(v*energy)/mass((1-v)*energy))
        assert np.all(true>=lower-1e-6) and np.all(true<=upper+1e-6),'Recorded active SNR outside float32 component rounding interval'
        delta=np.abs(point-true);errors.extend(delta[np.isfinite(delta)].tolist())
        wide+=int((delta>2e-3).sum());count+=len(true)
    return {'positive_examples_checked':count,'all_recorded_values_inside_float32_rounding_intervals':True,
        'point_reconstruction_error_above_0p002db_examples':wide,
        'max_finite_point_reconstruction_error_db':max(errors) if errors else None,
        'method':'IEEE float32 half-ULP q/db intervals; no target or physical component altered'}

def audit_minute_snr(db,cell_share,recorded_snr,ceiling_db=20.,batch=512):
    recorded=np.asarray(recorded_snr,dtype=float);maximum=-np.inf;count=0
    assert recorded.shape==(len(db),15)
    for start in range(0,len(db),batch):
        values=np.asarray(db[start:start+batch],dtype=np.float32);q=np.asarray(cell_share[start:start+batch],dtype=np.float32)
        valid=np.isfinite(values);x=values.astype(float);e=np.where(valid,10**((x-np.nanmax(x,axis=(1,2),keepdims=True))/10),0)
        v=q.astype(float);loq=np.clip((v+np.nextafter(q,np.float32(-np.inf)).astype(float))/2,0,1)
        hiq=np.clip((v+np.nextafter(q,np.float32(np.inf)).astype(float))/2,0,1)
        d=np.where(valid,np.abs(np.spacing(values)).astype(float)/2,0)
        loe=e*10**(-d/10);hie=e*10**(d/10)
        def mass(z):return np.where(valid,z,0).sum(2)
        with np.errstate(divide='ignore',invalid='ignore'):
            lower=10*np.log10(mass(loq*loe)/mass((1-loq)*hie))
            upper=10*np.log10(mass(hiq*hie)/mass((1-hiq)*loe))
            point=10*np.log10(mass(v*e)/mass((1-v)*e))
        actual=recorded[start:start+batch];use=np.isfinite(actual)
        assert np.array_equal(use,np.isfinite(point)),'Minute SNR support differs from saved observed added energy'
        assert np.all(actual[use]<=ceiling_db+1e-8),'Known minute SNR exceeds declared synthetic ceiling'
        assert np.all(actual[use]>=lower[use]-1e-6) and np.all(actual[use]<=upper[use]+1e-6),'Known minute SNR outside float32 rounding interval'
        assert np.all(point[np.isfinite(point)]<=ceiling_db+1e-3),'Saved minute SNR exceeds ceiling beyond float32 quantization'
        if use.any():maximum=max(maximum,float(actual[use].max()));count+=int(use.sum())
    return {'active_minutes_checked':count,'known_component_max_minute_snr_db':maximum if count else None,
        'declared_synthetic_ceiling_db':ceiling_db,'all_minutes_within_ceiling':True,
        'all_known_minute_SNRs_inside_float32_rounding_intervals':True,
        'physical_calibration_or_challenge_distribution_verified':False}
