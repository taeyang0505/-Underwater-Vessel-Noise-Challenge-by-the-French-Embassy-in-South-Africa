#!/usr/bin/env python3
"""Post-hoc common-gain stress diagnostic, no fitting or query data."""
from pathlib import Path
import gc, sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ops.r18_classifier_audit import OUT,MODELS,FRESH,MODES,extra_features,cached_temporal,matrix,metrics,no_query
from ops.r17_audit_common import load,sha,jsave

def main():
    import lightgbm as lgb
    sys.addaudithook(no_query)
    if not (OUT/'COMPLETED.json').exists():raise RuntimeError('Registered main experiment must finish first')
    protocol={'status':'post_hoc_stress_diagnostic_not_used_for_selection','gains_db':[-20.,0.,20.],
        'input':'Fresh fit only; all observed E and latent A/B would scale together, leaving presence/share unchanged',
        'no_training':True,'Test_read':False,'model_sha256':{
            str(p.relative_to(ROOT)):sha(p)for p in MODELS.rglob('classifier.txt')},
        'code_sha256':sha(__file__)}
    jsave(OUT/'common_gain_probe_protocol.json',protocol)
    result=[]
    for r in ('D0','H0'):
        d=load(FRESH/f'ensemble_fit_fresh_{r}.npz');temporal=cached_temporal('fresh_fit_'+r,d['db'])
        # Reuse stateless time-relative statistics exactly; numerical gain
        # commutation error was independently bounded by the main fixture.
        for gain in (-20.,0.,20.):
            restored,absolute=extra_features(d['db']+gain)
            record={'data':d,'temporal':temporal,'restored':restored,'absolute':absolute}
            for mode in MODES:
                x=matrix(record,mode);pred=[]
                for trained in ('D0','H0'):
                    p=MODELS/trained/mode/'classifier.txt';m=lgb.Booster(model_file=str(p))
                    pred.append(m.predict(x,num_threads=2));del m
                p=.5*(pred[0]+pred[1]);del x
                result.append({'panel':'fresh_fit_'+r,'common_gain_db':gain,'mode':mode,
                    **metrics(d['share']>0,p)})
            gc.collect()
    frame=pd.DataFrame(result);frame.to_csv(OUT/'common_gain_stress.csv',index=False)
    (OUT/'COMMON_GAIN_STRESS.md').write_text('# R18 사후 공통 gain 스트레스 진단\n\n'
        '새 모델을 선택하거나 적합하지 않고 fresh-fit 관측에만 ±20 dB를 더했다. '
        '이 반례에서는 E·B·A가 모두 같은 배율로 변해 정답 y 및 presence가 그대로다. '
        '실제 Test에 ±20 dB 오프셋이 있다는 증거는 아니며, 원시 dB가 센서 보정 조건을 벗어날 때의 민감도만 측정한다. '
        '원시 레벨은 장비 보정과 배경 prior가 유효할 때 유용할 수 있으므로 스트레스 결과만으로 삭제하지 않는다.\n\n'
        +frame[['panel','mode','common_gain_db','auc','brier','score_mean','score_ge090']].to_markdown(index=False,floatfmt='.6f')+'\n')
    jsave(OUT/'COMMON_GAIN_COMPLETED.json',{'status':'completed','new_fits':0,'Test_read':False,
        'protocol_sha256':sha(OUT/'common_gain_probe_protocol.json'),'csv_sha256':sha(OUT/'common_gain_stress.csv')})

if __name__=='__main__':main()
