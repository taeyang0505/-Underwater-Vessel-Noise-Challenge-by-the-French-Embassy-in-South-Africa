"""Finite analytical probes of the frozen R9 objective, no model training.

Test whether perfect known masks minimize the combined loss when score uses
temporal max-min contrast. A lower loss for an incorrect q is a property of
this objective on these inputs, not proof of the historical training cause.
"""
import json,hashlib,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS','2')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    out=ROOT/'reports/r11_diagnostics/derived_score_loss_counterexample_r1';out.mkdir(parents=True,exist_ok=False)
    pin={n:sha(ROOT/n) for n in ['pipeline/training_r9.py','pipeline/relative_models_r6_v2.py','pipeline/models_r9.py']}
    plan={'probe':'finite objective counterexample, not a trained model','targets':[.05,.2,.5],
          'weights':['uniform375','LF_dominant_static_spectrum'],
          'perfect_mask':'all q=true_share; A=y*E,B=(1-y)*E are physically nonnegative',
          'perturbed_mask':'two incorrect cells at t0/f22=0,t1/f22=1',
          'settings':'call frozen production physical_multitask_loss and fixed_presence_score',
          'comparison_without_presence':'algebraically omit .05BCE; candidate diagnostic only, production unchanged',
          'source_SHA':pin,'new_training_Test_models_or_policy_change':False}
    (out/'predeclared_plan.json').write_text(json.dumps(plan,indent=2))
    import torch,numpy as np
    from pipeline.training_r9 import physical_multitask_loss
    from pipeline.relative_models_r6_v2 import fixed_presence_score,SCORE_EPSILON
    torch.set_num_threads(2);records=[]
    for weightkind in plan['weights']:
        if weightkind=='uniform375':w=torch.ones(1,15,25)/375.
        else:
            band=torch.pow(10.,-torch.arange(25,dtype=torch.float32)/5.)
            w=band[None,None,:].repeat(1,15,1);w=w/w.sum()
        for y in plan['targets']:
            truth=torch.full((1,15,25),y);target=(torch.tensor([y]),torch.ones(1),truth)
            rows={}
            for kind in ['perfect_q','incorrect_two_cells']:
                q=truth.clone()
                if kind=='incorrect_two_cells':q[0,0,22]=0.;q[0,1,22]=1.
                score,share,contrast=fixed_presence_score(q,w)
                total,parts=physical_multitask_loss((share,torch.logit(score.clamp(SCORE_EPSILON,1-SCORE_EPSILON)),q),target,w)
                without=parts['share_mse']+.25*parts['weighted_cell_mse']
                rows[kind]={'share':float(share[0]),'score':float(score[0]),'contrast':float(contrast[0]),
                    'total_R9_loss':float(total[0]),'share_MSE':float(parts['share_mse'][0]),
                    'weighted_cell_MSE':float(parts['weighted_cell_mse'][0]),
                    'presence_BCE':float(parts['derived_presence_bce'][0]),
                    'physical_losses_without_presence':float(without[0])}
            if not rows['incorrect_two_cells']['total_R9_loss']<rows['perfect_q']['total_R9_loss']:raise AssertionError('Counterexample did not occur')
            if not rows['incorrect_two_cells']['physical_losses_without_presence']>rows['perfect_q']['physical_losses_without_presence']:raise AssertionError('Physical loss comparison invalid')
            # Independent scalar formula checks the production implementation.
            for k,r in rows.items():
                hand=r['share_MSE']+.25*r['weighted_cell_MSE']-.05*np.log(r['score'])
                if abs(hand-r['total_R9_loss'])>3e-8:raise AssertionError('Loss formula mismatch')
            records.append({'weight_kind':weightkind,'target_share':y,'modified_cell_weight':float(w[0,0,22]),'values':rows})
    for n,s in pin.items():
        if sha(ROOT/n)!=s:raise AssertionError('Production source changed')
    d={'status':'passed','counterexamples':records,'source_SHA_unchanged':True,
       'causal_limit':'This proves a finite objective incentive to corrupt perfect masks, not how often it occurred in the historical R9 training or the sole Public cause.',
       'production_loss_weights_or_outputs_changed':False}
    (out/'RESULTS.json').write_text(json.dumps(d,indent=2))
    lines=['# R11 추가 검산: contrast BCE와 정답 마스크의 목적 충돌','',
       '학습하지 않고 기존 생산 함수 두 개를 그대로 호출했다. E>0,B=(1−y)E,A=yE이면 모든시간의정답q=y는물리적으로가능하다. 두개셀의q를일부러0과1로틀리게만들어비교했다.','',
       '|w|정답y|정확q의전체loss|틀린q의전체loss|정확q score|틀린q score|틀린q share|','|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        a=r['values']['perfect_q'];b=r['values']['incorrect_two_cells'];lines.append(f"|{r['weight_kind']}|{r['target_share']}|{a['total_R9_loss']:.8f}|{b['total_R9_loss']:.8f}|{a['score']:.6f}|{b['score']:.6f}|{b['share']:.8f}|")
    lines+=['', '6개반례모두정답마스크를일부러틀리게만들었을때기존전체loss가낮아졌다. share/cell손실은늘었지만,0.05×BCE가시간최대−최소대비를키운보상을더크게줬기때문이다. presence항을제외한물리적손실만비교하면정확q가더낮았다.',
       '이는현재결합목적함수의구체적인한계다. 모든R9오류가이경로로발생했다고확정하거나BCE제거만으로공식점수가개선된다고보장하지않는다. 실제개선은새데이터의독립소형모델ablation으로검증해야한다. 기존R9loss·모델·제출출력은바꾸지않았다.', '', '[계획](predeclared_plan.json) · [전수치](RESULTS.json)']
    (out/'FINDINGS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'completed':str(out),'finite_counterexamples':len(records),'y020_uniform':records[1]},ensure_ascii=False))
if __name__=='__main__':main()
