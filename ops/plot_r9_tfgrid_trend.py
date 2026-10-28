"""Render a read-only snapshot of actual TFGrid epoch logs; no model/data fit."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'reports/R9_TFGRID_FOLD0_TREND_SNAPSHOT.json'
j=json.loads(source.read_text());p=j['points']
e=np.array([r['epoch'] for r in p])
best=min(p,key=lambda r:r['selection_rmse']);last=p[-1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,(ax,lr)=plt.subplots(2,1,figsize=(10,6),gridspec_kw={'height_ratios':[4,1]},sharex=True)
for key,label,color,style in [('train_rmse','TRAIN online RMSE','#356b9a','-'),('selection_rmse','Selection validation RMSE','#ce6a2b','-'),('site_macro_rmse','Site-macro validation RMSE','#54806e','--')]:
    ax.plot(e,[r[key] for r in p],style,color=color,lw=1.8,marker='o',markersize=3,label=label)
ax.axvline(best['epoch'],color='#ce6a2b',ls=':',lw=1)
ax.scatter([best['epoch']],[best['selection_rmse']],s=60,facecolor='white',edgecolor='#ce6a2b',zorder=4)
ax.annotate(f"Best selection: {best['selection_rmse']:.5f}\nepoch {best['epoch']}",xy=(best['epoch'],best['selection_rmse']),xytext=(best['epoch']-5,best['selection_rmse']+.02),arrowprops={'arrowstyle':'->','color':'#555'},fontsize=9)
ax.set_ylabel('RMSE');ax.set_ylim(.045,.113);ax.grid(axis='y',alpha=.18);ax.legend(loc='upper right',frameon=False)
ax.set_title(f"TFGrid–FDY–Conformer · Fold 0 · {last['epoch']}/100 epochs",loc='left',fontsize=15,pad=12)
lr.step(e,[r['learning_rate'] for r in p],where='post',color='#71688e',lw=1.8)
lr.set_yscale('log');lr.set_ylabel('Learning rate');lr.set_xlabel('Epoch');lr.grid(axis='y',alpha=.18)
lr.set_xticks(sorted(set([1,5,10,15,best['epoch'],20,last['epoch']])));lr.set_xlim(.5,last['epoch']+.5)
fig.text(.11,.02,'Fixed selection validation: 1,000 examples · Development metrics; official Test performance is unmeasured',fontsize=9,color='#555')
fig.subplots_adjust(left=.11,right=.98,top=.91,bottom=.12,hspace=.12)
out=ROOT/'reports/r9_live_curves/tfgrid_fold0_rmse.png';out.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(out,dpi=180,facecolor='white');plt.close(fig)
receipt={'source':str(source.relative_to(ROOT)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'plot':str(out.relative_to(ROOT)),'plot_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'epochs':len(p),'last':last,'best_window':best,'model_or_dataset_modified':False}
(out.parent/'tfgrid_fold0_plot_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
