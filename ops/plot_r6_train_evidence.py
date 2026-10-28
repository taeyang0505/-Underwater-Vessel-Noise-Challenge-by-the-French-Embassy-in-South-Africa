"""Plot predeclared first-in-order TRAIN examples; no model or heldout I/O."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_DATA=ROOT/'data/synthetic/r6_cpu_guardrails_v3_schemafix'
FREQUENCIES=np.array([31.6,39.8,50.1,63.1,79.4,100,125.9,158.5,199.5,251.2,
                      316.2,398.1,501.2,631,794.3,1000,1258.9,1584.9,1995.3,
                      2511.9,3162.3,3981.1,5011.9,6309.6,7943.3])
REGIONS={'LF <=200 Hz':np.arange(9),'MF 316--2512 Hz':np.arange(10,20),
         'HF 3162--7943 Hz':np.arange(20,25)}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):h.update(chunk)
    return h.hexdigest()


def regional_trajectories(db):
    out={}
    for name,bands in REGIONS.items():
        a=db[:,bands].astype(float);valid=np.isfinite(a).all(1)
        values=np.full(15,np.nan)
        if valid.any():
            v=a[valid];peak=v.max(1)
            levels=peak+10*np.log10((10**((v-peak[:,None])/10)).sum(1))
            values[valid]=levels-np.median(levels)
        out[name]={'relative_db':values,'bump_db':float(np.nanmax(values)) if valid.any() else None,
                   'valid_minutes':int(valid.sum())}
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/r6_guardrails_training_examples.png')
    args=parser.parse_args();folder=args.data_dir.resolve();output=args.output.resolve()
    selection=output.with_suffix('.selection.json')
    if output.exists() or selection.exists():raise FileExistsError('Fresh figure/selection output required')
    # Fail closed against accidentally opening an inference or heldout array.
    allowed={str((folder/n).resolve()) for n in ['train_r6.npz','train_r4.npz']}
    def audit(event,values):
        if event!='open' or not values or not isinstance(values[0],(str,bytes)):return
        p=Path(values[0]).resolve()
        if p.suffix in ['.npz','.npy','.parquet','.wav','.flac'] and ('data.nosync' in p.parts or 'data' in p.parts):
            if str(p) not in allowed:raise PermissionError('TRAIN-only visualization forbids '+str(p))
    sys.addaudithook(audit)
    manifest_path=folder/'manifest.json';manifest=json.loads(manifest_path.read_text())
    file_hashes={name:sha(folder/name) for name in ['train_r6.npz','train_r4.npz']}
    assert all(manifest['arrays'][n]==h for n,h in file_hashes.items())
    rows=[]
    for name in ['train_r6.npz','train_r4.npz']:
        with np.load(folder/name,allow_pickle=False) as z:
            if name=='train_r6.npz':
                choices=[('hard_negative',np.flatnonzero(z['hard_negative'])),
                         ('R6_positive',np.flatnonzero((z['presence']>0)&~z['no_source']))]
            else:choices=[('R4_positive',np.flatnonzero((z['presence']>0)&~z['no_source']))]
            for label,candidates in choices:
                if not len(candidates):raise ValueError('Predeclared case absent: '+label)
                index=int(candidates[0]);keys=['db','unrounded_db','cell_share','share','presence','site',
                                             'base_window_index','reference_group','family','no_source','hard_negative']
                row={k:z[k][index].copy() for k in keys};row.update(case=label,file=name,stored_row_index=index)
                if label=='hard_negative':
                    assert bool(row['no_source']) and row['presence']==0 and row['share']==0
                    assert (row['cell_share']==0).all()
                rows.append(row)
    from pipeline.relative_grid_r6_v2 import relative_grid_v2
    for row in rows:
        x,w=relative_grid_v2(row['db'][None]);row['sqrt_w']=np.sqrt(w[0]);row['q05_relative']=x[0,6]
        row['regional']=regional_trajectories(row['db'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
                         'axes.labelsize':9,'figure.facecolor':'white','axes.facecolor':'white',
                         'savefig.facecolor':'white','svg.fonttype':'none'})
    fig,axes=plt.subplots(3,5,figsize=(21.6,11.7),gridspec_kw={'width_ratios':[1,1,1,1,1.12]})
    fig.subplots_adjust(left=.055,right=.97,top=.87,bottom=.12,wspace=.32,hspace=.75)
    all_db=np.concatenate([r['db'][np.isfinite(r['db'])] for r in rows])
    db_limits=(float(np.floor(all_db.min()/5)*5),float(np.ceil(all_db.max()/5)*5))
    sqrt_max=max(float(r['sqrt_w'].max()) for r in rows)
    q05_max=max(float(np.abs(r['q05_relative']).max()) for r in rows)
    q05_max=max(q05_max,1e-4)
    cmaps=[]
    for name in ['viridis','Blues','magma','RdBu_r']:
        c=plt.get_cmap(name).copy();c.set_bad('#bfc3c8');cmaps.append(c)
    row_titles={'hard_negative':'(a) TRAIN hard negative: known artificial added energy A = 0',
                'R6_positive':'(b) TRAIN R6 positive: SNR-based, unwarped reference addition',
                'R4_positive':'(c) TRAIN R4 positive: time-warped reference addition'}
    column_titles=['Rounded observation (dB)','Known synthetic cell fraction q','sqrt(rounded energy weight)',
                   '(dB - band q05) / 15','Mixture band-power trajectories']
    handles=[]
    for i,row in enumerate(rows):
        observed=np.isfinite(row['db']);data=[row['db'],row['cell_share'],row['sqrt_w'],row['q05_relative']]
        for j,a in enumerate(data):
            ax=axes[i,j];a=np.ma.array(a.T,mask=~observed.T)
            opts=[{'vmin':db_limits[0],'vmax':db_limits[1]},{'vmin':0,'vmax':1},
                  {'vmin':0,'vmax':sqrt_max},{'norm':TwoSlopeNorm(vmin=-q05_max,vcenter=0,vmax=q05_max)}][j]
            image=ax.imshow(a,origin='lower',aspect='auto',interpolation='nearest',extent=(0,15,0,25),cmap=cmaps[j],**opts)
            ax.set_xticks([.5,5.5,10.5,14.5],['0','5','10','14']);ax.set_xlabel('Minute bin (0--14)')
            ax.set_yticks(np.array([0,5,10,15,20,24])+.5,['31.6','100','316','1000','3162','7943'])
            if j==0:ax.set_ylabel('Band centre (Hz)')
            ax.set_title(column_titles[j],pad=7)
            if i==0:handles.append(image)
        ax=axes[i,4]
        colors=['#24678c','#c47b24','#7861a6']
        for (name,values),color in zip(row['regional'].items(),colors):
            ax.plot(np.arange(15)+.5,values['relative_db'],drawstyle='steps-mid',lw=1.8,color=color,
                    label=f'{name}\nbump {values["bump_db"]:.2f} dB' if values['bump_db'] is not None else name+' undefined')
        ax.axhline(0,color='.65',lw=.7,ls='--');ax.set_xlim(0,15)
        ax.set_xticks([.5,5.5,10.5,14.5],['0','5','10','14']);ax.set_xlabel('Minute bin (0--14)')
        ax.set_ylabel('dB relative to own temporal median');ax.set_title(column_titles[4],pad=7)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
        ax.legend(loc='best',fontsize=7.5,frameon=False)
        pos=axes[i,0].get_position()
        subtitle=f'{row["site"]} | stored row {row["stored_row_index"]} | source window {int(row["base_window_index"])} | share = {float(row["share"]):.4f}'
        fig.text(pos.x0,pos.y1+.039,row_titles[row['case']],fontsize=13,fontweight='bold',va='bottom')
        fig.text(pos.x0,pos.y1+.017,subtitle,fontsize=9.5,color='#444',va='bottom')
    for j,image in enumerate(handles):
        bbox=axes[2,j].get_position();cax=fig.add_axes([bbox.x0,.066,bbox.width,.010])
        fig.colorbar(image,cax=cax,orientation='horizontal').ax.tick_params(labelsize=8)
    fig.suptitle('TRAIN-only evidence: natural LF transient versus known reference additions',fontsize=19,fontweight='bold',y=.977)
    fig.text(.055,.938,'Fixed selection: first stored hard-negative row, first stored R6 positive, first stored R4 positive. No prediction/error-based selection.',fontsize=11,color='#444')
    fig.text(.055,.018,'Gray = unobserved; heatmap cells are not interpolated. q is the known artificial added-energy fraction, not all natural vessel energy.\n'
             'q05 is an observed-window statistic, not verified background truth. A=0 means no artificial addition; natural ship absence is not certified. No model performance is shown.',
             fontsize=10,color='#444',linespacing=1.6)
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=200);svg=output.with_suffix('.svg');fig.savefig(svg);plt.close(fig)
    source_manifest=ROOT/manifest['source_version']/'manifest.json'
    if sha(source_manifest)!=manifest['source_manifest_sha256']:raise ValueError('Source metadata changed')
    source=json.loads(source_manifest.read_text())
    selected=[]
    for row in rows:
        selected.append({'case':row['case'],'TRAIN_file':row['file'],'stored_row_index':row['stored_row_index'],
                         'base_window_index':int(row['base_window_index']),'site':str(row['site']),
                         'reference_group':str(row['reference_group']),'family':str(row['family']),
                         'share':float(row['share']),'presence':int(row['presence']),'no_source':bool(row['no_source']),
                         'hard_negative':bool(row['hard_negative']),'max_known_q':float(row['cell_share'].max()),
                         'observed_cells':int(np.isfinite(row['db']).sum()),
                         'regional_mixture_bumps_db':{k:v['bump_db'] for k,v in row['regional'].items()}})
    assert file_hashes=={n:sha(folder/n) for n in file_hashes}
    receipt={'format':'r6_train_first_stored_evidence_figure_v1','status':'completed','scope':'TRAIN-only descriptive evidence',
             'selection_rule':'first stored hard_negative; first presence>0 and not no_source R6; same first-positive rule R4',
             'selection_not_based_on_predictions_errors_or_appearance':True,'selected':selected,
             'dataset_version':str(folder.relative_to(ROOT)),'dataset_manifest_sha256':sha(manifest_path),
             'TRAIN_array_sha256':file_hashes,'source_version':manifest['source_version'],
             'source_manifest_sha256':manifest['source_manifest_sha256'],'source_hash':source.get('source_hash'),
             'hard_negative_manifest_sha256':manifest['hard_negative_manifest_sha256'],
             'script_sha256':sha(__file__),'feature_code_sha256':{n:sha(ROOT/n) for n in ['pipeline/relative_grid_r6.py','pipeline/relative_grid_r6_v2.py']},
             'figures_sha256':{output.name:sha(output),svg.name:sha(svg)},'source_arrays_unchanged':True,
             'heatmap_interpolation':'nearest;15minute bins x25physical band indices','missing_color':'#bfc3c8',
             'region_bump_formula':'max(observed complete-region sum-power dB) minus median of same temporal levels',
             'model_predictions_or_performance_used':False,'heldout_INNER_OUTER_Test_or_GPU_SSH_used':False,
             'q05_is_not_certified_background':True,'A0_does_not_certify_natural_ship_absence':True}
    selection.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'figure':str(output),'selection':str(selection),'rows':selected,'png_sha256':sha(output)},ensure_ascii=False))


if __name__=='__main__':main()
