"""Paper-style editable schematic inspired by the supplied layout, not its content."""
from pathlib import Path
import os,sys,runpy,json,hashlib,xml.etree.ElementTree as ET
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/underwater-mpl')
sys.path[:0]=['/private/tmp/underwater-eda-deps']
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Polygon,Rectangle
from PIL import Image
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
helpers=runpy.run_path('/Users/hongtaeyang/.agents/skills/open-science-figure-composer/kernel.py')
outline={'claim':'Known-energy controlled additions support source-isolated regression and frozen inference.',
         'width_mm':280,'ncol':1,'row_heights_mm':[118,77],
         'panels':[{'letter':'a','role':'schematic','row':0,'col':0,'colspan':1,'chart_family':'flow diagram',
                    'message':'Split sources before constructing mixtures and optimize window MSE.',
                    'ask':'Show data collection, quality checks, exact targets, models and validation.'},
                   {'letter':'b','role':'schematic','row':1,'col':0,'colspan':1,'chart_family':'flow diagram',
                    'message':'All inference transformations and postprocessing are frozen on development data.',
                    'ask':'Show per-window inference, ensemble, permitted postprocessing and submission checks.'}]}
helpers['grid_geom'](outline,dpi=300)
(OUT/'pipeline_outline.json').write_text(json.dumps(outline,indent=2))
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'svg.fonttype':'none',
                     'mathtext.fontset':'dejavuserif','savefig.facecolor':'white'})
SAGE='#e8eee5';OLIVE='#a9b08c';BLUE='#e4f0fa';INK='#111820';ORANGE='#fbe7d2';GREY='#f4f5f2';PURPLE='#eee9f6'
def text(ax,x,y,s,fs=8,bold=False,ha='center',color=INK):
    return ax.text(x,y,s,fontsize=fs,fontweight='bold' if bold else 'normal',ha=ha,va='center',color=color)
def card(ax,x,y,w,h,title,lines=(),color=GREY,dashed=False):
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.25,rounding_size=0.8',facecolor=color,
                     edgecolor=INK,linewidth=.8,linestyle=(0,(4,3)) if dashed else 'solid')
    ax.add_patch(p)
    text(ax,x+w/2,y+h-3.7,title,bold=True)
    if lines:
        for yy,line in zip([y+h-8.0-i*4.9 for i in range(len(lines))],lines):text(ax,x+w/2,yy,line,fs=7)
    return p
def arrow(ax,start,end,label=None,dashed=False,elbow=None):
    kw={'arrowstyle':'-|>','mutation_scale':10,'color':INK,'linewidth':.85,
        'linestyle':(0,(4,3)) if dashed else 'solid','shrinkA':0,'shrinkB':0}
    if elbow is not None:
        points=[start,*elbow,end]
        for a,b in zip(points[:-2],points[1:-1]):ax.plot([a[0],b[0]],[a[1],b[1]],color=INK,lw=.85,ls='--' if dashed else '-')
        ax.add_patch(FancyArrowPatch(points[-2],points[-1],**kw))
    else:ax.add_patch(FancyArrowPatch(start,end,**kw))
    if label:text(ax,(start[0]+end[0])/2,(start[1]+end[1])/2+3,label,fs=7)
def stack(ax,x,y,w,h,title,lines,color):
    for d in [1.2,.6]:ax.add_patch(FancyBboxPatch((x+d,y+d),w,h,boxstyle='round,pad=.25,rounding_size=.6',fc=color,ec=INK,lw=.7))
    card(ax,x,y,w,h,title,lines,color)
def panel(letter):
    w,h=helpers['panel_px'](outline,letter,dpi=300)
    fig=plt.figure(figsize=(w/300,h/300),dpi=300);ax=fig.add_axes([0,0,1,1])
    ax.set(xlim=(0,100),ylim=(0,100));ax.axis('off')
    return fig,ax,w,h

fig,ax,w,h=panel('a')
text(ax,2,96,'Training: measured sources, exact mixture targets',fs=12,bold=True,ha='left')
stack(ax,2,58,15,26,'South Africa',['1,715,779 real rows','10 sites · 25 bands','No vessel truth labels'],SAGE)
stack(ax,2,16,15,31,'External audio',['VTUAD v1.3.2','HearMyShip V1','Licence + hashes','Keep source provenance'],SAGE)
card(ax,22,62,13,21,'Quality + clock',['Sentinels / duration','Align 15-min grids','Keep missing mask'],SAGE)
card(ax,22,19,13,26,'Audio → power',['Native WAV sample rate','2-s Hann periodograms','25 base-10 bands','Linear averaging'],SAGE)
arrow(ax,(18.4,71),(21.7,71))
arrow(ax,(18.4,31),(21.7,31))
card(ax,40,79,42,10,'Partition before mixing',[],SAGE)
text(ax,61,81.5,'SA sensor / time blocks  +  external hull / day / hash groups',fs=7)
arrow(ax,(35.4,72),(40,83),elbow=[(37.5,72),(37.5,83)])
arrow(ax,(35.4,32),(40,83),elbow=[(38.5,32),(38.5,83)])
ax.add_patch(Polygon([(40,42),(40,70),(58,64),(58,48)],closed=True,fc=OLIVE,ec='none'))
text(ax,49,61,'Controlled additions',bold=True)
text(ax,49,55,r'$E_{mix}=E_{bg}+\alpha E_{ref}$',fs=8)
text(ax,49,49,'Placement · gain · smooth tilt',fs=7)
arrow(ax,(49,78.5),(49,70.5))
card(ax,40,18,18,19,'Exact observed-cell target',[],GREY)
text(ax,49,26,r'$y=\frac{\sum M\,\alpha E_{ref}}{\sum M\,E_{mix}}$',fs=11)
arrow(ax,(49,41.5),(49,37.5))
model_x=64;model_w=18
for y0,title,subtitle,color,dash in [(62,'TabM ensemble','375 values + masks + summaries',BLUE,False),
                                    (47,'Energy-mask CNN','Time–frequency + energy pooling',BLUE,False),
                                    (32,'LightGBM','Spectral / temporal statistics',BLUE,False),
                                    (17,'TabICLv2','Foundation-model comparison',PURPLE,True)]:
    card(ax,model_x,y0,model_w,11,title,[],color,dash)
    text(ax,model_x+model_w/2,y0+2.3,subtitle,fs=6.8)
    arrow(ax,(58.5,56),(63.5,y0+5.5),elbow=[(61,56),(61,y0+5.5)])
card(ax,86,34,12,48,'Held-out evaluation',[],ORANGE)
for yy,line in [(73,'Validation RMSE'),(66,'Unseen source groups'),(59,'Gain / mask controls'),(52,'Independent holdout'),(45,'Calibration split'),(39,'Freeze selected models')]:text(ax,92,yy,line,fs=7)
for yy in [67.5,52.5,37.5,22.5]:
    arrow(ax,(82.5,yy),(85.5,58),elbow=[(84,yy),(84,58)])
text(ax,69,7,r'$\mathcal{L}_{main}=\frac{1}{N}\sum_i(\hat y_i-y_i)^2$',fs=10)
text(ax,69,1.7,'Diagnostic presence head and optional cell loss are ablated separately.',fs=7)
card(ax,2,1,33,9,'All real rows remain traceable',[],GREY)
text(ax,18.5,2.2,'Separate weak teacher scores + uncertainty + QC; no invented truth.',fs=6.8)
text(ax,40,92,'Dashed: optional candidates; retain only if held-out MSE improves.',fs=7,ha='left')
card(ax,40,1,18,10,'Training prototypes',[],PURPLE,True)
text(ax,49,2.8,'Grouped spectra; no clip matching',fs=6.8)
arrow(ax,(58.5,6),(63.5,67.5),dashed=True,elbow=[(62,6),(62,67.5)])
fig.savefig(OUT/'panel_a.png',dpi=300);fig.savefig(OUT/'panel_a.svg');plt.close(fig)

fig,ax,w,h=panel('b')
text(ax,2,95,'Inference: fixed transforms, frozen models and postprocessing chosen on validation',fs=12,bold=True,ha='left')
stack(ax,2,46,15,33,'Test.csv',['1,500 windows','15 minutes × 25 bands','Missing = unobserved'],SAGE)
card(ax,22,50,13,25,'Fixed representation',['Per-window transforms','Values + explicit mask','No fitting on Test.csv'],SAGE)
card(ax,41,42,18,39,'Frozen ensemble',[],BLUE)
for yy,line in [(71,'TabM  /  CNN'),(63,'LightGBM'),(55,'Optional TabICLv2'),(47,'Optional training prototypes')]:text(ax,50,yy,line,fs=7)
card(ax,65,49,13,27,'Postprocessing',['Weights from validation','Calibration split only','Clip predictions to [0,1]'],ORANGE)
card(ax,84,44,14,36,'Submission checks',['ID coverage + order','Finite [0,1] outputs','share_added_vessel','score_added_vessel'],SAGE)
for a,b in [((18.5,62),(21.5,62)),((35.5,62),(40.5,62)),((59.5,62),(64.5,62)),((78.5,62),(83.5,62))]:arrow(ax,a,b)
card(ax,41,9,18,20,'Fixed per-window TTA',['Time reversal / gain variants','Use only if validation helps'],PURPLE,True)
arrow(ax,(50,29.5),(50,41.5),dashed=True)
card(ax,65,8,33,21,'Leaderboard objective',[],GREY)
text(ax,81.5,18,r'$\mathrm{Score}=1-\sqrt{\mathrm{mean}(\hat y-y)^2}$',fs=10)
text(ax,81.5,10,'One energy fraction per window; presence score is diagnostic.',fs=7)
text(ax,2,27,'Training-only prototypes may support regression.',fs=7,ha='left')
text(ax,2,19,'No source-clip identification, hidden-background matching,',fs=7,ha='left')
text(ax,2,12,'test-distribution tuning or test-time gradient updates.',fs=7,ha='left')
fig.savefig(OUT/'panel_b.png',dpi=300);fig.savefig(OUT/'panel_b.svg');plt.close(fig)

paths={l:str(OUT/f'panel_{l}.png') for l in ['a','b']}
helpers['compose_figure'](outline,paths,str(OUT/'pipeline_overview.png'),dpi=300,letter_pt=10,
                          letter_font=font_manager.findfont('DejaVu Serif:bold'))
composite=Image.open(OUT/'pipeline_overview.png')
composite.resize((1800,round(composite.height*1800/composite.width))).save(OUT/'pipeline_overview_preview.png')
# Combine native vector panels in the same validated geometry.
ns='http://www.w3.org/2000/svg';ET.register_namespace('',ns)
W,_,_,rh,ry,g=helpers['grid_geom'](outline,dpi=300)
H=ry[-1]+rh[-1]
root=ET.Element('{'+ns+'}svg',{'width':str(W),'height':str(H),'viewBox':f'0 0 {W} {H}'})
for l in ['a','b']:
    x,y=helpers['panel_xy'](outline,l,dpi=300);ww,hh=helpers['panel_px'](outline,l,dpi=300)
    elem=ET.parse(OUT/f'panel_{l}.svg').getroot()
    elem.set('x',str(x));elem.set('y',str(y));elem.set('width',str(ww));elem.set('height',str(hh))
    root.append(elem)
    label=ET.SubElement(root,'{'+ns+'}text',{'x':str(x+3),'y':str(y+30),'font-family':'serif','font-size':'28','font-weight':'bold'})
    label.text=f'({l})'
ET.ElementTree(root).write(OUT/'pipeline_overview.svg',encoding='utf-8',xml_declaration=True)
manifest={'outline':outline,'producer':str(Path(__file__).resolve()),
          'style_reference':'/Users/hongtaeyang/Downloads/image (2).png',
          'panels':{l:{'path':paths[l],'sha256':hashlib.sha256(Path(paths[l]).read_bytes()).hexdigest()} for l in paths},
          'status':'Design of the implemented candidate pipeline; dashed branches require separate validation.'}
(OUT/'pipeline_provenance.json').write_text(json.dumps(manifest,indent=2))
print('Wrote',OUT/'pipeline_overview.png',composite.size)
