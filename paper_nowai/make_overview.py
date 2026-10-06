"""Figure 1: cost-accuracy frontiers (ours vs median-length pricing) on LCB test, fresh MMLU-Pro and fresh Omni-MATH, billed prices."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
import numpy as np

BILLED=json.loads((Path(__file__).resolve().parent/('data/fresh_billed_curves'+__import__('os').environ.get('RESULT_TAG','')+'.json')).read_text())
COLORS={'learned':'#2166AC','median':'#D67C27'}
LABELS={'learned':'Prefill cost readouts','median':'Training-median length'}


def panel(ax,dataset,title):
 d=BILLED['datasets'][dataset]
 for arm in ['median','learned']:
  c=d['curves'][arm]
  ax.scatter(np.array(c['mean_cost_usd'])*1000,np.array(c['accuracy'])*100,s=1.2,color=COLORS[arm],alpha=.25,lw=0,zorder=2)
  ax.plot(np.array(c['hull_cost_usd'])*1000,np.array(c['hull_accuracy'])*100,color=COLORS[arm],lw=1.4,solid_capstyle='round',label=LABELS[arm],zorder=3)
  sel=d['selected'][arm].values()
  ax.scatter([x['mean_cost_usd']*1000 for x in sel],[x['accuracy']*100 for x in sel],s=14,facecolors='white',edgecolors=COLORS[arm],linewidths=.9,zorder=5)
 sat=[]
 for arm in ['median','learned']:
  c=d['curves'][arm];sp=np.asarray(c['mean_cost_usd'])*1000;ac=np.asarray(c['accuracy'])*100
  sat.append(float(sp[np.flatnonzero(ac>=ac.max()-.5)[0]]))
 ax.set_xlim(0,1.3 if dataset=='lcb' else max(sat)*1.08)   # LCB: its flat plateau past $1 would hide the frontiers
 # matched-accuracy connectors: the saving is a lateral cost shift from the median rule's frontier to ours
 for k in d['connectors']:
  y=k['accuracy']*100;x0=k['median_cost_usd']*1000;x1=k['ours_cost_usd']*1000
  ax.annotate('',xy=(x1,y),xytext=(x0,y),arrowprops=dict(arrowstyle='-|>',color=INK,lw=.7,mutation_scale=6,shrinkA=0,shrinkB=1),zorder=4)
  ax.text(x0+.01*ax.get_xlim()[1],y-.6,f"−{k['saved']*100:.0f}%",fontsize=5.8,color=INK,ha='left',va='top')
 ax.set_title(title,loc='left',fontsize=8,fontweight='bold',pad=4)
 ax.grid(color='#E6EBF0',lw=.6);ax.set_axisbelow(True)
 for side in ('top','right'):ax.spines[side].set_visible(False)
 ax.set_xlabel('Billed spend ($ / 1,000 queries)',fontsize=6.5);ax.set_ylabel('Accuracy (%)',fontsize=7)
 ax.tick_params(labelsize=6)
 ax.set_ylim({'mmlupro':(60,88),'omni500':(38,73),'lcb':(48,92)}[dataset])

HERE=Path(__file__).resolve().parent
BLUE='#2166AC';INK='#263240';GREY='#718096'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(7.1,2.35),facecolor='white',gridspec_kw=dict(wspace=.28,left=.06,right=.99,bottom=.3,top=.88))
for ax,(ds,title) in zip(axes,[('lcb','LiveCodeBench (test, 341)'),('mmlupro','MMLU-Pro (fresh, 6,500)'),('omni500','Omni-MATH (fresh, 1,000)')]):
 panel(ax,ds,title)
for ax in axes[1:]:ax.set_ylabel('')
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,fontsize=6.8,bbox_to_anchor=(.5,.035),markerscale=2)
fig.text(.5,.0,'Dots: single-V policies; lines: their frontiers; hollow circles: calibration-selected policies; arrows: cost saved at matched accuracy.',ha='center',fontsize=6.2,color=GREY)
for suffix in ['pdf','png','svg']:fig.savefig(HERE/'figures'/f'shared_prefill_overview.{suffix}',dpi=250,bbox_inches='tight',pad_inches=.07)
svg=HERE/'figures/shared_prefill_overview.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
print('Saved shared_prefill_overview.pdf/png/svg')
