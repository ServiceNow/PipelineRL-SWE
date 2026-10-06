"""Overview figure: shared-prefill architecture and fresh deterministic curves."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
import numpy as np

BILLED=json.loads((Path(__file__).resolve().parent/'data/fresh_billed_curves'+__import__('os').environ.get('RESULT_TAG','')+'.json').read_text())
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
 ax.set_xlim(0,max(sat)*1.08)
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
 ax.set_ylim((60,88) if dataset=='mmlupro' else (38,73))

HERE=Path(__file__).resolve().parent
BLUE='#2166AC';TEAL='#138A80';INK='#263240';GREY='#718096';PALE='#EEF5FB'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'pdf.fonttype':42,'ps.fonttype':42})
fig=plt.figure(figsize=(7.1,3.05),facecolor='white')
gs=fig.add_gridspec(1,2,width_ratios=[1.7,1],left=.02,right=.97,bottom=.22,top=.86,wspace=.29)
a=fig.add_subplot(gs[0]);a.set_xlim(-.1,10);a.set_ylim(0,6);a.axis('off')
a.text(0,6.25,'(a) Routing architecture',weight='bold',fontsize=9)
def box(x,y,w,h,label,fc='white',ec=GREY,fs=8):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08,rounding_size=.15',linewidth=1,edgecolor=ec,facecolor=fc))
 a.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=fs)
def arrow(start,end,color=GREY):
 a.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=9,lw=1.2,color=color,connectionstyle='arc3'))
box(.05,2.85,1.1,.8,'Query $x$')
box(1.75,2.35,2.0,1.8,'Frozen 4B\nencoder\nPrefill pass',PALE,BLUE)
arrow((1.2,3.25),(1.65,3.25))
box(4.4,3.5,2.05,1.15,'Success readouts\n'+r'$\hat{p}_1,\ldots,\hat{p}_5$',PALE,BLUE,6.3)
box(4.4,1.85,2.05,1.15,'Cost readouts\n'+r'$\hat{c}_1,\ldots,\hat{c}_5$','#EAF6F3',TEAL,6.3)
arrow((3.85,3.55),(4.3,4.03),BLUE);arrow((3.85,2.95),(4.3,2.32),TEAL)
box(7.12,2.8,2.5,.9,'Route selection','white',INK,8)
arrow((6.55,4.0),(7.02,3.4),BLUE);arrow((6.55,2.3),(7.02,3.05),TEAL)
box(7.12,.65,2.5,1.4,'Selected route\ngenerates answer',PALE,BLUE,7)
arrow((8.37,2.7),(8.37,2.15),INK)
a.text(2.75,1.8,'Shared activations',ha='center',fontsize=7.4,color=GREY)
a.text(0,-.45,'Route selection uses only the encoder activations.',fontsize=7,color=GREY)
right=gs[1].subgridspec(2,1,hspace=.9)
b=fig.add_subplot(right[0]);c=fig.add_subplot(right[1])
panel(b,'mmlupro','(b) Fresh MMLU-Pro, billed')
panel(c,'omni500','Fresh Omni-MATH, billed')
b.set_xlabel('')
handles,labels=c.get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,fontsize=6.6,bbox_to_anchor=(.5,.035),markerscale=2)
fig.text(.49,.012,'Dots: single-V policies; lines: their frontiers; hollow circles: calibration-selected policies; arrows: cost saved at matched accuracy.',ha='center',fontsize=6.2,color=GREY)
for suffix in ['pdf','png','svg']:fig.savefig(HERE/'figures'/f'shared_prefill_overview.{suffix}',dpi=250,bbox_inches='tight',pad_inches=.07)
svg=HERE/'figures/shared_prefill_overview.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
print('Saved shared_prefill_overview.pdf/png/svg')
