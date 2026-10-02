"""Overview figure: shared-prefill architecture and fresh deterministic curves."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
import numpy as np
from make_fresh_curves import panel, COLORS, LABELS

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
box(4.4,3.5,2.05,1.15,'Success readouts\n'+r'$\hat{p}_1,\ldots,\hat{p}_5$',PALE,BLUE,7)
box(4.4,1.85,2.05,1.15,'Cost readouts\n'+r'$\hat{c}_1,\ldots,\hat{c}_5$','#EAF6F3',TEAL,7)
arrow((3.85,3.55),(4.3,4.03),BLUE);arrow((3.85,2.95),(4.3,2.32),TEAL)
box(7.12,2.8,2.5,.9,'Route selection','white',INK,8)
arrow((6.55,4.0),(7.02,3.4),BLUE);arrow((6.55,2.3),(7.02,3.05),TEAL)
box(7.12,.65,2.5,1.4,'Selected route\ngenerates answer',PALE,BLUE,7)
arrow((8.37,2.7),(8.37,2.15),INK)
a.text(2.75,1.8,'Shared activations',ha='center',fontsize=7.4,color=GREY)
a.text(0,-.45,'Route selection uses only the encoder activations.',fontsize=7,color=GREY)
right=gs[1].subgridspec(2,1,hspace=.9)
b=fig.add_subplot(right[0]);c=fig.add_subplot(right[1])
panel(b,'mmlupro','(b) Fresh MMLU-Pro',compact=True)
panel(c,'omni500','Fresh Omni-MATH',compact=True)
b.set_xlabel('')
handles,labels=c.get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,fontsize=6.6,bbox_to_anchor=(.5,.035),markerscale=2)
fig.text(.49,.012,'Single-V policies; hollow circles mark calibration selections. Axes stop just past the first point within 0.5 pp of each arm’s peak. Generation spending excludes encoder overhead.',ha='center',fontsize=6.2,color=GREY)
for suffix in ['pdf','png','svg']:fig.savefig(HERE/'figures'/f'shared_prefill_overview.{suffix}',dpi=250,bbox_inches='tight',pad_inches=.07)
svg=HERE/'figures/shared_prefill_overview.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
print('Saved shared_prefill_overview.pdf/png/svg')
