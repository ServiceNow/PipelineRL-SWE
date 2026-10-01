"""Overview figure: architecture + existing fixed-split savings.
Uses the saved paper manifest, not pending expansion or cross-fit results.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
import numpy as np

HERE=Path(__file__).resolve().parent
DATA=json.loads((HERE/'figures/data_manifest.json').read_text())
BLUE='#2166AC';TEAL='#138A80';INK='#263240';GREY='#718096';PALE='#EEF5FB'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'pdf.fonttype':42,'ps.fonttype':42})
fig=plt.figure(figsize=(7.1,2.65),facecolor='white')
gs=fig.add_gridspec(1,2,width_ratios=[1.7,1],left=.02,right=.97,bottom=.22,top=.86,wspace=.23)
a=fig.add_subplot(gs[0]);a.set_xlim(-.1,10);a.set_ylim(0,6);a.axis('off')
a.text(0,6.25,'(a) One encoding prices the entire pool',weight='bold',fontsize=9)
def box(x,y,w,h,label,fc='white',ec=GREY,fs=8):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08,rounding_size=.15',linewidth=1,edgecolor=ec,facecolor=fc))
 a.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=fs)
def arrow(start,end,color=GREY):
 a.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=9,lw=1.2,color=color,connectionstyle='arc3'))
box(.05,2.85,1.1,.8,'Query $x$')
box(1.75,2.35,2.0,1.8,'Frozen 4B\nencoder\nOne prefill',PALE,BLUE)
arrow((1.2,3.25),(1.65,3.25))
box(4.4,3.5,2.05,1.15,'Success readouts\n'+r'$\hat{p}_1,\ldots,\hat{p}_5$',PALE,BLUE,7)
box(4.4,1.85,2.05,1.15,'Cost readouts\n'+r'$\hat{c}_1,\ldots,\hat{c}_5$','#EAF6F3',TEAL,7)
arrow((3.85,3.55),(4.3,4.03),BLUE);arrow((3.85,2.95),(4.3,2.32),TEAL)
box(7.12,2.8,2.5,.9,'Route selection','white',INK,8)
arrow((6.55,4.0),(7.02,3.4),BLUE);arrow((6.55,2.3),(7.02,3.05),TEAL)
box(7.12,.65,2.5,1.4,'Selected route\ngenerates answer',PALE,BLUE,7)
arrow((8.37,2.7),(8.37,2.15),INK)
a.text(2.75,1.8,'Shared activations',ha='center',fontsize=7.4,color=GREY)
a.text(0,-.45,'Target activations and generations are unnecessary before dispatch.',fontsize=7,color=GREY)
b=fig.add_subplot(gs[1]);b.spines[['top','right','left']].set_visible(False);b.spines['bottom'].set_color('#B5BEC7')
b.set_title('(b) Same success head, better pricing',fontsize=9,fontweight='bold',loc='left',pad=15)
labels=['LCB','Omni','MMLU-Pro'];keys=['LiveCodeBench','Omni-MATH','MMLU-Pro']
x=np.arange(3);vals=np.array([DATA[k]['gain']*100 for k in keys]);ci=np.array([DATA[k]['gain_ci'] for k in keys])*100
b.bar(x,vals,width=.55,color=BLUE,alpha=.95,zorder=2)
b.errorbar(x,vals,yerr=np.vstack([vals-ci[:,0],ci[:,1]-vals]),fmt='none',ecolor=INK,lw=1,capsize=2.5,zorder=3)
for xx,v,hi in zip(x,vals,ci[:,1]):b.text(xx,hi+2,f'{v:.1f}%',ha='center',fontsize=8,fontweight='bold',color=BLUE)
b.set_xticks(x,labels,fontsize=8);b.tick_params(axis='x',length=0,pad=5);b.set_xlim(-.6,2.6);b.set_ylim(0,52);b.set_yticks([0,20,40]);b.grid(axis='y',color='#E5EBF1',lw=.6,zorder=0)
b.set_ylabel('Generation cost saved (%)',fontsize=8,labelpad=3)
fig.text(.49,.055,'Against median-length pricing at matched accuracy; whiskers: paired 95% bootstrap CIs.',ha='center',fontsize=7,color=GREY)
for suffix in ['pdf','png','svg']:fig.savefig(HERE/'figures'/f'shared_prefill_overview.{suffix}',dpi=250,bbox_inches='tight',pad_inches=.07)
print('Saved shared_prefill_overview.pdf/png/svg')
