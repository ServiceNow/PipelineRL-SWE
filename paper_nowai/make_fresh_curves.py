"""Observed deterministic routing sweeps for all three evaluation datasets."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE/'data/fresh_routing_curves.json').read_text())
COLORS = {'learned':'#2166AC','median':'#D67C27','mean':'#138A80'}
LABELS = {'learned':'Prefill cost prediction','median':'Training median length','mean':'Training mean length'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,
                     'axes.labelcolor':'#263240','text.color':'#263240','savefig.facecolor':'white'})


def panel(ax, dataset, title, compact=False):
    d = DATA['datasets'][dataset]
    size = 1.8 if compact else 3.5
    for arm in ['median','mean','learned']:
        c = d['curves'][arm]
        x = np.array(c['mean_cost_usd'])*1000
        y = np.array(c['accuracy'])*100
        # Points retain V order, including nonmonotonic test behavior.
        ax.plot(x,y,color=COLORS[arm],lw=.8 if compact else 1.2,alpha=.85,zorder=2)
        ax.scatter(x,y,s=size,color=COLORS[arm],alpha=.65,label=LABELS[arm],zorder=3)
        selected = [r['arms'][arm] for r in d['calibration_selected'].values() if arm in r['arms']]
        ax.scatter([a['mean_cost_usd']*1000 for a in selected],
                   [a['accuracy']*100 for a in selected],s=16 if compact else 40,
                   facecolors='white',edgecolors=COLORS[arm],linewidths=.8 if compact else 1.3,zorder=4)
    # Keep the plot focused on the tradeoff. The cutoff is set by a fixed,
    # outcome-agnostic plotting rule: 10% past the most expensive arm's first
    # point within 0.5 percentage points of that arm's observed maximum.
    saturation_costs = []
    for arm in ['median','mean','learned']:
        curve = d['curves'][arm]
        spend = np.asarray(curve['mean_cost_usd'])*1000
        accuracy = np.asarray(curve['accuracy'])*100
        near_best = np.flatnonzero(accuracy >= accuracy.max() - .5)
        saturation_costs.append(float(spend[near_best[0]]))
    ax.set_xlim(left=0, right=max(saturation_costs)*1.10)
    ax.set_title(title,loc='left',fontsize=8 if compact else 12,fontweight='bold',pad=6)
    ax.grid(color='#E6EBF0',lw=.6)
    ax.set_xlabel('Generation spend ($ / 1,000 queries)',fontsize=6.5 if compact else 10)
    ax.set_ylabel('Accuracy (%)',fontsize=7 if compact else 10)
    ax.tick_params(labelsize=6 if compact else 9)
    if dataset in {'mmlupro','omni500'}:
        ax.set_ylim((60,88) if dataset=='mmlupro' else (38,73))


if __name__ == '__main__':
    fig,axs = plt.subplots(1,3,figsize=(14.8,4.2))
    panel(axs[0],'lcb','LiveCodeBench · 341 original test problems')
    panel(axs[1],'mmlupro','MMLU-Pro · 6,500 new problems')
    panel(axs[2],'omni500','Omni-MATH · 1,000 new problems')
    handles,labels = axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,1.015))
    fig.text(.5,.025,'Each point is a deterministic V policy; lines follow the V sweep. Hollow circles: original-calibration selections. X-axis ends 10% past the first point within 0.5 pp of each arm’s best observed accuracy.\nObserved costs include output tokens; encoder overhead is excluded. Higher-spend tails are clipped; paired CIs are reported separately.',
             ha='center',fontsize=8,color='#65758B')
    fig.subplots_adjust(left=.045,right=.99,bottom=.23,top=.86,wspace=.25)
    for ext in ['png','pdf','svg']:
        fig.savefig(HERE/'figures'/f'fresh_accuracy_cost_curves.{ext}',dpi=250,bbox_inches='tight',pad_inches=.08)
    plt.close(fig)
    print(HERE/'figures/fresh_accuracy_cost_curves.png')
