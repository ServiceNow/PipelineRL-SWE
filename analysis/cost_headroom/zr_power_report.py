"""Render completed free analyses and audit variation between held-out folds."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import zr_power as z

summary=json.loads((z.DEST/'summary.json').read_text())
fold_audit={}
for pool in z.POOLS:
 d=z.load(pool); pred=np.load(z.DEST/f'{pool[0]}_crossfit_predictions.npz')
 ch=[z.choices(d,pred['P'],pred['C']),z.choices(d,pred['Pzr'],pred['Czr']),z.choices(d,pred['P'],pred['Cref'])]
 folds=[]
 for fold in range(5):
  ii=np.flatnonzero(pred['fold']==fold)
  effect=z.effects(z.curves(d,ch,ii))
  folds.append({'fold':fold,'n':len(ii),'ours_pp':float(effect[0]),'zr_pp':float(effect[1]),'difference_pp':float(effect[2])})
 fold_audit[pool[0]]=folds
(z.DEST/'fold_effects.json').write_text(json.dumps(fold_audit,indent=2))
labels=[p[0] for p in z.POOLS]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(7.2,3.8),layout='constrained')
for tag,offset,color,title in [('fixed',-.13,'#2B6CB0','Original fixed split'),('crossfit_conditional',.13,'#C05621','Five-fold cross-fit (conditional interval)')]:
 vals=[summary['datasets'][l][tag]['original_bands'] for l in labels]
 points=np.array([v['difference_pp'] for v in vals]); ci=np.array([v['ci95_pp'] for v in vals])
 ax.errorbar(points,np.arange(3)+offset,xerr=np.c_[points-ci[:,0],ci[:,1]-points].T,fmt='o',capsize=4,color=color,label=title)
ax.axvline(0,color='#718096',lw=1,ls='--');ax.set_yticks(range(3),labels);ax.invert_yaxis();ax.set_xlabel('Ours − ZeroRouter: difference in cost savings (percentage points)');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.22),frameon=False,fontsize=8)
fig.suptitle('Free follow-up: fixed split and cross-fitted routing',fontsize=12)
for suffix in ['png','pdf']:fig.savefig(z.DEST/f'comparison.{suffix}',dpi=220)
plt.close(fig)
lines=['# Free ZeroRouter follow-up results','',
'All four approved analyses completed locally without new generations or API spending. Protocol: [ZR_POWER_PROTOCOL.md](../ZR_POWER_PROTOCOL.md). These are exploratory analyses of already observed data.','',
'## Routing comparison','',
'Effects are percentage-point differences in savings versus median-output routing. The original utility sweep and metric are preserved.','',
'| Dataset | Fixed split, effect [95% CI] | Cross-fit, effect [conditional 95% interval] |',
'| --- | --- | --- |']
for label in labels:
 vals=[summary['datasets'][label][k]['original_bands'] for k in ['fixed','crossfit_conditional']]
 cells=[f"{v['difference_pp']:+.1f} [{v['ci95_pp'][0]:+.1f}, {v['ci95_pp'][1]:+.1f}]" for v in vals]
 lines.append(f'| {label} | '+ ' | '.join(cells)+' |')
lines+=['', '**Interpretation:** LCB remains the only clear win in the original fixed-split analysis. Cross-fitting favors our method on all three pools, but its intervals condition on fitted heads and omit uncertainty from overlapping training folds. Larger training sets and randomized splits change the setting; these intervals do not establish three independent confirmatory wins. Keep the original comparison primary.', '',
'![Routing comparison](comparison.png)','',
'One accuracy band shared by all three arms gives similar conclusions; see `summary.json`. Fixed point effects differ slightly from the previously reported bootstrap means. All original per-arm savings reconstruct to within 7e-14 percentage points.','',
'## Generation uncertainty diagnostic','',
'| Dataset | Problem-only SD (pp) | Generation-only SD (pp) | Nested SD (pp) |',
'| --- | --- | --- | --- |']
for label in labels:
 v=summary['datasets'][label]['uncertainty_diagnostic'];lines.append(f"| {label} | {v['problem_only_sd_pp']:.2f} | {v['generation_only_sd_pp']:.2f} | {v['nested_total_sd_pp']:.2f} |")
lines+=['','Generation noise is noticeable, but smaller than problem-resampling uncertainty in this diagnostic. These are empirical resampling distributions, not a decomposition of population variance: problem resampling already includes noise in the observed problem means. The small number of draws limits the inner bootstrap. Extra generations may help, but these results do not quantify the benefit or establish an optimal collection plan.','',
'## Pooled evidence and equivalence','']
p=summary['exploratory_stouffer']['original_bands']['one_sided_p']
lines+=[f'Equal-weight exploratory Stouffer test on the original fixed splits: one-sided p = {p:.4g}. Individual one-sided centered-bootstrap p estimates: '+', '.join(f"{l} {summary['exploratory_stouffer']['original_bands']['individual_p'][l]:.4g}" for l in labels)+'. This is pooled directional evidence, substantially helped by LCB; it does not establish superiority separately on Omni or MMLU-Pro. Bootstrap p values have finite resolution (1/1001).','']
for split,label in [('fixed','Original Omni split'),('crossfit_conditional','Cross-fitted Omni')]:
 v=summary['omni_equivalence'][split]
 lines.append(f"{label}: 90% interval [{v['ci90_pp'][0]:+.1f}, {v['ci90_pp'][1]:+.1f}] pp; +/-5 pp equivalence {'established' if v['percentile_equivalent'] else 'not established'}. Normal approximation TOST p = {v['normal_approximation_tost_p']:.3f}.")
lines+=['','Do not call Omni equivalent. The fixed split remains inconclusive; the cross-fitted result instead favors our method in its different evaluation setting.','',
'## Fold variation and implementation checks','',
'| Dataset | Individual held-out fold effects (pp) |', '| --- | --- |']
for label in labels:lines.append('| '+label+' | '+', '.join(f"{r['difference_pp']:+.1f}" for r in fold_audit[label])+' |')
lines+=['','Individual folds are small and their training sets overlap. Their differences are descriptive, not independent replications. Only one fold partition was run, as specified in the protocol.','',
'The full training row-space reduction preserves the linear model geometry. A direct full-feature check on Omni fold 0 / route 0 gave max probability difference 0.000155 and max relative cost difference 3.7e-7 (solver precision); ridge selected the same penalty. All routes have at least one valid draw. For fidelity, first-draw C selection and Platt calibration retain the legacy treatment of invalid draw 0 as failure (8/6/10 cells on LCB/Omni/MMLU-Pro); all-draw likelihoods and evaluation exclude invalid draws.','',
'Artifacts: `summary.json`, per-pool JSON and bootstrap NPZ files, held-out predictions, fold assignments/selections, `fold_effects.json`, `verification.json`, `provenance.json`, and `comparison.pdf`. Reproduce with `zr_power.py --pool <LCB|Omni|MMLU-Pro>`, then `zr_power.py --aggregate` and `zr_power_report.py`.','',
'## Recommendation','',
'Keep LCB as the established baseline win and describe MMLU-Pro/Omni cautiously. Cross-fitting is useful robustness evidence, suitable for supplementary material with the dependency caveat. The pooled test is not needed for the four-page paper. If a stronger separate MMLU-Pro claim is essential, prioritize additional untouched problems; generation-only uncertainty is smaller but not negligible. No paid follow-up has been launched.']
(z.DEST/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
