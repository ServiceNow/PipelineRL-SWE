"""Calibration-select one value per correct answer; evaluate frozen routers on test."""
import json
from pathlib import Path
import numpy as np
from jev_pilot import POOLS,load_pool
from carrot_compare import VS
from decompose import R

OUT=Path('analysis/cost_headroom/fixed_policy_20261001')

def main():
    OUT.mkdir(parents=True,exist_ok=True);result={'protocol':'Per pool, select V using only calibration labels: maximize mean [V * expected success - realized mean generation cost] for OUR success and learned cost router. Freeze V, then evaluate exactly one nonrandom route/problem on untouched original test. Compare OUR learned costs to median training-length costs with OUR same success predictions and same V; include paired accuracy and cost differences (not matched-accuracy or test-selected mixtures). 2000 paired problem-bootstrap draws, seed20261001, all routes/draws clustered by problem. V grid frozen as analysis/cost_headroom/carrot_compare.py VS. Conditional on fitted predictions; this is a prespecified policy-learning demonstration on the historical split, not fresh external validation.','pools':{}}
    for label in POOLS:
        d=load_pool(label);p=d['p'];c=d['cost'];cm=d['median'];q=d['q'];paid=d['paid'];te=d['te']
        name,_=POOLS[label];split=json.loads((R/name/'split_manifest.json').read_text());index={x:i for i,x in enumerate(d['ids'])}
        cal=np.asarray([index[str(x)] for x in split['calibration_problem_ids']])
        # Optimize expected utility under the calibrated success and observed mean costs.
        cal_choices=[]
        for v in VS:cal_choices.append(np.argmax(v*p[cal]-c[cal],axis=1))
        cal_choices=np.asarray(cal_choices)
        row=np.arange(len(cal))
        qcal=q[cal][row[None,:],cal_choices];ccal=paid[cal][row[None,:],cal_choices]
        utility=(VS[:,None]*qcal-ccal).mean(1);j=int(np.argmax(utility));v=float(VS[j])
        choices={}
        for arm,cost in [('learned',c),('median',cm)]:choices[arm]=np.argmax(v*p[te]-cost[te],axis=1)
        testrow=np.arange(len(te));ql=q[te][testrow,choices['learned']];qmd=q[te][testrow,choices['median']]
        cl=paid[te][testrow,choices['learned']];cmd=paid[te][testrow,choices['median']]
        point=1-cl.mean()/cmd.mean();delta_cost=cmd-cl;delta_acc=ql-qmd
        rng=np.random.default_rng(20261001);bs=rng.integers(0,len(te),(2000,len(te)))
        bcost=np.asarray([1-cl[x].mean()/cmd[x].mean() for x in bs]);bacc=np.asarray([(ql[x]-qmd[x]).mean() for x in bs])
        report={'n_calibration':len(cal),'n_test':len(te),'selected_V_dollars_per_correct':v/100,'selected_V_cents_per_correct':v,
            'calibration_mean_utility_dollars':float(utility[j]/100),'learned_cost_policy':{'accuracy':float(ql.mean()),'generation_cost_dollars_per_problem':float(cl.mean()/100)},
            'median_cost_policy':{'accuracy':float(qmd.mean()),'generation_cost_dollars_per_problem':float(cmd.mean()/100)},
            'learned_savings_vs_median':float(point),'paired_accuracy_delta':float(delta_acc.mean()),
            'paired_cost_delta_dollars_per_problem':float(delta_cost.mean()/100),
            'cost_savings_ci95':np.percentile(bcost,[2.5,97.5]).tolist(),'accuracy_delta_ci95':np.percentile(bacc,[2.5,97.5]).tolist(),
            'problem_ids':[d['ids'][i] for i in te], 'learned_choices':choices['learned'].tolist(),'median_choices':choices['median'].tolist(),
            'bootstrap_seed':20261001,'bootstrap_replicates':2000}
        result['pools'][label]=report;print(label,json.dumps({k:x for k,x in report.items() if k not in ['problem_ids','learned_choices','median_choices']}),flush=True)
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Calibration-selected fixed-policy evaluation','','Select V by maximizing OUR router’s mean validation utility, V × expected correctness − realized mean target-generation cost. Freeze V before test. Route each test problem once, with no randomized mixtures. Compare learned costs with median training-length costs while keeping our success predictions fixed. Positive savings and positive accuracy difference favor learned costs. Paired problem bootstrap 95% intervals, 2,000 draws.','','| Dataset | Validation-selected V ($/correct) | Learned-cost accuracy | Median-cost accuracy | Cost savings | Accuracy difference |','|---|---:|---:|---:|---:|---:|']
    for label,d in result['pools'].items():lines.append(f"| {label} | {d['selected_V_dollars_per_correct']:.4g} | {d['learned_cost_policy']['accuracy']:.3f} | {d['median_cost_policy']['accuracy']:.3f} | {100*d['learned_savings_vs_median']:+.1f}% [{100*d['cost_savings_ci95'][0]:+.1f}, {100*d['cost_savings_ci95'][1]:+.1f}] | {100*d['paired_accuracy_delta']:+.1f}pp [{100*d['accuracy_delta_ci95'][0]:+.1f}, {100*d['accuracy_delta_ci95'][1]:+.1f}] |")
    lines+=['','This is a fixed-policy check on the historical split, whose test outcomes have informed earlier exploratory analyses. It complements, but does not replace, fresh expanded-test evaluation. Paired intervals keep problems intact and condition on fitted predictors and selected validation V. Predictor overhead excluded.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
