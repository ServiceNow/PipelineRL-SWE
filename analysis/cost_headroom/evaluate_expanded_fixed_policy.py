"""Apply frozen original success/cost readouts on genuinely fresh expansion problems."""
import argparse,json
from pathlib import Path
import numpy as np
from carrot_compare import POOLS,read_predictions,curve_arrays,compare_pair
from jev_pilot import load_pool
from decompose import R
OUT=Path('/mnt/llmd/results/exps/aristides/reason/expanded_eval_20261001')
VREPORT=Path('analysis/cost_headroom/fixed_policy_20261001/results.json')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['mmlupro','omni500'],required=True)
    ap.add_argument('--cost-file',default='cost_preds.jsonl')
    ap.add_argument('--output-file',default='fixed_policy_results.json')
    a=ap.parse_args()
    label=a.dataset;pool='MMLU-Pro' if label=='mmlupro' else 'Omni';folder=OUT/label.replace('-','_');t=np.load(folder/'tensors.npz',allow_pickle=True)
    ids=[str(x) for x in t['problem_ids']];slots=[str(x) for x in t['model_slots']];probs=[json.loads(x) for x in (folder/'problems.jsonl').read_text().splitlines()]
    old=load_pool(pool);n_old=len(old['ids']);newidx=np.arange(n_old,len(ids));n=len(newidx);v=t['valid'][newidx].astype(bool);cnt=v.sum(2)
    if n!=len(json.loads((folder/'expansion_manifest.json').read_text())['new_problem_ids']) or not (cnt>0).all():raise ValueError('Expansion sample incomplete/misaligned')
    q=np.where(v,t['final_outcome'][newidx],0).sum(2)/cnt
    length=np.where(v,t['completion_tokens'][newidx],0).sum(2)/cnt;inp=np.where(v,t['prompt_tokens'][newidx],0).sum(2)/cnt
    from decompose import MK
    rates=dict(MK)
    pricepath=R/POOLS[pool][0]/'prices.json'
    if pricepath.exists():rates.update(json.loads(pricepath.read_text()))
    pin=np.asarray([rates[s][0] for s in slots])/1e6;pout=np.asarray([rates[s][1] for s in slots])/1e6
    paid=(inp*pin+length*pout)*100
    z=np.load(folder/'prefill_combined.npz',allow_pickle=True);pids=[str(x) for x in z['problem_ids']]
    pfile=folder/'success_preds.jsonl';cfile=folder/a.cost_file
    if not pfile.exists() or not cfile.exists():raise RuntimeError('Frozen readouts have not been run')
    p=read_predictions(pfile,ids,'p_successes',len(slots))[newidx]
    c=read_predictions(cfile,ids,'expected_costs',len(slots))[newidx]*100
    # Reconstruct the original route medians from the frozen training labels.
    base=np.load(R/POOLS[pool][0]/'tensors.npz',allow_pickle=True);split=json.loads((R/POOLS[pool][0]/'split_manifest.json').read_text());bi={str(x):i for i,x in enumerate(base['problem_ids'])};tr=np.asarray([bi[str(x)] for x in split['train_problem_ids']]);bv=base['valid'].astype(bool)
    med=np.asarray([np.median(base['completion_tokens'][tr,j][bv[tr,j]]) for j in range(len(slots))])
    median=(inp*pin+med*pout)*100
    vreport=json.loads(VREPORT.read_text());V=vreport['pools'][pool]['selected_V_cents_per_correct']
    gains={};chosen={}
    for arm,cost in [('learned',c),('median',median)]:chosen[arm]=(V*p-cost).argmax(1)
    ri=np.arange(n);acc={arm:float(q[ri,ix].mean()) for arm,ix in chosen.items()};spend={arm:float(paid[ri,ix].mean()/100) for arm,ix in chosen.items()}
    bootrng=np.random.default_rng(20261002);bs=bootrng.integers(0,n,(2000,n));ql=q[ri,chosen['learned']];qm=q[ri,chosen['median']];cl=paid[ri,chosen['learned']];cm=paid[ri,chosen['median']]
    bcost=np.asarray([1-cl[b].mean()/cm[b].mean() for b in bs]);bacc=np.asarray([(ql[b]-qm[b]).mean() for b in bs])
    point=1-cl.mean()/cm.mean()
    # Frozen sample-stratum weights were declared before collection. Report both the
    # unweighted sample mean and design-weighted target-population estimate.
    strata=([str(x.get('subject','')) for x in probs[n_old:]] if label=='mmlupro'
            else [str(round(float(x.get('difficulty',0)))) for x in probs[n_old:]])
    weights=json.loads((folder/'expansion_manifest.json').read_text())['stratum_weights']
    groups={s:np.flatnonzero(np.asarray(strata)==s) for s in weights}
    if any(len(ix)==0 for ix in groups.values()):raise ValueError('A target stratum has no sampled problems')
    def weighted_mean(x):
        return sum(float(weights[s])*float(np.mean(x[ix])) for s,ix in groups.items())
    wa_l=weighted_mean(ql);wa_m=weighted_mean(qm);wc_l=weighted_mean(cl);wc_m=weighted_mean(cm)
    wrng=np.random.default_rng(20261003);wbcost=[];wbacc=[]
    for _ in range(2000):
        draws={s:ix[wrng.integers(0,len(ix),len(ix))] for s,ix in groups.items()}
        wql=sum(float(weights[s])*ql[draws[s]].mean() for s in groups);wqm=sum(float(weights[s])*qm[draws[s]].mean() for s in groups)
        wcl=sum(float(weights[s])*cl[draws[s]].mean() for s in groups);wcm=sum(float(weights[s])*cm[draws[s]].mean() for s in groups)
        wbcost.append(1-wcl/wcm);wbacc.append(wql-wqm)
    weighted={'learned_cost_accuracy':wa_l,'median_cost_accuracy':wa_m,
      'learned_mean_generation_cost_usd':wc_l/100,'median_mean_generation_cost_usd':wc_m/100,
      'learned_savings_vs_median':1-wc_l/wc_m,'cost_savings_ci95':np.percentile(wbcost,[2.5,97.5]).tolist(),
      'accuracy_delta':wa_l-wa_m,'accuracy_delta_ci95':np.percentile(wbacc,[2.5,97.5]).tolist(),
      'target_stratum_weights':weights,'sample_counts':{s:int(len(ix)) for s,ix in groups.items()}}
    # The calibration-selected V is frozen for the primary comparison. It landed
    # on the old grid's upper edge, so include a small, prespecified operating-point
    # sensitivity sweep instead of presenting that one choice as representative.
    sensitivity={}
    values=sorted(set([float(V/100),1e-4,1e-3,1e-2]))
    for vd in values:
        vc=vd*100
        il=(vc*p-c).argmax(1);im=(vc*p-median).argmax(1)
        ql_v=q[ri,il];qm_v=q[ri,im];cl_v=paid[ri,il];cm_v=paid[ri,im]
        sb_cost=np.asarray([1-cl_v[b].mean()/cm_v[b].mean() for b in bs])
        sb_acc=np.asarray([(ql_v[b]-qm_v[b]).mean() for b in bs])
        sensitivity[f'{vd:g}']={'learned_accuracy':float(ql_v.mean()),'median_accuracy':float(qm_v.mean()),
          'accuracy_delta':float((ql_v-qm_v).mean()),'accuracy_delta_ci95':np.percentile(sb_acc,[2.5,97.5]).tolist(),
          'learned_mean_generation_cost_usd':float(cl_v.mean()/100),'median_mean_generation_cost_usd':float(cm_v.mean()/100),
          'cost_savings':float(1-cl_v.mean()/cm_v.mean()),'cost_savings_ci95':np.percentile(sb_cost,[2.5,97.5]).tolist()}
    front={}
    # Supplementary outcome-swept matched-accuracy frontier, same metric as original.
    arms={'learned':(p,c),'median':(p,median)};curves={k:curve_arrays(pp,cc,q,paid,np.arange(n)) for k,(pp,cc) in arms.items()}
    pt,band=compare_pair(curves,'learned','median',np.arange(n));bs2=[bootrng.integers(0,n,n) for _ in range(1000)]
    fb=np.asarray([compare_pair(curves,'learned','median',b)[0] for b in bs2]);finite=np.isfinite(fb)
    front={'direct_savings':pt,'accuracy_band':band,'ci95':np.percentile(fb[finite],[2.5,97.5]).tolist() if finite.any() else None,'valid_bootstrap':int(finite.sum())}
    report={'dataset':pool,'n_fresh':n,'problem_ids':[ids[i] for i in newidx],'selected_V_dollars_per_correct':V/100,
      'learned_cost_policy':{'accuracy':acc['learned'],'mean_generation_cost_usd':spend['learned']},
      'median_cost_policy':{'accuracy':acc['median'],'mean_generation_cost_usd':spend['median']},
      'learned_savings_vs_median':point,'cost_savings_ci95':np.percentile(bcost,[2.5,97.5]).tolist(),
      'accuracy_delta':float((ql-qm).mean()),'accuracy_delta_ci95':np.percentile(bacc,[2.5,97.5]).tolist(),
      'frontier_supplementary':front,'stratum_weighted':weighted,
      'fixed_V_sensitivity_dollars_per_correct':sensitivity,
      'cost_prediction_file':str(cfile),
      'protocol':'Fresh expansion only. Heads fitted using original training/calibration data and V fixed from original calibration; expansion labels used only for evaluation. Routing at a calibration-selected fixed V, one route/problem; weighted/unweighted estimates, paired problem bootstrap. Frontier is secondary/descriptive and uses evaluation outcomes. No model/readout fitting on expansion labels.'}
    (folder/a.output_file).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['problem_ids','mmlu_stratum_weights']}),flush=True)
if __name__=='__main__':main()
