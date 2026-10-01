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
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['mmlupro','omni500'],required=True);a=ap.parse_args()
    label=a.dataset;pool='MMLU-Pro' if label=='mmlupro' else 'Omni';folder=OUT/label.replace('-','_');t=np.load(folder/'tensors.npz',allow_pickle=True)
    ids=[str(x) for x in t['problem_ids']];slots=[str(x) for x in t['model_slots']];probs=[json.loads(x) for x in (folder/'problems.jsonl').read_text().splitlines()]
    old=load_pool(pool);n_old=len(old['ids']);newidx=np.arange(n_old,len(ids));n=len(newidx);v=t['valid'][newidx].astype(bool);cnt=v.sum(2)
    if n!=len(json.loads((folder/'expansion_manifest.json').read_text())['new_problem_ids']) or not (cnt>0).all():raise ValueError('Expansion sample incomplete/misaligned')
    q=np.where(v,t['final_outcome'][newidx],0).sum(2)/cnt
    length=np.where(v,t['completion_tokens'][newidx],0).sum(2)/cnt;inp=np.where(v,t['prompt_tokens'][newidx],0).sum(2)/cnt
    rates=dict(old['slots'] and __import__('decompose').MK)
    pricepath=R/POOLS[pool][0]/'prices.json'
    if pricepath.exists():rates.update(json.loads(pricepath.read_text()))
    pin=np.asarray([rates[s][0] for s in slots])/1e6;pout=np.asarray([rates[s][1] for s in slots])/1e6
    paid=(inp*pin+length*pout)*100
    z=np.load(folder/'prefill_combined.npz',allow_pickle=True);pids=[str(x) for x in z['problem_ids']]
    pfile=folder/'success_preds.jsonl';cfile=folder/'cost_preds.jsonl'
    if not pfile.exists() or not cfile.exists():raise RuntimeError('Frozen readouts have not been run')
    p=read_predictions(pfile,ids,'p_successes',len(slots))[newidx]
    c=read_predictions(cfile,ids,'expected_costs',len(slots))[newidx]*100
    median=np.broadcast_to(old['median'][newidx*0] if False else old['median'][0],(n,len(slots))).copy()
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
    point=1-cl.mean()/cm.mean();front={}
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
      'frontier_supplementary':front,'mmlu_stratum_weights':json.loads((folder/'expansion_manifest.json').read_text()).get('stratum_weights'),
      'protocol':'Fresh expansion only primary. Original frozen train/cal heads and V; expansion labels used only for evaluation. Routing at a calibration-selected fixed V, one route/problem; weighted/unweighted MMLU estimates, paired problem bootstrap. Frontier is secondary/descriptive and uses evaluation outcomes. No model/readout retraining on expansion labels.'}
    (folder/'fixed_policy_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['problem_ids','mmlu_stratum_weights']}),flush=True)
if __name__=='__main__':main()
