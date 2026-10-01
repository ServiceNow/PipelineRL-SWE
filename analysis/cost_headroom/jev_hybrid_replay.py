"""Replay saved Jev success predictions with our learned prefill costs. No API calls."""
import hashlib,json
from pathlib import Path
import numpy as np
from jev_pilot import POOLS,load_pool,curve_arrays,compare_pair

SOURCE=Path('analysis/cost_headroom/jev_pilot_20261001')
OUT=Path('analysis/cost_headroom/jev_hybrid_20261001')

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    manifest=json.loads((SOURCE/'manifest.json').read_text())
    rows={(r['pool'],r['problem_id']):r for r in map(json.loads,(SOURCE/'responses.jsonl').open()) if r['status']=='ok'}
    results={'success_request_sha256':manifest['request_sha256'],
             'responses_sha256':hashlib.sha256((SOURCE/'responses.jsonl').read_bytes()).hexdigest(),
             'protocol':'Saved Jev success probabilities, unchanged prompts and no calibration. Our existing learned prefill costs fixed for primary success comparison. Original 100-per-pool test subsets; all valid generation draws retained per problem. 1000 paired problem bootstrap draws, seed0. Conditional descriptive matched-accuracy convex-hull frontiers with pair-specific shared bands. API/encoder overhead excluded from target generation costs. No new paid calls or target generations.',
             'pools':{}}
    comparisons=[('jev_our_cost','ours_our_cost'),('jev_our_cost','jev_median_cost'),('ours_our_cost','ours_median_cost'),('jev_median_cost','ours_median_cost')]
    for label in POOLS:
        d=load_pool(label);selected=[r['problem_id'] for r in manifest['requests'] if r['pool']==label]
        index={p:i for i,p in enumerate(d['ids'])};ii=np.asarray([index[p] for p in selected])
        if any((label,p) not in rows for p in selected):raise ValueError('Missing saved success response')
        pj=d['p'].copy();pj[ii]=np.asarray([rows[label,p]['p_successes'] for p in selected])
        if not np.isfinite(pj[ii]).all() or not ((pj[ii]>=0)&(pj[ii]<=1)).all():raise ValueError('Invalid probability')
        arms={'jev_our_cost':(pj,d['cost']),'ours_our_cost':(d['p'],d['cost']),
              'jev_median_cost':(pj,d['median']),'ours_median_cost':(d['p'],d['median'])}
        curves={name:curve_arrays(p,c,d['q'],d['paid'],ii) for name,(p,c) in arms.items()}
        rng=np.random.default_rng(0);resamples=[rng.integers(0,len(ii),len(ii)) for _ in range(1000)]
        contrasts={}
        for left,right in comparisons:
            point,band=compare_pair(curves,left,right,np.arange(len(ii)))
            boot=np.asarray([compare_pair(curves,left,right,b)[0] for b in resamples]);finite=boot[np.isfinite(boot)]
            contrasts[left+'_vs_'+right]={'direct_cost_saved':point,'accuracy_band':band,
                'ci95':np.percentile(finite,[2.5,97.5]).tolist() if len(finite) else None,
                'valid_bootstrap':len(finite),'bootstrap':boot.tolist()}
        results['pools'][label]={'n_test':len(ii),'problem_ids':selected,'jev_successes':pj[ii].tolist(),'contrasts':contrasts}
        print(label,json.dumps({k:{x:y for x,y in v.items() if x!='bootstrap'} for k,v in contrasts.items()}),flush=True)
    (OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    lines=['# Jev success + our prefill cost replay','','No new API calls. Same 100 held-out problems per dataset. Prompts, probabilities and trained heads unchanged.','','## Primary comparison','','Direct generation cost savings from **Jev success + our costs** relative to **our success + our costs**, at matched accuracy. Positive favors Jev success. Paired problem-bootstrap 95% intervals (1,000 draws).','','| Dataset | Hybrid savings vs our full router |','|---|---|']
    def fmt(x):return f"{100*x['direct_cost_saved']:+.1f}% [{100*x['ci95'][0]:+.1f}, {100*x['ci95'][1]:+.1f}]"
    for label,d in results['pools'].items():lines.append('| '+label+' | '+fmt(d['contrasts']['jev_our_cost_vs_ours_our_cost'])+' |')
    lines+=['','## Cost-head effect with Jev success held fixed','','Our learned costs compared with median training output length, keeping Jev success predictions fixed:','','| Dataset | Learned-cost savings |','|---|---|']
    for label,d in results['pools'].items():lines.append('| '+label+' | '+fmt(d['contrasts']['jev_our_cost_vs_jev_median_cost'])+' |')
    lines+=['','## Interpretation limits','','Exploratory replay of an already analyzed pilot subset. All routes and stored valid generation draws remain clustered by problem. The frontiers use evaluation outcomes to choose convex-hull mixtures; they are descriptive and do not establish an independently selected deployment policy. Intervals condition on existing predictions and omit training uncertainty. Each contrast uses its own shared accuracy band. These are generation costs; Jev inference and encoder overhead are excluded. No calibration, prompt tuning, or paper edits. Full contrasts, success predictions and bootstrap samples are in results.json.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
