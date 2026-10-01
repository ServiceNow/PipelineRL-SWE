"""Jev cost bucket pilot: frozen success predictions, training-only length bins."""
import argparse,asyncio,hashlib,json,math
from pathlib import Path
import numpy as np
from jev_pilot import MODEL,ENDPOINT,ROUTES,GRADING,POOLS,load_pool,collect,curve_arrays,compare_pair

SUCCESS_OUT=Path('analysis/cost_headroom/jev_pilot_20261001')

def prepare(a):
    a.out.mkdir(parents=True,exist_ok=True)
    path=a.out/'manifest.json'
    if path.exists():return json.loads(path.read_text())
    old=json.loads((SUCCESS_OUT/'manifest.json').read_text())
    requests=[];bins={}
    for label in POOLS:
        d=load_pool(label);bins[label]={};routes={};questions={}
        for j,s in enumerate(d['slots']):
            train=d['length'][d['tr'],j]
            edges=np.unique(np.quantile(train,[.2,.4,.6,.8]))
            groups=np.searchsorted(edges,train,side='right')
            criteria={};means={};priors={}
            for k in range(len(edges)+1):
                subset=train[groups==k]
                if not len(subset):continue
                key=f'b{k}'
                lo=0 if k==0 else float(edges[k-1]);hi=float(edges[k]) if k<len(edges) else None
                criteria[key]=f'Mean output length is at least {lo:.6f} tokens'+(f' and less than {hi:.6f} tokens.' if hi is not None else '.')
                means[key]=float(subset.mean());priors[key]=float(len(subset)/len(train))
            bins[label][s]={'edges':edges.tolist(),'representative_tokens':means,'training_probabilities':priors}
            routes[s]={'description':ROUTES[s],'training_mean_output_tokens':float(train.mean()),
                       'bucket_training_frequencies':priors,'bucket_representative_tokens':means}
            questions[s]={'type':'choice','instructions':
                f'Predict the mean total OUTPUT tokens used by route routes.{s} to answer problem under grading_rule, across independent generations. Count hidden reasoning and final answer tokens, exclude input tokens. Select the length bucket. Use problem-specific difficulty and the training-only route statistics as priors. No answer has been generated yet.',
                'criteria':criteria}
        index={p:i for i,p in enumerate(d['ids'])}
        for r in old['requests']:
            if r['pool']!=label:continue
            i=index[r['problem_id']]
            if i not in d['te']:raise ValueError('Success sample not held out')
            body={'model':MODEL,'state':{'problem':d['meta'][r['problem_id']]['problem_statement'],
                  'grading_rule':GRADING[label],'routes':routes},'questions':questions}
            requests.append({'pool':label,'problem_id':r['problem_id'],'model_slots':d['slots'],'body':body})
    manifest={'model':MODEL,'endpoint':ENDPOINT,'success_manifest_sha256':old['request_sha256'],
              'bins':bins,'requests':requests,'request_sha256':hashlib.sha256(json.dumps(requests,sort_keys=True,ensure_ascii=False).encode()).hexdigest()}
    path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    return manifest

def parse_response(response,slots):
    values=[]
    for slot in slots:
        answer=response['answers'][slot]
        if answer.get('type')!='choice':raise ValueError('Expected choice answer')
        probs=answer['probabilities']
        if not isinstance(probs,dict) or not probs:raise ValueError('Missing distribution')
        if any(not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in probs.values()):raise ValueError('Invalid probabilities')
        total=sum(probs.values())
        if abs(total-1)>.02:raise ValueError('Probability sum')
        values.append({k:v/total for k,v in probs.items()})
    return values

def analyze(a,manifest):
    rows={(r['pool'],r['problem_id']):r for r in map(json.loads,(a.out/'responses.jsonl').open()) if r['status']=='ok'}
    result={'request_sha256':manifest['request_sha256'],'pools':{}}
    for label in POOLS:
        d=load_pool(label);selected=[r['problem_id'] for r in manifest['requests'] if r['pool']==label]
        index={p:i for i,p in enumerate(d['ids'])};ii=np.asarray([index[p] for p in selected])
        lengths=d['length'].copy()
        for i,p in zip(ii,selected):
            for j,s in enumerate(d['slots']):
                probs=rows[label,p]['bucket_probabilities'][j];means=manifest['bins'][label][s]['representative_tokens']
                if set(probs)!=set(means):raise ValueError('Bucket mismatch')
                lengths[i,j]=sum(probs[k]*means[k] for k in means)
        jev=(d['inp']*d['pin']+lengths*d['pout'])*100
        train_mean=(d['inp']*d['pin']+d['length'][d['tr']].mean(0)*d['pout'])*100
        costs={'jev':jev,'ours':d['cost'],'median':d['median'],'training_mean':train_mean}
        curves={k:curve_arrays(d['p'],c,d['q'],d['paid'],ii) for k,c in costs.items()}
        rng=np.random.default_rng(0);bs=[rng.integers(0,len(ii),len(ii)) for _ in range(1000)]
        contrasts={}
        for left,right in [('jev','median'),('jev','training_mean'),('jev','ours'),('ours','median')]:
            point,band=compare_pair(curves,left,right,np.arange(len(ii)))
            boot=np.asarray([compare_pair(curves,left,right,b)[0] for b in bs]);valid=boot[np.isfinite(boot)]
            contrasts[left+'_vs_'+right]={'direct_cost_saved':point,'band':band,'ci95':np.percentile(valid,[2.5,97.5]).tolist() if len(valid) else None,'valid_bootstrap':len(valid),'bootstrap':boot.tolist()}
        metrics={}
        for name,c in costs.items():
            predicted=(c/100-d['inp']*d['pin'])/d['pout']
            y=d['length'][ii];pred=predicted[ii]
            metrics[name]={'mae_tokens':float(np.abs(pred-y).mean()),'route_r2':dict(zip(d['slots'],[float(1-np.sum((pred[:,j]-y[:,j])**2)/np.sum((y[:,j]-y[:,j].mean())**2)) for j in range(len(d['slots']))])),
                           'mean_predicted_tokens':float(pred.mean()),'mean_observed_tokens':float(y.mean())}
        result['pools'][label]={'n_test':len(ii),'problem_ids':selected,'predicted_output_tokens':lengths[ii].tolist(),'metrics':metrics,'contrasts':contrasts}
        print(label,json.dumps({k:{x:y for x,y in v.items() if x!='bootstrap'} for k,v in contrasts.items()}),flush=True)
    (a.out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Jev cost-bucket pilot','','Our success predictions are fixed in all arms. Five training-quantile output-length buckets per route; arithmetic training bucket means weighted by Jev probabilities. Input cost and recorded route prices retained. Same 100 held-out problems per dataset as the success pilot; all valid stored generation draws. No new target generations.','','Direct generation cost savings at matched accuracy (paired problem bootstrap, 1,000 draws):','','| Dataset | Jev vs median | Jev vs training mean | Jev vs ours |','|---|---|---|---|']
    def fmt(x):return f"{100*x['direct_cost_saved']:+.1f}% [{100*x['ci95'][0]:+.1f}, {100*x['ci95'][1]:+.1f}]"
    for label,d in result['pools'].items():lines.append('| '+label+' | '+' | '.join(fmt(d['contrasts'][k]) for k in ['jev_vs_median','jev_vs_training_mean','jev_vs_ours'])+' |')
    status=json.loads((a.out/'collection_status.json').read_text())
    lines+=['',f"Collection: {status['valid_calls']}/{status['expected_calls']} calls; recorded API spend ${status['guard_spend_usd']:.9f}.",'','Exploratory reuse of the success-pilot test subset; cost prompt and bins frozen before cost calls, with no tuning or calibration on test outcomes. Intervals condition on fitted predictors and use pair-specific shared accuracy bands. API prediction overhead excluded from generation savings. Bucket expectations are bounded by training bucket means and may miss extreme tails. This tests a prompted, untrained Jev predictor with training aggregate priors; it is not a trained cost head. Raw responses are retained locally in responses.jsonl; manifest and derived predictions/results are saved.']
    (a.out/'REPORT.md').write_text('\n'.join(lines)+'\n')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=Path('analysis/cost_headroom/jev_cost_pilot_20261001'))
    ap.add_argument('--budget-usd',type=float,default=1);ap.add_argument('--concurrency',type=int,default=8)
    ap.add_argument('--api-key-file',default='/home/toolkit/.secrets/openrouter_api_key');ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--analyze-only',action='store_true')
    a=ap.parse_args()
    if a.budget_usd<=0 or a.concurrency<=0 or a.limit<0:ap.error('Invalid guard/count')
    m=prepare(a)
    if a.prepare_only:return
    if a.analyze_only:analyze(a,m);return
    status=asyncio.run(collect(a,m,parse_response,'bucket_probabilities'))
    if status['complete']:analyze(a,m)
    elif not a.limit:raise RuntimeError('Incomplete cost pilot')

if __name__=='__main__':main()
