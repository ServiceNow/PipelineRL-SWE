"""Pinned Intern-Decision success pilot; original Jev prompts and fixed cost replay."""
import argparse,hashlib,importlib.util,json,math,time
from pathlib import Path
import numpy as np
from jev_pilot import POOLS,load_pool,curve_arrays,compare_pair,parse_response

MODEL='internlm/Intern-Decision-4B'
REVISION='0e5e6aa7d6d750e2b1504ba11a8136cb58aeb3cd'
TEMPERATURE=1.99241824
SOURCE=Path('analysis/cost_headroom/jev_pilot_20261001')
HYBRID=Path('analysis/cost_headroom/jev_hybrid_20261001/results.json')
PROTOCOL=Path('analysis/cost_headroom/intern_decision_20261001')

def prepare():
    PROTOCOL.mkdir(parents=True,exist_ok=True)
    path=PROTOCOL/'manifest.json'
    if path.exists():return json.loads(path.read_text())
    source=json.loads((SOURCE/'manifest.json').read_text());requests=[]
    for r in source['requests']:
        body={'state':r['body']['state'],'questions':r['body']['questions']}
        requests.append({'pool':r['pool'],'problem_id':r['problem_id'],'model_slots':r['model_slots'],'body':body})
    result={'model':MODEL,'revision':REVISION,'temperature':TEMPERATURE,'max_length':8192,
            'success_request_sha256':source['request_sha256'],'requests':requests,
            'request_sha256':hashlib.sha256(json.dumps(requests,sort_keys=True,ensure_ascii=False).encode()).hexdigest()}
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n');return result

def infer(a,m):
    import fcntl,torch
    from huggingface_hub import snapshot_download
    a.out.mkdir(parents=True,exist_ok=True)
    lock=(a.out/'collector.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required')
    checkpoint=snapshot_download(MODEL,revision=REVISION,token=False)
    module_path=Path(checkpoint)/'inference.py'
    spec=importlib.util.spec_from_file_location('intern_published_inference',module_path)
    import sys
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    dtype='bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
    engine=module.DecisionEngine(checkpoint=checkpoint,temperature=TEMPERATURE,max_length=8192,dtype=dtype,device='cuda')
    metadata={'model':MODEL,'revision':REVISION,'inference_sha256':hashlib.sha256(module_path.read_bytes()).hexdigest(),
              'request_sha256':m['request_sha256'],'temperature':TEMPERATURE,'dtype':dtype,
              'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'api_calls':0}
    # Check every request BEFORE inference, rejecting overlength inputs without truncation.
    token_counts=[]
    for r in m['requests']:
        _,batch,_=engine.backend.encode(module.validate_request(r['body']))
        token_counts.append(int(batch['input_ids'].shape[-1]))
    metadata['max_input_tokens']=max(token_counts);metadata['input_tokens']=token_counts
    (a.out/'execution.json').write_text(json.dumps(metadata,indent=2)+'\n')
    response_path=a.out/'responses.jsonl';done=set()
    if response_path.exists():
        for r in map(json.loads,response_path.open()):
            if r['request_sha256']!=m['request_sha256'] or r['revision']!=REVISION:raise ValueError('Resume protocol mismatch')
            done.add((r['pool'],r['problem_id']))
    with response_path.open('a') as output:
        for r in m['requests']:
            if (r['pool'],r['problem_id']) in done:continue
            started=time.time();response=engine.predict(r['body']);probabilities=parse_response(response,r['model_slots'])
            row={'pool':r['pool'],'problem_id':r['problem_id'],'model_slots':r['model_slots'],'p_successes':probabilities,
                 'request_sha256':m['request_sha256'],'revision':REVISION,'elapsed_seconds':time.time()-started,'response':response}
            output.write(json.dumps(row)+'\n');output.flush();done.add((r['pool'],r['problem_id']))
            if len(done)%25==0:print(json.dumps({'completed':len(done),'expected':len(m['requests'])}),flush=True)
    status={'complete':len(done)==len(m['requests']),'valid_calls':len(done),'expected_calls':len(m['requests']),'api_spend_usd':0}
    (a.out/'collection_status.json').write_text(json.dumps(status,indent=2)+'\n')
    lock.close()

def analyze(a,m):
    rows={(r['pool'],r['problem_id']):r for r in map(json.loads,(a.out/'responses.jsonl').open())}
    jev=json.loads(HYBRID.read_text())
    results={'model':MODEL,'revision':REVISION,'request_sha256':m['request_sha256'],'temperature':TEMPERATURE,'pools':{}}
    for label in POOLS:
        d=load_pool(label);selected=[r['problem_id'] for r in m['requests'] if r['pool']==label]
        index={p:i for i,p in enumerate(d['ids'])};ii=np.asarray([index[p] for p in selected])
        intern=d['p'].copy();intern[ii]=np.asarray([rows[label,p]['p_successes'] for p in selected])
        jp=d['p'].copy();saved=jev['pools'][label]
        if saved['problem_ids']!=selected:raise ValueError('Jev alignment mismatch')
        jp[ii]=np.asarray(saved['jev_successes'])
        arms={'intern_our_cost':(intern,d['cost']),'ours_our_cost':(d['p'],d['cost']),'jev_our_cost':(jp,d['cost']),
              'intern_median_cost':(intern,d['median']),'ours_median_cost':(d['p'],d['median']),'jev_median_cost':(jp,d['median'])}
        curves={name:curve_arrays(p,c,d['q'],d['paid'],ii) for name,(p,c) in arms.items()}
        rng=np.random.default_rng(0);bs=[rng.integers(0,len(ii),len(ii)) for _ in range(1000)]
        pairs=[('intern_our_cost','ours_our_cost'),('intern_our_cost','jev_our_cost'),
               ('intern_median_cost','ours_median_cost'),('intern_median_cost','jev_median_cost'),('intern_our_cost','intern_median_cost')]
        contrasts={}
        for left,right in pairs:
            point,band=compare_pair(curves,left,right,np.arange(len(ii)))
            boot=np.asarray([compare_pair(curves,left,right,b)[0] for b in bs]);valid=boot[np.isfinite(boot)]
            contrasts[left+'_vs_'+right]={'direct_cost_saved':point,'accuracy_band':band,'ci95':np.percentile(valid,[2.5,97.5]).tolist() if len(valid) else None,'valid_bootstrap':len(valid),'bootstrap':boot.tolist()}
        metrics={}
        for name,p in [('intern',intern),('ours',d['p']),('jev',jp)]:
            pp=np.clip(p[ii],1e-6,1-1e-6);q=d['q'][ii]
            metrics[name]={'expected_draw_brier':float(np.mean(pp**2-2*pp*q+q)),
                          'expected_draw_logloss':float(-np.mean(q*np.log(pp)+(1-q)*np.log(1-pp))),
                          'mean_predicted_success':float(pp.mean()),'mean_observed_success':float(q.mean())}
        results['pools'][label]={'n_test':len(ii),'problem_ids':selected,'intern_successes':intern[ii].tolist(),'metrics':metrics,'contrasts':contrasts}
        print(label,json.dumps({k:{x:y for x,y in v.items() if x!='bootstrap'} for k,v in contrasts.items()}),flush=True)
    (a.out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    lines=['# Intern-Decision-4B success pilot','','Same frozen 100 problems per dataset and original Jev success questions/training aggregate priors. Published inference with pinned weights and default temperature; no prompt tuning, domain calibration, or target-model generations.','','Direct generation cost savings at matched accuracy; 1,000 paired problem bootstrap draws. Positive favors Intern.','','| Dataset | Intern vs ours, our costs fixed | Intern vs Jev, our costs fixed | Intern vs ours, median costs fixed |','|---|---|---|---|']
    def fmt(v):return f"{100*v['direct_cost_saved']:+.1f}% [{100*v['ci95'][0]:+.1f}, {100*v['ci95'][1]:+.1f}]" if v['ci95'] else 'No shared band'
    for label,d in results['pools'].items():lines.append('| '+label+' | '+' | '.join(fmt(d['contrasts'][k]) for k in ['intern_our_cost_vs_ours_our_cost','intern_our_cost_vs_jev_our_cost','intern_median_cost_vs_ours_median_cost'])+' |')
    lines+=['','Conditional exploratory replay of previously examined test problems. All routes and valid generation draws remain clustered by problem. Convex-hull frontiers use evaluation outcomes, not a separately selected deployment policy. Pair-specific accuracy bands; no training uncertainty or multiplicity correction. Predictor inference overhead excluded from generation spending. No API spending. All probability metrics, derived predictions, contrasts and bootstrap samples in results.json.']
    (a.out/'REPORT.md').write_text('\n'.join(lines)+'\n')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=Path('/mnt/llmd/results/exps/aristides/reason/intern_decision_20261001'))
    ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--analyze-only',action='store_true');a=ap.parse_args()
    m=prepare()
    if a.prepare_only:return
    if not a.analyze_only:infer(a,m)
    analyze(a,m)

if __name__=='__main__':main()
