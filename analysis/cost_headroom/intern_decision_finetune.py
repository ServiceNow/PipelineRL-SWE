"""Train dataset-specific Intern LoRA success predictors; fixed costs and median controls."""
import argparse,hashlib,importlib.util,json,math,sys,time
from pathlib import Path
import numpy as np
from intern_decision_pilot import MODEL,REVISION,TEMPERATURE,PROTOCOL
from jev_pilot import POOLS,ROUTES,GRADING,load_pool,curve_arrays,compare_pair
from decompose import R

FROZEN=Path('analysis/cost_headroom/intern_finetune_20261001')

def data(pool):
    d=load_pool(pool);sp=json.loads((R/POOLS[pool][0]/'split_manifest.json').read_text())
    index={p:i for i,p in enumerate(d['ids'])}
    d['cal']=np.asarray([index[str(p)] for p in sp['calibration_problem_ids']])
    splits=[set(d[k]) for k in ['tr','cal','te']]
    if any(splits[i]&splits[j] for i in range(3) for j in range(i)):raise ValueError('Split overlap')
    old=json.loads((PROTOCOL/'manifest.json').read_text())
    template=next(r['body'] for r in old['requests'] if r['pool']==pool)
    priors=d['q'][d['tr']].mean(0)
    d['requests']=[{'state':{'problem':str(d['meta'][p]['problem_statement']),'grading_rule':GRADING[pool],
        'routes':{s:{'description':ROUTES[s],'training_accuracy':round(float(priors[j]),4)} for j,s in enumerate(d['slots'])}},
        'questions':template['questions']} for p in d['ids']]
    return d

def prepare():
    FROZEN.mkdir(parents=True,exist_ok=True);p=FROZEN/'manifest.json'
    if p.exists():return json.loads(p.read_text())
    pools={}
    for label in POOLS:
        d=data(label)
        pools[label]={'model_slots':d['slots'],'splits':{k:[d['ids'][i] for i in d[k]] for k in ['tr','cal','te']},
            'requests_sha256':hashlib.sha256(json.dumps(d['requests'],sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
            'train_cal_targets_sha256':hashlib.sha256(d['q'][np.r_[d['tr'],d['cal']]].astype('<f8').tobytes()).hexdigest()}
    result={'model':MODEL,'revision':REVISION,'temperature':TEMPERATURE,'seed':42,'epochs':5,'lr':1e-4,
        'microbatch':1,'gradient_accumulation':8,'lora_rank':16,'lora_alpha':32,'lora_dropout':.05,
        'weight_decay':.01,'warmup_fraction':.1,'max_length':8192,'pools':pools,
        'protocol':'Dataset-specific LoRA. Equal problem/route BCE on valid-draw success means; published binary candidate-logit margin divided by fixed published temperature. Epoch0 and five epochs selected by calibration NLL only. Published prompts and all five decision placeholders unchanged. Full original test and original100-problem pilot replays. Our costs fixed primary; median costs fixed secondary. No fresh expansion labels used.'}
    p.write_text(json.dumps(result,indent=2)+'\n');return result

def evaluate(pool,d,pred,out):
    untuned=json.loads((PROTOCOL/'results.json').read_text())['pools'][pool]
    idx={p:i for i,p in enumerate(d['ids'])};pilot=np.asarray([idx[p] for p in untuned['problem_ids']])
    base=d['p'].copy();base[pilot]=np.asarray(untuned['intern_successes'])
    report={}
    for subset,ii in [('full_test',d['te']),('original_pilot',pilot)]:
        arms={'ft_our_cost':(pred,d['cost']),'ours_our_cost':(d['p'],d['cost']),
              'ft_median_cost':(pred,d['median']),'ours_median_cost':(d['p'],d['median'])}
        pairs=[('ft_our_cost','ours_our_cost'),('ft_median_cost','ours_median_cost'),('ft_our_cost','ft_median_cost')]
        metrics_p={'finetuned':pred,'ours':d['p']}
        if subset=='original_pilot':
            arms['untuned_our_cost']=(base,d['cost']);arms['untuned_median_cost']=(base,d['median'])
            pairs += [('ft_our_cost','untuned_our_cost'),('ft_median_cost','untuned_median_cost')];metrics_p['untuned']=base
        curves={k:curve_arrays(p,c,d['q'],d['paid'],ii) for k,(p,c) in arms.items()}
        rng=np.random.default_rng(0);bs=[rng.integers(0,len(ii),len(ii)) for _ in range(1000)];contrasts={}
        for left,right in pairs:
            point,band=compare_pair(curves,left,right,np.arange(len(ii)))
            boot=np.asarray([compare_pair(curves,left,right,b)[0] for b in bs]);valid=boot[np.isfinite(boot)]
            contrasts[left+'_vs_'+right]={'direct_cost_saved':point,'accuracy_band':band,'ci95':np.percentile(valid,[2.5,97.5]).tolist() if len(valid) else None,'valid_bootstrap':len(valid),'bootstrap':boot.tolist()}
        metrics={}
        for name,p in metrics_p.items():
            pp=np.clip(p[ii],1e-6,1-1e-6);q=d['q'][ii]
            metrics[name]={'expected_draw_brier':float(np.mean(pp**2-2*pp*q+q)),
                'expected_draw_logloss':float(-np.mean(q*np.log(pp)+(1-q)*np.log(1-pp))),
                'mean_predicted_success':float(pp.mean()),'mean_observed_success':float(q.mean())}
        report[subset]={'n_test':len(ii),'problem_ids':[d['ids'][i] for i in ii],'metrics':metrics,'contrasts':contrasts}
        print(pool,subset,json.dumps({k:{x:y for x,y in v.items() if x!='bootstrap'} for k,v in contrasts.items()}),flush=True)
    (out/'results.json').write_text(json.dumps({'pool':pool,'subsets':report},indent=2)+'\n')
    lines=[f'# Fine-tuned Intern: {pool}','','Positive savings favor the fine-tuned success predictor. Same cost estimates per success comparison. Paired problem-bootstrap 95% intervals, 1000 draws.','','| Subset | Fine-tuned vs ours, our costs fixed | Fine-tuned vs ours, median costs fixed |','|---|---|---|']
    def fmt(v):return f"{100*v['direct_cost_saved']:+.1f}% [{100*v['ci95'][0]:+.1f}, {100*v['ci95'][1]:+.1f}]" if v['ci95'] else 'No shared accuracy band'
    for subset,x in report.items():lines.append('| '+subset+' | '+' | '.join(fmt(x['contrasts'][k]) for k in ['ft_our_cost_vs_ours_our_cost','ft_median_cost_vs_ours_median_cost'])+' |')
    lines+=['','Full original test includes the already examined pilot subset; exploratory evaluation. Train/cal/test remain disjoint. Calibration alone selected checkpoint, including epoch0. No new expansion labels, test calibration or test-selected hyperparameters. Descriptive convex-hull frontiers use outcomes to select mixtures, with pair-specific shared accuracy bands. Intervals condition on one training seed and omit training variability. Predictor overhead excluded from generation spending; no API costs. Adapters and calibration predictions saved for future untouched expansion evaluation.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')

def train(a,m):
    import fcntl,torch
    from huggingface_hub import snapshot_download
    from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required')
    torch.set_num_threads(8);torch.manual_seed(m['seed']);np.random.seed(m['seed'])
    out=a.out/a.pool.lower().replace('-','_');out.mkdir(parents=True,exist_ok=True)
    lock=(out/'training.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    d=data(a.pool);frozen=m['pools'][a.pool]
    actual=hashlib.sha256(json.dumps(d['requests'],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    if actual!=frozen['requests_sha256']:raise ValueError('Frozen requests changed')
    if hashlib.sha256(d['q'][np.r_[d['tr'],d['cal']]].astype('<f8').tobytes()).hexdigest()!=frozen['train_cal_targets_sha256']:raise ValueError('Training/calibration labels changed')
    checkpoint=snapshot_download(MODEL,revision=REVISION,token=False)
    code=Path(checkpoint)/'inference.py';spec=importlib.util.spec_from_file_location('intern_training_inference',code)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    dtype='bfloat16' if torch.cuda.is_bf16_supported() else 'float32'
    engine=module.DecisionEngine(checkpoint=checkpoint,temperature=TEMPERATURE,max_length=m['max_length'],device='cuda',dtype=dtype)
    encoded=[]
    for request in d['requests']:
        compiled,batch,positions=engine.backend.encode(module.validate_request(request))
        if list(compiled.fields)!=d['slots']:raise ValueError('Route order mismatch')
        encoded.append((batch,positions))
    token_ids=[engine.tokenizer.encode(s,add_special_tokens=False) for s in ['A','B']]
    if any(len(t)!=1 for t in token_ids):raise ValueError('Non-single-token options')
    no,yes=[t[0] for t in token_ids]
    language_linear={'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj',
                     'in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','out_proj'}
    targets=[name for name,layer in engine.backend.model.named_modules() if isinstance(layer,torch.nn.Linear)
             and '.language_model.' in name and name.rsplit('.',1)[-1] in language_linear]
    if not targets:raise ValueError('No language LoRA targets matched')
    engine.backend.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    engine.backend.model.enable_input_require_grads()
    model=get_peft_model(engine.backend.model,LoraConfig(r=m['lora_rank'],lora_alpha=m['lora_alpha'],
        lora_dropout=m['lora_dropout'],target_modules=targets,bias='none'))
    engine.backend.model=model;parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=m['lr'],weight_decay=m['weight_decay'])
    updates=math.ceil(len(d['tr'])/m['gradient_accumulation']);total=updates*m['epochs'];warmup=max(1,int(total*m['warmup_fraction']))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min(1.,(step+1)/warmup)*max(0.,(total-step)/max(1,total-warmup)) if step>=warmup else (step+1)/warmup)
    def forward(i):
        batch,positions=encoded[i]
        output=model(**batch.to('cuda'),use_cache=False,logits_to_keep=positions.to('cuda')).logits[0]
        return (output[:,yes].float()-output[:,no].float())/TEMPERATURE
    def predict(ii):
        model.eval();values=[]
        with torch.no_grad():
            for i in ii:values.append(torch.sigmoid(forward(int(i))).cpu().numpy())
        return np.asarray(values)
    def loss_metric(p,ii):
        pp=np.clip(p,1e-6,1-1e-6);q=d['q'][ii]
        return float(-np.mean(q*np.log(pp)+(1-q)*np.log(1-pp)))
    start=time.time();history=[];best=float('inf');selected=None
    metadata={'pool':a.pool,'protocol':m,'lora_targets':targets,'trainable_parameters':sum(p.numel() for p in parameters),
        'total_parameters':sum(p.numel() for p in model.parameters()),'dtype':dtype,'gpu':torch.cuda.get_device_name(),
        'torch':torch.__version__,'inference_sha256':hashlib.sha256(code.read_bytes()).hexdigest(),
        'max_input_tokens':max(batch['input_ids'].shape[-1] for batch,_ in encoded)}
    (out/'execution.json').write_text(json.dumps(metadata,indent=2)+'\n')
    rng=np.random.default_rng(m['seed'])
    for epoch in range(m['epochs']+1):
        train_loss=None
        if epoch:
            model.train();order=rng.permutation(d['tr']);losses=[]
            for start_batch in range(0,len(order),m['gradient_accumulation']):
                group=order[start_batch:start_batch+m['gradient_accumulation']];optimizer.zero_grad(set_to_none=True)
                for i in group:
                    logits=forward(int(i));target=torch.as_tensor(d['q'][i],dtype=torch.float32,device='cuda')
                    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,target)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                    (loss/len(group)).backward();losses.append(float(loss.detach()))
                torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step();scheduler.step()
                if (start_batch//m['gradient_accumulation'])%20==0:print(json.dumps({'pool':a.pool,'epoch':epoch,'trained_problems':min(start_batch+len(group),len(order)),'mean_loss':float(np.mean(losses))}),flush=True)
            train_loss=float(np.mean(losses))
        cal_pred=predict(d['cal']);cal_loss=loss_metric(cal_pred,d['cal'])
        if not math.isfinite(cal_loss):raise RuntimeError('Nonfinite calibration loss')
        row={'epoch':epoch,'train_loss':train_loss,'calibration_nll':cal_loss,'elapsed_seconds':time.time()-start}
        history.append(row);print(json.dumps({'pool':a.pool,**row}),flush=True)
        if cal_loss<best:
            best=cal_loss;selected=epoch
            state={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
            temp=out/'best_adapter.tmp.pt';torch.save(state,temp);temp.replace(out/'best_adapter.pt')
            np.savez_compressed(out/'calibration_predictions.npz',problem_ids=np.asarray([d['ids'][i] for i in d['cal']]),model_slots=np.asarray(d['slots']),p_successes=cal_pred)
        (out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    set_peft_model_state_dict(model,torch.load(out/'best_adapter.pt',map_location='cpu',weights_only=True))
    model.save_pretrained(out/'adapter');engine.tokenizer.save_pretrained(out/'adapter')
    predictions=d['p'].copy();predictions[d['te']]=predict(d['te'])
    np.savez_compressed(out/'test_predictions.npz',problem_ids=np.asarray([d['ids'][i] for i in d['te']]),model_slots=np.asarray(d['slots']),p_successes=predictions[d['te']])
    (out/'training.json').write_text(json.dumps({'pool':a.pool,'selected_epoch':selected,'best_calibration_nll':best,'elapsed_seconds':time.time()-start,'api_calls':0},indent=2)+'\n')
    evaluate(a.pool,d,predictions,out)
    (out/'status.json').write_text(json.dumps({'complete':True,'selected_epoch':selected,'test_problems':len(d['te'])})+'\n')
    lock.close()

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--pool',choices=POOLS)
    ap.add_argument('--out',type=Path,default=Path('/mnt/llmd/results/exps/aristides/reason/intern_finetune_20261001'))
    ap.add_argument('--prepare-only',action='store_true');a=ap.parse_args();m=prepare()
    if a.prepare_only:return
    if not a.pool:ap.error('--pool required for training')
    train(a,m)

if __name__=='__main__':main()
