"""CARROT-style trained success/cost predictors with Jina-137M, original splits.
Adaptations: Jina mean pooling, existing calibration split, train-standardized
raw output lengths, six epochs. Not a reproduction of CARROT-RoBERTa.
"""
import argparse
import contextlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from carrot_compare import POOLS, R, run as evaluate

ENCODER='jinaai/jina-embeddings-v2-base-code'


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError('GPU is required for this fine-tuning run')
    torch.set_num_threads(8)
    name,_=POOLS[args.pool]; folder=R/name
    out=args.out/args.pool.lower().replace('-','_');out.mkdir(parents=True,exist_ok=True)
    t=np.load(folder/'tensors.npz',allow_pickle=True)
    ids=[str(p) for p in t['problem_ids']];slots=[str(s) for s in t['model_slots']]
    pi={p:i for i,p in enumerate(ids)}
    sp=json.loads((folder/'split_manifest.json').read_text())
    tr=np.asarray([pi[str(p)] for p in sp['train_problem_ids']])
    cal=np.asarray([pi[str(p)] for p in sp['calibration_problem_ids']])
    if set(tr)&set(cal):raise ValueError('Training and calibration overlap')
    v=t['valid'].astype(bool);n=v.sum(2)
    if not (n>0).all():raise ValueError('Missing route labels')
    q=np.where(v,t['final_outcome'],0).sum(2)/n
    length=np.where(v,t['completion_tokens'],0).sum(2)/n
    mu=length[tr].mean(0);sd=np.maximum(length[tr].std(0),1)
    z=(length-mu)/sd
    meta={str(r['problem_id']):r for r in map(json.loads,(folder/'problems.jsonl').open())}
    texts=[str(meta[p]['problem_statement']) for p in ids]
    # Do not import optional DeepSpeed; this uses a plain PyTorch loop.
    import transformers.integrations.deepspeed as ds
    ds.is_deepspeed_available=lambda:False
    from transformers import AutoTokenizer,AutoModel
    tok=AutoTokenizer.from_pretrained(ENCODER)
    histories={};predictions={};start=time.time()
    for task,targets in [('success',q),('cost',z)]:
        torch.manual_seed(args.seed);np.random.seed(args.seed)
        enc=AutoModel.from_pretrained(ENCODER,trust_remote_code=True).cuda()
        head=torch.nn.Sequential(torch.nn.Dropout(.1),torch.nn.Linear(enc.config.hidden_size,enc.config.hidden_size),
                                 torch.nn.Tanh(),torch.nn.Dropout(.1),torch.nn.Linear(enc.config.hidden_size,len(slots))).cuda()
        params=list(enc.parameters())+list(head.parameters())
        optimizer=torch.optim.AdamW(params,lr=args.lr,weight_decay=.01)
        total=math.ceil(len(tr)/args.batch)*args.epochs;warmup=max(1,int(.1*total))
        def lr_scale(step):
            return step/warmup if step<warmup else max(0,(total-step)/max(1,total-warmup))
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_scale)
        amp_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else None
        def autocast():
            return torch.autocast('cuda',dtype=amp_dtype) if amp_dtype else contextlib.nullcontext()
        def forward(ii):
            batch=tok([texts[i] for i in ii],padding=True,truncation=True,max_length=args.max_length,
                      return_tensors='pt').to('cuda')
            hidden=enc(**batch).last_hidden_state
            mask=batch['attention_mask'].unsqueeze(-1).to(hidden.dtype)
            pooled=(hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
            return head(pooled)
        def predict(ii):
            enc.eval();head.eval();pred=[]
            with torch.no_grad(),autocast():
                for j in range(0,len(ii),args.batch):pred.append(forward(ii[j:j+args.batch]).float().cpu().numpy())
            return np.concatenate(pred)
        best=float('inf');history=[]
        for epoch in range(args.epochs):
            enc.train();head.train();order=np.random.permutation(tr);losses=[]
            for j in range(0,len(order),args.batch):
                ii=order[j:j+args.batch];y=torch.tensor(targets[ii],dtype=torch.float32,device='cuda')
                with autocast():
                    logits=forward(ii).float()
                    loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,y) if task=='success'
                          else torch.nn.functional.mse_loss(logits,y))
                if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step();scheduler.step()
                losses.append(float(loss.detach()))
            pv=torch.tensor(predict(cal));yv=torch.tensor(targets[cal],dtype=torch.float32)
            vl=float(torch.nn.functional.binary_cross_entropy_with_logits(pv,yv) if task=='success'
                     else torch.nn.functional.mse_loss(pv,yv))
            row={'epoch':epoch+1,'train_loss':float(np.mean(losses)),'calibration_loss':vl,
                 'elapsed_seconds':time.time()-start};history.append(row)
            print(json.dumps({'pool':args.pool,'task':task,**row}),flush=True)
            if not math.isfinite(vl):raise RuntimeError('Non-finite calibration loss')
            if vl<best:
                best=vl
                checkpoint={'encoder_state':{k:w.detach().cpu() for k,w in enc.state_dict().items()},
                            'head_state':{k:w.detach().cpu() for k,w in head.state_dict().items()},
                            'encoder':ENCODER,'model_revision':getattr(enc.config,'_commit_hash',None),
                            'pool':args.pool,'task':task,'seed':args.seed,'epoch':epoch+1,'model_slots':slots,
                            'max_length':args.max_length,'output_mean':mu,'output_std':sd}
                tmp=out/f'{task}.tmp.pt';torch.save(checkpoint,tmp);tmp.replace(out/f'{task}.pt')
                raw=predict(np.arange(len(ids)))
                predictions[task]=(1/(1+np.exp(-np.clip(raw,-40,40))) if task=='success'
                                   else np.maximum(raw*sd+mu,0))
                histories[task]={'selected_epoch':epoch+1,'calibration_loss':vl}
            (out/f'{task}_history.json').write_text(json.dumps(history,indent=2)+'\n')
        del enc,head,optimizer,params
        torch.cuda.empty_cache()
    candidate=out/'jina_predictions.npz'
    np.savez_compressed(candidate,problem_ids=np.asarray(ids),model_slots=np.asarray(slots),encoder=ENCODER,
                        carrot_success=predictions['success'],carrot_output_tokens=predictions['cost'])
    metadata={'variant':'CARROT-style-Jina-137M','encoder':ENCODER,'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              'selected_heads':histories,'elapsed_seconds':time.time()-start,
              'protocol':'Separate fine-tuned mean-pooled encoders; BCE on valid-draw success means; MSE on train-standardized raw output-token means; checkpoint selected on existing calibration only. Six epochs, one seed; no test-based configuration choice. Negative output estimates clipped to zero.'}
    (out/'training.json').write_text(json.dumps(metadata,indent=2)+'\n')
    args.candidate=candidate;args.variant='CARROT-style-Jina-137M'
    # Bootstrap seed matches the completed kNN comparison.
    training_seed=args.seed;args.seed=0
    try:evaluate(args,args.pool)
    finally:args.seed=training_seed


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pool',choices=POOLS,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--epochs',type=int,default=6)
    ap.add_argument('--batch',type=int,default=8)
    ap.add_argument('--max-length',type=int,default=1024)
    ap.add_argument('--lr',type=float,default=2e-5)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--bootstrap',type=int,default=1000)
    args=ap.parse_args()
    if min(args.epochs,args.batch,args.max_length,args.bootstrap)<1:ap.error('Counts must be positive')
    train(args)

if __name__=='__main__':main()
