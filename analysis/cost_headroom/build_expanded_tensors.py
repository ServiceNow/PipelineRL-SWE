"""Assemble frozen expansion generations into tensor files for original-head inference."""
import argparse,json
from pathlib import Path
import numpy as np
R=Path('/mnt/llmd/results/exps/aristides/reason')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['mmlupro','omni500'],required=True);a=ap.parse_args()
    label=a.dataset;oldname='mmlupro_tensors' if label=='mmlupro' else 'omni500_tensors'
    source=R/'math_expand_20261001'/label;out=R/'expanded_eval_20261001'/label;out.mkdir(parents=True,exist_ok=True)
    status=source/'COMPLETE.json'
    if not status.exists():raise RuntimeError('Generation expansion has no COMPLETE.json')
    plan=json.loads((source/'collection_plan.json').read_text());tasks=[json.loads(x) for x in (Path('analysis/cost_headroom/expansion_20261001')/f'{label}_tasks.jsonl').read_text().splitlines()]
    n=len(tasks);old=np.load(R/oldname/'tensors.npz',allow_pickle=True);oldids=[str(x) for x in old['problem_ids']];slots=[str(x) for x in old['model_slots']]
    if n!=plan['estimates'][label]['new_problems']:raise ValueError('Wrong expansion size')
    rows={s:{} for s in slots}
    for j,s in enumerate(slots):
        f=source/f'{s}_d0.jsonl'
        for line in f.open():
            r=json.loads(line)
            if r.get('finish_reason')=='error':continue
            rows[s][r['problem_id']]=r
        if set(rows[s])!={t['problem_id'] for t in tasks}:raise RuntimeError(f'{s}: incomplete expansion responses')
    if len(set(oldids)&{t['problem_id'] for t in tasks}):raise ValueError('Expansion overlaps original IDs')
    dct={k:old[k] for k in old.files}
    shapes={'final_outcome':(n,len(slots),4),'execution_outcome':(n,len(slots),4),'weak_verifier_outcome':(n,len(slots),4),'valid':(n,len(slots),4),'prompt_tokens':(n,len(slots),4),'completion_tokens':(n,len(slots),4)}
    for name,shape in shapes.items():
        dtype=bool if name.endswith('outcome') or name=='valid' else old[name].dtype
        tail=np.zeros(shape,dtype=dtype)
        if name in {'final_outcome','execution_outcome','weak_verifier_outcome','valid','prompt_tokens','completion_tokens'}:
            for i,t in enumerate(tasks):
                for j,s in enumerate(slots):
                    r=rows[s][t['problem_id']]
                    tail[i,j,0]=True if name=='valid' else (bool(r['resolved']) if name.endswith('outcome') else (int(r.get('prompt_tokens') or 0) if name=='prompt_tokens' else int(r.get('completion_tokens') or 0)))
        dct[name]=np.concatenate([old[name],tail],axis=0)
    dct['problem_ids']=np.asarray(oldids+[t['problem_id'] for t in tasks],dtype=str)
    tmp=out/'tensors.tmp.npz';np.savez_compressed(tmp,**dct);tmp.replace(out/'tensors.npz')
    oldprobs=[json.loads(x) for x in (R/oldname/'problems.jsonl').read_text().splitlines()]
    newprobs=[{'problem_id':t['problem_id'],'problem_statement':t['problem'],'difficulty':float(t.get('difficulty',0)),
        'subject':t.get('subject',''),'answer':t['answer'],'kind':t.get('kind','')} for t in tasks]
    (out/'problems.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in oldprobs+newprobs))
    (out/'split_manifest.json').write_text((R/oldname/'split_manifest.json').read_text())
    (out/'expansion_manifest.json').write_text(json.dumps({'dataset':label,'n_new':n,'new_problem_ids':[t['problem_id'] for t in tasks],
       'stratum_weights':plan['evaluation_stratum_weights'],
       'policy':'Original train/calibration IDs only. Expansion examples are held out and not added to either split.'},indent=2)+'\n')
    print(json.dumps({'dataset':label,'old':len(oldids),'new':n,'valid_new':int(dct['valid'][len(oldids):].sum()),'out':str(out)}),flush=True)
if __name__=='__main__':main()
