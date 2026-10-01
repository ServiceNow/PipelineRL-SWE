"""Prepare fixed, disjoint expansion samples and empirical collection cost estimates."""
import hashlib,json,random,re,sys,unicodedata
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import requests
from datasets import Dataset
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'pipelinerl/swe/scripts/math_pool'))
from reasoning_datasets import _mc_prompt

ROOT=Path(__file__).parent/'expansion_20261001'
R=Path('/mnt/llmd/results/exps/aristides/reason')
DRAWS={'oss20lo':4,'oss20md':3,'dsv4f':3,'oss120md':2,'oss120hi':2}
CAPS={'oss20lo':(.04,.15),'oss20md':(.04,.15),'dsv4f':(.14,.28),'oss120md':(.15,.6),'oss120hi':(.15,.6)}

def norm(s):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',s)).strip()
def read(path):return [json.loads(l) for l in path.read_text().splitlines()]
def sample(items,n,key,reference,seed):
    # Allocate proportional to the EXISTING sample's strata, then deterministic
    # largest-remainder rounding, capacity-aware redistribution within strata.
    groups=defaultdict(list)
    for t in items:groups[key(t)].append(t)
    counts=Counter(key(t) for t in reference)
    rng=random.Random(seed)
    for v in groups.values():rng.shuffle(v)
    targets={k:n*v/sum(counts.values()) for k,v in counts.items()}
    allocations={k:min(int(v),len(groups[k])) for k,v in targets.items()}
    while sum(allocations.values())<n:
        eligible=[k for k in targets if allocations[k]<len(groups[k])]
        if not eligible:raise ValueError('Insufficient capacity in reference strata')
        k=max(eligible,key=lambda k:(targets[k]-allocations[k],str(k)))
        allocations[k]+=1
    result=[t for k in sorted(allocations,key=str) for t in groups[k][:allocations[k]]]
    rng.shuffle(result)
    return result,{str(k):v for k,v in allocations.items()}

def main():
 ROOT.mkdir(exist_ok=True)
 old={name:read(R/'math_pool'/name/'problems.jsonl') for name in ['mmlupro','omni500']}
 cached=next(Path('/home/toolkit/.cache/huggingface/datasets/TIGER-Lab___mmlu-pro').rglob('mmlu-pro-test.arrow'))
 ds=Dataset.from_file(str(cached))
 mmlu=[dict(problem_id=f"mmlup_{r['question_id']}",problem=r['question'],prompt=_mc_prompt(r['question'],r['options']),answer=r['answer'],difficulty=0.,subject=r['category'],kind='mc') for r in ds]
 meta=requests.get('https://huggingface.co/api/datasets/KbsdJames/Omni-MATH',timeout=30);meta.raise_for_status();revision=meta.json()['sha']
 response=requests.get(f'https://huggingface.co/datasets/KbsdJames/Omni-MATH/resolve/{revision}/test.jsonl',timeout=120);response.raise_for_status()
 full=[json.loads(l) for l in response.text.splitlines()]
 omni=[dict(problem_id='omni_full_'+hashlib.sha256(norm(r['problem']).encode()).hexdigest()[:20],problem=r['problem'],answer=r['answer'],difficulty=float(r['difficulty']),subject=str(r['domain'])[:120],source=r.get('source')) for r in full]
 provenance={'mmlupro_source':'TIGER-Lab/MMLU-Pro cached test split','mmlupro_cache_fingerprint':ds._fingerprint,'omni_source':'KbsdJames/Omni-MATH','omni_revision':revision,'omni_source_sha256':hashlib.sha256(response.content).hexdigest(),'seed':20261001,'draws':DRAWS,'price_caps_per_million':CAPS,'evaluation':'All new problems are held-out evaluation. Preserve existing training/calibration splits and configuration choices; no significance-based stopping.'}
 estimates={}
 for name,tasks,n in [('mmlupro',mmlu,2000),('omni500',omni,1000)]:
  oldids={r['problem_id'] for r in old[name]};oldtexts={norm(r['problem_statement']) for r in old[name]}
  seen=set(oldtexts);eligible=[];overlap=0;duplicates=0
  for t in tasks:
   text=norm(t['problem'])
   if t['problem_id'] in oldids or text in oldtexts:overlap+=1;continue
   if text in seen:duplicates+=1;continue
   seen.add(text);eligible.append(t)
  strat=(lambda r:r['subject']) if name=='mmlupro' else (lambda r:round(r['difficulty']))
  selected,allocation=sample(eligible,n,strat,old[name],20261001)
  assert len({r['problem_id'] for r in selected})==n
  assert not ({norm(r['problem']) for r in selected}&oldtexts)
  content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected)
  (ROOT/f'{name}_tasks.jsonl').write_text(content)
  tensor=np.load(R/(name+'_tensors')/'tensors.npz',allow_pickle=True);valid=tensor['valid'];detail={}
  for m,slot in enumerate(tensor['model_slots']):
   slot=str(slot);pt=float(tensor['prompt_tokens'][:,m][valid[:,m]].mean());ct=float(tensor['completion_tokens'][:,m][valid[:,m]].mean())
   pin,pout=CAPS[slot];detail[slot]={'calls':n*DRAWS[slot],'mean_prompt_tokens':pt,'mean_completion_tokens':ct,'price_cap_estimate_usd':n*DRAWS[slot]*(pt*pin+ct*pout)/1e6}
  estimate=sum(v['price_cap_estimate_usd'] for v in detail.values())
  estimates[name]={'new_problems':n,'existing_problems':len(old[name]),'combined_problems':len(old[name])+n,'calls':n*sum(DRAWS.values()),'source_rows':len(tasks),'excluded_existing_rows':overlap,'excluded_source_duplicates':duplicates,'eligible_rows':len(eligible),'sample_sha256':hashlib.sha256(content.encode()).hexdigest(),'strata_counts':allocation,'routes':detail,'estimated_usd_at_provider_price_caps':estimate,'planning_usd_with_25pct_buffer':1.25*estimate}
 provenance['estimates']=estimates
 (ROOT/'plan.json').write_text(json.dumps(provenance,indent=2))
 print(json.dumps(estimates,indent=2))
if __name__=='__main__':main()
