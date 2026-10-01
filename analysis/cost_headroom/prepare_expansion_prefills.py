"""Freeze raw-problem prefill requests for held-out expansion IDs (no labels included)."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path('analysis/cost_headroom/expansion_20261001')
OUT=Path('/mnt/llmd/results/exps/aristides/reason/expansion_prefills_20261001')

def prepare(label):
    OUT.mkdir(parents=True,exist_ok=True)
    src=ROOT/f'{label}_tasks.jsonl';rows=[json.loads(x) for x in src.read_text().splitlines()]
    expected={'mmlupro':6500,'omni500':1000}[label]
    if len(rows)!=expected:raise ValueError(f'{label}: expected {expected} tasks, got {len(rows)}')
    # Match the original feature inputs: raw problem_statement, before the reasoning/MMLU
    # generation wrapper. Answer choices are not in the original problems.jsonl or features.
    requests=[{'problem_id':r['problem_id'],'prompt':r['problem']} for r in rows]
    text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in requests)
    path=OUT/f'{label}_prompts.jsonl';digest=hashlib.sha256(text.encode()).hexdigest()
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Existing prompts do not match frozen tasks')
    if not path.exists():path.write_text(text)
    meta={'dataset':label,'task_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'prompt_sha256':digest,
          'n':len(rows),'model':{'mmlupro':'Qwen/Qwen3-4B-Instruct-2507','omni500':'Qwen/Qwen3-4B-Thinking-2507'}[label],
          'system_prompt':'pool_activation_probe.py default SYSTEM','readout':'same eight relative layers, last/mean, as frozen paper features',
          'limit_tokens':8192,'request_fields':'problem_id and raw problem text only; no answers, generation outputs or split labels'}
    (OUT/f'{label}_manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['mmlupro','omni500'],required=True);prepare(p.parse_args().dataset)
