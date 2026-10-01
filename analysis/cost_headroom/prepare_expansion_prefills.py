"""Freeze raw-problem prefill requests for held-out expansion IDs (no labels included)."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path('analysis/cost_headroom/expansion_20261001')
OUT=Path('/mnt/llmd/results/exps/aristides/reason/expansion_prefills_verified_20261001')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'pipelinerl/swe/scripts/math_pool'))
from collect_math_pool import PROMPT

def prepare(label):
    OUT.mkdir(parents=True,exist_ok=True)
    src=ROOT/f'{label}_tasks.jsonl';rows=[json.loads(x) for x in src.read_text().splitlines()]
    expected={'mmlupro':6500,'omni500':1000}[label]
    if len(rows)!=expected:raise ValueError(f'{label}: expected {expected} tasks, got {len(rows)}')
    # Original probe_prompts.jsonl contains the full solving prompt. In particular
    # MMLU answer choices are model inputs; gold answers never enter encoder inputs.
    if label=='mmlupro' and any(not r.get('prompt') for r in rows):
        raise ValueError('A multiple-choice task has no full prompt/options')
    requests=[{'problem_id':r['problem_id'],'prompt':r.get('prompt') or PROMPT.format(problem=r['problem'])} for r in rows]
    text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in requests)
    path=OUT/f'{label}_prompts.jsonl';digest=hashlib.sha256(text.encode()).hexdigest()
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Existing prompts do not match frozen tasks')
    if not path.exists():path.write_text(text)
    meta={'dataset':label,'task_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'prompt_sha256':digest,
          'n':len(rows),'model':{'mmlupro':'Qwen/Qwen3-4B-Instruct-2507','omni500':'Qwen/Qwen3-4B-Thinking-2507'}[label],
          'system_prompt':'You are a helpful assistant. (matched to original cached feature metadata)','readout':'same eight relative layers, last/mean, as frozen paper features',
          'limit_tokens':8192,'request_fields':'problem_id and full solving prompt, including MC options; no gold answers, generation outputs or split labels',
          'prompt_protocol':'Same full user prompts as original *_probe_prompts.jsonl and generation collection'}
    (OUT/f'{label}_manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['mmlupro','omni500'],required=True);prepare(p.parse_args().dataset)
