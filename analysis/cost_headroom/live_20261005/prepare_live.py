"""Live run (NEW_PATH 4.A.52), step 1: freeze 1,000 MMLU-Pro problems never used anywhere (not in the original 1,000, the 6,500 fresh,
or by normalized text), stratified to the original subject mix, seed 20261005. Writes tasks + prefill prompts (full solving prompt
with options; no answers) to /mnt/llmd/results/exps/aristides/reason/live_run_20261005/. Usage: python prepare_live.py
"""
import hashlib, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[2] / "pipelinerl/swe/scripts/math_pool"))
from prepare_expansion import norm, read, sample, R
from reasoning_datasets import _mc_prompt
from datasets import Dataset

OUT = R / "live_run_20261005"; N = 1000; SEED = 20261005
old = read(R / "math_pool" / "mmlupro" / "problems.jsonl"); fresh = read(HERE.parent / "expansion_20261001" / "mmlupro_tasks.jsonl")
used_ids = {r["problem_id"] for r in old} | {r["problem_id"] for r in fresh}
used_txt = {norm(r["problem_statement"]) for r in old} | {norm(r["problem"]) for r in fresh}
ds = Dataset.from_file(str(next(Path("/home/toolkit/.cache/huggingface/datasets/TIGER-Lab___mmlu-pro").rglob("mmlu-pro-test.arrow"))))
seen, eligible = set(used_txt), []
for r in ds:
    t = dict(problem_id=f"mmlup_{r['question_id']}", problem=r["question"], prompt=_mc_prompt(r["question"], r["options"]), answer=r["answer"],
             difficulty=0., subject=r["category"], kind="mc")
    x = norm(t["problem"])
    if t["problem_id"] in used_ids or x in seen:
        continue
    seen.add(x); eligible.append(t)
sel, alloc = sample(eligible, N, lambda r: r["subject"], old, SEED)
assert len({r["problem_id"] for r in sel}) == N and not ({norm(r["problem"]) for r in sel} & used_txt)
OUT.mkdir(parents=True, exist_ok=True)
tasks = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sel)
prompts = "".join(json.dumps({"problem_id": r["problem_id"], "prompt": r["prompt"]}, ensure_ascii=False) + "\n" for r in sel)
for name, text in (("tasks.jsonl", tasks), ("prompts.jsonl", prompts)):
    p = OUT / name
    if p.exists() and p.read_text() != text:
        raise SystemExit(f"{p} exists and differs; refusing to overwrite a frozen sample")
    p.write_text(text)
meta = dict(n=N, seed=SEED, eligible=len(eligible), strata=alloc, tasks_sha256=hashlib.sha256(tasks.encode()).hexdigest(),
            excluded="original 1,000 + fresh 6,500 MMLU-Pro ids and normalized texts")
(OUT / "manifest.json").write_text(json.dumps(meta, indent=1) + "\n"); print(json.dumps(meta)[:400])
