"""What the router's own prefill costs (NEW_PATH 4.A.86). Per pool with a probe prompts file: Qwen3-4B-Instruct-2507 prompt length
(chat template + the system prompt the probe used, as pool_activation_probe.py extract), for every problem; and on one GPU, the
latency of the forward pass that produces the features (bf16, output_hidden_states, batch 1, as extracted) on 100 random problems
per pool after a warm-up, plus batched throughput (batch 16, padded). Writes prefill_overhead.json to OUT_DIR. The $ comparison with
routing savings is done in the paper text from this file and tmlr_strengthen deploy outputs.
Usage (1 GPU): OUT_DIR=... python prefill_overhead.py
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, "/home/toolkit/PipelineRL-SWE")
from pipelinerl.swe.scripts.livecodebench.pool_activation_probe import SYSTEM

R = Path("/mnt/llmd/results/exps/aristides/reason"); OUT = Path(os.environ.get("OUT_DIR", "."))
MATH = (R / "math_system_prompt.txt").read_text()
POOLS = {"AIME": ("aime_probe_prompts.jsonl", MATH), "Omni500": ("omni500_probe_prompts.jsonl", MATH), "MMLU-Pro": ("mmlupro_probe_prompts.jsonl", MATH),
         "SuperGPQA": ("supergpqa_probe_prompts.jsonl", MATH), "BBEH": ("bbeh_probe_prompts.jsonl", MATH), "APPS": ("apps_probe_prompts.jsonl", SYSTEM),
         "BCB": ("bcb_probe_prompts.jsonl", SYSTEM)}
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
try:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda", output_hidden_states=True).eval()
except TypeError:                                                               # older transformers spell it torch_dtype
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda", output_hidden_states=True).eval()
res = {"gpu": torch.cuda.get_device_name(0), "model": MODEL}
texts_of = lambda f, sysmsg: [tok.apply_chat_template([{"role": "system", "content": sysmsg}, {"role": "user", "content": json.loads(l)["prompt"]}],
                                                      tokenize=False, add_generation_prompt=True) for l in open(R / f) if l.strip()]
with torch.no_grad():
    for _ in range(5):                                                          # warm-up
        model(**tok("warm up " * 200, return_tensors="pt").to("cuda"))
    for P, (f, sysmsg) in POOLS.items():
        if not (R / f).exists():
            continue
        texts = texts_of(f, sysmsg); ntok = np.array([len(tok(t)["input_ids"]) for t in texts])
        sample = [texts[i] for i in np.random.default_rng(0).choice(len(texts), min(100, len(texts)), replace=False)]
        lat = []
        for t in sample:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=8192).to("cuda"); torch.cuda.synchronize(); t0 = time.perf_counter()
            model(**enc); torch.cuda.synchronize(); lat.append(time.perf_counter() - t0)
        torch.cuda.synchronize(); t0 = time.perf_counter(); done = 0
        for i in range(0, len(sample), 16):
            enc = tok(sample[i:i + 16], return_tensors="pt", padding=True, truncation=True, max_length=8192).to("cuda")
            model(**enc); done += int(enc["attention_mask"].sum())
        torch.cuda.synchronize(); thr = done / (time.perf_counter() - t0)
        res[P] = dict(n=len(texts), tokens_mean=float(ntok.mean()), tokens_median=float(np.median(ntok)), tokens_p90=float(np.percentile(ntok, 90)),
                      latency_ms_median=float(np.median(lat) * 1e3), latency_ms_p90=float(np.percentile(lat, 90) * 1e3), batched_tokens_per_s=float(thr))
        print(P, json.dumps(res[P]), flush=True)
json.dump(res, open(OUT / "prefill_overhead.json", "w"), indent=1)
print("DONE", flush=True)
