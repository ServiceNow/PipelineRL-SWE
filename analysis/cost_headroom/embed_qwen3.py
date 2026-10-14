"""GPU (NEW_PATH 4.A.101): Qwen3-Embedding-{4B,8B} text embeddings of every problem of the nine pools, for size-matched text
baselines (MixLLM-style, kNN, ridge, and a fully text-based router). As fresh_embeddings.py: problem_statement, last-token pooling
with left padding, L2-normalized, max 4096 tokens, bf16. Writes <pool dir>/qwen3emb<size>.npz (problem_ids, emb).
Usage: python embed_qwen3.py 4B|8B [POOL_DIRS]
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

assert torch.cuda.is_available(), "GPU job"
SZ = sys.argv[1]; R = Path("/mnt/llmd/results/exps/aristides/reason_pinned")
DIRS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["pool_v2_tensors_5rung", "apps_tensors", "bcb_tensors_5r", "cc_tensors", "aime_tensors",
                                                         "supergpqa_tensors", "bbeh_tensors", "expanded_eval_20261001/omni500", "expanded_eval_20261001/mmlupro"]
name = f"Qwen/Qwen3-Embedding-{SZ}"; tok = AutoTokenizer.from_pretrained(name, padding_side="left")
m = AutoModel.from_pretrained(name, torch_dtype=torch.bfloat16).eval().cuda()
for dd in DIRS:
    out_f = R / dd / f"qwen3emb{SZ.lower()}.npz"
    if out_f.exists():
        print(dd, "exists", flush=True); continue
    rows = [json.loads(l) for l in (R / dd / "problems.jsonl").read_text().splitlines() if l.strip()]
    ids = [str(r["problem_id"]) for r in rows]; texts = [str(r.get("problem_statement", "")) for r in rows]
    order = np.argsort([len(t) for t in texts]); E = np.zeros((len(texts), m.config.hidden_size), np.float32)
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            ii = order[i:i + 16]; b = tok([texts[j] for j in ii], padding=True, truncation=True, max_length=4096, return_tensors="pt").to("cuda")
            E[ii] = torch.nn.functional.normalize(m(**b).last_hidden_state[:, -1].float(), dim=-1).cpu().numpy()
    np.savez_compressed(out_f, problem_ids=np.array(ids), emb=E); print(dd, len(ids), "embedded", E.shape, flush=True)
print("ALL DONE", flush=True)
