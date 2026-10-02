"""GPU: text embeddings of every problem in the expanded pools (original + fresh) for the fresh-set CARROT and MixLLM-style
baselines and the representation x head grid (NEW_PATH 4.A.43/4.A.47). Also Qwen3-Embedding-8B (last-token pooling, L2-normalized,
as on its model card): a text EMBEDDING model at least as large as our 4B prefill encoder. MiniLM-L12-v2 (CARROT's kNN encoder, sentence-transformers, as in carrot_compare.py) and jina-embeddings-v2-base-code
(our MixLLM-style encoder, max_length 1024). Text = problem_statement from expanded_eval_20261001/<ds>/problems.jsonl, the same
field the original-pool baselines embedded. Writes expanded_eval_20261001/<ds>/text_embeddings.npz.
"""
import json
from pathlib import Path
import numpy as np
import torch

R = Path("/mnt/llmd/results/exps/aristides/reason") / "expanded_eval_20261001"
assert torch.cuda.is_available(), "GPU job"
from sentence_transformers import SentenceTransformer
from transformers import AutoModel
mini = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2", device="cuda")
jina = AutoModel.from_pretrained("jinaai/jina-embeddings-v2-base-code", trust_remote_code=True).eval().cuda()
from transformers import AutoTokenizer
qtok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-8B", padding_side="left")
qwen = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-8B", torch_dtype=torch.bfloat16).eval().cuda()
for ds in ("omni500", "mmlupro"):
    rows = [json.loads(l) for l in (R / ds / "problems.jsonl").read_text().splitlines()]
    ids = [r["problem_id"] for r in rows]; texts = [str(r.get("problem_statement", "")) for r in rows]
    E1 = mini.encode(texts, batch_size=128, show_progress_bar=False)            # as carrot_compare.py (cosine kNN)
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            out.append(np.asarray(jina.encode(texts[i:i + 64], max_length=1024, device="cuda")))
    E3 = []
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            b = qtok(texts[i:i + 16], padding=True, truncation=True, max_length=4096, return_tensors="pt").to("cuda")
            h = qwen(**b).last_hidden_state[:, -1]                       # left padding -> last position is the last token
            E3.append(torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy())
    np.savez_compressed(R / ds / "text_embeddings.npz", problem_ids=np.array(ids), minilm=np.asarray(E1, np.float32), jina=np.concatenate(out).astype(np.float32),
                        qwen3emb8b=np.concatenate(E3).astype(np.float32))
    print(ds, len(ids), "embedded", flush=True)
print("ALL DONE")
