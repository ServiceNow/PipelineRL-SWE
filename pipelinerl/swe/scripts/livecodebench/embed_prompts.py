#!/usr/bin/env python3
"""Embed routing prompts with a small CODE embedding model, as an alternative representation.

Why this is a METHOD and not just a baseline. The closest prior work (arXiv 2603.20895) already
publishes single-shot routing on prefill activations, so "4B prefill predicts pool success" is not
open ground. A small code encoder is different in kind: at 137M it can be FINE-TUNED end to end on
our own states, which a frozen 4B cannot be cheaply, and it costs ~50x less per state -- which
matters most for the history arm, where we pay one encode per decision rather than one per problem.

Runs on CPU by design: the extraction GPUs are contended, and 137M x ~12k short sequences is
minutes here. jina-embeddings-v2-base-code takes 8192 tokens, enough for problem + failed attempt.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", required=True, help="jsonl with problem_id + prompt")
    ap.add_argument("--model", default="jinaai/jina-embeddings-v2-base-code")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    import torch
    from transformers import AutoModel, AutoTokenizer
    rows = [json.loads(l) for l in open(a.prompts) if l.strip()]
    print(f"{len(rows)} prompts; loading {a.model}")
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(a.model, trust_remote_code=True).to(a.device).eval()
    ids, out = [], []
    with torch.no_grad():
        for i in range(0, len(rows), a.batch):
            b = rows[i:i + a.batch]
            enc = tok([r["prompt"] for r in b], padding=True, truncation=True,
                      max_length=a.max_len, return_tensors="pt").to(a.device)
            h = mdl(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            mean = (h * m).sum(1) / m.sum(1).clamp(min=1)          # mean pool over real tokens
            last = h[torch.arange(len(b)), enc["attention_mask"].sum(1) - 1]
            out.append(torch.cat([mean, last], -1).float().cpu().numpy())
            ids += [str(r["problem_id"]) for r in b]
            if (i // a.batch) % 10 == 0:
                print(f"  {i + len(b)}/{len(rows)}", flush=True)
    X = np.concatenate(out)
    np.savez_compressed(a.out, problem_ids=np.array(ids), emb=X.astype(np.float32))
    print(f"wrote {X.shape} -> {a.out}")


if __name__ == "__main__":
    main()
