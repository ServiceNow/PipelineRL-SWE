#!/usr/bin/env python3
"""Fine-tune a small code encoder end-to-end to predict per-route success from a routing state.

Why this can win where the frozen version does not. Frozen jina-embeddings-v2-base-code is trained
for RETRIEVAL, and still lands within 0.01-0.02 AUC of a 20,480-d Qwen3-4B prefill on two of three
rungs (it loses badly only on gpt-oss-20b-low). So most of the gap is plausibly objective mismatch,
not capacity: nothing in contrastive retrieval training asks "will a 20B model solve this". At 137M
the encoder can be trained on that question directly, which a frozen 4B prefill cannot be cheaply --
and it is ~50x cheaper per state, which matters because the history arm pays one encode per
decision, not one per problem.

Trained on the same states, labels and split as the frozen comparison, so the delta is the training
objective and nothing else. Multi-task: one shared encoder, one head per route, masked where a
route has no remaining draws. Count features are concatenated at the head, exactly as in
fit_history_counts_head.py, so the encoder is only asked for the semantic part.
"""
from __future__ import annotations
import argparse, itertools, json
from pathlib import Path
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts-dir", required=True)
    ap.add_argument("--variant", default="code")
    ap.add_argument("--tensors-dir", required=True)
    ap.add_argument("--model", default="jinaai/jina-embeddings-v2-base-code")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--max-states-per-problem", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch, torch.nn as nn
    from transformers import AutoModel, AutoTokenizer

    PD = Path(a.prompts_dir); T = Path(a.tensors_dir)
    text = {}
    for f in sorted(PD.glob(f"{a.variant}_shard*.jsonl")):
        for line in open(f):
            if line.strip():
                r = json.loads(line); text[str(r["problem_id"])] = r["prompt"]
    t = np.load(T / "tensors.npz", allow_pickle=True)
    ok = (t["final_outcome"] & t["valid"]).astype(bool); valid = t["valid"].astype(bool)
    slots = [str(s) for s in t["model_slots"]]; M = len(slots)
    pidx = {str(p): i for i, p in enumerate(t["problem_ids"])}
    sm = json.loads((T / "split_manifest.json").read_text())
    grp = {**{str(p): 0 for p in sm["train_problem_ids"]},
           **{str(p): 1 for p in sm["calibration_problem_ids"]},
           **{str(p): 2 for p in sm["test_problem_ids"]}}
    rng = np.random.default_rng(a.seed); rows = []
    for pid, pi in pidx.items():
        fails = {m: [k for k in range(ok.shape[2]) if valid[pi, m, k] and not ok[pi, m, k]]
                 for m in range(M)}
        cand = []
        for counts in itertools.product(*[range(len(fails[m]) + 1) for m in range(M)]):
            if sum(counts) == 0 or sum(counts) > 8:
                continue
            for r in range(M):
                if counts[r] >= 1:
                    cand.append((counts, r, fails[r][counts[r] - 1]))
        if len(cand) > a.max_states_per_problem:
            cand = [cand[i] for i in rng.choice(len(cand), a.max_states_per_problem, replace=False)]
        for counts, r, dr in cand:
            eid = f"{pid}||{slots[r]}{dr}"
            if eid not in text:
                continue
            used = {(m, k) for m in range(M) for k in fails[m][:counts[m]]}
            lab, msk = np.zeros(M, np.float32), np.zeros(M, np.float32)
            for m in range(M):
                rem = [k for k in range(ok.shape[2]) if valid[pi, m, k] and (m, k) not in used]
                if rem:
                    msk[m] = 1.0; lab[m] = float(any(ok[pi, m, k] for k in rem))
            rows.append((eid, np.array(counts, np.float32), grp.get(pid, 3), lab, msk, r))
    tr = [r for r in rows if r[2] == 0]; te = [r for r in rows if r[2] == 2]
    print(f"{len(rows)} states: {len(tr)} train / {len(te)} test", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    enc = AutoModel.from_pretrained(a.model, trust_remote_code=True).to(dev)
    D = enc.config.hidden_size
    head = nn.Sequential(nn.Linear(D + 2 * M + 1 + M, 256), nn.ReLU(), nn.Linear(256, M)).to(dev)
    opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": a.lr},
                             {"params": head.parameters(), "lr": a.head_lr}])
    bce = nn.BCEWithLogitsLoss(reduction="none")

    def batch(sel):
        e = tok([text[r[0]] for r in sel], padding=True, truncation=True,
                max_length=a.max_len, return_tensors="pt").to(dev)
        C = torch.tensor(np.array([r[1] for r in sel]), device=dev)
        last = torch.zeros(len(sel), M, device=dev)
        for i, r in enumerate(sel):
            last[i, r[5]] = 1
        side = torch.cat([C, torch.log1p(C), C.sum(1, keepdim=True), last], 1)
        return e, side, (torch.tensor(np.array([r[3] for r in sel]), device=dev),
                         torch.tensor(np.array([r[4] for r in sel]), device=dev))

    def forward(e, side):
        h = enc(**e).last_hidden_state
        m = e["attention_mask"].unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return head(torch.cat([pooled, side], 1))

    for ep in range(a.epochs):
        enc.train(); head.train(); perm = rng.permutation(len(tr)); tot = 0.0
        for i in range(0, len(perm), a.batch):
            sel = [tr[j] for j in perm[i:i + a.batch]]
            e, side, (y, msk) = batch(sel)
            loss = (bce(forward(e, side), y) * msk).sum() / msk.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
            if (i // a.batch) % 50 == 0:
                print(f"  ep{ep} {i}/{len(perm)} loss {tot/max(1,(i//a.batch)+1):.4f}", flush=True)
        enc.eval(); head.eval(); P, Y, MK = [], [], []
        with torch.no_grad():
            for i in range(0, len(te), a.batch):
                sel = te[i:i + a.batch]
                e, side, (y, msk) = batch(sel)
                P.append(torch.sigmoid(forward(e, side)).cpu().numpy())
                Y.append(y.cpu().numpy()); MK.append(msk.cpu().numpy())
        P, Y, MK = np.concatenate(P), np.concatenate(Y), np.concatenate(MK)
        line = []
        for m, s in enumerate(slots):
            k = MK[:, m] > 0; p, y = P[k, m], Y[k, m].astype(bool)
            if y.all() or not y.any():
                continue
            o = np.argsort(p); rk = np.empty(len(p)); rk[o] = np.arange(1, len(p) + 1)
            line.append(f"{s} {(rk[y].sum()-y.sum()*(y.sum()+1)/2)/(y.sum()*(~y).sum()):.3f}")
        print(f"epoch {ep}: test AUC  " + "  ".join(line), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head": head.state_dict()}, a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
