"""A TRAINED small reader for cost: fine-tune jina-embeddings-v2-base-code (137M) end-to-end to predict every route's log
mean output tokens from text -- the problem alone, or the problem + gpt-oss-20b-low's 512-token prefix. Complements the
frozen 4B probe (linear head on fixed activations): does TRAINING the representation help where the frozen one saturates?
Mean-pooled encoder -> Linear(768, n_routes) on per-route z-scored targets (train statistics), masked MSE. Train on TRAIN,
pick the epoch on CALIBRATION loss, score TEST once. Writes <pool>/cost_preds_ft_<tag>.jsonl (market prices, smearing +
train-level match, same post-processing as the other heads) and prints test log-output R2 per route.
Usage: python finetune_cost_reader.py --pool <tensors dir name> --prompts <jsonl> --tag <tag> [--max-len 2048]
"""
import argparse, json, sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from decompose import MK, R

ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True); ap.add_argument("--prompts", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--max-len", type=int, default=2048); ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--batch", type=int, default=8); ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--head-lr", type=float, default=1e-3)
a = ap.parse_args()
torch.manual_seed(0); np.random.seed(0)
assert torch.cuda.is_available(), "GPU job only"
D = R / a.pool; t = np.load(D / "tensors.npz", allow_pickle=True)
S = [str(s) for s in t["model_slots"]]; pids = [str(p) for p in t["problem_ids"]]; pi = {p: i for i, p in enumerate(pids)}
v = t["valid"].astype(bool); ct = t["completion_tokens"].astype(float); pt = t["prompt_tokens"].astype(float); n = v.sum(2)
Y = np.log(np.maximum(np.where(n > 0, np.where(v, ct, 0).sum(2) / np.maximum(n, 1), np.nan), 1)); Y[n == 0] = np.nan
inp = np.nanmean(np.where(v, pt, np.nan), 2)
text = {json.loads(l)["problem_id"]: json.loads(l)["prompt"] for l in open(a.prompts)}
sp = json.load(open(D / "split_manifest.json"))
idx = {k: np.array([pi[str(p)] for p in sp[f"{k}_problem_ids"] if str(p) in text]) for k in ("train", "calibration", "test")}
mu, sd = np.nanmean(Y[idx["train"]], 0), np.nanstd(Y[idx["train"]], 0)
Z = (Y - mu) / sd

from transformers import AutoModel, AutoTokenizer
tok = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v2-base-code")
enc = AutoModel.from_pretrained("jinaai/jina-embeddings-v2-base-code", trust_remote_code=True).cuda()
head = torch.nn.Linear(enc.config.hidden_size, len(S)).cuda()
opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": a.lr}, {"params": head.parameters(), "lr": a.head_lr}], weight_decay=0.01)


def forward(ids):
    b = tok([text[pids[i]] for i in ids], truncation=True, max_length=a.max_len, padding=True, return_tensors="pt").to("cuda")
    h = enc(**b).last_hidden_state; m = b["attention_mask"].unsqueeze(-1).float()
    return head((h * m).sum(1) / m.sum(1))


def predict(ids):
    enc.eval(); out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(ids), 16):
            out.append(forward(ids[i:i + 16]).float().cpu().numpy())
    enc.train(); return np.concatenate(out)


def loss_on(ids):
    p = predict(ids); z = Z[ids]; ok = np.isfinite(z); return float(((p - np.nan_to_num(z)) ** 2)[ok].mean())


best = (np.inf, None, -1)
for ep in range(a.epochs):
    perm = np.random.permutation(idx["train"])
    for i in range(0, len(perm), a.batch):
        ids = perm[i:i + a.batch]; z = torch.tensor(np.nan_to_num(Z[ids]), dtype=torch.float32, device="cuda")
        mask = torch.tensor(np.isfinite(Z[ids]), dtype=torch.float32, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = forward(ids).float()
        l = (((p - z) ** 2) * mask).sum() / mask.sum().clamp(min=1)
        opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), 1.0); opt.step()
    cl = loss_on(idx["calibration"])
    print(f"epoch {ep}: calibration MSE (z units) {cl:.3f}", flush=True)
    if cl < best[0]:
        best = (cl, predict(np.arange(len(pids))), ep)
P = best[1] * sd + mu                                        # back to log output tokens
tr, te = idx["train"], idx["test"]; C = np.zeros_like(P); r2 = []
for m, s in enumerate(S):
    ok = tr[np.isfinite(Y[tr, m])]
    o = np.exp(P[:, m]) * np.mean(np.exp(Y[ok, m] - P[ok, m])); o *= np.exp(Y[ok, m]).mean() / o[ok].mean()
    C[:, m] = (np.nan_to_num(inp[:, m], nan=np.nanmean(inp[:, m])) * MK[s][0] + o * MK[s][1]) / 1e6
    tt = te[np.isfinite(Y[te, m])]; r2.append(1 - ((Y[tt, m] - P[tt, m]) ** 2).sum() / ((Y[tt, m] - Y[tt, m].mean()) ** 2).sum())
with open(D / f"cost_preds_ft_{a.tag}.jsonl", "w") as f:
    for i, p in enumerate(pids):
        if p in text:
            f.write(json.dumps({"problem_id": p, "expected_costs": [float(x) for x in C[i]]}) + "\n")
print(f"{a.pool} ft_{a.tag}: best epoch {best[2]}; test log-output R2 " + "  ".join(f"{s} {x:+.2f}" for s, x in zip(S, r2)) + f"   mean {np.mean(r2):+.2f}")
