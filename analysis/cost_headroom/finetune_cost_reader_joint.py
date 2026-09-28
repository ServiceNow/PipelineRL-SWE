"""JOINT trained cost reader: one fully fine-tuned jina-embeddings-v2-base-code (137M) encoder shared across five pools
(LCB, CodeContests, Omni-MATH, BCB, TACO -- ~2150 training problems, ~5x any single pool), a separate linear head per
pool (its own routes, per-route z-scored log mean output tokens). Problem text only. Each batch comes from one pool
(shuffled pool-chunks). Epoch chosen on the summed per-pool calibration MSE; each pool scored on its own TEST split.
Writes <pool>/cost_preds_ft_joint.jsonl (market prices, same post-processing as finetune_cost_reader.py).
Answers: is the single-pool reader's weakness (fewer than 450 problems) fixed by sharing the representation?
"""
import json, sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from decompose import MK, R

POOLS = {"pool_v2_tensors_5rung": R / "pool_probe_prompts.jsonl", "cc_tensors": R / "cc_pool/prompts.jsonl",
         "omni500_tensors": R / "omni500_probe_prompts.jsonl", "bcb_tensors_5r": R / "bcb_probe_prompts.jsonl",
         "taco_tensors_ha": R / "taco_probe_prompts.jsonl"}
EPOCHS, BATCH, MAXLEN = 6, 8, 2048
torch.manual_seed(0); np.random.seed(0); assert torch.cuda.is_available()
data = {}
for name, pf in POOLS.items():
    D = R / name; t = np.load(D / "tensors.npz", allow_pickle=True)
    S = [str(s) for s in t["model_slots"]]; pids = [str(p) for p in t["problem_ids"]]; pi = {p: i for i, p in enumerate(pids)}
    v = t["valid"].astype(bool); ct = t["completion_tokens"].astype(float); pt = t["prompt_tokens"].astype(float); n = v.sum(2)
    Y = np.log(np.maximum(np.where(n > 0, np.where(v, ct, 0).sum(2) / np.maximum(n, 1), np.nan), 1)); Y[n == 0] = np.nan
    text = {json.loads(l)["problem_id"]: json.loads(l)["prompt"] for l in open(pf)}
    sp = json.load(open(D / "split_manifest.json"))
    idx = {k: np.array([pi[str(p)] for p in sp[f"{k}_problem_ids"] if str(p) in text]) for k in ("train", "calibration", "test")}
    mu, sd = np.nanmean(Y[idx["train"]], 0), np.nanstd(Y[idx["train"]], 0)
    data[name] = dict(D=D, S=S, pids=pids, Y=Y, Z=(Y - mu) / sd, mu=mu, sd=sd, text=text, idx=idx,
                      inp=np.nanmean(np.where(v, pt, np.nan), 2))
    print(f"{name}: train {len(idx['train'])} / cal {len(idx['calibration'])} / test {len(idx['test'])}", flush=True)

from transformers import AutoModel, AutoTokenizer
tok = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v2-base-code")
enc = AutoModel.from_pretrained("jinaai/jina-embeddings-v2-base-code", trust_remote_code=True).cuda()
heads = torch.nn.ModuleDict({k.replace(".", "_"): torch.nn.Linear(enc.config.hidden_size, len(d["S"])) for k, d in data.items()}).cuda()
opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": 2e-5}, {"params": heads.parameters(), "lr": 1e-3}], weight_decay=0.01)


def forward(name, ids):
    d = data[name]; b = tok([d["text"][d["pids"][i]] for i in ids], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to("cuda")
    h = enc(**b).last_hidden_state; m = b["attention_mask"].unsqueeze(-1).float()
    return heads[name.replace(".", "_")]((h * m).sum(1) / m.sum(1))


def predict(name, ids):
    enc.eval(); out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(ids), 16):
            out.append(forward(name, ids[i:i + 16]).float().cpu().numpy())
    enc.train(); return np.concatenate(out)


best = (np.inf, None, -1)
for ep in range(EPOCHS):
    chunks = [(k, c) for k, d in data.items() for c in np.array_split(np.random.permutation(d["idx"]["train"]), max(1, len(d["idx"]["train"]) // BATCH))]
    np.random.shuffle(chunks)
    for name, ids in chunks:
        Z = data[name]["Z"][ids]; z = torch.tensor(np.nan_to_num(Z), dtype=torch.float32, device="cuda")
        mask = torch.tensor(np.isfinite(Z), dtype=torch.float32, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = forward(name, ids).float()
        l = (((p - z) ** 2) * mask).sum() / mask.sum().clamp(min=1)
        opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(heads.parameters()), 1.0); opt.step()
    cal = {}
    for k, d in data.items():
        c = d["idx"]["calibration"]; pz = predict(k, c); ok = np.isfinite(d["Z"][c]); cal[k] = float(((pz - np.nan_to_num(d["Z"][c])) ** 2)[ok].mean())
    tot = sum(cal.values()); print(f"epoch {ep}: calibration MSE " + " ".join(f"{k.split('_')[0]} {v:.3f}" for k, v in cal.items()) + f"  sum {tot:.3f}", flush=True)
    if tot < best[0]:
        best = (tot, {k: predict(k, np.arange(len(d["pids"]))) for k, d in data.items()}, ep)
print(f"best epoch {best[2]}")
for k, d in data.items():
    P = best[1][k] * d["sd"] + d["mu"]; tr, te = d["idx"]["train"], d["idx"]["test"]; C = np.zeros_like(P); r2 = []
    for m, s in enumerate(d["S"]):
        ok = tr[np.isfinite(d["Y"][tr, m])]
        o = np.exp(P[:, m]) * np.mean(np.exp(d["Y"][ok, m] - P[ok, m])); o *= np.exp(d["Y"][ok, m]).mean() / o[ok].mean()
        C[:, m] = (np.nan_to_num(d["inp"][:, m], nan=np.nanmean(d["inp"][:, m])) * MK[s][0] + o * MK[s][1]) / 1e6
        tt = te[np.isfinite(d["Y"][te, m])]; r2.append(1 - ((d["Y"][tt, m] - P[tt, m]) ** 2).sum() / ((d["Y"][tt, m] - d["Y"][tt, m].mean()) ** 2).sum())
    with open(d["D"] / "cost_preds_ft_joint.jsonl", "w") as f:
        for i, p in enumerate(d["pids"]):
            f.write(json.dumps({"problem_id": p, "expected_costs": [float(x) for x in C[i]]}) + "\n")
    print(f"{k} ft_joint: test log-output R2 " + "  ".join(f"{s} {x:+.2f}" for s, x in zip(d["S"], r2)) + f"   mean {np.mean(r2):+.2f}")
