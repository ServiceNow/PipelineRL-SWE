"""Fine-tuned text encoder as a COST ESTIMATOR arm for fresh_baselines.py (NEW_PATH 4.A.90): the CARROT-RoBERTa / ZeroRouter-DistilBERT
kind of per-query length predictor, trained end-to-end on the same training problems and length labels as our frozen-prefill readout.
Encoders (all cached locally): distilbert (66M, 512 tokens; ZeroRouter's / IRT-Router's encoder), jina (jina-embeddings-v2-base-code,
137M, 2048 tokens), qwen06 (Qwen3-0.6B, full fine-tune, 2048 tokens; the prefill's own family, 7x smaller).
Input: the text the prefill reads (LCB / Omni-MATH problem statement; MMLU-Pro question WITH its options, rebuilt as in
reasoning_datasets._mc_prompt). Model: mean-pooled encoder -> Linear(hidden, n_routes) on per-route z-scored log mean output tokens
(train statistics), masked MSE. Learning rate and epoch chosen on CALIBRATION MSE; TEST never touched. Smearing and the per-route
level match are estimated on CALIBRATION (a fine-tuned net nearly memorises its training problems, so in-sample residuals would
understate both). Writes $OUT_DIR/ft_cost_<pool>_<enc>.jsonl {problem_id, pred_tokens[route]} for every problem in the pool, which
fresh_baselines.py reads as arm ft-<enc> when FT_DIR is set.
Usage: python finetune_cost_estimator.py --pool LCB|Omni|MMLU-Pro --enc distilbert|jina|qwen06
"""
import argparse, json, os, sys, time, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from decompose import R, split_manifest

ENC = {"distilbert": ("distilbert-base-uncased", 512, (2e-5, 5e-5), 16), "jina": ("jinaai/jina-embeddings-v2-base-code", 2048, (1e-5, 3e-5), 8),
       "qwen06": ("Qwen/Qwen3-0.6B", 2048, (5e-6, 2e-5), 8)}
POOL_DIRS = {"LCB": ("pool_v2_tensors_5rung", "pool_v2_tensors_5rung"), "Omni": ("omni500_tensors", "expanded_eval_20261001/omni500"),
             "MMLU-Pro": ("mmlupro_tensors", "expanded_eval_20261001/mmlupro")}
ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True, choices=list(POOL_DIRS)); ap.add_argument("--enc", required=True, choices=list(ENC))
ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--head-lr", type=float, default=1e-3)
a = ap.parse_args()
assert torch.cuda.is_available(), "GPU job only"
name, maxlen, LRS, BATCH = ENC[a.enc]
old, F = (R / d for d in POOL_DIRS[a.pool])
t = np.load(F / "tensors.npz", allow_pickle=True)
ids = [str(p) for p in t["problem_ids"]]; S = [str(s) for s in t["model_slots"]]; idx = {p: i for i, p in enumerate(ids)}; M = len(S)
sp = split_manifest(old); tr, ca = [np.array([idx[str(p)] for p in sp[k + "_problem_ids"]]) for k in ("train", "calibration")]
v = t["valid"].astype(bool); cnt = np.maximum(v.sum(2), 1); L = np.where(v, t["completion_tokens"], 0).sum(2) / cnt
Y = np.log(np.maximum(L, 1)); Y[v.sum(2) == 0] = np.nan                  # same target as fresh_baselines (log mean output)
probs = {json.loads(l)["problem_id"]: json.loads(l) for l in (F / "problems.jsonl").read_text().splitlines()}
text = {p: probs[p]["problem_statement"] for p in ids}
if a.pool == "MMLU-Pro":                                                  # the prefill read the question with its ten options
    # prompts_with_options.jsonl: reasoning_datasets._mc_prompt over the cached TIGER-Lab/MMLU-Pro test split, written once locally
    # (the dataset cache is not visible from eai jobs); matches mmlupro_probe_prompts.jsonl on all 1000 problems it covers
    text.update({r["problem_id"]: r["prompt"] for r in map(json.loads, (F / "prompts_with_options.jsonl").open())})
    assert all(t_.startswith("Answer the following") for t_ in text.values()), "MMLU-Pro prompt rebuild incomplete"
mu, sd = np.nanmean(Y[tr], 0), np.nanstd(Y[tr], 0); Z = (Y - mu) / sd
print(f"{a.pool} {a.enc}: n={len(ids)} train {len(tr)} calibration {len(ca)}; routes {S}", flush=True)

from transformers import AutoModel, AutoTokenizer
tok = AutoTokenizer.from_pretrained(name)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"


def run(lr, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    enc = AutoModel.from_pretrained(name, trust_remote_code=True, torch_dtype=torch.float32).cuda()
    if a.enc == "qwen06":
        enc.gradient_checkpointing_enable()
    head = torch.nn.Linear(enc.config.hidden_size, M).cuda()
    opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": lr}, {"params": head.parameters(), "lr": a.head_lr}], weight_decay=0.01)
    steps = a.epochs * int(np.ceil(len(tr) / BATCH)); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / (0.06 * steps)) * max(0, 1 - s / steps))

    def forward(ii):
        b = tok([text[ids[i]] for i in ii], truncation=True, max_length=maxlen, padding=True, return_tensors="pt").to("cuda")
        h = enc(**b).last_hidden_state.float(); m = b["attention_mask"].unsqueeze(-1).float()
        return head((h * m).sum(1) / m.sum(1))

    def predict(ii):
        enc.eval(); out = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(ii), 32):
                out.append(forward(ii[i:i + 32]).float().cpu().numpy())
        enc.train(); return np.concatenate(out)

    best = (np.inf, None, -1)
    for ep in range(a.epochs):
        perm = rng.permutation(tr)
        for i in range(0, len(perm), BATCH):
            ii = perm[i:i + BATCH]; z = torch.tensor(np.nan_to_num(Z[ii]), dtype=torch.float32, device="cuda")
            mask = torch.tensor(np.isfinite(Z[ii]), dtype=torch.float32, device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = forward(ii)
            l = (((p - z) ** 2) * mask).sum() / mask.sum().clamp(min=1)
            opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), 1.0); opt.step(); sched.step()
        pc = predict(ca); ok = np.isfinite(Z[ca]); cl = float(((pc - np.nan_to_num(Z[ca])) ** 2)[ok].mean())
        print(f"  lr {lr:g} epoch {ep}: calibration MSE (z units) {cl:.3f}", flush=True)
        if cl < best[0]:
            best = (cl, predict(np.arange(len(ids))), ep)
    del enc, head, opt; torch.cuda.empty_cache()
    return best


t0 = time.time(); runs = {lr: run(lr) for lr in LRS}
lr = min(runs, key=lambda k: runs[k][0]); cl, Pz, ep = runs[lr]
P = Pz * sd + mu                                                          # log output tokens
O = np.zeros_like(P); r2 = {}
te = np.array([idx[str(p)] for p in sp["test_problem_ids"]]) if a.pool == "LCB" else np.setdiff1d(np.arange(len(ids)), np.r_[tr, ca, [idx[str(p)] for p in sp["test_problem_ids"]]])
for k in range(M):
    c = ca[np.isfinite(Y[ca, k])]
    o = np.exp(P[:, k]) * np.mean(np.exp(Y[c, k] - P[c, k]))                # smearing on calibration
    O[:, k] = o * (L[c, k].mean() / o[c].mean())                          # level match on calibration
    tt = te[np.isfinite(Y[te, k])]; r2[S[k]] = float(1 - ((Y[tt, k] - P[tt, k]) ** 2).sum() / ((Y[tt, k] - Y[tt, k].mean()) ** 2).sum())
out = Path(os.environ.get("OUT_DIR", ".")) / f"ft_cost_{a.pool}_{a.enc}{os.environ.get('DRAW_TAG', '')}.jsonl"
with open(out, "w") as f:
    for i, p in enumerate(ids):
        f.write(json.dumps({"problem_id": p, "pred_tokens": [float(x) for x in O[i]]}) + "\n")
print(f"{a.pool} ft-{a.enc}: chosen lr {lr:g}, epoch {ep}, calibration MSE {cl:.3f} (all: " + ", ".join(f"{k:g}: {v_[0]:.3f}@{v_[2]}" for k, v_ in runs.items())
      + f"); evaluated-set log-output R2 " + "  ".join(f"{s} {x:+.2f}" for s, x in r2.items()) + f"  mean {np.mean(list(r2.values())):+.2f}; "
      f"{(time.time() - t0) / 60:.0f} min -> {out}", flush=True)
