"""Endpoints, not models: do providers of deepseek-v4-flash behave as 'the same model plus a small offset'?
Per-draw dsv4f records (problem, provider, output length, correct) on MMLU-Pro and Omni (original pools + fresh sets).
Shared readouts = our frozen-prefill heads for dsv4f (predicted mean output tokens and P(correct) per problem; archived heads,
fitted without any provider information). Fit problems = original TRAIN split; evaluation = original test + fresh problems.
Provider groups with >= 150 evaluation draws are evaluated; per group g, on its evaluation draws:
  pooled     shared readouts, no provider information
  offset     shared readouts + provider offset fitted on g's TRAIN draws (length: ratio of means; success: logit shift)
  onboard-k  g treated as a NEW endpoint: offset from k of g's draws on train problems only (k = 5, 10, 25, 50; 20 draws)
  scratch-k  g treated as an unrelated model from k draws: constant length (mean of the k) and base rate
Metrics per draw: log-length MSE (and predicted/realized mean length), success log-loss. Also: share of within-problem
log-length variance explained by provider (fixed effects after removing problem means), for every route.
Usage: python provider_endpoints.py
"""
import glob, json, sys, collections
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from carrot_compare import POOLS, read_predictions
from decompose import MK, R

sig = lambda z: 1 / (1 + np.exp(-z)); lgt = lambda p: np.log(p / (1 - p))
KS = (5, 10, 25, 50); out = {}
for ds, label in (("mmlupro", "MMLU-Pro"), ("omni500", "Omni")):
    name, cost_file = POOLS[label]; old = R / name; folder = R / "expanded_eval_20261001" / ds
    t = np.load(folder / "tensors.npz", allow_pickle=True); ids = list(map(str, t["problem_ids"])); slots = list(map(str, t["model_slots"]))
    j = slots.index("dsv4f"); index = {p: i for i, p in enumerate(ids)}
    n_old = len(np.load(old / "tensors.npz", allow_pickle=True)["problem_ids"])
    split = json.loads((old / "split_manifest.json").read_text()); train = set(map(str, split["train_problem_ids"]))
    cost = read_predictions(folder / "paper_cost_preds.jsonl", ids, "expected_costs", len(slots))
    cost[:n_old] = read_predictions(old / cost_file, ids[:n_old], "expected_costs", len(slots))
    p = read_predictions(folder / "success_preds.jsonl", ids, "p_successes", len(slots))
    p[:n_old] = read_predictions(old / "content_preds.jsonl", ids[:n_old], "p_successes", len(slots))
    inp = np.where(t["valid"].astype(bool), t["prompt_tokens"], 0).sum(2) / np.maximum(t["valid"].sum(2), 1)
    pin, pout = MK["dsv4f"][0] / 1e6, MK["dsv4f"][1] / 1e6
    predL = np.maximum((cost[:, j] - inp[:, j] * pin) / pout, 1.0)                 # predicted MEAN output tokens per problem
    P = np.clip(p[:, j], 1e-4, 1 - 1e-4)
    rows = []
    for f in glob.glob(f"{R}/math_pool/{ds}/dsv4f_d*.jsonl") + glob.glob(f"{R}/math_expand_20261001/{ds}/dsv4f*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if r.get("finish_reason") == "error" or not r.get("provider") or not r.get("completion_tokens") or r["problem_id"] not in index:
                continue
            rows.append((index[r["problem_id"]], r["provider"], float(r["completion_tokens"]), float(bool(r.get("resolved")))))
    pi = np.array([x[0] for x in rows]); pv = np.array([x[1] for x in rows]); L = np.array([x[2] for x in rows]); y = np.array([x[3] for x in rows])
    is_tr = np.array([ids[i] in train for i in pi])
    groups = [g for g, c in collections.Counter(pv[~is_tr]).most_common() if c >= 150]
    print(f"\n===== {label}: {len(rows)} dsv4f draws, {len(set(pv))} providers; evaluated groups: {groups}")

    def metrics(ev, Lhat, Phat):
        return dict(msle=float(np.mean((np.log(L[ev]) - np.log(Lhat)) ** 2)), calib=float(Lhat.mean() / L[ev].mean()),
                    ll=float(-np.mean(y[ev] * np.log(Phat) + (1 - y[ev]) * np.log(1 - Phat))))

    def offsets(fit):                                                              # length ratio of means, success logit shift
        a = np.log(L[fit].sum() / predL[pi[fit]].sum())
        lo, hi = -6.0, 6.0                                                         # solve mean(sig(logit p + b)) = mean(y) by bisection
        for _ in range(60):
            b = (lo + hi) / 2; lo, hi = (b, hi) if sig(lgt(P[pi[fit]]) + b).mean() < y[fit].mean() else (lo, b)
        return a, (lo + hi) / 2
    rng = np.random.default_rng(0); res = {}
    for g in groups:
        ev = np.flatnonzero((pv == g) & ~is_tr); trg = np.flatnonzero((pv == g) & is_tr)
        base = metrics(ev, predL[pi[ev]], P[pi[ev]])
        a, b = offsets(trg) if len(trg) >= 20 else (0.0, 0.0)
        off = metrics(ev, predL[pi[ev]] * np.exp(a), sig(lgt(P[pi[ev]]) + b))
        row = {"n_eval": len(ev), "n_train": len(trg), "pooled": base, "offset": off, "offset_len_x": float(np.exp(a)), "offset_logit": float(b)}
        pool_tr = np.flatnonzero(is_tr & (pv == g))
        for k in KS:
            if len(pool_tr) < k:
                continue
            ob, sc = [], []
            for _ in range(20):
                kk = rng.choice(pool_tr, k, replace=False); a_, b_ = offsets(kk)
                ob.append(metrics(ev, predL[pi[ev]] * np.exp(a_), sig(lgt(P[pi[ev]]) + b_)))
                sc.append(metrics(ev, np.full(len(ev), L[kk].mean()), np.full(len(ev), np.clip(y[kk].mean(), .02, .98))))
            row[f"onboard{k}"] = {m: float(np.mean([x[m] for x in ob])) for m in ob[0]}
            row[f"scratch{k}"] = {m: float(np.mean([x[m] for x in sc])) for m in sc[0]}
        res[g] = row
        line = lambda m: f"MSLE {m['msle']:.3f} calib {m['calib']:.2f} LL {m['ll']:.3f}"
        print(f"  {g:<14} eval n={len(ev):5d} (train draws {len(trg)})  offset: length x{np.exp(a):.2f}, logit {b:+.2f}")
        print(f"     pooled (no provider info) {line(base)}\n     + provider offset         {line(off)}")
        for k in KS:
            if f"onboard{k}" in row:
                print(f"     k={k:<3} onboard {line(row[f'onboard{k}'])}   | scratch {line(row[f'scratch{k}'])}")
    out[label] = res
json.dump(out, open(Path(__file__).parent / "provider_endpoints.json", "w"), indent=1)
