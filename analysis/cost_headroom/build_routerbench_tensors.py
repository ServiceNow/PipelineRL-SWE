"""RouterBench (withmartian/routerbench, 0-shot) as a NON-REASONING pool for the cost-predictability rule: 11 chat
models (GPT-4, Claude v1/v2/instant, Llama-2-70B, Mixtral, ...), one response each, per-query total cost.

Prediction written BEFORE this run (NEW_PATH 4.A.7): output is a small share of cost and short, so HEADROOM is small.
Tokens are not in the release: prompt / completion tokens are approximated as chars / 4, and each model's input and
output price is RECOVERED by least squares of its total_cost on (prompt chars, response chars) -- saved to
prices.json, which decompose.py uses for this pool. A stratified 6000-query subsample (seed 0; every eval kept in
proportion) with the probe activations that already exist (routerbench_activations/scout.npz).
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

R = Path("/mnt/llmd/results/exps/aristides/reason")
OUT = R / "routerbench_tensors"; N = 6000
d = pd.read_pickle(R / "routerbench_data" / "routerbench_0shot.pkl")
z = np.load(R / "routerbench_activations" / "scout.npz", allow_pickle=True)
have = {str(p) for p in z["problem_ids"]}
d = d[d["sample_id"].astype(str).isin(have)].reset_index(drop=True)
models = [c for c in d.columns if f"{c}|total_cost" in d.columns]
rng = np.random.default_rng(0)
keep = d.groupby("eval_name", group_keys=False).apply(lambda g: g.sample(max(1, round(len(g) * N / len(d))), random_state=0))
keep = keep.reset_index(drop=True)
n, M = len(keep), len(models)
pl = keep["prompt"].astype(str).str.len().to_numpy(float)
fo = np.zeros((n, M, 1), bool); pt = np.zeros((n, M, 1), np.float32); ct = np.zeros((n, M, 1), np.float32)
prices = {}
for m, c in enumerate(models):
    rl = keep[f"{c}|model_response"].astype(str).str.len().to_numpy(float)
    cost = keep[f"{c}|total_cost"].to_numpy(float)
    A = np.c_[pl / 4, rl / 4]; w, *_ = np.linalg.lstsq(A, cost, rcond=None)
    prices[c] = (max(float(w[0]) * 1e6, 0.0), max(float(w[1]) * 1e6, 0.0))      # $/M in, out
    fo[:, m, 0] = keep[c].to_numpy(float) >= 0.5; pt[:, m, 0] = pl / 4; ct[:, m, 0] = np.maximum(rl / 4, 1)
ids = keep["sample_id"].astype(str).tolist()
perm = rng.permutation(n); ntr, ncal = int(0.55 * n), int(0.15 * n)
split = {"train_problem_ids": [ids[k] for k in perm[:ntr]], "calibration_problem_ids": [ids[k] for k in perm[ntr:ntr + ncal]],
         "test_problem_ids": [ids[k] for k in perm[ntr + ncal:]]}
OUT.mkdir(parents=True, exist_ok=True)
np.savez(OUT / "tensors.npz", final_outcome=fo, execution_outcome=fo, weak_verifier_outcome=fo, valid=np.ones((n, M, 1), bool),
         prompt_tokens=pt, completion_tokens=ct, problem_ids=np.array(ids), model_slots=np.array(models), schema_version=3)
(OUT / "split_manifest.json").write_text(json.dumps(split))
(OUT / "prices.json").write_text(json.dumps(prices, indent=1))
with open(OUT / "problems.jsonl", "w") as f:
    for i, r in keep.iterrows():
        f.write(json.dumps({"problem_id": ids[i], "platform": r["eval_name"], "problem_statement": str(r["prompt"])}) + "\n")
print(f"{n} queries, {M} models -> {OUT}")
for m, c in enumerate(models):
    share = (ct[:, m, 0] * prices[c][1]).sum() / max((pt[:, m, 0] * prices[c][0] + ct[:, m, 0] * prices[c][1]).sum(), 1e-12)
    print(f"  {c:<36} acc {fo[:, m, 0].mean():.2f}  price in/out ${prices[c][0]:.2f}/${prices[c][1]:.2f} per M  "
          f"mean out {ct[:, m, 0].mean():.0f} tok  output share of $ {share*100:.0f}%")
