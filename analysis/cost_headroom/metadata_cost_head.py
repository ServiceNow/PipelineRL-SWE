"""Cost heads from DIFFICULTY METADATA instead of (or on top of) the 4B prefill -- a causal test of "predictability
drives capture". If LCB's cost head works because the probe reads the easy/medium/hard tier, and CodeContests'
fails because the probe cannot read a Codeforces rating, then handing the router the metadata should move each
pool along the capture curve.

Per route, on TRAIN: OLS of log mean output tokens on the features, Duan smearing, level matched to the train mean;
cost = the problem's own input x in-price + predicted output x out-price (market). Output: cost_preds_<tag>.jsonl
in the pool's tensors dir, read by decompose.py.
Features: meta = the pool's metadata (LCB/TACO: difficulty tier + platform one-hots + log statement length;
CC: cubic in Codeforces rating + log statement length); meta+probe = the same plus the existing 4B head's log
prediction (stacked, so the metadata adds to what the probe already knows).
Also reports how well the 4B probe predicts the metadata itself (ridge on the rich activations, 5-fold CV R2).
Usage: python metadata_cost_head.py <pool> <activations.npz>
"""
import json, sys, numpy as np
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict
sys.path.insert(0, str(Path(__file__).parent))
from decompose import MK, R

name, act = sys.argv[1], sys.argv[2]
D = R / name; t = np.load(D / "tensors.npz", allow_pickle=True)
S = [str(s) for s in t["model_slots"]]; pids = [str(p) for p in t["problem_ids"]]; pi = {p: i for i, p in enumerate(pids)}
v = t["valid"].astype(bool); ct = t["completion_tokens"].astype(float); pt = t["prompt_tokens"].astype(float)
n = v.sum(2); outm = np.where(n > 0, np.where(v, ct, 0).sum(2) / np.maximum(n, 1), np.nan)
inp = np.nanmean(np.where(v, pt, np.nan), 2)
meta = {str(json.loads(l)["problem_id"]): json.loads(l) for l in open(D / "problems.jsonl")}
L = np.log1p(np.array([len(str(meta[p].get("problem_statement", ""))) for p in pids], float))
sp = json.load(open(D / "split_manifest.json")); tr = np.array([pi[str(p)] for p in sp["train_problem_ids"]])
if name.startswith("cc"):
    tasks = {json.loads(l)["problem_id"]: json.loads(l) for l in open(R / "cc_pool" / "cc_tasks.jsonl")}
    rat = np.array([tasks.get(p, {}).get("cf_rating") or np.nan for p in pids], float)
    rat = np.where(np.isfinite(rat), rat, np.nanmedian(rat)); z = (rat - rat[tr].mean()) / rat[tr].std()
    F = np.c_[z, z ** 2, z ** 3, L]; target_meta = rat; meta_name = "Codeforces rating"
else:
    cols = []
    for key in ("difficulty", "platform"):
        vals = [str(meta[p].get(key, "")) for p in pids]; u = sorted(set(vals))
        if 1 < len(u) < 30:
            cols.append(np.array([[x == c for c in u[1:]] for x in vals], float))
    F = np.c_[np.concatenate(cols, 1), L]
    tier = {"easy": 0, "medium": 1, "hard": 2}
    target_meta = np.array([tier.get(str(meta[p].get("difficulty", "")), np.nan) for p in pids], float); meta_name = "difficulty tier"
lcf = {json.loads(l)["problem_id"]: json.loads(l)["expected_costs"][:len(S)] for l in open(D / "cost_preds_market.jsonl")}
LC = np.array([lcf[p] for p in pids])

# how well does the probe read the metadata?
z_ = np.load(act, allow_pickle=True); X = np.concatenate([z_[k].reshape(len(z_[k]), -1) for k in ("mean", "last") if k in z_.files], 1)
aid = {str(p): i for i, p in enumerate(z_["problem_ids"])}; X = X[[aid[p] for p in pids]].astype(np.float32)
X = (X - X.mean(0)) / (X.std(0) + 1e-6); ok = np.isfinite(target_meta)
pred = cross_val_predict(RidgeCV(alphas=np.geomspace(1e1, 1e6, 12)), X[ok], target_meta[ok], cv=5)
r2 = 1 - ((target_meta[ok] - pred) ** 2).sum() / ((target_meta[ok] - target_meta[ok].mean()) ** 2).sum()
print(f"{name}: 4B probe predicts the {meta_name}: 5-fold CV R2 {r2:.2f}")

for tag, extra in (("meta", None), ("metaprobe", "probe")):
    C = np.zeros((len(pids), len(S)))
    for m, s in enumerate(S):
        pin, pout = MK[s][0] / 1e6, MK[s][1] / 1e6
        y = np.log(np.maximum(outm[:, m], 1.0)); a = np.isfinite(outm[:, m])
        G = F if extra is None else np.c_[F, np.log(np.maximum((LC[:, m] - np.nan_to_num(inp[:, m]) * pin) / pout, 1.0))]
        A = np.c_[np.ones(len(pids)), G]; trm = np.intersect1d(tr, np.where(a)[0])
        w, *_ = np.linalg.lstsq(A[trm], y[trm], rcond=None)
        yh = A @ w; smear = np.mean(np.exp(y[trm] - yh[trm]))
        out_tok = np.exp(yh) * smear; out_tok *= np.nanmean(outm[trm, m]) / out_tok[trm].mean()
        C[:, m] = np.nan_to_num(inp[:, m], nan=np.nanmean(inp[:, m])) * pin + out_tok * pout
    with open(D / f"cost_preds_{tag}.jsonl", "w") as f:
        for i, p in enumerate(pids):
            f.write(json.dumps({"problem_id": p, "expected_costs": [float(x) for x in C[i]]}) + "\n")
    print(f"  wrote cost_preds_{tag}.jsonl")
