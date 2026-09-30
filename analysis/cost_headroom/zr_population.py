"""Does ZeroRouter recover with its model POPULATION? MMLU-Pro, our 5-model pool, their stage 1 fitted on N Open LLM Leaderboard
models (the paper's own data source; leaderboard_mmlupro.py) + our 5 pool models, TRAIN questions only (as in the paper: test
questions get their position from stage 2). N in {0, 5, 10, 20, 50, 100, 200}, 3 random population subsets per N (< all), D in
{1, 5, 20}. Stage 2 = frozen 4B reader -> PCA(256) -> ridge (the reader that did at least as well as their DistilBERT, 4.A.34).
Evaluated on our pool's TEST problems: success log-loss of the pool models; routing gain vs the paper rule for zr (K = 5 / 10 / 20,
best reported), zr success + our cost, our success + zr cost; ours for reference.
Usage: python zr_population.py
"""
import glob, json, sys, numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
sys.path.insert(0, str(Path(__file__).parent))
from decompose import MK, R, hull, cost_at
from baseline_cost_heads import rich
from zr_dimsweep import fit_stage1

sig = lambda z: 1 / (1 + np.exp(-z)); VS = np.geomspace(1e-5, 100, 250)
D_ = R / "mmlupro_tensors"; t = np.load(D_ / "tensors.npz", allow_pickle=True)
S = [str(s) for s in t["model_slots"]]; M = len(S); pids = [str(p) for p in t["problem_ids"]]; pi = {p: i for i, p in enumerate(pids)}
v = t["valid"].astype(bool); okd = (t["final_outcome"] & t["valid"]).astype(bool); ct = t["completion_tokens"].astype(float)
pt = t["prompt_tokens"].astype(float); n = v.sum(2); avail = n > 0; succ = (okd & v).sum(2)
inp = np.nan_to_num(np.nanmean(np.where(v, pt, np.nan), 2))
pin = np.array([MK[s][0] for s in S]) / 1e6 * 100; pout = np.array([MK[s][1] for s in S]) / 1e6 * 100
outm = np.where(avail, np.where(v, ct, 0).sum(2) / np.maximum(n, 1), np.nan)
Q = np.where(avail, succ / np.maximum(n, 1), 0); Cr = np.where(avail, inp * pin + outm * pout, 1e9)
sp = json.load(open(D_ / "split_manifest.json")); tr = np.array([pi[str(p)] for p in sp["train_problem_ids"]]); te = np.array([pi[str(p)] for p in sp["test_problem_ids"]])
_lp = {json.loads(l)["problem_id"]: json.loads(l)["p_successes"][:M] for l in open(D_ / "content_preds.jsonl")}
P = np.clip(np.array([_lp[p] for p in pids]), 1e-4, 1 - 1e-4)
lc = {json.loads(l)["problem_id"]: json.loads(l)["expected_costs"][:M] for l in open(D_ / "cost_preds_probe_instruct.jsonl")}
LC = np.array([lc[p] for p in pids]) * 100; med = np.array([np.median(ct[tr, m][v[tr, m]]) for m in range(M)]); PC = inp * pin + med[None] * pout
X = rich(R / "mmlupro_probe/instruct.npz", pids); Z = PCA(256, random_state=0).fit(X[tr]).transform(X); Z /= Z[tr].std(0) + 1e-6

pop = []                                                   # [n_models, n_problems] 0/1 or nan
for f in sorted(glob.glob(str(R / "leaderboard_mmlupro" / "outcomes" / "*.json"))):
    o = json.load(open(f))["outcomes"]; pop.append([o.get(p.split("_")[1], np.nan) for p in pids])
pop = np.array(pop, float); print(f"population: {len(pop)} leaderboard models", flush=True)


def gain(Pm, C):
    def curve(Pm_, C_):
        U = np.where(avail[te][None], Pm_[te][None] * VS[:, None, None] - C_[te][None], -np.inf); m = U.argmax(2)
        a = np.take_along_axis(np.broadcast_to(Q[te], U.shape), m[..., None], 2)[..., 0].mean(1)
        c = np.take_along_axis(np.broadcast_to(Cr[te], U.shape), m[..., None], 2)[..., 0].mean(1); return hull(list(zip(c, a)))
    H, H0 = curve(Pm, C), curve(P, PC); lo, hi = max(H[0][1], H0[0][1]), min(H[-1][1], H0[-1][1])
    T = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 12)
    return 1 - float(np.exp(np.nanmean(np.log([cost_at(H, x) / cost_at(H0, x) for x in T]))))


def nll(PP):
    tot = []
    for m in range(M):
        ii = te[n[te, m] > 0]; p_ = np.clip(PP[ii, m], 1e-4, 1 - 1e-4)
        tot.append(-((succ[ii, m] * np.log(p_) + (n[ii, m] - succ[ii, m]) * np.log(1 - p_)) / n[ii, m]).mean())
    return float(np.mean(tot))


ours = gain(P, LC); res = {"ours": ours, "ours_nll": nll(P)}
print(f"ours: gain {ours*100:.1f}%, success log-loss {nll(P):.3f}", flush=True)
rng = np.random.default_rng(0)
for N in sorted({0, 5, 10, 20, 50, 100, 200, len(pop)}):
    if N > len(pop):
        continue
    subsets = [np.arange(len(pop))] if N == len(pop) else [rng.choice(len(pop), N, replace=False) for _ in range(3 if N else 1)]
    for D in (1, 5, 20):
        rows = []
        for sub in subsets:
            s_pop = np.nan_to_num(pop[sub][:, tr].T); n_pop = np.isfinite(pop[sub][:, tr]).T.astype(float)
            la, b, th = fit_stage1(np.c_[succ[tr], s_pop], np.c_[n[tr], n_pop], D, 0)          # pool models first, then population
            pr = RidgeCV(alphas=np.geomspace(1, 1e5, 11)).fit(Z[tr], np.c_[la, b]).predict(Z)
            A = np.exp(pr[:, :D]); B = pr[:, D:]; A[tr] = np.exp(la); B[tr] = b
            Pz = sig((A[:, None, :] * (th[:M][None] - B[:, None, :])).sum(-1))
            g = {}
            for K in (5, 10, 20):
                s = (A * B).sum(1); e = np.quantile(s[tr], np.linspace(0, 1, K + 1)[1:-1]); bn = np.searchsorted(e, s)
                tab = np.array([[np.nanmean(outm[tr[bn[tr] == j], m]) if (bn[tr] == j).any() else np.nanmean(outm[tr, m]) for j in range(K)] for m in range(M)])
                g[f"zr_K{K}"] = gain(Pz, inp * pin + tab.T[bn] * pout)
                if K == 10:
                    g["our succ + zr cost"] = gain(P, inp * pin + tab.T[bn] * pout)
            g["zr succ + our cost"] = gain(Pz, LC); g["nll"] = nll(Pz); rows.append(g)
        m_ = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        res[f"N{N}|D{D}"] = m_
        best = max(m_[f"zr_K{K}"] for K in (5, 10, 20))
        print(f"  N={N:<3} D={D:<2} success log-loss {m_['nll']:.3f} (ours {res['ours_nll']:.3f}) | zr best-K {best*100:5.1f}% "
              f"| zr success + our cost {m_['zr succ + our cost']*100:5.1f}% | our success + zr cost {m_['our succ + zr cost']*100:5.1f}% "
              f"(ours {ours*100:.1f}%)", flush=True)
json.dump(res, open(Path(__file__).parent / "zr_population.json", "w"), indent=1, default=float)
