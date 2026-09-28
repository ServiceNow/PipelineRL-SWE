"""Can the cost head be boosted? Free CPU tests on the frozen 4B Instruct probe (the scout), five pools.
Target: log mean output tokens per (problem, route); metric: TEST log-output R2 per route (and the routing gain
via decompose.py for the variants written out).
  base        plain RidgeCV per route (the reference, = baseline_cost_heads 'probe')
  curve       learning curve: the same head trained on 25 / 50 / 75 / 100% of the TRAIN problems -> data-limited?
  pooled      one ridge trained on the train problems of ALL pools (per-route targets matched by route label; a
              pool indicator appended) -> does data from other datasets help?
  lowrank     shared-factor multi-output head: reduced-rank ridge (predict the 5 routes jointly through k latent
              factors, k chosen on calibration) -> denoises routes with few draws
  knn         nearest-neighbour cost: mean log length of the k most similar TRAIN problems in (standardised, PCA-256)
              probe space, k on calibration; and knn+ridge averaged
Writes cost_preds_boost_<variant>.jsonl (market prices) for the best variants.
"""
import json, sys, numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).parent))
from decompose import MK, R
from baseline_cost_heads import rich

POOLS = {"pool_v2_tensors_5rung": R / "pv2_scout_prefill_1756715297/scout.npz", "cc_tensors": R / "cc_pool/scout_prefill.npz",
         "taco_tensors_ha": R / "taco_activations_1788500841/scout.npz", "bcb_tensors_5r": R / "bcb_scout_prefill.npz",
         "omni500_tensors": R / "omni500_probe/instruct.npz"}
ALPHAS = np.geomspace(1e1, 1e7, 13)


def load(name):
    D = R / name; t = np.load(D / "tensors.npz", allow_pickle=True)
    S = [str(s) for s in t["model_slots"]]; pids = [str(p) for p in t["problem_ids"]]; pi = {p: i for i, p in enumerate(pids)}
    v = t["valid"].astype(bool); ct = t["completion_tokens"].astype(float); pt = t["prompt_tokens"].astype(float); n = v.sum(2)
    Y = np.log(np.maximum(np.where(n > 0, np.where(v, ct, 0).sum(2) / np.maximum(n, 1), np.nan), 1.0))
    Y[n == 0] = np.nan
    sp = json.load(open(D / "split_manifest.json"))
    idx = {k: np.array([pi[str(p)] for p in sp[f"{k}_problem_ids"]]) for k in ("train", "calibration", "test")}
    X = rich(POOLS[name], pids)
    return dict(D=D, S=S, pids=pids, Y=Y, X=X, idx=idx, inp=np.nanmean(np.where(v, pt, np.nan), 2))


def r2(y, p):
    ok = np.isfinite(y); return 1 - ((y[ok] - p[ok]) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum()


def ridge_fit(X, y, tr):
    ok = tr[np.isfinite(y[tr])]; sc = StandardScaler().fit(X[ok])
    m = RidgeCV(alphas=ALPHAS).fit(sc.transform(X[ok]), y[ok]); return sc.transform(X) @ m.coef_ + m.intercept_


def write(d, tag, P):
    tr = d["idx"]["train"]; C = np.zeros_like(P)
    for m, s in enumerate(d["S"]):
        ok = tr[np.isfinite(d["Y"][tr, m])]
        o = np.exp(P[:, m]) * np.mean(np.exp(d["Y"][ok, m] - P[ok, m])); o *= np.exp(d["Y"][ok, m]).mean() / o[ok].mean()
        C[:, m] = np.nan_to_num(d["inp"][:, m], nan=np.nanmean(d["inp"][:, m])) * MK[s][0] / 1e6 + o * MK[s][1] / 1e6
    with open(d["D"] / f"cost_preds_boost_{tag}.jsonl", "w") as f:
        for i, p in enumerate(d["pids"]):
            f.write(json.dumps({"problem_id": p, "expected_costs": [float(x) for x in C[i]]}) + "\n")


def main():
    data = {k: load(k) for k in POOLS}
    rng = np.random.default_rng(0); res = {}
    for name, d in data.items():
        X, Y, tr, cal, te = d["X"], d["Y"], d["idx"]["train"], d["idx"]["calibration"], d["idx"]["test"]
        M = Y.shape[1]; out = {}
        base = np.stack([ridge_fit(X, Y[:, m], tr) for m in range(M)], 1)
        out["base"] = [r2(Y[te, m], base[te, m]) for m in range(M)]
        # learning curve
        for frac in (0.25, 0.5, 0.75):
            sub = rng.permutation(tr)[: int(frac * len(tr))]
            P = np.stack([ridge_fit(X, Y[:, m], sub) for m in range(M)], 1)
            out[f"curve{int(frac*100)}"] = [r2(Y[te, m], P[te, m]) for m in range(M)]
        # reduced-rank ridge: ridge to all routes jointly, then project predictions on the top-k SVD directions of
        # the fitted train predictions (k chosen on calibration)
        best = None
        for k in (1, 2, 3):
            U, s_, Vt = np.linalg.svd(base[tr] - base[tr].mean(0), full_matrices=False)
            Pk = base.mean(0) * 0 + base[tr].mean(0) + (base - base[tr].mean(0)) @ Vt[:k].T @ Vt[:k]
            sc_ = np.nanmean([r2(Y[cal, m], Pk[cal, m]) for m in range(M)])
            best = max(best or (sc_, k, Pk), (sc_, k, Pk), key=lambda z: z[0])
        out[f"lowrank(k={best[1]})"] = [r2(Y[te, m], best[2][te, m]) for m in range(M)]
        # kNN in PCA space, and kNN + ridge average
        Z = PCA(256, random_state=0).fit(StandardScaler().fit_transform(X[tr])).transform(StandardScaler().fit(X[tr]).transform(X))
        kbest = None
        for k in (5, 10, 20, 40):
            Pk = np.stack([KNeighborsRegressor(k, weights="distance").fit(Z[tr[np.isfinite(Y[tr, m])]], Y[tr[np.isfinite(Y[tr, m])], m]).predict(Z)
                           for m in range(M)], 1)
            sc_ = np.nanmean([r2(Y[cal, m], Pk[cal, m]) for m in range(M)])
            kbest = max(kbest or (sc_, k, Pk), (sc_, k, Pk), key=lambda z: z[0])
        out[f"knn(k={kbest[1]})"] = [r2(Y[te, m], kbest[2][te, m]) for m in range(M)]
        avg = 0.5 * (base + kbest[2]); out["knn+ridge"] = [r2(Y[te, m], avg[te, m]) for m in range(M)]
        write(d, "lowrank", best[2]); write(d, "knnridge", avg)
        res[name] = dict(routes=d["S"], **out)
    # pooled across datasets, by route label (all pools share the 5 LCB routes except TACO)
    labels = ["oss20lo", "oss20md", "dsv4f", "oss120md", "oss120hi"]
    for target in data:
        if data[target]["S"] != labels:
            continue
        d = data[target]; te = d["idx"]["test"]; P = np.zeros_like(d["Y"])
        pools = [k for k in data if data[k]["S"] == labels]
        for m in range(len(labels)):
            Xs, ys = [], []
            for k in pools:
                dk = data[k]; trk = dk["idx"]["train"]; ok = trk[np.isfinite(dk["Y"][trk, m])]
                ind = np.zeros((len(ok), len(pools))); ind[:, pools.index(k)] = 1
                sc = StandardScaler().fit(dk["X"][dk["idx"]["train"]])       # standardise within pool
                Xs.append(np.c_[sc.transform(dk["X"][ok]), ind * 10]); ys.append(dk["Y"][ok, m])
            mdl = RidgeCV(alphas=ALPHAS).fit(np.concatenate(Xs), np.concatenate(ys))
            ind = np.zeros((len(d["pids"]), len(pools))); ind[:, pools.index(target)] = 1
            sc = StandardScaler().fit(d["X"][d["idx"]["train"]])
            P[:, m] = mdl.predict(np.c_[sc.transform(d["X"]), ind * 10])
        res[target]["pooled"] = [r2(d["Y"][te, m], P[te, m]) for m in range(len(labels))]
        write(d, "pooled", P)
    json.dump(res, open("analysis/cost_headroom/boost_cost_head.json", "w"), indent=1, default=float)
    for name, r in res.items():
        print(f"== {name}  routes {r['routes']}")
        for k, v in r.items():
            if k != "routes":
                print(f"   {k:<14}" + "  ".join(f"{x:+.2f}" for x in v) + f"   mean {np.mean(v):+.3f}")


if __name__ == "__main__":
    main()
