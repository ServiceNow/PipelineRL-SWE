"""How legible is each route's length from the prompt, independent of OUR predictor? (NEW_PATH 4.A.108)
Per pool, route and training set (DRAW=s, readouts refitted as in tmlr_strengthen.py draws): test R2 of log mean output length for
every pre-generation estimator we have -- ours (4B prefill ridge), pricing from our success readouts, ZeroRouter bins (its latent read
from the prefill), prompt-feature GBM, MixLLM-style (jina), kNN (jina), Qwen3-Embedding-4B ridge / MixLLM / kNN, and the fine-tuned
Qwen3-0.6B where stored (LCB) -- and the best of them per route (an empirical LOWER bound on legibility from the prompt), next to the
noise ceiling of legibility.py (an UPPER bound for any predictor). Also two leaky references that are not pre-generation estimators:
the problem's true solve rate over the routes (difficulty, with its square) and benchmark-free mean length of the other routes on the
same problem (what other models' realized lengths say).
Usage: REASON_ROOT=.../reason_pinned OUT_DIR=... python best_estimators.py POOL [S0 S1]
"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
POOL_ = sys.argv[1]; S0, S1 = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 10)
sys.argv = [sys.argv[0], "bestest", POOL_]
exec(open(Path(__file__).parent / "cost_generalization.py").read().split('MODE, POOL = sys.argv[1], sys.argv[2]')[0])
OUT = Path(os.environ.get("OUT_DIR", Path(__file__).parent)); FT = Path(os.environ.get("FT_DIR", "/mnt/llmd/results/exps/aristides/reason/reason_pinned_logs/draws"))
r2 = lambda y, yh: float(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())
name_, _, ds_ = POOLS[POOL_]; Fd = R / "expanded_eval_20261001" / ds_ if ds_ else R / name_
z = np.load(Fd / "qwen3emb4b.npz", allow_pickle=True); eid = {p: i for i, p in enumerate(map(str, z["problem_ids"]))}
res = {"pool": POOL_, "seeds": {}}
for s in range(S0, S1):
    os.environ["DRAW"] = str(s); d = load(POOL_); N = len(d["ids"]); tr, ca, ev = d["tr"], d["ca"], d["ev"]; M = d["M"]
    S, T, Ca = view(d, tr), view(d, np.arange(N)), view(d, ca); cfg = choose(d, S, Ca)
    tk = {a: fit_predict(a, S, T, cfg.get(a)) for a in ("ours", "fromsuccess", "gbm", "mixllm", "knn")}
    tk["zr-bins"] = fit_predict("zerorouter", S, T, cfg["zerorouter"])
    Ez = z["emb"].astype(np.float32)[[eid[p] for p in d["ids"]]]; dE = dict(d, X=Ez, E=Ez); SE, TE, CaE = view(dE, tr), view(dE, np.arange(N)), view(dE, ca)
    tk["q4b-ridge"] = fit_predict("ours", SE, TE); tk["q4b-mixllm"] = fit_predict("mixllm", SE, TE)
    H0c = curve(d, d["P"], cost_of(d, np.repeat(d["med"][None], N, 0)), ca); sc = {}
    for nn in (1, 3, 5, 10, 20):
        t_ = np.zeros((N, M)); t_[ca] = fit_predict("knn", SE, CaE, nn); sc[nn] = saved(curve(d, d["P"], cost_of(d, t_), ca), H0c)
    tk["q4b-knn"] = fit_predict("knn", SE, TE, max(sc, key=lambda c: -np.inf if np.isnan(sc[c]) else sc[c]))
    ftf = FT / f"ft_cost_{POOL_}_qwen06_d{s}.jsonl"
    if ftf.exists():
        tk["ft-qwen06"] = read_predictions(ftf, d["ids"], "pred_tokens", M)
    Y = np.log(d["L"]); pre = list(tk)
    leaky = {"true_difficulty": fit_predict("oracle_diff", S, T)}
    other = np.stack([np.delete(Y, k, 1).mean(1) for k in range(M)], 1)           # other routes' realized mean log length, same problem
    r_ = {"r2": {a: [r2(Y[ev, k], np.log(np.maximum(x[ev, k], 1))) for k in range(M)] for a, x in {**tk, **leaky}.items()}}
    r_["r2"]["other_routes_realized"] = []
    for k in range(M):
        A = np.c_[np.ones(len(tr)), other[tr, k]]; b = np.linalg.lstsq(A, Y[tr, k], rcond=None)[0]; r_["r2"]["other_routes_realized"].append(r2(Y[ev, k], b[0] + b[1] * other[ev, k]))
    r_["best_pre"] = [max(pre, key=lambda a: r_["r2"][a][k]) for k in range(M)]; r_["best_pre_r2"] = [r_["r2"][r_["best_pre"][k]][k] for k in range(M)]
    res["seeds"][s] = r_; res["slots"] = d["slots"]
    print(f"  set {s}: " + "  ".join(f"{sl}: ours {r_['r2']['ours'][k]:.2f} best {r_['best_pre_r2'][k]:.2f} ({r_['best_pre'][k]}) "
                                     f"difficulty {r_['r2']['true_difficulty'][k]:.2f} other-routes {r_['r2']['other_routes_realized'][k]:.2f}" for k, sl in enumerate(d["slots"])), flush=True)
    json.dump(res, open(OUT / f"best_estimators_{POOL_.lower().replace('-', '')}_{S0}_{S1}{os.environ.get('RESULT_TAG', '')}.json", "w"), indent=1, default=float)
print("DONE", flush=True)
