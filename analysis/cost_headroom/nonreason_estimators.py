"""Every cost estimator on the NON-reasoning routes, 10 training sets (NEW_PATH 4.A.113). nonreason_compare.py measured only our
readouts on the five non-reasoning routes (tab:nonreason); here every pre-generation estimator of tab:allest / tab:textemb is fitted
the same way on those routes, to tell "our readout misses it" from "no estimator we have reads it".
Per training set s (DRAW=s; the success readouts nonreason_compare.py refitted for that set are read from
nonreason_eval_20261009/<ds>/d<s>/success_preds.jsonl): problems with a valid answer on all five non-reasoning routes (training) and,
for test, also on all five reasoning routes (the common test problems of tab:nonreason). Realized cost: billed usage_cost where
recorded, else tokens x the route's billed rate (as nonreason_compare.py).
Arms (all with the same non-reasoning success readouts): ours, fromsuccess, mean, gbm, mixllm, knn (nn on calibration), zr-pricing
(ZeroRouter latent + bins, (D, K) on calibration), q4b-ridge / q4b-mixllm / q4b-knn (Qwen3-Embedding-4B), oracle (realized length).
Per arm: saving vs median pricing and ours vs it (point + NBOOT test resamples, rng 0), test log-length R2 per route. Leaky
references (not estimators): other non-reasoning routes' realized lengths, and the reasoning routes' realized lengths, on the same
problem (linear map fitted on training) -- how much of a non-reasoning route's length is a shared property of the problem.
Usage: REASON_ROOT=.../reason_pinned OUT_DIR=... python nonreason_estimators.py LCB|Omni|MMLU-Pro S0 S1
"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
POOL_, S0, S1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); sys.argv = [sys.argv[0], "nrest", POOL_]
exec(open(Path(__file__).parent / "cost_generalization.py").read().split('MODE, POOL = sys.argv[1], sys.argv[2]')[0])
OUT = Path(os.environ.get("OUT_DIR", Path(__file__).parent)); REALR = Path("/mnt/llmd/results/exps/aristides/reason")
NR = ["nr_ds4off", "nr_llama8", "nr_qw30", "nr_llama70", "nr_qw235"]; ds_ = {"Omni": "omni500", "MMLU-Pro": "mmlupro", "LCB": "lcb"}[POOL_]
r2 = lambda y, yh: float(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def latest_valid(files, rule):                    # as nonreason_compare.py
    rows = {}
    for f in files:
        if not f.exists():
            continue
        for l in open(f):
            r = json.loads(l)
            if rule == "lcb":
                good = isinstance(r.get("resolved"), bool) and (str(r.get("full_output") or "").strip() or r.get("finish_reason") == "length")
            else:
                good = r.get("finish_reason") != "error" and not r.get("error") and r.get("completion_tokens")
            if good:
                rows[str(r["problem_id"])] = r
    return rows


rateN, rowsN = [], []
for r_ in NR:
    mr = [r for d_ in ("math_pool_nonreason", "math_expand_nonreason") for ds2 in ("omni500", "mmlupro")
          for r in latest_valid([REALR / d_ / ds2 / f"{r_}_d0.jsonl"], "math").values() if r.get("usage_cost") is not None]
    A = np.array([[r["prompt_tokens"], r["completion_tokens"]] for r in mr], float); b = np.array([r["usage_cost"] for r in mr])
    rateN.append(np.maximum(np.linalg.lstsq(A, b, rcond=None)[0], 0))
    files = ([REALR / "pool_v2_lcb_nonreason" / f"{r_}_{s}_d0.jsonl" for s in ("train", "eval")] if POOL_ == "LCB"
             else [REALR / d_ / ds_ / f"{r_}_d0.jsonl" for d_ in ("math_pool_nonreason", "math_expand_nonreason")])
    rowsN.append(latest_valid(files, "lcb" if POOL_ == "LCB" else "math"))
rateN = np.array(rateN)
name_, _, dsx = POOLS[POOL_]; Fd = R / "expanded_eval_20261001" / dsx if dsx else R / name_
z4 = np.load(Fd / "qwen3emb4b.npz", allow_pickle=True); e4 = {p: i for i, p in enumerate(map(str, z4["problem_ids"]))}
res = {"pool": POOL_, "slots": NR, "rates_per_M": (rateN * 1e6).tolist(), "seeds": {}}
for s in range(S0, S1):
    os.environ["DRAW"] = str(s); d = load(POOL_); ids = d["ids"]; idx = {p: i for i, p in enumerate(ids)}; N = len(ids)
    n_ = np.zeros((N, 5), int); q = np.zeros((N, 5)); L = np.ones((N, 5)); I = np.tile(d["I"].mean(1, keepdims=True), (1, 5)); paid = np.zeros((N, 5))
    for k, rows in enumerate(rowsN):
        for p, r in rows.items():
            if p in idx:
                i = idx[p]; n_[i, k] = 1; q[i, k] = float(bool(r["resolved"])); L[i, k] = max(r["completion_tokens"], 1); I[i, k] = r["prompt_tokens"]
                paid[i, k] = r["usage_cost"] if r.get("usage_cost") is not None else I[i, k] * rateN[k, 0] + L[i, k] * rateN[k, 1]
    D = REALR / "nonreason_eval_20261009" / ds_ / f"d{s}"
    tN = np.load(D / "tensors.npz", allow_pickle=True); assert list(map(str, tN["problem_ids"])) == ids
    P = np.clip(read_predictions(D / "success_preds.jsonl", ids, "p_successes", 5), 1e-4, 1 - 1e-4)
    allv = (n_ > 0).all(1); tr, ca, ev = d["tr"][allv[d["tr"]]], d["ca"][allv[d["ca"]]], d["ev"][allv[d["ev"]]]
    CT = np.full((N, 5, 16), np.nan); CT[:, :, 0] = np.where(n_ > 0, L, np.nan)
    dN = dict(d, slots=NR, M=5, tr=tr, ca=ca, ev=ev, n=n_, succ=q * n_, q=q, L=L, I=I, rates=rateN, paid=paid, P=P, y0=q.astype(int), CT=CT,
              med=np.array([np.median(L[tr, k]) for k in range(5)]))
    S, T, Ca = view(dN, tr), view(dN, np.arange(N)), view(dN, ca); cfg = choose(dN, S, Ca)
    tk = {a: fit_predict(a, S, T, cfg.get(a)) for a in ("ours", "fromsuccess", "mean", "gbm", "mixllm", "knn")}
    tk["zr-pricing"] = fit_predict("zerorouter", S, T, cfg["zerorouter"])
    Ez = z4["emb"].astype(np.float32)[[e4[p] for p in ids]]; dE = dict(dN, X=Ez, E=Ez); SE, TE, CaE = view(dE, tr), view(dE, np.arange(N)), view(dE, ca)
    tk["q4b-ridge"] = fit_predict("ours", SE, TE); tk["q4b-mixllm"] = fit_predict("mixllm", SE, TE)
    H0c = curve(dN, P, cost_of(dN, np.repeat(dN["med"][None], N, 0)), ca); sc = {}
    for nn in (1, 3, 5, 10, 20):
        t_ = np.zeros((N, 5)); t_[ca] = fit_predict("knn", SE, CaE, nn); sc[nn] = saved(curve(dN, P, cost_of(dN, t_), ca), H0c)
    tk["q4b-knn"] = fit_predict("knn", SE, TE, max(sc, key=lambda c: -np.inf if np.isnan(sc[c]) else sc[c])); del dE, SE, TE, CaE
    pre = list(tk); tk["oracle"] = L
    C = {a: cost_of(dN, x) for a, x in tk.items()}; Cm = cost_of(dN, np.repeat(dN["med"][None], N, 0))
    rb = np.random.default_rng(0); BS = [ev] + [ev[rb.integers(0, len(ev), len(ev))] for _ in range(NBOOT)]
    Hs = [{a: curve(dN, P, C[a], ii) for a in C} | {"median": curve(dN, P, Cm, ii)} for ii in BS]; r_ = {"n_train": len(tr), "n_test": len(ev)}
    Y = np.log(L)
    for a in C:
        vm = np.array([saved(H[a], H["median"]) for H in Hs]) * 100; vd = np.array([saved(H["ours"], H[a]) for H in Hs]) * 100
        r_[a] = dict(vs_median=float(vm[0]), vs_median_bs=vm[1:].tolist(), ours_vs_it=float(vd[0]), ours_vs_it_bs=vd[1:].tolist(),
                     r2=[r2(Y[ev, k], np.log(np.maximum(tk[a][ev, k], 1))) for k in range(5)])
    YR = np.log(d["L"]); leaky = {"other_nr_routes": lambda k: np.delete(Y, k, 1).mean(1), "reasoning_routes": lambda k: YR.mean(1)}
    for nm, f in leaky.items():
        r_[nm] = []
        for k in range(5):
            x = f(k); A_ = np.c_[np.ones(len(tr)), x[tr]]; b_ = np.linalg.lstsq(A_, Y[tr, k], rcond=None)[0]; r_[nm].append(r2(Y[ev, k], b_[0] + b_[1] * x[ev]))
    r_["best_pre"] = [max(pre, key=lambda a: r_[a]["r2"][k]) for k in range(5)]; res["seeds"][s] = r_
    pr = lambda a: f"{a} {r_[a]['vs_median']:+5.1f}"
    print(f"  set {s} (train {len(tr)}, test {len(ev)}): vs median " + " ".join(pr(a) for a in C) + "\n    R2 per route: " + "  ".join(
        f"{sl}: ours {r_['ours']['r2'][k]:.2f} best {r_[r_['best_pre'][k]]['r2'][k]:.2f} ({r_['best_pre'][k]}) other-nr {r_['other_nr_routes'][k]:.2f} "
        f"reasoning {r_['reasoning_routes'][k]:.2f}" for k, sl in enumerate(NR)), flush=True)
    json.dump(res, open(OUT / f"nonreason_estimators_{ds_}_{S0}_{S1}{os.environ.get('RESULT_TAG', '')}.json", "w"), indent=1, default=float)
print("DONE", flush=True)
