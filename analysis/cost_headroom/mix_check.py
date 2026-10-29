"""Every estimator against a random mix of single routes (NEW_PATH 4.A.123): the question "where our router does not beat the mix (APPS,
CodeContests, BigCodeBench), does any other estimator?". A random mix with fixed probabilities reaches any point on the convex hull of the
single routes' (cost, accuracy) points without a router (tmlr_strengthen.py `singles`, the _mix entry). Per training set (DRAW=s), every
arm of tab:allest routes with its own success predictions (ours for the cost-only arms; ZeroRouter's own for zr_4b / zr_pub) and its
saving relative to the mix's hull is computed at matched accuracy, point + NBOOT test resamples (rng 0, the same in every set).
Arms: ours, median, mean, fromsuccess, zr-bins, gbm, mixllm, knn, zr-pricing, scx, rtr, zr_4b, zr_pub, oracle.
Usage: (GPU) REASON_ROOT=.../reason_pinned RESULT_TAG=_pinned NBOOT=200 OUT_DIR=... python mix_check.py POOL S0 S1
"""
import json, os, sys
from pathlib import Path
HERE = Path(__file__).parent
_lc = open(HERE / "learning_curve.py").read()
exec(_lc[:_lc.index("texts = texts_of(POOL_); Q = encode(texts)")])      # every estimator of learning_curve.py, unchanged
texts = texts_of(POOL_); Q = encode(texts)
enc = tok(texts, truncation=True, max_length=512, padding="max_length", return_tensors="pt"); Fraw = np.array([feats(x) for x in texts], np.float32)
res = {"pool": POOL_, "seeds": {}}; outf = OUT / f"mix_check_{POOL_.lower().replace('-', '')}_{S0}_{S1}{TAG}.json"
for s in range(S0, S1):
    os.environ["DRAW"] = str(s); d = load(POOL_); N = len(d["ids"]); tr, ca, ev = d["tr"], d["ca"], d["ev"]
    S, T, Ca = view(d, tr), view(d, np.arange(N)), view(d, ca); cfg = choose(d, S, Ca); tk, Ps = {}, {}
    for a in ("ours", "median", "mean", "fromsuccess", "gbm", "mixllm", "knn"):
        tk[a] = fit_predict(a, S, T, cfg.get(a))
    tk["zr-pricing"] = fit_predict("zerorouter", S, T, cfg["zerorouter"]); tk["zr-bins"] = bins(d, d["P"], tr)
    tk["scx"] = scx(d); tk["rtr"], _ = rtr(d, Q, False)
    Ps["zr_4b"], tk["zr_4b"], _ = zr_4b(d, S)
    H0c = curve(d, d["P"], cost_of(d, np.repeat(d["med"][None], N, 0)), ca); Ps["zr_pub"], tk["zr_pub"], _ = zr_pub(d, S, H0c)
    tk["oracle"] = d["L"]; C = {a: cost_of(d, x) for a, x in tk.items()}; Pa = lambda a: Ps.get(a, d["P"])
    rb = np.random.default_rng(0); BS = [ev] + [ev[rb.integers(0, len(ev), len(ev))] for _ in range(NBOOT)]; r_ = {a: [] for a in C}
    for ii in BS:
        Hm = hull([(d["paid"][ii, k].mean(), d["q"][ii, k].mean()) for k in range(d["M"])])
        for a in C:
            r_[a].append(saved(curve(d, Pa(a), C[a], ii), Hm) * 100)
    res["seeds"][s] = {a: dict(vs_mix=float(v[0]), vs_mix_bs=[float(x) for x in v[1:]]) for a, v in r_.items()}
    print(f"  set {s}: vs mix " + "  ".join(f"{a} {v[0]:+5.1f}" for a, v in r_.items()), flush=True)
    json.dump(res, open(outf, "w"), indent=1, default=float)
print("DONE", flush=True)
