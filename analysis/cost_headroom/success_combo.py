"""ZeroRouter's success model with our cost probe (NEW_PATH 4.A.125). ZeroRouter's item-response success (a shared D-dim latent: one
difficulty / discrimination per query, one ability per model) is coupled across routes, ours is five independent logistic probes; on
BigCodeBench and CodeContests its routers match or beat ours although its success AUC is lower everywhere. Per training set (DRAW=s),
the SAME cost probe (ours) routes with each success model:
  ours          our success probes (the paper)
  zr4b+cost     ZeroRouter's success, latent read from our 4B prefill (tmlr_strengthen.zr_paper, PCA capped)
  zrpub+cost    ZeroRouter's success as published (fine-tuned DistilBERT reader)
  ens+cost      per route, the mean of our logit and ZeroRouter-4B's logit (one fixed rule for every dataset)
Saving relative to median pricing with OUR success (the tab:allest reference) and relative to a random mix of single routes, point +
NBOOT test resamples (rng 0, the same in every set); success AUC per arm.
Usage: (GPU) REASON_ROOT=.../reason_pinned RESULT_TAG=_pinned NBOOT=200 OUT_DIR=... python success_combo.py POOL S0 S1
"""
import json, os, sys
from pathlib import Path
HERE = Path(__file__).parent
_lc = open(HERE / "learning_curve.py").read()
exec(_lc[:_lc.index("texts = texts_of(POOL_); Q = encode(texts)")])      # zr_4b, zr_pub and the shared code, unchanged
texts = texts_of(POOL_)
enc = tok(texts, truncation=True, max_length=512, padding="max_length", return_tensors="pt"); Fraw = np.array([feats(x) for x in texts], np.float32)
lgt = lambda P: np.log(P / (1 - P))
res = {"pool": POOL_, "seeds": {}}; outf = OUT / f"success_combo_{POOL_.lower().replace('-', '')}_{S0}_{S1}{TAG}.json"
for s in range(S0, S1):
    os.environ["DRAW"] = str(s); d = load(POOL_); N = len(d["ids"]); tr, ca, ev = d["tr"], d["ca"], d["ev"]
    S, T = view(d, tr), view(d, np.arange(N)); tk = fit_predict("ours", S, T); C = cost_of(d, tk)
    Pz4, _, _ = zr_4b(d, S); H0c = curve(d, d["P"], cost_of(d, np.repeat(d["med"][None], N, 0)), ca); Pzp, _, _ = zr_pub(d, S, H0c)
    Ps = {"ours": d["P"], "zr4b+cost": Pz4, "zrpub+cost": Pzp, "ens+cost": np.clip(sig((lgt(d["P"]) + lgt(Pz4)) / 2), 1e-4, 1 - 1e-4)}
    Cm = cost_of(d, np.repeat(d["med"][None], N, 0)); rb = np.random.default_rng(0); BS = [ev] + [ev[rb.integers(0, len(ev), len(ev))] for _ in range(NBOOT)]
    r_ = {a: dict(vs_median=[], vs_mix=[], ours_vs_it=[]) for a in Ps}
    for ii in BS:
        H0 = curve(d, d["P"], Cm, ii); Hm = hull([(d["paid"][ii, k].mean(), d["q"][ii, k].mean()) for k in range(d["M"])])
        H = {a: curve(d, P, C, ii) for a, P in Ps.items()}
        for a in Ps:
            r_[a]["vs_median"].append(saved(H[a], H0) * 100); r_[a]["vs_mix"].append(saved(H[a], Hm) * 100); r_[a]["ours_vs_it"].append(saved(H["ours"], H[a]) * 100)
    out = {}
    for a, v in r_.items():
        out[a] = {k: float(x[0]) for k, x in v.items()} | {k + "_bs": [float(y) for y in x[1:]] for k, x in v.items()}
        out[a]["auc"] = float(np.mean([auc(d["y0"][ev, m], Ps[a][ev, m]) for m in range(d["M"])]))
    res["seeds"][s] = out
    print(f"  set {s}: " + "  ".join(f"{a}: vs median {o['vs_median']:+5.1f} vs mix {o['vs_mix']:+5.1f} AUC {o['auc']:.3f}" for a, o in out.items()), flush=True)
    json.dump(res, open(outf, "w"), indent=1, default=float)
print("DONE", flush=True)
