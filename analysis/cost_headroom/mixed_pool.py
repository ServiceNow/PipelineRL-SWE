"""One router for mixed traffic (NEW_PATH 4.A.103): the nine pools' problems together (Omni-MATH as its 500-problem pool, whose
prefill matches the others'), the same five routes. Train / calibration = the union of every pool's train / calibration split
(DRAW=s training set per pool); test = an equal-weight sample of each pool's test split (CAP problems per pool, rng 0), so no
benchmark dominates. Success readouts (paper recipe) and our cost readout are refitted on the union. Arms, all with the same
success predictions:
  ours            prefill cost readout trained on the mixture
  median          one median length per route (the paper's rule, over the mixture)
  median_by_ds    median length per (route, benchmark): a router that knows which benchmark a query comes from; the control for
                  "segmenting by dataset" -- ours vs it is the per-query value beyond benchmark identity
  fromsuccess_by_ds, bins_by_ds  (4.A.105) difficulty-aware pricing fitted within each benchmark (ridge on the success logits;
                  ten bins of the mean success logit), level-matched per benchmark; fromsuccess: the same over the whole mixture
  mean_by_ds      mean length per (route, benchmark)
  oracle          realized length (headroom)
Saving vs median and vs median_by_ds, and ours vs each arm; point + NBOOT test resamples stratified by pool (rng 0).
Usage: REASON_ROOT=.../reason_pinned RESULT_TAG=_pinned OUT_DIR=... DRAW=s python mixed_pool.py
"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
sys.argv = [sys.argv[0], "mixed", "ALL"]
exec(open(Path(__file__).parent / "cost_generalization.py").read().split('MODE, POOL = sys.argv[1], sys.argv[2]')[0])
OUT = Path(os.environ.get("OUT_DIR", Path(__file__).parent)); CAP = int(os.environ.get("CAP", "150"))
PL = ["LCB", "APPS", "BCB", "CC", "Omni500", "MMLU-Pro", "AIME", "SuperGPQA", "BBEH"]
parts, slots0 = [], None
for p in PL:
    d = load(p); slots0 = slots0 or d["slots"]; assert d["slots"] == slots0, (p, d["slots"])
    parts.append(d); print(f"  {p}: train {len(d['tr'])} cal {len(d['ca'])} test {len(d['ev'])} features {d['X'].shape[1]}", flush=True)
assert len({d["X"].shape[1] for d in parts}) == 1, "prefill feature dims differ across pools"
off = np.cumsum([0] + [len(d["ids"]) for d in parts]); N = off[-1]; M = len(slots0)
cat = lambda k: np.concatenate([d[k] for d in parts])
D = dict(ids=[f"{p}:{i}" for p, d in zip(PL, parts) for i in d["ids"]], slots=slots0, M=M, rates=parts[0]["rates"],
         **{k: cat(k) for k in ("n", "succ", "q", "L", "I", "paid", "X", "E", "TF", "y0")})
D["CT"] = np.concatenate([d["CT"] for d in parts]); D["P"] = np.full((N, M), .5)
pool_of = np.concatenate([np.full(len(d["ids"]), j) for j, d in enumerate(parts)])
tr = np.concatenate([d["tr"] + off[j] for j, d in enumerate(parts)]); ca = np.concatenate([d["ca"] + off[j] for j, d in enumerate(parts)])
rng = np.random.default_rng(0)
evs = [np.sort(rng.choice(d["ev"], min(CAP, len(d["ev"])), replace=False)) + off[j] for j, d in enumerate(parts)]; ev = np.concatenate(evs)
D["tr"], D["ca"], D["ev"] = tr, ca, ev; del parts
A = dict(X=D["X"], y0=D["y0"], succ=D["succ"], n=D["n"]); sub = lambda ii: {k: x[ii] for k, x in A.items()}
D["P"] = np.clip(success_linear("X", sub(tr), sub(ca), sub(np.arange(N))), 1e-4, 1 - 1e-4)
D["med"] = np.array([np.nanmedian(D["CT"][tr, k]) for k in range(M)])
print(f"===== mixed traffic, training set {os.environ.get('DRAW', 'manifest')}: train {len(tr)} cal {len(ca)} test {len(ev)} "
      f"({CAP} per pool cap); success AUC {np.mean([auc(D['y0'][ev, k], D['P'][ev, k]) for k in range(M)]):.3f}", flush=True)
tk = {"ours": fit_predict("ours", view(D, tr), view(D, np.arange(N))), "median": np.repeat(D["med"][None], N, 0), "oracle": D["L"]}
mbd, abd = np.zeros((N, M)), np.zeros((N, M))
for j in range(len(PL)):
    tj, ij = tr[pool_of[tr] == j], pool_of == j
    mbd[ij] = np.array([np.nanmedian(D["CT"][tj, k]) for k in range(M)])[None]; abd[ij] = D["L"][tj].mean(0)[None]
tk["median_by_ds"], tk["mean_by_ds"] = mbd, abd
# (4.A.105) difficulty-aware pricing WITHIN each benchmark: a router that knows the benchmark and prices from the success readouts there
fsd, bnd = np.zeros((N, M)), np.zeros((N, M)); lgD = np.log(D["P"] / (1 - D["P"]))
for j in range(len(PL)):
    tj, ij = tr[pool_of[tr] == j], np.where(pool_of == j)[0]
    fsd[ij] = fit_predict("fromsuccess", view(D, tj), view(D, ij))
    db = lgD.mean(1); e_ = np.quantile(db[tj], np.linspace(0, 1, 11)[1:-1]); b_ = np.searchsorted(e_, db[ij]); bt = np.searchsorted(e_, db[tj])
    zb = np.stack([np.array([D["L"][tj][bt == q, k].mean() if (bt == q).any() else D["L"][tj, k].mean() for q in range(10)])[b_] for k in range(M)], 1)
    bnd[ij] = zb * (D["L"][tj].mean(0) / np.stack([np.array([D["L"][tj][bt == q, k].mean() if (bt == q).any() else D["L"][tj, k].mean() for q in range(10)])[bt] for k in range(M)], 1).mean(0))
tk["fromsuccess_by_ds"], tk["bins_by_ds"] = fsd, bnd
tk["fromsuccess"] = fit_predict("fromsuccess", view(D, tr), view(D, np.arange(N)))
C = {a: cost_of(D, x) for a, x in tk.items()}
BS = [ev] + [np.concatenate([e[rng.integers(0, len(e), len(e))] for e in evs]) for _ in range(NBOOT)]
Hs = [{a: curve(D, D["P"], C[a], ii) for a in C} for ii in BS]
res = {"draw": os.environ.get("DRAW"), "cap": CAP, "n": dict(train=len(tr), calibration=len(ca), test=len(ev)), "arms": {}}
for a in C:
    res["arms"][a] = {f"vs_{b}": [float(saved(H[a], H[b]) * 100) for H in Hs] for b in ("median", "median_by_ds")} | {"ours_vs_it": [float(saved(H["ours"], H[a]) * 100) for H in Hs]}
pct = lambda v: f"{v[0]:+5.1f} [{np.nanpercentile(v[1:], 2.5):+.1f}, {np.nanpercentile(v[1:], 97.5):+.1f}]"
for a, r in res["arms"].items():
    print(f"  {a:<13} vs median {pct(r['vs_median'])}  vs median_by_ds {pct(r['vs_median_by_ds'])}  ours vs it {pct(r['ours_vs_it'])}", flush=True)
json.dump(res, open(OUT / f"mixed_pool{os.environ.get('RESULT_TAG', '')}.json", "w"), indent=1, default=float)
print("DONE", flush=True)
