"""What the data says about per-query cost, with no predictor (NEW_PATH 4.A.109).
(a) Two-way variance decomposition of log output length per draw, problems x routes x draws (random problem effect, fixed route
    effect, method of moments, unbalanced draws via the harmonic mean draw count n_h):
      route        variance of the route means (some models are longer; median pricing already knows this)
      problem      shared level: long for every route (difficulty / work)
      interaction  problem x route: long for this route but not that one (what choosing between routes needs)
      noise        within-cell, run-to-run (no predictor can read it)
    E[var(problem means)] = s_b + s_ab / K + s_e / (K n_h);  E[resid MS] = s_ab + s_e / n_h. Computed for the full five-route pool and
    for every subset of tmlr_strengthen.py drop (full, leave-one-out, pairs), on all problems with a valid draw on every route of the
    subset; noise from cells with >= 2 draws (APPS has one draw per problem: noise n/a, interaction includes noise). Price spread: s.d.
    over the subset's routes of log mean billed cost per call.
(b) Headroom by source, on the 10 training sets with our success readouts: oracles that know each test problem's realized shared level
    only (level_i + the route's training-mean offset), its route-specific part only (the training-mean level + realized offsets), or
    both (the paper's headroom); each smeared and level-matched to the route's training mean like the readouts. Saving vs median.
Usage: REASON_ROOT=.../reason_pinned OUT_DIR=... python data_decomp.py POOL
"""
import json, os, sys, itertools
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
POOL_ = sys.argv[1]; sys.argv = [sys.argv[0], "decomp", POOL_]
exec(open(Path(__file__).parent / "cost_generalization.py").read().split('MODE, POOL = sys.argv[1], sys.argv[2]')[0])
OUT = Path(os.environ.get("OUT_DIR", Path(__file__).parent))
name_, _, ds_ = POOLS[POOL_]; Fd = R / "expanded_eval_20261001" / ds_ if ds_ else R / name_
t = np.load(Fd / "tensors.npz", allow_pickle=True); v = t["valid"].astype(bool)
LD = np.where(v, np.log(np.maximum(t["completion_tokens"], 1)), np.nan).astype(float); nD = v.sum(2)
os.environ.pop("DRAW", None); d0 = load(POOL_); slots = d0["slots"]; M = len(slots); assert d0["ids"] == list(map(str, t["problem_ids"]))


def decomp(cols):
    ok = (nD[:, cols] >= 1).all(1); x = LD[ok][:, cols]; n = nD[ok][:, cols].astype(float); m = np.nanmean(x, 2); N, K = m.shape
    rep = n >= 2
    s_e = float(np.nansum(np.nanvar(x, 2, ddof=1)[rep] * (n[rep] - 1)) / max((n[rep] - 1).sum(), 1)) if rep.sum() >= 50 else None
    nh = float(len(n.ravel()) / (1 / n).sum()); mu = m.mean(); a = m.mean(0) - mu; b = m.mean(1) - mu; e = m - mu - a[None] - b[:, None]
    ms_e = float((e ** 2).sum() / ((N - 1) * (K - 1))); s_ab = ms_e - (s_e / nh if s_e is not None else 0)
    s_b = float(b.var(ddof=1)) - s_ab / K - (s_e / (K * nh) if s_e is not None else 0)
    paid = d0["paid"][:, cols]; ps = float(np.log(paid[ok].mean(0)).std())
    return dict(route=float(a.var()), problem=s_b, interaction=s_ab, noise=s_e, price_spread=ps, n_problems=int(N), K=K,
                corr_mean=float(np.mean([np.corrcoef(m[:, i], m[:, j])[0, 1] for i, j in itertools.combinations(range(K), 2)])))


subsets = {"full": list(range(M))} | {"-" + slots[k]: [j for j in range(M) if j != k] for k in range(M)} | {"+".join(slots[j] for j in c): list(c) for c in itertools.combinations(range(M), 2)}
res = {"pool": POOL_, "slots": slots, "decomp": {k: decomp(c) for k, c in subsets.items()}, "oracle_split": {}}
f = res["decomp"]["full"]; tot = sum(f[k] or 0 for k in ("problem", "interaction", "noise"))
print(f"===== {POOL_}: {f['n_problems']} problems; log-length variance (excluding route means): problem {f['problem']:.3f} ({f['problem'] / tot:.0%})  interaction "
      f"{f['interaction']:.3f} ({f['interaction'] / tot:.0%})  noise " + ("n/a" if f["noise"] is None else f"{f['noise']:.3f} ({f['noise'] / tot:.0%})")
      + f"; route-mean variance {f['route']:.3f}; price spread {f['price_spread']:.2f}; mean corr of route lengths {f['corr_mean']:.2f}", flush=True)
for s in range(10):
    os.environ["DRAW"] = str(s); d = load(POOL_); N = len(d["ids"]); tr, ev = d["tr"], d["ev"]; Y = np.log(d["L"])
    lvl = Y.mean(1); dif = Y - lvl[:, None]
    def tok(Yh):
        o = np.exp(Yh); o *= np.mean(np.exp(Y[tr] - Yh[tr]), 0)[None]; return o * (d["L"][tr].mean(0) / o[tr].mean(0))[None]
    arms = {"both": d["L"], "level_only": tok(lvl[:, None] + dif[tr].mean(0)[None]), "route_specific_only": tok(lvl[tr].mean() + dif)}
    Cm = cost_of(d, np.repeat(d["med"][None], N, 0)); H0 = curve(d, d["P"], Cm, ev)
    res["oracle_split"][s] = {a: float(saved(curve(d, d["P"], cost_of(d, x), ev), H0) * 100) for a, x in arms.items()}
    print(f"  set {s}: headroom (both) {res['oracle_split'][s]['both']:+5.1f}  level only {res['oracle_split'][s]['level_only']:+5.1f}  route-specific only "
          f"{res['oracle_split'][s]['route_specific_only']:+5.1f}", flush=True)
json.dump(res, open(OUT / f"data_decomp_{POOL_.lower().replace('-', '')}{os.environ.get('RESULT_TAG', '')}.json", "w"), indent=1, default=float)
print("DONE", flush=True)
