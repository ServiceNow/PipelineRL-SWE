"""Does a few-call sample pick the right provider to pin, and what does a wrong pick cost? (NEW_PATH 4.A.55; free, offline)
Backs the reframed provider paragraph: "endpoint = model + offset; a few calls fix drift and say which provider to pin".
Reuses provider_routing.py's loader, offsets and frontier (arms: pinned-P = gpt-oss routes + dsv4f pinned to P, P's offsets + price).
  (1) Regret of each pinned arm on EVAL: extra cost of pinned-P over the best pinned arm at matched accuracy (frontier over V,
      12 targets over the shared band), and the endpoints router (all providers as routes) for reference.
  (2) Selection from k calls per provider: draw k problems from the non-eval (FIT) problems, pick the provider with the lowest
      billed cost per correct dsv4f answer on those k calls; 1,000 draws. Report P(pick = eval-best) and expected regret.
Usage: python provider_pinning_check.py
"""
import json
from pathlib import Path
import numpy as np
src = open(__file__.replace("provider_pinning_check.py", "provider_routing.py")).read().split("res = {}")[0]
g = {"__file__": __file__.replace("provider_pinning_check.py", "provider_routing.py"), "__name__": "x"}; exec(src, g)
load, frontier, GO, PROV, RATE, PRATE, sig, lgt, cost_at = (g[k] for k in ("load", "frontier", "GO", "PROV", "RATE", "PRATE", "sig", "lgt", "cost_at"))
KS = [5, 10, 20, 50, 100, 200]; NREP = 1000
out = {}
for ds in [d for d, _ in g["DSETS"]]:
    ids, q, L, I, paid, p_pred, len_pred, fit, ev, _ = load(ds)
    rate_of = g["rate_of"]
    p = {s: p_pred[s] for s in GO}; c = {s: I[s] * rate_of(s)[0] + len_pred[s] * rate_of(s)[1] for s in GO}
    for pp in PROV:
        a = L[pp][fit].sum() / len_pred["dsv4f"][fit].sum(); lo, hi = -6.0, 6.0
        for _ in range(60):
            b = (lo + hi) / 2; lo, hi = (b, hi) if sig(lgt(p_pred["dsv4f"][fit]) + b).mean() < q[pp][fit].mean() else (lo, b)
        rate = PRATE.get(("dsv4f", pp), RATE["dsv4f"]); p[pp] = sig(lgt(p_pred["dsv4f"]) + (lo + hi) / 2); c[pp] = I[pp] * rate[0] + len_pred["dsv4f"] * a * rate[1]
    arms = {"endpoints": GO + PROV, **{f"pinned-{pp}": GO + [pp] for pp in PROV}}
    E = np.arange(len(ev)); H = {k: frontier(v, p, c, q, paid, ev[E]) for k, v in arms.items()}

    def extra(A, B):                       # extra cost of arm A over arm B at matched accuracy (positive = A dearer)
        Ha, Hb = H[A], H[B]; lo_, hi_ = max(Ha[0][1], Hb[0][1]), min(Ha[-1][1], Hb[-1][1])
        T = np.linspace(lo_ + .05 * (hi_ - lo_), hi_ - .05 * (hi_ - lo_), 12)
        return float(np.exp(np.nanmean(np.log([cost_at(Ha, x) / cost_at(Hb, x) for x in T])))) - 1
    pair = {(a, b): extra(f"pinned-{a}", f"pinned-{b}") for a in PROV for b in PROV}
    best = min(PROV, key=lambda a: max(pair[(a, b)] for b in PROV))     # the pinned arm no other pinned arm beats
    regret = {a: pair[(a, best)] for a in PROV}
    vs_end = {a: extra(f"pinned-{a}", "endpoints") for a in PROV}
    cpc = {a: (paid[a][fit].sum() / max(q[a][fit].sum(), 1)) for a in PROV}
    print(f"\n===== {ds}: fit {len(fit)}, eval {len(ev)}; eval-best pinned provider = {best}")
    for a in PROV:
        print(f"  pinned-{a:<13} regret vs best {regret[a]*100:+6.1f}%  vs endpoints router {vs_end[a]*100:+6.1f}%  "
              f"| FIT cost/correct {cpc[a]*1e3:.3f} m$  acc {q[a][fit].mean():.3f}")
    rng = np.random.default_rng(0); sel = {}
    for k in [k for k in KS if k <= len(fit)]:
        picks = []
        for _ in range(NREP):
            s = fit[rng.choice(len(fit), k, replace=False)]
            picks.append(min(PROV, key=lambda a: paid[a][s].sum() / max(q[a][s].sum(), 0.5)))
        reg = np.array([regret[x] for x in picks])
        sel[k] = dict(p_best=float(np.mean([x == best for x in picks])), exp_regret=float(reg.mean()), p90_regret=float(np.percentile(reg, 90)),
                      share={a: float(np.mean([x == a for x in picks])) for a in PROV})
        print(f"  k={k:>3} calls/provider: P(pick best) {sel[k]['p_best']:.3f}  expected regret {reg.mean()*100:+.2f}%  90th pct {np.percentile(reg,90)*100:+.1f}%  "
              f"picks {', '.join(f'{a} {v:.2f}' for a, v in sel[k]['share'].items())}")
    out[ds] = dict(best=best, regret=regret, vs_endpoints=vs_end, fit_cost_per_correct=cpc, selection=sel, n_fit=len(fit), n_eval=len(ev))
json.dump(out, open(Path(__file__).parent / "provider_pinning_check.json", "w"), indent=1)
