"""Endpoints router vs the BEST single pinned provider, directly at matched accuracy (NEW_PATH 4.A.49). The "best" provider is chosen
on the FIT problems (lowest billed cost per correct dsv4f answer), not on evaluation. Reuses provider_routing.py's loader, offsets
and frontier (arms: endpoints = gpt-oss routes + every pinned dsv4f provider; pinned-P = gpt-oss routes + dsv4f pinned to P).
Paired problem bootstrap (300).
Usage: python provider_best_pinned.py
"""
import numpy as np
src = open(__file__.replace("provider_best_pinned.py", "provider_routing.py")).read().split("res = {}")[0]
g = {"__file__": __file__.replace("provider_best_pinned.py", "provider_routing.py"), "__name__": "x"}; exec(src, g)
load, frontier, GO, PROV, RATE, PRATE, sig, lgt, cost_at = (g[k] for k in ("load", "frontier", "GO", "PROV", "RATE", "PRATE", "sig", "lgt", "cost_at"))
for ds in [d for d, _ in g["DSETS"]]:
    ids, q, L, I, paid, p_pred, len_pred, fit, ev, _ = load(ds)
    rate_of = g["rate_of"]
    p = {s: p_pred[s] for s in GO}; c = {s: I[s] * rate_of(s)[0] + len_pred[s] * rate_of(s)[1] for s in GO}
    for pp in PROV:
        a = L[pp][fit].sum() / len_pred["dsv4f"][fit].sum(); lo, hi = -6.0, 6.0
        for _ in range(60):
            b = (lo + hi) / 2; lo, hi = (b, hi) if sig(lgt(p_pred["dsv4f"][fit]) + b).mean() < q[pp][fit].mean() else (lo, b)
        rate = PRATE.get(("dsv4f", pp), RATE["dsv4f"]); p[pp] = sig(lgt(p_pred["dsv4f"]) + (lo + hi) / 2); c[pp] = I[pp] * rate[0] + len_pred["dsv4f"] * a * rate[1]
    best_fit = min(PROV, key=lambda pp: paid[pp][fit].sum() / max(q[pp][fit].sum(), 1))
    arms = {"endpoints": GO + PROV, **{f"pinned-{pp}": GO + [pp] for pp in PROV}}

    def saved(A, B, sub):
        e = ev[sub]; Ha, Hb = frontier(arms[A], p, c, q, paid, e), frontier(arms[B], p, c, q, paid, e)
        lo_, hi_ = max(Ha[0][1], Hb[0][1]), min(Ha[-1][1], Hb[-1][1]); T = np.linspace(lo_ + .05 * (hi_ - lo_), hi_ - .05 * (hi_ - lo_), 12)
        return 1 - float(np.exp(np.nanmean(np.log([cost_at(Ha, x) / cost_at(Hb, x) for x in T])))), (lo_, hi_)
    rng = np.random.default_rng(1); BS = [rng.integers(0, len(ev), len(ev)) for _ in range(300)]
    print(f"\n{ds}: best provider chosen on FIT problems = {best_fit}")
    for pp in PROV:
        gsv, band = saved("endpoints", f"pinned-{pp}", np.arange(len(ev))); bs = [saved("endpoints", f"pinned-{pp}", b)[0] for b in BS]
        print(f"  endpoints router vs pinned-{pp:<13} cost saved {gsv*100:+6.1f}% [{np.percentile(bs,2.5)*100:+.1f}, {np.percentile(bs,97.5)*100:+.1f}]  over accuracy {band[0]*100:.1f}-{band[1]*100:.1f}%")
