"""ZeroRouter's item-response structure in our success probes (NEW_PATH 4.A.127). ZeroRouter's success orders the routes within a query
better than ours (4.A.125) through its 2PL structure: a query latent b_i, a query discrimination alpha_i that scales how much the routes'
abilities matter on that query, and D-dimensional route abilities theta_m that let the order of the routes change from query to query.
ZeroRouter fits that structure in two stages (item-response fit on training outcomes, then a PCA-256 ridge reader of the latents). Here
the same structure is read end to end from our full prefill features and fitted on the outcomes directly:
  logit P(route m solves query i) = beta_m + sum_d alpha_id (theta_md - b_id),  alpha_i = exp(c + A x_i),  b_i = e + B x_i
binomial over all draws, L2 on A, B and theta (c fixed: see fit_mirt), D in {1, 2, 5} and the penalties chosen by calibration log
loss pooled over routes, Platt per route as our probes. One fixed rule for every dataset, routed with the SAME cost probe (ours):
  ours        the paper's independent probes
  shared      4.A.126 (multi-task shrinkage)
  mirt        the item-response reader above, Platt per route
  mirt-raw    the same without Platt
  auto        per training set, whichever of ours / shared / mirt has the lowest calibration log loss (a deployable choice)
  zr4b+cost   ZeroRouter's success from our 4B prefill, the reference
The fit is exact on the training rows' span (success_pool.project). Reports as success_pool.py; per-problem success predictions of every
arm and our cost probe's tokens are saved (.npz) for top-of-frontier diagnostics.
Usage: (GPU) REASON_ROOT=.../reason_pinned RESULT_TAG=_pinned NBOOT=200 OUT_DIR=... python success_mirt.py POOL S0 S1
"""
import json, os, sys
from pathlib import Path
HERE = Path(__file__).parent
_sp = open(HERE / "success_pool.py").read()
exec(_sp[:_sp.index('res = {"pool": POOL_, "seeds": {}}')])            # learning_curve's shared code, project, shared, pair_acc, bll
import torch
DS = (1, 2, 5); PENS = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2); PTHS = (1.0, 0.1)


def fit_mirt(Z, succ, n, D, pen, pth, seed=0):
    """Z: projected features scaled to unit mean row norm on the training rows. Every factor that multiplies a query-dependent term is
    penalized (A, B, theta) and the discrimination scale c is fixed at ZeroRouter's prior mean log(1/sqrt(D)); a free c, or a free
    offset on b, rescales the query effects and defeats the penalty (it fitted the training draws almost exactly). beta_m is free."""
    g = torch.Generator().manual_seed(seed); N, r = Z.shape; M = succ.shape[1]; T = lambda x: torch.tensor(x, dtype=torch.float64)
    z, s_, n_ = T(Z), T(succ), T(n); tot = float(n.sum())
    rate = np.clip((succ.sum(0) + .5) / (n.sum(0) + 1), .02, .98)
    A = torch.zeros(r, D, dtype=torch.float64); B = torch.zeros(r, D, dtype=torch.float64); c = torch.full((D,), float(np.log(1 / np.sqrt(D))), dtype=torch.float64)
    th = 0.1 * torch.randn(M, D, generator=g, dtype=torch.float64); be = T(np.log(rate / (1 - rate)))
    ps = [A, B, th, be]
    for p in ps:
        p.requires_grad_(True)
    opt = torch.optim.LBFGS(ps, lr=1, max_iter=3000, history_size=20, line_search_fn="strong_wolfe")

    def loss():
        opt.zero_grad()
        lo = be[None] + (torch.exp(c + z @ A)[:, None, :] * (th[None] - (z @ B)[:, None, :])).sum(-1)
        ll = (s_ * torch.nn.functional.logsigmoid(lo) + (n_ - s_) * torch.nn.functional.logsigmoid(-lo)).sum() / tot
        L = -ll + pen * ((A ** 2).sum() + (B ** 2).sum() + pth * (th ** 2).sum())
        L.backward(); return L
    opt.step(loss)
    A_, B_, th_, be_ = [p.detach().numpy() for p in ps]; c_ = c.numpy()
    return lambda Zq: be_[None] + (np.exp(c_ + Zq @ A_)[:, None, :] * (th_[None] - (Zq @ B_)[:, None, :])).sum(-1)


def mirt(d, Z, tr, ca):
    M = d["M"]; best = None; Z = Z / np.linalg.norm(Z[tr], axis=1).mean()
    for D in DS:
        for pen in PENS:
            for pth in PTHS:
                f = fit_mirt(Z[tr], d["succ"][tr], d["n"][tr], D, pen, pth)
                v = bll(d["succ"][ca].ravel(), d["n"][ca].ravel(), sig(f(Z[ca])).ravel())
                if np.isfinite(v) and (best is None or v < best[0]):
                    best = (v, (D, pen, pth), f)
    _, (D, pen, pth), f = best; lo = f(Z)
    cal = np.stack([platt(lo[ca, k], d["y0"][ca, k], lo[:, k]) for k in range(M)], 1)
    return np.clip(cal, 1e-4, 1 - 1e-4), np.clip(sig(lo), 1e-4, 1 - 1e-4), dict(D=D, pen=pen, pth=pth)


res = {"pool": POOL_, "seeds": {}}; outf = OUT / f"success_mirt_{POOL_.lower().replace('-', '')}_{S0}_{S1}{TAG}.json"; SAVE = {}
for s in range(S0, S1):
    os.environ["DRAW"] = str(s); d = load(POOL_); N = len(d["ids"]); tr, ca, ev = d["tr"], d["ca"], d["ev"]
    S, T = view(d, tr), view(d, np.arange(N)); tk = fit_predict("ours", S, T); C = cost_of(d, tk)
    Z, scale = project(d, tr)
    Ps = {"ours": d["P"]}
    Ps["shared"], _, cfg_s = shared(d, Z, scale, tr, ca)
    Ps["mirt"], Ps["mirt-raw"], cfg_m = mirt(d, Z, tr, ca)
    cl = {a: bll(d["succ"][ca].ravel(), d["n"][ca].ravel(), Ps[a][ca].ravel()) for a in ("ours", "shared", "mirt")}
    pick = min(cl, key=cl.get); Ps["auto"] = Ps[pick]
    Ps["zr4b+cost"], _, _ = zr_4b(d, S)
    Cm = cost_of(d, np.repeat(d["med"][None], N, 0)); rb = np.random.default_rng(0); BS = [ev] + [ev[rb.integers(0, len(ev), len(ev))] for _ in range(NBOOT)]
    r_ = {a: dict(vs_median=[], vs_mix=[], ours_vs_it=[]) for a in Ps}
    for ii in BS:
        H0 = curve(d, d["P"], Cm, ii); Hm = hull([(d["paid"][ii, k].mean(), d["q"][ii, k].mean()) for k in range(d["M"])])
        H = {a: curve(d, P, C, ii) for a, P in Ps.items()}
        for a in Ps:
            r_[a]["vs_median"].append(saved(H[a], H0) * 100); r_[a]["vs_mix"].append(saved(H[a], Hm) * 100); r_[a]["ours_vs_it"].append(saved(H["ours"], H[a]) * 100)
    out = {"_cfg": dict(shared=cfg_s, mirt=cfg_m, auto=pick, cal_logloss=cl)}
    for a, v in r_.items():
        out[a] = {k: float(x[0]) for k, x in v.items()} | {k + "_bs": [float(y) for y in x[1:]] for k, x in v.items()}
        out[a]["auc"] = float(np.mean([auc(d["y0"][ev, m], Ps[a][ev, m]) for m in range(d["M"])])); out[a]["pair_acc"] = pair_acc(d["q"][ev], Ps[a][ev])
    res["seeds"][s] = out
    SAVE[f"s{s}_ev"] = ev; SAVE[f"s{s}_cost_ours_tok"] = tk.astype(np.float32)
    for a, P in Ps.items():
        SAVE[f"s{s}_{a}_P"] = P.astype(np.float32)
    print(f"  set {s}: {out['_cfg']}  " + "  ".join(f"{a}: vs median {o['vs_median']:+5.1f} AUC {o['auc']:.3f} pair {o['pair_acc']:.3f}"
                                               for a, o in out.items() if a != "_cfg"), flush=True)
    json.dump(res, open(outf, "w"), indent=1, default=float); np.savez_compressed(outf.with_suffix(".npz"), ids=np.array(d["ids"]), slots=np.array(d["slots"]), **SAVE)
print("DONE", flush=True)
