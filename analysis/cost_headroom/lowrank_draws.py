"""Is the prefill's cost signal just difficulty? A shared rank-k bottleneck for success AND cost, on the 10 training sets (NEW_PATH 4.A.117).
z = X W (k columns, X the standardized Qwen3-4B prefill features of the paper probes); per route m a logistic success head
logit_m = a_m . z + b_m (binomial over all draws) and a linear cost head log L_m = c_m . z + e_m, all fitted jointly on the training
problems: loss = mean binomial NLL + mean over routes of squared error / var(log L_m) + lam (|W|^2 + |heads|^2), full-batch L-BFGS,
two random restarts; lam chosen on the calibration problems by the same loss. With five routes, k = 10 cannot bind (five success
directions plus five cost directions), so lr10 is the SAME pipeline without a bottleneck: the matched control. Post-processing as the
paper probes: success Platt-scaled per route on calibration (first-draw labels), cost smeared from the training residuals and
level-matched to the route's training mean.
Arms (every training set refits the paper success readouts P exactly as tmlr_strengthen.py `draws`):
  ours           paper success probes + paper cost probe (ridge)                    -- reference, as in tab:pools
  lr<k>          success AND cost from the rank-k bottleneck (k = 1, 2, 4, 10)
  lr<k>-cost     paper success probes + cost from the rank-k bottleneck
  oracle         realized length (headroom)
Per arm: saving vs the median rule (with P) and ours vs the arm, point + NBOOT test resamples (rng 0, the same in every set);
test success AUC and log-length R2 per route.
Usage: REASON_ROOT=.../reason_pinned RESULT_TAG=_pinned NBOOT=300 OUT_DIR=... python lowrank_draws.py POOL S0 S1
"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
POOL_, S0, S1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); sys.argv = [sys.argv[0], "lowrank", POOL_]
exec(open(Path(__file__).parent / "cost_generalization.py").read().split('MODE, POOL = sys.argv[1], sys.argv[2]')[0])
from sklearn.linear_model import LogisticRegression
OUT = Path(os.environ.get("OUT_DIR", Path(__file__).parent)); TAG = os.environ.get("RESULT_TAG", "")
KS = tuple(int(x) for x in os.environ.get("KS", "1,2,4,10").split(",")); LAMS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0); RESTARTS = 2


def fit_lowrank(Xs, succ, n, Y, k, lam, seed):
    """joint rank-k fit on standardized features Xs [N, p]; returns parameters as numpy arrays"""
    g = torch.Generator().manual_seed(seed); X = torch.tensor(Xs, dtype=torch.float32); N, p = X.shape; M = Y.shape[1]
    s_, n_ = torch.tensor(succ, dtype=torch.float32), torch.tensor(n, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32); mu, sd = Yt.mean(0), Yt.std(0) + 1e-6; Yz = (Yt - mu) / sd
    W = (torch.randn(p, k, generator=g) / np.sqrt(p)).requires_grad_()
    A = (torch.randn(k, M, generator=g) * .1).requires_grad_(); b = torch.zeros(M, requires_grad=True)
    Cw = (torch.randn(k, M, generator=g) * .1).requires_grad_(); e = torch.zeros(M, requires_grad=True)
    q0 = ((s_.sum(0) + 1) / (n_.sum(0) + 2)); b.data = torch.log(q0 / (1 - q0))
    ps = [W, A, b, Cw, e]; opt = torch.optim.LBFGS(ps, lr=1, max_iter=500, history_size=20, line_search_fn="strong_wolfe")

    def loss_fn():
        opt.zero_grad(); z = X @ W; lg = z @ A + b
        nll = (n_ * torch.nn.functional.softplus(lg) - s_ * lg).sum() / n_.sum()       # binomial NLL per draw
        mse = ((z @ Cw + e - Yz) ** 2).mean()
        L_ = nll + mse + lam * ((W ** 2).sum() + (A ** 2).sum() + (Cw ** 2).sum()); L_.backward(); return L_
    opt.step(loss_fn)
    return dict(W=W.detach().numpy(), A=A.detach().numpy(), b=b.detach().numpy(), C=Cw.detach().numpy(), e=e.detach().numpy(),
                mu=mu.numpy(), sd=sd.numpy())


def heads(par, Xs):
    z = Xs @ par["W"]; return z @ par["A"] + par["b"], (z @ par["C"] + par["e"]) * par["sd"] + par["mu"]


def cal_loss(par, Xc, succ, n, Y):
    lg, yh = heads(par, Xc); p_ = np.clip(sig(lg), 1e-6, 1 - 1e-6)
    nll = -(succ * np.log(p_) + (n - succ) * np.log(1 - p_)).sum() / n.sum()
    return float(nll + (((yh - Y) / par["sd"]) ** 2).mean())


def lowrank(d, k):
    """rank-k success [N, M] and tokens [N, M] for every problem, fitted on train, lam chosen on calibration"""
    tr, ca = d["tr"], d["ca"]; sc = StandardScaler().fit(d["X"][tr]); Xa = sc.transform(d["X"]).astype(np.float32)
    Y = np.log(d["L"]); best = None
    for lam in LAMS:
        for r in range(RESTARTS):
            par = fit_lowrank(Xa[tr], d["succ"][tr], d["n"][tr], Y[tr], k, lam, 1000 * k + r)
            l_ = cal_loss(par, Xa[ca], d["succ"][ca], d["n"][ca], Y[ca])
            if best is None or l_ < best[0]:
                best = (l_, lam, par)
    _, lam, par = best; lg, yh = heads(par, Xa); M = d["M"]; P = np.zeros_like(lg); tk = np.zeros_like(yh)
    for m in range(M):                                                                  # Platt on calibration, as success_linear
        yc = d["y0"][ca, m]
        P[:, m] = LogisticRegression(max_iter=2000, C=1e6).fit(lg[ca, m][:, None], yc).predict_proba(lg[:, m][:, None])[:, 1] if 0 < yc.mean() < 1 else sig(lg[:, m])
        sm = np.mean(np.exp(Y[tr, m] - yh[tr, m])); es, et = np.exp(yh[tr, m]) * sm, np.exp(yh[:, m]) * sm   # smearing + level match
        tk[:, m] = et * np.exp(Y[tr, m]).mean() / es.mean()
    return np.clip(P, 1e-4, 1 - 1e-4), tk, lam


res = {"pool": POOL_, "ks": KS, "seeds": {}}
for s in range(S0, S1):
    os.environ["DRAW"] = str(s); d = load(POOL_); N = len(d["ids"]); tr, ca, ev = d["tr"], d["ca"], d["ev"]
    S, T = view(d, tr), view(d, np.arange(N))
    tk = {"ours": fit_predict("ours", S, T)}; Ps = {"ours": d["P"]}; lam_ = {}
    for k in KS:
        Pk, tkk, lam_[k] = lowrank(d, k)
        tk[f"lr{k}"], Ps[f"lr{k}"] = tkk, Pk
        tk[f"lr{k}-cost"], Ps[f"lr{k}-cost"] = tkk, d["P"]
    tk["oracle"] = d["L"]; Ps["oracle"] = d["P"]
    C = {a: cost_of(d, x) for a, x in tk.items()}; Cm = cost_of(d, np.repeat(d["med"][None], N, 0))
    rb = np.random.default_rng(0); BS = [ev] + [ev[rb.integers(0, len(ev), len(ev))] for _ in range(NBOOT)]
    Hs = [{a: curve(d, Ps[a], C[a], ii) for a in C} | {"median": curve(d, d["P"], Cm, ii)} for ii in BS]
    r_ = {}
    for a in C:
        vm = np.array([saved(H[a], H["median"]) for H in Hs]) * 100; vd = np.array([saved(H["ours"], H[a]) for H in Hs]) * 100
        r_[a] = dict(vs_median=float(vm[0]), vs_median_bs=vm[1:].tolist(), ours_vs_it=float(vd[0]), ours_vs_it_bs=vd[1:].tolist(),
                     auc=[float(auc(d["y0"][ev, m], Ps[a][ev, m])) for m in range(d["M"])], r2=r2_routes(d, tk[a], ev))
    r_["_lam"] = lam_; r_["_n"] = dict(train=len(tr), calibration=len(ca), test=len(ev))
    res["seeds"][s] = r_
    print(f"  set {s}: vs median " + "  ".join(f"{a} {r_[a]['vs_median']:+5.1f}" for a in C), flush=True)
    print("        ours vs it " + "  ".join(f"{a} {r_[a]['ours_vs_it']:+5.1f}" for a in C if a != "ours"), flush=True)
    print("        AUC " + "  ".join(f"{a} {np.mean(r_[a]['auc']):.3f}" for a in C if not a.endswith("-cost") and a != "oracle")
          + " | R2 " + "  ".join(f"{a} {np.mean(r_[a]['r2']):.2f}" for a in C if a != "oracle" and not (a.startswith("lr") and not a.endswith("-cost"))), flush=True)
    json.dump(res, open(OUT / f"lowrank_draws_{POOL_.lower().replace('-', '')}_{S0}_{S1}{TAG}.json", "w"), indent=1, default=float)
print("DONE", flush=True)
