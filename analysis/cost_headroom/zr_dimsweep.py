"""ZeroRouter (arXiv 2601.06220) fitted on OUR 5-model pools: what happens to its latent with 5 models instead of ~200?
Stage 1 (their Eq. 1): D-dim 2PL IRT, P(u solves i) = sigmoid(alpha_i^T (theta_u - b_i)), alpha_i = exp(la_i) > 0, Gaussian priors
  (la ~ N(log(1/sqrt(D)), 0.5^2), b ~ N(0, 1), theta ~ N(0, 1)); MAP by Adam on TRAIN outcomes (binomial over draws). The paper
  uses SVI on a hierarchical version; MAP is the same model's point estimate.
Stage 2: predict (la, b) for every query from text. Their encoder is fine-tuned DistilBERT + 11 linguistic features; here the
  frozen Qwen3-4B prefill activations (mean + last, 8 layers) -> PCA(256) -> ridge -- a stronger reader, generous to them.
Cost (their Eq. 8-10): s = alpha^T b, K quantile bins on train, per-model mean TRAIN output per bin.
Evaluated per D in {1, 2, 5, 20}, test split:
  (1) success prediction: per-route AUC and log-loss vs our prefill success probes
  (2) routing gain vs the paper rule at matched accuracy: ours / zr / zr success + our cost / our success + zr cost
  (3) onboarding (hold out each route; stage 1 + 2 refit on the other 4; theta_h MAP from k random anchors; per-bin length from
      the anchors) at k = 10, 50, 20 draws -- mean over held-out routes, vs ours (onboard_full.py's method, same anchors)
Usage: python zr_dimsweep.py
"""
import json, sys, numpy as np, torch
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).parent))
from decompose import MK, R, hull, cost_at
from baseline_cost_heads import rich

POOLS = [("LCB", "pool_v2_tensors_5rung", "cost_preds_probe.jsonl", "pv2_scout_prefill_1756715297/scout.npz"),
         ("Omni", "omni500_tensors", "cost_preds_probe_thinking.jsonl", "omni500_probe/thinking.npz"),
         ("MMLU-Pro", "mmlupro_tensors", "cost_preds_probe_instruct.jsonl", "mmlupro_probe/instruct.npz")]
DS = [1, 2, 5, 20]; K = 10; VS = np.geomspace(1e-5, 100, 250)
sig = lambda z: 1 / (1 + np.exp(-z))
torch.set_num_threads(8)


def fit_stage1(succ, n, D, seed=0):
    """succ, n: [N, U] counts -> la [N, D], b [N, D], th [U, D]"""
    g = torch.Generator().manual_seed(seed); N, U = succ.shape
    la = torch.full((N, D), float(np.log(1 / np.sqrt(D)))) + 0.01 * torch.randn(N, D, generator=g)
    b = 0.1 * torch.randn(N, D, generator=g); th = 0.1 * torch.randn(U, D, generator=g)
    for x in (la, b, th):
        x.requires_grad_(True)
    s_, n_ = torch.tensor(succ, dtype=torch.float32), torch.tensor(n, dtype=torch.float32)
    opt = torch.optim.Adam([la, b, th], lr=0.05); mu_la = float(np.log(1 / np.sqrt(D)))
    for _ in range(1500):
        z = (torch.exp(la)[:, None, :] * (th[None] - b[:, None, :])).sum(-1)
        ll = (s_ * torch.nn.functional.logsigmoid(z) + (n_ - s_) * torch.nn.functional.logsigmoid(-z)).sum()
        pr = ((la - mu_la) ** 2).sum() / (2 * 0.25) + (b ** 2).sum() / 2 + (th ** 2).sum() / 2
        loss = -ll + pr; opt.zero_grad(); loss.backward(); opt.step()
    return la.detach().numpy(), b.detach().numpy(), th.detach().numpy()


def fit_theta(a, bb, s, n, D):
    """new model's ability from anchors: a, bb [k, D]; s, n [k]"""
    th = torch.zeros(D, requires_grad=True); a_, b_ = torch.tensor(a, dtype=torch.float32), torch.tensor(bb, dtype=torch.float32)
    s_, n_ = torch.tensor(s, dtype=torch.float32), torch.tensor(n, dtype=torch.float32); opt = torch.optim.Adam([th], lr=0.05)
    for _ in range(400):
        z = (a_ * (th[None] - b_)).sum(-1)
        loss = -(s_ * torch.nn.functional.logsigmoid(z) + (n_ - s_) * torch.nn.functional.logsigmoid(-z)).sum() + (th ** 2).sum() / 2
        opt.zero_grad(); loss.backward(); opt.step()
    return th.detach().numpy()


out = {}
for label, name, cfile, act in (POOLS if __name__ == "__main__" else []):     # importable without running
    D_ = R / name; t = np.load(D_ / "tensors.npz", allow_pickle=True)
    S = [str(s) for s in t["model_slots"]]; M = len(S); pids = [str(p) for p in t["problem_ids"]]; pi = {p: i for i, p in enumerate(pids)}
    v = t["valid"].astype(bool); okd = (t["final_outcome"] & t["valid"]).astype(bool); ct = t["completion_tokens"].astype(float)
    pt = t["prompt_tokens"].astype(float); n = v.sum(2); avail = n > 0; succ = (okd & v).sum(2)
    inp = np.nan_to_num(np.nanmean(np.where(v, pt, np.nan), 2))
    pin = np.array([MK[s][0] for s in S]) / 1e6 * 100; pout = np.array([MK[s][1] for s in S]) / 1e6 * 100
    outm = np.where(avail, np.where(v, ct, 0).sum(2) / np.maximum(n, 1), np.nan); Y = np.log(np.maximum(outm, 1))
    Q = np.where(avail, succ / np.maximum(n, 1), 0); Cr = np.where(avail, inp * pin + outm * pout, 1e9)
    sp = json.load(open(D_ / "split_manifest.json")); tr = np.array([pi[str(p)] for p in sp["train_problem_ids"]]); te = np.array([pi[str(p)] for p in sp["test_problem_ids"]])
    _lp = {json.loads(l)["problem_id"]: json.loads(l)["p_successes"][:M] for l in open(D_ / "content_preds.jsonl")}
    P = np.clip(np.array([_lp[p] for p in pids]), 1e-4, 1 - 1e-4); LOGIT = np.log(P / (1 - P))
    lc = {json.loads(l)["problem_id"]: json.loads(l)["expected_costs"][:M] for l in open(D_ / cfile)}
    LC = np.array([lc[p] for p in pids]) * 100; MU = np.log(np.maximum((LC - inp * pin) / pout, 1.0))
    med = np.array([np.median(ct[tr, m][v[tr, m]]) for m in range(M)])
    X = rich(R / act, pids); pca = PCA(256, random_state=0).fit(X[tr]); Z = pca.transform(X); Z /= Z[tr].std(0) + 1e-6

    def frontier(Pm, C, ii):
        pts = []
        for V in VS:
            m = np.where(avail[ii], Pm[ii] * V - C[ii], -np.inf).argmax(1); r = np.arange(len(ii))
            pts.append((np.mean(Cr[ii][r, m]), np.mean(Q[ii][r, m])))
        return hull(pts)
    H0 = frontier(P, inp * pin + med[None] * pout, te)

    def gain(Pm, C):
        H = frontier(Pm, C, te); lo, hi = max(H[0][1], H0[0][1]), min(H[-1][1], H0[-1][1])
        T = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 12)
        return 1 - float(np.exp(np.nanmean(np.log([cost_at(H, x) / cost_at(H0, x) for x in T]))))

    def latent(routes, D):
        la, b, th = fit_stage1(succ[np.ix_(tr, routes)], n[np.ix_(tr, routes)], D)
        f = RidgeCV(alphas=np.geomspace(1, 1e5, 11)).fit(Z[tr], np.c_[la, b])
        pr = f.predict(Z); A = np.exp(pr[:, :D]); B = pr[:, D:]
        A[tr] = np.exp(la); B[tr] = b                                   # training queries keep their stage-1 positions
        return A, B, th

    def zr_cost(A, B, routes_len_from, idx_for_bins):
        s = (A * B).sum(1); edges = np.quantile(s[tr], np.linspace(0, 1, K + 1)[1:-1]); bn = np.searchsorted(edges, s)
        return bn, edges

    res = {"ours_full": gain(P, LC)}
    auc_ours = [roc_auc_score(Q[te, m] > .5, P[te, m]) for m in range(M)]
    print(f"\n===== {label}: ours full heads {res['ours_full']*100:.1f}% vs paper rule; our success-probe AUC per route " + " ".join(f"{x:.2f}" for x in auc_ours))
    rng = np.random.default_rng(0)
    for D in DS:
        A, B, th = latent(list(range(M)), D)
        Pz = sig((A[:, None, :] * (th[None] - B[:, None, :])).sum(-1))
        auc = [roc_auc_score(Q[te, m] > .5, Pz[te, m]) for m in range(M)]
        nll = lambda PP: float(-np.mean([(succ[i, m] * np.log(np.clip(PP[i, m], 1e-4, 1)) + (n[i, m] - succ[i, m]) * np.log(np.clip(1 - PP[i, m], 1e-4, 1))) / max(n[i, m], 1) for i in te for m in range(M) if n[i, m]]))
        s = (A * B).sum(1); edges = np.quantile(s[tr], np.linspace(0, 1, K + 1)[1:-1]); bn = np.searchsorted(edges, s)
        tab = np.array([[np.nanmean(outm[tr[bn[tr] == j], m]) if (bn[tr] == j).any() else np.nanmean(outm[tr, m]) for j in range(K)] for m in range(M)])
        Cz = inp * pin + tab.T[bn] * pout
        g = {"zr": gain(Pz, Cz), "zr success + our cost": gain(Pz, LC), "our success + zr cost": gain(P, Cz)}
        # onboarding at this D
        ob = {k: {"ours": [], "zr": []} for k in (10, 50)}
        for h in range(M):
            others = [m for m in range(M) if m != h]
            Ah, Bh, thh = latent(others, D)
            sh = (Ah * Bh).sum(1); eh = np.quantile(sh[tr], np.linspace(0, 1, 6)[1:-1]); bh = np.searchsorted(eh, sh)
            lvl = np.nanmean(np.delete(MU, h, 1), 1); dbar = np.nanmean(np.delete(LOGIT, h, 1), 1); trh = tr[avail[tr, h]]
            for k in (10, 50):
                for _ in range(10):
                    kk = rng.choice(trh, k, replace=False); yk = np.concatenate([okd[i, h][v[i, h]] for i in kk]).astype(int)
                    Pm, C = P.copy(), LC.copy()                             # ours
                    off = np.mean(Y[kk, h] - lvl[kk]); sm = np.mean(np.exp(Y[kk, h] - (lvl[kk] + off)))
                    C[:, h] = inp[:, h] * pin[h] + np.exp(lvl + off) * sm * pout[h]
                    Xk = np.repeat(dbar[kk], n[kk, h])
                    Pm[:, h] = LogisticRegression(C=1.0).fit(Xk[:, None], yk).predict_proba(dbar[:, None])[:, 1] if len(set(yk)) == 2 else np.clip(yk.mean(), .02, .98)
                    ob[k]["ours"].append(gain(Pm, C))
                    Pm, C = P.copy(), LC.copy()                             # ZeroRouter at this D
                    tnew = fit_theta(Ah[kk], Bh[kk], succ[kk, h], n[kk, h], D)
                    Pm[:, h] = sig((Ah * (tnew[None] - Bh)).sum(1))
                    bm = np.array([np.nanmean(outm[kk, h][bh[kk] == j]) if (bh[kk] == j).any() else np.nanmean(outm[kk, h]) for j in range(5)])
                    C[:, h] = inp[:, h] * pin[h] + bm[bh] * pout[h]
                    ob[k]["zr"].append(gain(Pm, C))
        res[f"D={D}"] = {"auc": auc, "nll": nll(Pz), "nll_ours": nll(P), "routing": g,
                         "onboard": {k: {a: float(np.mean(x)) for a, x in r.items()} for k, r in ob.items()}}
        print(f"  D={D:<3} success AUC " + " ".join(f"{x:.2f}" for x in auc) + f" | log-loss zr {nll(Pz):.3f} (ours {nll(P):.3f})"
              + " | routing gain: " + ", ".join(f"{a} {x*100:5.1f}%" for a, x in g.items())
              + " | onboarding k=10: ours {:.1f}% zr {:.1f}%; k=50: ours {:.1f}% zr {:.1f}%".format(
                  *[res[f"D={D}"]["onboard"][k][a] * 100 for k in (10, 50) for a in ("ours", "zr")]), flush=True)
    out[label] = res
if __name__ == "__main__":
    json.dump(out, open(Path(__file__).parent / "zr_dimsweep.json", "w"), indent=1, default=float)
