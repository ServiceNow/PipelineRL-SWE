#!/usr/bin/env python3
"""Cost-head replication on the SWE-Smith REASONING pool (plan D: 500 instances x 5 open reasoning routes): one-shot routing, learned per-query cost vs the median rule.

SWE prompts are input-heavy (~4.8k in / ~1k out), so input and output tokens are priced separately.
Both arms share one success head and differ ONLY in the cost estimate:
  paper  (arXiv 2603.20895): this query's input tokens * in_price + route's median TRAIN output * out_price
  ours:                      this query's input tokens * in_price + predicted output tokens * out_price
Heads read the 4B prefill activations (PCA on train); per-route logistic success and ridge on log output
tokens, regularisation chosen on a calibration split carved from train. Evaluation is on the held-out
eval split with the single real draw per route (real Daytona labels, real token counts):
  - matched spend: value grid chosen on calibration, mixed to each budget, applied once to test
  - matched accuracy: cost at target accuracy on each arm's test frontier (same treatment for both)
Both with problem-level paired bootstrap CIs.
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss

ROUTES = ["oss20lo", "oss20md", "dsv4f", "oss120md", "oss120hi"]
# $/M (in, out), OpenRouter 2026-09-25. Qwen3-4B is self-hosted (not listed): priced at gpt-oss-20b as a floor.
PRICE = {"oss20lo": (0.018, 0.09), "oss20md": (0.018, 0.09), "dsv4f": (0.04704, 0.09408),
         "oss120md": (0.15, 0.6), "oss120hi": (0.15, 0.6)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--eval-logs", default="logs/run_evaluation")
    ap.add_argument("--activations", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pca", type=int, default=128)
    ap.add_argument("--cal-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    W = Path(a.pool_dir); ids = json.load(open(W / "full" / "ids.json"))
    z = np.load(a.activations, allow_pickle=True)
    feats = np.concatenate([z[k].reshape(len(z[k]), -1) for k in ("mean", "last") if k in z.files], 1)
    fid = {str(p): i for i, p in enumerate(z["problem_ids"])}
    ids = [i for i in ids if i in fid]
    X = feats[[fid[p] for p in ids]].astype(np.float32)
    Y = np.zeros((len(ids), len(ROUTES)), int); IN = np.zeros(Y.shape); OUT = np.zeros(Y.shape)
    for m, r in enumerate(ROUTES):
        tok = {json.loads(l)["instance_id"]: json.loads(l) for l in open(W / "full" / f"predictions_{r}_d0.jsonl")}
        for i, iid in enumerate(ids):
            IN[i, m] = tok[iid]["prompt_tokens"]; OUT[i, m] = tok[iid]["completion_tokens"]
            rep = Path(a.eval_logs) / f"swesmith_reason_v2_predictions_{r}_d0" / iid / "report.json"
            Y[i, m] = int(rep.exists() and bool(json.loads(rep.read_text()).get("resolved")))
    pin = np.array([PRICE[r][0] for r in ROUTES]) / 1e6 * 100; pout = np.array([PRICE[r][1] for r in ROUTES]) / 1e6 * 100
    COST = IN * pin + OUT * pout
    perm = rng.permutation(len(ids)); ntr, ncal = int(0.55 * len(ids)), int(0.15 * len(ids))
    tr, cal, te = perm[:ntr], perm[ntr:ntr + ncal], perm[ntr + ncal:]
    print(f"{len(ids)} problems: train {len(tr)} / calibration {len(cal)} / test {len(te)}; "
          f"test success {Y[te].mean(0).round(3)}; mean cost/route (c) {COST[te].mean(0).round(4)}")
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Z = PCA(min(a.pca, len(tr) - 1), random_state=0).fit((X[tr] - mu) / sd).transform((X - mu) / sd)
    P = np.zeros(Y.shape); LOUT = np.zeros(Y.shape)
    for m in range(len(ROUTES)):
        best = min(((log_loss(Y[cal, m], LogisticRegression(C=C, max_iter=3000).fit(Z[tr], Y[tr, m]).predict_proba(Z[cal])[:, 1]), C)
                    for C in (1e-4, 1e-3, 1e-2, 1e-1, 1.0)))
        P[:, m] = LogisticRegression(C=best[1], max_iter=3000).fit(Z[tr], Y[tr, m]).predict_proba(Z)[:, 1]
        ly = np.log(OUT[:, m] + 1)
        besta = min(((np.mean((Ridge(alpha=al).fit(Z[tr], ly[tr]).predict(Z[cal]) - ly[cal]) ** 2), al)
                     for al in (1, 10, 100, 1e3, 1e4, 1e5)))
        rg = Ridge(alpha=besta[1]).fit(Z[tr], ly[tr])
        smear = np.mean(np.exp(ly[tr] - rg.predict(Z[tr])))           # Duan smearing: log-fit -> mean
        LOUT[:, m] = np.exp(rg.predict(Z)) * smear
        r2 = 1 - np.mean((rg.predict(Z[te]) - ly[te]) ** 2) / np.var(ly[te])
        print(f"  {ROUTES[m]:<9} success C={best[1]} | log-output ridge alpha={besta[1]} test R2 {r2:.3f}")
    med = np.array([np.median(OUT[tr, m]) for m in range(len(ROUTES))])
    C_paper = IN * pin + med * pout
    C_ours = IN * pin + LOUT * pout
    Vs = np.geomspace(1e-4, 100, 300)

    def run(idx, C, V):
        r = np.argmax(P[idx] * V - C[idx], 1)
        return Y[idx, r].astype(float), COST[idx, r]

    def matched_spend(C, budget):
        pts = sorted([(0.0, 0.0, None)] + [(run(cal, C, V)[1].mean(), run(cal, C, V)[0].mean(), V) for V in Vs])
        best = None
        for lo in pts:
            for hi in pts:
                if lo[0] <= budget <= hi[0] and hi[0] > lo[0]:
                    w = (budget - lo[0]) / (hi[0] - lo[0]); acc = (1 - w) * lo[1] + w * hi[1]
                elif lo is hi and lo[0] <= budget:
                    w, acc = 0.0, lo[1]
                else:
                    continue
                if best is None or acc > best[0]:
                    best = (acc, lo[2], hi[2], w)
        _, vl, vh, w = best
        al, cl = run(te, C, vl) if vl is not None else (np.zeros(len(te)), np.zeros(len(te)))
        ah, ch = run(te, C, vh)
        return (1 - w) * al + w * ah, (1 - w) * cl + w * ch

    out = {"routes": ROUTES, "prices": PRICE, "matched_spend": [], "matched_accuracy": []}
    lo_b, hi_b = np.percentile(COST[te].min(1), 50), np.percentile(COST[te].max(1), 60)
    budgets = np.round(np.geomspace(max(lo_b, 1e-3), hi_b, 6), 4)
    print(f"\nmatched spend (test {len(te)}; calibration-chosen value mix); ours - paper, 95% paired CI")
    for b in budgets:
        (A0, C0), (A1, C1) = matched_spend(C_paper, b), matched_spend(C_ours, b)
        dd = A1 - A0; bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(3000)]
        print(f"  {b:.4f}c  paper {A0.mean()*100:5.1f}%@{C0.mean():.4f}  ours {A1.mean()*100:5.1f}%@{C1.mean():.4f}  "
              f"delta {dd.mean()*100:+.1f} [{np.percentile(bs,2.5)*100:+.1f},{np.percentile(bs,97.5)*100:+.1f}]")
        out["matched_spend"].append({"budget": float(b), "paper": [A0.mean(), C0.mean()], "ours": [A1.mean(), C1.mean()],
                                     "delta": dd.mean(), "ci": [np.percentile(bs, 2.5), np.percentile(bs, 97.5)]})

    def cost_at(C, idx, target):
        pts = sorted({(run(idx, C, V)[1].mean(), run(idx, C, V)[0].mean()) for V in Vs})
        h = []
        for p in pts:
            while len(h) >= 2 and (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0]) <= (p[1] - h[-2][1]) * (h[-1][0] - h[-2][0]):
                h.pop()
            h.append(p)
        for (c0, a0), (c1, a1) in zip(h, h[1:]):
            if a0 <= target <= a1:
                return c0 + (c1 - c0) * (target - a0) / max(a1 - a0, 1e-12)
        return np.nan
    accs = [run(te, C_paper, V)[0].mean() for V in Vs]
    targets = np.round(np.linspace(min(accs) + 0.02, max(accs) - 0.01, 5), 3)
    print("\nmatched accuracy: cost (cents/problem) on each arm's test frontier; ratio ours/paper, 95% CI")
    for t in targets:
        c0, c1 = cost_at(C_paper, te, t), cost_at(C_ours, te, t)
        bs = []
        for _ in range(300):
            idx = te[rng.integers(0, len(te), len(te))]
            x, y = cost_at(C_ours, idx, t), cost_at(C_paper, idx, t)
            if np.isfinite(x) and np.isfinite(y) and y > 0:
                bs.append(x / y)
        ci = f"[{np.percentile(bs,2.5):.2f},{np.percentile(bs,97.5):.2f}]" if len(bs) > 50 else "[n/a]"
        print(f"  {t*100:5.1f}%  paper {c0:.4f}c  ours {c1:.4f}c  ratio {c1/c0:.2f} {ci}")
        out["matched_accuracy"].append({"accuracy": float(t), "paper": c0, "ours": c1, "ratio": c1 / c0})
    json.dump(out, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
