"""Per-query BUDGET: "spend at most X cents on this query". One-shot routing, one call per query, market prices; exact replay
from stored draws. Does per-query cost prediction matter more under a per-query budget than on average cost?

HARD cap (enforced): the call gets max_tokens = (X - input cost) / output price for its model; a draw longer than that is
  cut off and fails. Realised per draw: success = solved AND length <= cap; cost = input + min(length, cap) x price_out.
  Router: argmax_m p_m(x) * P(L_m(x) <= cap_m); the length model is the only thing that differs between arms:
    median      the paper rule: model m fits iff its median TRAIN length <= cap (deterministic)
    constant    P(L <= cap) from model m's TRAIN length distribution (same for every query)
    ours        log L ~ N(mu_m(x), s_m^2): mu from the 4B cost probe, s_m from train per-draw residuals
    oracle      the problem's own draws' length distribution (leaky ceiling for any length predictor)
  Metric: test accuracy at each budget X; paired bootstrap of ours - constant (and - median) at every X.
SOFT cap (not enforced): the call runs to completion; a query whose realised cost exceeds X is a VIOLATION.
  Router: most likely-to-succeed model whose predicted cost x theta <= X (theta >= 1 is a safety margin, swept); predicted cost
  = input + median train output (paper) or the cost probe's expected cost (ours) or the true mean (oracle).
  Metric: accuracy at matched violation rate (5%, 10%) per X, plus mean overshoot (mean of max(0, cost - X) / X).
All arms share the prefill success head p. Usage: python budget_cap.py
"""
import json, os, sys, numpy as np
from pathlib import Path
from scipy.stats import norm
sys.path.insert(0, str(Path(__file__).parent))
from decompose import MK, R

POOLS = [("LCB", "pool_v2_tensors_5rung", "cost_preds_probe.jsonl"), ("Omni", "omni500_tensors", "cost_preds_probe_thinking.jsonl"),
         ("MMLU-Pro", "mmlupro_tensors", "cost_preds_probe_instruct.jsonl")]
THETAS = np.geomspace(0.5, 8, 30)
out = {}
for label, name, cfile in POOLS:
    D = R / name; t = np.load(D / "tensors.npz", allow_pickle=True)
    S = [str(s) for s in t["model_slots"]]; M = len(S); pids = [str(p) for p in t["problem_ids"]]; pi = {p: i for i, p in enumerate(pids)}
    v = t["valid"].astype(bool); ok = (t["final_outcome"] & t["valid"]).astype(bool)
    ct = t["completion_tokens"].astype(float); pt = t["prompt_tokens"].astype(float); n = v.sum(2); avail = n > 0
    pin = np.array([MK[s][0] for s in S]) / 1e6 * 100; pout = np.array([MK[s][1] for s in S]) / 1e6 * 100      # cents / token
    inp = np.nan_to_num(np.nanmean(np.where(v, pt, np.nan), 2))
    sp = json.load(open(D / "split_manifest.json")); tr = np.array([pi[str(p)] for p in sp["train_problem_ids"]]); te = np.array([pi[str(p)] for p in sp["test_problem_ids"]])
    lp = {json.loads(l)["problem_id"]: json.loads(l)["p_successes"][:M] for l in open(D / "content_preds.jsonl")}
    P = np.clip(np.array([lp[p] for p in pids]), 0, 1)
    lc = {json.loads(l)["problem_id"]: json.loads(l)["expected_costs"][:M] for l in open(D / cfile)}
    LC = np.array([lc[p] for p in pids]) * 100
    MU = np.log(np.maximum((LC - inp * pin) / pout, 1.0)); LOGL = np.log(np.maximum(np.where(v, ct, np.nan), 1.0))
    SIG = np.array([np.nanstd((LOGL[tr, m] - MU[tr, m][:, None])[v[tr, m]]) for m in range(M)])
    RES = [np.sort((LOGL[tr, m] - MU[tr, m][:, None])[v[tr, m]]) for m in range(M)]
    trainL = [np.sort(ct[tr, m][v[tr, m]]) for m in range(M)]; med = np.array([np.median(x) for x in trainL])
    rcost = np.where(v, pt * pin[None, :, None] + ct * pout[None, :, None], np.nan)                  # realised uncapped cost per draw
    T = te[avail[te].all(1)]
    Xs = np.geomspace(np.nanpercentile(rcost[T, int(np.argmin(pout))], 10), np.nanpercentile(rcost[T, int(np.argmax(med * pout))], 90), 24)

    def hard(X, arm):
        cap = np.maximum((X - inp[T] * pin) / pout, 0.0)                                            # [n, M] tokens
        if arm == "median":
            fit = (med[None] <= cap).astype(float)
        elif arm == "constant":
            fit = np.stack([np.searchsorted(trainL[m], cap[:, m], side="right") / len(trainL[m]) for m in range(M)], 1)
        elif arm == "ours":                                   # empirical TRAIN residual distribution around the probe's per-query shift
            fit = np.stack([np.searchsorted(RES[m], np.log(np.maximum(cap[:, m], 1e-9)) - MU[T, m], side="right") / len(RES[m]) for m in range(M)], 1)
        elif arm == "ours-lognormal":
            fit = norm.cdf((np.log(np.maximum(cap, 1e-9)) - MU[T]) / SIG[None])
        else:
            fit = np.nanmean(np.where(v[T], ct[T] <= cap[..., None], np.nan), 2)
        score = P[T] * fit; score = np.where(cap > 0, score, -1)
        m = score.argmax(1); r = np.arange(len(T))
        fb = (score.max(1) <= 0) & (cap > 0).any(1)                 # nothing predicted to fit: call the model with the most room
        m = np.where(fb, (cap / np.maximum(med[None], 1)).argmax(1), m)
        c_ = cap[r, m][:, None]; vv = v[T][r, m]
        succ = np.nanmean(np.where(vv, ok[T][r, m] & (ct[T][r, m] <= c_), np.nan), 1)
        cost = np.nanmean(np.where(vv, pt[T][r, m] * pin[m][:, None] + np.minimum(ct[T][r, m], c_) * pout[m][:, None], np.nan), 1)
        called = (cap > 0).any(1); succ = np.where(called, succ, 0.0); cost = np.where(called, cost, 0.0)
        return succ, cost

    def soft(X, arm, theta):
        pred = {"paper": inp[T] * pin + med[None] * pout, "ours": LC[T], "oracle": np.nanmean(rcost[T], 2)}[arm]
        feas = pred * theta <= X
        score = np.where(feas, P[T], -1.0); m = score.argmax(1); r = np.arange(len(T)); none = ~feas.any(1)
        m = np.where(none, int(np.argmin(pout)), m)                                                # nothing fits: cheapest model anyway
        c = rcost[T][r, m]; vv = v[T][r, m]
        succ = np.nanmean(np.where(vv, ok[T][r, m], np.nan), 1)
        viol = np.nanmean(np.where(vv, c > X, np.nan), 1); over = np.nanmean(np.where(vv, np.maximum(c - X, 0) / X, np.nan), 1)
        return succ, viol, over

    rng = np.random.default_rng(0); B = [rng.integers(0, len(T), len(T)) for _ in range(500)]
    print(f"\n===== {label}: {len(T)} test problems; budget X from {Xs[0]:.4f}c to {Xs[-1]:.4f}c; median train output per model "
          + ", ".join(f"{s} {m_:.0f}" for s, m_ in zip(S, med)))
    print("HARD cap (enforced by max_tokens): test accuracy % [ours - constant, 95% CI]")
    print(f"   {'X (cents)':>10} {'median':>7} {'constant':>9} {'ours':>6} {'oracle':>7}   ours-constant        ours-median")
    H = {}
    for X in Xs:
        res = {a: hard(X, a) for a in ("median", "constant", "ours", "ours-lognormal", "oracle")}
        d1 = np.array([(res["ours"][0][b] - res["constant"][0][b]).mean() for b in B]) * 100
        d2 = np.array([(res["ours"][0][b] - res["median"][0][b]).mean() for b in B]) * 100
        H[float(X)] = {a: [float(res[a][0].mean()), float(res[a][1].mean())] for a in res} | {"ours-constant": [float(d1.mean()), *np.percentile(d1, [2.5, 97.5])],
                                                                                            "ours-median": [float(d2.mean()), *np.percentile(d2, [2.5, 97.5])]}
        print(f"   {X:>10.4f} {res['median'][0].mean()*100:7.1f} {res['constant'][0].mean()*100:9.1f} {res['ours'][0].mean()*100:6.1f} "
              f"{res['oracle'][0].mean()*100:7.1f}   {d1.mean():+5.1f} [{np.percentile(d1,2.5):+.1f},{np.percentile(d1,97.5):+.1f}]"
              f"   {d2.mean():+5.1f} [{np.percentile(d2,2.5):+.1f},{np.percentile(d2,97.5):+.1f}]")
    def budget_needed(arm, acc):                              # smallest budget X on the grid (log-interpolated) reaching acc
        ys = np.maximum.accumulate([H[float(X)][arm][0] for X in Xs])
        if acc > ys[-1] or acc < ys[0]:
            return np.nan
        return float(np.exp(np.interp(acc, ys, np.log(Xs))))
    top = min(max(H[float(X)][a][0] for X in Xs) for a in ("constant", "ours", "median"))
    lo_ = max(H[float(Xs[0])][a][0] for a in ("constant", "ours", "median"))
    targets = np.linspace(lo_ + .05 * (top - lo_), top - .02 * (top - lo_), 15)
    for base in ("constant", "median"):
        r = [budget_needed(base, x) / budget_needed("ours", x) for x in targets]
        print(f"   budget {base} needs / budget ours needs, at matched accuracy {targets[0]*100:.0f}-{targets[-1]*100:.0f}%: "
              + " ".join(f"{x:.2f}" for x in r) + f"   geo-mean {np.exp(np.nanmean(np.log(r))):.2f}")
        out.setdefault(label, {})[f"budget_ratio_{base}"] = [float(np.exp(np.nanmean(np.log(r)))), [float(x) for x in r], [float(x) for x in targets]]
    print("SOFT cap (not enforced): accuracy % at matched violation rate (theta swept per arm; linear interpolation) | mean overshoot")
    Sft = {}
    for X in Xs[::3]:
        row = {}
        for arm in ("paper", "ours", "oracle"):
            pts = [soft(X, arm, th) for th in THETAS]
            vr = np.array([p_[1].mean() for p_ in pts]); ac = np.array([p_[0].mean() for p_ in pts]); ov = np.array([p_[2].mean() for p_ in pts])
            o = np.argsort(vr)
            row[arm] = {f"acc@{int(q*100)}%viol": float(np.interp(q, vr[o], ac[o], left=np.nan)) for q in (0.05, 0.10)} | \
                       {f"overshoot@{int(q*100)}%viol": float(np.interp(q, vr[o], ov[o], left=np.nan)) for q in (0.05, 0.10)}
        Sft[float(X)] = row
        print(f"   X {X:.4f}c: " + " | ".join(f"{a} acc@5% {r['acc@5%viol']*100:5.1f} @10% {r['acc@10%viol']*100:5.1f} (overshoot@10% {r['overshoot@10%viol']*100:4.0f}%)"
                                            for a, r in row.items()))
    out.setdefault(label, {}).update({"hard": H, "soft": Sft})
json.dump(out, open(Path(__file__).parent / f"budget_cap{os.environ.get('RESULT_TAG', '')}.json", "w"), indent=1, default=float)
