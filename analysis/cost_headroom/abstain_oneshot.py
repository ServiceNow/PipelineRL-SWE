"""One-shot routing (no verifier, one submission) WITH an abstain option.

Per problem: answer with route m, or abstain (cost 0, score 0). A correct answer scores +1, a wrong one -lam
(lam = 0: abstaining only saves money; lam > 0: selective prediction, a wrong answer is worse than none).
Rule: with reward V per correct answer, utility of m = V*((1+lam)*p_m - lam) - c_m; abstain iff every utility < 0.
Arms (same success head p for all except the oracle; test frontiers over V, upper hull, mixing allowed; EVERY arm's hull
includes the origin, so random abstention is always available -- informed abstention must beat it):
  fixed        always one route (the best mix of single routes); no per-problem information
  paper        paper cost rule (input + median train output), always answers
  paper+A      paper cost rule, may abstain
  ours         4B-prefill cost head, always answers
  ours+A       4B-prefill cost head, may abstain
  oracle+A     the problem's TRUE per-route success rate and cost, may abstain (ceiling for any predictor)
Metric: cost saved at matched score (score = accuracy - lam * error rate, over ALL problems), 1 - geometric-mean cost
ratio over 12 score targets in the band where BOTH compared arms have real (non-origin) points; paired bootstrap over test problems (identical resamples).
Usage: python abstain_oneshot.py [lam,lam,...]
"""
import json, sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from decompose import MK, R, cost_at

POOLS = [("LCB", "pool_v2_tensors_5rung", "cost_preds_probe.jsonl"), ("Omni", "omni500_tensors", "cost_preds_probe_thinking.jsonl"),
         ("MMLU-Pro", "mmlupro_tensors", "cost_preds_probe_instruct.jsonl")]
LAMS = [float(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0.0, 1.0, 3.0]
VS = np.geomspace(1e-5, 100, 300)


def hull0(pts):                          # upper hull through the origin (abstain on everything = (0, 0))
    pts = sorted(set(pts) | {(0.0, 0.0)}); h = []
    for c, a in pts:
        if h and a <= h[-1][1]:
            continue
        while len(h) >= 2 and (h[-1][1] - h[-2][1]) * (c - h[-2][0]) <= (a - h[-2][1]) * (h[-1][0] - h[-2][0]):
            h.pop()
        h.append((c, a))
    return h


def load(name, cfile):
    D = R / name; t = np.load(D / "tensors.npz", allow_pickle=True)
    S = [str(s) for s in t["model_slots"]]; M = len(S); pids = [str(p) for p in t["problem_ids"]]; pi = {p: i for i, p in enumerate(pids)}
    v = t["valid"].astype(bool); ok = (t["final_outcome"] & t["valid"]).astype(float)
    pt, ct = t["prompt_tokens"].astype(float), t["completion_tokens"].astype(float)
    real = np.stack([(pt[:, m] * MK[s][0] + ct[:, m] * MK[s][1]) / 1e6 * 100 for m, s in enumerate(S)], 1)
    n = v.sum(2); avail = n > 0
    Q = np.where(avail, (ok * v).sum(2) / np.maximum(n, 1), 0); Cr = np.where(avail, (real * v).sum(2) / np.maximum(n, 1), 1e9)
    sp = json.load(open(D / "split_manifest.json")); tr = np.array([pi[str(p)] for p in sp["train_problem_ids"]])
    te = np.array([pi[str(p)] for p in sp["test_problem_ids"]])
    lp = {json.loads(l)["problem_id"]: json.loads(l)["p_successes"][:M] for l in open(D / "content_preds.jsonl")}
    lc = {json.loads(l)["problem_id"]: json.loads(l)["expected_costs"][:M] for l in open(D / cfile)}
    P = np.clip(np.array([lp[p] for p in pids]), 0, 1); LC = np.array([lc[p] for p in pids]) * 100
    inp = np.nanmean(np.where(v, pt, np.nan), 2); med = np.array([np.median(ct[tr, m][v[tr, m]]) for m in range(M)])
    PC = np.stack([(np.nan_to_num(inp[:, m], nan=np.nanmean(inp[tr, m])) * MK[S[m]][0] + med[m] * MK[S[m]][1]) / 1e6 * 100 for m in range(M)], 1)
    return dict(S=S, M=M, Q=Q, Cr=Cr, avail=avail, te=te, P=P, LC=LC, PC=PC)


def points(d, Pm, C, abstain, lam):
    """per V: (per-problem score, per-problem cost) on the TEST problems -> arrays [nV, nte]"""
    te = d["te"]; U = np.where(d["avail"][te][None], VS[:, None, None] * ((1 + lam) * Pm[te][None] - lam) - C[te][None], -np.inf)
    m = U.argmax(2); best = np.take_along_axis(U, m[..., None], 2)[..., 0]
    q = np.take_along_axis(np.broadcast_to(d["Q"][te], (len(VS),) + d["Q"][te].shape), m[..., None], 2)[..., 0]
    c = np.take_along_axis(np.broadcast_to(d["Cr"][te], (len(VS),) + d["Cr"][te].shape), m[..., None], 2)[..., 0]
    ans = (best > 0) if abstain else np.ones_like(best, bool)
    return np.where(ans, q - lam * (1 - q), 0.0), np.where(ans, c, 0.0), ans


def fixed_points(d, lam):
    te = d["te"]; sc, co = [], []
    for m in range(d["M"]):
        if d["avail"][te, m].all():
            sc.append(d["Q"][te, m] - lam * (1 - d["Q"][te, m])); co.append(d["Cr"][te, m])
    return np.array(sc), np.array(co)


def run(label, name, cfile):
    d = load(name, cfile); te = d["te"]; out = {}
    for lam in LAMS:
        arms = {"fixed": fixed_points(d, lam)}
        for a, (Pm, C, ab) in {"paper": (d["P"], d["PC"], False), "paper+A": (d["P"], d["PC"], True), "ours": (d["P"], d["LC"], False),
                               "ours+A": (d["P"], d["LC"], True), "oracle+A": (d["Q"], d["Cr"], True)}.items():
            s, c, ans = points(d, Pm, C, ab, lam); arms[a] = (s, c)
            if a == "ours+A":
                ansrate = ans
        def summary(ii):
            H = {a: hull0(list(zip(c[:, ii].mean(1), s[:, ii].mean(1)))) for a, (s, c) in arms.items()}
            def g(a, b):                     # band where BOTH arms have real (non-origin) points: no credit for random-abstention mixing
                lo = max(H[a][1][1] if len(H[a]) > 1 else np.inf, H[b][1][1] if len(H[b]) > 1 else np.inf)
                hi = min(H[a][-1][1], H[b][-1][1])
                if not hi > lo:
                    return np.nan
                T = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 12)
                return 1 - float(np.exp(np.nanmean(np.log([cost_at(H[a], x) / cost_at(H[b], x) for x in T]))))
            T = [0.0, max(h[-1][1] for h in H.values())]
            return {"ours+A vs paper+A": g("ours+A", "paper+A"), "ours+A vs ours": g("ours+A", "ours"),
                    "paper+A vs paper": g("paper+A", "paper"), "ours vs paper": g("ours", "paper"),
                    "ours+A vs fixed": g("ours+A", "fixed"), "oracle+A vs ours+A": g("oracle+A", "ours+A")}, H, T
        g, H, T = summary(np.arange(len(te)))
        rng = np.random.default_rng(0); B = [summary(rng.integers(0, len(te), len(te)))[0] for _ in range(300)]
        ci = {k: np.nanpercentile([b[k] for b in B], [2.5, 97.5]) for k in g}
        print(f"\n{label} lam={lam:g}: (max score: " +
              ", ".join(f"{a} {h[-1][1]:.3f}" for a, h in H.items()) + ")")
        for k in g:
            print(f"   {k:<22} cost saved at matched score {g[k]*100:6.1f}% [{ci[k][0]*100:6.1f}, {ci[k][1]*100:6.1f}]")
        out[str(lam)] = {k: [g[k], *ci[k]] for k in g} | {"max_score": {a: h[-1][1] for a, h in H.items()}}
    return out


res = {lab: run(lab, n, c) for lab, n, c in POOLS}
json.dump(res, open(Path(__file__).parent / "abstain_oneshot.json", "w"), indent=1, default=float)
