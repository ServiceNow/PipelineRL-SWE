"""When does per-query cost prediction pay in one-shot routing? Headroom x capture, per pool.

One-shot rule argmax_m p_m(x)*V - c_m(x), same success head p for every arm; only the cost estimate changes:
  paper    input tokens of x + the route's median TRAIN output length (arXiv 2603.20895)
  learned  the 4B-prefill cost head (activation_cost_preds.py, floored at the train minimum)
  ORACLE   the problem's TRUE expected cost on that route (mean over its valid draws) -- perfect cost knowledge
HEADROOM = how much cheaper the oracle is than the paper rule at matched accuracy: what knowing output length
per problem is worth at all. CAPTURE = learned gain / headroom. Both from TEST frontiers (V swept; upper hull,
mixing allowed), identical treatment for every arm; summarised as 1 - geometric-mean cost ratio over accuracy
targets reachable by all arms; paired bootstrap over test problems.
Per-route descriptives: spread of per-problem cost (p90/p10, sd of log), ICC of log cost (share of variance that
is between problems, i.e. predictable in principle), output share of cost, learned head's dollar R2 on test, and
the REROUTE rate (share of test problems where the oracle and paper rules pick different routes at the median
operating point).
Usage: python decompose.py <tensors_dir>[:<cost_preds_file>] ...   (prices below; out: analysis/cost_headroom/)
"""
import json, sys, numpy as np
from pathlib import Path

R = Path("/mnt/llmd/results/exps/aristides/reason")
PR = {"oss20lo": 0.12, "oss20md": 0.57, "dsv4f": 0.111, "oss120md": 1.43, "oss120hi": 1.43}
VS = np.geomspace(1e-5, 100, 300)


def hull(pts):
    pts = sorted(set(pts)); h = []
    for c, a in pts:
        if h and a <= h[-1][1]:
            continue
        while len(h) >= 2 and (h[-1][1] - h[-2][1]) * (c - h[-2][0]) <= (a - h[-2][1]) * (h[-1][0] - h[-2][0]):
            h.pop()
        h.append((c, a))
    return h


def cost_at(h, t):
    if not h or t > h[-1][1] or t < h[0][1]:
        return np.nan
    for (c1, a1), (c2, a2) in zip(h, h[1:]):
        if a1 <= t <= a2:
            return c1 + (c2 - c1) * (t - a1) / max(a2 - a1, 1e-12)
    return h[0][0]


def pool(spec):
    name, _, cfile = spec.partition(":")
    D = R / name; t = np.load(D / "tensors.npz", allow_pickle=True)
    S = [str(s) for s in t["model_slots"]]; M = len(S)
    pids = [str(p) for p in t["problem_ids"]]; pi = {p: i for i, p in enumerate(pids)}
    v = t["valid"].astype(bool); ok = (t["final_outcome"] & t["valid"]).astype(float)
    pt, ct = t["prompt_tokens"].astype(float), t["completion_tokens"].astype(float)
    real = np.stack([(pt[:, m] + ct[:, m]) * PR[s] / 1e6 * 100 for m, s in enumerate(S)], 1)   # cents per draw
    n = v.sum(2); avail = n > 0
    Q = np.where(avail, (ok * v).sum(2) / np.maximum(n, 1), 0)
    Cr = np.where(avail, (real * v).sum(2) / np.maximum(n, 1), 1e9)
    sp = json.load(open(D / "split_manifest.json"))
    idx = {k: np.array([pi[str(p)] for p in sp[f"{k}_problem_ids"]]) for k in ("train", "calibration", "test")}
    lp = {json.loads(l)["problem_id"]: json.loads(l)["p_successes"][:M] for l in open(D / "content_preds.jsonl")}
    lc = {json.loads(l)["problem_id"]: json.loads(l)["expected_costs"][:M] for l in open(D / (cfile or "cost_preds.jsonl"))}
    P = np.array([lp[p] for p in pids]); LC = np.array([lc[p] for p in pids]) * 100
    tr = idx["train"]; inp = np.nanmean(np.where(v, pt, np.nan), 2)
    med = np.array([np.median(ct[tr, m][v[tr, m]]) for m in range(M)])
    PC = np.stack([(np.nan_to_num(inp[:, m], nan=np.nanmean(inp[tr, m])) + med[m]) * PR[S[m]] / 1e6 * 100 for m in range(M)], 1)
    te = idx["test"]
    arms = {"paper": PC, "learned": LC, "ORACLE": np.where(avail, Cr, 1e9)}
    # per arm, per V: the route chosen on each test problem -> accuracy / cost arrays [nV, nte]
    curves = {}
    for a, C in arms.items():
        U = np.where(avail[te][None], P[te][None] * VS[:, None, None] - C[te][None], -np.inf)
        ch = U.argmax(2)
        curves[a] = (np.take_along_axis(np.broadcast_to(Q[te], (len(VS),) + Q[te].shape), ch[..., None], 2)[..., 0],
                     np.take_along_axis(np.broadcast_to(Cr[te], (len(VS),) + Cr[te].shape), ch[..., None], 2)[..., 0], ch)

    def summary(ii):
        H = {a: hull(list(zip(c[1][:, ii].mean(1), c[0][:, ii].mean(1)))) for a, c in curves.items()}
        lo = max(h[0][1] for h in H.values()); hi = min(h[-1][1] for h in H.values())
        lo, hi = lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo)
        T = np.linspace(lo, hi, 12)
        base = np.array([cost_at(H["paper"], x) for x in T])
        r = {a: np.array([cost_at(H[a], x) for x in T]) / base for a in ("learned", "ORACLE")}
        g = {a: 1 - float(np.exp(np.nanmean(np.log(r[a])))) for a in r}
        return g, T, r, H
    g, T, r, H = summary(np.arange(len(te)))
    rng = np.random.default_rng(0); B = []
    for _ in range(500):
        gb, *_ = summary(rng.integers(0, len(te), len(te))); B.append(gb)
    ci = {a: np.percentile([b[a] for b in B], [2.5, 97.5]).tolist() for a in g}
    # reroute rate at the V whose paper-rule accuracy is closest to the middle of the matched band
    mid = T[len(T) // 2]; vi = int(np.argmin(np.abs(curves["paper"][0].mean(1) - mid)))
    reroute = float((curves["ORACLE"][2][vi] != curves["paper"][2][vi]).mean())
    reroute_l = float((curves["learned"][2][vi] != curves["paper"][2][vi]).mean())
    routes = {}
    for m, s in enumerate(S):
        ok_p = avail[:, m]; c = Cr[ok_p, m]
        lv = np.log(np.where(v[:, m], real[:, m], np.nan))
        tot = np.nanvar(lv[ok_p]); within = np.nanmean(np.nanvar(lv[ok_p], 1))
        tte = te[avail[te, m]]
        r2 = 1 - ((LC[tte, m] - Cr[tte, m]) ** 2).sum() / ((Cr[tte, m] - Cr[tte, m].mean()) ** 2).sum()
        out_share = float(np.nansum(np.where(v[:, m], ct[:, m], 0)) / np.nansum(np.where(v[:, m], pt[:, m] + ct[:, m], 0)))
        routes[s] = dict(acc=float(Q[ok_p, m].mean()), mean_cost_c=float(c.mean()),
                         p90_p10=float(np.percentile(c, 90) / np.percentile(c, 10)), sd_log=float(np.log(c).std()),
                         icc=float(1 - within / tot) if tot > 0 else np.nan, output_share=out_share, test_r2=float(r2))
    return dict(pool=name, cost_file=cfile or "cost_preds.jsonl", n_test=int(len(te)), band=[float(T[0]), float(T[-1])],
                headroom=g["ORACLE"], headroom_ci=ci["ORACLE"], learned_gain=g["learned"], learned_ci=ci["learned"],
                capture=g["learned"] / g["ORACLE"] if g["ORACLE"] > 0.01 else np.nan,
                reroute_oracle=reroute, reroute_learned=reroute_l, routes=routes,
                ratios={a: dict(zip([round(float(x), 3) for x in T], [float(y) for y in r[a]])) for a in r})


def main():
    res = [pool(s) for s in sys.argv[1:]]
    out = Path("analysis/cost_headroom"); out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "decompose.json", "w"), indent=1, default=float)
    print("HEADROOM = cost saved at matched accuracy by PERFECT per-problem cost knowledge vs the paper rule; "
          "learned = the 4B-prefill head; capture = learned / headroom (test frontiers, band = accuracies all arms reach)")
    print(f"{'pool':<26}{'band':>13}{'headroom [95% CI]':>24}{'learned [95% CI]':>24}{'capture':>9}{'reroute or/lrn':>16}")
    for x in res:
        print(f"{x['pool']:<26}{x['band'][0]*100:6.0f}-{x['band'][1]*100:3.0f}%   {x['headroom']*100:5.1f}% "
              f"[{x['headroom_ci'][0]*100:5.1f},{x['headroom_ci'][1]*100:5.1f}]   {x['learned_gain']*100:5.1f}% "
              f"[{x['learned_ci'][0]*100:5.1f},{x['learned_ci'][1]*100:5.1f}]{x['capture']:9.2f}"
              f"{x['reroute_oracle']*100:8.0f}%/{x['reroute_learned']*100:.0f}%")
    print("\nper route: acc, mean cost (c), p90/p10 per-problem cost, sd log cost, ICC (between-problem share), output share, head test R2")
    for x in res:
        print(f"  {x['pool']}")
        for s, d in x["routes"].items():
            print(f"    {s:<9} acc {d['acc']*100:5.1f}%  {d['mean_cost_c']:.4f}c  p90/p10 {d['p90_p10']:6.1f}  sdlog {d['sd_log']:.2f}  "
                  f"ICC {d['icc']:.2f}  out {d['output_share']*100:3.0f}%  R2 {d['test_r2']:+.2f}")


if __name__ == "__main__":
    main()
