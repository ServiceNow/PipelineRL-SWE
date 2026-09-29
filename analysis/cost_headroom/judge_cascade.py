"""No-verifier, single-submission CASCADE with a learned answer judge vs one-shot routing (LCB 5-rung pool).

Every arm submits at most ONE answer per problem and never sees a test result. Each tier call = one real stored draw (3 draw
orderings averaged), priced at market rates. Score = accuracy - lam x error rate over all problems (abstain = 0, cost 0 after
what was already spent).
Arms:
  router        one-shot prefill router (success head + 4B cost head), rule V((1+lam)p - lam) - c; "+A" may abstain
  paper         same, paper cost rule (input + median train output)
  cascade[J]    FrugalGPT-style: an ordered subset of tiers (cheapest first); at each tier draw once, the judge J scores the
                attempt, submit if score >= that tier's threshold, else escalate; the last tier submits (lam = 0) or submits
                iff score >= its own threshold (lam > 0, else abstain)
  hybrid[J]     the prefill router picks the ENTRY tier (value V_h); the cascade runs from there upward
Judges J: 4B = frozen Qwen3-4B prefill probe on problem + code + "Is this solution correct?" (pool_v2_judge_full), charged at
gpt-oss-20b's input price for its prompt tokens (uncached) or for the tokens after the shared problem prefix (cached);
137M = per-tier fine-tuned jina-code scorer (judge_preds_ft137m.jsonl, if present), charged 0; oracle = the true outcome
(cost 0; a free perfect verifier with one submission -- ceiling, not a deployable arm).
Selection (one protocol for all arms): each arm's candidate set (plans x thresholds, or V grid) is evaluated on CALIBRATION;
for each target score the cheapest two-point mix on the calibration upper hull (through the origin) that reaches it is
applied once to TEST; cost ratio vs the router arm and score difference, paired bootstrap over test problems.
Usage: python judge_cascade.py [lams] [targets]
"""
import itertools, json, sys, numpy as np, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from decompose import MK, R

LAMS = [float(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0.0, 1.0, 3.0]
FRACS = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0.5, 0.7, 0.8, 0.9, 0.95]   # of the router's max score
T = "pool_v2_tensors_5rung"; D = R / T; J = R / "pool_v2_judge_full"
t = np.load(D / "tensors.npz", allow_pickle=True)
S = [str(s) for s in t["model_slots"]]; M = len(S); ids = [str(p) for p in t["problem_ids"]]; pi = {p: i for i, p in enumerate(ids)}
valid = t["valid"].astype(bool); ND = valid.shape[2]
pt, ct = t["prompt_tokens"].astype(float), t["completion_tokens"].astype(float)
real = np.stack([(pt[:, m] * MK[s][0] + ct[:, m] * MK[s][1]) / 1e6 * 100 for m, s in enumerate(S)], 1)      # cents per draw
sp = json.load(open(D / "split_manifest.json")); idx = {k: np.array([pi[str(p)] for p in sp[f"{k}_problem_ids"]]) for k in ("train", "calibration", "test")}
tr, cal, te = idx["train"], idx["calibration"], idx["test"]
# judge scores, truth and judge prompt lengths per (problem, route, draw)
man = [json.loads(l) for l in open(J / "judge_manifest.jsonl")]
key = {r["example_id"]: (pi[r["problem_id"]], S.index(r["slot"]), int(r["draw"])) for r in man}
Y = np.full(valid.shape, np.nan); [Y.__setitem__(key[r["example_id"]], float(r["correct"])) for r in man]
prompts = {}
for f in glob.glob(str(J / "judge_shard*.jsonl")):
    for l in open(f):
        r = json.loads(l); prompts[r["problem_id"]] = r["prompt"]
LEN = np.full(valid.shape, np.nan); NEW = np.full(valid.shape, np.nan)
byp = {}
for e, k in key.items():
    byp.setdefault(k[0], []).append(e)
import os
for i, es in byp.items():                         # shared problem prefix = longest common prefix of the problem's judge prompts
    pre = os.path.commonprefix([prompts[e] for e in es]) if len(es) > 1 else ""
    for e in es:
        LEN[key[e]] = len(prompts[e]) / 3.5; NEW[key[e]] = (len(prompts[e]) - len(pre)) / 3.5     # ~3.5 chars per token
JPRICE = MK["oss20lo"][0] / 1e6 * 100                                                                  # cents per input token
JUDGES = {}
for tag, f in (("4B", "judge_preds_causal.jsonl"), ("137M", "judge_preds_ft137m.jsonl")):
    if (J / f).exists():
        A = np.full(valid.shape, np.nan)
        for l in open(J / f):
            r = json.loads(l); A[key[r["example_id"]]] = r["p_correct"]
        JUDGES[tag] = A
JCOST = {"4B": JPRICE * LEN, "4B cached": JPRICE * NEW, "137M": np.zeros(valid.shape), "oracle": np.zeros(valid.shape)}
JSCORE = {"4B": JUDGES["4B"], "4B cached": JUDGES["4B"], "oracle": np.nan_to_num(Y)}
if "137M" in JUDGES:
    JSCORE["137M"] = JUDGES["137M"]
print(f"judge cost per attempt (cents): 4B uncached mean {np.nanmean(JCOST['4B']):.5f}, cached {np.nanmean(JCOST['4B cached']):.5f}; "
      f"oss20lo call mean {np.nanmean(np.where(valid[:, 0], real[:, 0], np.nan)):.5f}")
# one attempt per (problem, ordering, route): the first valid draw of a random permutation
rng = np.random.default_rng(0); NO = 3
DR = np.zeros((len(ids), NO, M), int)
for i in range(len(ids)):
    for o in range(NO):
        for m in range(M):
            vd = np.where(valid[i, m] & np.isfinite(Y[i, m]))[0]
            DR[i, o, m] = rng.permutation(vd)[0] if len(vd) else -1
OK = DR >= 0
take = lambda A: np.where(OK, np.take_along_axis(A[:, None].repeat(NO, 1), np.maximum(DR, 0)[..., None], 3)[..., 0], np.nan)
y = np.nan_to_num(take(Y)); c = take(real)
c = np.where(OK, c, 1e9)
# prefill router inputs
lp = {json.loads(l)["problem_id"]: json.loads(l)["p_successes"][:M] for l in open(D / "content_preds.jsonl")}
lc = {json.loads(l)["problem_id"]: json.loads(l)["expected_costs"][:M] for l in open(D / "cost_preds_probe.jsonl")}
P = np.clip(np.array([lp[p] for p in ids]), 0, 1); LC = np.array([lc[p] for p in ids]) * 100
inp = np.nanmean(np.where(valid, pt, np.nan), 2); med = np.array([np.median(ct[tr, m][valid[tr, m]]) for m in range(M)])
PC = np.stack([(np.nan_to_num(inp[:, m], nan=np.nanmean(inp[tr, m])) * MK[S[m]][0] + med[m] * MK[S[m]][1]) / 1e6 * 100 for m in range(M)], 1)
order = list(np.argsort(np.nanmean(np.where(valid[tr], real[tr], np.nan), (0, 2))))                  # tiers by train mean cost
VS = np.geomspace(1e-4, 1e2, 60)


def router_pts(C, lam, abstain):
    out = []
    for V in VS:
        U = V * ((1 + lam) * P - lam) - C; m = U.argmax(1); ans = (U.max(1) > 0) if abstain else np.ones(len(ids), bool)
        yy = y[np.arange(len(ids)), :, m]; cc = c[np.arange(len(ids)), :, m]
        out.append((np.where(ans[:, None], yy - lam * (1 - yy), 0).mean(1), np.where(ans[:, None], cc, 0).mean(1)))
    return out


def cascade_pts(jtag, lam, entry_V=None):
    sc, jc = take(JSCORE[jtag]), take(JCOST[jtag])
    q = {m: np.nanquantile(JSCORE[jtag][tr, m][np.isfinite(JSCORE[jtag][tr, m])], [.1, .25, .4, .55, .7, .85, .95]) for m in range(M)}
    if jtag == "oracle":
        q = {m: np.array([0.5]) for m in range(M)}
    entries = [None] if entry_V is None else entry_V
    N = len(ids); ar = np.arange(N)
    g = lambda A, m: A[:, :, m]                                   # [N, NO] slice of tier m
    out = []
    for Vh in entries:
        er = np.zeros(N, int) if Vh is None else np.array([order.index(e) for e in (Vh * P - LC).argmax(1)])   # entry rank
        ent = np.array(order)[er]
        for L in (1, 2, 3):
            for tiers in itertools.combinations(order, L):
                grids = [q[m] for m in tiers[:-1]] + [np.array([-np.inf]) if lam == 0 else np.r_[-np.inf, q[tiers[-1]]]]
                for taus in itertools.product(*grids):
                    score = np.zeros((N, NO)); cost = np.zeros((N, NO)); done = np.zeros((N, NO), bool)
                    for m, tau in zip(tiers, taus):
                        act = ~done & (er <= order.index(m))[:, None]        # tiers below the entry are skipped
                        judged = np.isfinite(tau)
                        cost += act * (g(c, m) + (g(jc, m) if judged else 0))
                        sub = act & ((g(sc, m) >= tau) if judged else True)
                        score += sub * (g(y, m) - lam * (1 - g(y, m))); done |= sub
                    fb = ~done & (er > order.index(tiers[-1]))[:, None]      # entry above every listed tier: call it, submit
                    ye = y[ar, :, ent]; ce = c[ar, :, ent]
                    cost += fb * ce; score += fb * (ye - lam * (1 - ye))
                    out.append((score.mean(1), cost.mean(1)))
    return out


def pick(pts, target):
    cands = sorted(((float(cc[cal].mean()), float(s[cal].mean()), j) for j, (s, cc) in enumerate(pts)))
    h = [(0.0, 0.0, None)]
    for x in cands:
        if x[1] <= h[-1][1]:
            continue
        while len(h) >= 2 and (h[-1][1] - h[-2][1]) * (x[0] - h[-2][0]) <= (x[1] - h[-2][1]) * (h[-1][0] - h[-2][0]):
            h.pop()
        h.append(x)
    for lo, hi in zip(h, h[1:]):
        if lo[1] <= target <= hi[1]:
            w = (target - lo[1]) / max(hi[1] - lo[1], 1e-12)
            f = lambda j, k: pts[j][k][te] if j is not None else np.zeros(len(te))
            return (1 - w) * f(lo[2], 0) + w * f(hi[2], 0), (1 - w) * f(lo[2], 1) + w * f(hi[2], 1)
    return None


res = {}
for lam in LAMS:
    arms = {"router": router_pts(LC, lam, lam > 0), "paper": router_pts(PC, lam, lam > 0)}
    for jt in JSCORE:
        arms[f"cascade[{jt}]"] = cascade_pts(jt, lam)
    for jt in [j for j in JSCORE if j != "oracle"]:
        arms[f"hybrid[{jt}]"] = cascade_pts(jt, lam, entry_V=list(VS))
    top = max(np.mean(s[cal]) for s, _ in arms["router"])
    print(f"\n=== lam = {lam:g}  (router max calibration score {top:.3f}; targets = fractions of it)")
    for fr in FRACS:
        base = pick(arms["router"], fr * top)
        if base is None:
            continue
        print(f"  target {fr*top:.3f}: router test score {base[0].mean():.3f} @ {base[1].mean():.4f}c")
        for a in arms:
            if a == "router":
                continue
            x = pick(arms[a], fr * top)
            if x is None:
                print(f"     {a:<18} unreachable"); continue
            B = []
            for _ in range(1000):
                ii = rng.integers(0, len(te), len(te)); B.append((x[1][ii].mean() / base[1][ii].mean(), (x[0][ii] - base[0][ii]).mean()))
            B = np.array(B)
            print(f"     {a:<18} test {x[0].mean():.3f} @ {x[1].mean():.4f}c   cost ratio {x[1].mean()/base[1].mean():.2f} "
                  f"[{np.percentile(B[:,0],2.5):.2f},{np.percentile(B[:,0],97.5):.2f}]   score diff {(x[0]-base[0]).mean()*100:+.1f} "
                  f"[{np.percentile(B[:,1],2.5)*100:+.1f},{np.percentile(B[:,1],97.5)*100:+.1f}]")
            res.setdefault(str(lam), {}).setdefault(str(fr), {})[a] = [x[0].mean(), x[1].mean(), x[1].mean() / base[1].mean()]
        res[str(lam)][str(fr)]["router"] = [base[0].mean(), base[1].mean(), 1.0]
json.dump(res, open(Path(__file__).parent / "judge_cascade.json", "w"), indent=1, default=float)

# ---- matched-score view (decompose.py protocol): TEST upper hulls through the origin, cost at matched score over the band
# where both arms have real points; paired bootstrap. Selection on test favours arms with MORE candidates (cascades ~1000
# plans vs the router's 60 V values), i.e. it is generous to the cascades.
from decompose import cost_at


def thull(pts, ii):
    P_ = sorted(set((float(cc[ii].mean()), float(s[ii].mean())) for s, cc in pts) | {(0.0, 0.0)}); h = []
    for x in P_:
        if h and x[1] <= h[-1][1]:
            continue
        while len(h) >= 2 and (h[-1][1] - h[-2][1]) * (x[0] - h[-2][0]) <= (x[1] - h[-2][1]) * (h[-1][0] - h[-2][0]):
            h.pop()
        h.append(x)
    return h


def saved(ha, hb):
    lo = max(ha[1][1] if len(ha) > 1 else np.inf, hb[1][1] if len(hb) > 1 else np.inf); hi = min(ha[-1][1], hb[-1][1])
    if not hi > lo:
        return np.nan
    Tt = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 12)
    return 1 - float(np.exp(np.nanmean(np.log([cost_at(ha, x) / cost_at(hb, x) for x in Tt]))))


print("\n==== MATCHED-SCORE (test hulls): cost saved vs the one-shot router [95% CI]; + = cheaper than the router")
mres = {}
for lam in LAMS:
    arms = {"router": router_pts(LC, lam, lam > 0), "paper": router_pts(PC, lam, lam > 0)}
    for jt in JSCORE:
        arms[f"cascade[{jt}]"] = cascade_pts(jt, lam)
    for jt in [j for j in JSCORE if j != "oracle"]:
        arms[f"hybrid[{jt}]"] = cascade_pts(jt, lam, entry_V=list(VS))
    print(f"  lam = {lam:g}: max test score " + ", ".join(f"{a} {thull(p, te)[-1][1]:.3f}" for a, p in arms.items()))
    brng = np.random.default_rng(1); BS = [brng.choice(te, len(te)) for _ in range(200)]
    for a in arms:
        if a == "router":
            continue
        g = saved(thull(arms[a], te), thull(arms["router"], te))
        b = [saved(thull(arms[a], ii), thull(arms["router"], ii)) for ii in BS]
        print(f"     {a:<18} {g*100:6.1f}% [{np.nanpercentile(b,2.5)*100:6.1f}, {np.nanpercentile(b,97.5)*100:6.1f}]")
        mres.setdefault(str(lam), {})[a] = [g, *np.nanpercentile(b, [2.5, 97.5])]
json.dump({"calibration_protocol": res, "matched_test_hull": mres}, open(Path(__file__).parent / "judge_cascade.json", "w"), indent=1, default=float)
