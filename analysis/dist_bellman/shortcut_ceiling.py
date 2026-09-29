"""Why doesn't a shared difficulty latent beat the fixed cascade? An information ladder (free perfect verifier).

Every non-cascade arm runs the SAME policy -- only the per-problem information differs:
  index policy: repeatedly call the route with the lowest c_m/q_m among routes with draws left and V*q_m > c_m;
  stop at the first verified success or when no route is worth calling (optimal for known q, iid draws).
Information levels:
  none            q, c = train route means (no per-problem information)
  ours            q = prefill success prior, c = prefill cost head
  oracle latent   q_m = sigmoid(a_m + b_m d(x)), d = the problem's TRUE shared difficulty (mean over routes of the
                  smoothed logit of its empirical pass rate), (a_m, b_m) fitted on train; c_m = exp(true level + train
                  route offset), level = mean over routes of the problem's true log mean cost
  oracle full     q_m, c_m = the problem's own empirical pass rate (smoothed) and mean cost per route (leaky ceiling
                  for any prediction-based policy)
  clairvoyant     knows which draws succeed: pays only the cheapest successful draw (absolute bound)
Cascade: all enumerated fixed plans (as clean_comparison.py). Selection for every arm: operating point chosen on
CALIBRATION (upper hull, two-point mix), applied once to TEST; paired bootstrap. Also: the cascade's spend on
FAILED draws at each target (the most any shortcut can remove) and how often each arm skips the cheapest route.
Usage: python shortcut_ceiling.py <pool> <targets> <out.json> [legacy|market] [expensive-price factor]
"""
import itertools, json, sys
from pathlib import Path
import numpy as np
from pipelinerl.swe.scripts.livecodebench.mdp_utils import load_split_manifest, split_indices

R = "/mnt/llmd/results/exps/aristides/reason"
T = sys.argv[1]; TARGETS = [float(x) for x in sys.argv[2].split(",")]; OUT = sys.argv[3]
PRICES = sys.argv[4] if len(sys.argv) > 4 else "legacy"; FX = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
PR = {"oss20lo": 0.12, "oss20md": 0.57, "dsv4f": 0.111, "oss120md": 1.43, "oss120hi": 1.43}           # clean_comparison.py
MK = {"oss20lo": (0.018, 0.09), "oss20md": (0.018, 0.09), "dsv4f": (0.04704, 0.09408), "oss120md": (0.15, 0.6), "oss120hi": (0.15, 0.6)}
d = np.load(f"{R}/{T}/tensors.npz", allow_pickle=True)
ids = [str(x) for x in d["problem_ids"]]; S = [str(x) for x in d["model_slots"]]; M = len(S); ND = d["valid"].shape[2]
valid = d["valid"].astype(bool); truth = d["execution_outcome"].astype(bool)
pt, ct = d["prompt_tokens"].astype(float), d["completion_tokens"].astype(float)
leg = np.stack([(pt[:, m] + ct[:, m]) * PR[s] / 1e6 for m, s in enumerate(S)], 1)
real = leg if PRICES == "legacy" else np.stack([(pt[:, m] * MK[s][0] + ct[:, m] * MK[s][1]) / 1e6 for m, s in enumerate(S)], 1)
real = real * np.array([FX if s.startswith("oss120") else 1.0 for s in S])[None, :, None]
tr, cal, te = split_indices(load_split_manifest(Path(f"{R}/{T}/split_manifest.json"), ids), ids)
n = valid.sum(2); ok = (truth & valid).sum(2)
p = np.array([truth[tr, m][valid[tr, m]].mean() for m in range(M)])
c = np.array([real[tr, m][valid[tr, m]].mean() for m in range(M)])
cmean = np.where(n > 0, np.where(valid, real, 0).sum(2) / np.maximum(n, 1), np.nan)
lc = {json.loads(l)["problem_id"]: json.loads(l)["expected_costs"][:M] for l in open(f"{R}/{T}/cost_preds.jsonl")}
lp = {json.loads(l)["problem_id"]: json.loads(l)["p_successes"][:M] for l in open(f"{R}/{T}/content_preds.jsonl")}
LC = np.array([lc[i] for i in ids], float); LP = np.clip(np.array([lp[i] for i in ids], float), 1e-3, 1 - 1e-3)
legm = np.array([leg[tr, m][valid[tr, m]].mean() for m in range(M)])
LC = LC * (c / legm)[None]                                   # cost head was fitted in legacy dollars: rescale per route
# oracle quantities
qemp = (ok + 0.5) / (n + 1.0); lg = np.log(qemp / (1 - qemp)); dlat = np.nanmean(np.where(n > 0, lg, np.nan), 1)
QLAT = np.zeros((len(ids), M))
for m in range(M):
    A = np.c_[np.ones(len(tr)), dlat[tr]]; w = np.linalg.lstsq(A[n[tr, m] > 0], lg[tr, m][n[tr, m] > 0], rcond=None)[0]
    QLAT[:, m] = 1 / (1 + np.exp(-(w[0] + w[1] * dlat)))
logc = np.log(cmean); lvl = np.nanmean(logc, 1); off = np.nanmean(logc[tr] - lvl[tr, None], 0)
CLAT = np.exp(lvl[:, None] + off[None]); CFULL = np.where(np.isfinite(cmean), cmean, c[None])
# synthetic latents: the true shared difficulty and level with calibrated noise at R2 rho2 (then refit on train)
def noisy(x, rho2, seed):
    r = np.random.default_rng(seed); mu, sd = np.nanmean(x[tr]), np.nanstd(x[tr])
    return mu + np.sqrt(rho2) * (x - mu) + np.sqrt(1 - rho2) * sd * r.standard_normal(len(x))
def latent_info(dd, ll):
    Q = np.zeros((len(ids), M))
    for m in range(M):
        A = np.c_[np.ones(len(tr)), dd[tr]]; ww = np.linalg.lstsq(A[n[tr, m] > 0], lg[tr, m][n[tr, m] > 0], rcond=None)[0]
        Q[:, m] = 1 / (1 + np.exp(-(ww[0] + ww[1] * dd)))
    o = np.nanmean(logc[tr] - ll[tr, None], 0); return Q, np.exp(ll[:, None] + o[None])
dOurs = np.log(LP / (1 - LP)).mean(1); lOurs = np.log(LC).mean(1)
r2 = lambda a, b: 1 - np.mean((b[te] - np.polyval(np.polyfit(a[tr], b[tr], 1), a[te])) ** 2) / np.var(b[te])
print(f"OUR prefill vs the TRUE shared latent on test: difficulty R2 {r2(dOurs, dlat):.2f}, cost level R2 {r2(lOurs, lvl):.2f}")
INFO = {"none": (np.repeat(p[None], len(ids), 0), np.repeat(c[None], len(ids), 0)), "ours": (LP, LC),
        "oracle latent": (QLAT, CLAT), "oracle full": (qemp, CFULL)}
for rho2 in (0.5, 0.7, 0.85):
    INFO[f"latent R2={rho2}"] = latent_info(noisy(dlat, rho2, 1), noisy(lvl, rho2, 2))
rng = np.random.default_rng(0); ORD = [[rng.permutation(ND) for _ in S] for _ in range(3)]
cheapest = int(np.argmin(c))


def episode(i, o, chooser):
    """-> (solved, spent, spent on failed draws, first route)"""
    ptr = [0] * M; fails = [0] * M; spent = 0.0; wasted = 0.0; first = None
    while True:
        m = chooser(fails)
        if m is None:
            return 0.0, spent, spent, first
        while ptr[m] < ND and not valid[i, m, ORD[o][m][ptr[m]]]:
            ptr[m] += 1
        if ptr[m] >= ND:
            return 0.0, spent, spent, first
        first = m if first is None else first
        k = ORD[o][m][ptr[m]]; ptr[m] += 1; spent += real[i, m, k]
        if truth[i, m, k]:
            return 1.0, spent, wasted, first
        wasted += real[i, m, k]; fails[m] += 1


def per_problem(idx, make):
    out = np.zeros((len(idx), 4))                      # acc, cost (cents), wasted (cents), skipped-the-cheapest
    for j, i in enumerate(idx):
        ch = make(i)
        for o in range(3):
            a, s, w, f = episode(i, o, ch); out[j] += (a / 3, s * 100 / 3, w * 100 / 3, (f is not None and f != cheapest) / 3)
    return out


def cascade_maker(plan):
    seq = [m for m in sorted(range(M), key=lambda m: c[m]) for _ in range(plan[m])]
    def make(i):
        def ch(fails):
            used = [0] * M
            for m in seq:
                used[m] += 1
                if used[m] > fails[m] and fails[m] < n[i, m]:
                    return m
            return None
        return ch
    return make


def index_maker(V, q, cc):
    def make(i):
        def ch(fails):
            best, bi = None, np.inf
            for m in range(M):
                if fails[m] < n[i, m] and V * q[i, m] > cc[i, m] and cc[i, m] / q[i, m] < bi:
                    best, bi = m, cc[i, m] / q[i, m]
            return best
        return ch
    return make


def clair(idx, V):
    out = np.zeros((len(idx), 4))
    for j, i in enumerate(idx):
        win = np.where(truth[i] & valid[i], real[i], np.inf); b = win.min()
        if np.isfinite(b) and V > b:
            out[j] = (1.0, b * 100, 0.0, float(np.unravel_index(win.argmin(), win.shape)[0] != cheapest))
    return out


grid = {s: range(0, 9) if i == cheapest else range(0, 3) for i, s in enumerate(S)}
plans = [pl for pl in itertools.product(*[grid[s] for s in S]) if sum(pl)]
Vs = np.geomspace(c.min() / p.max() * 0.3, c.max() / p.min() * 300, 60)
curves = {"cascade": [(per_problem(cal, cascade_maker(pl)), per_problem(te, cascade_maker(pl))) for pl in plans]}
for name, (q, cc) in INFO.items():
    curves[name] = [(per_problem(cal, index_maker(V, q, cc)), per_problem(te, index_maker(V, q, cc))) for V in Vs]
curves["clairvoyant"] = [(clair(cal, V), clair(te, V)) for V in Vs]


def pick(name, target):
    pts = sorted(((cl[:, 1].mean(), cl[:, 0].mean(), j) for j, (cl, _) in enumerate(curves[name])))
    h = [(0.0, 0.0, None)]
    for x in pts:
        while len(h) >= 2 and (h[-1][1]-h[-2][1])*(x[0]-h[-2][0]) <= (x[1]-h[-2][1])*(h[-1][0]-h[-2][0]):
            h.pop()
        h.append(x)
    for lo, hi in zip(h, h[1:]):
        if lo[1] <= target <= hi[1]:
            w = (target - lo[1]) / max(hi[1] - lo[1], 1e-12)
            tl = curves[name][lo[2]][1] if lo[2] is not None else np.zeros((len(te), 4))
            return (1 - w) * tl + w * curves[name][hi[2]][1]
    return None


print(f"{T} [{PRICES} prices, oss120 x{FX}]: {len(tr)}/{len(cal)}/{len(te)}; route mean cost (c) {dict(zip(S, (c*100).round(4)))}; "
      f"train pass {dict(zip(S, p.round(2)))}")
res = {}
for t in TARGETS:
    base = pick("cascade", t)
    if base is None:
        print(f"  {t*100:.0f}%: cascade unreachable"); continue
    print(f"  target {t*100:.0f}%: cascade test {base[:,0].mean()*100:.1f}% @ {base[:,1].mean():.4f}c; spend on FAILED draws "
          f"{base[:,2].mean()/base[:,1].mean()*100:.0f}%; skips cheapest {base[:,3].mean()*100:.0f}%")
    for name in curves:
        if name == "cascade":
            continue
        x = pick(name, t)
        if x is None:
            print(f"     {name:<14} unreachable"); continue
        bs = [];
        for _ in range(1000):
            ii = rng.integers(0, len(te), len(te)); bs.append((x[ii, 1].mean() / base[ii, 1].mean(), (x[ii, 0] - base[ii, 0]).mean()))
        bs = np.array(bs); r = x[:, 1].mean() / base[:, 1].mean()
        print(f"     {name:<14} test {x[:,0].mean()*100:.1f}% @ {x[:,1].mean():.4f}c  ratio {r:.2f} [{np.percentile(bs[:,0],2.5):.2f},"
              f"{np.percentile(bs[:,0],97.5):.2f}]  acc {(x[:,0]-base[:,0]).mean()*100:+.1f}  failed-draw share "
              f"{x[:,2].mean()/max(x[:,1].mean(),1e-12)*100:.0f}%  skips cheapest {x[:,3].mean()*100:.0f}%")
        res.setdefault(str(t), {})[name] = {"acc": x[:, 0].mean(), "cost": x[:, 1].mean(), "ratio": r,
                                            "ci": list(np.percentile(bs[:, 0], [2.5, 97.5]))}
    res.setdefault(str(t), {})["cascade"] = {"acc": base[:, 0].mean(), "cost": base[:, 1].mean(), "failed_share": base[:, 2].mean() / base[:, 1].mean()}
Path(OUT).write_text(json.dumps(res, indent=1, default=float))
