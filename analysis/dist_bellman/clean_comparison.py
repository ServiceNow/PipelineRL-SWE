"""ONE protocol for every arm (fixed before looking at results): free perfect verifier, cost vs fixed cascade.

Arms (all candidates simulated identically, 3 draw orderings, stop at first verified success):
  cascade                 every plan in the enumerated grid (draw counts per route, run cheapest-first)
  bellman: no prefill     route train pass rates + train-mean costs (no per-problem information)
  bellman: + cost head    learned per-problem costs, route pass rates
  bellman: + cost + prior learned per-problem costs + prefill per-problem pass-rate prior
Each Bellman arm is run with the post-failure decay q_f = q * k/(k+f) at the fixed k=2 AND at k FITTED ON TRAIN
(max log-likelihood of each next draw's outcome given the failures before it, same beliefs). Bellman candidates:
a 60-point value grid. Selection: for each target accuracy, each arm's operating point is the cheapest mix of two
adjacent points on its CALIBRATION upper hull that reaches the target on calibration; that fixed mix is applied
once to TEST. Reported: test accuracy and cost, cost ratio vs the cascade and accuracy difference, paired
bootstrap over test problems (the chosen mixes held fixed).
"""
import itertools, json, sys
from pathlib import Path
import numpy as np
from pipelinerl.swe.scripts.livecodebench.replay_bellman_verification import build_solver
from pipelinerl.swe.scripts.livecodebench.mdp_utils import load_split_manifest, split_indices

R = "/mnt/llmd/results/exps/aristides/reason"
T = sys.argv[1]; TARGETS = [float(x) for x in sys.argv[2].split(",")]; OUT = sys.argv[3]
PR = {"oss20lo": 0.12, "oss20md": 0.57, "dsv4f": 0.111, "oss120md": 1.43, "oss120hi": 1.43}
d = np.load(f"{R}/{T}/tensors.npz", allow_pickle=True)
ids = [str(x) for x in d["problem_ids"]]; S = [str(x) for x in d["model_slots"]]; M = len(S)
valid = d["valid"].astype(bool); truth = d["execution_outcome"].astype(bool)
real = np.stack([(d["prompt_tokens"][:, m] + d["completion_tokens"][:, m]) * PR[s] / 1e6 for m, s in enumerate(S)], 1)
tr, cal, te = split_indices(load_split_manifest(Path(f"{R}/{T}/split_manifest.json"), ids), ids)
p = np.array([truth[tr, m][valid[tr, m]].mean() for m in range(M)])
c = np.array([real[tr, m][valid[tr, m]].mean() for m in range(M)])
lc = {json.loads(l)["problem_id"]: json.loads(l)["expected_costs"][:M] for l in open(f"{R}/{T}/cost_preds.jsonl")}
lp = {json.loads(l)["problem_id"]: json.loads(l)["p_successes"][:M] for l in open(f"{R}/{T}/content_preds.jsonl")}
rng = np.random.default_rng(0)
ORD = [[rng.permutation(valid.shape[2]) for _ in S] for _ in range(3)]


def episode(i, o, chooser):
    ptr = [0] * M; fails = [0] * M; spent = 0.0; t = 0
    while True:
        m = chooser(fails, [int(valid[i, k].sum()) - fails[k] for k in range(M)], t)
        if m is None:
            return 0.0, spent
        while ptr[m] < valid.shape[2] and not valid[i, m, ORD[o][m][ptr[m]]]:
            ptr[m] += 1
        if ptr[m] >= valid.shape[2]:
            return 0.0, spent
        k = ORD[o][m][ptr[m]]; ptr[m] += 1; spent += real[i, m, k]; t += 1
        if truth[i, m, k]:
            return 1.0, spent
        fails[m] += 1


def per_problem(idx, make):
    """(acc, cost-in-cents) per problem, averaged over orderings."""
    out = np.zeros((len(idx), 2))
    for j, i in enumerate(idx):
        ch = make(i)
        for o in range(3):
            a, s = episode(i, o, ch); out[j] += (a / 3, s * 100 / 3)
    return out


def cascade_maker(plan):
    seq = [m for m in sorted(range(M), key=lambda m: c[m]) for _ in range(plan[m])]
    def make(i):
        def ch(fails, remaining, t):
            used = [0] * M
            for m in seq:
                used[m] += 1
                if used[m] > fails[m] and remaining[m] > 0:
                    return m
            return None
        return ch
    return make


def bellman_maker(V, k, use_cost, use_prior):
    def make(i):
        caps = tuple(int(valid[i, m].sum()) for m in range(M))
        prior = np.array(lp[ids[i]]) if use_prior else p
        costs = np.array(lc[ids[i]]) if use_cost else c
        solve, action = build_solver(prior, costs, V, 0.0, caps, 8, k)
        def ch(fails, remaining, t):
            rem = tuple(max(0, caps[m] - fails[m]) for m in range(M))
            val, route, _ = action(tuple(fails), rem, 8 - t)
            return route if route is not None and val > 0 else None
        return ch
    return make


def fit_decay(use_prior):
    """k maximising the train log-likelihood of each draw's outcome after the route's earlier failures."""
    best = None
    for k in (0.25, 0.5, 1, 2, 4, 8, 16, 1e6):
        ll = 0.0
        for i in tr:
            prior = np.array(lp[ids[i]]) if use_prior else p
            for m in range(M):
                f = 0
                for kk in ORD[0][m]:
                    if not valid[i, m, kk]:
                        continue
                    q = np.clip(prior[m] * k / (k + f), 1e-4, 1 - 1e-4)
                    y = truth[i, m, kk]; ll += np.log(q if y else 1 - q)
                    if y:
                        break
                    f += 1
        if best is None or ll > best[0]:
            best = (ll, k)
    return best[1]


grid = {s: range(0, 9) if i == int(np.argmin(c)) else range(0, 3) for i, s in enumerate(S)}
arms = {"cascade": [cascade_maker([pl[m] for m in range(M)]) for pl in
                    (dict(enumerate(x)) for x in itertools.product(*[grid[s] for s in S])) if sum(pl.values())]}
Vs = np.geomspace(c.min() / p.max() * 0.3, c.max() / p.min() * 300, 60)
for label, use_cost, use_prior in (("no prefill", False, False), ("+ cost head", True, False), ("+ cost head + prior", True, True)):
    kfit = fit_decay(use_prior)
    for k, tag in ((2.0, "k=2"), (kfit, f"k fitted={kfit:g}")):
        arms[f"bellman {label} [{tag}]"] = [bellman_maker(V, k, use_cost, use_prior) for V in Vs]
print(f"{T}: {len(tr)}/{len(cal)}/{len(te)} train/cal/test; cascade plans {len(arms['cascade'])}; bellman V grid {len(Vs)}", flush=True)
curves = {}
for name, makers in arms.items():
    curves[name] = [(per_problem(cal, mk), per_problem(te, mk)) for mk in makers]
    print(f"  simulated {name}", flush=True)


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
            tl = curves[name][lo[2]][1] if lo[2] is not None else np.zeros((len(te), 2))
            return (1 - w) * tl + w * curves[name][hi[2]][1]
    return None


results = {}
print(f"\nTEST accuracy / cost (cents) at operating points chosen on CALIBRATION; ratio vs cascade [95% CI]")
for t in TARGETS:
    base = pick("cascade", t)
    if base is None:
        print(f"  {t*100:.0f}%: cascade cannot reach it on calibration"); continue
    print(f"\n  target {t*100:.0f}% (cal):  cascade  test {base[:,0].mean()*100:.1f}% @ {base[:,1].mean():.4f}c")
    for name in arms:
        if name == "cascade":
            continue
        x = pick(name, t)
        if x is None:
            print(f"     {name:<42} unreachable on calibration"); continue
        ratio = x[:, 1].mean() / base[:, 1].mean(); dacc = (x[:, 0] - base[:, 0]).mean()
        bs = []
        for _ in range(2000):
            ii = rng.integers(0, len(te), len(te))
            bs.append((x[ii, 1].mean() / base[ii, 1].mean(), (x[ii, 0] - base[ii, 0]).mean()))
        bs = np.array(bs)
        print(f"     {name:<42} test {x[:,0].mean()*100:.1f}% @ {x[:,1].mean():.4f}c   cost ratio {ratio:.2f} "
              f"[{np.percentile(bs[:,0],2.5):.2f},{np.percentile(bs[:,0],97.5):.2f}]   acc diff {dacc*100:+.1f} "
              f"[{np.percentile(bs[:,1],2.5)*100:+.1f},{np.percentile(bs[:,1],97.5)*100:+.1f}]")
        results.setdefault(str(t), {})[name] = {"test_acc": x[:, 0].mean(), "test_cost": x[:, 1].mean(), "ratio": ratio,
                                                 "ratio_ci": [np.percentile(bs[:, 0], 2.5), np.percentile(bs[:, 0], 97.5)],
                                                 "acc_diff": dacc}
    results.setdefault(str(t), {})["cascade"] = {"test_acc": base[:, 0].mean(), "test_cost": base[:, 1].mean()}
Path(OUT).write_text(json.dumps(results, indent=1, default=float))
