"""Why does exact planning (Bellman) beat the fixed cascade when a one-step greedy rule does not?

Same beliefs and costs for all three (NO prefill: route train pass rates p_m, hyperbolic decay after f failures
p_m * 2/(2+f); route train-mean costs), free perfect verifier, LCB test split, 3 draw orderings:
  cascade   best fixed plan from analysis/priced_verification/lcb_best_fixed_cascade.json (draw counts per route,
            routes in price order, stop at first success)
  greedy    each step: draw argmax_m q_m*V - c_m; stop when <= 0          (ignores what a failure is worth)
  bellman   exact finite-horizon value (horizon 8): q*V - c + (1-q)*continuation
V is set per policy so test accuracy lands near the target (diagnostic, not a headline number).
Reported per policy: accuracy, cost, first-draw route shares, draws per problem by route, what happens after a
failure (resample same route / switch / stop), and where the money goes (problems finally solved vs not).
"""
import json, sys
from pathlib import Path
from collections import Counter
import numpy as np
from pipelinerl.swe.scripts.livecodebench.replay_bellman_verification import build_solver
from pipelinerl.swe.scripts.livecodebench.mdp_utils import load_split_manifest, split_indices

R = "/mnt/llmd/results/exps/aristides/reason"
T = sys.argv[1] if len(sys.argv) > 1 else "pool_v2_tensors_5rung"
TARGET = float(sys.argv[2]) if len(sys.argv) > 2 else 0.85
PR = {"oss20lo": 0.12, "oss20md": 0.57, "dsv4f": 0.111, "oss120md": 1.43, "oss120hi": 1.43}
d = np.load(f"{R}/{T}/tensors.npz", allow_pickle=True)
ids = [str(x) for x in d["problem_ids"]]; S = [str(x) for x in d["model_slots"]]
valid = d["valid"].astype(bool); truth = d["execution_outcome"].astype(bool)
real = np.stack([(d["prompt_tokens"][:, m] + d["completion_tokens"][:, m]) * PR[s] / 1e6 for m, s in enumerate(S)], 1)
tr, cal, te = split_indices(load_split_manifest(Path(f"{R}/{T}/split_manifest.json"), ids), ids)
p = np.array([truth[tr, m][valid[tr, m]].mean() for m in range(len(S))])
c = np.array([real[tr, m][valid[tr, m]].mean() for m in range(len(S))])
rng = np.random.default_rng(0)
orders = [[rng.permutation(valid.shape[2]) for _ in S] for _ in range(3)]


def episode(i, o, chooser):
    ptr = [0] * len(S); fails = [0] * len(S); log = []; spent = 0.0
    while True:
        m = chooser(fails, [int(valid[i, k].sum()) - fails[k] for k in range(len(S))], len(log))
        if m is None:
            return False, spent, log
        while ptr[m] < valid.shape[2] and not valid[i, m, orders[o][m][ptr[m]]]:
            ptr[m] += 1
        if ptr[m] >= valid.shape[2]:
            return False, spent, log
        k = orders[o][m][ptr[m]]; ptr[m] += 1
        spent += real[i, m, k]; log.append(m)
        if truth[i, m, k]:
            return True, spent, log
        fails[m] += 1


def make_greedy(V):
    def ch(fails, remaining, t):
        vals = [(p[m] * 2 / (2 + fails[m]) * V - c[m], m) for m in range(len(S)) if remaining[m] > 0]
        if not vals:
            return None
        best = max(vals)
        return best[1] if best[0] > 0 and t < 8 else None
    return ch


def make_bellman(i, V):
    caps = tuple(int(valid[i, m].sum()) for m in range(len(S)))
    solve, action = build_solver(p, c, V, 0.0, caps, 8, 2.0)
    def ch(fails, remaining, t):
        rem = tuple(max(0, caps[m] - fails[m]) for m in range(len(S)))
        val, route, _ = action(tuple(fails), rem, 8 - t)
        return route if route is not None and val > 0 else None
    return ch


def make_cascade(plan):
    seq = [m for m in sorted(range(len(S)), key=lambda m: c[m]) for _ in range(plan.get(S[m], 0))]
    def ch(fails, remaining, t):
        used = Counter(); j = 0
        for m in seq:                       # next planned draw whose route still has draws left
            used[m] += 1
            if used[m] > fails[m] and remaining[m] > 0:
                return m
        return None
    return ch


def run(policy_for_problem):
    res = []
    for i in te:
        for o in range(3):
            res.append((i,) + episode(i, o, policy_for_problem(i)))
    return res


def summarize(name, res):
    acc = np.mean([r[1] for r in res]); cost = np.mean([r[2] for r in res]) * 100
    first = Counter(S[r[3][0]] for r in res if r[3]); n = len(res)
    draws = Counter(S[m] for r in res for m in r[3])
    after = Counter()
    for r in res:
        for a, b in zip(r[3], r[3][1:]):
            after["resample" if a == b else "switch"] += 1
        if not r[1] and r[3]:
            after["give up after failure"] += 1
    unsolved = [r for r in res if not r[1]]
    waste = sum(r[2] for r in unsolved) / max(1e-12, sum(r[2] for r in res))
    print(f"\n{name}: accuracy {acc*100:.1f}%  cost {cost:.4f}c/problem  draws/problem {sum(draws.values())/n:.2f}")
    print("   first draw: " + ", ".join(f"{k} {v/n*100:.0f}%" for k, v in first.most_common()))
    print("   draws by route per problem: " + ", ".join(f"{k} {v/n:.2f}" for k, v in sorted(draws.items(), key=lambda x: -x[1])))
    tot = sum(after.values())
    print("   after a failed draw: " + ", ".join(f"{k} {v/tot*100:.0f}%" for k, v in after.items()))
    print(f"   unsolved problems: {len(unsolved)/n*100:.1f}% of episodes, absorbing {waste*100:.0f}% of all spend "
          f"({np.mean([r[2] for r in unsolved])*100 if unsolved else 0:.4f}c each vs {np.mean([r[2] for r in res if r[1]])*100:.4f}c per solved)")


print(f"{T}: routes {S}; train p {np.round(p,3)}; train cost (c) {np.round(c*100,4)}; target accuracy ~{TARGET*100:.0f}%")
casc = json.load(open(f"analysis/priced_verification/{'lcb' if 'pool_v2' in T else 'bcb'}_best_fixed_cascade.json"))
best = min(casc, key=lambda x: abs(x["test_acc"] / 100 - TARGET) if x["test_acc"] > 1 else abs(x["test_acc"] - TARGET))
print("cascade plan closest to target:", best["plan"])
summarize("CASCADE (fixed plan)", run(lambda i: make_cascade(best["plan"])))
Vs = np.geomspace(c.min() / p.max() * 0.5, c.max() / p.min() * 200, 40)
for name, mk in (("GREEDY one-step", lambda V: (lambda i: make_greedy(V))), ("BELLMAN exact", lambda V: (lambda i: make_bellman(i, V)))):
    cands = [(abs(np.mean([r[1] for r in run(mk(V))]) - TARGET), V) for V in Vs[::4]]
    V = min(cands)[1]
    summarize(f"{name} (V={V:.5f})", run(mk(V)))


# ---- matched-accuracy comparison: full curves, then the cheapest point reaching each target -------------------
if "--matched" in sys.argv:
    import itertools
    cheap_first = sorted(range(len(S)), key=lambda m: c[m])
    grid = {"oss20lo": range(0, 9), "oss20md": range(0, 3), "dsv4f": range(0, 5), "oss120md": range(0, 2), "oss120hi": range(0, 2)}
    pts = {"cascade": [], "greedy": [], "bellman": []}
    for counts in itertools.product(*[grid[s] for s in S]):
        if sum(counts) == 0:
            continue
        plan = dict(zip(S, counts)); res = run(lambda i: make_cascade(plan))
        pts["cascade"].append((np.mean([r[2] for r in res]) * 100, np.mean([r[1] for r in res]), plan, res))
    for V in Vs:
        for name, mk in (("greedy", lambda V: (lambda i: make_greedy(V))), ("bellman", lambda V: (lambda i: make_bellman(i, V)))):
            res = run(mk(V)); pts[name].append((np.mean([r[2] for r in res]) * 100, np.mean([r[1] for r in res]), V, res))
    print("\n=== cheapest point reaching each accuracy (test split; diagnostic, selection on test for every policy) ===")
    for tgt in (0.85, 0.90, 0.93, 0.95):
        print(f"\n--- target {tgt*100:.0f}% ---")
        for name in ("cascade", "greedy", "bellman"):
            ok = [x for x in pts[name] if x[1] >= tgt]
            if not ok:
                print(f"   {name}: unreachable"); continue
            best = min(ok, key=lambda x: x[0])
            summarize(f"{name.upper()}  [{best[2]}]", best[3])


# ---- same matched protocol, Bellman WITH per-problem information (learned cost head, optional prefill prior) ----
if "--perproblem" in sys.argv:
    LC = np.array([json.loads(l)["expected_costs"][:len(S)] for l in open(f"{R}/{T}/cost_preds.jsonl")], dtype=object)
    lc = {json.loads(l)["problem_id"]: json.loads(l)["expected_costs"][:len(S)] for l in open(f"{R}/{T}/cost_preds.jsonl")}
    lp = {json.loads(l)["problem_id"]: json.loads(l)["p_successes"][:len(S)] for l in open(f"{R}/{T}/content_preds.jsonl")}

    def make_bellman_pp(i, V, use_prior):
        caps = tuple(int(valid[i, m].sum()) for m in range(len(S)))
        prior = np.array(lp[ids[i]]) if use_prior else p
        solve, action = build_solver(prior, np.array(lc[ids[i]]), V, 0.0, caps, 8, 2.0)
        def ch(fails, remaining, t):
            rem = tuple(max(0, caps[m] - fails[m]) for m in range(len(S)))
            val, route, _ = action(tuple(fails), rem, 8 - t)
            return route if route is not None and val > 0 else None
        return ch
    import itertools
    grid = {"oss20lo": range(0, 9), "oss20md": range(0, 3), "dsv4f": range(0, 5), "oss120md": range(0, 2), "oss120hi": range(0, 2)}
    pts = {"cascade": [], "bellman no-prefill": [], "bellman + cost head": [], "bellman + cost head + prior": []}
    for counts in itertools.product(*[grid[s] for s in S]):
        if sum(counts):
            plan = dict(zip(S, counts)); res = run(lambda i: make_cascade(plan))
            pts["cascade"].append((np.mean([r[2] for r in res]) * 100, np.mean([r[1] for r in res]), plan, res))
    for V in Vs:
        for name, f in (("bellman no-prefill", lambda i: make_bellman(i, V)),
                        ("bellman + cost head", lambda i: make_bellman_pp(i, V, False)),
                        ("bellman + cost head + prior", lambda i: make_bellman_pp(i, V, True))):
            res = run(f); pts[name].append((np.mean([r[2] for r in res]) * 100, np.mean([r[1] for r in res]), V, res))
    print("\n=== per-problem information, matched (cheapest point reaching each accuracy; test-selected for all) ===")
    for tgt in (0.85, 0.90, 0.93, 0.95):
        row = []
        for name in pts:
            ok = [x for x in pts[name] if x[1] >= tgt]
            row.append(f"{name} {min(x[0] for x in ok):.4f}c" if ok else f"{name} unreachable")
        print(f"   {tgt*100:.0f}%:  " + " | ".join(row))
    for name in ("cascade", "bellman + cost head"):
        ok = [x for x in pts[name] if x[1] >= 0.93]
        if ok:
            best = min(ok, key=lambda x: x[0]); summarize(f"{name.upper()} at 93%  [{best[2]}]", best[3])
