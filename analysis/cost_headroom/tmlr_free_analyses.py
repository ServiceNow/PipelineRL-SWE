"""TMLR analyses on the pinned test sets (NEW_PATH 4.A.63), all free. Same loading and frontier arithmetic as headroom_routes.py.
  savings    WHERE the saving comes from: at a mid-band matched accuracy, each arm's deterministic policy (V closest to the target);
             route shares, reroute matrix (median route -> our route), and spend by decile of predicted length / of difficulty
  learning   label learning curve: cost readout (paper ridge recipe) refit on n training problems (25..all, 5 seeds), and with ONE
             draw per training problem instead of all; success fixed (paper readout); saving vs median at matched accuracy
  prices     counterfactual price ladders: every route's price scaled by (its billed $/token / cheapest)^(s-1), s in a grid (s=1 billed,
             s<1 flatter, s>1 steeper); headroom (oracle) and ours vs median at each
  effort     model vs effort choice: ours vs median and headroom in sub-pools (20b effort only, 120b effort only, one effort per model,
             full pool)
Usage: REASON_ROOT=.../reason_pinned python tmlr_free_analyses.py LCB|Omni|MMLU-Pro
"""
import glob, json, os, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).parent))
from carrot_compare import POOLS, read_predictions
from decompose import MK, R, hull, cost_at
from billed import RATE
from baseline_cost_heads import rich

VALUES = np.geomspace(1e-7, 1, 300)
rate_of = lambda s: RATE["oss120" if "120" in s else ("oss20" if s.startswith("oss20") else "dsv4f")]
POOL = sys.argv[1]
REAL = "/mnt/llmd/results/exps/aristides/reason"
SINGLE = {   # pools evaluated on their own test split (tokens x billed rates), not on the fresh test sets
    "LCB": ("pool_v2_tensors_5rung", "cost_preds_probe.jsonl", f"{REAL}/pv2_scout_prefill_1756715297/scout.npz"),
    "SuperGPQA": ("supergpqa_tensors", "cost_preds_probe_instruct.jsonl", f"{REAL}/supergpqa_probe/instruct.npz"),
    "BBEH": ("bbeh_tensors", "cost_preds_probe_instruct.jsonl", f"{REAL}/bbeh_probe/instruct.npz")}
if POOL in SINGLE:
    name, cost_file, feat = SINGLE[POOL]; old = F = R / name; feat = Path(feat)
else:
    name, cost_file = POOLS[POOL]; old = R / name; ds = "mmlupro" if POOL == "MMLU-Pro" else "omni500"; F = R / "expanded_eval_20261001" / ds
    feat = F / "prefill_combined.npz"
t = np.load(F / "tensors.npz", allow_pickle=True)
ids, slots = list(map(str, t["problem_ids"])), list(map(str, t["model_slots"])); idx = {p: i for i, p in enumerate(ids)}; M = len(slots)
sp = json.loads((old / "split_manifest.json").read_text()); tr = np.array([idx[str(p)] for p in sp["train_problem_ids"]])
n_old = len(np.load(old / "tensors.npz", allow_pickle=True)["problem_ids"])
ev = np.array([idx[str(p)] for p in sp["test_problem_ids"]]) if POOL in SINGLE else np.arange(n_old, len(ids))
v = t["valid"].astype(bool); cnt = np.maximum(v.sum(2), 1)
q = np.where(v, t["final_outcome"], 0).sum(2) / cnt; L = np.maximum(np.where(v, t["completion_tokens"], 0).sum(2) / cnt, 1)
I = np.where(v, t["prompt_tokens"], 0).sum(2) / cnt
rates = np.array([rate_of(s) for s in slots]); toks = I * rates[:, 0] + L * rates[:, 1]; paid = toks.copy()
if POOL not in SINGLE:
    for f in glob.glob(f"{R}/math_expand_20261001/{ds}/*_d0.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if r.get("finish_reason") != "error" and r.get("usage_cost") is not None and r["problem_id"] in idx and r["route_label"] in slots:
                paid[idx[r["problem_id"]], slots.index(r["route_label"])] = r["usage_cost"]
ev = ev[(v[ev].sum(2) > 0).all(1)]
if POOL in SINGLE:
    learned = read_predictions(old / cost_file, ids, "expected_costs", M); P = read_predictions(old / "content_preds.jsonl", ids, "p_successes", M)
else:
    learned = read_predictions(F / "paper_cost_preds.jsonl", ids, "expected_costs", M); learned[:n_old] = read_predictions(old / cost_file, ids[:n_old], "expected_costs", M)
    P = read_predictions(F / "success_preds.jsonl", ids, "p_successes", M); P[:n_old] = read_predictions(old / "content_preds.jsonl", ids[:n_old], "p_successes", M)
P = np.clip(P, 1e-4, 1 - 1e-4)
asg = np.array([[MK[s][0] / 1e6, MK[s][1] / 1e6] for s in slots]); tok = np.maximum((learned - I * asg[:, 0]) / asg[:, 1], 1)
med = np.array([np.median(t["completion_tokens"][tr, k][v[tr, k]]) for k in range(M)])


def costs(tk, rt):
    return I * rt[:, 0] + tk * rt[:, 1]


def front(Pm, C, pd, ii, cols):
    pts = []
    for V in VALUES:
        m = (V * Pm[ii][:, cols] - C[ii][:, cols]).argmax(1); k = np.arange(len(ii)); pts.append((pd[ii][:, cols][k, m].mean(), q[ii][:, cols][k, m].mean()))
    return hull(pts)


def saved(Ca, Cb, pd, ii, cols=None):
    cols = list(range(M)) if cols is None else cols
    Ha, Hb = front(P, Ca, pd, ii, cols), front(P, Cb, pd, ii, cols); lo, hi = max(Ha[0][1], Hb[0][1]), min(Ha[-1][1], Hb[-1][1])
    if hi <= lo:
        return np.nan
    T = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 12)
    return 1 - float(np.exp(np.nanmean(np.log([cost_at(Ha, x) / cost_at(Hb, x) for x in T]))))


C_ours, C_med = costs(tok, rates), costs(med[None].repeat(len(ids), 0), rates)
out = {"pool": POOL, "n_eval": int(len(ev))}
print(f"===== {POOL} test n={len(ev)}: ours vs median {saved(C_ours, C_med, paid, ev)*100:+.1f}%  headroom {saved(paid, C_med, paid, ev)*100:+.1f}%", flush=True)

# ---------------------------------------------------------------- 1. where the saving comes from
def policy(C, target):
    best = None
    for V in VALUES:
        m = (V * P[ev] - C[ev]).argmax(1); acc = q[ev][np.arange(len(ev)), m].mean()
        if best is None or abs(acc - target) < abs(best[0] - target):
            best = (acc, m)
    return best
Ho, Hm = front(P, C_ours, paid, ev, list(range(M))), front(P, C_med, paid, ev, list(range(M)))
lo, hi = max(Ho[0][1], Hm[0][1]), min(Ho[-1][1], Hm[-1][1]); res1 = {}
for frac in (.25, .5, .75):
    target = lo + frac * (hi - lo); (ao, mo), (am, mm) = policy(C_ours, target), policy(C_med, target)
    k = np.arange(len(ev)); so, sm = paid[ev][k, mo], paid[ev][k, mm]
    share = lambda m: {s: float((m == j).mean()) for j, s in enumerate(slots)}
    rer = {f"{slots[a]}->{slots[b]}": float(((mm == a) & (mo == b)).mean()) for a in range(M) for b in range(M) if a != b and ((mm == a) & (mo == b)).mean() > .01}
    lvl = np.log(tok[ev]).mean(1); dif = -np.log(P[ev] / (1 - P[ev])).mean(1)
    dec = lambda x: np.minimum((np.argsort(np.argsort(x)) * 10) // len(x), 9)
    by = {}
    for nm, x in (("predicted_length", lvl), ("difficulty", dif)):
        d = dec(x); by[nm] = [dict(decile=int(b), spend_ours=float(so[d == b].mean()), spend_median=float(sm[d == b].mean()),
                                   share_of_saving=float((sm[d == b] - so[d == b]).sum() / max((sm - so).sum(), 1e-12))) for b in range(10)]
    res1[f"{frac:.2f}"] = dict(target=float(target), acc_ours=float(ao), acc_median=float(am), spend_ours=float(so.mean()), spend_median=float(sm.mean()),
                               share_ours=share(mo), share_median=share(mm), reroutes=rer, by_decile=by, changed=float((mo != mm).mean()))
    top = sorted(rer.items(), key=lambda kv: -kv[1])[:4]
    print(f"[savings] {frac:.2f} of band: acc ours {ao:.3f} / median {am:.3f}; spend {so.mean()*1e3:.3f} vs {sm.mean()*1e3:.3f} m$; "
          f"{(mo != mm).mean()*100:.0f}% of problems rerouted; top reroutes {top}", flush=True)
    print("          saving share by predicted-length decile (short->long): " + " ".join(f"{b['share_of_saving']*100:.0f}" for b in by["predicted_length"]), flush=True)
    print("          saving share by difficulty decile (easy->hard):        " + " ".join(f"{b['share_of_saving']*100:.0f}" for b in by["difficulty"]), flush=True)
out["savings"] = res1

# ---------------------------------------------------------------- 2. label learning curve
z = np.load(feat, allow_pickle=True)
X = rich(feat, ids); sc_all = None
Y = np.log(L)
def fit_tok(train, Yt):
    sc = StandardScaler().fit(X[train]); Xs = sc.transform(X); tk = np.zeros((len(ids), M))
    for k in range(M):
        m = RidgeCV(alphas=np.geomspace(1e1, 1e7, 13)).fit(Xs[train], Yt[train, k]); yh = m.predict(Xs)
        s_ = np.mean(np.exp(Yt[train, k] - yh[train])); e = np.exp(yh) * s_; tk[:, k] = e * np.exp(Yt[train, k]).mean() / e[train].mean()
    return tk
res2 = {}; rng = np.random.default_rng(0)
for n in [25, 50, 100, 200, len(tr)]:
    reps = 1 if n == len(tr) else 5; g = []
    for _ in range(reps):
        sub = tr if n == len(tr) else rng.choice(tr, n, replace=False)
        g.append(saved(costs(fit_tok(sub, Y), rates), C_med, paid, ev))
    res2[str(n)] = [float(np.mean(g)), float(np.std(g))]
    print(f"[learning] n_train={n:4d}: ours vs median {np.mean(g)*100:+.1f}% (sd {np.std(g)*100:.1f}, {reps} seeds)", flush=True)
Y1 = np.log(np.maximum(np.where(v[:, :, 0], t["completion_tokens"][:, :, 0], L), 1))   # first draw only (falls back to mean if missing)
g1 = saved(costs(fit_tok(tr, Y1), rates), C_med, paid, ev); res2["one_draw"] = float(g1)
print(f"[learning] one draw per training problem: {g1*100:+.1f}% (max draws per route {v.sum(2).max(0).tolist()})", flush=True)
out["learning"] = res2

# ---------------------------------------------------------------- 3. counterfactual price ladders
res3 = {}; cheapest = rates[:, 1].min()
for s_ in (0.0, 0.5, 1.0, 1.5, 2.0):
    mult = (rates[:, 1] / cheapest) ** (s_ - 1.0); rt = rates * mult[:, None]
    pd = costs(L, rt) if POOL in SINGLE else paid * mult[None]                 # realized cost rescaled by the same factor
    g, h = saved(costs(tok, rt), costs(med[None].repeat(len(ids), 0), rt), pd, ev), saved(pd, costs(med[None].repeat(len(ids), 0), rt), pd, ev)
    gap = float(rt[:, 1].max() / rt[:, 1].min()); res3[str(s_)] = dict(price_gap=gap, ours=g, headroom=h)
    print(f"[prices] steepness {s_:.1f} (out-price gap {gap:.1f}x): ours vs median {g*100:+.1f}%  headroom {h*100:+.1f}%", flush=True)
out["prices"] = res3

# ---------------------------------------------------------------- 4. effort vs model
res4 = {}; j = {s: i for i, s in enumerate(slots)}
for nm, cols in (("20b effort only", ["oss20lo", "oss20md"]), ("120b effort only", ["oss120md", "oss120hi"]),
                 ("one effort per model", ["oss20md", "dsv4f", "oss120md"]), ("full pool", slots)):
    c = [j[s] for s in cols if s in j]
    g, h = saved(C_ours, C_med, paid, ev, c), saved(paid, C_med, paid, ev, c); res4[nm] = dict(routes=cols, ours=g, headroom=h)
    print(f"[effort] {nm:<22}: ours vs median {g*100:+.1f}%  headroom {h*100:+.1f}%", flush=True)
out["effort"] = res4
json.dump(out, open(Path(__file__).parent / f"tmlr_free_{POOL.replace('-', '').lower()}{os.environ.get('RESULT_TAG', '')}.json", "w"), indent=1, default=float)
