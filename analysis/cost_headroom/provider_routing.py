"""Provider-routing pilot analysis (NEW_PATH 4.A.40). deepseek-v4-flash pinned to StreamLake / GMICloud / DigitalOcean on the same
problems (collect_provider_pilot.py), plus the four gpt-oss routes from the existing collections.
Pools: MMLU-Pro = the 2,000 pinned fresh problems (prefill heads frozen, from expanded_eval_20261001); APPS = the 382 TEST problems
(heads in apps_tensors). Prices = BILLED rates: pinned dsv4f calls use their own usage_cost; other calls are priced at the model's
effective billed $/M (least squares over the fresh collection's usage_cost), or the served provider's rate for unpinned dsv4f.
(1) Descriptives per pinned provider: accuracy, billed cost/call, median output; pairwise correct-agreement and log-length
    correlation across problems (is a provider 'the same model + an offset'?).
(2) Routing, all arms share the frozen prefill readouts; decision argmax_m V p_m - c_m over the arm's routes:
      unpinned        4 gpt-oss + dsv4f as collected (random provider), pooled dsv4f readout, average dsv4f billed price
      pinned-<P>      4 gpt-oss + dsv4f pinned to P (P's offsets + P's price)    -> best fixed provider
      endpoints       4 gpt-oss + all 3 pinned providers as separate routes (shared dsv4f readouts + per-provider offsets)
    Offsets (length ratio of means, success logit shift) are fitted on a FIT subset that is excluded from evaluation:
    APPS = pinned rows on TRAIN problems; MMLU-Pro = 300 random pinned problems (seed 0), evaluated on the other 1,700.
    Metric: cost saved vs 'unpinned' at matched accuracy (frontier over V, 12 targets over the shared band) and net-utility
    gain averaged over the V range; paired problem bootstrap (1,000).
Usage: python provider_routing.py
"""
import glob, json, sys, collections
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from carrot_compare import read_predictions
from decompose import R, hull, cost_at

PROV = ["StreamLake", "GMICloud", "DigitalOcean"]; GO = ["oss20lo", "oss20md", "oss120md", "oss120hi"]
PIL = R / "provider_pilot_20261002"; VS = np.geomspace(1e-4, 30, 120)
sig = lambda z: 1 / (1 + np.exp(-z)); lgt = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))


def billed_rates():
    by = collections.defaultdict(list); byp = collections.defaultdict(list)
    for f in glob.glob(f"{R}/math_expand_20261001/*/*.jsonl"):
        for l in open(f):
            try: r = json.loads(l)
            except Exception: continue
            if r.get("usage_cost") is None or r.get("finish_reason") == "error": continue
            m = "oss120" if "120" in r["route_label"] else ("oss20" if r["route_label"].startswith("oss20") else "dsv4f")
            x = (r["prompt_tokens"], r["completion_tokens"], r["usage_cost"]); by[m].append(x); byp[(m, r["provider"])].append(x)
    fit = lambda rows: np.linalg.lstsq(np.array([[a, b] for a, b, _ in rows], float), np.array([c for *_, c in rows]), rcond=None)[0]
    return {m: fit(v) for m, v in by.items()}, {k: fit(v) for k, v in byp.items() if len(v) >= 30}


RATE, PRATE = billed_rates()
rate_of = lambda route: RATE["oss120" if "120" in route else ("oss20" if route.startswith("oss20") else "dsv4f")]


def pinned(ds):
    out = {}
    for p in PROV:
        rows = {}
        for l in open(PIL / ds / f"dsv4f_pin_{p}.jsonl"):
            r = json.loads(l)
            if r.get("finish_reason") != "error" and r.get("completion_tokens"):
                rows[r["problem_id"]] = (float(r["resolved"]), float(r["completion_tokens"]), float(r["prompt_tokens"]), float(r["usage_cost"] or 0))
        out[p] = rows
    return out


def load(ds):
    """-> ids, q[n, routes], cost_realized, p_pred, len_pred, inp, routes, fit/eval index arrays, unpinned dsv4f provider"""
    pins = pinned(ds)
    if ds == "mmlupro":
        F = R / "expanded_eval_20261001" / "mmlupro"; t = np.load(F / "tensors.npz", allow_pickle=True)
        ids_all = list(map(str, t["problem_ids"])); slots = list(map(str, t["model_slots"]))
        P = read_predictions(F / "success_preds.jsonl", ids_all, "p_successes", len(slots))
        C = read_predictions(F / "paper_cost_preds.jsonl", ids_all, "expected_costs", len(slots))
        raw = {}
        for f in glob.glob(f"{R}/math_expand_20261001/mmlupro/*_d0.jsonl"):
            for l in open(f):
                r = json.loads(l)
                if r.get("finish_reason") != "error": raw[(r["problem_id"], r["route_label"])] = r
        cand = json.load(open(PIL / ds / "problem_ids.json"))
    else:                                                  # apps / cc (CodeContests, NEW_PATH 4.A.55): same layout
        F = R / ("apps_tensors" if ds == "apps" else "cc_tensors"); t = np.load(F / "tensors.npz", allow_pickle=True)
        ids_all = list(map(str, t["problem_ids"])); slots = list(map(str, t["model_slots"]))
        P = read_predictions(F / "content_preds.jsonl", ids_all, "p_successes", len(slots))
        C = read_predictions(F / "cost_preds_probe.jsonl", ids_all, "expected_costs", len(slots))
        raw = {}
        for f in glob.glob(f"{R}/apps_pool/*_d0.jsonl" if ds == "apps" else f"{R}/cc_pool/full/*_d0.jsonl"):
            for l in open(f):
                r = json.loads(l)
                if r.get("finish_reason") != "error": raw[(r["problem_id"], r["route_label"])] = r
        cand = ids_all
    idx = {p: i for i, p in enumerate(ids_all)}
    ids = [p for p in cand if all(p in pins[pp] for pp in PROV) and all((p, s) in raw for s in slots)]
    ii = np.array([idx[p] for p in ids])
    q = {s: np.array([float(bool(raw[(p, s)]["resolved"])) for p in ids]) for s in slots}
    L = {s: np.array([float(raw[(p, s)]["completion_tokens"]) for p in ids]) for s in slots}
    I = {s: np.array([float(raw[(p, s)]["prompt_tokens"]) for p in ids]) for s in slots}
    prov_unpinned = [raw[(p, "dsv4f")].get("provider") for p in ids]
    paid = {s: (I[s] * rate_of(s)[0] + L[s] * rate_of(s)[1]) for s in GO}
    pr_un = np.array([PRATE.get(("dsv4f", pv), RATE["dsv4f"]) @ np.array([a, b]) for pv, a, b in zip(prov_unpinned, I["dsv4f"], L["dsv4f"])])
    paid["dsv4f"] = np.array([raw[(p, "dsv4f")]["usage_cost"] if raw[(p, "dsv4f")].get("usage_cost") is not None else c for p, c in zip(ids, pr_un)])
    for pp in PROV:
        q[pp] = np.array([pins[pp][p][0] for p in ids]); L[pp] = np.array([pins[pp][p][1] for p in ids])
        I[pp] = np.array([pins[pp][p][2] for p in ids]); paid[pp] = np.array([pins[pp][p][3] for p in ids])
    from decompose import MK
    j = {s: slots.index(s) for s in slots}
    p_pred = {s: P[ii, j[s]] for s in slots}
    len_pred = {s: np.maximum((C[ii, j[s]] - I[s] * MK[s][0] / 1e6) / (MK[s][1] / 1e6), 1.0) for s in slots}   # back out tokens at assumed prices
    if ds == "mmlupro":
        rng = np.random.default_rng(0); perm = rng.permutation(len(ids)); fit, ev = perm[:300], perm[300:]
    else:
        sp = json.load(open(F / "split_manifest.json")); trset = set(map(str, sp["train_problem_ids"]) ) | set(map(str, sp["calibration_problem_ids"]))
        fit = np.array([k for k, p in enumerate(ids) if p in trset]); ev = np.array([k for k, p in enumerate(ids) if p in set(map(str, sp["test_problem_ids"]))])
    return ids, q, L, I, paid, p_pred, len_pred, fit, ev, prov_unpinned


def describe(ds, q, L, paid, ev):
    print(f"  per pinned provider (eval problems, n={len(ev)}):")
    for pp in PROV + ["dsv4f"]:
        tag = pp if pp != "dsv4f" else "unpinned dsv4f"
        print(f"     {tag:<15} acc {q[pp][ev].mean():.3f}  billed cost/call {paid[pp][ev].mean()*1e3:.3f} m$  median out {np.median(L[pp][ev]):7.0f}")
    for a in range(3):
        for b in range(a + 1, 3):
            A, B = PROV[a], PROV[b]
            agree = (q[A][ev] == q[B][ev]).mean(); exp = q[A][ev].mean() * q[B][ev].mean() + (1 - q[A][ev].mean()) * (1 - q[B][ev].mean())
            cor = np.corrcoef(np.log(L[A][ev]), np.log(L[B][ev]))[0, 1]
            print(f"     {A}-{B}: correct agreement {agree:.3f} (independent would give {exp:.3f}); log-length corr {cor:.2f}")


def route(arm_routes, p, c, q, paid, ev, V):
    U = np.stack([V * p[r][ev] - c[r][ev] for r in arm_routes], 1); m = U.argmax(1); k = np.arange(len(ev))
    return np.stack([q[r][ev] for r in arm_routes], 1)[k, m], np.stack([paid[r][ev] for r in arm_routes], 1)[k, m]


def frontier(arm_routes, p, c, q, paid, ev):
    return hull([tuple(np.array(route(arm_routes, p, c, q, paid, ev, V))[[1, 0]].mean(1)) for V in VS])


DSETS = [("mmlupro", "MMLU-Pro fresh (pinned 2,000)"), ("apps", "APPS test")] + ([("cc", "CodeContests test")] if (PIL / "cc").exists() else [])
res = {}
for ds, label in DSETS:
    ids, q, L, I, paid, p_pred, len_pred, fit, ev, prov_un = load(ds)
    print(f"\n===== {label}: {len(ids)} problems with all routes + 3 pinned providers; fit {len(fit)}, eval {len(ev)}")
    describe(ds, q, L, paid, ev)
    # predictions: shared dsv4f readouts; pinned endpoints = readout + offsets fitted on FIT problems
    p = {s: p_pred[s] for s in GO}; c = {s: I[s] * rate_of(s)[0] + len_pred[s] * rate_of(s)[1] for s in GO}
    p["dsv4f"] = p_pred["dsv4f"]; c["dsv4f"] = I["dsv4f"] * RATE["dsv4f"][0] + len_pred["dsv4f"] * RATE["dsv4f"][1]
    offs = {}
    for pp in PROV:
        a = L[pp][fit].sum() / len_pred["dsv4f"][fit].sum()
        lo, hi = -6.0, 6.0
        for _ in range(60):
            b = (lo + hi) / 2; lo, hi = (b, hi) if sig(lgt(p_pred["dsv4f"][fit]) + b).mean() < q[pp][fit].mean() else (lo, b)
        b = (lo + hi) / 2; rate = PRATE.get(("dsv4f", pp), RATE["dsv4f"]); offs[pp] = (a, b)
        p[pp] = sig(lgt(p_pred["dsv4f"]) + b); c[pp] = I[pp] * rate[0] + len_pred["dsv4f"] * a * rate[1]
    print("  offsets fitted on FIT problems: " + ", ".join(f"{pp} length x{a:.2f} logit {b:+.2f}" for pp, (a, b) in offs.items()))
    arms = {"unpinned": GO + ["dsv4f"], **{f"pinned-{pp}": GO + [pp] for pp in PROV}, "endpoints": GO + PROV}
    rng = np.random.default_rng(1); BS = [rng.integers(0, len(ev), len(ev)) for _ in range(1000)]

    def saved(arm, sub):
        e = ev[sub]; H, H0 = frontier(arms[arm], p, c, q, paid, e), frontier(arms["unpinned"], p, c, q, paid, e)
        lo, hi = max(H[0][1], H0[0][1]), min(H[-1][1], H0[-1][1]); T = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 12)
        return 1 - float(np.exp(np.nanmean(np.log([cost_at(H, x) / cost_at(H0, x) for x in T]))))

    def util(arm, sub):
        e = ev[sub]; Vg = np.geomspace(1e-3, 3e-2, 15)                         # $ per correct answer, the band where accuracy moves
        d = np.mean([(V * route(arms[arm], p, c, q, paid, e, V)[0] - route(arms[arm], p, c, q, paid, e, V)[1])
                     - (V * route(arms["unpinned"], p, c, q, paid, e, V)[0] - route(arms["unpinned"], p, c, q, paid, e, V)[1]) for V in Vg], 0)
        base = np.mean([route(arms["unpinned"], p, c, q, paid, e, V)[1].mean() for V in Vg]); return 100 * d.mean() / base
    out = {}
    allk = np.arange(len(ev))
    for arm in arms:
        Hm = frontier(arms[arm], p, c, q, paid, ev)
        if arm == "unpinned":
            print(f"  {arm:<20} max acc {Hm[-1][1]*100:.1f}%"); continue
        g = saved(arm, allk); u = util(arm, allk)
        gb = [saved(arm, b) for b in BS[:300]]; ub = [util(arm, b) for b in BS[:300]]
        out[arm] = dict(saved=g, saved_ci=list(np.nanpercentile(gb, [2.5, 97.5])), util=u, util_ci=list(np.nanpercentile(ub, [2.5, 97.5])), max_acc=Hm[-1][1])
        print(f"  {arm:<20} max acc {Hm[-1][1]*100:.1f}%  cost saved vs unpinned {g*100:+6.1f}% [{np.nanpercentile(gb,2.5)*100:+.1f}, {np.nanpercentile(gb,97.5)*100:+.1f}]"
              f"  net-utility gain {u:+6.1f}% [{np.nanpercentile(ub,2.5):+.1f}, {np.nanpercentile(ub,97.5):+.1f}]")
    # how often does the endpoints router pick each provider, at a mid V?
    V = 0.01; U = np.stack([V * p[r][ev] - c[r][ev] for r in arms["endpoints"]], 1); mm = U.argmax(1)
    print("  endpoints router choice shares at V=$0.01/correct: " + ", ".join(f"{r} {np.mean(mm == k)*100:.0f}%" for k, r in enumerate(arms["endpoints"])))
    res[label] = out
json.dump(res, open(Path(__file__).parent / "provider_routing.json", "w"), indent=1, default=float)
