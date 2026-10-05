"""Price-leakage check (NEW_PATH 4.A.49): the billed $/token rates used to PRICE PREDICTIONS were fitted on the same fresh calls we
evaluate. Here they are cross-fitted: fresh problems are split in two halves by a hash of the problem id; each problem's predictions
are priced with rates fitted only on the OTHER half's calls (all datasets). Realized cost is still each call's usage_cost.
Compared with in-sample rates (the paper) and with the list prices we originally assumed (decompose.MK).
Arms (as fresh_baselines.py, same success predictions): ours, median, mean, cost-from-success, difficulty bins.
Reported: cost saved by ours vs each arm at matched accuracy on the fresh problems, paired problem bootstrap (300).
Usage: python heldout_rates.py
"""
import collections, glob, hashlib, json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from carrot_compare import POOLS, read_predictions
from decompose import MK, R, hull, cost_at

VALUES = np.geomspace(1e-7, 1, 300)
fam = lambda s: "oss120" if "120" in s else ("oss20" if s.startswith("oss20") else "dsv4f")
half = lambda pid: int(hashlib.md5(pid.encode()).hexdigest(), 16) % 2

rows = collections.defaultdict(list)                                  # (half, model) -> (in, out, usd)
for f in glob.glob(f"{R}/math_expand_20261001/*/*.jsonl"):
    for l in open(f):
        try: r = json.loads(l)
        except Exception: continue
        if r.get("usage_cost") is None or r.get("finish_reason") == "error": continue
        rows[(half(r["problem_id"]), fam(r["route_label"]))].append((r["prompt_tokens"], r["completion_tokens"], r["usage_cost"]))
fit = lambda rr: np.linalg.lstsq(np.array([[a, b] for a, b, _ in rr], float), np.array([c for *_, c in rr]), rcond=None)[0]
RATE_H = {h: {m: fit(rows[(h, m)]) for m in ("oss20", "dsv4f", "oss120")} for h in (0, 1)}
RATE_ALL = {m: fit(rows[(0, m)] + rows[(1, m)]) for m in ("oss20", "dsv4f", "oss120")}
print("output $/M  in-sample | half0 | half1 | list:  " + "  ".join(
    f"{m} {RATE_ALL[m][1]*1e6:.4f} | {RATE_H[0][m][1]*1e6:.4f} | {RATE_H[1][m][1]*1e6:.4f} | {MK[m][1]:.3f}" for m in RATE_ALL))
out = {}
for ds in ("omni500", "mmlupro"):
    label = "MMLU-Pro" if ds == "mmlupro" else "Omni"; name, cost_file = POOLS[label]; old = R / name
    F = R / "expanded_eval_20261001" / ds; t = np.load(F / "tensors.npz", allow_pickle=True)
    ids, slots = list(map(str, t["problem_ids"])), list(map(str, t["model_slots"])); idx = {p: i for i, p in enumerate(ids)}; M = len(slots)
    sp = json.loads((old / "split_manifest.json").read_text()); tr = np.array([idx[str(p)] for p in sp["train_problem_ids"]])
    n_old = len(np.load(old / "tensors.npz", allow_pickle=True)["problem_ids"]); fresh = np.arange(n_old, len(ids))
    v = t["valid"].astype(bool); cnt = np.maximum(v.sum(2), 1)
    q = np.where(v, t["final_outcome"], 0).sum(2) / cnt; L = np.where(v, t["completion_tokens"], 0).sum(2) / cnt; I = np.where(v, t["prompt_tokens"], 0).sum(2) / cnt
    paid = np.full(L.shape, np.nan)
    for f in glob.glob(f"{R}/math_expand_20261001/{ds}/*_d0.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if r.get("finish_reason") != "error" and r.get("usage_cost") is not None and r["problem_id"] in idx and r["route_label"] in slots:
                paid[idx[r["problem_id"]], slots.index(r["route_label"])] = r["usage_cost"]
    fresh = fresh[(v[fresh].sum(2) > 0).all(1) & np.isfinite(paid[fresh]).all(1)]
    learned = read_predictions(F / "paper_cost_preds.jsonl", ids, "expected_costs", M)
    learned[:n_old] = read_predictions(old / cost_file, ids[:n_old], "expected_costs", M)
    P = np.clip(read_predictions(F / "success_preds.jsonl", ids, "p_successes", M), 1e-4, 1 - 1e-4); Lg = np.log(P / (1 - P))
    asg = np.array([[MK[s][0] / 1e6, MK[s][1] / 1e6] for s in slots])
    tok = {"ours": np.maximum((learned - I * asg[:, 0]) / asg[:, 1], 1)}
    tok["median"] = np.repeat([[np.median(t["completion_tokens"][tr, k][v[tr, k]]) for k in range(M)]], len(ids), 0)
    tok["mean"] = np.repeat([L[tr].mean(0)], len(ids), 0)
    Y = np.log(np.maximum(L, 1)); FS = np.c_[Lg, Lg ** 2]; fs = []
    for k in range(M):
        m_ = make_pipeline(StandardScaler(), RidgeCV(alphas=np.geomspace(1e-3, 1e4, 15))).fit(FS[tr], Y[tr, k]); yh = m_.predict(FS)
        fs.append(np.exp(yh) * np.mean(np.exp(Y[tr, k] - yh[tr])))
    tok["cost-from-success"] = np.stack(fs, 1)
    dbar = Lg.mean(1); e = np.quantile(dbar[tr], np.linspace(0, 1, 11)[1:-1]); b = np.searchsorted(e, dbar)
    tok["difficulty bins"] = np.stack([np.array([L[tr][b[tr] == j, k].mean() if (b[tr] == j).any() else L[tr, k].mean() for j in range(10)])[b] for k in range(M)], 1)
    for a in ("cost-from-success", "difficulty bins"):
        tok[a] = tok[a] * (L[tr].mean(0) / tok[a][tr].mean(0))
    hp = np.array([half(p) for p in ids])

    def priced(scheme):                                               # per-problem (in, out) $/token per route
        if scheme == "list":
            r_ = np.repeat(asg[None], len(ids), 0)
        elif scheme == "in-sample":
            r_ = np.repeat(np.array([RATE_ALL[fam(s)] for s in slots])[None], len(ids), 0)
        else:                                                         # cross-fitted: rates from the other half's calls
            r_ = np.stack([np.array([RATE_H[1 - h][fam(s)] for s in slots]) for h in hp])
        return {a: I * r_[..., 0] + x * r_[..., 1] for a, x in tok.items()}

    def front(c, ii):
        pts = []
        for V in VALUES:
            m = (V * P[ii] - c[ii]).argmax(1); pts.append((paid[ii][np.arange(len(ii)), m].mean(), q[ii][np.arange(len(ii)), m].mean()))
        return hull(pts)

    def saved(ca, cb, ii):
        Ha, Hb = front(ca, ii), front(cb, ii); lo, hi = max(Ha[0][1], Hb[0][1]), min(Ha[-1][1], Hb[-1][1])
        T = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 12)
        return 1 - float(np.exp(np.nanmean(np.log([cost_at(Ha, x) / cost_at(Hb, x) for x in T]))))
    rng = np.random.default_rng(0); BS = [fresh[rng.integers(0, len(fresh), len(fresh))] for _ in range(300)]
    print(f"\n===== {label} fresh n={len(fresh)}: cost saved by ours vs each arm at matched accuracy [95% CI]")
    res = {}
    for scheme in ("in-sample", "cross-fitted", "list"):
        C = priced(scheme); res[scheme] = {}
        for a in C:
            if a == "ours": continue
            g = saved(C["ours"], C[a], fresh); bs = [saved(C["ours"], C[a], bb) for bb in BS]
            res[scheme][a] = [g, *np.percentile(bs, [2.5, 97.5])]
            print(f"  {scheme:<13} vs {a:<18} {g*100:+6.1f}% [{np.percentile(bs,2.5)*100:+.1f}, {np.percentile(bs,97.5)*100:+.1f}]", flush=True)
    out[label] = res
json.dump(out, open(Path(__file__).parent / "heldout_rates.json", "w"), indent=1, default=float)
