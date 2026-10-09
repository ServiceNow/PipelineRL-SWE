"""Is ZeroRouter's CodeContests edge (9.3% [1.1, 20.0] vs median; ours 6.3 [-0.2, 17.0]; ours - it -3.3 n.s.; NEW_PATH 4.A.61) its
SUCCESS model or its PRICING, and is it robust to its configuration? Same loading, prices and protocol as fresh_baselines.py --pool CC
(pinned, billed, test split). For every D in {1, 2, 5} x K in {5, 10, 20} x stage-1 seed in {0, 1, 2}, cost saved vs median pricing
(with OUR success readouts) at matched accuracy on test for:
  zr          ZeroRouter's success model + its bin pricing (the suite arm)
  zr_succ     ZeroRouter's success model + OUR cost readout       (is the edge its success side?)
  zr_price    OUR success readouts + ZeroRouter's bin pricing     (is it its pricing?)
plus the suite's calibration-chosen configuration and paired bootstraps (300) for ours - zr and ours - zr_succ at that configuration.
Usage: REASON_ROOT=.../reason_pinned OUT_DIR=... python zr_cc_check.py [POOL]   (default CC)
"""
import json, os, sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
sys.path.insert(0, str(Path(__file__).parent))
from carrot_compare import read_predictions
from decompose import MK, R, hull, cost_at
from billed import RATE
from zr_dimsweep import fit_stage1
src = open(Path(__file__).parent / "fresh_baselines.py").read(); ns = {}
exec(src[src.index("ORIG ="):src.index("POOL =")], ns); ORIG = ns["ORIG"]

POOL = sys.argv[1] if len(sys.argv) > 1 else "CC"; VALUES = np.geomspace(1e-7, 1, 300); sig = lambda z: 1 / (1 + np.exp(-z))
rate_of = lambda s: RATE["oss120" if "120" in s else ("oss20" if s.startswith("oss20") else "dsv4f")]
name, cost_file, feat = ORIG[POOL]; F = R / name; t = np.load(F / "tensors.npz", allow_pickle=True)
ids, slots = list(map(str, t["problem_ids"])), list(map(str, t["model_slots"])); idx = {p: i for i, p in enumerate(ids)}; M = len(slots)
sp = json.loads((F / "split_manifest.json").read_text())
tr, ca, te = [np.array([idx[str(p)] for p in sp[k + "_problem_ids"]]) for k in ("train", "calibration", "test")]
v = t["valid"].astype(bool); cnt = np.maximum(v.sum(2), 1); succ = np.where(v, t["final_outcome"], 0).sum(2); n = v.sum(2)
q = succ / cnt; L = np.where(v, t["completion_tokens"], 0).sum(2) / cnt; I = np.where(v, t["prompt_tokens"], 0).sum(2) / cnt
rates = np.array([rate_of(s) for s in slots]); paid = I * rates[:, 0] + L * rates[:, 1]
te = te[(v[te].sum(2) > 0).all(1)]; ca = ca[(v[ca].sum(2) > 0).all(1)]
learned = read_predictions(F / cost_file, ids, "expected_costs", M); P = np.clip(read_predictions(F / "content_preds.jsonl", ids, "p_successes", M), 1e-4, 1 - 1e-4)
asg = np.array([[MK[s][0] / 1e6, MK[s][1] / 1e6] for s in slots]); tok = np.maximum((learned - I * asg[:, 0]) / asg[:, 1], 1)
C_ours = I * rates[:, 0] + tok * rates[:, 1]
C_med = I * rates[:, 0] + np.array([np.median(t["completion_tokens"][tr, k][v[tr, k]]) for k in range(M)])[None] * rates[:, 1]
z = np.load(feat, allow_pickle=True); zid = {str(p): i for i, p in enumerate(z["problem_ids"])}
X = np.concatenate([z[k].reshape(len(z[k]), -1) for k in ("mean", "last")], 1)[[zid[p] for p in ids]].astype(np.float32); del z
Z = PCA(256, random_state=0).fit(X[tr]).transform(X); Z /= Z[tr].std(0) + 1e-6; del X


def front(p_, c_, ii):
    m = (VALUES[:, None, None] * p_[ii][None] - c_[ii][None]).argmax(2); r = np.arange(len(ii))[None]
    return hull(list(zip(paid[ii][r, m].mean(1), q[ii][r, m].mean(1))))


def saved(pa, ca_, pb, cb, ii):
    Ha, Hb = front(pa, ca_, ii), front(pb, cb, ii); lo, hi = max(Ha[0][1], Hb[0][1]), min(Ha[-1][1], Hb[-1][1])
    if hi <= lo:
        return np.nan
    T = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 12)
    return 1 - float(np.exp(np.nanmean(np.log([cost_at(Ha, x) / cost_at(Hb, x) for x in T]))))


ours = saved(P, C_ours, P, C_med, te); res = {"pool": POOL, "n_test": int(len(te)), "ours_vs_median": ours, "configs": []}
print(f"===== ZeroRouter check {POOL}: test n={len(te)}, cal n={len(ca)}; ours vs median {ours*100:+.1f}%", flush=True)
best = None
for D in (1, 2, 5):
    for seed in (0, 1, 2):
        la, bb, th = fit_stage1(succ[tr], n[tr], D, seed); pr = RidgeCV(alphas=np.geomspace(1, 1e5, 11)).fit(Z[tr], np.c_[la, bb]).predict(Z)
        A = np.exp(pr[:, :D]); B = pr[:, D:]; A[tr] = np.exp(la); B[tr] = bb; Pz = np.clip(sig((A[:, None, :] * (th[None] - B[:, None, :])).sum(-1)), 1e-4, 1 - 1e-4)
        s = (A * B).sum(1)
        for K in (5, 10, 20):
            ee = np.quantile(s[tr], np.linspace(0, 1, K + 1)[1:-1]); bn = np.searchsorted(ee, s)
            tab = np.array([[L[tr][bn[tr] == j, k].mean() if (bn[tr] == j).any() else L[tr, k].mean() for j in range(K)] for k in range(M)])
            Cz = I * rates[:, 0] + tab.T[bn] * rates[:, 1]
            row = dict(D=D, seed=seed, K=K, zr=saved(Pz, Cz, P, C_med, te), zr_succ=saved(Pz, C_ours, P, C_med, te), zr_price=saved(P, Cz, P, C_med, te),
                       cal_zr=saved(Pz, Cz, P, C_med, ca))
            res["configs"].append(row)
            print(f"  D={D} seed={seed} K={K:<2}  zr {row['zr']*100:+6.1f}  zr_succ+our cost {row['zr_succ']*100:+6.1f}  our succ+zr price "
                  f"{row['zr_price']*100:+6.1f}   (calibration: zr {row['cal_zr']*100:+6.1f})", flush=True)
            if seed == 0 and (best is None or row["cal_zr"] > best[0]):         # the suite's rule: seed 0, D x K chosen on calibration
                best = (row["cal_zr"], D, K, Pz, Cz)
cf = np.array([[r["zr"], r["zr_succ"], r["zr_price"]] for r in res["configs"]]) * 100
res["summary"] = {k: dict(mean=float(np.nanmean(cf[:, j])), min=float(np.nanmin(cf[:, j])), max=float(np.nanmax(cf[:, j])),
                         share_above_ours=float(np.mean(cf[:, j] > ours * 100))) for j, k in enumerate(("zr", "zr_succ", "zr_price"))}
_, D, K, Pz, Cz = best; rng = np.random.default_rng(0); BS = [te[rng.integers(0, len(te), len(te))] for _ in range(300)]
d1 = np.array([saved(P, C_ours, Pz, Cz, b) for b in BS]) * 100; d2 = np.array([saved(P, C_ours, Pz, C_ours, b) for b in BS]) * 100
res["cal_chosen"] = dict(D=D, K=K, ours_minus_zr=[float(saved(P, C_ours, Pz, Cz, te) * 100), *np.nanpercentile(d1, [2.5, 97.5]).tolist()],
                         ours_minus_zr_succ_our_cost=[float(saved(P, C_ours, Pz, C_ours, te) * 100), *np.nanpercentile(d2, [2.5, 97.5]).tolist()])
print("  over all 27 configurations (test): " + "  ".join(f"{k} mean {x['mean']:+.1f} [{x['min']:+.1f}, {x['max']:+.1f}], above ours in {x['share_above_ours']:.0%}"
                                                    for k, x in res["summary"].items()), flush=True)
c = res["cal_chosen"]
print(f"  calibration-chosen D={D} K={K}: ours - zr {c['ours_minus_zr'][0]:+.1f} [{c['ours_minus_zr'][1]:+.1f}, {c['ours_minus_zr'][2]:+.1f}]; "
      f"ours - (zr success + our cost) {c['ours_minus_zr_succ_our_cost'][0]:+.1f} [{c['ours_minus_zr_succ_our_cost'][1]:+.1f}, {c['ours_minus_zr_succ_our_cost'][2]:+.1f}]", flush=True)
json.dump(res, open(Path(os.environ.get("OUT_DIR", Path(__file__).parent)) / f"zr_check_{POOL.lower()}_pinned.json", "w"), indent=1, default=float)
print("DONE", flush=True)
