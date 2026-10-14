"""Does the paper's condition predict the saving out of sample? (NEW_PATH 4.A.104)
For every route subset of the five-route pools (full, leave-one-out, pairs; drop_summary_pinned.json, 10-set means) and the
eight-route sub-pools (newfamily_draws_summary_pinned.json), compute from TRAINING problems only (manifest split):
  within   mean over the subset's routes of the s.d. of log billed cost per call
  between  s.d. over the subset's routes of log mean billed cost per call (the price gap)
  ratio    log(within / between)
  legible  mean over the subset's routes of our cost readout's log-length R2 on the calibration problems (fit on train)
Targets: headroom (oracle vs median) and our saving vs median. Models (least squares): headroom ~ ratio; saving ~ headroom x
legible. Evaluated leave-one-pool-out on the five-route subsets, and fitted on all five-route subsets then applied unchanged to the
eight-route sub-pools (held out: other model families).
Usage: REASON_ROOT=.../reason_pinned python when_law.py -> when_law_pinned.json
"""
import json, os, sys, itertools
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
os.environ.pop("DRAW", None)
exec(open(Path(__file__).parent / "newfamily_eval.py").read().split("res = {}; built = {}")[0])
H = Path(__file__).parent; drop = json.load(open(H / "drop_summary_pinned.json")); nf = json.load(open(H / "newfamily_draws_summary_pinned.json"))


def route_r2(L, X, tr, ca):
    sc = StandardScaler().fit(X[tr]); Xt, Xc = sc.transform(X[tr]), sc.transform(X[ca]); out = []
    for k in range(L.shape[1]):
        y = np.log(L[:, k]); m = RidgeCV(alphas=np.geomspace(1e1, 1e7, 13)).fit(Xt, y[tr]); yc = y[ca]
        out.append(float(1 - ((yc - m.predict(Xc)) ** 2).sum() / ((yc - yc.mean()) ** 2).sum()))
    return np.array(out)


def stats(paid, r2k, tr, cols):
    lp = np.log(paid[tr][:, cols]); within = float(lp.std(0).mean()); between = float(np.log(paid[tr][:, cols].mean(0)).std())
    return dict(within=within, between=between, ratio=float(np.log(within / max(between, 1e-6))), legible=float(r2k[cols].mean()))


rows = []
for pool in drop:
    d = load(pool); slots = d["slots"]; r2k = route_r2(d["L"], d["X"], d["tr"], d["ca"])
    for sp, o in drop[pool].items():
        if sp == "n_sets" or o["ours"]["vs_median"][0] is None or o["oracle"]["vs_median"][0] is None:
            continue
        cols = list(range(len(slots))) if sp == "full" else [k for k in range(len(slots)) if slots[k] != sp[1:]] if sp.startswith("-") else [slots.index(x) for x in sp.split("+")]
        st = stats(d["paid"], r2k, d["tr"], cols)
        rows.append(dict(pool=pool, subset=sp, group="five", headroom=o["oracle"]["vs_median"][0], ours=o["ours"]["vs_median"][0], **st))
    print(pool, "done", flush=True)
SUB = {"newpool": [0, 1, 5, 6, 7], "no_dsv4f": [0, 1, 3, 4, 5, 6, 7], "all8": list(range(8))}
for pool in ("LCB", "Omni", "MMLU-Pro"):
    S = build(pool); r2k = route_r2(S["L"], S["X"], S["tr"], S["ca"])
    for sp, cols in SUB.items():
        o = nf[pool][sp]["arms"]
        st = stats(S["paid"], r2k, S["tr"], cols)
        rows.append(dict(pool=pool, subset=sp, group="eight", headroom=o["oracle"]["vs_median"]["v"][0], ours=o["ours"]["vs_median"]["v"][0], **st))
R_ = {"rows": rows}


def fit(Xm, y):
    A = np.c_[np.ones(len(Xm)), Xm]; return np.linalg.lstsq(A, y, rcond=None)[0]


pred = lambda b, Xm: np.c_[np.ones(len(Xm)), Xm] @ b
r2 = lambda y, yh: float(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())
five = [r for r in rows if r["group"] == "five"]; eight = [r for r in rows if r["group"] == "eight"]
for tgt, feats in (("headroom", lambda r: [r["ratio"]]), ("ours", lambda r: [r["headroom"] * r["legible"]]), ("ours_from_stats", lambda r: [r["ratio"], r["legible"], r["ratio"] * r["legible"]])):
    y_of = (lambda r: r["ours"]) if tgt.startswith("ours") else (lambda r: r["headroom"])
    Xf, yf = np.array([feats(r) for r in five]), np.array([y_of(r) for r in five]); lopo = np.zeros(len(five))
    for pool in {r["pool"] for r in five}:
        m = np.array([r["pool"] == pool for r in five]); lopo[m] = pred(fit(Xf[~m], yf[~m]), Xf[m])
    b = fit(Xf, yf); Xe, ye = np.array([feats(r) for r in eight]), np.array([y_of(r) for r in eight]); ye_h = pred(b, Xe)
    R_[tgt] = dict(coef=b.tolist(), r2_in=r2(yf, pred(b, Xf)), r2_lopo=r2(yf, lopo), r2_eight=r2(ye, ye_h), mae_eight=float(np.abs(ye - ye_h).mean()),
                   eight=[dict(pool=r["pool"], subset=r["subset"], actual=y, predicted=float(p)) for r, y, p in zip(eight, ye, ye_h)])
    print(f"{tgt:<16} coef {np.round(b, 2).tolist()}  R2 in-sample {R_[tgt]['r2_in']:.2f}  leave-one-pool-out {R_[tgt]['r2_lopo']:.2f}  "
          f"held-out eight-route pools {R_[tgt]['r2_eight']:.2f} (MAE {R_[tgt]['mae_eight']:.1f} pt)", flush=True)
    for e in R_[tgt]["eight"]:
        print(f"    {e['pool']:<9} {e['subset']:<9} actual {e['actual']:+5.1f}  predicted {e['predicted']:+5.1f}", flush=True)
json.dump(R_, open(Path(os.environ.get("OUT_DIR", H)) / "when_law_pinned.json", "w"), indent=1, default=float)
print("DONE", flush=True)
