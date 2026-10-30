"""Frontiers relative to deepseek-v4-flash alone: for every arm of tab:allest (plus perfect cost knowledge), the cost of reaching a given
share of deepseek-v4-flash's test accuracy, as a share of what calling deepseek-v4-flash on every query costs. Each training set's
frontier (upper concave hull of the V sweep, billed costs, the test problems every training set keeps) is rebuilt from the per-problem
predictions saved by learning_curve.py (n = all: the tab:allest protocol), read at accuracy shares 0.50-1.00, and the ten training sets
are combined by geometric mean where all ten frontiers reach that accuracy.
Usage: REASON_ROOT=.../reason_pinned RESULT_TAG=_pinned python relative_frontiers.py LCB [POOL ...] -> relative_frontiers_pinned.json (merged)
"""
import glob, json, sys
from pathlib import Path
WANT = sys.argv[1:]
__file__ = str(Path(__file__).resolve()); _cg = Path(__file__).parent / "cost_generalization.py"; sys.argv = ["x", "labels", WANT[0]]
exec(open(_cg).read().split('MODE, POOL = sys.argv[1], sys.argv[2]')[0])
LC = Path("/mnt/llmd/results/exps/aristides/reason/reason_pinned_logs/lcurve")
COST = ["ours", "median", "mean", "gbm", "mixllm", "rtr", "zr-bins", "fromsuccess", "scx"]; SYSTEM = ["zr_pub", "zr_4b"]
F = np.round(np.arange(0.50, 1.0001, 0.01), 2)
outf = Path(__file__).parent / "relative_frontiers_pinned.json"; res = json.load(open(outf)) if outf.exists() else {}
for P in WANT:
    d = load(P); ev = d["ev"]; k_ds = d["slots"].index("dsv4f")
    X, Y = float(d["paid"][ev, k_ds].mean()), float(d["q"][ev, k_ds].mean())
    per = {a: [] for a in COST + SYSTEM + ["oracle"]}
    for f in sorted(glob.glob(str(LC / f"learning_curve_{P.lower().replace('-', '')}_*_pinned.npz"))):
        z = np.load(f); assert (z["ids"] == np.array(d["ids"])).all()
        for s in sorted({k.split("_")[0] for k in z.files if k.endswith("_ev")}):
            assert (z[f"{s}_ev"] == ev).all()
            arms = {a: (z[f"{s}_all_cost_{a}_P"], z[f"{s}_all_cost_{a}_tok"]) for a in COST}
            arms |= {a: (z[f"{s}_all_system_{a}_P"], z[f"{s}_all_system_{a}_tok"]) for a in SYSTEM}
            arms["oracle"] = (z[f"{s}_P_full"], d["L"])
            for a, (Pa, tk) in arms.items():
                H = curve(d, Pa.astype(float), cost_of(d, tk.astype(float)), ev)
                per[a].append([cost_at(H, x * Y) / X for x in F])
    out = {"deepseek": dict(usd_per_1k=X * 1000, accuracy=Y), "share_of_accuracy": F.tolist(), "n_sets": len(per["ours"]), "arms": {}}
    for a, v in per.items():
        v = np.array(v, float); gm = np.where(np.isnan(v).any(0), np.nan, np.exp(np.log(np.where(v > 0, v, np.nan)).mean(0)))
        out["arms"][a] = dict(cost_share=[None if not np.isfinite(x) else float(x) for x in gm])
    res[P] = out
    r = lambda a, x: out["arms"][a]["cost_share"][int(round((x - .5) * 100))]
    print(P, f"deepseek ${X * 1000:.3f}/1k acc {Y * 100:.1f}%  n_sets {out['n_sets']}")
    for a in per:
        print(f"  {a:12s} " + "  ".join(f"{x:.0%}:{r(a, x) * 100:5.1f}" if r(a, x) is not None else f"{x:.0%}:  -  " for x in (.6, .7, .8, .9, .95, .99)))
json.dump(res, open(outf, "w"), indent=1)
