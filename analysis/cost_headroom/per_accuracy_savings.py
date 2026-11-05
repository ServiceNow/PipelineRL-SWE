"""Saving of our cost probe vs median pricing at each of saved()'s 12 matched accuracies (make_hero_fig.py), on the ten training sets of
the tab:allest protocol: same success probes, billed costs. The geometric mean of the 12 savings in a set is that set's saved().
Usage: REASON_ROOT=.../reason_pinned RESULT_TAG=_pinned python per_accuracy_savings.py [POOL ...] -> per_accuracy_savings_pinned.json
"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
POOLS_ = sys.argv[1:] or ["LCB", "Omni", "MMLU-Pro", "AIME", "APPS", "BCB", "CC", "SuperGPQA", "BBEH"]; sys.argv = [sys.argv[0], "peracc", "ALL"]
exec(open(Path(__file__).parent / "cost_generalization.py").read().split('MODE, POOL = sys.argv[1], sys.argv[2]')[0])


def saved_at(H, H0):
    """saved()'s 12 matched accuracies and the saving (%) at each, low to high"""
    lo, hi = max(H[0][1], H0[0][1]), min(H[-1][1], H0[-1][1])
    T = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 12)
    return T, np.array([100 * (1 - float(cost_at(H, x) / cost_at(H0, x))) for x in T])


out = {}
for pool in POOLS_:
    A, V, M = [], [], []
    for s in range(10):
        os.environ["DRAW"] = str(s); d = load(pool); N = len(d["ids"]); ev = d["ev"]
        ours = fit_predict("ours", view(d, d["tr"]), view(d, np.arange(N)))
        H, H0 = curve(d, d["P"], cost_of(d, ours), ev), curve(d, d["P"], cost_of(d, np.repeat(d["med"][None], N, 0)), ev)
        t, v = saved_at(H, H0); A.append(t); V.append(v); M.append(100 * saved(H, H0))
    out[pool] = dict(accuracy=np.mean(A, 0).tolist(), saved=np.mean(V, 0).tolist(), saved_by_set=np.array(V).tolist(), band_mean=float(np.mean(M)))
    print(f"{pool:<10} mean {np.mean(M):5.1f}  acc {np.mean(A, 0)[0] * 100:.0f}-{np.mean(A, 0)[-1] * 100:.0f}  " + " ".join(f"{x:.0f}" for x in np.mean(V, 0)), flush=True)
    json.dump(out, open(Path(__file__).parent / "per_accuracy_savings_pinned.json", "w"), indent=1)
