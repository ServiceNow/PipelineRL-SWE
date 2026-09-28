"""Is routing's cost headroom in the SHARED per-problem cost LEVEL or in the RELATIVE cost across routes? (the caveat: a
problem that is long for every model multiplies all routes' costs and should push toward cheaper routes even at equal p.)

log cost L[x, m] (dollars, market prices) = level l(x) + difference d(x, m), with l(x) = mean over routes of L[x, m].
Cost files written for decompose.py (same success head, same protocol, vs the paper rule):
  oracle_full    exp(L)                                   (= the ORACLE arm)
  oracle_level   exp(l(x) + dbar_m)                       true level, each route's TRAIN-average difference
  oracle_diff    exp(lbar + d(x, m))                      true differences, the TRAIN-average level
  <pred>_level / <pred>_diff: the same split applied to a real predictor's costs (probe; probe_prefixread where it exists)
"""
import json, sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from decompose import MK, R

name = sys.argv[1]; preds = sys.argv[2:]
D = R / name; t = np.load(D / "tensors.npz", allow_pickle=True)
S = [str(s) for s in t["model_slots"]]; pids = [str(p) for p in t["problem_ids"]]; pi = {p: i for i, p in enumerate(pids)}
PT = dict(MK)
if (D / "prices.json").exists():
    PT.update({k: tuple(v) for k, v in json.load(open(D / "prices.json")).items()})
v = t["valid"].astype(bool); n = v.sum(2)
real = np.stack([(t["prompt_tokens"][:, m] * PT[s][0] + t["completion_tokens"][:, m] * PT[s][1]) / 1e6 for m, s in enumerate(S)], 1)
Cr = np.where(n > 0, (real * v).sum(2) / np.maximum(n, 1), np.nan)                      # USD
tr = np.array([pi[str(p)] for p in json.load(open(D / "split_manifest.json"))["train_problem_ids"]])


def split(C):
    L = np.log(C); lvl = np.nanmean(L, 1); d = L - lvl[:, None]
    return L, lvl, d, np.nanmean(d[tr], 0), np.nanmean(lvl[tr])


def write(tag, C):
    C = np.where(np.isfinite(C), C, np.nanmedian(C[tr], 0)[None].repeat(len(C), 0))
    with open(D / f"cost_preds_lvr_{tag}.jsonl", "w") as f:
        for i, p in enumerate(pids):
            f.write(json.dumps({"problem_id": p, "expected_costs": [float(x) for x in C[i]]}) + "\n")


L, lvl, d, dbar, lbar = split(Cr)
write("oracle_full", np.exp(L)); write("oracle_level", np.exp(lvl[:, None] + dbar[None])); write("oracle_diff", np.exp(lbar + d))
print(f"{name}: sd of true log-cost LEVEL {np.nanstd(lvl):.2f}; sd of true DIFFERENCE per route "
      + " ".join(f"{s} {np.nanstd(d[:, m]):.2f}" for m, s in enumerate(S)))
for cf in preds:
    lc = {json.loads(l)["problem_id"]: json.loads(l)["expected_costs"][:len(S)] for l in open(D / cf)}
    P = np.array([lc[p] for p in pids]); Lp, lp, dp, dpbar, lpbar = split(P)
    tag = cf.replace("cost_preds_", "").replace(".jsonl", "")
    write(f"{tag}_level", np.exp(lp[:, None] + dpbar[None])); write(f"{tag}_diff", np.exp(lpbar + dp))
    te = np.array([pi[str(p)] for p in json.load(open(D / "split_manifest.json"))["test_problem_ids"]])
    r2 = lambda y, x: 1 - np.nansum((y - x) ** 2) / np.nansum((y - np.nanmean(y)) ** 2)
    print(f"   {tag}: R2 of the LEVEL {r2(lvl[te], lp[te]):+.2f}; R2 of the DIFFERENCES (pooled over routes) "
          f"{r2((d[te] - dbar).ravel(), (dp[te] - dpbar).ravel()):+.2f}")
