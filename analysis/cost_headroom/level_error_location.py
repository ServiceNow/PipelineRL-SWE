"""Where are the real level predictor's errors? CC: real 4B-reads-prefix level (R2 .72) captures 2.0% of a 12.7% level
headroom while synthetic noise at equal R2 captures 6.2%. Compare their log-level errors (test) by quartile of the TRUE
level, and on the problems where the ORACLE-level router and the paper rule choose differently (decision-relevant)."""
import json, numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from decompose import MK, R, VS
D = R / "cc_tensors"; t = np.load(D / "tensors.npz", allow_pickle=True)
S = [str(s) for s in t["model_slots"]]; pids = [str(p) for p in t["problem_ids"]]; pi = {p: i for i, p in enumerate(pids)}
v = t["valid"].astype(bool); n = v.sum(2); ok = (t["final_outcome"] & t["valid"]).astype(float)
real = np.stack([(t["prompt_tokens"][:, m] * MK[s][0] + t["completion_tokens"][:, m] * MK[s][1]) / 1e6 for m, s in enumerate(S)], 1)
L = np.log(np.where(n > 0, (real * v).sum(2) / np.maximum(n, 1), np.nan)); lvl = np.nanmean(L, 1)
te = np.array([pi[str(p)] for p in json.load(open(D / "split_manifest.json"))["test_problem_ids"]])
ld = lambda f: np.log(np.array([[json.loads(l)["expected_costs"][m] for m in range(len(S))] for l in open(D / f)]))
lc = {json.loads(l)["problem_id"]: i for i, l in enumerate(open(D / "cost_preds_lvr_probe_prefixread_level.jsonl"))}
real_lvl = np.nanmean(ld("cost_preds_lvr_probe_prefixread_level.jsonl"), 1); syn_lvl = np.nanmean(ld("cost_preds_lvr_synlevel_72_s0.jsonl"), 1)
orc = ld("cost_preds_lvr_oracle_level.jsonl"); pap_raw = {json.loads(l)["problem_id"]: json.loads(l)["expected_costs"] for l in open(D / "cost_preds_market.jsonl")}
_lp = {json.loads(l)["problem_id"]: json.loads(l)["p_successes"][:len(S)] for l in open(D / "content_preds.jsonl")}
P = np.array([_lp[p] for p in pids])                     # keyed by problem_id
# bias-free comparison: remove each predictor's mean offset (level R2 is offset-sensitive; decisions via ratios are not)
def err(x): e = x[te] - lvl[te]; return e - np.nanmean(e)
er, es = err(real_lvl), err(syn_lvl)
q = np.digitize(lvl[te], np.nanpercentile(lvl[te], [25, 50, 75]))
print("CC test, |error| of the log LEVEL (offset removed): real 4B-reads-prefix vs synthetic (both R2 ~.72)")
for k in range(4):
    m = q == k; print(f"   true-level quartile {k+1} ({'cheapest' if k==0 else 'most expensive' if k==3 else 'middle'}): real {np.nanmean(np.abs(er[m])):.2f}  synthetic {np.nanmean(np.abs(es[m])):.2f}   n={m.sum()}")
print(f"   correlation of error with true level: real {np.corrcoef(er, lvl[te])[0,1]:+.2f}  synthetic {np.corrcoef(es, lvl[te])[0,1]:+.2f}  (negative = under-predicts the long ones)")
# decision-relevant problems: where the ORACLE-level router and the paper rule choose different routes, at the V whose
# paper-rule test accuracy is closest to the middle of the band
Q = np.where(n > 0, (ok * v).sum(2) / np.maximum(n, 1), 0)
paper = np.log(np.array([pap_raw[p] for p in pids]))                  # not the paper rule exactly; use the median-cost arm:
med = np.nanmedian(L[np.array([pi[str(p)] for p in json.load(open(D / "split_manifest.json"))["train_problem_ids"]])], 0)
paper = np.repeat(med[None], len(pids), 0)
def choose(C, V): return np.argmax(P[te] * V - np.exp(C[te]), 1)
accs = [(V, Q[te, choose(paper, V)].mean() if False else np.mean(Q[te][np.arange(len(te)), choose(paper, V)])) for V in VS]
V = min(accs, key=lambda x: abs(x[1] - 0.67))[0]
rel = choose(orc, V) != choose(paper, V)
print(f"decision-relevant (oracle-level vs median-cost choose differently at the ~67%-accuracy operating point): {rel.sum()} of {len(te)}")
print(f"   |error| there: real {np.nanmean(np.abs(er[rel])):.2f}  synthetic {np.nanmean(np.abs(es[rel])):.2f};  elsewhere: real {np.nanmean(np.abs(er[~rel])):.2f}  synthetic {np.nanmean(np.abs(es[~rel])):.2f}")
# redundancy with the success head: the part of the TRUE level NOT explained by the success predictions (logit p, CV)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
lp = np.log(np.clip(P[te], 1e-4, 1 - 1e-4) / (1 - np.clip(P[te], 1e-4, 1 - 1e-4)))
fit = cross_val_predict(LinearRegression(), lp, lvl[te], cv=5); resid = lvl[te] - fit
r2 = lambda y, x: 1 - ((y - x) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print(f"success predictions explain {r2(lvl[te], fit):.2f} of the true level (CV)")
for nm, x in (("real 4B-reads-prefix", real_lvl[te]), ("synthetic (R2 .72)", syn_lvl[te])):
    xr = x - cross_val_predict(LinearRegression(), lp, x, cv=5)            # the predictor's part not explained by p
    print(f"   {nm:<22}: R2 of the level's part BEYOND the success head: {max(-9, 1 - ((resid - np.polyval(np.polyfit(xr, resid, 1), xr)) ** 2).sum() / ((resid - resid.mean()) ** 2).sum()):+.2f}")
