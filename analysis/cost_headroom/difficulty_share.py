"""How much of each candidate's log output length is difficulty, and how much of that the prefill reads (paper split, one training set).
Difficulty here is an ORACLE: a problem's measured solve rate across the other candidates (the candidate itself left out, so the
difficulty and the length never come from the same draws), known only after running and grading every model; no router has it.
Per candidate, test R2 of log mean output length for three linear fits on the calibration problems (where the prefill's predictions
are out of sample): prefill (our cost probe's log prediction), difficulty (solve rate and its square), both. Also the all-candidate
solve rate (the candidate included), as best_estimators.py's true_difficulty.
Usage: REASON_ROOT=.../reason_pinned RESULT_TAG=_pinned python difficulty_share.py [POOL ...] -> difficulty_share_pinned.json
"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
WANT = sys.argv[1:] or ["LCB", "Omni", "MMLU-Pro", "AIME", "APPS", "BCB", "CC", "SuperGPQA", "BBEH"]; sys.argv = [sys.argv[0], "diffshare", "ALL"]
exec(open(Path(__file__).parent / "cost_generalization.py").read().split('MODE, POOL = sys.argv[1], sys.argv[2]')[0])
r2 = lambda y, yh: float(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())
def lin(Fc, yc, Ft):
    A = np.c_[np.ones(len(Fc)), Fc]; b = np.linalg.lstsq(A, yc, rcond=None)[0]; return np.c_[np.ones(len(Ft)), Ft] @ b
res = {}
for P in WANT:
    d = load(P); N = len(d["ids"]); tr, ca, ev = d["tr"], d["ca"], d["ev"]; M = d["M"]; Y = np.log(d["L"])
    pre = np.log(np.maximum(fit_predict("ours", view(d, tr), view(d, np.arange(N))), 1))
    out = {"slots": d["slots"], "n_cal": int(len(ca)), "n_test": int(len(ev))}
    for key in ("prefill", "difficulty", "both", "difficulty_all"):
        out[key] = []
    for k in range(M):
        s = np.delete(d["q"], k, 1).mean(1); sa = d["q"].mean(1)
        F = {"prefill": pre[:, k:k + 1], "difficulty": np.c_[s, s ** 2], "both": np.c_[pre[:, k], s, s ** 2], "difficulty_all": np.c_[sa, sa ** 2]}
        for key, X in F.items():
            out[key].append(r2(Y[ev, k], lin(X[ca], Y[ca, k], X[ev])))
    out["gain_from_difficulty"] = [b - p for b, p in zip(out["both"], out["prefill"])]
    out["beyond_difficulty"] = [b - q for b, q in zip(out["both"], out["difficulty"])]
    res[P] = out
    f = lambda x: f"{min(x):.2f}-{max(x):.2f}"
    print(f"{P:<10} cal {len(ca)} test {len(ev)}  prefill {f(out['prefill'])}  difficulty {f(out['difficulty'])} (all {f(out['difficulty_all'])})  "
          f"both {f(out['both'])}  +difficulty {f(out['gain_from_difficulty'])}  beyond {f(out['beyond_difficulty'])}", flush=True)
json.dump(res, open(Path(__file__).parent / f"difficulty_share{os.environ.get('RESULT_TAG', '')}.json", "w"), indent=1)
