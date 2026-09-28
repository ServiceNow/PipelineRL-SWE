"""Omni-MATH-500 step 1 of the PRE-REGISTERED test (NEW_PATH 4.A.9): compute the rule's inputs from gpt-oss-20b-low x1
on all 500 problems + the Qwen3-4B Instruct / Thinking prefills, and print the resulting prediction.
  (a) probe 5-fold CV log-output R2 on gpt-oss-20b-low   (the decision variable)
  (b) difficulty -> length: 5-fold CV R2 of log length on a cubic in the Omni difficulty rating
  (c) readability: 5-fold CV R2 of the difficulty rating from the prefill
"""
import json, sys, numpy as np
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).parent))
from baseline_cost_heads import rich
from why_predictable import cv_r2

R = Path("/mnt/llmd/results/exps/aristides/reason")
rows = [json.loads(l) for l in open(R / "math_pool/omni500/oss20lo_d0.jsonl") if json.loads(l).get("finish_reason") != "error"]
pids = [r["problem_id"] for r in rows]
y = np.log(np.maximum([r["completion_tokens"] for r in rows], 1.0)); dif = np.array([r["difficulty"] for r in rows], float)
acc = np.mean([r["resolved"] for r in rows])
print(f"{len(rows)} problems; gpt-oss-20b-low accuracy {acc:.2f}; mean output {np.exp(y).mean():.0f} tok; sd log length {y.std():.2f}")
z = (dif - dif.mean()) / dif.std()
print(f"(b) difficulty -> log length, CV R2: {cv_r2(np.c_[z, z ** 2, z ** 3], y):.2f}")


def cv(X, t):
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.geomspace(1e1, 1e7, 13)))
    p = cross_val_predict(m, X, t, cv=5)
    return 1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum()


best = None
for tag in ("instruct", "thinking"):
    X = rich(R / "omni500_probe" / f"{tag}.npz", pids)
    a, c = cv(X, y), cv(X, dif)
    print(f"probe={tag:<9} (a) CV log-output R2 on oss20lo {a:.2f}   (c) readability of the rating {c:.2f}")
    best = max(best or (a, tag), (a, tag))
a, tag = best
call = "GAIN (>= 10%, CI excluding 0)" if a >= 0.50 else ("NO GAIN (CI includes 0)" if a <= 0.35 else "no directional call")
print(f"\nPRE-REGISTERED PREDICTION for the full pool: probe = {tag} (a = {a:.2f}) -> {call}; headroom >= 15% regardless")
