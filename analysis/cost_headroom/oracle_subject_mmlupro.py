"""MMLU-Pro, full test set (6,500), paper split, one training set: does an ORACLE that knows each problem's difficulty and subject price
queries as well as the prefill cost probe? Oracle difficulty = the problem's measured solve rate across the other candidates (known only
after running and grading every model; no router has it). Per candidate, ridge of log mean output length on the training problems from
(a) oracle difficulty and its square, (b) (a) + subject one-hot; Duan smearing as the cost probe. Routed with the same success
predictions as the cost probe and median pricing; billed costs. Cost saved at matched accuracy (the paper's measure) with paired bootstrap
intervals over test problems (NBOOT resamples). oracle1_*: the solve rate from each training problem's first draw only, matching
the test set's one draw per candidate (no train/test shift in how noisy the oracle is).
Replaces mmlupro_length_driver.py's 'difficulty + subject' row, whose 'difficulty' was our predicted success logits on a 300-problem split.
Usage: REASON_ROOT=.../reason_pinned RESULT_TAG=_pinned python oracle_subject_mmlupro.py -> oracle_subject_mmlupro_pinned.json
"""
import json, os, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).parent))
sys.argv = [sys.argv[0], "oraclesubj", "MMLU-Pro"]
exec(open(Path(__file__).parent / "cost_generalization.py").read().split('MODE, POOL = sys.argv[1], sys.argv[2]')[0])
d = load("MMLU-Pro"); N = len(d["ids"]); tr, ev, M = d["tr"], d["ev"], d["M"]; Y = np.log(d["L"])
RP = Path(os.environ["REASON_ROOT"]); sub = {}
for f in (RP / "math_pool/mmlupro/problems.jsonl", RP / "math_expand_20261001/mmlupro/problems.jsonl"):
    for l in open(f):
        r = json.loads(l); sub.setdefault(str(r["problem_id"]), r.get("subject"))
subj = [sub.get(p) for p in d["ids"]]; miss = sum(s is None for s in subj); print("problems", N, "missing subject", miss, flush=True)
assert miss == 0
SUB = sorted(set(subj)); SJ = np.array([[s == k for k in SUB] for s in subj], float)
Q1 = d["y0"].astype(float)                       # first valid draw only: the test set has one draw per candidate, so training gets one too
def oracle_tokens(with_subject, Q=None):
    Q = d["q"] if Q is None else Q; out = np.zeros((N, M))
    for k in range(M):
        s = np.delete(Q, k, 1).mean(1); F = np.c_[s, s ** 2, SJ] if with_subject else np.c_[s, s ** 2]
        sc = StandardScaler().fit(F[tr]); m = RidgeCV(alphas=np.geomspace(1e-3, 1e4, 15)).fit(sc.transform(F[tr]), Y[tr, k])
        smear = np.mean(np.exp(Y[tr, k] - m.predict(sc.transform(F[tr])))); out[:, k] = np.exp(m.predict(sc.transform(F))) * smear
    return out
tk = {"median": np.repeat(d["med"][None], N, 0), "ours": fit_predict("ours", view(d, tr), view(d, np.arange(N))),
      "oracle_difficulty": oracle_tokens(False), "oracle_difficulty_subject": oracle_tokens(True),
      "oracle1_difficulty": oracle_tokens(False, Q1), "oracle1_difficulty_subject": oracle_tokens(True, Q1)}
C = {a: cost_of(d, x) for a, x in tk.items()}
r2 = lambda y, yh: float(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())
res = {"n_test": int(len(ev)), "n_train": int(len(tr)), "subjects": len(SUB),
       "r2": {a: [r2(Y[ev, k], np.log(np.maximum(tk[a][ev, k], 1))) for k in range(M)] for a in tk if a != "median"}}
def est(ii):
    H = {a: curve(d, d["P"], C[a], ii) for a in C}
    return {**{f"{a}_vs_median": saved(H[a], H["median"]) for a in C if a != "median"},
            "ours_vs_oracle_difficulty_subject": saved(H["ours"], H["oracle_difficulty_subject"]),
            "ours_vs_oracle_difficulty": saved(H["ours"], H["oracle_difficulty"]),
            "ours_vs_oracle1_difficulty_subject": saved(H["ours"], H["oracle1_difficulty_subject"])}
pt = est(ev); rng = np.random.default_rng(0); B = [est(ev[rng.integers(0, len(ev), len(ev))]) for _ in range(NBOOT)]
res["saved"] = {k: [pt[k], float(np.nanpercentile([b[k] for b in B], 2.5)), float(np.nanpercentile([b[k] for b in B], 97.5))] for k in pt}
for k, v in res["saved"].items():
    print(f"  {k:<36} {v[0]*100:6.1f} [{v[1]*100:.1f}, {v[2]*100:.1f}]")
for a, v in res["r2"].items():
    print(f"  R2 {a:<28} " + " ".join(f"{x:.2f}" for x in v))
json.dump(res, open(Path(__file__).parent / f"oracle_subject_mmlupro{os.environ.get('RESULT_TAG', '')}.json", "w"), indent=1)
