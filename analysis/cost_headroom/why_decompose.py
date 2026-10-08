"""WHY is output length predictable? Variance decomposition of log output length (per route, TEST problems) into what
the probe's out-of-sample prediction shares with DIFFICULTY and what it adds beyond it.

For target y (log mean output tokens), probe prediction P (plain-ridge head fitted on train, log output tokens) and
difficulty features D (5-fold CV R2 throughout, test problems only):
  R2_D  = R2(y ~ D),  R2_P = R2(y ~ P),  R2_DP = R2(y ~ D + P)
  shared          = R2_D + R2_P - R2_DP   (length explained by difficulty that the probe also captures)
  difficulty-only = R2_DP - R2_P          (difficulty the probe misses)
  probe-only      = R2_DP - R2_D          (length the probe predicts beyond difficulty)
Two difficulty measures:
  empirical  cubic in the route's own solve rate and the pool-mean solve rate (from the labels; NOT available at
             routing time -- this is the ceiling of "difficulty" as the models experience it)
  labelled   the dataset's human difficulty label (LCB easy/medium/hard, CodeContests Codeforces rating, TACO tier,
             Omni-MATH rating), where it exists
Also: the length of failed vs solved draws within a problem-route (is "hard" long because failing is long?).
"""
import json, os, sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from decompose import MK, R
from why_predictable import cv_r2

POOLS = {"pool_v2_tensors_5rung": "cost_preds_probe.jsonl", "cc_tensors": "cost_preds_probe.jsonl",
         "taco_tensors_ha": "cost_preds_probe.jsonl", "bcb_tensors_5r": "cost_preds_probe.jsonl",
         "omni500_tensors": "cost_preds_probe_thinking.jsonl",
         "mmlupro_tensors": "cost_preds_probe_instruct.jsonl", "aime_tensors": "cost_preds_probe_instruct.jsonl", "apps_tensors": "cost_preds_probe.jsonl"}


def labels(name, pids):
    meta = {str(json.loads(l)["problem_id"]): json.loads(l) for l in open(R / name / "problems.jsonl")}
    if name.startswith("cc"):
        tasks = {json.loads(l)["problem_id"]: json.loads(l) for l in open(R / "cc_pool" / "cc_tasks.jsonl")}
        r = np.array([tasks.get(p, {}).get("cf_rating") or np.nan for p in pids], float)
    elif name.startswith("omni"):
        r = np.array([float(meta[p].get("difficulty", np.nan)) for p in pids])
    else:
        tier = {"easy": 0, "medium": 1, "hard": 2}
        r = np.array([tier.get(str(meta[p].get("difficulty", "")), np.nan) for p in pids], float)
    if np.isfinite(r).sum() < 0.5 * len(r) or np.nanstd(r) == 0:
        return None
    r = np.where(np.isfinite(r), r, np.nanmedian(r)); z = (r - r.mean()) / r.std()
    return np.c_[z, z ** 2, z ** 3]


def main():
    out = {}
    print("per route, TEST problems, 5-fold CV R2 of log output length; shared = difficulty the probe also captures")
    print(f"{'pool / route':<30}{'probe':>7} | {'empirical difficulty: R2_D  shared  diff-only  probe-only':<58}| labelled: R2_D  shared  probe-only")
    for name, cf in POOLS.items():
        D = R / name; t = np.load(D / "tensors.npz", allow_pickle=True)
        S = [str(s) for s in t["model_slots"]]; pids = [str(p) for p in t["problem_ids"]]; pi = {p: i for i, p in enumerate(pids)}
        v = t["valid"].astype(bool); ok = (t["final_outcome"] & t["valid"]); ct = t["completion_tokens"].astype(float)
        pt = t["prompt_tokens"].astype(float); n = v.sum(2)
        outm = np.where(n > 0, np.where(v, ct, 0).sum(2) / np.maximum(n, 1), np.nan)
        Q = np.where(n > 0, (ok & v).sum(2) / np.maximum(n, 1), np.nan); qbar = np.nanmean(Q, 1)
        inp = np.nanmean(np.where(v, pt, np.nan), 2)
        te = np.array([pi[str(p)] for p in json.load(open(D / "split_manifest.json"))["test_problem_ids"]])
        lc = {json.loads(l)["problem_id"]: json.loads(l)["expected_costs"][:len(S)] for l in open(D / cf)}
        LC = np.array([lc[p] for p in pids]); L = labels(name, pids)
        for m, s in enumerate(S):
            tt = te[np.isfinite(outm[te, m])]
            y = np.log(np.maximum(outm[tt, m], 1))
            P = np.log(np.maximum((LC[tt, m] - np.nan_to_num(inp[tt, m]) * MK[s][0] / 1e6) / (MK[s][1] / 1e6), 1))[:, None]
            q, qb = np.nan_to_num(Q[tt, m]), qbar[tt]
            De = np.c_[q, q ** 2, q ** 3, qb, qb ** 2, qb ** 3]
            rp = cv_r2(P, y); rd = cv_r2(De, y); rdp = cv_r2(np.c_[De, P], y)
            row = dict(probe=rp, emp_D=rd, emp_shared=rd + rp - rdp, emp_diff_only=rdp - rp, emp_probe_only=rdp - rd)
            lab = ""
            if L is not None:
                rl = cv_r2(L[tt], y); rlp = cv_r2(np.c_[L[tt], P], y)
                row.update(lab_D=rl, lab_shared=rl + rp - rlp, lab_probe_only=rlp - rl)
                lab = f"{rl:6.2f}{rl + rp - rlp:8.2f}{rlp - rl:12.2f}"
            # failed vs solved draws of the same problem-route: is failing long?
            fr = []
            for i in tt:
                a, b = ct[i, m][v[i, m] & ~ok[i, m]], ct[i, m][v[i, m] & ok[i, m]]
                if len(a) and len(b):
                    fr.append(np.log(a.mean() / max(b.mean(), 1)))
            row["fail_vs_solve_ratio"] = float(np.exp(np.mean(fr))) if fr else np.nan
            out[f"{name}/{s}"] = row
            print(f"{name[:18] + ' / ' + s:<30}{rp:7.2f} | {rd:9.2f}{rd + rp - rdp:8.2f}{rdp - rp:11.2f}{rdp - rd:12.2f}{'':>17}| {lab}"
                  f"   fail/solve x{row['fail_vs_solve_ratio']:.2f}")
    json.dump(out, open(f"analysis/cost_headroom/why_decompose{os.environ.get('RESULT_TAG', '')}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
