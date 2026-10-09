"""SWE-Smith one-shot reasoning pool v2 (NEW_PATH 4.A.76): 1,432 instances x 5 open reasoning routes, one draw each, deepseek-v4-flash
pinned to StreamLake, REAL Daytona labels (report.json `resolved`; never the collection's proxy route_successes) -> tensors in the
pinned shadow root, the format fresh_baselines.py --pool reads. A draw with no patch, or one that failed to apply, is a paid, failed draw
(valid = 1 for every draw with a predictions row and a report). Split: seed-0 permutation, 55 / 15 / 30 train / calibration / test.
problems.jsonl carries the issue text (problem_statement) for the prompt-feature baseline.
Usage: python build_swesmith_v2.py
"""
import glob, json
from pathlib import Path
import numpy as np, pandas as pd

R = Path("/mnt/llmd/results/exps/aristides/reason"); W = R / "swesmith_reasoning_pool_v2"
OUT = Path("/mnt/llmd/results/exps/aristides/reason_pinned/swesmith_v2_tensors"); LOGS = Path("/home/toolkit/PipelineRL-SWE/logs/run_evaluation")
ROUTES = ["oss20lo", "oss20md", "dsv4f", "oss120md", "oss120hi"]
RUN = {r: f"swesmith_reason_v2_predictions_{r}_d0" for r in ROUTES} | {"dsv4f": "swesmith_reason_pinned_predictions_dsv4f_d0"}
COLL = R / "offline_router_swe_smith_train1500_real_labels_4route_1780639659" / "collect"

ids = json.load(open(W / "full" / "ids.json"))
z = np.load(R / "swesmith_costhead" / "scout_prefill.npz", allow_pickle=True); have = {str(p) for p in z["problem_ids"]}
ids = [i for i in ids if i in have]; n, M = len(ids), len(ROUTES)
fo = np.zeros((n, M, 1), bool); va = np.zeros((n, M, 1), bool); pt = np.zeros((n, M, 1), np.float32); ct = np.zeros((n, M, 1), np.float32)
missing = {}
for m, r in enumerate(ROUTES):
    tok = {}
    for l in open(W / "full" / f"predictions_{r}_d0.jsonl"):
        if l.strip():
            x = json.loads(l); tok[x["instance_id"]] = x                     # latest row per instance
    missing[r] = 0
    for i, iid in enumerate(ids):
        rep = LOGS / RUN[r] / iid / "report.json"
        if iid not in tok or (tok[iid].get("why_empty") or "").startswith("error"):
            continue                                                         # no paid draw (API error): invalid
        va[i, m, 0] = True; pt[i, m, 0] = tok[iid].get("prompt_tokens") or 0; ct[i, m, 0] = tok[iid].get("completion_tokens") or 0
        if rep.exists():
            fo[i, m, 0] = bool(json.loads(rep.read_text()).get("resolved"))
        elif (tok[iid].get("model_patch") or "").strip():
            missing[r] += 1                                                  # a patch that was never labelled: treat as invalid
            va[i, m, 0] = False
stmt = {}
for f in sorted(glob.glob(str(COLL / "*" / "*.parquet"))):
    df = pd.read_parquet(f, columns=["problem_id", "problem_statement"])
    stmt.update(dict(zip(df.problem_id, df.problem_statement)))
rng = np.random.default_rng(0); perm = rng.permutation(n); ntr, ncal = int(0.55 * n), int(0.15 * n)
split = {"train_problem_ids": [ids[k] for k in perm[:ntr]], "calibration_problem_ids": [ids[k] for k in perm[ntr:ntr + ncal]],
         "test_problem_ids": [ids[k] for k in perm[ntr + ncal:]]}
OUT.mkdir(parents=True, exist_ok=True)
np.savez(OUT / "tensors.npz", final_outcome=fo, execution_outcome=fo, weak_verifier_outcome=fo, valid=va, prompt_tokens=pt, completion_tokens=ct,
         problem_ids=np.array(ids), model_slots=np.array(ROUTES), schema_version=3)
(OUT / "split_manifest.json").write_text(json.dumps(split))
with open(OUT / "problems.jsonl", "w") as f:
    for iid in ids:
        f.write(json.dumps({"problem_id": iid, "problem_statement": str(stmt.get(iid, ""))}) + "\n")
print(f"{n} instances -> {OUT}; resolve rate per route {fo[:, :, 0].sum(0) / np.maximum(va[:, :, 0].sum(0), 1)}; valid {va.mean(0)[:, 0]}; "
      f"patches without a report (set invalid) {missing}; split {ntr}/{ncal}/{n - ntr - ncal}; mean out tokens {ct[:, :, 0].mean(0).round(0)}")
