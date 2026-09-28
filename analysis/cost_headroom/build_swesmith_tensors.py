"""Convert the SWE-Smith REASONING pool (plan D: 500 instances x 5 open reasoning routes, one draw each, real
Daytona labels) into the tensors format the cost-headroom decomposition reads.

final_outcome = report.json `resolved` of that route's draw; a draw with no patch (or one that failed to apply)
is a paid, failed draw, so valid = 1 for every draw that has a predictions row. Split = the one
swesmith_reason_cost_head.py used (seed-0 permutation, 55 / 15 / 30 train / calibration / test).
"""
import json
from pathlib import Path
import numpy as np

R = Path("/mnt/llmd/results/exps/aristides/reason")
W = R / "swesmith_reasoning_pool"; OUT = R / "swesmith_reason_tensors"
LOGS = Path("/home/toolkit/PipelineRL-SWE/logs/run_evaluation")
ROUTES = ["oss20lo", "oss20md", "dsv4f", "oss120md", "oss120hi"]

ids = json.load(open(W / "full" / "ids.json"))
z = np.load(R / "swesmith_costhead" / "scout_prefill.npz", allow_pickle=True)
have = {str(p) for p in z["problem_ids"]}
ids = [i for i in ids if i in have]
n, M = len(ids), len(ROUTES)
fo = np.zeros((n, M, 1), bool); va = np.zeros((n, M, 1), bool)
pt = np.zeros((n, M, 1), np.float32); ct = np.zeros((n, M, 1), np.float32)
for m, r in enumerate(ROUTES):
    tok = {json.loads(l)["instance_id"]: json.loads(l) for l in open(W / "full" / f"predictions_{r}_d0.jsonl")}
    for i, iid in enumerate(ids):
        if iid not in tok:
            continue
        va[i, m, 0] = True; pt[i, m, 0] = tok[iid]["prompt_tokens"]; ct[i, m, 0] = tok[iid]["completion_tokens"]
        rep = LOGS / f"swesmith_reason_v2_predictions_{r}_d0" / iid / "report.json"
        fo[i, m, 0] = rep.exists() and bool(json.loads(rep.read_text()).get("resolved"))
rng = np.random.default_rng(0); perm = rng.permutation(n); ntr, ncal = int(0.55 * n), int(0.15 * n)
split = {"train_problem_ids": [ids[k] for k in perm[:ntr]], "calibration_problem_ids": [ids[k] for k in perm[ntr:ntr + ncal]],
         "test_problem_ids": [ids[k] for k in perm[ntr + ncal:]]}
OUT.mkdir(parents=True, exist_ok=True)
np.savez(OUT / "tensors.npz", final_outcome=fo, execution_outcome=fo, weak_verifier_outcome=fo, valid=va,
         prompt_tokens=pt, completion_tokens=ct, problem_ids=np.array(ids), model_slots=np.array(ROUTES), schema_version=3)
(OUT / "split_manifest.json").write_text(json.dumps(split))
with open(OUT / "problems.jsonl", "w") as f:
    for iid in ids:
        f.write(json.dumps({"problem_id": iid}) + "\n")
print(f"{n} problems -> {OUT}; solve rate per route {fo[:, :, 0].mean(0).round(3)}; valid {va.mean():.3f}; "
      f"split {ntr}/{ncal}/{n - ntr - ncal}; mean in/out tokens {pt.mean((0, 2)).round(0)} / {ct.mean((0, 2)).round(0)}")
