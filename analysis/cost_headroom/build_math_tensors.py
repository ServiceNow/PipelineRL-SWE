"""Math-pool dataset (Omni-MATH-500 by default; usage: build_math_tensors.py <dataset>) (math_pool/omni500/<route>_d<k>.jsonl) -> the tensors format decompose.py reads.
valid = a draw that returned (finish_reason != error); final_outcome = math_verify-graded `resolved`.
Split: seed-0 permutation, 55 / 15 / 30 train / calibration / test (as SWE-Smith). problems.jsonl keeps the
difficulty rating and statement.
"""
import json, sys
from pathlib import Path
import numpy as np

DS = sys.argv[1] if len(sys.argv) > 1 else "omni500"          # math_pool/<DS>/ -> <DS>_tensors
R = Path("/mnt/llmd/results/exps/aristides/reason"); P = R / "math_pool" / DS; OUT = R / f"{DS}_tensors"
DRAWS = {"oss20lo": 4, "oss20md": 3, "dsv4f": 3, "oss120md": 2, "oss120hi": 2}
probs = [json.loads(l) for l in open(P / "problems.jsonl")]
ids = [p["problem_id"] for p in probs]; pi = {p: i for i, p in enumerate(ids)}
n, M, K = len(ids), len(DRAWS), max(DRAWS.values())
fo = np.zeros((n, M, K), bool); va = np.zeros((n, M, K), bool); pt = np.zeros((n, M, K), np.float32); ct = np.zeros((n, M, K), np.float32)
for m, (r, k) in enumerate(DRAWS.items()):
    for d in range(k):
        f = P / f"{r}_d{d}.jsonl"
        if not f.exists():
            continue
        for l in open(f):
            x = json.loads(l)
            if x.get("finish_reason") == "error" or x["problem_id"] not in pi:
                continue
            i = pi[x["problem_id"]]
            va[i, m, d] = True; fo[i, m, d] = bool(x["resolved"])
            pt[i, m, d] = x["prompt_tokens"]; ct[i, m, d] = x["completion_tokens"]
rng = np.random.default_rng(0); perm = rng.permutation(n); ntr, ncal = int(0.55 * n), int(0.15 * n)
split = {"train_problem_ids": [ids[k] for k in perm[:ntr]], "calibration_problem_ids": [ids[k] for k in perm[ntr:ntr + ncal]],
         "test_problem_ids": [ids[k] for k in perm[ntr + ncal:]]}
OUT.mkdir(parents=True, exist_ok=True)
np.savez(OUT / "tensors.npz", final_outcome=fo, execution_outcome=fo, weak_verifier_outcome=fo, valid=va, prompt_tokens=pt,
         completion_tokens=ct, problem_ids=np.array(ids), model_slots=np.array(list(DRAWS)), schema_version=3)
(OUT / "split_manifest.json").write_text(json.dumps(split))
with open(OUT / "problems.jsonl", "w") as f:
    for p in probs:
        f.write(json.dumps({"problem_id": p["problem_id"], "difficulty": p["difficulty"], "platform": DS,
                            "problem_statement": p["problem_statement"]}) + "\n")
acc = (fo & va).sum((0, 2)) / np.maximum(va.sum((0, 2)), 1)
print(f"{n} problems -> {OUT}; valid draws per route {va.sum((0, 2))}; accuracy per route {acc.round(3)}; "
      f"mean output per route {(np.where(va, ct, 0).sum((0, 2)) / np.maximum(va.sum((0, 2)), 1)).round(0)}")
