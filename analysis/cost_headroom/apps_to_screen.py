"""APPS screen: merge collect_cc_expert's oss20lo_{train,eval}_d0.jsonl into math_pool/apps/oss20lo_d0.jsonl (the format
screen_step1.py reads), attaching each task's tier (introductory 1 / interview 2 / competition 3) as `difficulty`."""
import json
from pathlib import Path
R = Path("/mnt/llmd/results/exps/aristides/reason")
tier = {json.loads(l)["problem_id"]: json.loads(l)["difficulty"] for l in open(R / "apps_tasks.jsonl")}
out = R / "math_pool" / "apps"; out.mkdir(parents=True, exist_ok=True); n = 0
with open(out / "oss20lo_d0.jsonl", "w") as f:
    for part in ("train", "eval"):
        p = R / "apps_pool" / f"oss20lo_{part}.jsonl"
        for l in open(p):
            r = json.loads(l)
            if r.get("finish_reason") == "error":
                continue
            f.write(json.dumps({"problem_id": r["problem_id"], "completion_tokens": r["completion_tokens"], "prompt_tokens": r["prompt_tokens"],
                                "resolved": bool(r["resolved"]), "difficulty": tier.get(r["problem_id"], 0.0), "finish_reason": r["finish_reason"]}) + "\n")
            n += 1
print(f"{n} APPS rows -> {out / 'oss20lo_d0.jsonl'}")
