"""nebius/SWE-agent-trajectories (80k SWE-agent runs, 3591 tasks, mostly Llama-3.1-70B, ~19 runs per task): per run, the
agent's length and a billing proxy. No dollar cost is released, so:
  steps       number of agent ('ai') turns
  out_chars   agent output characters
  in_chars    characters re-sent as context at every agent turn (cumulative sum of the conversation so far) -- how an
              uncached API bill accrues; cost_proxy = in_chars + 4 * out_chars (output ~4x input price, typical)
The issue statement (first user turn) is saved once per task for the prefill. Output: light.jsonl, issues.jsonl."""
import glob, json, pyarrow.parquet as pq
D = "/mnt/llmd/results/exps/aristides/reason/nebius_swe_agent_traj"
issues = {}; n = 0
with open(f"{D}/light.jsonl", "w") as out:
    for f in sorted(glob.glob(f"{D}/data/*.parquet")):
        pf = pq.ParquetFile(f)
        for b in pf.iter_batches(columns=["instance_id", "model_name", "target", "trajectory"], batch_size=500):
            for r in b.to_pylist():
                tr = r["trajectory"] or []; ctx = 0; in_chars = 0; out_chars = 0; steps = 0
                for s in tr:
                    txt = s.get("text") if s.get("text") not in (None, "None") else (s.get("system_prompt") or "")
                    txt = str(txt or "")
                    if s.get("role") == "ai":
                        steps += 1; in_chars += ctx; out_chars += len(txt)
                    ctx += len(txt)
                    if s.get("role") == "user" and r["instance_id"] not in issues and "issue" in txt[:400].lower():
                        issues[r["instance_id"]] = txt[:12000]
                out.write(json.dumps({"instance_id": r["instance_id"], "model": r["model_name"], "resolved": str(r["target"]) in ("True", "true", "1"),
                                      "steps": steps, "in_chars": in_chars, "out_chars": out_chars,
                                      "cost_proxy": in_chars + 4 * out_chars}) + "\n"); n += 1
with open(f"{D}/issues.jsonl", "w") as f:
    for i, t in issues.items():
        f.write(json.dumps({"problem_id": i, "prompt": t}) + "\n")
print(n, "runs;", len(issues), "issues")
