"""Export the paper's data into the release layout of github.com/ServiceNow/scout-before-you-route.

Internal tool: reads the pinned results root on /mnt and writes a self-contained data tree with NO prompt or response text.
Per dataset (data/<key>/):
  calls.npz          per-call numbers as the analysis reads them: outcome, valid, prompt_tokens, completion_tokens
                     [problem, candidate, sample], problem_ids, candidates (exact copy of the pinned tensors.npz fields used)
  calls.csv.gz       the same, one row per call (for people who do not use the code)
  splits.json        train / calibration / test problem IDs (the pinned split_manifest.json)
  problems.csv       problem_id, row index, benchmark metadata, sha256 of our formatted problem statement (the text loader's check)
  prompt_format.json the scout prompt template and layers recorded in our activation file (no problem text)
  expanded/          Omni-MATH and MMLU-Pro only: the expanded evaluation set (calls.npz, splits.json, problems.csv)
Raw API rows (data/rows/<collection>/...jsonl): every numeric / categorical field the scripts read (tokens, billed cost, provider,
finish reason, grade), text fields replaced by output_nonempty. Usage: python export_data.py <out_dir>
"""
import csv, gzip, hashlib, json, os, shutil, sys
from pathlib import Path
import numpy as np

P = Path("/mnt/llmd/results/exps/aristides/reason_pinned"); R = Path("/mnt/llmd/results/exps/aristides/reason")
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)

# the user prompt each activation file was extracted from (under R), by dataset key; "<key>/expanded" for the expanded sets
PROMPTS = {"lcb": "pool_probe_prompts.jsonl", "apps": "apps_probe_prompts.jsonl", "bcb": "bcb_probe_prompts.jsonl",
           "cc": "cc_pool/prompts.jsonl", "omni": "omni500_probe_prompts.jsonl", "mmlupro": "mmlupro_probe_prompts.jsonl",
           "aime": "aime_probe_prompts.jsonl", "supergpqa": "supergpqa_probe_prompts.jsonl", "bbeh": "bbeh_probe_prompts.jsonl",
           "swesmith": "swesmith_costhead/prompts.jsonl", "omni/expanded": "prefill_size_20261005/omni500_prompts.jsonl",
           "mmlupro/expanded": "prefill_size_20261005/mmlupro_prompts.jsonl"}
DROP = {"problem_id", "problem_statement", "prompt", "question", "answer"}   # never exported: benchmark text and gold answers
# key: (tensors dir under P, activation file under R or None, expanded dir name or None)
DATASETS = {
    "lcb": ("pool_v2_tensors_5rung", "pv2_scout_prefill_1756715297/scout.npz", None),
    "apps": ("apps_tensors", "apps_probe/instruct.npz", None),
    "bcb": ("bcb_tensors_5r", "bcb_scout_prefill.npz", None),
    "cc": ("cc_tensors", "cc_pool/scout_prefill.npz", None),
    "omni": ("omni500_tensors", "omni500_probe/instruct.npz", "omni500"),
    "mmlupro": ("mmlupro_tensors", "mmlupro_probe/instruct.npz", "mmlupro"),
    "aime": ("aime_tensors", "aime_probe/instruct.npz", None),
    "supergpqa": ("supergpqa_tensors", "supergpqa_probe/instruct.npz", None),
    "bbeh": ("bbeh_tensors", "bbeh_probe/instruct.npz", None),
    "swesmith": ("swesmith_v2_tensors", "swesmith_costhead/scout_prefill.npz", None),
}
FIELDS = ("final_outcome", "valid", "prompt_tokens", "completion_tokens")
# raw API row collections the scripts read (relative to P, falling back to R), and the fields kept
ROWS = ["math_expand_20261001", "math_pool_nonreason", "math_expand_nonreason", "pool_v2_lcb_nonreason",
        "second_family_20261002", "second_family_20261009", "pool_v2_lcb_second_family"]
KEEP = {"problem_id", "route_label", "model", "draw", "dataset", "difficulty", "prompt_tokens", "completion_tokens",
        "finish_reason", "provider", "usage_cost", "resolved", "public_resolved", "_reasoning_effort", "_reasoning_enabled",
        "_generation_temperature", "_lcb_evaluator_commit", "_lcb_dataset_revision"}
TEXT = ("full_output", "content", "code", "reasoning", "thinking_text")
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()


def export_tables(src, dst, act=None, pkey=None):
    dst.mkdir(parents=True, exist_ok=True)
    t = np.load(src / "tensors.npz", allow_pickle=True)
    ids, slots = [str(x) for x in t["problem_ids"]], [str(x) for x in t["model_slots"]]
    np.savez_compressed(dst / "calls.npz", problem_ids=np.array(ids), candidates=np.array(slots), **{k: t[k] for k in FIELDS})
    with gzip.open(dst / "calls.csv.gz", "wt", newline="") as f:
        w = csv.writer(f); w.writerow(["problem_id", "candidate", "sample", "valid", "correct", "prompt_tokens", "completion_tokens"])
        v = t["valid"]
        for i, p in enumerate(ids):
            for m, s in enumerate(slots):
                for k in range(v.shape[2]):
                    if v[i, m, k]:
                        w.writerow([p, s, k, 1, int(t["final_outcome"][i, m, k]), int(t["prompt_tokens"][i, m, k]), int(t["completion_tokens"][i, m, k])])
    shutil.copy(src / "split_manifest.json", dst / "splits.json")
    probs = [json.loads(l) for l in open(src / "problems.jsonl")]
    assert [str(r["problem_id"]) for r in probs] == ids, f"{src}: problems.jsonl order differs from tensors"
    meta = sorted({k for r in probs for k in r if k not in DROP})
    prompts = {json.loads(l)["problem_id"]: json.loads(l)["prompt"] for l in open(R / PROMPTS[pkey])} if pkey in PROMPTS else {}
    with open(dst / "problems.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["problem_id", "row"] + meta + ["statement_sha256", "prompt_sha256"])
        for i, r in enumerate(probs):
            pr = prompts.get(str(r["problem_id"]))
            w.writerow([r["problem_id"], i] + [r.get(k, "") for k in meta] + [sha(str(r["problem_statement"])), sha(pr) if pr is not None else ""])
    if act:
        z = np.load(act if Path(act).is_absolute() else R / act, allow_pickle=True)
        fmt = {k: (z[k].tolist() if k in z.files else None) for k in ("model", "layers", "system_prompt", "user_suffix", "route_label")}
        fmt["source_file"] = Path(act).name
        json.dump(fmt, open(dst / "prompt_format.json", "w"), indent=1)
    return len(ids), slots, int(t["valid"].sum())


def export_rows(name):
    src = P / name if (P / name).exists() else R / name
    n = 0
    for f in sorted(src.rglob("*.jsonl")):
        rel = f.relative_to(src); dst = OUT / "rows" / name / rel; dst.parent.mkdir(parents=True, exist_ok=True)
        with open(f) as fi, open(dst, "w") as fo:
            for l in fi:
                try: r = json.loads(l)
                except Exception: continue
                o = {k: r[k] for k in KEEP if k in r}
                o["error"] = bool(r.get("error")); o["output_nonempty"] = any(str(r.get(k) or "").strip() for k in TEXT)
                fo.write(json.dumps(o) + "\n"); n += 1
    return src, n


if __name__ == "__main__":
    report = {}
    for key, (tdir, act, exp) in DATASETS.items():
        n, slots, nv = export_tables(P / tdir, OUT / key, act, key)
        report[key] = {"problems": n, "candidates": slots, "valid_calls": nv}
        if exp:
            E = P / "expanded_eval_20261001" / exp
            n2, _, nv2 = export_tables(E, OUT / key / "expanded", E / "prefill_combined.npz", key + "/expanded")
            report[key]["expanded"] = {"problems": n2, "valid_calls": nv2}
        print(key, report[key], flush=True)
    for name in ([] if "--tables-only" in sys.argv else ROWS):
        src, n = export_rows(name); report["rows/" + name] = {"source": str(src), "rows": n}; print(name, n, flush=True)
    if "--tables-only" not in sys.argv:
        json.dump(report, open(OUT / "export_report.json", "w"), indent=1)
