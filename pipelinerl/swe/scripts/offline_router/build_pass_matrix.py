#!/usr/bin/env python3
"""Track B, P0: one pass-matrix record per instance, shared by LCB (dev) and SWE Verified (evidence).

{benchmark, instance_id, split,
 candidates: [{cid, gen, draw, correct, gen_cost_c}],
 tests:      [{writer, mode, write_cost_c, valid, passes: {cid: bool}}],
 run_cost_c}
- `valid` is the LABEL-FREE check: the test fails on the unpatched repo (SWE). LCB has no unpatched
  program to run, so valid is None there.
- `passes[cid]` for SWE = the reproduction script exits 0 on that patch; for LCB = every case passes.
- Costs in cents at OpenRouter list prices (input and output priced separately); Qwen3-4B is
  self-hosted and priced at gpt-oss-20b as a floor. SWE run cost = ~1.5 s of a 1 vCPU / 1 GiB Daytona
  sandbox per check (NEW_PATH 3.3); LCB runs locally, cost 0.
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np, pandas as pd

P = {  # $/M (in, out)
    "qwen4b": (0.018, 0.09), "oss20": (0.018, 0.09), "oss20lo": (0.018, 0.09), "oss20md": (0.018, 0.09),
    "qwen30": (0.07, 0.28), "qcoder30": (0.07, 0.28), "oss120": (0.15, 0.6), "oss120md": (0.15, 0.6),
    "oss120hi": (0.15, 0.6), "gemini": (0.5, 3.0), "opus": (5.0, 25.0), "dsv4f": (0.04704, 0.09408),
    "devstral": (0.4, 2.0)}
R = Path("/mnt/llmd/results/exps/aristides/reason")


def c(model, tin, tout):
    return (tin * P[model][0] + tout * P[model][1]) / 1e6 * 100


def build_swe(pilot_dir: Path, exec_files: list[str]):
    runs = json.loads((pilot_dir / "patch_runs.json").read_text())
    truth = {k: {json.loads(l)["instance_id"]: bool(json.loads(l)["resolved"]) for l in
                 open(R / f"opus_verified_daytona_eval_{r}/predictions/predictions_opus_verified.results.jsonl")}
             for k, r in runs.items()}
    col = R / "offline_router_swe_bench_train_all_16k_verified_eval_collect_5route_4b_scout_oss20_qwen30_oss120_gemini/collect/eval"
    d5 = pd.concat([pd.read_parquet(f) for f in glob.glob(str(col / "*.parquet"))]).drop_duplicates("problem_id").set_index("problem_id")
    do = pd.concat([pd.read_parquet(f) for f in glob.glob(str(R / "verified_collect_anthropic_claude_opus_5_openrouter_1786656236/collect/eval/*.parquet"))]).drop_duplicates("problem_id").set_index("problem_id")
    order5 = ["qwen4b", "oss20", "qwen30", "oss120", "gemini"]
    wcost = {}
    for f in glob.glob(str(pilot_dir / "scripts" / "scripts_*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if r.get("script"):
                wcost[(r["instance_id"], r["writer"])] = c(r["writer"], r["prompt_tokens"], r["completion_tokens"])
    rows = {}
    for ef in exec_files:
        for l in open(ef):
            r = json.loads(l)
            if r.get("error") or not r["base"]:
                continue
            iid = r["instance_id"]
            rec = rows.setdefault(iid, {"benchmark": "swe_verified", "instance_id": iid, "split": None,
                                        "candidates": [], "tests": [], "run_cost_c": 1.5 * 1.85e-5 * 100})
            if not rec["candidates"]:
                for g in runs:
                    if not r["apply"].get(g):
                        continue
                    if g == "opus":
                        gc = c("opus", do.loc[iid].route_prompt_tokens[1], do.loc[iid].route_output_tokens[1]) if iid in do.index else np.nan
                    else:
                        i = order5.index(g); gc = c(g, d5.loc[iid].route_prompt_tokens[i], d5.loc[iid].route_output_tokens[i]) if iid in d5.index else np.nan
                    rec["candidates"].append({"cid": g, "gen": g, "draw": 0, "correct": truth[g][iid], "gen_cost_c": float(gc)})
            have = {t["writer"] for t in rec["tests"]}
            for w, code in r["base"].items():
                if w in have:
                    continue
                rec["tests"].append({"writer": w, "mode": "issue", "write_cost_c": wcost.get((iid, w), np.nan),
                                     "valid": code not in (0, None),
                                     "passes": {cand["cid"]: r["patches"].get(cand["cid"], {}).get(w) == 0 for cand in rec["candidates"]}})
    return list(rows.values())


def build_lcb(smoke_dir: Path, tensors_dir: Path):
    t = np.load(tensors_dir / "tensors.npz", allow_pickle=True)
    slots = [str(s) for s in t["model_slots"]]; pid = {str(p): i for i, p in enumerate(t["problem_ids"])}
    sp = json.loads((tensors_dir / "split_manifest.json").read_text())
    split = {**{p: "train" for p in sp["train_problem_ids"]}, **{p: "calibration" for p in sp["calibration_problem_ids"]},
             **{p: "test" for p in sp["test_problem_ids"]}}
    wcost = {}
    for f in glob.glob(str(smoke_dir / "suites_*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            wcost[(r["problem_id"], r["writer"])] = c(r["writer"], r.get("prompt_tokens", 0), r.get("completion_tokens", 0))
    rows = {}
    for f in glob.glob(str(smoke_dir / "verdicts" / "verdicts_*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            iid = r["problem_id"]; cid = f'{r["slot"]}:{r["draw_index"]}'
            rec = rows.setdefault(iid, {"benchmark": "lcb", "instance_id": iid, "split": split.get(iid), "candidates": {},
                                        "tests": {}, "run_cost_c": 0.0})
            if cid not in rec["candidates"]:
                i, m, k = pid[iid], slots.index(r["slot"]), r["draw_index"]
                rec["candidates"][cid] = {"cid": cid, "gen": r["slot"], "draw": k, "correct": bool(r["truth"]),
                                          "gen_cost_c": c(r["slot"], float(t["prompt_tokens"][i, m, k]), float(t["completion_tokens"][i, m, k]))}
            tst = rec["tests"].setdefault(r["writer"], {"writer": r["writer"], "mode": "issue",
                                                        "write_cost_c": wcost.get((iid, r["writer"]), np.nan),
                                                        "valid": None, "passes": {}})
            tst["passes"][cid] = bool(r["suite_ok"]) and r["n_case_pass"] == r["n_cases"]
    out = []
    for rec in rows.values():
        rec["candidates"] = list(rec["candidates"].values()); rec["tests"] = list(rec["tests"].values()); out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--swe-exec", default="", help="comma list of SWE exec jsonl files (latest writer rows win)")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    pilot = R / "swe_testwriter_pilot"
    swe = build_swe(pilot, [x for x in (a.swe_exec or str(pilot / "exec_clean.jsonl")).split(",") if x])
    lcb = build_lcb(R / "testwriter_smoke_lcb", R / "pool_v2_tensors_5rung")
    for name, recs in (("swe_verified", swe), ("lcb", lcb)):
        with open(out / f"pass_matrix_{name}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r, default=float) + "\n")
        nc = np.mean([len(r["candidates"]) for r in recs]); nt = np.mean([len(r["tests"]) for r in recs])
        print(f"{name}: {len(recs)} instances, {nc:.1f} candidates and {nt:.1f} tests per instance")


if __name__ == "__main__":
    main()
