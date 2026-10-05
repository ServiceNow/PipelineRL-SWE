"""Live run (NEW_PATH 4.A.52), step 3: route 1,000 never-used MMLU-Pro problems with the FROZEN readouts, then call only the chosen
routes live (OpenRouter, unpinned, as deployed) and grade them. Nothing is fitted on live data.
  readouts   success: activation_content_preds.py --rich --select-C on the original train/cal split (same command as the fresh set);
             cost: the archived train-only RidgeCV heads (paper_cost_heads.joblib). Anchor checks: both reproduce the archived FRESH
             predictions (success_preds.jsonl / paper_cost_preds.jsonl) before any live call is made.
  pricing    effective billed $/token (billed.py RATE); input tokens are not known before the call, so each route's prompt tokens are
             predicted from the Qwen prompt length (per-route least squares on the fresh set). Median arm = same success, median TRAIN
             output length (the paper rule).
  policies   for each accuracy target (.65/.75/.85) and arm, the cheapest V reaching the target on ORIGINAL calibration (the Table 2 /
             deploy_matched.py rule); both arms route every live problem once.
  calls      the union of chosen (problem, route) pairs; one call each; billed usage_cost recorded; spend guard (--budget-usd).
Outputs in /mnt/llmd/results/exps/aristides/reason/live_run_20261005/: decisions.json, calls.jsonl, anchor.json.
Usage: python live_route.py [--budget-usd 8] [--dry-run]
"""
import argparse, asyncio, glob, json, shutil, subprocess, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; CH = HERE.parent; REPO = HERE.parents[2]
sys.path.insert(0, str(CH)); sys.path.insert(0, str(REPO / "pipelinerl/swe/scripts/math_pool"))
from carrot_compare import POOLS, read_predictions
from decompose import MK, R
from billed import RATE
from baseline_cost_heads import rich

OUT = R / "live_run_20261005"; F = R / "expanded_eval_20261001" / "mmlupro"; OLD = R / POOLS["MMLU-Pro"][0]
TARGETS = (0.65, 0.75, 0.85); VALUES = np.geomspace(1e-7, 1, 400)                 # deploy_matched.py grid
rate_of = lambda s: RATE["oss120" if "120" in s else ("oss20" if s.startswith("oss20") else "dsv4f")]


def build_inputs():
    """Combined features (original+fresh+live) and a tensors dir with live rows appended as invalid (never used in fitting)."""
    live = np.load(OUT / "prefill.npz", allow_pickle=True); comb = np.load(F / "prefill_combined.npz", allow_pickle=True)
    for k in ("layers", "model", "system_prompt", "user_suffix"):
        if not np.array_equal(live[k], comb[k]):   # checked before the anchor rows are dropped
            raise SystemExit(f"live feature metadata differs from the frozen features: {k}")
    # feature anchor: 32 fresh prompts re-extracted in the same job must match their frozen features (then dropped)
    anc = set(json.load(open(OUT / "anchor_ids.json"))); allid = [str(p) for p in live["problem_ids"]]; cid = {str(p): i for i, p in enumerate(comb["problem_ids"])}
    ai = [i for i, p in enumerate(allid) if p in anc]; ci = [cid[allid[i]] for i in ai]
    rel = max(float(np.linalg.norm((live[k][ai] - comb[k][ci]).astype(float)) / np.linalg.norm(comb[k][ci].astype(float))) for k in ("mean", "last"))
    print(f"feature anchor: {len(ai)} fresh prompts re-extracted, relative RMS deviation {rel:.2e} (tolerance .05, as verify_paper_prefill_anchor.py)", flush=True)
    (OUT / "feature_anchor.json").write_text(json.dumps(dict(n=len(ai), max_rel_dev=rel)))
    if len(ai) != 32 or rel > .05:
        raise SystemExit("live features do not reproduce the frozen fresh features; refusing to route")
    keep = [i for i, p in enumerate(allid) if p not in anc]; lid = [allid[i] for i in keep]
    live = {k: live[k][keep] for k in ("mean", "last")}
    np.savez(OUT / "features_all.npz", problem_ids=np.array([str(p) for p in comb["problem_ids"]] + lid),
             mean=np.concatenate([comb["mean"], live["mean"]]), last=np.concatenate([comb["last"], live["last"]]), layers=comb["layers"])
    t = np.load(F / "tensors.npz", allow_pickle=True); n = len(lid); T = OUT / "tensors_live"; T.mkdir(exist_ok=True)
    arr = {k: t[k] for k in t.files}
    for k in ("final_outcome", "execution_outcome", "weak_verifier_outcome", "valid", "prompt_tokens", "completion_tokens"):
        arr[k] = np.concatenate([t[k], np.zeros((n,) + t[k].shape[1:], t[k].dtype)])
    arr["problem_ids"] = np.array(list(map(str, t["problem_ids"])) + lid)
    np.savez(T / "tensors.npz", **arr); shutil.copy(F / "split_manifest.json", T / "split_manifest.json")
    tasks = [json.loads(l) for l in open(OUT / "tasks.jsonl")]
    (T / "problems.jsonl").write_text((F / "problems.jsonl").read_text() + "".join(
        json.dumps({"problem_id": x["problem_id"], "difficulty": 0.0, "platform": "mmlupro", "problem_statement": x["problem"]}) + "\n" for x in tasks))
    return lid, tasks


def readouts(lid):
    import joblib
    sp = OUT / "success_all.jsonl"
    if not sp.exists():
        subprocess.run([sys.executable, str(REPO / "pipelinerl/swe/scripts/livecodebench/activation_content_preds.py"), "--activations",
                        str(OUT / "features_all.npz"), "--rich", "--tensors-dir", str(OUT / "tensors_live"), "--select-C", "--out", str(sp)],
                       check=True, cwd=REPO, stdout=subprocess.DEVNULL)
    t = np.load(F / "tensors.npz", allow_pickle=True); slots = list(map(str, t["model_slots"])); fids = list(map(str, t["problem_ids"]))
    n_old = len(np.load(OLD / "tensors.npz", allow_pickle=True)["problem_ids"]); fresh = fids[n_old:]
    P_live = read_predictions(sp, lid, "p_successes", len(slots))
    d_s = np.abs(read_predictions(sp, fresh, "p_successes", len(slots)) - read_predictions(F / "success_preds.jsonl", fresh, "p_successes", len(slots))).max()
    H = joblib.load(F / "paper_cost_heads.joblib"); assert H["slots"] == slots
    X = rich(OUT / "features_all.npz", fresh + lid)
    tok = np.stack([np.exp(h["model"].predict(h["scaler"].transform(X))) * h["smear"] * h["level"] for h in H["heads"]], 1)
    tok_fresh, tok_live = tok[:len(fresh)], tok[len(fresh):]
    I_f = np.where(t["valid"], t["prompt_tokens"], 0).sum(2)[n_old:] / np.maximum(t["valid"].sum(2), 1)[n_old:]
    arch = read_predictions(F / "paper_cost_preds.jsonl", fresh, "expected_costs", len(slots))
    mk = dict(MK); mk.update(json.loads((OLD / "prices.json").read_text()) if (OLD / "prices.json").exists() else {})   # as reconstruct_paper_cost_heads.py
    asg = np.array([[mk[s][0] / 1e6, mk[s][1] / 1e6] for s in slots])
    d_c = float(np.nanmax(np.abs((I_f * asg[:, 0] + tok_fresh * asg[:, 1]) / arch - 1)))
    anchor = dict(success_max_abs_diff=float(d_s), cost_max_rel_diff=d_c, n_fresh=len(fresh))
    (OUT / "anchor.json").write_text(json.dumps(anchor, indent=1)); print("anchor", anchor, flush=True)
    if d_s > 1e-3 or d_c > 1e-4:          # success: lbfgs numerics on the larger matrix give ~4e-4 (set before any live call)
        raise SystemExit("readouts do not reproduce the archived fresh predictions; refusing to make live calls")
    return slots, P_live, tok_live, I_f, n_old


def qwen_len(prompts):
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    return np.array([len(tk(p)["input_ids"]) for p in prompts], float)


def select_V(slots):
    """Table 2 rule (deploy_matched.py): cheapest V reaching each target on ORIGINAL calibration, billed prices, both arms."""
    t = np.load(F / "tensors.npz", allow_pickle=True); ids = list(map(str, t["problem_ids"])); idx = {p: i for i, p in enumerate(ids)}
    sp = json.loads((OLD / "split_manifest.json").read_text()); tr, ca = [np.array([idx[str(p)] for p in sp[k + "_problem_ids"]]) for k in ("train", "calibration")]
    n_old = len(np.load(OLD / "tensors.npz", allow_pickle=True)["problem_ids"]); v = t["valid"].astype(bool); cnt = np.maximum(v.sum(2), 1)
    q = np.where(v, t["final_outcome"], 0).sum(2) / cnt; L = np.where(v, t["completion_tokens"], 0).sum(2) / cnt; I = np.where(v, t["prompt_tokens"], 0).sum(2) / cnt
    rates = np.array([rate_of(s) for s in slots]); paid = I * rates[:, 0] + L * rates[:, 1]
    learned = read_predictions(OLD / POOLS["MMLU-Pro"][1], ids[:n_old], "expected_costs", len(slots))
    p = read_predictions(OLD / "content_preds.jsonl", ids[:n_old], "p_successes", len(slots))
    asg = np.array([[MK[s][0] / 1e6, MK[s][1] / 1e6] for s in slots]); tok = np.maximum((learned - I[:n_old] * asg[:, 0]) / asg[:, 1], 1)
    med = np.array([np.median(t["completion_tokens"][tr, k][v[tr, k]]) for k in range(len(slots))])
    cL = I[:n_old] * rates[:, 0] + tok * rates[:, 1]; cM = I[:n_old] * rates[:, 0] + med[None] * rates[:, 1]

    def pick(pp, c, qq, pd, target):
        ch = (VALUES[:, None, None] * pp[None] - c[None]).argmax(2); r = np.arange(len(pp))[None]
        acc, s = qq[r, ch].mean(1), pd[r, ch].mean(1); ok = np.flatnonzero(acc >= target - 1e-12)
        return None if not len(ok) else float(VALUES[ok[np.argmin(s[ok])]])
    return {tg: dict(ours=pick(p[ca], cL[ca], q[ca], paid[ca], tg), median=pick(p[ca], cM[ca], q[ca], paid[ca], tg)) for tg in TARGETS}, med


async def call_all(pairs, tasks, budget):
    import aiohttp
    from collect_math_pool import call, _grade_task
    key = Path("/home/toolkit/.secrets/openrouter_api_key").read_text().strip(); sem = asyncio.Semaphore(256)
    path = OUT / "calls.jsonl"; done = set(); spent = {"usd": 0.0, "stop": False}
    if path.exists():
        for l in open(path):
            r = json.loads(l); spent["usd"] += float(r.get("usage_cost") or 0)
            if r.get("finish_reason") != "error": done.add((r["problem_id"], r["route_label"]))
    todo = [x for x in pairs if x not in done]; print(f"{len(pairs)} live calls, {len(todo)} to make, ${spent['usd']:.2f} already spent", flush=True)
    fh = open(path, "a"); T = {x["problem_id"]: x for x in tasks}
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as session:
        async def one(pid, route):
            if spent["stop"]:
                return
            r = await call(session, key, route, T[pid]["prompt"], sem, 64000)
            r.update(problem_id=pid, route_label=route, resolved=_grade_task(T[pid], r) if r["finish_reason"] != "error" else False)
            r.pop("reasoning", None); fh.write(json.dumps(r) + "\n"); fh.flush()
            spent["usd"] += float(r.get("usage_cost") or 0)
            if spent["usd"] > budget:
                spent["stop"] = True
        await asyncio.gather(*[one(*x) for x in todo])
    fh.close(); print(f"spent ${spent['usd']:.3f} (budget stop {spent['stop']})", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--budget-usd", type=float, default=8.0); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    lid, tasks = build_inputs(); slots, P, tok, I_f, n_old = readouts(lid)
    # input tokens per route from the Qwen prompt length (fit on the fresh set's prompts)
    fresh_tasks = {json.loads(l)["problem_id"]: json.loads(l) for l in open(CH / "expansion_20261001" / "mmlupro_tasks.jsonl")}
    t = np.load(F / "tensors.npz", allow_pickle=True); fids = list(map(str, t["problem_ids"]))[n_old:]
    ql_f = qwen_len([fresh_tasks[p]["prompt"] for p in fids]); ql_l = qwen_len([x["prompt"] for x in tasks])
    I_live = np.stack([np.polyval(np.polyfit(ql_f, I_f[:, k], 1), ql_l) for k in range(len(slots))], 1)
    Vs, med = select_V(slots)
    rates = np.array([rate_of(s) for s in slots]); cL = I_live * rates[:, 0] + tok * rates[:, 1]; cM = I_live * rates[:, 0] + med[None] * rates[:, 1]
    dec = {}
    for tg, vv in Vs.items():
        dec[str(tg)] = dict(V=vv, ours=[slots[k] for k in (vv["ours"] * P - cL).argmax(1)], median=[slots[k] for k in (vv["median"] * P - cM).argmax(1)])
    pairs = sorted({(pid, d[arm][i]) for d in dec.values() for arm in ("ours", "median") for i, pid in enumerate(lid)})
    exp_cost = sum(cL[lid.index(p), slots.index(s)] for p, s in pairs)
    json.dump(dict(problem_ids=lid, slots=slots, decisions=dec, predicted_cost_of_union_usd=float(exp_cost), n_calls=len(pairs),
                   p_success=P.tolist(), tokens_pred=tok.tolist(), input_tokens_pred=I_live.tolist(), median_tokens=med.tolist()),
              open(OUT / "decisions.json", "w"))
    print(json.dumps({tg: dict(V=d["V"], ours={s: d["ours"].count(s) for s in slots}, median={s: d["median"].count(s) for s in slots}) for tg, d in dec.items()}), flush=True)
    print(f"union {len(pairs)} calls, predicted cost ${exp_cost:.2f}", flush=True)
    if not a.dry_run:
        asyncio.run(call_all(pairs, tasks, a.budget_usd))


if __name__ == "__main__":
    main()
