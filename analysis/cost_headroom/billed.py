"""Effective billed $/token per model from the fresh collection's usage_cost (least squares); shared by the fresh-set analyses.
RATE_FIT=train (NEW_PATH 4.A.88): fit on TRAINING-split calls only, never on a test problem's calls. deepseek-v4-flash (pinned
StreamLake) from the training calls of the main collection (Omni-MATH / MMLU-Pro / AIME, same period as the test calls); gpt-oss
from the SuperGPQA / BIG-Bench Extra Hard training calls (the only training calls with a billed cost). RATE_POOL=SuperGPQA|BBEH:
that pool's own training calls for every route (its deepseek-v4-flash calls were billed at a later, higher StreamLake price)."""
import glob, json, collections, os
import numpy as np
from decompose import R

fam = lambda s: "oss120" if "120" in s else ("oss20" if s.startswith("oss20") else "dsv4f")
fit = lambda rows: np.linalg.lstsq(np.array([[a, b] for a, b, _ in rows], float), np.array([c for *_, c in rows]), rcond=None)[0]


def billed_rates():
    by = collections.defaultdict(list); byp = collections.defaultdict(list)
    for f in glob.glob(f"{R}/math_expand_20261001/*/*.jsonl"):
        for l in open(f):
            try: r = json.loads(l)
            except Exception: continue
            if r.get("usage_cost") is None or r.get("finish_reason") == "error": continue
            m = fam(r["route_label"])
            x = (r["prompt_tokens"], r["completion_tokens"], r["usage_cost"]); by[m].append(x); byp[(m, r["provider"])].append(x)
    return {m: fit(v) for m, v in by.items()}, {k: fit(v) for k, v in byp.items() if len(v) >= 30}


def train_rates(pool=None):
    RP, RR = "/mnt/llmd/results/exps/aristides/reason_pinned", "/mnt/llmd/results/exps/aristides/reason"
    main = [(f"{RP}/math_pool_pinned/{ds}", f"{RP}/{t}") for ds, t in (("omni500", "omni500_tensors"), ("mmlupro", "mmlupro_tensors"), ("aime", "aime_tensors"))]
    hetero = {"SuperGPQA": (f"{RR}/math_pool/supergpqa", f"{RR}/supergpqa_tensors"), "BBEH": (f"{RR}/math_pool/bbeh", f"{RR}/bbeh_tensors")}
    src = {"dsv4f": main, "oss20": list(hetero.values()), "oss120": list(hetero.values())}
    if pool in hetero:
        src = {m: [hetero[pool]] for m in src}
    out = {}
    for m, dirs in src.items():
        rows = []
        for d, t in dirs:
            ids = set(map(str, json.load(open(f"{t}/split_manifest.json"))["train_problem_ids"]))
            for f in glob.glob(f"{d}/*_d*.jsonl"):
                for l in open(f):
                    try: r = json.loads(l)
                    except Exception: continue
                    if r.get("usage_cost") is None or r.get("finish_reason") == "error" or str(r.get("problem_id")) not in ids: continue
                    if fam(r.get("route_label") or "") != m or (m == "dsv4f" and r.get("provider") != "StreamLake"): continue
                    rows.append((r["prompt_tokens"], r["completion_tokens"], r["usage_cost"]))
        out[m] = fit(rows)
        print(f"[billed] RATE_FIT=train{' ' + pool if pool else ''}: {m} {len(rows)} training calls -> $/M in {out[m][0]*1e6:.4f} out {out[m][1]*1e6:.4f}", flush=True)
    return out


RATE, PRATE = billed_rates()
if os.environ.get("RATE_FIT") == "train":
    RATE = train_rates(os.environ.get("RATE_POOL") or None)
