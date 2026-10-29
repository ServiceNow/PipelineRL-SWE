"""Prefill-size scale table (NEW_PATH 4.A.122): pools prefill_draws.py over the 10 training sets for Qwen3 hybrid 0.6B-14B (LCB, Omni-MATH,
MMLU-Pro; the Omni-MATH / MMLU-Pro 0.6B / 1.7B / 4B runs of tab:size are reused from prefilldraws/) and adds each prefill's measured cost
(prefill_overhead_<tag>.json: mean prompt tokens / batch-16 throughput on one H100 at $2 per H100-hour). Per row: mean test length R2 and
success AUC; cost saved relative to median pricing with the row's own success probes (the prefill's cost cancels: both pay it); prefill
$ per 1,000 queries and its share of our router's spend; ours vs the row's router at matched accuracy, without and with each router's
prefill priced. Spend level: our router's spend per 1,000 queries at the middle of the shared accuracy band (route_shares_pinned.json,
one training set), the operating point of Figure 1. Intervals: 95% percentiles of the bootstrap replicates pooled over training sets.
Usage: python prefill_scale_aggregate.py
"""
import glob, json
from pathlib import Path
import numpy as np

L = Path("/mnt/llmd/results/exps/aristides/reason/reason_pinned_logs"); HERE = Path(__file__).parent
TAGS = [("q06b", "Qwen3-0.6B"), ("q17b", "Qwen3-1.7B"), ("q4bhyb", "Qwen3-4B"), ("q8b", "Qwen3-8B"), ("q14b", "Qwen3-14B")]
POOLS = [("LCB", "lcb", "LCB"), ("Omni", "omni", "Omni500"), ("MMLU-Pro", "mmlupro", "MMLU-Pro")]
SH = json.load(open(HERE / "route_shares_pinned.json")); H100 = 2.0


def usd(tag, key):
    r = json.load(open(L / "prefillscale" / f"prefill_overhead_{tag}.json"))[key]
    return r["tokens_mean"] * 1000 / r["batched_tokens_per_s"] * H100 / 3600


def seeds(p, tag):
    out = {}
    for d in ("prefillscale", "prefilldraws"):
        for f in sorted(glob.glob(str(L / d / f"prefill_draws_{p}_{tag}_*_pinned.json"))):
            out.update(json.load(open(f))["seeds"])
        if len(out) == 10:
            return out
    raise SystemExit(f"{p} {tag}: {len(out)} training sets")


res = {}
for P, p, key in POOLS:
    So = SH[P]["ours"]["cost_per_1k"]; po = usd("paper", key); rows = {}; ours_vm = []
    for tag, name in TAGS:
        S = seeds(p, tag); v = list(S.values()); pi = usd(tag, key)
        vm = np.array([x["vs_median_bs"] for x in v]).ravel(); dd = np.array([x["ours_vs_it_bs"] for x in v]).ravel()
        d_pt = np.mean([x["ours_vs_it"] for x in v]); ours_vm += [x["ours_vs_median"] for x in v]
        net = lambda d: (1 - (So + po) / (So / (1 - d / 100) + pi)) * 100       # ours vs it, each router paying its own prefill
        rows[name] = dict(r2=float(np.mean([x["r2"] for x in v])), auc=float(np.mean([x["auc"] for x in v])),
                          vs_median=[float(np.mean([x["vs_median"] for x in v])), *map(float, np.nanpercentile(vm, [2.5, 97.5]))],
                          prefill_usd_per_1k=pi, prefill_share_of_spend=pi / (So / (1 - d_pt / 100) + pi) * 100,
                          ours_vs_it=[float(d_pt), *map(float, np.nanpercentile(dd, [2.5, 97.5]))],
                          ours_vs_it_priced=[float(net(d_pt)), *map(float, np.nanpercentile(net(dd), [2.5, 97.5]))],
                          ahead_sets=int(sum(np.nanpercentile(x["ours_vs_it_bs"], 2.5) > 0 for x in v)),
                          behind_sets=int(sum(np.nanpercentile(x["ours_vs_it_bs"], 97.5) < 0 for x in v)))
    x = v[0]
    rows["Qwen3-4B-2507 (ours)"] = dict(r2=float(np.mean([x["r2_ours"] for x in v])), auc=float(np.mean([x["auc_ours"] for x in v])),
                                        vs_median=[float(np.mean(ours_vm[:10]))], prefill_usd_per_1k=po, prefill_share_of_spend=po / (So + po) * 100)
    res[P] = dict(spend_ours_per_1k=So, rows=rows)
    print(f"\n{P} (our router's spend at Figure 1's operating point: ${So:.2f} per 1k queries)")
    print(f"  {'prefill':22s} {'R2':>5s} {'AUC':>5s} {'vs median':>22s} {'$/1k':>7s} {'share':>6s} {'ours vs it':>22s} {'ours vs it, priced':>22s} ahead/behind")
    for name, r in rows.items():
        f = lambda a: f"{a[0]:5.1f}" + (f" [{a[1]:5.1f}, {a[2]:5.1f}]" if len(a) > 1 else " " * 15)
        print(f"  {name:22s} {r['r2']:5.2f} {r['auc']:5.3f} {f(r['vs_median']):>22s} {r['prefill_usd_per_1k']:7.4f} {r['prefill_share_of_spend']:5.1f}% "
              + (f"{f(r['ours_vs_it']):>22s} {f(r['ours_vs_it_priced']):>22s} {r['ahead_sets']}/{r['behind_sets']}" if "ours_vs_it" in r else ""))
json.dump(res, open(HERE / "prefill_scale_summary_pinned.json", "w"), indent=1)
