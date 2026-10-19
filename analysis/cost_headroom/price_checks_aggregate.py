import json, glob, numpy as np
D = "/mnt/llmd/results/exps/aristides/reason/reason_pinned_logs/pricechecks"
base = json.load(open("/home/toolkit/PipelineRL-SWE/analysis/cost_headroom/draws_summary_pinned.json"))["draws"]
ARMS = ["mean", "gbm", "mixllm", "zerorouter", "fromsuccess", "zr-bins", "oracle"]
for tag in ("ratetrain", "nobilled"):
    for pool in ("LCB", "Omni", "MMLU-Pro"):
        S = {}
        for f in glob.glob(f"{D}/tmlr_strengthen_draws_{pool.lower().replace('-', '')}_*_{tag}.json"): S.update(json.load(open(f))["seeds"])
        ks = [str(s) for s in range(10) if str(s) in S]
        if not ks: continue
        def agg(a, k):
            pts = np.mean([S[s][a][k] for s in ks]); bs = np.concatenate([S[s][a][k + "_bs"] for s in ks]); per = [np.percentile(S[s][a][k + "_bs"], [2.5, 97.5]) for s in ks]
            return pts, np.percentile(bs, 2.5), np.percentile(bs, 97.5), sum(p[0] > 0 for p in per)
        o = agg("ours", "vs_median"); b = base[pool]["ours"]["vs_median"]
        line = f"{tag:<9} {pool:<8} n={len(ks)} ours vs median {o[0]:.1f} [{o[1]:.1f}, {o[2]:.1f}] (pinned {b[0]:.1f}, diff {o[0]-b[0]:+.1f})"
        for a in ARMS:
            v = agg(a, "ours_vs_it"); bb = base[pool][a]["ours_vs_it"][0] if a in base[pool] else float("nan")
            line += f" | {a} {v[0]:.1f} [{v[1]:.1f},{v[2]:.1f}] ({v[3]}) Δ{v[0]-bb:+.1f}"
        print(line)
