"""Few-label results relative to the median fitted on the same n problems (launch_nov06_labels_samen.sh). Mean over the ten training sets; intervals pool the
paired resamples, as draws_aggregate.py. vs_full is the old baseline, kept to check the rerun against draws_summary_pinned.json.
Usage: python labels_samen_aggregate.py -> labels_samen_summary_pinned.json
"""
import json
from pathlib import Path
import numpy as np
D = Path("/mnt/llmd/results/exps/aristides/reason/reason_pinned_logs/draws/labels_samen"); OLD = json.load(open(Path(__file__).parent / "draws_summary_pinned.json"))["labels"]; out = {}; worst = 0
for P in "LCB Omni MMLU-Pro SuperGPQA BBEH".split():
    p = P.lower().replace("-", ""); runs = [json.load(open(D / f"cost_generalization_labels_{p}_pinned_d{s}.json")) for s in range(10)]; out[P] = {}
    for n in runs[0]["by_n"]:
        row = out[P][n] = {}
        for a in runs[0]["by_n"][n]:
            if a.startswith("_"):
                continue
            v = np.array([r["by_n"][n][a]["vs_same_n"] for r in runs]) * 100; f = np.array([r["by_n"][n][a]["vs_median"] for r in runs]) * 100
            row[a] = dict(vs_same_n=float(v.mean()), sd=float(v.std()), vs_full=float(f.mean()))
            worst = max(worst, abs(row[a]["vs_full"] - OLD[P][n][a]["vs_median"]))
            if a != "ours":
                b = np.concatenate([r["by_n"][n][a]["ours_minus_it_same_n_bs"] for r in runs]); row[a]["ours_minus_it"] = [float(row["ours"]["vs_same_n"] - row[a]["vs_same_n"]), *map(float, np.nanpercentile(b, [2.5, 97.5]))]
json.dump(out, open(Path(__file__).parent / "labels_samen_summary_pinned.json", "w"), indent=1)
print(f"largest change on the old baseline: {worst:.2f} points")
for n in ("20", "50", "100"):
    print(f"n={n}")
    for a in "ours fromsuccess gbm mixllm knn zerorouter median".split():
        print(f"  {a:12s}" + "  ".join(f"{P} {out[P][n][a]['vs_same_n']:6.1f}" + ("*" if a not in ("ours", "fromsuccess", "median") and out[P][n][a]["ours_minus_it"][1] > 0 else " ") for P in out))
print("ours at all problems", {P: round(out[P][max(out[P], key=int)]["ours"]["vs_same_n"], 1) for P in out})
