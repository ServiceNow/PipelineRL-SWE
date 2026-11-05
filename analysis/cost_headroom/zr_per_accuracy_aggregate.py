"""Saving of ours vs ZeroRouter as published at each of saved()'s 12 matched accuracies, mean over the ten training sets (fig:overview).
Reads the rerun launch_nov05_zr_peracc.sh wrote to zrfaithful/peracc -> zr_per_accuracy_savings_pinned.json
"""
import glob, json
from pathlib import Path
import numpy as np
D = "/mnt/llmd/results/exps/aristides/reason/reason_pinned_logs/zrfaithful/peracc/"; out = {}
for p in "LCB Omni MMLU-Pro AIME APPS BCB CC SuperGPQA BBEH".split():
    S = {}
    for f in glob.glob(D + f"zr_faithful_draws_{p.lower().replace('-', '')}_*_pinned.json"):
        S.update(json.load(open(f))["seeds"])
    ks = sorted(S, key=int); V = np.array([S[k]["zr_pub"]["ours_vs_it_at"]["saved"] for k in ks]); A = np.array([S[k]["zr_pub"]["ours_vs_it_at"]["accuracy"] for k in ks])
    out[p] = dict(accuracy=A.mean(0).tolist(), saved=V.mean(0).tolist(), saved_by_set=V.tolist(), band_mean=float(np.mean([S[k]["zr_pub"]["ours_vs_it"] for k in ks])))
json.dump(out, open(Path(__file__).parent / "zr_per_accuracy_savings_pinned.json", "w"), indent=1)
