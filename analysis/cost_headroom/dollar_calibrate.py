"""Dollar-space calibration test: is the gap between the capture curve and the ideal theory (capture ~ R2) caused by
heads that are calibrated in LOG space while the rule argmax p*V - c spends DOLLARS?

Per route, isotonic regression of the observed per-problem mean cost (market dollars) on the head's predicted cost,
fitted on the CALIBRATION split only (the heads were fitted on train), applied to every problem. Monotone, so it only
fixes the dollar scale of the prediction, never its ranking. Writes <cost file>_dcal.jsonl (+ its overhead file).
Usage: python dollar_calibrate.py <pool> <cost_file> [<cost_file> ...]
"""
import json, shutil, sys, numpy as np
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
sys.path.insert(0, str(Path(__file__).parent))
from decompose import MK, R

name = sys.argv[1]; D = R / name; t = np.load(D / "tensors.npz", allow_pickle=True)
S = [str(s) for s in t["model_slots"]]; pids = [str(p) for p in t["problem_ids"]]; pi = {p: i for i, p in enumerate(pids)}
PT = dict(MK)
if (D / "prices.json").exists():
    PT.update({k: tuple(v) for k, v in json.load(open(D / "prices.json")).items()})
v = t["valid"].astype(bool); n = v.sum(2)
real = np.stack([(t["prompt_tokens"][:, m] * PT[s][0] + t["completion_tokens"][:, m] * PT[s][1]) / 1e6 for m, s in enumerate(S)], 1)
Cr = np.where(n > 0, (real * v).sum(2) / np.maximum(n, 1), np.nan)              # USD, like expected_costs
cal = np.array([pi[str(p)] for p in json.load(open(D / "split_manifest.json"))["calibration_problem_ids"]])
for cf in sys.argv[2:]:
    lc = {json.loads(l)["problem_id"]: json.loads(l)["expected_costs"][:len(S)] for l in open(D / cf)}
    C = np.array([lc[p] for p in pids]); out = C.copy(); msg = []
    for m, s in enumerate(S):
        ok = cal[np.isfinite(Cr[cal, m])]
        iso = IsotonicRegression(out_of_bounds="clip", increasing=True).fit(C[ok, m], Cr[ok, m])
        out[:, m] = np.maximum(iso.predict(C[:, m]), 1e-12)
        msg.append(f"{s} bias before {np.mean(C[ok, m]) / np.mean(Cr[ok, m]):.2f}x")
    new = cf.replace(".jsonl", "_dcal.jsonl")
    with open(D / new, "w") as f:
        for i, p in enumerate(pids):
            f.write(json.dumps({"problem_id": p, "expected_costs": [float(x) for x in out[i]]}) + "\n")
    oh = D / cf.replace(".jsonl", ".overhead.json")
    if oh.exists():
        shutil.copy(oh, D / new.replace(".jsonl", ".overhead.json"))
    print(f"{name} {cf} -> {new}: " + "; ".join(msg))
