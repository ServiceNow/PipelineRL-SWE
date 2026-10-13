"""Claude Opus 5.5 as a frontier reference on LCB (launch_oct13_opus_anchor.sh; 100 manifest test problems, one sample, effort high).
On the same 100 problems, for each of the 10 training sets (DRAW=s, readouts refitted as in tmlr_strengthen.py draws): the routers'
frontiers (ours, median pricing; same success readouts), each route's accuracy and cost, and Opus's (billed usage_cost). Reported:
our router's accuracy at Opus's cost per call, its cost at Opus's accuracy (if reached), and the most accurate point on its frontier.
Usage: REASON_ROOT=.../reason_pinned OUT_DIR=... python opus_anchor.py -> opus_anchor_lcb_pinned.json
"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
_src = open(Path(__file__).parent / "cost_generalization.py").read()
exec(_src.split('MODE, POOL = sys.argv[1], sys.argv[2]')[0])
A = Path("/mnt/llmd/results/exps/aristides/reason/lcb_opus55_anchor_20261013")
rows = {}
for f in A.glob("opus55_*_d0.jsonl"):
    for l in open(f):
        r = json.loads(l); rows[r["problem_id"]] = r
ids_o = sorted(rows); acc_o = np.mean([bool(rows[p]["resolved"]) for p in ids_o]); cost_o = np.mean([float(rows[p]["usage_cost"] or 0) for p in ids_o])
tok_o = np.mean([rows[p]["completion_tokens"] for p in ids_o]); think_o = np.mean([rows[p]["thinking_chars"] > 0 for p in ids_o])
print(f"Opus 5.5 on {len(ids_o)} LCB test problems: accuracy {acc_o:.3f}, ${cost_o * 1e3:.2f} per 1k calls, mean output {tok_o:.0f} tokens, "
      f"thinking on {think_o:.0%} of problems", flush=True)
res = {"opus": dict(n=len(ids_o), accuracy=acc_o, usd_per_1k=cost_o * 1e3, mean_output_tokens=tok_o, frac_thinking=think_o), "draws": {}}


def at_cost(H, c):                                   # accuracy of a frontier (list of (cost, acc), cost ascending) at cost c
    cs, as_ = np.array([h[0] for h in H]), np.array([h[1] for h in H])
    return float(np.interp(c, cs, as_)) if cs[0] <= c else np.nan


for s in range(10):
    os.environ["DRAW"] = str(s); d = load("LCB"); N = len(d["ids"]); idx = {p: i for i, p in enumerate(d["ids"])}
    ii = np.array([idx[p] for p in ids_o if p in idx]); assert len(ii) == len(ids_o)
    S, T = view(d, d["tr"]), view(d, np.arange(N))
    C = {"ours": cost_of(d, fit_predict("ours", S, T)), "median": cost_of(d, np.repeat(d["med"][None], N, 0))}
    H = {a: curve(d, d["P"], c, ii) for a, c in C.items()}
    r_ = {a: dict(acc_at_opus_cost=at_cost(h, cost_o), cost_at_opus_acc=cost_at(h, acc_o), max_acc=float(h[-1][1]), cost_at_max=float(h[-1][0])) for a, h in H.items()}
    r_["routes"] = {sl: dict(acc=float(d["q"][ii, k].mean()), usd_per_1k=float(d["paid"][ii, k].mean() * 1e3)) for k, sl in enumerate(d["slots"])}
    res["draws"][s] = r_
    print(f"  draw {s}: ours acc at Opus cost {r_['ours']['acc_at_opus_cost']:.3f} (median {r_['median']['acc_at_opus_cost']:.3f}); max acc ours "
          f"{r_['ours']['max_acc']:.3f} at ${r_['ours']['cost_at_max'] * 1e3:.2f}/1k; cost at Opus acc ours {r_['ours']['cost_at_opus_acc'] * 1e3:.2f}", flush=True)
m = lambda a, k: float(np.nanmean([res["draws"][s][a][k] for s in res["draws"]]))
res["mean"] = {a: {k: m(a, k) for k in ("acc_at_opus_cost", "cost_at_opus_acc", "max_acc", "cost_at_max")} for a in ("ours", "median")}
res["routes"] = res["draws"][0]["routes"]
print("MEAN:", json.dumps(res["mean"]), "\nroutes on these problems:", json.dumps(res["routes"]), flush=True)
json.dump(res, open(Path(os.environ.get("OUT_DIR", Path(__file__).parent)) / "opus_anchor_lcb_pinned.json", "w"), indent=1, default=float)
print("DONE", flush=True)
