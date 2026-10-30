"""Every single route as a constant policy on each dataset's test set (billed spend per 1,000 queries and accuracy; the paper split's
test problems, which every training set keeps). Used for the constant-deepseek point in Figure 2 (make_headline_fig.py).
Usage: REASON_ROOT=.../reason_pinned RESULT_TAG=_pinned python single_routes.py LCB [POOL ...] -> single_routes_pinned.json (merged)
"""
import json, sys
from pathlib import Path
WANT = sys.argv[1:]
__file__ = str(Path(__file__).resolve()); _cg = Path(__file__).parent / "cost_generalization.py"; sys.argv = ["x", "labels", WANT[0]]
exec(open(_cg).read().split('MODE, POOL = sys.argv[1], sys.argv[2]')[0])
outf = Path(__file__).parent / "single_routes_pinned.json"; res = json.load(open(outf)) if outf.exists() else {}
for P in WANT:
    d = load(P); ev = d["ev"]
    res[P] = {s: dict(usd_per_1k=float(d["paid"][ev, k].mean() * 1000), accuracy=float(d["q"][ev, k].mean()), n=int(len(ev))) for k, s in enumerate(d["slots"])}
    print(P, {s: (round(v["usd_per_1k"], 3), round(v["accuracy"] * 100, 1)) for s, v in res[P].items()})
json.dump(res, open(outf, "w"), indent=1)
