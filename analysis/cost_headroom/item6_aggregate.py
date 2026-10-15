"""Aggregate the remaining analyses run on the 10 training sets (NEW_PATH 4.A.102). Per analysis: mean over sets of each point
estimate, s.d. over sets, and the number of sets whose own interval excludes zero (the per-set outputs store intervals, not
resamples). deploy: tmlr_strengthen.py deploy; nonreason: nonreason_compare.py; transfer: cost_generalization.py fulltransfer;
families: newfamily_eval.py (routing and onboarding).
Usage: python item6_aggregate.py [DIR] -> item6_summary_pinned.json
"""
import json, sys
from pathlib import Path
import numpy as np

D = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/llmd/results/exps/aristides/reason/reason_pinned_logs/item6"); out = {}
m_ = lambda xs: [float(np.nanmean(xs)), float(np.nanstd(xs))]


def triples(vals):                       # list of [pt, lo, hi] -> mean pt, mean lo, mean hi, sets with lo > 0, sets with hi < 0
    a = np.array([v for v in vals if v is not None], float)
    return dict(mean=float(a[:, 0].mean()), sd=float(a[:, 0].std()), lo=float(a[:, 1].mean()), hi=float(a[:, 2].mean()), pos=int((a[:, 1] > 0).sum()), neg=int((a[:, 2] < 0).sum()), n=len(a))


fmt = lambda t: f"{t['mean']:+6.1f} (sd {t['sd']:.1f}; >0 in {t['pos']}/{t['n']}, <0 in {t['neg']}/{t['n']})"
# ---- deployable policies
out["deploy"] = {}
for f in sorted((D / "deploy").glob("tmlr_strengthen_deploy_*_pinned_d0.json")):
    pool = json.load(open(f))["pool"]; runs = [json.load(open(g)) for g in sorted((D / "deploy").glob(f.name.replace("_d0.json", "_d*.json")))]
    arms = [a for a in runs[0] if isinstance(runs[0][a], dict) and "vs_median_deployed" in runs[0][a]]; o = {"n": len(runs)}
    for a in arms:
        o[a] = {k: m_([r[a][k] * 100 for r in runs]) for k in ("vs_median_deployed", "own_frontier_loss", "miss")}
        if a != "ours":
            o[a]["ours_minus_it_deployed"] = triples([r[a].get("ours_minus_it_deployed") for r in runs])
    out["deploy"][pool] = o
    ext = {a: o[a]["ours_minus_it_deployed"]["mean"] for a in ("mean", "gbm", "mixllm", "knn", "zerorouter") if a in o}; best = min(ext, key=ext.get)
    print(f"deploy {pool:<10} ours vs median deployed {o['ours']['vs_median_deployed'][0]:+5.1f} (sd {o['ours']['vs_median_deployed'][1]:.1f})  own-frontier loss "
          f"{o['ours']['own_frontier_loss'][0]:.1f}  miss {o['ours']['miss'][0]:.1f} pt | ours - best external ({best}) {fmt(o[best]['ours_minus_it_deployed'])}"
          f" | ours - fromsuccess {fmt(o['fromsuccess']['ours_minus_it_deployed'])} | median own-frontier loss {o['median']['own_frontier_loss'][0]:.1f}")
# ---- reasoning vs non-reasoning
out["nonreason"] = {}
for ds in ("lcb", "omni500", "mmlupro"):
    runs = [json.load(open(g)) for g in sorted((D / "nonreason").glob(f"nonreason_compare_{ds}_pinned_d*.json"))]
    if not runs:
        continue
    o = {"n": len(runs), "n_test_common": runs[0]["n_test_common"]}
    for nm in runs[0]["routing"]:
        o[nm] = {a: triples([[x * 100 for x in r["routing"][nm][a]] for r in runs]) for a in ("headroom", "ours", "from_success")}
        o[nm]["capture"] = m_([r["routing"][nm]["capture"] for r in runs])
    out["nonreason"][ds] = o
    print(f"nonreason {ds:<8} " + "  ".join(f"{nm}: headroom {o[nm]['headroom']['mean']:+5.1f} ours {fmt(o[nm]['ours'])} capture {o[nm]['capture'][0]:.2f}" for nm in ("reasoning", "nonreasoning")))
# ---- full transfer
out["transfer"] = {}
for f in sorted((D / "transfer").glob("cost_generalization_fulltransfer_*_pinned_d0.json")):
    runs = [json.load(open(g)) for g in sorted((D / "transfer").glob(f.name.replace("_d0.json", "_d*.json")))]; pool = runs[0]["pool"]
    o = {"n": len(runs), "in_domain_ours": m_([r["in_domain_ours"]["vs_median"] * 100 for r in runs])}
    for src in ("same_domain", "all_other"):
        for var in ("zero_shot", "level10"):
            arms = runs[0][src][var]
            o[f"{src}|{var}"] = {a_: {"vs_median": m_([r[src][var][a_]["vs_median"] * 100 for r in runs])}
                                 | ({"ours_minus_it": triples([r[src][var][a_]["ours_minus_it"] for r in runs])} if "ours_minus_it" in arms[a_] else {}) for a_ in arms}
    out["transfer"][pool] = o; z = o["same_domain|zero_shot"]; ext = {a_: z[a_]["vs_median"][0] for a_ in ("mixllm", "knn", "zerorouter")}; best = max(ext, key=ext.get)
    print(f"transfer {pool:<9} in-domain {o['in_domain_ours'][0]:+5.1f} | same-domain zero-shot: ours {z['ours']['vs_median'][0]:+5.1f}  prefill rule {z['prefill_router']['vs_median'][0]:+5.1f}"
          f"  best external {best} {z[best]['vs_median'][0]:+5.1f}; ours - best external {fmt(z[best]['ours_minus_it'])}; ours - prefill rule {fmt(z['prefill_router']['ours_minus_it'])}")
# ---- new families (routing + onboarding)
runs = [json.load(open(g)) for g in sorted((D / "families").glob("newfamily_eval_pinned_d*.json"))]
if runs:
    o = {"n": len(runs)}
    for P in ("LCB", "Omni", "MMLU-Pro"):
        if not all(P in r for r in runs):
            continue
        o[P] = {"onboarding_newpool": {}}
        for mdl in ("qw32", "glm47f", "nemo120"):
            ob = [r[P]["onboarding_newpool"][mdl] for r in runs]
            o[P]["onboarding_newpool"][mdl] = {"without": m_([x["without"] * 100 for x in ob])} | {
                k: {a: m_([x[k][a] * 100 for x in ob]) for a in ("ours", "zr", "knn", "naive")} | {"ours_minus_zr": triples([x[k]["ours_minus_zr"] for x in ob])} for k in ("10", "50")}
        o[P]["onboarding_newpool"]["full"] = m_([r[P]["onboarding_newpool"]["full"] * 100 for r in runs])
        nm = o[P]["onboarding_newpool"]["nemo120"]
        print(f"families {P:<9} newpool full {o[P]['onboarding_newpool']['full'][0]:+5.1f}  without nemo120 {nm['without'][0]:+5.1f}  nemo120 k=10: ours {nm['10']['ours'][0]:+5.1f} zr {nm['10']['zr'][0]:+5.1f} "
              f"naive {nm['10']['naive'][0]:+5.1f}  ours - zr {fmt(nm['10']['ours_minus_zr'])}; k=50 ours - zr {fmt(nm['50']['ours_minus_zr'])}")
    out["families"] = o
json.dump(out, open(Path(__file__).parent / "item6_summary_pinned.json", "w"), indent=1, default=float)
