"""Aggregate the training-draw runs of launch_oct13_draws.sh (NEW_PATH 4.A.92) into the paper's numbers. Every number is the MEAN over
training draws 0..9; every interval is the 2.5 / 97.5 percentile of the bootstrap replicates POOLED over draws (each draw's test
resamples around its own fit), so it covers both which training problems were drawn and which test problems. Draw -1 (the paper split)
is printed as a control only.
  draws   tmlr_strengthen_draws_<pool>_*  -> per pool and arm: saving vs median, ours vs the arm (direct), capture, s.d. over draws,
          draws where the per-draw interval excludes zero
  labels  cost_generalization_labels_<pool>_pinned_d<s>  -> per n and arm: saving vs median, ours - arm (points)
  onboard onboard_compare_<pool>_groups_pinned_d<s>      -> per k and arm: saving, ours - arm (points)
Usage: python draws_aggregate.py [DIR]  (default /mnt/.../reason_pinned_logs/draws) -> draws_summary_pinned.json
"""
import glob, json, sys
from pathlib import Path
import numpy as np

D = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/llmd/results/exps/aristides/reason/reason_pinned_logs/draws")
POOLS = ["LCB", "Omni", "MMLU-Pro", "AIME", "APPS", "BCB", "CC", "SuperGPQA", "BBEH"]; DRAWS = [str(s) for s in range(10)]
lc = lambda p: p.lower().replace("-", "")
ci = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]
out = {"draws": {}, "labels": {}, "onboard": {}}

for pool in POOLS:
    seeds = {}
    for f in sorted(glob.glob(str(D / f"tmlr_strengthen_draws_{lc(pool)}_*_pinned.json"))):
        seeds.update(json.load(open(f))["seeds"])
    have = [s for s in DRAWS if s in seeds]
    if not have:
        continue
    arms = [a for a in seeds[have[0]] if not a.startswith("_")]; res = {"n_draws": len(have), "n": seeds[have[0]]["_n"]}
    for a in arms:
        r = {}
        for key in ("vs_median", "ours_vs_it"):
            pts = np.array([seeds[s][a][key] for s in have]); bs = np.concatenate([seeds[s][a][key + "_bs"] for s in have])
            r[key] = [float(pts.mean()), *ci(bs)]; r[key + "_sd"] = float(pts.std())
            r[key + "_draws_excl0"] = int(sum(ci(seeds[s][a][key + "_bs"])[0] > 0 for s in have))
            if "-1" in seeds and a in seeds["-1"]:
                r[key + "_paper_split"] = float(seeds["-1"][a][key])
        res[a] = r
    o, m = res["oracle"]["vs_median"][0], res["ours"]["vs_median"][0]; res["capture"] = m / o
    res["capture_per_draw"] = float(np.mean([seeds[s]["ours"]["vs_median"] / seeds[s]["oracle"]["vs_median"] for s in have]))
    out["draws"][pool] = res
    print(f"\n===== {pool}: {len(have)} draws; test n={res['n']['test']}   (paper split in parentheses)")
    print(f"  {'arm':<12} {'vs median':>26} {'ours vs it':>26}")
    for a in arms:
        vm, vd = res[a]["vs_median"], res[a]["ours_vs_it"]
        print(f"  {a:<12} {vm[0]:+6.1f} [{vm[1]:+6.1f}, {vm[2]:+6.1f}] ({res[a].get('vs_median_paper_split', np.nan):+5.1f})"
              f"  {vd[0]:+6.1f} [{vd[1]:+6.1f}, {vd[2]:+6.1f}] ({res[a].get('ours_vs_it_paper_split', np.nan):+5.1f})  sd {res[a]['ours_vs_it_sd']:.1f}"
              f"  draws>0 {res[a]['ours_vs_it_draws_excl0']}/{len(have)}")
    print(f"  capture {res['capture']:.2f}")

for kind, pat, top in (("labels", "cost_generalization_labels_{p}_pinned_d{s}.json", "by_n"), ("onboard", "onboard_compare_{p}_groups_pinned_d{s}.json", "by_k")):
    for pool in ("LCB", "Omni", "MMLU-Pro", "SuperGPQA", "BBEH"):
        runs = [json.load(open(D / pat.format(p=lc(pool), s=s))) for s in DRAWS if (D / pat.format(p=lc(pool), s=s)).exists()]
        if not runs:
            continue
        res = {"n_draws": len(runs)}
        for key in runs[0][top]:
            row = {}
            for a, x in (runs[0][top][key] if kind == "labels" else runs[0][top][key]["arms"]).items():
                if a.startswith("_"):
                    continue
                get = (lambda r: r[top][key][a]) if kind == "labels" else (lambda r: r[top][key]["arms"][a])
                val = np.array([get(r)["vs_median" if kind == "labels" else "mean"] for r in runs]) * 100; rr = {"vs_median": float(val.mean()), "sd": float(val.std())}
                if "ours_minus_it" in x:
                    pts = np.array([get(r)["ours_minus_it"][0] for r in runs]); bs = np.concatenate([get(r)["ours_minus_it_bs"] for r in runs])
                    rr["ours_minus_it"] = [float(pts.mean()), *ci(bs)]
                row[a] = rr
            res[key] = row
        out[kind][pool] = res
        print(f"\n===== {kind} {pool}: {len(runs)} draws")
        for key, row in res.items():
            if key == "n_draws":
                continue
            print(f"  {top[3:]}={key:<4} " + "  ".join(f"{a} {r['vs_median']:+5.1f}" + (f" (d {r['ours_minus_it'][0]:+.1f} [{r['ours_minus_it'][1]:+.1f}, {r['ours_minus_it'][2]:+.1f}])" if "ours_minus_it" in r else "")
                                               for a, r in row.items()))
json.dump(out, open(Path(__file__).parent / "draws_summary_pinned.json", "w"), indent=1)
