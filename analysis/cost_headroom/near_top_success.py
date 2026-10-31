"""Near the top of the frontier (NEW_PATH 4.A.129): cost of reaching 90/95/98/100% of deepseek-v4-flash's accuracy, as a share of calling it
on every query, for the success variants of success_mirt.py (all with our cost probe) and ZeroRouter-4B as a whole router; a training set
whose frontier does not reach the accuracy counts as deepseek alone. Also the most accurate policy's share of queries sent to deepseek.
Usage: REASON_ROOT=.../reason_pinned RESULT_TAG=_pinned python near_top_success.py > near_top_success_pinned.txt
"""
import sys, glob, numpy as np
__file__ = "/home/toolkit/PipelineRL-SWE/analysis/cost_headroom/cost_generalization.py"; sys.argv = ["x", "labels", "LCB"]
exec(open(__file__).read().split('MODE, POOL = sys.argv[1], sys.argv[2]')[0])
SM = "/mnt/llmd/results/exps/aristides/reason/reason_pinned_logs/succmirt2"; LC = "/mnt/llmd/results/exps/aristides/reason/reason_pinned_logs/lcurve"
FS = (0.90, 0.95, 0.98, 1.00); ARMS = ("ours", "shared", "mirt", "mirt-raw", "zr4b+cost", "ZR4B-whole")
print("cost % of deepseek alone (gm over 10 sets, non-reaching set -> 100) [n reaching]; top = share of queries the top policy sends to deepseek / top acc as % of ds")
for P in ["LCB", "Omni", "MMLU-Pro", "AIME", "APPS", "BCB", "CC", "SuperGPQA", "BBEH", "SWESmith"]:
    d = load(P); ev = d["ev"]; k = d["slots"].index("dsv4f"); q = d["q"][ev]; Y, X = q[:, k].mean(), d["paid"][ev, k].mean(); r = np.arange(len(ev))
    RS = {a: {f: [] for f in FS} | {"sh": [], "ta": []} for a in ARMS}
    def add(a, Pa, tk):
        H = curve(d, Pa.astype(float), cost_of(d, tk.astype(float)), ev)
        for f in FS: RS[a][f].append(cost_at(H, f * Y) / X)
        m = Pa[ev].argmax(1); RS[a]["sh"].append((m == k).mean()); RS[a]["ta"].append(q[r, m].mean() / Y)
    for fn in sorted(glob.glob(f"{SM}/success_mirt_{P.lower().replace('-', '')}_*_pinned.npz")):
        z = np.load(fn); assert (z["ids"] == np.array(d["ids"])).all()
        for s in sorted({x.split('_')[0] for x in z.files if x.endswith('_ev')}):
            for a in ARMS[:-1]: add(a, z[f"{s}_{a}_P"], z[f"{s}_cost_ours_tok"])
    for fn in sorted(glob.glob(f"{LC}/learning_curve_{P.lower().replace('-', '')}_*_pinned.npz")):
        z = np.load(fn)
        for s in sorted({x.split('_')[0] for x in z.files if x.endswith('_ev')}):
            add("ZR4B-whole", z[f"{s}_all_system_zr_4b_P"], z[f"{s}_all_system_zr_4b_tok"])
    print(f"== {P}  (n sets {len(RS['ours']['sh'])}/{len(RS['ZR4B-whole']['sh'])})")
    for a in ARMS:
        cells = []
        for f in FS:
            v = np.array(RS[a][f]); n = int(np.isfinite(v).sum()); fb = np.where(np.isfinite(v), np.minimum(v, 1.0), 1.0)
            cells.append(f"{f:.0%} {np.exp(np.log(fb).mean())*100:4.0f}[{n:2d}]")
        print(f"   {a:11s} " + "  ".join(cells) + f"   top: {np.mean(RS[a]['sh']):.2f} ds, acc {np.mean(RS[a]['ta'])*100:5.1f}%")
