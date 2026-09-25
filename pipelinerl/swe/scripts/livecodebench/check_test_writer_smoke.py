"""Checks on the test-writer smoke test. Selection = submit the candidate passing the most suite cases;
ties broken uniformly (computed exactly: mean truth over the tied top set).
(1) is any writer's suite better than NO suite (random pick)?  (2) honest per-problem oracle:
choose the writer on half the candidates, score on the other half.  (3) the user's framing:
generator X's own candidates, verified by X's own suite vs by another writer Y."""
import json, glob, numpy as np
from collections import defaultdict
D="/mnt/llmd/results/exps/aristides/reason/testwriter_smoke_lcb/verdicts"
W=["qwen4b","oss20lo","oss20md","dsv4f","oss120md"]
V={}  # (writer,pid) -> {cand: (score,truth)}
for w in W:
    for l in open(f"{D}/verdicts_{w}.jsonl"):
        r=json.loads(l)
        if not r["suite_ok"]: continue
        V.setdefault((w,r["problem_id"]),{})[(r["slot"],r["draw_index"])]=(r["n_case_pass"],bool(r["truth"]))
pids=sorted({p for (w,p) in V if all((x,p) in V for x in W)})
cands={p:sorted(set.intersection(*[set(V[(w,p)]) for w in W])) for p in pids}
pids=[p for p in pids if len(cands[p])>=4]
def sel(w,p,C):  # expected accuracy of argmax-with-random-ties over candidate list C
    s=np.array([V[(w,p)][c][0] for c in C]); t=np.array([V[(w,p)][c][1] for c in C])
    return t[s==s.max()].mean()
def rnd(p,C): return np.mean([V[(W[0],p)][c][1] for c in C])
def perfect(p,C): return float(any(V[(W[0],p)][c][1] for c in C))
rng=np.random.default_rng(0)
def ci(x):
    x=np.asarray(x); b=[x[rng.integers(0,len(x),len(x))].mean() for _ in range(2000)]
    return f"{x.mean()*100:+.1f} [{np.percentile(b,2.5)*100:+.1f},{np.percentile(b,97.5)*100:+.1f}]"
print(f"{len(pids)} problems with suites from all 5 writers and >=4 common candidates "
      f"(mean {np.mean([len(cands[p]) for p in pids]):.1f} candidates)")
print("\n(1) pick 1 among all candidates:")
R={p:rnd(p,cands[p]) for p in pids}
print(f"   no suite (random)   {np.mean(list(R.values()))*100:5.1f}%   perfect {np.mean([perfect(p,cands[p]) for p in pids])*100:5.1f}%")
for w in W:
    a=[sel(w,p,cands[p]) for p in pids]
    print(f"   {w:<18}  {np.mean(a)*100:5.1f}%   vs random {ci(np.array(a)-np.array([R[p] for p in pids]))}")
print("\n(2) per-problem writer choice: same-data oracle vs split-half honest oracle (20 random splits):")
same=[max(sel(w,p,cands[p]) for w in W) for p in pids]
hon=[];fix=[]
for p in pids:
    C=cands[p]; acc=[];facc=[]
    for _ in range(20):
        perm=rng.permutation(len(C)); A=[C[i] for i in perm[:len(C)//2]]; B=[C[i] for i in perm[len(C)//2:]]
        # choose writer by within-A AUC-like score: mean(score|correct)-mean(score|wrong); ties -> random writer
        def fit(w):
            y=np.array([V[(w,p)][c][1] for c in A]); s=np.array([V[(w,p)][c][0] for c in A])
            return (s[y].mean()-s[~y].mean()) if 0<y.sum()<len(y) else 0.0
        f=np.array([fit(w) for w in W]); best=rng.choice(np.flatnonzero(f==f.max()))
        acc.append(sel(W[best],p,B)); facc.append(B)
    hon.append(np.mean(acc)); fix.append(facc)
bestfixed=max(W,key=lambda w: np.mean([sel(w,p,cands[p]) for p in pids]))
fixB=[np.mean([sel(bestfixed,p,B) for B in fix[i]]) for i,p in enumerate(pids)]
rndB=[np.mean([rnd(p,B) for B in fix[i]]) for i,p in enumerate(pids)]
print(f"   same-data oracle (inflated)  {np.mean(same)*100:5.1f}%  on all candidates")
print(f"   on held-out halves:  random {np.mean(rndB)*100:5.1f}%   best fixed ({bestfixed}) {np.mean(fixB)*100:5.1f}%   "
      f"honest oracle {np.mean(hon)*100:5.1f}%   oracle - fixed {ci(np.array(hon)-np.array(fixB))}")
print("\n(3) generator X's own candidates (problems with >=2 X candidates): verify with X's suite vs others'")
for X in ["oss20lo","oss20md","dsv4f","oss120md"]:
    ps=[p for p in pids if sum(c[0]==X for c in cands[p])>=2]
    if not ps: continue
    row={w:[sel(w,p,[c for c in cands[p] if c[0]==X]) for p in ps] for w in W}
    r=[rnd(p,[c for c in cands[p] if c[0]==X]) for p in ps]
    others=[w for w in W if w!=X]; bo=max(others,key=lambda w: np.mean(row[w]))
    print(f"   X={X:<9} n={len(ps):>3}  random {np.mean(r)*100:5.1f}%  self-suite {np.mean(row[X])*100:5.1f}%  "
          + "  ".join(f"{w} {np.mean(row[w])*100:4.1f}" for w in others)
          + f"   best-other({bo}) - self {ci(np.array(row[bo])-np.array(row[X]))}")
