"""What is it worth to decide, PER PROBLEM, between many cheap draws and few expensive ones?

A plan is a vector k = (k_1..k_M), draws per rung. With per-draw success q_{p,m},
    value(p,k) = P(solve) - sum_m k_m c_m / R = 1 - prod_m (1-q_pm)^{k_m} - sum_m k_m c_m/R
Three policies, all with the SAME action space, so the differences isolate one thing each:
  FIXED     : one k for every problem (the strongest Zero Router: best fixed plan, not a cascade)
  DEPTH     : per-problem scaling of the fixed plan's MIX (how many, not which)
  FULL      : per-problem choice of k (which rung AND how many)
Planning uses half of each rung's draws, evaluation the other half, so the oracle cannot
exploit the noise in its own q estimates. Naive (plan=eval on all draws) is printed for contrast.
"""
import json,glob,os,re,itertools
import numpy as np
from collections import defaultdict

C={"oss20lo":0.011,"oss20md":0.042,"dsv4f":0.098,"oss120md":0.106,"oss120hi":0.607}  # cents/draw
D=defaultdict(lambda: defaultdict(list))
for f in glob.glob("*_d*.jsonl"):
    lab,sp,dr=re.match(r"(.+?)_(train|eval)_d(\d+)\.jsonl$",os.path.basename(f)).groups()
    if lab in C:
        for l in open(f):
            if l.strip():
                r=json.loads(l); D[lab][(sp,str(r["problem_id"]))].append(bool(r.get("resolved")))
RUNGS=[m for m in C if len(D[m])>0]
MAXK={m:min(6,int(np.median([len(v) for v in D[m].values()]))) for m in RUNGS}
probs=set.intersection(*[{p for p,v in D[m].items() if len(v)>=2} for m in RUNGS])
print(f"rungs {RUNGS}\nmax draws used per rung {MAXK}\n{len(probs)} problems present in every rung\n")

rng=np.random.default_rng(0)
def halves(v):
    v=list(v); rng.shuffle(v); h=max(1,len(v)//2); return np.mean(v[:h]), np.mean(v[h:])
QA=np.array([[halves(D[m][p])[0] for m in RUNGS] for p in sorted(probs)])
QB=np.array([[halves(D[m][p])[1] for m in RUNGS] for p in sorted(probs)])
QALL=np.array([[np.mean(D[m][p]) for m in RUNGS] for p in sorted(probs)])
PLANS=[k for k in itertools.product(*[range(MAXK[m]+1) for m in RUNGS]) if sum(k)<=8]
K=np.array(PLANS); COST=K@np.array([C[m] for m in RUNGS])
def vals(Q,cR_scale):
    # P(solve) for every (problem, plan): 1 - prod (1-q)^k
    lp=np.log(np.clip(1-Q,1e-9,1))@K.T
    return (1-np.exp(lp)) - COST*cR_scale
print(f"{'R (value of a correct answer)':<30}{'FIXED':>9}{'+depth':>9}{'+tier choice (FULL)':>21}{'naive FULL':>12}")
for R in [1.0,3.0,10.0,30.0]:
    s=1.0/R
    vA,vB=vals(QA,s),vals(QB,s)
    fixed_k=int(np.argmax(vA.mean(0)));  fixed=vB[:,fixed_k].mean()
    # DEPTH: keep the fixed plan's MIX, scale it per problem (0.5x,1x,1.5x,2x rounded)
    mix=K[fixed_k]; cand=[np.rint(mix*f).astype(int) for f in (0,0.5,1,1.5,2,3)]
    idx=[np.where((K==c).all(1))[0] for c in cand]; idx=[i[0] for i in idx if len(i)]
    depth=vB[:,idx][np.arange(len(QB)),np.argmax(vA[:,idx],1)].mean()
    full=vB[np.arange(len(QB)),np.argmax(vA,1)].mean()
    naive=vals(QALL,s).max(1).mean()
    print(f"R={R:>5.1f}c  best fixed plan {str(dict(zip(RUNGS,K[fixed_k]))):<40}"[:30].ljust(30)
          +f"{fixed:>9.4f}{depth:>9.4f}{full:>21.4f}{naive:>12.4f}")
    print(f"    plan={ {m:int(x) for m,x in zip(RUNGS,K[fixed_k]) if x} }   "
          f"depth +{(depth-fixed)/abs(fixed)*100:.1f}%   tier +{(full-depth)/abs(fixed)*100:.1f}%   "
          f"total +{(full-fixed)/abs(fixed)*100:.1f}%")
