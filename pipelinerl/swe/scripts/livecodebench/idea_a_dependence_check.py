"""Idea A kill test: is there cross-model failure dependence BEYOND the probe's prior?
For target model m' and observed model m: fit on TRAIN problems
   base: y_m' ~ logit p_m'(x)                 (independent belief: prior only)
   dep:  y_m' ~ logit p_m'(x) + y_m          (plus one observed draw of model m on the same problem)
Evaluate held-out TEST log-loss gain per draw and the y_m coefficient. Same-model (m=m') rows are the
resample case. Draw pairs: one random draw of m and a DIFFERENT random draw of m' per problem, 20 reps."""
import json, sys, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
R="/mnt/llmd/results/exps/aristides/reason"
for T in ["bcb_tensors_5r","pool_v2_tensors_5rung"]:
    t=np.load(f"{R}/{T}/tensors.npz",allow_pickle=True)
    ok=(t["final_outcome"]&t["valid"]).astype(bool); v=t["valid"].astype(bool)
    S=[str(s) for s in t["model_slots"]]; pids=[str(p) for p in t["problem_ids"]]; pi={p:i for i,p in enumerate(pids)}
    cp={json.loads(l)["problem_id"]:json.loads(l)["p_successes"] for l in open(f"{R}/{T}/content_preds.jsonl")}
    sp=json.load(open(f"{R}/{T}/split_manifest.json")); rng=np.random.default_rng(0)
    def rows(split,m,mp):
        X=[];y=[]
        for p in sp[f"{split}_problem_ids"]:
            i=pi[p]; a=np.flatnonzero(v[i,m]); b=np.flatnonzero(v[i,mp])
            if not len(a) or len(b)<(2 if m==mp else 1): continue
            lp=np.log(np.clip(cp[p][mp],1e-4,1-1e-4)/np.clip(1-cp[p][mp],1e-4,1))
            for _ in range(20):
                da=rng.choice(a); db=rng.choice(b[b!=da] if m==mp else b)
                X.append([lp,float(ok[i,m,da])]); y.append(ok[i,mp,db])
        return np.array(X),np.array(y)
    print(f"\n== {T}: held-out log-loss gain (nats x100 per draw) from observing ONE draw of m, beyond the prior for m'  [coef on y_m]")
    print("   m' \\ m   " + "".join(f"{s:>15}" for s in S))
    for mp in range(len(S)):
        line=f"   {S[mp]:<9}"
        for m in range(len(S)):
            Xtr,ytr=rows("train",m,mp); Xte,yte=rows("test",m,mp)
            b=LogisticRegression().fit(Xtr[:,:1],ytr); d=LogisticRegression().fit(Xtr,ytr)
            g=(log_loss(yte,b.predict_proba(Xte[:,:1])[:,1])-log_loss(yte,d.predict_proba(Xte)[:,1]))*100
            line+=f"{g:>8.1f} [{d.coef_[0][1]:+.1f}]"
        print(line)
