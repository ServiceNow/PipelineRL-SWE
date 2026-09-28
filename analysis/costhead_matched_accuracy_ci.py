"""Cost head, one-shot routing: cost at matched accuracy with operating points chosen on CALIBRATION.
For each arm (paper median-length rule vs learned cost head; same success head), sweep V in argmax p*V - c
on calibration; for each target accuracy, find the calibration mix of two adjacent V's hitting it; apply
that fixed mix once to test. Test outcome per problem = expected over that route's valid draws. CI: paired
bootstrap over test problems (mix held fixed)."""
import json, numpy as np
R="/mnt/llmd/results/exps/aristides/reason"; PR={"oss20lo":0.12,"oss20md":0.57,"dsv4f":0.111,"oss120md":1.43,"oss120hi":1.43}
import os
T=os.environ.get("CI_TENSORS","pool_v2_tensors_5rung"); t=np.load(f"{R}/{T}/tensors.npz",allow_pickle=True)
S=[str(s) for s in t["model_slots"]]; pids=[str(p) for p in t["problem_ids"]]; pi={p:i for i,p in enumerate(pids)}
v=t["valid"].astype(bool); ok=(t["final_outcome"]&t["valid"]).astype(float)
# CI_MARKET=1: market input/output prices (analysis/cost_headroom/decompose.MK) instead of the legacy blended table
MKT=os.environ.get("CI_MARKET")=="1"
if MKT:
    import sys; sys.path.insert(0,"analysis/cost_headroom"); from decompose import MK
PIN={s:(MK[s][0] if MKT else PR[s]) for s in S}; POUT={s:(MK[s][1] if MKT else PR[s]) for s in S}
real=np.stack([(t["prompt_tokens"][:,m]*PIN[s]+t["completion_tokens"][:,m]*POUT[s])/1e6*100 for m,s in enumerate(S)],1)
n=v.sum(2); Q=np.where(n>0,(ok*v).sum(2)/np.maximum(n,1),0); Cr=np.where(n>0,(real*v).sum(2)/np.maximum(n,1),1e9)
sp=json.load(open(f"{R}/{T}/split_manifest.json")); idx={k:np.array([pi[p] for p in sp[f"{k}_problem_ids"]]) for k in ("train","calibration","test")}
lp={json.loads(l)["problem_id"]:json.loads(l)["p_successes"][:5] for l in open(f"{R}/{T}/content_preds.jsonl")}
lc={json.loads(l)["problem_id"]:json.loads(l)["expected_costs"][:5] for l in open(f"{R}/{T}/"+os.environ.get("CI_COST","cost_preds.jsonl"))}
P=np.array([lp[p] for p in pids]); LC=np.array([lc[p] for p in pids])*100
tr=idx["train"]; inp=np.nanmean(np.where(v,t["prompt_tokens"],np.nan),2)
med=np.array([np.median(t["completion_tokens"][tr,m][v[tr,m]]) for m in range(5)])
PC=np.stack([(np.nan_to_num(inp[:,m],nan=np.nanmean(inp[tr,m]))*PIN[S[m]]+med[m]*POUT[S[m]])/1e6*100 for m in range(5)],1)
avail=n>0
Vs=np.geomspace(1e-5,100,400)
def choose(C,ii,V): return np.argmax(np.where(avail[ii],P[ii]*V-C[ii],-np.inf),1)
def outcome(C,ii,V):
    r=choose(C,ii,V); return Q[ii,r], Cr[ii,r]
def mix_for(C,target):
    cal=idx["calibration"]; pts=sorted({(outcome(C,cal,V)[1].mean(),outcome(C,cal,V)[0].mean(),V) for V in Vs})
    # upper hull on calibration
    h=[]
    for p in pts:
        while len(h)>=2 and (h[-1][1]-h[-2][1])*(p[0]-h[-2][0])<=(p[1]-h[-2][1])*(h[-1][0]-h[-2][0]): h.pop()
        h.append(p)
    for a,b in zip(h,h[1:]):
        if a[1]<=target<=b[1]:
            w=(target-a[1])/max(b[1]-a[1],1e-12); return a[2],b[2],w
    return None
te=idx["test"]; rng=np.random.default_rng(0)
print(f"LCB one-shot, {len(te)} test problems; operating points chosen on calibration ({len(idx['calibration'])})")
print(f"{'target(cal)':>11} | {'paper: test acc @ cost':>24} | {'ours: test acc @ cost':>23} | cost ratio ours/paper [95% CI] | acc diff [95% CI]")
out=[]
for tgt in (0.60,0.65,0.70,0.75,0.80,0.85):
    arms={}
    for name,C in (("paper",PC),("ours",LC)):
        m=mix_for(C,tgt)
        if m is None: arms=None; break
        a0,c0=outcome(C,te,m[0]); a1,c1=outcome(C,te,m[1]); w=m[2]
        arms[name]=((1-w)*a0+w*a1,(1-w)*c0+w*c1)
    if arms is None: print(f"{tgt*100:>10.0f}% | unreachable on calibration"); continue
    (A0,C0),(A1,C1)=arms["paper"],arms["ours"]
    bs_r=[];bs_a=[]
    for _ in range(4000):
        i=rng.integers(0,len(te),len(te)); bs_r.append(C1[i].mean()/C0[i].mean()); bs_a.append(A1[i].mean()-A0[i].mean())
    r=C1.mean()/C0.mean()
    print(f"{tgt*100:>10.0f}% | {A0.mean()*100:>9.1f}% @ {C0.mean():.4f}c | {A1.mean()*100:>8.1f}% @ {C1.mean():.4f}c | "
          f"{r:.2f} [{np.percentile(bs_r,2.5):.2f},{np.percentile(bs_r,97.5):.2f}] ({(1-r)*100:.0f}% cheaper) | "
          f"{(A1.mean()-A0.mean())*100:+.1f} [{np.percentile(bs_a,2.5)*100:+.1f},{np.percentile(bs_a,97.5)*100:+.1f}]")
    out.append({"target_cal":tgt,"paper":[A0.mean(),C0.mean()],"ours":[A1.mean(),C1.mean()],"ratio":r,
                "ratio_ci":[np.percentile(bs_r,2.5),np.percentile(bs_r,97.5)],"acc_diff_ci":[np.percentile(bs_a,2.5),np.percentile(bs_a,97.5)]})
json.dump(out,open(os.environ.get("CI_OUT","analysis/costhead_matched_accuracy_ci.json"),"w"),indent=1,default=float)
