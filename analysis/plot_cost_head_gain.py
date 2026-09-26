import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="/mnt/llmd/results/exps/aristides/reason"; PR={"oss20lo":0.12,"oss20md":0.57,"dsv4f":0.111,"oss120md":1.43,"oss120hi":1.43}
def hull(pts):
    pts=sorted(pts); h=[]
    for x in pts:
        while len(h)>=2 and (h[-1][1]-h[-2][1])*(x[0]-h[-2][0])<=(x[1]-h[-2][1])*(h[-1][0]-h[-2][0]): h.pop()
        h.append(x)
    return np.array(h)
def curves(T):
    t=np.load(f"{R}/{T}/tensors.npz",allow_pickle=True); S=[str(s) for s in t["model_slots"]]; pids=[str(p) for p in t["problem_ids"]]; pi={p:i for i,p in enumerate(pids)}
    v=t["valid"].astype(bool); ok=(t["final_outcome"]&t["valid"]).astype(float)
    real=np.stack([(t["prompt_tokens"][:,m]+t["completion_tokens"][:,m])*PR[s]/1e6*100 for m,s in enumerate(S)],1)
    rank=np.cumsum(v,2)-1; A=v&(rank%2==0); B=v&(rank%2==1)
    def est(M): n=M.sum(2); return np.where(n>0,(ok*M).sum(2)/np.maximum(n,1),np.nan), np.where(n>0,(real*M).sum(2)/np.maximum(n,1),np.nan)
    qA,cA=est(A); qB,cB=est(B)
    sp=json.load(open(f"{R}/{T}/split_manifest.json"))
    te=np.array([pi[p] for p in sp["test_problem_ids"]]); te=te[np.isfinite(qA[te]).all(1)&np.isfinite(qB[te]).all(1)]
    trn=np.array([pi[p] for p in sp["train_problem_ids"]])
    lp={json.loads(l)["problem_id"]:json.loads(l)["p_successes"][:5] for l in open(f"{R}/{T}/content_preds.jsonl")}
    lc={json.loads(l)["problem_id"]:json.loads(l)["expected_costs"][:5] for l in open(f"{R}/{T}/cost_preds.jsonl")}
    LP=np.array([lp[pids[i]] for i in te]); LC=np.array([lc[pids[i]] for i in te])*100
    # paper rule: this query's input tokens + route's median TRAIN output tokens, priced
    inp=np.nanmean(np.where(v,t["prompt_tokens"],np.nan),2)
    med=np.array([np.median(t["completion_tokens"][trn,m][v[trn,m]]) for m in range(5)])
    PC=np.stack([(inp[te,m]+med[m])*PR[S[m]]/1e6*100 for m in range(5)],1)
    def one(P,C):
        pts=[(0,0)]
        for V in np.geomspace(1e-5,100,500):
            r=np.argmax(P*V-C,1); pts.append((cB[te][np.arange(len(te)),r].mean(), qB[te][np.arange(len(te)),r].mean()*100))
        return hull(pts)
    return one(LP,PC), one(LP,LC), one(qA[te],cA[te]), len(te)
fig,axes=plt.subplots(1,2,figsize=(12,4.8))
col={"paper":"#8a8f98","ours":"#1f6feb","oracle":"#2da44e"}
for ax,(T,name,xmax,ylo) in zip(axes,[("pool_v2_tensors_5rung","LiveCodeBench — cost estimate matters",0.2,50),("bcb_tensors_5r","BigCodeBench — flat ladder, no gain (control)",0.05,40)]):
    P,O,Or,n=curves(T)
    ax.plot(P[:,0],P[:,1],color=col["paper"],lw=2.2,label="Prefill router, median-length cost (2603.20895)")
    ax.plot(O[:,0],O[:,1],color=col["ours"],lw=2.6,label="Same router + our per-problem cost head")
    ax.plot(Or[:,0],Or[:,1],color=col["oracle"],lw=1.6,ls="--",label="Oracle (true per-problem cost & success)")
    ax.set_xlim(0,xmax); ax.set_ylim(ylo, max(P[:,1].max(),O[:,1].max(),Or[:,1].max())+3)
    ax.set_xlabel("Spend per problem (cents)"); ax.set_ylabel("Accuracy (%)"); ax.set_title(f"{name}\n({n} held-out test problems; one model per problem, no verifier)",fontsize=10.5)
    ax.grid(alpha=0.25); ax.spines[["top","right"]].set_visible(False)
    if "tensors_5rung" in T:
        for acc in (70,):
            cp=np.interp(acc,P[:,1],P[:,0]); co=np.interp(acc,O[:,1],O[:,0])
            ax.annotate("",xy=(co,acc),xytext=(cp,acc),arrowprops=dict(arrowstyle="<-",color="black",lw=1.2))
            ax.text((cp+co)/2+0.004,acc-4.2,f"same {acc}% accuracy:\n{cp/co:.1f}× cheaper",ha="center",fontsize=9)
        ax.legend(loc="lower right",fontsize=8.5,frameon=False)
fig.tight_layout(); out="/tmp/claude-13011/-home-toolkit-PipelineRL-SWE/29f3ed3b-1f85-424a-8576-97a9148bdc53/scratchpad/cost_head_gain.png"
fig.savefig(out,dpi=160); print(out)
