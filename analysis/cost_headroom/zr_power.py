"""Free follow-up analyses; see ZR_POWER_PROTOCOL.md. No external API calls."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from decompose import R, MK, hull, cost_at
from baseline_cost_heads import rich
from zr_dimsweep import POOLS, fit_stage1

HERE = Path(__file__).parent
DEST = HERE / 'zr_power_results'
VS = np.geomspace(1e-5, 100, 250)
B = 1000


def load(pool):
    label, name, cfile, act = pool
    t = np.load(R / name / 'tensors.npz', allow_pickle=True)
    ids = list(map(str, t['problem_ids']))
    valid = t['valid'].astype(bool)
    n = valid.sum(2)
    success = (t['final_outcome'] & valid).sum(2)
    ct, pt = t['completion_tokens'].astype(float), t['prompt_tokens'].astype(float)
    inp = np.where(valid, pt, 0).sum(2) / np.maximum(n, 1)
    out = np.where(valid, ct, 0).sum(2) / np.maximum(n, 1)
    slots = list(map(str, t['model_slots']))
    pin = np.array([MK[s][0] for s in slots]) * 100 / 1e6
    pout = np.array([MK[s][1] for s in slots]) * 100 / 1e6
    manifest = json.loads((R / name / 'split_manifest.json').read_text())
    ix = {p:i for i,p in enumerate(ids)}
    split = {k:np.array([ix[str(p)] for p in manifest[k+'_problem_ids']]) for k in ['train','calibration','test']}
    def preds(file, key):
        rows = [json.loads(l) for l in (R/name/file).read_text().splitlines()]
        mapping = {str(r['problem_id']):r[key][:len(slots)] for r in rows}
        return np.array([mapping[p] for p in ids])
    return dict(label=label, ids=ids, valid=valid, n=n, succ=success,
                ok=(t['final_outcome'] & valid).astype(float), ct=ct, pt=pt,
                inp=inp, out=out, pin=pin, pout=pout, avail=n>0,
                Q=success/np.maximum(n,1), Cr=inp*pin+out*pout,
                P=np.clip(preds('content_preds.jsonl','p_successes'),1e-4,1-1e-4),
                C=100*preds(cfile,'expected_costs'), split=split, X=rich(R/act,ids))


def reference(d, tr):
    med = np.array([np.median(d['ct'][tr,m][d['valid'][tr,m]]) for m in range(d['n'].shape[1])])
    return d['inp']*d['pin']+med*d['pout']


def choices(d, P, C):
    return np.where(d['avail'][None],VS[:,None,None]*P[None]-C[None],-np.inf).argmax(2)


def curves(d, ch, ii, Q=None, Cr=None):
    q, c = (d['Q'][ii], d['Cr'][ii]) if Q is None else (Q, Cr)
    rows = np.arange(len(ii))[None]
    return [hull(zip(c[rows,x[:,ii]].mean(1),q[rows,x[:,ii]].mean(1))) for x in ch]


def effects(H, common=False):
    def savings(h, ref, limits=None):
        lo,hi = limits or (max(h[0][1],ref[0][1]),min(h[-1][1],ref[-1][1]))
        if hi<=lo: return np.nan
        targets = np.linspace(lo+.05*(hi-lo),hi-.05*(hi-lo),12)
        return 1-np.exp(np.mean(np.log([cost_at(h,a)/cost_at(ref,a) for a in targets])))
    limits = (max(h[0][1] for h in H),min(h[-1][1] for h in H)) if common else None
    a,b = savings(H[0],H[2],limits),savings(H[1],H[2],limits)
    return np.array([a*100,b*100,(a-b)*100])


def zr(d, tr, cal, fixed=None):
    X = d['X']
    Z = PCA(256,random_state=0).fit(X[tr]).transform(X)
    Z /= Z[tr].std(0)+1e-6
    ref = reference(d,tr)
    basechoices = choices(d,d['P'],ref)
    # During cross-fitting d['P'] is the newly fitted linear success head.
    configs = [fixed] if fixed else [f'D{D}|s{s}|K{k}' for D in (1,5) for s in (0,1,2) for k in (5,10,20)]
    fitted, results = {}, {}
    for cfg in configs:
        D,s,K = [int(v[1:]) for v in cfg.split('|')]
        if (D,s) not in fitted:
            la,b,th = fit_stage1(d['succ'][tr],d['n'][tr],D,s)
            pr = RidgeCV(alphas=np.geomspace(1,1e5,11)).fit(Z[tr],np.c_[la,b]).predict(Z)
            A,BB = np.exp(pr[:,:D]),pr[:,D:]
            A[tr],BB[tr] = np.exp(la),b
            fitted[D,s] = expit((A[:,None]*(th[None]-BB[:,None])).sum(-1)), (A*BB).sum(1)
        P,score = fitted[D,s]
        bins = np.searchsorted(np.quantile(score[tr],np.linspace(0,1,K+1)[1:-1]),score)
        tab = np.array([[np.nanmean(d['out'][tr[bins[tr]==j],m]) if (bins[tr]==j).any() else np.nanmean(d['out'][tr,m]) for j in range(K)] for m in range(d['n'].shape[1])])
        C = d['inp']*d['pin']+tab.T[bins]*d['pout']
        H = curves(d,[choices(d,P,C),basechoices,basechoices],cal)
        gain = effects(H)[0]
        results[cfg] = (gain,P,C)
    selected = max(results,key=lambda k:results[k][0])
    return results[selected][1:], {'chosen':selected,'calibration_gains_pp':{k:float(v[0]) for k,v in results.items()}}


def linear(d,tr,cal):
    # Full training row-space SVD: preserves the L2 geometry without feature truncation.
    xs = StandardScaler().fit(d['X'][tr]).transform(d['X']).astype(float)
    _,_,V = np.linalg.svd(xs[tr],full_matrices=False)
    F = xs @ V.T
    scale = max(1,d['X'].shape[1]//2560)
    P,C = np.zeros_like(d['Q']),np.zeros_like(d['Cr'])
    settings=[]
    for m in range(P.shape[1]):
        y = d['ok'][:,m,0]
        best,score = .05,-np.inf
        def auc(yy,p):
            ranks=np.argsort(np.argsort(p))+1
            positives=yy.sum()
            return (ranks[yy==1].sum()-positives*(positives+1)/2)/(positives*(len(yy)-positives))
        if len(cal)>=20 and 0<y[cal].mean()<1:
            for c in [1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1]:
                model=LogisticRegression(max_iter=2000,C=c/scale).fit(F[tr],y[tr])
                a=auc(y[cal],model.predict_proba(F[cal])[:,1])
                if a>score: best,score=c,a
        weights=np.r_[d['succ'][tr,m],d['n'][tr,m]-d['succ'][tr,m]]
        keep=weights>1e-9
        model=LogisticRegression(max_iter=2000,C=best/scale).fit(np.vstack([F[tr],F[tr]])[keep],np.r_[np.ones(len(tr)),np.zeros(len(tr))][keep],sample_weight=weights[keep])
        raw=model.predict_proba(F)[:,1]
        if len(cal)>=20 and 0<y[cal].mean()<1:
            p=np.clip(raw,1e-6,1-1e-6); logit=np.log(p/(1-p))[:,None]
            raw=LogisticRegression(max_iter=2000,C=1e6).fit(logit[cal],y[cal]).predict_proba(logit)[:,1]
        P[:,m]=np.clip(raw,1e-4,1-1e-4)
        yy=np.log(np.maximum(d['out'][:,m],1))
        ridge=RidgeCV(alphas=np.geomspace(1e1,1e7,13)).fit(F[tr],yy[tr])
        yh=ridge.predict(F)
        tok=np.exp(yh)*np.mean(np.exp(yy[tr]-yh[tr]))
        tok*=d['out'][tr,m].mean()/tok[tr].mean()
        C[:,m]=d['inp'][:,m]*d['pin'][m]+tok*d['pout'][m]
        settings.append({'C':best,'alpha':float(ridge.alpha_)})
    return P,C,settings


def draw_means(d,ii,rng):
    q,c=np.zeros((len(ii),d['n'].shape[1])),np.zeros((len(ii),d['n'].shape[1]))
    for m in range(q.shape[1]):
        for count in np.unique(d['n'][ii,m]):
            if not count: continue
            rows=np.flatnonzero(d['n'][ii,m]==count)
            idx=np.array([np.flatnonzero(d['valid'][i,m]) for i in ii[rows]])
            sampled=np.take_along_axis(idx,rng.integers(count,size=idx.shape),1)
            q[rows,m]=d['ok'][ii[rows,None],m,sampled].mean(1)
            c[rows,m]=(d['pt'][ii[rows,None],m,sampled]*d['pin'][m]+d['ct'][ii[rows,None],m,sampled]*d['pout'][m]).mean(1)
    return q,c


def summarize(point,boot):
    values=boot[:,2]
    if not np.isfinite(boot).all(): raise ValueError('Undefined bootstrap frontier; inspect saved data')
    p=(1+np.sum(values-point[2]>=point[2]))/(len(values)+1)
    return {'ours_pp':float(point[0]),'zr_pp':float(point[1]),'difference_pp':float(point[2]),
            'ci95_pp':np.percentile(values,[2.5,97.5]).tolist(),
            'ci90_pp':np.percentile(values,[5,95]).tolist(),
            'bootstrap_sd_pp':float(values.std(ddof=1)), 'one_sided_centered_bootstrap_p':float(p)}


def boot_problem(d,ch,ii,rng):
    draws=np.zeros((B,2,3))
    for b in range(B):
        H=curves(d,ch,rng.choice(ii,len(ii)))
        for common in (0,1): draws[b,common]=effects(H,common)
    return draws


def run(pool):
    started=time.time(); d=load(pool); label=d['label']; print(label,'loaded',flush=True)
    tr,cal,te=[d['split'][k] for k in ['train','calibration','test']]
    fixed=json.loads((HERE/'zr_best_ci.json').read_text())[label]['calibration-chosen']['config']
    (Pz,Cz),selection=zr(d,tr,cal,fixed)
    ch=[choices(d,d['P'],d['C']),choices(d,Pz,Cz),choices(d,d['P'],reference(d,tr))]
    H=curves(d,ch,te); points=np.array([effects(H,i) for i in (0,1)])
    rng=np.random.default_rng(20261001)
    bs=boot_problem(d,ch,te,rng)
    result={'fixed':{tag:summarize(points[i],bs[:,i]) for i,tag in enumerate(['original_bands','common_band'])},'fixed_configuration':fixed}
    print(label,'fixed',result['fixed'],flush=True)
    within=np.zeros((B,3)); nested=np.zeros((200,5,3))
    for b in range(B): within[b]=effects(curves(d,ch,te,*draw_means(d,te,rng)))
    for b in range(200):
        ii=rng.choice(te,len(te))
        for j in range(5): nested[b,j]=effects(curves(d,ch,ii,*draw_means(d,ii,rng)))
    result['uncertainty_diagnostic']={'problem_only_sd_pp':float(bs[:,0,2].std(ddof=1)),
       'generation_only_sd_pp':float(within[:,2].std(ddof=1)),
       'nested_total_sd_pp':float(nested[:,:,2].ravel().std(ddof=1)),
       'nested_mean_inner_variance_pp2':float(nested[:,:,2].var(axis=1,ddof=1).mean()),
       'nested_variance_of_outer_means_pp2':float(nested[:,:,2].mean(axis=1).var(ddof=1)),
       'warning':'Descriptive resampling variances, not identified population variance components.'}
    np.savez_compressed(DEST/f'{label}_fixed_bootstrap.npz',problem=bs,within=within,nested=nested)
    (DEST/f'{label}.json').write_text(json.dumps(result,indent=2))
    print(label,'diagnostic',result['uncertainty_diagnostic'],flush=True)
    originalP=d['P'].copy()
    pred=[np.zeros_like(d['Q']) for _ in range(5)]
    foldid=np.full(len(d['ids']),-1); folds=[]
    for fold,(rest,test) in enumerate(KFold(5,shuffle=True,random_state=17).split(d['ids'])):
        frac=len(cal)/(len(tr)+len(cal))
        train,ca=train_test_split(rest,test_size=frac,random_state=1700+fold)
        assert not (set(train)&set(test) or set(ca)&set(test) or set(train)&set(ca))
        print(label,'fold',fold,'linear fit',flush=True)
        P,C,settings=linear(d,train,ca); d['P']=P
        (Pz,Cz),selection=zr(d,train,ca)
        for store,values in zip(pred,[P,C,Pz,Cz,reference(d,train)]): store[test]=values[test]
        foldid[test]=fold
        folds.append({'fold':fold,'train':train.tolist(),'calibration':ca.tolist(),'test':test.tolist(),'linear':settings,'zr':selection})
        print(label,'fold',fold,'done',selection['chosen'],flush=True)
        np.savez_compressed(DEST/f'{label}_crossfit_predictions.npz',P=pred[0],C=pred[1],Pzr=pred[2],Czr=pred[3],Cref=pred[4],fold=foldid,ids=np.array(d['ids']))
        (DEST/f'{label}_folds.json').write_text(json.dumps(folds,indent=2))
    ii=np.arange(len(d['ids']))
    ch=[choices(d,pred[0],pred[1]),choices(d,pred[2],pred[3]),choices(d,pred[0],pred[4])]
    H=curves(d,ch,ii); points=np.array([effects(H,i) for i in (0,1)])
    bs=boot_problem(d,ch,ii,rng)
    result['crossfit_conditional']={tag:summarize(points[i],bs[:,i]) for i,tag in enumerate(['original_bands','common_band'])}
    result['crossfit_warning']='Conditional bootstrap omits training variability and fold dependence; does not establish algorithm-level significance.'
    result['elapsed_seconds']=time.time()-started
    np.savez_compressed(DEST/f'{label}_crossfit_bootstrap.npz',problem=bs)
    (DEST/f'{label}.json').write_text(json.dumps(result,indent=2))
    print(label,'FINISHED',result['crossfit_conditional'],flush=True)


def aggregate():
    data={p[0]:json.loads((DEST/f'{p[0]}.json').read_text()) for p in POOLS}
    pooled={}
    for band in ['original_bands','common_band']:
        ps=[data[p[0]]['fixed'][band]['one_sided_centered_bootstrap_p'] for p in POOLS]
        z=float(np.sum(norm.isf(np.clip(ps,1e-12,1-1e-12)))/np.sqrt(3))
        pooled[band]={'individual_p':dict(zip([p[0] for p in POOLS],ps)),'z':z,'one_sided_p':float(norm.sf(z))}
    equivalence={}
    for split in ['fixed','crossfit_conditional']:
        v=data['Omni'][split]['original_bands']; effect,se=v['difference_pp'],v['bootstrap_sd_pp']
        p=max(float(norm.sf((effect+5)/se)),float(norm.cdf((effect-5)/se)))
        equivalence[split]={'margin_pp':5,'ci90_pp':v['ci90_pp'],'percentile_equivalent':v['ci90_pp'][0]>-5 and v['ci90_pp'][1]<5,'normal_approximation_tost_p':p}
    summary={'datasets':data,'exploratory_stouffer':pooled,'omni_equivalence':equivalence}
    (DEST/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--pool',choices=[p[0] for p in POOLS]); parser.add_argument('--aggregate',action='store_true'); args=parser.parse_args()
    DEST.mkdir(exist_ok=True)
    if args.aggregate: aggregate()
    else:
        for pool in POOLS:
            if args.pool is None or args.pool==pool[0]: run(pool)
