"""CARROT-KNN-SBERT comparison on frozen original splits; no API calls.
Upstream: somerstep/CARROT at 3e6acff6aecf4cbcb8f31a118d04c799c2ea1655.
Cosine uniform kNN, separate multi-output success/output-token regressors,
5-fold training CV maximizing mean route R2 over powers-of-two k.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decompose import R, MK, hull, cost_at

POOLS = {
    'LCB': ('pool_v2_tensors_5rung', 'cost_preds_probe.jsonl'),
    'Omni': ('omni500_tensors', 'cost_preds_probe_thinking.jsonl'),
    'MMLU-Pro': ('mmlupro_tensors', 'cost_preds_probe_instruct.jsonl'),
}
UPSTREAM = '3e6acff6aecf4cbcb8f31a118d04c799c2ea1655'
MODEL = 'sentence-transformers/all-MiniLM-L12-v2'
VS = np.geomspace(1e-5, 100, 300)


def embeddings(texts, ids, out):
    cache = out / 'embeddings.npz'
    digest = hashlib.sha256(json.dumps(list(zip(ids, texts)), ensure_ascii=False).encode()).hexdigest()
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if str(z['text_sha256']) != digest or z['problem_ids'].tolist() != ids:
            raise ValueError('Embedding cache does not match problem texts/order')
        return z['embeddings']
    import torch
    torch.set_num_threads(8)
    # Transformers imports installed DeepSpeed even for inference. Its Triton
    # initialization fails without GPUs in this environment; it is unused here.
    import transformers.integrations.deepspeed as ds
    ds.is_deepspeed_available = lambda: False
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL, device='cpu')
    x = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    np.savez_compressed(cache, embeddings=x, problem_ids=np.asarray(ids), text_sha256=digest,
                        model=MODEL, max_seq_length=model.max_seq_length)
    return x


def fit_knn(x, y, tr):
    # Restrict k so every CV training fold can support it. No calibration/test labels.
    folds = list(KFold(5, shuffle=False).split(tr))
    limit = min(len(a) for a, _ in folds)
    scores = {}
    for k in [2**i for i in range(1, 10) if 2**i <= limit]:
        values = []
        for a, b in folds:
            model = KNeighborsRegressor(k, metric='cosine', weights='uniform', n_jobs=4)
            model.fit(x[tr[a]], y[tr[a]])
            values.append(r2_score(y[tr[b]], model.predict(x[tr[b]]), multioutput='uniform_average'))
        scores[k] = float(np.mean(values))
    chosen = max(scores, key=scores.get)
    model = KNeighborsRegressor(chosen, metric='cosine', weights='uniform', n_jobs=4)
    return model.fit(x[tr], y[tr]).predict(x), {'neighbors': chosen, 'cv_r2': scores}


def read_predictions(path, ids, key, m):
    rows = {str(r['problem_id']): r[key][:m] for r in map(json.loads, path.open())}
    return np.asarray([rows[p] for p in ids], dtype=float)


def curve_arrays(p, c, q, paid, te):
    choices = (VS[:, None, None] * p[te][None] - c[te][None]).argmax(2)
    rows = np.arange(len(te))[None]
    return q[te][rows, choices], paid[te][rows, choices]


def compare_pair(curves, left, right, indices):
    hs = {key: hull(list(zip(curves[key][1][:, indices].mean(1),
                            curves[key][0][:, indices].mean(1)))) for key in (left, right)}
    lo = max(h[0][1] for h in hs.values()); hi = min(h[-1][1] for h in hs.values())
    if hi <= lo:
        return np.nan, [lo, hi]
    targets = np.linspace(lo + .05*(hi-lo), hi - .05*(hi-lo), 12)
    costs = {key: np.asarray([cost_at(h, a) for a in targets]) for key, h in hs.items()}
    return float(1-np.exp(np.mean(np.log(costs[left]/costs[right])))), [float(targets[0]), float(targets[-1])]


def run(args, label):
    name, costfile = POOLS[label]
    folder = R / name
    out = args.out / label.lower().replace('-', '_'); out.mkdir(parents=True, exist_ok=True)
    t = np.load(folder/'tensors.npz', allow_pickle=True)
    ids = [str(p) for p in t['problem_ids']]; slots = [str(s) for s in t['model_slots']]
    pi = {p:i for i,p in enumerate(ids)}; m = len(slots)
    split = json.loads((folder/'split_manifest.json').read_text())
    tr = np.asarray([pi[str(p)] for p in split['train_problem_ids']])
    te = np.asarray([pi[str(p)] for p in split['test_problem_ids']])
    valid = t['valid'].astype(bool); n = valid.sum(2)
    # All compared routes need observed labels. Do not silently impute or drop problems.
    if not (n[tr] > 0).all() or not (n[te] > 0).all():
        raise ValueError('A train/test problem has no valid draws for a route')
    q = np.where(valid, t['final_outcome'], 0).sum(2)/np.maximum(n,1)
    length = np.where(valid, t['completion_tokens'], 0).sum(2)/np.maximum(n,1)
    inp = np.where(valid, t['prompt_tokens'], 0).sum(2)/np.maximum(n,1)
    if not (n > 0).all():
        raise ValueError('Missing route labels in full prediction pool')
    rates = dict(MK)
    if (folder/'prices.json').exists():
        rates.update(json.loads((folder/'prices.json').read_text()))
    pin = np.asarray([rates[s][0] for s in slots])/1e6
    pout = np.asarray([rates[s][1] for s in slots])/1e6
    paid = (inp*pin + length*pout)*100
    p_ours = read_predictions(folder/'content_preds.jsonl', ids, 'p_successes', m)
    c_ours = read_predictions(folder/costfile, ids, 'expected_costs', m)*100
    meta = {str(r['problem_id']):r for r in map(json.loads, (folder/'problems.jsonl').open())}
    texts = [str(meta[p]['problem_statement']) for p in ids]
    x = embeddings(texts, ids, out)
    p_carrot, pspec = fit_knn(x, q, tr)
    l_carrot, cspec = fit_knn(x, length, tr)
    c_carrot = (inp*pin + l_carrot*pout)*100
    mean_cost = (inp*pin + length[tr].mean(0)*pout)*100
    median_length = np.asarray([np.median(t['completion_tokens'][tr,j][valid[tr,j]]) for j in range(m)])
    median_cost = (inp*pin + median_length*pout)*100
    arms = {
        'ours': (p_ours, c_ours),
        'ours_success_median_cost': (p_ours, median_cost),
        'ours_success_carrot_cost': (p_ours, c_carrot),
        'carrot': (p_carrot, c_carrot),
        'carrot_success_mean_cost': (p_carrot, mean_cost),
        'carrot_success_our_cost': (p_carrot, c_ours),
    }
    curves = {key:curve_arrays(p,c,q,paid,te) for key,(p,c) in arms.items()}
    pairs = {
        'ours_vs_carrot_cost_fixed_success': ('ours','ours_success_carrot_cost'),
        'ours_vs_carrot_full': ('ours','carrot'),
        'carrot_cost_vs_constant_fixed_success': ('carrot','carrot_success_mean_cost'),
        'our_cost_swap_into_carrot': ('carrot_success_our_cost','carrot'),
        'ours_vs_median': ('ours','ours_success_median_cost'),
        'carrot_cost_vs_median_our_success': ('ours_success_carrot_cost','ours_success_median_cost'),
    }
    rng = np.random.default_rng(args.seed)
    resamples = [rng.integers(0,len(te),len(te)) for _ in range(args.bootstrap)]
    comparisons = {}
    for tag,(left,right) in pairs.items():
        point,band = compare_pair(curves,left,right,np.arange(len(te)))
        boot = np.asarray([compare_pair(curves,left,right,ii)[0] for ii in resamples])
        finite = np.isfinite(boot)
        comparisons[tag] = {'left':left,'right':right,'direct_cost_saved':point,'accuracy_band':band,
                            'ci95':np.percentile(boot[finite],[2.5,97.5]).tolist() if finite.any() else None,
                            'valid_bootstrap':int(finite.sum()),'bootstrap':boot.tolist()}
        print(label, tag, json.dumps({k:v for k,v in comparisons[tag].items() if k!='bootstrap'}),flush=True)
    np.savez_compressed(out/'predictions.npz', problem_ids=np.asarray(ids), model_slots=np.asarray(slots),
                        train_indices=tr,test_indices=te,carrot_success=p_carrot,
                        carrot_output_tokens=l_carrot,carrot_cost_dollars=c_carrot/100)
    report = {'pool':label,'source_pool':name,'upstream_commit':UPSTREAM,'variant':'CARROT-KNN-SBERT',
              'embedding_model':MODEL,'n_train':len(tr),'n_test':len(te),'success_knn':pspec,'cost_knn':cspec,
              'route_log_length_r2':dict(zip(slots,[float(r2_score(np.log(np.maximum(length[te,j],1)),
                                                       np.log(np.maximum(l_carrot[te,j],1)))) for j in range(m)])),
              'comparisons':comparisons,
              'protocol':'All valid draw means; raw output-token regression; original frozen train/test splits; training-only 5-fold k selection; paired problem bootstrap conditional on fitted predictors; each contrast uses its own common band and reports direct geometric-mean cost savings, not differences of separately normalized gains; generation spend excludes encoder overhead.'}
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', choices=list(POOLS)+['all'],default='all')
    parser.add_argument('--out', type=Path,required=True)
    parser.add_argument('--bootstrap',type=int,default=1000)
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    if args.bootstrap<1:parser.error('--bootstrap must be positive')
    for label in (POOLS if args.pool=='all' else [args.pool]):run(args,label)

if __name__=='__main__':main()
