"""Reconstruct the archived train-only RidgeCV head before scoring expansion data.

This intentionally follows baseline_cost_heads.py's probe, without the extra
calibration/target-space transformations in activation_cost_preds.py.
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from baseline_cost_heads import rich
from carrot_compare import POOLS, read_predictions
from decompose import MK, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['mmlupro', 'omni500'], required=True)
    a = ap.parse_args()
    pool = 'MMLU-Pro' if a.dataset == 'mmlupro' else 'Omni'
    name, archived_file = POOLS[pool]
    original = R / name
    expanded = R / 'expanded_eval_20261001' / a.dataset
    features = R / ('mmlupro_probe/instruct.npz' if a.dataset == 'mmlupro'
                    else 'omni500_probe/thinking.npz')
    t = np.load(original / 'tensors.npz', allow_pickle=True)
    ids = list(map(str, t['problem_ids']))
    slots = list(map(str, t['model_slots']))
    split = json.loads((original / 'split_manifest.json').read_text())
    index = {p: i for i, p in enumerate(ids)}
    train = np.array([index[str(p)] for p in split['train_problem_ids']])
    valid = t['valid'].astype(bool)
    counts = valid.sum(2)
    lengths = np.where(valid, t['completion_tokens'], 0).sum(2) / np.maximum(counts, 1)
    X = rich(features, ids)
    combined = np.load(expanded / 'tensors.npz', allow_pickle=True)
    all_ids = list(map(str, combined['problem_ids']))
    if all_ids[:len(ids)] != ids or list(map(str, combined['model_slots'])) != slots:
        raise ValueError('Original IDs or route order changed')
    all_X = rich(expanded / 'prefill_combined.npz', all_ids)
    if not np.array_equal(X, all_X[:len(ids)]):
        raise ValueError('Original feature values changed')
    all_valid = combined['valid'].astype(bool)
    inputs = np.where(all_valid, combined['prompt_tokens'], 0).sum(2) / np.maximum(all_valid.sum(2), 1)
    rates = dict(MK)
    price_path = original / 'prices.json'
    if price_path.exists():
        rates.update(json.loads(price_path.read_text()))
    forecasts = np.zeros((len(all_ids), len(slots)))
    heads, settings = [], []
    for j, slot in enumerate(slots):
        tr = train[counts[train, j] > 0]
        scaler = StandardScaler().fit(X[tr])
        Xs = scaler.transform(X)
        y = np.log(np.maximum(lengths[:, j], 1.0))
        model = RidgeCV(alphas=np.geomspace(1e1, 1e7, 13)).fit(Xs[tr], y[tr])
        old_log = model.predict(Xs)
        smear = float(np.mean(np.exp(y[tr] - old_log[tr])))
        level = float(lengths[tr, j].mean() / (np.exp(old_log[tr]) * smear).mean())
        log_pred = model.predict(scaler.transform(all_X))
        tokens = np.exp(log_pred) * smear * level
        pin, pout = np.asarray(rates[slot]) / 1e6
        forecasts[:, j] = inputs[:, j] * pin + tokens * pout
        heads.append(dict(scaler=scaler, model=model, smear=smear, level=level))
        settings.append(dict(route=slot, alpha=float(model.alpha_), smear=smear, level=level))
        print(json.dumps(settings[-1]), flush=True)
    archived = read_predictions(original / archived_file, ids, 'expected_costs', len(slots))
    error = np.abs(forecasts[:len(ids)] - archived)
    relative = error / np.maximum(np.abs(archived), 1e-12)
    report = dict(dataset=pool, train_problems=len(train), settings=settings,
                  max_absolute_error_usd=float(error.max()), max_relative_error=float(relative.max()),
                  mean_relative_error=float(relative.mean()),
                  per_route_max_relative_error=dict(zip(slots, relative.max(0).tolist())),
                  archived_predictions_sha256=hashlib.sha256((original / archived_file).read_bytes()).hexdigest(),
                  protocol='Train-only RidgeCV, original features and labels, no expansion fitting or calibration; Duan smearing and training-mean matching. Same success predictions held fixed across cost arms.')
    (expanded / 'paper_cost_reconstruction.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    if relative.max() > 1e-3:
        raise ValueError('Reconstructed head failed to reproduce archived predictions; fresh scoring invalid')
    dest = expanded / 'paper_cost_preds.jsonl'
    dest.write_text(''.join(json.dumps(dict(problem_id=p, expected_costs=c.tolist())) + '\n'
                           for p, c in zip(all_ids, forecasts)))
    joblib.dump(dict(heads=heads, slots=slots, train_ids=split['train_problem_ids'],
                     feature_keys=['mean', 'last'], settings=settings), expanded / 'paper_cost_heads.joblib')


if __name__ == '__main__':
    main()
