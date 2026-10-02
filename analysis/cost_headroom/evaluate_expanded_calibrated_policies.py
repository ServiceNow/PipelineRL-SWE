"""Freeze accuracy-target policies on original calibration, score new problems.

The target grid is the existing costhead_matched_accuracy_ci.py grid. Fresh
outcomes never select V, mixture weights, targets, or readout parameters.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from carrot_compare import POOLS, read_predictions
from decompose import MK, R, hull

TARGETS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
VALUES = np.geomspace(1e-5, 100, 400)  # cents per correct answer, historical grid


def policy(p, cost, q, paid, target, deterministic=False):
    choices = (VALUES[:, None, None] * p[None] - cost[None]).argmax(2)
    rows = np.arange(len(p))[None]
    accuracy, spending = q[rows, choices].mean(1), paid[rows, choices].mean(1)
    if deterministic:
        eligible = np.flatnonzero(accuracy >= target - 1e-12)
        if not len(eligible):
            return None
        i = int(eligible[np.argmin(spending[eligible])])
        return dict(V_cents=[float(VALUES[i]), float(VALUES[i])], upper_weight=0.0,
                    calibration_target=target, calibration_accuracy=float(accuracy[i]),
                    calibration_mean_cost_cents=float(spending[i]), policy_type='deterministic')
    h = hull(zip(spending, accuracy))
    if not h[0][1] <= target <= h[-1][1]:
        return None
    if target == h[0][1]:
        edges = [h[0], h[0]]
    else:
        edges = next(([left, right] for left, right in zip(h, h[1:])
                      if left[1] <= target <= right[1]), None)
    if edges is None:
        return None
    indices = [int(np.argmin((spending-c)**2 + (accuracy-a)**2)) for c, a in edges]
    weight = ((target-edges[0][1]) / (edges[1][1]-edges[0][1])
              if edges[1][1] > edges[0][1] else 0.0)
    return dict(V_cents=[float(VALUES[i]) for i in indices],
                upper_weight=float(weight), calibration_target=target, policy_type='randomized')


def outcomes(p, cost, q, paid, selected):
    rows = np.arange(len(p))
    choices = [(value*p-cost).argmax(1) for value in selected['V_cents']]
    w = selected['upper_weight']
    return ((1-w)*q[rows, choices[0]] + w*q[rows, choices[1]],
            (1-w)*paid[rows, choices[0]] + w*paid[rows, choices[1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['mmlupro', 'omni500'], required=True)
    ap.add_argument('--policy-type', choices=['randomized', 'deterministic'], default='randomized')
    a = ap.parse_args()
    label = 'MMLU-Pro' if a.dataset == 'mmlupro' else 'Omni'
    name, cost_file = POOLS[label]
    old = R/name
    folder = R/'expanded_eval_20261001'/a.dataset
    t = np.load(folder/'tensors.npz', allow_pickle=True)
    ids, slots = list(map(str, t['problem_ids'])), list(map(str, t['model_slots']))
    split = json.loads((old/'split_manifest.json').read_text())
    index = {p:i for i,p in enumerate(ids)}
    tr, ca = [np.asarray([index[str(p)] for p in split[key+'_problem_ids']])
              for key in ['train', 'calibration']]
    n_old = len(np.load(old/'tensors.npz', allow_pickle=True)['problem_ids'])
    fresh = np.arange(n_old, len(ids))
    valid = t['valid'].astype(bool)
    counts = valid.sum(2)
    if not (counts > 0).all():
        raise ValueError('Missing observations for some route/problem')
    q = np.where(valid, t['final_outcome'], 0).sum(2)/counts
    lengths = np.where(valid, t['completion_tokens'], 0).sum(2)/counts
    inputs = np.where(valid, t['prompt_tokens'], 0).sum(2)/counts
    rates = dict(MK)
    if (old/'prices.json').exists():
        rates.update(json.loads((old/'prices.json').read_text()))
    pin = np.array([rates[s][0] for s in slots])/1e6
    pout = np.array([rates[s][1] for s in slots])/1e6
    paid = (inputs*pin+lengths*pout)*100
    median = np.asarray([np.median(t['completion_tokens'][tr,j][valid[tr,j]])
                         for j in range(len(slots))])
    average = lengths[tr].mean(0)
    costs = {'learned':read_predictions(folder/'paper_cost_preds.jsonl', ids,
                                       'expected_costs', len(slots))*100,
             'median':(inputs*pin+median*pout)*100,
             'mean':(inputs*pin+average*pout)*100}
    p = read_predictions(folder/'success_preds.jsonl', ids, 'p_successes', len(slots))
    archived_p = read_predictions(old/'content_preds.jsonl', ids[:n_old],
                                  'p_successes', len(slots))
    discrepancy = float(np.abs(p[:n_old]-archived_p).max())
    # Calibration policy selection reproduces historical scores exactly.
    p[:n_old] = archived_p
    costs['learned'][:n_old] = read_predictions(old/cost_file, ids[:n_old],
                                              'expected_costs', len(slots))*100
    reconstruction = json.loads((folder/'paper_cost_reconstruction.json').read_text())
    if reconstruction['max_relative_error'] > 1e-3:
        raise ValueError('Unverified reconstructed cost head')
    problems = [json.loads(line) for line in (folder/'problems.jsonl').read_text().splitlines()]
    strata = np.asarray([str(x.get('subject','')) if a.dataset=='mmlupro'
                         else str(round(float(x.get('difficulty',0))))
                         for x in problems[n_old:]])
    weights = json.loads((folder/'expansion_manifest.json').read_text())['stratum_weights']
    groups = {s:np.flatnonzero(strata==s) for s in weights}
    rng = np.random.default_rng(20261004)
    # Same paired, stratified samples for every estimator and operating point.
    draws = [{s:ix[rng.integers(0,len(ix),len(ix))] for s,ix in groups.items()}
             for _ in range(2000)]
    def mean(x, sample=None):
        return sum(float(weights[s])*float(x[(sample or groups)[s]].mean()) for s in groups)
    results = {}
    for target in TARGETS:
        selected = {arm:policy(p[ca],c[ca],q[ca],paid[ca],target,
                              deterministic=a.policy_type=='deterministic') for arm,c in costs.items()}
        observed = {arm:outcomes(p[fresh],costs[arm][fresh],q[fresh],paid[fresh],config)
                    for arm,config in selected.items() if config is not None}
        arms = {}
        for arm,(accuracy,spend) in observed.items():
            arms[arm] = dict(policy=selected[arm], accuracy=mean(accuracy),
                             mean_cost_usd=mean(spend)/100)
        comparisons = {}
        if 'learned' in observed:
            al,cl = observed['learned']
            for base in ['median','mean']:
                if base not in observed:
                    continue
                ab,cb = observed[base]
                boot_cost = [1-mean(cl,b)/mean(cb,b) for b in draws]
                boot_accuracy = [mean(al-ab,b) for b in draws]
                comparisons[base] = dict(savings=1-mean(cl)/mean(cb),
                    savings_ci95=np.percentile(boot_cost,[2.5,97.5]).tolist(),
                    accuracy_delta=mean(al-ab),
                    accuracy_delta_ci95=np.percentile(boot_accuracy,[2.5,97.5]).tolist())
        results[str(target)] = dict(arms=arms, comparisons=comparisons,
                                    unreachable_on_calibration=[arm for arm,v in selected.items() if v is None])
    report = dict(dataset=label, n_fresh=len(fresh), targets=TARGETS, policy_type=a.policy_type,
                  original_success_prediction_max_absolute_discrepancy=discrepancy,
                  cost_reconstruction=reconstruction, results=results,
                  protocol=('Historical target grid reused. '+
                    ('Select the cheapest single-V policy meeting each accuracy target on original calibration; no randomized routing. This deterministic follow-up was specified after the randomized fresh results were observed, in response to a request for a simpler deployable policy; all reachable targets are reported.'
                     if a.policy_type=='deterministic' else
                     'Policy Vs and mixture weights selected only on original calibration.')+
                    ' Same success scores and realized market spending for all arms. Median and mean length references fitted on original training only. Fresh outcomes used solely for achieved accuracy/cost and 2,000 paired stratified problem-bootstrap CIs. No accuracy-equivalence claim merely from non-significance. Pointwise intervals, no multiplicity adjustment; training and calibration held fixed. Report transparently as a corrected follow-up, not a new preregistration.'))
    filename=('paper_deterministic_policy_results.json' if a.policy_type=='deterministic'
              else 'paper_calibrated_policy_results.json')
    (folder/filename).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
