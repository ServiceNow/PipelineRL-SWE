"""Reject expansion inference if re-extracted original prompts change forecasts."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from baseline_cost_heads import rich

R = Path('/mnt/llmd/results/exps/aristides/reason')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['mmlupro', 'omni500'], required=True)
    label = ap.parse_args().dataset
    root = R / 'prefill_anchor_verified_20261001'
    original = R / ('mmlupro_probe/instruct.npz' if label == 'mmlupro'
                    else 'omni500_probe/thinking.npz')
    current = root / f'{label}.npz'
    old, new = np.load(original, allow_pickle=True), np.load(current, allow_pickle=True)
    for key in ('layers', 'model', 'system_prompt', 'user_suffix'):
        if not np.array_equal(old[key], new[key]):
            raise ValueError(f'Anchor metadata differs: {key}')
    ids = list(map(str, new['problem_ids']))
    if len(ids) != 32 or len(set(ids)) != 32:
        raise ValueError('Expected 32 distinct original anchor problems')
    source = {str(row['problem_id']): row['prompt'] for row in
              map(json.loads, (R / f'{label}_probe_prompts.jsonl').read_text().splitlines())}
    prompts_path = root / f'{label}_prompts.jsonl'
    prompts = list(map(json.loads, prompts_path.read_text().splitlines()))
    if [str(row['problem_id']) for row in prompts] != ids:
        raise ValueError('Anchor prompt IDs differ from extracted IDs')
    if any(row['prompt'] != source[str(row['problem_id'])] for row in prompts):
        raise ValueError('Anchor prompt text differs from the original extraction input')
    x_old, x_new = rich(original, ids), rich(current, ids)
    raw_relative_rms = float(np.linalg.norm(x_new.astype(float) - x_old) /
                             max(np.linalg.norm(x_old.astype(float)), 1e-12))
    saved = joblib.load(R / 'expanded_eval_20261001' / label / 'paper_cost_heads.joblib')
    relative = []
    for head in saved['heads']:
        def forecast(x):
            return np.exp(head['model'].predict(head['scaler'].transform(x))) * head['smear'] * head['level']
        old_tokens, new_tokens = forecast(x_old), forecast(x_new)
        relative.append(np.abs(new_tokens - old_tokens) / np.maximum(old_tokens, 1e-12))
    relative = np.asarray(relative)
    # Numerical reproduction tolerances, fixed independently of expansion outcomes.
    passed = bool(raw_relative_rms <= .05 and relative.max() <= .05 and np.median(relative) <= .01)
    report = dict(dataset=label, anchors=len(ids), original_feature_file=str(original),
                  reextracted_feature_file=str(current),
                  prompt_sha256=hashlib.sha256(prompts_path.read_bytes()).hexdigest(),
                  relative_feature_rms=raw_relative_rms,
                  max_relative_output_forecast_error=float(relative.max()),
                  median_relative_output_forecast_error=float(np.median(relative)),
                  tolerances=dict(relative_feature_rms=.05, max_relative_forecast=.05,
                                  median_relative_forecast=.01), passed=passed)
    (root / f'{label}_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    if not passed:
        raise ValueError('Original-feature anchor replay failed; expansion inference stopped')


if __name__ == '__main__':
    main()
