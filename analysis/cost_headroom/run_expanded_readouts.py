"""Assemble fresh features and apply the original frozen readout fitting protocol."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

R = Path('/mnt/llmd/results/exps/aristides/reason')
EXP = R / 'expansion_prefills_verified_20261001'
OUT = R / 'expanded_eval_20261001'
REPO = Path(__file__).resolve().parents[2]


def run(cmd):
    print('+', ' '.join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=REPO, check=True)


def combine_features(label):
    old_path = (R / 'mmlupro_probe/instruct.npz' if label == 'mmlupro'
                else R / 'omni500_probe/thinking.npz')
    new_path = EXP / f'{label}_prefill.npz'
    if not old_path.exists() or not new_path.exists():
        raise FileNotFoundError(f'Missing old/new feature file: {old_path} / {new_path}')
    old = np.load(old_path, allow_pickle=True)
    new = np.load(new_path, allow_pickle=True)
    old_ids = [str(x) for x in old['problem_ids']]
    new_ids = [str(x) for x in new['problem_ids']]
    if len(set(old_ids) & set(new_ids)):
        raise ValueError('New feature IDs overlap old features')
    for key in ('layers', 'model', 'scalar_names', 'system_prompt', 'user_suffix'):
        if key in old.files and key in new.files and not np.array_equal(old[key], new[key]):
            raise ValueError(f'Feature metadata differs: {key}')
    out = {}
    for key in old.files:
        # Rich heads consume mean+last only; omit duplicate/unused readouts.
        if key in ('pre','content_last','content_mean'):
            continue
        if key == 'problem_ids':
            out[key] = np.asarray(old_ids + new_ids, dtype=str)
        elif key in new.files and old[key].ndim > 0 and new[key].ndim > 0 and old[key].shape[0] == len(old_ids) and new[key].shape[0] == len(new_ids):
            if old[key].shape[1:] != new[key].shape[1:]:
                raise ValueError(f'Feature shape mismatch for {key}: {old[key].shape}, {new[key].shape}')
            out[key] = np.concatenate([old[key], new[key]], axis=0)
        else:
            out[key] = old[key]
    missing = set(old.files) - set(out) - {'pre', 'content_last', 'content_mean'}
    if missing:
        raise ValueError(f'Failed to assemble feature keys: {missing}')
    folder = OUT / label
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'prefill_combined.npz'
    tmp = folder / 'prefill_combined.tmp.npz'
    # Compression was the dominant previous CPU delay; these are local working features.
    np.savez(tmp, **out)
    tmp.replace(path)
    return path, len(old_ids), len(new_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['mmlupro', 'omni500'], required=True)
    args = ap.parse_args()
    label = args.dataset
    # The builder refuses to proceed until collection COMPLETE.json is present and
    # every planned route/problem has a valid response.
    run([sys.executable, 'analysis/cost_headroom/build_expanded_tensors.py', '--dataset', label])
    feature, n_old, n_new = combine_features(label)
    folder = OUT / label
    tensor_dir = folder
    success = folder / 'success_preds.jsonl'
    cost = folder / 'cost_preds.jsonl'
    scripts = REPO / 'pipelinerl/swe/scripts/livecodebench'
    run([sys.executable, scripts / 'activation_content_preds.py', '--activations', feature,
         '--rich', '--tensors-dir', tensor_dir, '--select-C', '--out', success])
    run([sys.executable, 'analysis/cost_headroom/reconstruct_paper_cost_heads.py', '--dataset', label])
    cost = folder / 'paper_cost_preds.jsonl'
    run([sys.executable, 'analysis/cost_headroom/evaluate_expanded_fixed_policy.py', '--dataset', label,
         '--cost-file',cost.name,'--output-file','verified_fixed_policy_results.json'])
    run([sys.executable, 'analysis/cost_headroom/evaluate_expanded_calibrated_policies.py', '--dataset', label])
    manifest = {'dataset': label, 'old_features': n_old, 'fresh_features': n_new,
                'combined_features': n_old + n_new, 'features': str(feature),
                'success_predictions': str(success), 'cost_predictions': str(cost),
                'readouts_fit_only_on_original_train_and_calibration_ids': True}
    (folder / 'readout_run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
