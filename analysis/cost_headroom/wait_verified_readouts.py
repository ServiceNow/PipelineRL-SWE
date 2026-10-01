"""Run CPU scoring once the guarded expansion feature archive is complete."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

import numpy as np

R = Path('/mnt/llmd/results/exps/aristides/reason')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['mmlupro', 'omni500'], required=True)
    label = ap.parse_args().dataset
    anchor = R / 'prefill_anchor_verified_20261001' / f'{label}_verification.json'
    feature = R / 'expansion_prefills_verified_20261001' / f'{label}_prefill.npz'
    expected = {'mmlupro': 6500, 'omni500': 1000}[label]
    deadline = time.monotonic() + 7200
    print(f'Waiting up to two hours for verified {label} prefills', flush=True)
    while time.monotonic() < deadline:
        if anchor.exists():
            if not json.loads(anchor.read_text())['passed']:
                raise ValueError('Original-prompt anchor failed; scoring cancelled')
            if feature.exists():
                try:
                    # ZIP central directory is written last, after all arrays.
                    with np.load(feature, allow_pickle=True) as z:
                        if len(z['problem_ids']) != expected:
                            raise ValueError('Expansion feature count differs from frozen plan')
                        if not {'mean', 'last'} <= set(z.files):
                            raise ValueError('Required rich features missing')
                    break
                except (zipfile.BadZipFile, EOFError):
                    pass
        time.sleep(30)
    else:
        raise TimeoutError('Corrected prefills were not ready within two hours')
    os.environ['OPENBLAS_NUM_THREADS'] = '8'
    os.environ['OMP_NUM_THREADS'] = '8'
    subprocess.run(['bash', 'analysis/cost_headroom/run_expanded_readouts.sh', label], check=True)


if __name__ == '__main__':
    main()
