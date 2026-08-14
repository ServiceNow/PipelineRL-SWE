#!/usr/bin/env bash
# Train an in-domain abstention predictor on SWE-bench Verified instances.
#
# Designed for the mixture study:
#   - Pure Verified in-domain baseline
#   - Or mixed with SWE-Smith (see MIX_FRAC_VERIFIED below)
#
# Prerequisites (all 500 instances must be collected first):
#   - Scout trajectories (with test feedback) for all 500 Verified instances
#   - Real oracle (120b) labels (Daytona eval) for all 500 instances
#
# Data sources:
#   TRAJ_ALL16K   -- trajectories_with_testfb for 369 all_16k instances
#   TRAJ_131      -- trajectories for 131 newly collected instances
#   LABELS_369    -- real Daytona oracle labels for 369 instances
#   LABELS_131    -- real Daytona oracle labels for 131 new instances
#   SWESMITH_TRAJ -- SWE-Smith trajectories dir (for mixture training)
#   SWESMITH_PARQUET -- SWE-Smith labels parquet dir (for mixture training)
#
# Optional env vars:
#   MIX_FRAC_VERIFIED  -- fraction of Verified to mix in (0.0=pure SWE-Smith,
#                          1.0=pure Verified). Default: 1.0 (pure Verified)
#   TRAIN_FRAC         -- fraction of Verified data to use for training (default: 0.8)
#   SEED               -- random seed for train/eval split (default: 42)
#   INCLUDE_TEST_FEEDBACK -- true | false (default: true)
#   NUM_EPOCHS / LORA_R / etc. -- passed to underlying launcher
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
PYTHON=/home/toolkit/.conda/envs/pipeline-rl/bin/python3

TIMESTAMP=$(date +%s)

# Data inputs
TRAJ_ALL16K=${TRAJ_ALL16K:-/mnt/llmd/results/exps/aristides/reason/verified_scout_daytona_eval_1786402548/trajectories_with_testfb/trajectories_eval.jsonl}
TRAJ_131=${TRAJ_131:-}   # set after launch_verified_expand_to_500.sh completes
LABELS_369_DIR=${LABELS_369_DIR:-logs/run_evaluation/verified_oracle_eval_1786405908}
LABELS_131_RUN_ID=${LABELS_131_RUN_ID:-}   # set after expand job's daytona step completes

# SWE-Smith data (for mixture experiments)
SWESMITH_TRAJ=${SWESMITH_TRAJ:-/mnt/llmd/results/exps/aristides/reason/instruct_patches_trajectories_$(ls /mnt/llmd/results/exps/aristides/reason/ | grep instruct_patches_trajectories | tail -1 | grep -o '[0-9]*$')}
SWESMITH_PARQUET=${SWESMITH_PARQUET:-/mnt/llmd/results/exps/aristides/reason/offline_router_swe_smith_train1500_real_labels_4route_1780639659/collect}

MIX_FRAC_VERIFIED=${MIX_FRAC_VERIFIED:-1.0}  # 1.0 = pure Verified
TRAIN_FRAC=${TRAIN_FRAC:-0.8}
SEED=${SEED:-42}
INCLUDE_TEST_FEEDBACK=${INCLUDE_TEST_FEEDBACK:-true}
NUM_EPOCHS=${NUM_EPOCHS:-10}
LORA_R=${LORA_R:-32}

# Workspace for prepared data
PREP_DIR=/mnt/llmd/results/exps/aristides/reason/verified_indomain_prep_${TIMESTAMP}
mkdir -p "${PREP_DIR}"

echo "=== Preparing Verified in-domain training data (mix_frac=${MIX_FRAC_VERIFIED}) ==="

"${PYTHON}" - << PYEOF
import json, random, shutil
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

SEED = ${SEED}
TRAIN_FRAC = ${TRAIN_FRAC}
PREP_DIR = Path("${PREP_DIR}")
INCLUDE_TF = "${INCLUDE_TEST_FEEDBACK}" == "true"
MIX_FRAC = float("${MIX_FRAC_VERIFIED}")

# ── 1. Load Verified trajectories ────────────────────────────────────────────
traj_all16k = Path("${TRAJ_ALL16K}")
trajs = []
if traj_all16k.exists():
    for line in traj_all16k.read_text().splitlines():
        if line.strip():
            trajs.append(json.loads(line))
print(f"Loaded {len(trajs)} trajectories from all_16k")

traj_131_path = "${TRAJ_131}"
if traj_131_path:
    p = Path(traj_131_path)
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                trajs.append(json.loads(line))
        print(f"Loaded additional trajectories → total {len(trajs)}")

if not trajs:
    raise SystemExit("No trajectories found. Run data collection first.")

# ── 2. Load real oracle labels ───────────────────────────────────────────────
oracle_labels = {}
label_dir = Path("${LABELS_369_DIR}")
if label_dir.exists():
    for f in label_dir.glob("*.jsonl"):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                iid = r.get("instance_id") or r.get("problem_id")
                if iid:
                    oracle_labels[iid] = bool(r.get("resolved", False))
print(f"Loaded {len(oracle_labels)} oracle labels (369 set)")

labels_131_run = "${LABELS_131_RUN_ID}"
if labels_131_run:
    label_dir_131 = Path("logs/run_evaluation") / labels_131_run
    if label_dir_131.exists():
        for f in label_dir_131.glob("*.jsonl"):
            for line in f.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    iid = r.get("instance_id") or r.get("problem_id")
                    if iid:
                        oracle_labels[iid] = bool(r.get("resolved", False))
        print(f"Loaded additional oracle labels → total {len(oracle_labels)}")

# Keep only trajectories with oracle labels
trajs = [t for t in trajs if t["problem_id"] in oracle_labels]
print(f"Trajectories with oracle labels: {len(trajs)}")

# ── 3. Train/eval split ───────────────────────────────────────────────────────
rng = random.Random(SEED)
pids = [t["problem_id"] for t in trajs]
rng.shuffle(pids)
n_train = int(len(pids) * TRAIN_FRAC)
train_pids = set(pids[:n_train])
eval_pids = set(pids[n_train:])
print(f"Verified split: {len(train_pids)} train, {len(eval_pids)} eval")

train_trajs = [t for t in trajs if t["problem_id"] in train_pids]
eval_trajs = [t for t in trajs if t["problem_id"] in eval_pids]

# ── 4. Write trajectory JSONL files ──────────────────────────────────────────
traj_dir = PREP_DIR / "trajectories"
traj_dir.mkdir(parents=True, exist_ok=True)

def write_trajs(trajs_list, path):
    with open(path, "w") as f:
        for t in trajs_list:
            out = {k: v for k, v in t.items()}
            if not INCLUDE_TF:
                out.pop("test_feedback", None)
                out.pop("_tf_failing", None)
                out.pop("_tf_passing", None)
                out.pop("_tf_resolved", None)
                out.pop("_tf_patch_exists", None)
            f.write(json.dumps(out) + "\n")

write_trajs(train_trajs, traj_dir / "trajectories_train.jsonl")
write_trajs(eval_trajs,  traj_dir / "trajectories_eval.jsonl")
print(f"Wrote trajectories → {traj_dir}")

# ── 5. Write label parquets ───────────────────────────────────────────────────
def write_parquet(pids_set, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for pid in pids_set:
        label = oracle_labels.get(pid, False)
        rows.append({
            "problem_id": pid,
            "route_successes": [False, False, False, label],
            "route_rewards": [0.0, 0.0, 0.0, float(label)],
        })
    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "labels.parquet", index=False)
    n_pos = sum(oracle_labels.get(p, False) for p in pids_set)
    print(f"  {out_dir.name}: n={len(rows)}, positive={n_pos} ({100*n_pos/max(1,len(rows)):.1f}%)")

write_parquet(train_pids, PREP_DIR / "parquet" / "train")
write_parquet(eval_pids,  PREP_DIR / "parquet" / "eval")
print(f"Wrote label parquets → {PREP_DIR}/parquet/")

print(f"\n[done] Prepared data in: {PREP_DIR}")
PYEOF

echo ""
echo "=== Launching in-domain Verified predictor training ==="

INCLUDE_THINKING=false \
INCLUDE_TEST_FEEDBACK="${INCLUDE_TEST_FEEDBACK}" \
TRAJECTORIES_DIR="${PREP_DIR}/trajectories" \
TRAIN_PARQUET_DIR="${PREP_DIR}/parquet/train" \
EVAL_PARQUET_DIR="${PREP_DIR}/parquet/eval" \
LABEL_ROUTE_IDX=3 \
NUM_EPOCHS="${NUM_EPOCHS}" \
LORA_R="${LORA_R}" \
  bash "${SCRIPT_DIR}/launch_cot_abstention_predictor.sh"
