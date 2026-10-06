#!/usr/bin/env bash
# NEW_PATH 4.A.59: rerun one pool with deepseek-v4-flash PINNED. Builds the pool in two shadow roots -- reason_anchor (unchanged
# tensors: regenerated predictions must reproduce the archived ones, which validates the feature file / recipe) and reason_pinned
# (dsv4f slot swapped) -- refits success + cost readouts with the paper recipes in each, and evaluates the pinned one:
# decompose.py (headroom + gain vs paper rule, list prices) and reprice_original.py (billed). Outputs tagged _pinned / _anchor.
# Usage: bash pinned_rerun.sh <pool>   pool in: lcb bcb cc apps omni mmlupro aime fresh
set -uo pipefail
cd /home/toolkit/PipelineRL-SWE
PY=/home/toolkit/.conda/envs/pipeline-rl/bin/python3; REAL=/mnt/llmd/results/exps/aristides/reason; CH=analysis/cost_headroom
export HF_HOME=/home/toolkit/.cache/huggingface HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
P=$1
case $P in
  lcb)     T=pool_v2_tensors_5rung; FEAT=$REAL/pv2_scout_prefill_1756715297/scout.npz; COST=cost_preds_probe.jsonl; HEAD=bch; LABEL=LCB ;;
  bcb)     T=bcb_tensors_5r;        FEAT=$REAL/bcb_scout_prefill.npz;              COST=cost_preds_probe.jsonl; HEAD=bch; LABEL=BCB ;;
  cc)      T=cc_tensors;            FEAT=$REAL/cc_pool/scout_prefill.npz;          COST=cost_preds_probe.jsonl; HEAD=bch; LABEL=CC ;;
  apps)    T=apps_tensors;          FEAT=$REAL/apps_probe/instruct.npz;            COST=cost_preds_probe.jsonl; HEAD=bch; LABEL=APPS ;;
  omni)    T=omni500_tensors;       FEAT=$REAL/omni500_probe/thinking.npz;         COST=cost_preds_probe_thinking.jsonl; HEAD=pmc:thinking; LABEL=Omni ;;
  mmlupro) T=mmlupro_tensors;       FEAT=$REAL/mmlupro_probe/instruct.npz;         COST=cost_preds_probe_instruct.jsonl; HEAD=pmc:instruct; LABEL=MMLU-Pro ;;
  aime)    T=aime_tensors;          FEAT=$REAL/aime_probe/instruct.npz;            COST=cost_preds_probe_instruct.jsonl; HEAD=pmc:instruct; LABEL=AIME ;;
  fresh)   ;;
  *) echo "unknown pool $P"; exit 1 ;;
esac
L=$REAL/reason_pinned_logs; mkdir -p $L; exec > >(tee -a $L/$P.log) 2>&1
readouts() {   # $1 = root
  local ROOT=$1 D=$1/$T
  $PY pipelinerl/swe/scripts/livecodebench/activation_content_preds.py --activations $FEAT --tensors-dir $D --rich --select-C \
    --out $D/content_preds.jsonl > $D/content.log 2>&1
  if [[ $HEAD == bch ]]; then REASON_ROOT=$ROOT $PY $CH/baseline_cost_heads.py $T $FEAT --only probe > $D/cost.log 2>&1
  else REASON_ROOT=$ROOT $PY $CH/probe_model_compare.py $T ${HEAD#pmc:}=$FEAT > $D/cost.log 2>&1; fi
}
compare() {    # archived (real) vs regenerated (anchor) predictions
  $PY - "$REAL/$T" "$REAL/../reason_anchor/$T" "$COST" <<'EOF'
import json, sys, numpy as np
real, anc, cost = sys.argv[1:]
for f, key in (("content_preds.jsonl", "p_successes"), (cost, "expected_costs")):
    a = {json.loads(l)["problem_id"]: json.loads(l)[key] for l in open(f"{real}/{f}")}
    b = {json.loads(l)["problem_id"]: json.loads(l)[key] for l in open(f"{anc}/{f}")}
    common = sorted(set(a) & set(b)); x = np.array([a[p][:len(b[p])] for p in common]); y = np.array([b[p] for p in common])
    rel = np.abs(x - y) / np.maximum(np.abs(x), 1e-12)
    print(f"ANCHOR {f}: {len(common)}/{len(a)} problems, max abs diff {np.abs(x - y).max():.2e}, max rel diff {rel.max():.2e}")
EOF
}
if [[ $P != fresh ]]; then
  $PY $CH/build_pinned_root.py $T --anchor && $PY $CH/build_pinned_root.py $T
  readouts $REAL/../reason_anchor & a=$!; readouts $REAL/../reason_pinned & b=$!; wait $a $b   # explicit PIDs: a bare wait also waits for the tee process substitution (deadlock)
  compare
  ROOT=$REAL/../reason_pinned
  REASON_ROOT=$ROOT RESULT_TAG=_pinned $PY $CH/decompose.py --curve --out ${P} $T:$COST:market > $L/${P}_decompose.txt 2>&1
  REASON_ROOT=$ROOT RESULT_TAG=_pinned $PY $CH/reprice_original.py $LABEL:$T:$COST > $L/${P}_reprice.txt 2>&1
  grep -E "^$T|headroom|saves" $L/${P}_decompose.txt $L/${P}_reprice.txt | head -8
else
  # fresh sets: needs the omni + mmlupro pool reruns first (their content/cost preds are the original-row predictions)
  until [[ -s $REAL/../reason_pinned/omni500_tensors/cost_preds_probe_thinking.jsonl && -s $REAL/../reason_pinned/mmlupro_tensors/cost_preds_probe_instruct.jsonl ]]; do sleep 60; done
  $PY $CH/build_pinned_root.py expanded_mmlupro expanded_omni500
  ROOT=$REAL/../reason_pinned
  for ds in mmlupro omni500; do
    E=$ROOT/expanded_eval_20261001/$ds
    $PY pipelinerl/swe/scripts/livecodebench/activation_content_preds.py --activations $E/prefill_combined.npz --tensors-dir $E --rich \
      --select-C --out $E/success_preds.jsonl > $E/content.log 2>&1 & pids="${pids:-} $!"
  done; wait $pids
  for ds in mmlupro omni500; do REASON_ROOT=$ROOT $PY $CH/reconstruct_paper_cost_heads.py --dataset $ds > $L/fresh_cost_$ds.txt 2>&1; tail -1 $L/fresh_cost_$ds.txt | cut -c1-300; done
  REASON_ROOT=$ROOT RESULT_TAG=_pinned $PY -u $CH/fresh_baselines.py > $L/fresh_baselines.txt 2>&1
  REASON_ROOT=$ROOT RESULT_TAG=_pinned $PY -u $CH/deploy_matched.py > $L/deploy_matched.txt 2>&1
  tail -30 $L/fresh_baselines.txt; cat $L/deploy_matched.txt | grep -v -i warn
fi
echo "PINNED RERUN $P DONE"
