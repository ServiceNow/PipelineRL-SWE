#!/usr/bin/env bash
# Rebuild the whole analysis chain from the locally collected pool.
#
# Run this once launch_lcb_local_collect.sh has finished all six draws on all three routes.
# It rebuilds tensors from the clean labels and re-derives every CPU-side result; the two
# GPU steps are printed at the end rather than submitted, because they need a job.
#
# The activations do NOT need re-extracting. A probe reads the problem prompt, and the prompt
# set is identical -- the same 892 LiveCodeBench problems that pool_probe_prompts.jsonl froze.
# Only the LABELS changed. So pool_activations_*/{scout,oss20,oss120}.npz carry over, and the
# expensive part of the pipeline is already done.
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
cd "${REPO_ROOT}"
R=/mnt/llmd/results/exps/aristides/reason
PY=${PY:-/home/toolkit/.conda/envs/pipeline-rl/bin/python3}

POOL=${POOL:?Need POOL, e.g. $R/lcb_local_pool_1788407418}
SOURCE_DIR=${SOURCE_DIR:-$R/lcb_corrected_temporal_qwen_qwen3_4b_instruct_2507_1787205448}
ACTS=${ACTS:-$R/pool_activations_1788397287}
DRAWS=${DRAWS:-6}
OUT=${OUT:-${POOL}_prepared}

echo "=== 0. completeness check: every route x split x draw must be present and full ==="
missing=0
for route in scout oss20 oss120; do
  for split in train eval; do
    want=$([ "$split" = train ] && echo 551 || echo 341)
    for d in $(seq 0 $((DRAWS-1))); do
      f="${POOL}/${route}_${split}_d${d}.jsonl"
      n=$([ -f "$f" ] && wc -l < "$f" || echo 0)
      if [ "$n" -lt "$want" ]; then
        echo "  INCOMPLETE ${route}_${split}_d${d}: ${n}/${want}"; missing=1
      fi
    done
  done
done
[ "$missing" = 0 ] || { echo "Collection is not finished. Stopping."; exit 1; }
echo "  all $((DRAWS*6)) files complete"

echo "=== 1. tensors from clean labels ==="
${PY} pipelinerl/swe/scripts/livecodebench/build_mdp_tensors_v2.py \
  --collection-dir "${POOL}" --source-collection-dir "${SOURCE_DIR}" \
  --output-dir "${OUT}/tensors_v3" --num-draws "${DRAWS}" --split-mode source_temporal

echo "=== 2. serving-path comparison: EmptyGeneration, local vs OpenRouter ==="
${PY} - "${POOL}" <<'PYEOF'
import json, glob, os, sys, collections
pool=sys.argv[1]
agg=collections.defaultdict(lambda:[0,0,0,0])
for f in glob.glob(pool+"/*_d*.jsonl"):
    route=os.path.basename(f).split("_")[0]
    for l in open(f):
        if not l.strip(): continue
        r=json.loads(l); a=agg[route]; a[0]+=1
        if (r.get("eval_metadata") or {}).get("error_message")=="EmptyGeneration": a[1]+=1
        a[2]+= bool(r.get("resolved")); a[3]+= r.get("finish_reason")=="length"
print(f"{'route':<8} {'n':>6} {'EmptyGen':>9} {'solved':>8} {'at 32k cap':>11}")
for k,(n,e,s,fl) in sorted(agg.items()):
    print(f"{k:<8} {n:6d} {100*e/n:8.1f}% {100*s/n:7.1f}% {100*fl/n:10.1f}%")
print("\nOpenRouter at the same cap, for contrast: oss20 22.6% EmptyGen, oss120 8.2% (train).")
PYEOF

echo "=== 3. transfer matrix on clean labels (activations reused, no re-extraction) ==="
${PY} pipelinerl/swe/scripts/livecodebench/pool_activation_probe.py --phase matrix \
  --activations-file scout="${ACTS}/scout.npz" \
  --activations-file oss20="${ACTS}/oss20.npz" \
  --activations-file oss120="${ACTS}/oss120.npz" \
  --tensors-dir "${OUT}/tensors_v3" \
  --cost-json "$R/pool_probe_prefill_costs.json" \
  | tee "${OUT}/transfer_matrix.txt"

echo "=== 4. content predictions for the replay ==="
${PY} pipelinerl/swe/scripts/livecodebench/activation_content_preds.py \
  --activations "${ACTS}/scout.npz" --tensors-dir "${OUT}/tensors_v3" \
  --output "${OUT}/content_preds.jsonl" || \
  echo "  (check activation_content_preds.py args if this failed)"

cat <<EOM

=== remaining, GPU, submit as jobs ===
1. replay with content_decay + LoRA arms:
     --tensors-dir ${OUT}/tensors_v3 --content-preds ${OUT}/content_preds.jsonl \\
     --sequential-model-dir <lora>/model --state-layout counts_last --num-orderings 5
2. then, on the resulting episode_traces.jsonl:
     compare_uniform_truncation.py --tensors-dir ${OUT}/tensors_v3 \\
       --policies content_decay_value,content_decay_abstain,content_decay

Note: the LoRA was trained on the OLD reachable dataset. For a like-for-like comparison it
must be retrained on the clean one, or reported as trained on stale labels. Do not quietly
compare a probe fit on clean data against an encoder fit on corrupted data.
EOM
