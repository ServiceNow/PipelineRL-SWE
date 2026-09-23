#!/usr/bin/env bash
# RECOLLECT THE LCB POOL at each model's recommended sampling settings -- OPEN RUNGS ONLY.
#
# Why these rungs (pool pilot, 100 problems, launch_pool_pilot.sh):
#   * The pool is a DIFFICULTY LADDER, not a set of specialists: across ten models from seven labs
#     no model solved a problem the others miss (union 95.7% with or without Claude). So the rungs
#     are chosen for price/accuracy position, not for complementarity.
#   * gpt-oss-120b high is the rescue rung: of the 12 problems both 20b rungs fail at draw 0 it
#     solves 8, against 7 for claude-opus-5 (9x its price) and 5 for claude-sonnet-5. The frontier
#     models are therefore NOT in this pool; add them later only if the open ladder tops out.
#   * Depth saturates early (20b low 84->90% over 6 draws, 20b medium 85->94% over 8, 120b medium
#     86->91% over 5), so depth is bought where a draw is ~0.01-0.05c and rationed at the top.
#
# Price spread 0.011c -> 0.652c per draw is 59x, against 7x for the old three-route pool at market
# prices: enough room for a give-up decision to be worth anything.
#
# One eai job per (route, draw); both splits (892 problems) inside each job. Resumable: a job
# rerun reuses complete rows, so a partial draw can be topped up by resubmitting the same runner.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
R=/mnt/llmd/results/exps/aristides/reason
SUBMIT=${SUBMIT:-0}
SNAPSHOT=${SNAPSHOT:-1}
# MAX_TOKENS is PER RUNG (last field): 128k where the endpoints allow it, the ceiling where they
# do not. gpt-oss's ceiling is 117,964 and NOTHING is above it (0 of 24 endpoints for 120b, 0 of 12
# for 20b at >=119,000), so a flat 128k there would fail every draw. 20b gets the full 117,964
# (7/12 endpoints, same as at 110k). 120b gets 115,264, which keeps 18/24 instead of 16/24: its
# truncation at 110k was already 0.0%, so 2.7k more headroom buys nothing and narrowing the
# endpoint pool is how the provider lottery bit us. DeepSeek: 128,000, 13/15 endpoints. The cap was binding unequally across rungs
# -- 12.2% of gpt-oss-20b-high eval draws were truncated against 4.1% for gpt-oss-120b-high and 0%
# for 20b-medium -- and a truncated draw is scored as a failure, so the cap was biasing the very
# rung comparison the pool is being chosen on. Raising it loses exactly one endpoint per model
# (120b 19->18 of 24, 20b 9->8 of 13, DeepSeek 14->13 of 15), so it does not re-introduce the
# provider lottery. Worst case cost is +6% on the pool, and only the truncating draws pay it.
MAX_TOKENS=${MAX_TOKENS:-110000}
KEYFILE=${KEYFILE:-/home/toolkit/.secrets/openrouter_api_key}
# Per-job request concurrency. The thinking-heavy rungs are GENERATION-latency bound, not grading
# bound -- gpt-oss-120b-high emits ~10k reasoning tokens per draw, so at 8 concurrent each job
# cleared only ~1.6 problems/min and would have needed until 04:14. Throughput scales with this.
CONCURRENCY=${CONCURRENCY:-8}
SRC=${SRC:-$R/lcb_corrected_temporal_qwen_qwen3_4b_instruct_2507_1787205448}
BASE=${BASE:-$R/pool_v2_lcb}
ONLY=${ONLY:-}                 # space-separated route labels
START_DRAW=${START_DRAW:-0}
DRAW_SCALE=${DRAW_SCALE:-1}    # 1 = the counts below; use a smaller pool for a smoke test

# label | model | temp | top_p | effort | draws | measured c/draw (pilot) | extra sampling args
# effort: low/medium/high for gpt-oss; "on" for a HYBRID model, whose reasoning default differs by
# endpoint -- deepseek-v4-flash answers with no reasoning at all on OpenInference/DigitalOcean
# (~300 tokens, 63-74%) and with reasoning on StreamLake/GMICloud (2-6k tokens, 95-97%).
ROUTES=(
  "oss20lo|openai/gpt-oss-20b|1.0|1.0|low|16|0.011||117964"
  "oss20md|openai/gpt-oss-20b|1.0|1.0|medium|12|0.042||117964"
  "dsv4f|deepseek/deepseek-v4-flash|0.7|0.95|on|8|0.158||128000"
  "oss120md|openai/gpt-oss-120b|1.0|1.0|medium|6|0.106||115264"
  "oss120hi|openai/gpt-oss-120b|1.0|1.0|high|3|0.569||115264"
  # Optional, not in the default pool: a second lab at the top tier. Adds no unique coverage in
  # the pilot (87.0% alone, +0.0pt to the union), so it buys a peer choice we showed does not pay.
  # "q235t|qwen/qwen3-235b-a22b-thinking-2507|0.6|0.95||2|0.802|--top-k 20 --min-p 0"
)
# c/draw above are SPLIT-WEIGHTED (551 train / 341 eval), from post-fix draws where they exist.
# They are not the first pilot's numbers: DeepSeek doubled once it was actually made to think
# (0.048 -> 0.098) and gpt-oss-20b medium rose 0.034 -> 0.042. Eval draws cost 1.5-2.5x train.
NPROB=$(( $(wc -l < "${SRC}/scout_train.jsonl") + $(wc -l < "${SRC}/scout_eval.jsonl") ))
mkdir -p "${BASE}"

TOTAL=0
for spec in "${ROUTES[@]}"; do
  IFS='|' read -r LABEL MODEL TEMP TOPP EFFORT NDRAW CPD EXTRA MAXTOK <<< "${spec}"
  MAXTOK=${MAXTOK:-${MAX_TOKENS}}
  if [[ -n "${ONLY}" && " ${ONLY} " != *" ${LABEL} "* ]]; then continue; fi
  NDRAW=$(( NDRAW / DRAW_SCALE )); (( NDRAW > 0 )) || NDRAW=1
  COST=$(/home/toolkit/.conda/envs/pipeline-rl/bin/python3 -c "print(f'{${CPD}*${NDRAW}*${NPROB}/100:.2f}')")
  TOTAL=$(/home/toolkit/.conda/envs/pipeline-rl/bin/python3 -c "print(f'{${TOTAL}+${COST}:.2f}')")
  printf '%-10s %-34s %2d draws x %d problems  ~$%s\n' "${LABEL}" "${MODEL}" "${NDRAW}" "${NPROB}" "${COST}"
  for DRAW in $(seq ${START_DRAW} $((START_DRAW + NDRAW - 1))); do
    case "${EFFORT}" in
      "")  REASON_ARG="" ;;
      on)  REASON_ARG=" --reasoning-enabled --require-parameters" ;;
      *)   REASON_ARG=" --reasoning-effort ${EFFORT}" ;;
    esac
    RUNNER="${BASE}/run_${LABEL}_d${DRAW}.sh"
    {
      echo '#!/usr/bin/env bash'; echo 'set -euo pipefail'
      echo "cd ${REPO_ROOT}"
      echo 'export HF_HUB_DISABLE_IMPLICIT_TOKEN=1'
      echo 'source pipelinerl/swe/scripts/livecodebench/ensure_lcb_runner.sh'
      echo "export OPENROUTER_API_KEY=\$(cat ${KEYFILE})"
      echo "python pipelinerl/swe/scripts/livecodebench/collect_lcb_expert.py \\"
      echo "  --source-collection-dir ${SRC} --output-dir ${BASE} --route-label ${LABEL} \\"
      echo "  --model '${MODEL}' --splits train,eval \\"
      echo "  --temperature ${TEMP} --top-p ${TOPP} ${EXTRA} --max-tokens ${MAXTOK} \\"
      echo "  --gen-timeout 3600 --max-invalid-frac 0.10 --output-suffix _d${DRAW} \\"
      echo "  --api-key-file ${KEYFILE} --concurrency ${CONCURRENCY}${REASON_ARG}"
    } > "${RUNNER}"
    chmod +x "${RUNNER}"
    if [[ "${SUBMIT}" == "1" ]]; then
      if out=$(make -C "${REPO_ROOT}" job JOB_NAME="pv2_${LABEL}_d${DRAW}_$(date +%s)" ENV=pipeline-rl \
          CONDA_EXE=/opt/conda/bin/conda SNAPSHOT="${SNAPSHOT}" NPROC=1 \
          GPU=0 GPU_MEM=0 CPU=8 CPU_MEM=32 COMMAND="bash ${RUNNER}" 2>&1); then
        echo "  ok   pv2_${LABEL}_d${DRAW}"
      else echo "  FAIL pv2_${LABEL}_d${DRAW}: $(echo "$out" | tail -1)"; fi
      sleep 20
    fi
  done
done
echo "-------- estimated total: ~\$${TOTAL} (pilot c/draw x draws x ${NPROB} problems) --------"
echo "Output: ${BASE}"
[[ "${SUBMIT}" == "1" ]] || echo "Prepared, not submitted. SUBMIT=1 to launch."
