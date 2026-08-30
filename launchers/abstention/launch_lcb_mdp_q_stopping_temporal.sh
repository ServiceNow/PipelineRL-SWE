#!/usr/bin/env bash
# How much of the oracle's stopping headroom can the beliefs we already have recover?
#
# The oracle arm showed stopping is worth ~67% of cost while routing is worth ~3%, but
# an oracle is an upper bound and we have no idea how tight it is. This arm replaces
# stopping with a swept threshold on the learned q(s) = 1 - P(nothing remains) -- the
# `nothing` head, which every value and Bellman arm currently leaves unread -- while
# leaving routing to the value rule, exactly as the oracle arm does.
#
# Unlike the oracle arm this is deployable: q(s) is a model output, not a leaked label.
# It puts a realistic floor under the stopping program before any retraining is spent.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

SUBMIT=${SUBMIT:-0}
TIMESTAMP=${TIMESTAMP:-$(date +%s)}
ARTIFACT_DIR=${ARTIFACT_DIR:-/mnt/llmd/results/exps/aristides/reason/lcb_mdp_temporal_551_341_counts_last_unweighted_1787939458b}
TENSORS_DIR=${TENSORS_DIR:-${ARTIFACT_DIR}/tensors_v3}
REPLAY_TAG=${REPLAY_TAG:-replay_q_stop_v1}
JOB_NAME=${JOB_NAME:-lcb_mdp_temporal_counts_last_q_stop_${TIMESTAMP}}
Q_STOP_GRID=${Q_STOP_GRID:-0.02,0.05,0.08,0.12,0.16,0.20,0.25,0.30,0.40,0.50}
BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES:-2000}

if [[ "${SUBMIT}" != "1" ]]; then
  cat <<EOF
Prepared but not submitted.
  job: ${JOB_NAME}
  artifact: ${ARTIFACT_DIR}
  output: ${ARTIFACT_DIR}/${REPLAY_TAG}
  policy: sequential_decay Bellman H=2, routing unchanged
  intervention: stopping replaced by a threshold on the learned q(s)
  q grid: ${Q_STOP_GRID}
  also runs: oracle stop / route / both, for the comparison this exists to make

Submit explicitly with:
  SUBMIT=1 bash ${BASH_SOURCE[0]}
EOF
  exit 0
fi

# Oracle arms run alongside so the achievable and upper-bound numbers land in one
# artifact on identical episodes, rather than being compared across jobs.
ARTIFACT_DIR="${ARTIFACT_DIR}" \
TENSORS_DIR="${TENSORS_DIR}" \
STATE_LAYOUT=counts_last \
START_PROTOCOL=scout_first \
NUM_ORDERINGS=5 \
REPLAY_TAG="${REPLAY_TAG}" \
JOB_NAME="${JOB_NAME}" \
SNAPSHOT=1 \
EXTRA_REPLAY_ARGS="--bellman-horizons 2 \
  --q-stop-family sequential_decay --q-stop-horizon 2 --q-stop-grid ${Q_STOP_GRID} \
  --oracle-stopping-family sequential_decay --oracle-stopping-horizon 2 \
  --bootstrap-samples ${BOOTSTRAP_SAMPLES}" \
bash "${SCRIPT_DIR}/launch_lcb_mdp_replay.sh"
