#!/usr/bin/env bash
# Collect LCB trajectories using Instruct-4B scout (Qwen3-4B-Instruct-2507).
#
# Instruct-4B performs better than Thinking-4B on SWE-B agentic tasks;
# on LCB competitive programming this ordering may differ. Running both
# scout variants lets us compare directly on the same domain.
#
# This is a separate collection from the Thinking-4B run
# (lcb_collect_qwen_qwen3_4b_thinking_2507_*) — different scout model,
# different patches, different failure modes.
#
# Optional env vars (override defaults in launch_lcb_collection.sh):
#   MAX_SAMPLES   -- default: 500
#   MIN_DATE      -- default: 2023-09-01
#   CONCURRENCY   -- default: 16
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

SCOUT_MODEL=Qwen/Qwen3-4B-Instruct-2507 \
  bash "${SCRIPT_DIR}/launch_lcb_collection.sh"
