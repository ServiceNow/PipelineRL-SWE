# SWE-Smith full-execution MDP preparation (protocol v2)

This is the agentic-domain counterpart to the corrected LiveCodeBench MDP. It
uses the same matched three-tier portfolio:

- mandatory scout: Qwen3-4B-Instruct-2507
- resample/escalate actions: scout, gpt-oss-20b, gpt-oss-120b
- routing and final verdict: the same real SWE-Smith sandbox execution
- policy state: issue, ordered failed patches/model outputs, and concise real
  FAIL_TO_PASS/PASS_TO_PASS feedback
- termination: immediate after any real sandbox pass
- cost: realized prompt + completion tokens; decisions use train-split expected
  cost, so future completion length is not exposed
- verifier economics: replay always reports the number of sandbox executions and
  accepts `--execution-cost-usd` (launcher: `EXECUTION_COST_USD`) for a fixed
  per-attempt sensitivity sweep; zero means model-token cost only

The Qwen3-Coder-30B-A3B generations remain available for a later portfolio-size
ablation but are intentionally excluded from the primary LCB-matched comparison.

## Data audit

The existing `eval150` collection has 150 problems x 3 draws for all three main
tiers. Real sandbox reports are missing for a few individual attempts. The
builder treats missing reports as invalid, never as failures, and its strict
default retains the 143 problems with all nine reports.

A local preparation check produced:

- canonical internal split: 72 train / 36 calibration / 35 test problems
- reachable states with 20 sampled histories/problem: 5,568 / 2,828 / 1,709
- policy state: problem statement, explicit per-route failed/remaining counts, and only the latest
  failed sandbox attempt (the full history is retained as provenance metadata)
- abstention target: positive only when no successful valid draw remains in any route
- problems represented after mandatory-scout failure: 58 / 25 / 22

This is enough for development and an end-to-end gating run, but the 35-problem
internal test is not a paper-scale independent evaluation.

The separate 300-instance no-overlap generation collection is complete, but no
real sandbox reports are present in this checkout. The preferred final design is:

1. use the 150-instance collection for train/calibration;
2. execute all nine main-route attempts for the 300-instance collection;
3. pass its ID file with `--heldout-ids` so those 300 problems become the exact,
   untouched test set;
4. never tune on the 300-instance outcomes.

The builder accepts repeated `--trace-root`, `--report-root`, and
`--report-split` triples for that combined bundle.

## Prepared launcher

`launchers/abstention/launch_swe_smith_mdp_full_execution.sh` is safe by
default: it only prints its configuration and exits. It submits only when
`SUBMIT=1` is explicitly supplied.

Development configuration using the existing 143 complete problems:

```bash
bash launchers/abstention/launch_swe_smith_mdp_full_execution.sh
```

Do not set `SUBMIT=1` until the corrected LCB result has been audited. Once the
300-instance reports exist, additionally set `EXTRA_TRACE_ROOT`,
`EXTRA_REPORT_ROOT`, and `HELDOUT_IDS` together before submission.

## Interpretation

This is not the older weak-verifier SWE-Smith controller. Proxy similarity
scores are retained in the tensor bundle only for an explicitly labeled
weak-verifier ablation. They never determine primary routing outcomes, labels,
termination, or reported correctness.
