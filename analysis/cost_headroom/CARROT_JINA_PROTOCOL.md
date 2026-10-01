# CARROT-style trained Jina-137M encoder

User requested using the existing 137M Jina encoder instead of RoBERTa.
This is an **adapted CARROT-style baseline**, not CARROT-RoBERTa reproduction.
The reference recipe is `somerstep/CARROT` at revision
`3e6acff6aecf4cbcb8f31a118d04c799c2ea1655`.

## Predictors

- Two separately fine-tuned `jinaai/jina-embeddings-v2-base-code` encoders, one
  for route success and one for raw output length, with attention-mask mean
  pooling and a dense/tanh/dropout multi-output head.
- Success: per-problem BCE against each route's mean valid-draw correctness.
- Cost: squared error on **training-standardized raw output-token means**.
  Transform predictions back into tokens; clip negative predictions to zero.
  No log targets, smearing, or training-mean matching. This follows CARROT's
  regression target family, but adapts normalization to the smaller training
  sets. It differs from our earlier log-length fine-tuned reader.
- Existing train/calibration/test splits. Original calibration selects each
  task's checkpoint; test labels never select epochs or hyperparameters.
- Six epochs, batch 8, max length 1,024, AdamW learning rate 2e-5 for all
  parameters, weight decay .01, warmup 10%, gradient norm cap 1, seed 42.
  BF16 on supporting GPUs, FP32 otherwise.
- Defaults differ from upstream RoBERTa/custom-loop training: encoder/pooling,
  six vs. three epochs, 1,024 vs. 256 tokens, existing calibration rather than
  an internal 10% training holdout, standardized token targets, and draw-mean
  rather than binary labels. Explicitly report these adaptations.
- Save best success/cost weights, model revisions, normalization statistics,
  calibration histories, and ID/route-aligned predictions for future held-out
  expansion evaluation. One seed is not a training-variance analysis.

## Evaluation

Reuse all six contrasts and 1,000 paired problem bootstrap resamples from
[CARROT_PROTOCOL.md](CARROT_PROTOCOL.md). Direct pair-specific matched-accuracy
cost ratios, fixed recorded prices, and conditional predictor intervals.
The original datasets are evaluated first; expanded problems remain held out.
No generation or embedding API calls. GPU cluster usage is separate.

```bash
SUBMIT=1 bash launchers/abstention/launch_carrot_jina.sh
```

Three single-GPU snapshot jobs (32 GB GPU memory, 8 CPUs, 32 GB RAM). Each job
trains the two tasks sequentially and evaluates immediately afterward.
Output: `/mnt/llmd/results/exps/aristides/reason/carrot_jina_20261001/`.
Startup logs persist under `logs/`. Results are pending until `results.json`
exists for each pool and scheduler completion is confirmed.
