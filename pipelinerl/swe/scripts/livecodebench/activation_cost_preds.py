#!/usr/bin/env python3
"""Per-problem expected costs for the replay's `--cost-preds`, from frozen activations.

The utility rule is `argmax_m (p_m(x)*R - c_m)`. Every system we surveyed conditions only the
numerator: RoR uses count-based costs, and the prefill-activation routers use "each model's
median training output tokens as a verbosity proxy". That constant is a poor summary of what it
replaces -- on LiveCodeBench, per-problem cost spans p90/p10 of 16-37x, 62-90% of the variance
is between problems rather than draw-to-draw, and failed draws cost 2.6-5.3x more than solved
ones, so cost and correctness are coupled while the rule treats them as independent.

Predicting length from activations is established (arXiv 2607.05316 shows total response length
is linearly decodable from the prompt's last hidden state; 2602.11812 does it for serving
throughput). Neither puts the prediction inside a decision rule, and neither predicts across
models. This emits c_m(x) for every route from ONE cheap model's prefill, in USD, in the same
units as the training-set constant it replaces.

Ridge on log total tokens, layer chosen on calibration, then exponentiated and priced. Log
space because cost is right-skewed and the rule needs a conditional mean, not a median; the
smearing correction below restores the mean under log-normal residuals.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

USD_PER_M_TOKENS = {"scout": 0.278, "oss20": 1.299, "oss120": 11.13}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--activations", required=True)
ap.add_argument("--tensors-dir", required=True)
ap.add_argument("--readout", default="mean")
ap.add_argument("--rich", action="store_true", help="Concatenate ALL layers x {mean,last} instead of one layer of one readout. The single-layer probe captured only 11-40%% of the between-problem variance ceiling on TACO; concatenating roughly doubles cost R2 there (oss20 0.093->0.216) and lifts LCB too (oss120 belief AUC 0.792->0.864, cost R2 0.700->0.792). The gain is FEATURES, not capacity: an MLP on the single layer collapses to negative R2.")
ap.add_argument("--layer-frac", type=float, default=0.5)
ap.add_argument("--alpha", type=float, default=100.0)
ap.add_argument("--cap", type=float, default=0.0, help=(
    "Exclude draws at or above this completion length from the cost target. DEFAULT OFF, and it "
    "should stay off: truncation censors the LATENT length but not the PAID cost -- a draw that "
    "hits the cap costs exactly the cap's worth of tokens, which we observe exactly. Excluding "
    "them raises R2 (TACO oss20 0.093 -> 0.189) by switching to an easier target that is not the "
    "quantity the policy spends, and under-prices every route by 30-48%%. Kept only for ablation."))
ap.add_argument("--no-target-space-select", action="store_true", help=(
    "Always regress in log space. Default OFF: the script fits log-space and direct-token "
    "regressions and picks per route on the calibration split, because log-space accuracy does "
    "not imply dollar-space accuracy and the policy spends dollars."))
ap.add_argument("--no-shrink", action="store_true", help=(
    "Disable calibration-fitted shrinkage. By default the prediction is linearly recalibrated "
    "on the CALIBRATION split: regressing true log-cost on the prediction gives a slope near 1 "
    "when the prediction is informative and near 0 when it is noise, in which case the estimate "
    "collapses to the per-route constant. This makes query-conditioned cost provably no worse "
    "than the constant it replaces -- the failure mode measured on TACO, where cost R2 falls to "
    "0.09 and conditioning actively hurt."))
ap.add_argument("--out", required=True)
a = ap.parse_args()

z = np.load(a.activations, allow_pickle=True)
if a.rich:
    parts = [z[k].reshape(z[k].shape[0], -1) for k in ("mean", "last") if k in z.files]
    X_all = np.concatenate(parts, axis=1)[:, None, :]          # single pseudo-layer
else:
    X_all = z[a.readout if a.readout in z.files else "pre"]
apids = [str(p) for p in z["problem_ids"]]; layers = list(z["layers"])
L = 0 if a.rich else min(range(len(layers)), key=lambda i: abs(layers[i] / max(layers) - a.layer_frac))

T = Path(a.tensors_dir)
t = np.load(T / "tensors.npz", allow_pickle=True)
tp = [str(p) for p in t["problem_ids"]]; ti = {p: i for i, p in enumerate(tp)}
slots = [str(s) for s in t["model_slots"]]
valid = t["valid"]
completion = t["completion_tokens"].astype(float)
total = t["prompt_tokens"].astype(float) + completion
probs = {str(json.loads(l)["problem_id"]): json.loads(l)
         for l in open(T / "problems.jsonl") if l.strip()}

pk = [p for p in apids if p in ti and p in probs]
X = X_all[[apids.index(p) for p in pk]][:, L]
# Use the canonical split manifest, NOT source_temporal_split. They coincide on LCB but
# not on TACO, whose dates are 79.8% Unix-epoch sentinels and whose manifest is therefore a
# random split -- fitting on the temporal "train" there would train on manifest test.
_man = json.loads((Path(a.tensors_dir) / "split_manifest.json").read_text())
_train_ids = {str(x) for x in _man["train_problem_ids"]}
tr = np.array([p in _train_ids for p in pk])
print(f"{len(pk)} problems, fitting on {tr.sum()} train, readout={a.readout} layer={layers[L]}")

_cal_ids = {str(x) for x in _man["calibration_problem_ids"]}
cal = np.array([p in _cal_ids for p in pk])

C = np.zeros((len(pk), len(slots)))
for j, s in enumerate(slots):
    def _mean_log(p):
        sel = valid[ti[p], j, :].copy()
        if a.cap:
            sel &= completion[ti[p], j, :] < a.cap
        x = total[ti[p], j, :][sel]
        if len(x) == 0:                      # every draw censored: fall back to all of them
            x = total[ti[p], j, :][valid[ti[p], j, :]]
        return np.log(max(x.mean(), 1.0)) if len(x) else np.log(1.0)

    def _mean_tokens(p):
        return float(np.exp(_mean_log(p)))

    y = np.array([_mean_log(p) for p in pk])
    sc = StandardScaler().fit(X[tr])
    alpha = a.alpha * max(1, X.shape[1] // 2560)
    m = Ridge(alpha=alpha).fit(sc.transform(X[tr]), y[tr])
    pred = m.predict(sc.transform(X))

    # Which space to regress in is a per-route choice, decided on held-out data.
    #
    # Log space is the right prior for a right-skewed target and is what the smearing correction
    # below assumes, but the POLICY spends dollars, and the two disagree: on TACO the log fit
    # reaches R2 0.32-0.45 in log space while delivering -0.21 to +0.33 in dollars, because
    # dollar error is dominated by a heavy tail the log fit is free to under-weight. Regressing
    # directly on tokens optimises the loss the decision actually pays, and wins on 4 of 6
    # route-dataset cells (TACO oss20 -0.207 -> +0.005, LCB oss20 +0.159 -> +0.370) while losing
    # on gpt-oss-120b, whose costs are the least skewed. So fit both and pick per route on
    # CALIBRATION, never on test.
    direct = None
    if not a.no_target_space_select and cal.sum() > 10:
        y_tok = np.array([_mean_tokens(p) for p in pk])
        md = Ridge(alpha=alpha).fit(sc.transform(X[tr]), y_tok[tr])
        cand = np.clip(md.predict(sc.transform(X)), 1.0, None)
        smear_c = float(np.mean(np.exp(y[tr] - pred[tr])))
        log_dollars = np.exp(pred) * smear_c
        def _sse(p_):
            return float(np.sum((y_tok[cal] - p_[cal]) ** 2))
        if _sse(cand) < _sse(log_dollars):
            direct = cand
            print(f"  {s:8s} target space: DIRECT (log-space fit loses on calibration)")
    if not a.no_shrink and cal.sum() > 10:
        # Linear recalibration fitted on held-out calibration data. b ~ 1 when the prediction
        # carries signal, b -> 0 when it does not, which recovers the constant automatically.
        A_cal = np.c_[np.ones(cal.sum()), pred[cal]]
        coef, *_ = np.linalg.lstsq(A_cal, y[cal], rcond=None)
        pred = coef[0] + coef[1] * pred
        print(f"  {s:8s} shrinkage slope b={coef[1]:.3f}"
              + ("  (prediction ~ignored, collapses to constant)" if coef[1] < 0.25 else ""))
    # Duan's smearing estimator: E[tokens] = exp(pred) * mean(exp(residual)) on TRAIN only.
    # Without it, exp() of a log-space fit returns a conditional median and systematically
    # under-prices every route, which would bias the policy toward buying too much.
    smear = float(np.mean(np.exp(y[tr] - pred[tr])))
    tokens = direct if direct is not None else np.exp(pred) * smear
    # Match the first moment on TRAIN. Shrinkage compresses pred toward its mean, and by
    # Jensen exp() of a compressed predictor has a lower mean than the quantity it estimates;
    # the smearing factor corrects the conditional mean but not this. Without the rescale the
    # routes came out 20-25% under-priced, which would make the policy systematically over-buy.
    # A problem can have no valid draw on a route once truncated draws are excluded (the
    # no-truncation ablation deletes the whole oss20 arm on 4 LCB problems, where all six
    # draws ran away). Those problems carry no cost evidence, so drop them from the moment
    # match rather than let one NaN poison the level and the constant.
    per_problem = np.array([
        total[ti[p], j, :][valid[ti[p], j, :]].mean() if valid[ti[p], j, :].any() else np.nan
        for p, k in zip(pk, tr) if k
    ])
    true_mean_tr = float(np.nanmean(per_problem))
    level = true_mean_tr / max(float(tokens[tr].mean()), 1e-9)
    tokens = tokens * level
    # Second recalibration, in the space the POLICY actually consumes.
    #
    # The shrinkage above is fitted on log tokens, so its guarantee -- b -> 0 recovers the
    # constant -- holds in LOG space. The decision rule spends dollars, and log-space signal is
    # not dollar-space signal: on TACO the log slope reads a healthy 0.83-0.96 while dollar-space
    # R2 for oss20 is 0.032, i.e. noise with an 8.5x p10-p90 spread. Feeding that into
    # `p*R - c` randomises routing, which is how a cost head that is marginally BETTER than the
    # constant on MSE still cost 26% on the frontier.
    #
    # So refit the same linear shrinkage on calibration in dollars. Where the query-conditioned
    # estimate carries no dollar-space signal this collapses to the per-route constant, which is
    # what the guarantee was supposed to deliver in the first place.
    if not a.no_shrink and cal.sum() > 10:
        obs_cal = np.array([
            total[ti[p], j, :][valid[ti[p], j, :]].mean() if valid[ti[p], j, :].any() else np.nan
            for p, k in zip(pk, cal) if k
        ])
        keep = ~np.isnan(obs_cal)
        if keep.sum() > 10:
            A_d = np.c_[np.ones(int(keep.sum())), tokens[cal][keep]]
            cd, *_ = np.linalg.lstsq(A_d, obs_cal[keep], rcond=None)
            if np.isfinite(cd).all() and cd[1] > 0:
                tokens = np.clip(cd[0] + cd[1] * tokens, 1.0, None)
                print(f"  {s:8s} dollar-space shrinkage slope b={cd[1]:.3f}"
                      + ("  (no dollar-space signal; collapses to the constant)"
                         if cd[1] < 0.25 else ""))
    C[:, j] = tokens * USD_PER_M_TOKENS[s] / 1e6
    const = true_mean_tr * USD_PER_M_TOKENS[s] / 1e6
    print(f"  {s:8s} constant ${const:.6f}  predicted mean ${C[:, j].mean():.6f}  "
          f"p10 ${np.percentile(C[:, j],10):.6f}  p90 ${np.percentile(C[:, j],90):.6f}  "
          f"smearing {smear:.3f}  level {level:.3f}")

with open(a.out, "w") as f:
    for i, p in enumerate(pk):
        f.write(json.dumps({"problem_id": p,
                            "expected_costs": [float(x) for x in C[i]]}) + "\n")
print(f"wrote {a.out}")
