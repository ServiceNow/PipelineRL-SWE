"""A conditioned cost must be fitted in the space the policy spends.

The cost head regressed log tokens, exponentiated with a smearing correction, and shrank toward
the per-route constant using a slope fitted ON THE LOG SCALE. The conversion back to dollars was
correct; the OBJECTIVE was not. Log-space accuracy does not imply dollar-space accuracy, because
dollar error is dominated by a heavy tail the log fit is free to under-weight -- on TACO the log
slope read a healthy 0.83-0.96 while dollar-space R2 for gpt-oss-20b was 0.032, i.e. noise with
an 8.5x p10-p90 spread fed straight into `p*R - c`.

That is why the "provably no worse than the constant" guarantee did not hold where it mattered.
"""
from __future__ import annotations

import numpy as np


def shrink(pred, truth, cal):
    """Linear recalibration, as applied in activation_cost_preds.py."""
    A = np.c_[np.ones(cal.sum()), pred[cal]]
    coef, *_ = np.linalg.lstsq(A, truth[cal], rcond=None)
    return coef[1], coef[0] + coef[1] * pred


def test_log_space_signal_can_be_dollar_space_noise():
    """The failure mode, reproduced: strong in logs, useless in the units that are spent."""
    rng = np.random.default_rng(0)
    n = 400
    true_log = rng.normal(7.0, 1.6, n)              # heavy-tailed cost
    pred_log = true_log + rng.normal(0, 0.9, n)     # a genuinely informative log predictor
    cal = np.zeros(n, bool); cal[:200] = True

    b_log, _ = shrink(pred_log, true_log, cal)
    assert b_log > 0.4, "the log-space fit looks healthy, which is the trap"

    truth_d, pred_d = np.exp(true_log), np.exp(pred_log)
    te = ~cal
    r2 = 1 - ((truth_d[te] - pred_d[te]) ** 2).sum() / (
        (truth_d[te] - truth_d[te].mean()) ** 2).sum()
    assert r2 < 0.2, "yet in dollars it explains almost nothing"


def test_dollar_space_shrinkage_collapses_when_dollars_carry_no_signal():
    """The fix: refit the shrinkage in the spent units, so it can fall back to the constant."""
    rng = np.random.default_rng(1)
    n = 400
    truth = np.exp(rng.normal(7.0, 1.6, n))
    pred = np.exp(rng.normal(7.0, 1.6, n))          # independent of truth
    cal = np.zeros(n, bool); cal[:200] = True
    b, out = shrink(pred, truth, cal)
    assert abs(b) < 0.25, "no dollar signal must drive the slope to zero"
    assert out.std() < 0.35 * truth.std(), "and the estimate must collapse toward a constant"


def test_dollar_space_shrinkage_keeps_a_genuinely_useful_predictor():
    rng = np.random.default_rng(2)
    n = 400
    truth = np.exp(rng.normal(7.0, 1.0, n))
    pred = truth * np.exp(rng.normal(0, 0.25, n))
    cal = np.zeros(n, bool); cal[:200] = True
    b, _ = shrink(pred, truth, cal)
    assert b > 0.6
