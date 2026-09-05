"""A conditioned belief must shrink toward the constant it replaces.

The cost head has carried calibration-fitted shrinkage from the start, precisely so that
query-conditioning is provably no worse than the per-route constant. The belief head shipped
without it, and that asymmetry is where every catastrophic number came from: a raw logistic
head on 40,960 features and ~550 problems emitted next-draw probabilities of 1e-5 for routes
that solve the problem, the utility rule read `p*R - c < 0`, and the policy abstained on
problems the count-based baseline attempts and wins.

These tests pin the property that makes conditioning safe: as the held-out fit finds no signal,
the prediction collapses to a constant, which is exactly what the baseline uses.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def platt(raw: np.ndarray, y: np.ndarray, cal: np.ndarray) -> np.ndarray:
    """The transform applied in activation_content_preds.py."""
    p = np.clip(raw, 1e-6, 1 - 1e-6)
    lo = np.log(p / (1 - p))
    fit = LogisticRegression(max_iter=2000, C=1e6).fit(lo[cal].reshape(-1, 1), y[cal])
    return fit.predict_proba(lo.reshape(-1, 1))[:, 1]


def test_pure_noise_collapses_to_the_base_rate():
    """No signal on held-out data => the constant, i.e. exactly what RoR uses."""
    rng = np.random.default_rng(0)
    n = 600
    raw = rng.uniform(0.001, 0.999, n)
    y = (rng.uniform(size=n) < 0.4).astype(int)     # outcome independent of raw
    cal = np.zeros(n, bool); cal[:300] = True
    out = platt(raw, y, cal)
    assert abs(out.mean() - y[cal].mean()) < 0.05
    assert out.std() < 0.06, "a noise predictor must not keep varying per problem"


def test_an_informative_predictor_is_preserved():
    rng = np.random.default_rng(1)
    n = 800
    z = rng.normal(size=n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-z))).astype(int)
    raw = 1 / (1 + np.exp(-z))
    cal = np.zeros(n, bool); cal[:400] = True
    out = platt(raw, y, cal)
    assert np.corrcoef(out, raw)[0, 1] > 0.95


def test_the_overconfident_left_tail_is_lifted():
    """The measured failure: 1e-5 predictions for routes that actually succeed."""
    rng = np.random.default_rng(2)
    n = 800
    z = rng.normal(size=n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-0.4 * z))).astype(int)   # true slope 0.4
    raw = 1 / (1 + np.exp(-3.0 * z))                                     # head is 7.5x too sharp
    cal = np.zeros(n, bool); cal[:400] = True
    out = platt(raw, y, cal)
    assert raw.min() < 1e-3
    assert out.min() > 10 * raw.min(), "calibration must pull the tail off the floor"
    assert out.max() < raw.max()


def test_calibration_never_uses_the_test_split():
    """Fitting on test would leak; the split must change the answer."""
    rng = np.random.default_rng(3)
    n = 400
    raw = rng.uniform(0.01, 0.99, n)
    y = (raw > 0.5).astype(int)
    a = np.zeros(n, bool); a[:200] = True
    assert not np.allclose(platt(raw, y, a), platt(raw, y, ~a))
