"""Per-query token CAPS for cost-aware routing (exploratory, offline). Instead of only choosing a model, the router chooses
(model m, cap c): max_tokens = c truncates the call. Simulated EXACTLY from the stored uncapped draws: under cap c a draw
succeeds iff it succeeded AND its output length <= c, and costs input + min(length, c) output tokens.
Predicted length distribution per (problem, model): log L ~ N(mu_m(x), s_m^2), mu from the plain-ridge probe (log output
tokens), s_m from TRAIN per-draw residuals (includes draw-to-draw noise). Decision:
    argmax_{m, c}  p_m(x) * Phi((log c - mu)/s) * V  -  [input + E_lognormal[min(L, c)]] * price_out   (+ input * price_in)
Arms (same success head p, same mu; test split; frontier over V; cost at matched accuracy vs the paper rule):
  nocap        c = inf for every call                                  (our current cost-head router)
  globalcap    c = a per-MODEL constant (the q-quantile of that model's TRAIN lengths), q chosen on calibration
  querycap     c chosen per query from {q-quantiles of the PREDICTED distribution, q in QS} U {inf}
  paper        the median-length rule (reference, as in decompose.py)
Usage: python percall_caps.py <pool> [<cost_file>]
"""
import json, sys, numpy as np
from pathlib import Path
from scipy.stats import norm
sys.path.insert(0, str(Path(__file__).parent))
from decompose import MK, R, hull, cost_at

name = sys.argv[1]; CF = sys.argv[2] if len(sys.argv) > 2 else "cost_preds_probe.jsonl"
D = R / name; t = np.load(D / "tensors.npz", allow_pickle=True)
S = [str(s) for s in t["model_slots"]]; M = len(S); pids = [str(p) for p in t["problem_ids"]]; pi = {p: i for i, p in enumerate(pids)}
v = t["valid"].astype(bool); ok = (t["final_outcome"] & t["valid"]).astype(bool)
ct = t["completion_tokens"].astype(float); pt = t["prompt_tokens"].astype(float); n = v.sum(2); avail = n > 0
inp = np.nanmean(np.where(v, pt, np.nan), 2)
pin = np.array([MK[s][0] for s in S]) / 1e6 * 100; pout = np.array([MK[s][1] for s in S]) / 1e6 * 100     # cents / token
sp = json.load(open(D / "split_manifest.json")); idx = {k: np.array([pi[str(p)] for p in sp[f"{k}_problem_ids"]]) for k in ("train", "calibration", "test")}
tr, cal, te = idx["train"], idx["calibration"], idx["test"]
_lp = {json.loads(l)["problem_id"]: json.loads(l)["p_successes"][:M] for l in open(D / "content_preds.jsonl")}
P = np.array([_lp[p] for p in pids])                     # keyed by problem_id (file order differs from tensor order)
lc = {json.loads(l)["problem_id"]: json.loads(l)["expected_costs"][:M] for l in open(D / CF)}
LC = np.array([lc[p] for p in pids]) * 100
MU = np.log(np.maximum((LC - np.nan_to_num(inp) * pin) / pout, 1.0))                     # predicted log output tokens
LOGL = np.log(np.maximum(np.where(v, ct, np.nan), 1.0))
SIG = np.array([np.nanstd((LOGL[tr, m] - MU[tr, m][:, None])[v[tr, m]]) for m in range(M)])
QS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98]
BIG = 1e9


def realised(cap):                        # cap [n, M] tokens -> realised success & cost (cents), mean over the draws
    capd = cap[:, :, None]
    succ = np.where(v, ok & (ct <= capd), np.nan); cost = np.where(v, pt * pin[None, :, None] + np.minimum(ct, capd) * pout[None, :, None], np.nan)
    return np.nanmean(succ, 2), np.nanmean(cost, 2)


KF = np.maximum((LC - np.nan_to_num(inp) * pin) / pout, 1.0) / np.exp(MU + SIG[None] ** 2 / 2)   # match the head's calibrated mean


def expected_cost(mu, s, cap):            # E[min(L, c)] for log L ~ N(mu, s^2), rescaled so the uncapped mean = the head's
    lc_ = np.log(np.maximum(cap, 1.0)); z = (lc_ - mu) / s
    part = np.exp(mu + s ** 2 / 2) * norm.cdf(z - s)
    return KF * np.where(cap >= BIG, np.exp(mu + s ** 2 / 2), part + cap * norm.sf(z))


# option set per (problem, model): caps
options = [np.full((len(pids), M), BIG)] + [np.exp(MU + SIG[None] * norm.ppf(q)) for q in QS]
REAL = [realised(c) for c in options]
PASS = [np.where(c >= BIG, 1.0, norm.cdf((np.log(c) - MU) / SIG[None])) for c in options]
ECOST = [np.nan_to_num(inp) * pin[None] + expected_cost(MU, SIG[None], c) * pout[None] for c in options]
Q0 = np.where(avail, np.nanmean(np.where(v, ok, np.nan), 2), 0); C0 = realised(options[0])[1]


def frontier(ii, opt_ids):
    pts = []
    for V in np.geomspace(1e-5, 100, 250):
        best_u = np.full((len(ii), M), -np.inf); best_o = np.zeros((len(ii), M), int)
        for o in opt_ids:
            u = P[ii] * PASS[o][ii] * V - ECOST[o][ii]
            better = u > best_u; best_u = np.where(better, u, best_u); best_o = np.where(better, o, best_o)
        best_u = np.where(avail[ii], best_u, -np.inf); m = best_u.argmax(1); o = best_o[np.arange(len(ii)), m]
        succ = np.array([REAL[oo][0][i, mm] for i, mm, oo in zip(ii, m, o)]); cost = np.array([REAL[oo][1][i, mm] for i, mm, oo in zip(ii, m, o)])
        pts.append((np.nanmean(cost), np.nanmean(succ)))
    return hull(pts)


def paper_frontier(ii):
    med = np.array([np.median(ct[tr, m][v[tr, m]]) for m in range(M)])
    PC = np.nan_to_num(inp) * pin[None] + med[None] * pout[None]
    pts = []
    for V in np.geomspace(1e-5, 100, 250):
        m = np.where(avail[ii], P[ii] * V - PC[ii], -np.inf).argmax(1)
        pts.append((np.nanmean(C0[ii, m]), np.nanmean(Q0[ii, m])))
    return hull(pts)


def oracle_frontier(ii):                  # true per-problem cost (uncapped) -- only to fix the accuracy band, as decompose.py does
    pts = []
    for V in np.geomspace(1e-5, 100, 250):
        m = np.where(avail[ii], P[ii] * V - C0[ii], -np.inf).argmax(1)
        pts.append((np.nanmean(C0[ii, m]), np.nanmean(Q0[ii, m])))
    return hull(pts)


def gain(H, H0, Ho=None):
    hs = [H, H0] + ([Ho] if Ho is not None else [])
    lo, hi = max(h[0][1] for h in hs), min(h[-1][1] for h in hs); T = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 12)
    r = np.array([cost_at(H, x) / cost_at(H0, x) for x in T]); return 1 - float(np.exp(np.nanmean(np.log(r)))), (lo, hi)


# global per-model cap: the q-quantile of that model's TRAIN lengths, q chosen on calibration
def global_cap_opts(q):
    capm = np.array([np.quantile(ct[tr, m][v[tr, m]], q) for m in range(M)])
    c = np.repeat(capm[None], len(pids), 0); REAL.append(realised(c)); PASS.append(norm.cdf((np.log(c) - MU) / SIG[None]))
    ECOST.append(np.nan_to_num(inp) * pin[None] + expected_cost(MU, SIG[None], c) * pout[None]); return len(REAL) - 1


H0c = paper_frontier(cal)
gq = max((0.8, 0.9, 0.95, 0.98, 0.99), key=lambda q: gain(frontier(cal, [global_cap_opts(q)]), H0c)[0])
g_opt = global_cap_opts(gq)
H0 = paper_frontier(te); HO = oracle_frontier(te)
arms = {"nocap": [0], f"globalcap(q={gq})": [g_opt], "querycap": list(range(len(QS) + 1))}
res = {}
for k, o in arms.items():
    g, band = gain(frontier(te, o), H0, HO)
    rng = np.random.default_rng(0); B = []
    for _ in range(200):
        ii = rng.choice(te, len(te)); B.append(gain(frontier(ii, o), paper_frontier(ii), oracle_frontier(ii))[0])
    res[k] = (g, np.percentile(B, [2.5, 97.5]), band)
print(f"{name}: predicted log-length spread per route {SIG.round(2)}; runaway share (draws > 4x the problem's median) "
      f"{np.nanmean(ct[v] > 4 * np.repeat(np.nanmedian(np.where(v, ct, np.nan), 2)[:, :, None], ct.shape[2], 2)[v]) * 100:.1f}%")
for k, (g, ci, band) in res.items():
    print(f"   {k:<20} gain vs paper rule at matched accuracy {g*100:6.1f}% [{ci[0]*100:5.1f}, {ci[1]*100:5.1f}]  (band {band[0]*100:.0f}-{band[1]*100:.0f}%)")
