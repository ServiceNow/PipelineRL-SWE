"""Why ZeroRouter's cost estimate falls short, and what the headline would be with its success model. Reads the per-query predictions the rerun
launch_nov05_zr_hybrid.sh saved (zrfaithful/hybrid/preds_*.npz: no refit). Billed costs, saved() at matched accuracy, ten training sets.
Ladder, all with OUR success probes so that only the price differs (saving vs median pricing):
  zr          ZeroRouter's pricing (4B scout): per-candidate mean training length in K quantile bins of one score s = alpha^T b, fitted on OUTCOMES only
  lvlbins     the same one-score table with a score fitted on LENGTH: bins of our probe's mean predicted log length (K chosen on calibration as theirs)
  lvlcont     that score used continuously: our shared part plus each candidate's training mean (level_vs_scx.py's ours-level)
  ownbins     a table per candidate, on that candidate's own predicted length (K chosen on calibration)
  ours        our cost probe
Also: test R2 of log length (per candidate, of the shared part, of the differences between candidates), rank correlation of the predicted shared length
with the actual one and with empirical difficulty, bins the test queries actually occupy; and with ZeroRouter's success (P_4b, P_pub): our probe vs median
pricing and perfect cost knowledge with the SAME success, the numbers tab:pools would show (NBOOT paired resamples for the 4B version).
Usage: REASON_ROOT=.../reason_pinned NBOOT=100 python zr_pricing_why.py POOL -> zr_pricing_why_<pool>_pinned.json
"""
import json, os, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).parent))
POOL_ = sys.argv[1]; sys.argv = [sys.argv[0], "zrwhy", POOL_]
exec(open(Path(__file__).parent / "cost_generalization.py").read().split('MODE, POOL = sys.argv[1], sys.argv[2]')[0])
success_linear = lambda *a, **k: None                                                 # P comes from the saved predictions
rich = lambda feat, ids: np.zeros((len(ids), 1))                                       # features are not needed
D_ = "/mnt/llmd/results/exps/aristides/reason/reason_pinned_logs/zrfaithful/hybrid/"; NB = int(os.environ.get("NBOOT", "100"))
r2 = lambda y, x: float(1 - ((y - x) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def table(score, d, K):
    """per-candidate mean training length in K quantile bins of a score [N] or [N, M] (their Eqs. 8-10)"""
    sc = score[:, None].repeat(d["M"], 1) if score.ndim == 1 else score; tr = d["tr"]; tk = np.zeros_like(d["L"])
    for m in range(d["M"]):
        e_ = np.quantile(sc[tr, m], np.linspace(0, 1, K + 1)[1:-1]); bn = np.searchsorted(e_, sc[:, m])
        tk[:, m] = np.array([d["L"][tr[bn[tr] == j], m].mean() if (bn[tr] == j).any() else d["L"][tr, m].mean() for j in range(K)])[bn]
    return tk


def best_K(score, d, Hca):
    c = [table(score, d, K) for K in (5, 10, 20)]
    return c[int(np.nanargmax([saved(curve(d, d["P"], cost_of(d, x), d["ca"]), Hca) for x in c]))]


def finish(d, yh):
    tr, Y = d["tr"], np.log(d["L"]); tk = np.zeros_like(d["L"])
    for m in range(d["M"]):
        sm = np.mean(np.exp(Y[tr, m] - yh[tr, m])); es, et = np.exp(yh[tr, m]) * sm, np.exp(yh[:, m]) * sm
        tk[:, m] = et * d["L"][tr, m].mean() / es.mean()
    return tk


res = {"pool": POOL_, "seeds": {}}; p_ = POOL_.lower().replace("-", "")
for s in range(10):
    os.environ["DRAW"] = str(s); d = load(POOL_); z = np.load(D_ + f"preds_{p_}_{s}.npz", allow_pickle=True)
    assert list(map(str, z["ids"])) == list(d["ids"]) and (z["ev"] == d["ev"]).all() and (z["tr"] == d["tr"]).all()
    d["P"] = z["P_ours"]; N = len(d["ids"]); tr, ca, ev = d["tr"], d["ca"], d["ev"]; Y = np.log(d["L"]); c = Y[tr].mean(0)
    Cm = cost_of(d, np.repeat(d["med"][None], N, 0)); lo = np.log(np.maximum(z["len_ours"], 1)); lvl = (lo - c[None]).mean(1)
    Hca = curve(d, d["P"], Cm, ca); H0 = curve(d, d["P"], Cm, ev)
    tk = {"zr": z["len_4b"], "zr_pub": z["len_pub"], "lvlbins": best_K(lvl, d, Hca), "lvlcont": finish(d, c[None] + lvl[:, None]), "ownbins": best_K(lo, d, Hca), "ours": z["len_ours"]}
    sv = {a: 100 * saved(curve(d, d["P"], cost_of(d, x), ev), H0) for a, x in tk.items()}
    # how well each predicts length on the test problems
    Ye = Y[ev]; sh = Ye.mean(1); dif = Ye - sh[:, None]; acc = {}
    for a in ("zr", "zr_pub", "ours"):
        X = np.log(np.maximum(tk[a][ev], 1)); xs = X.mean(1); xd = X - xs[:, None]
        acc[a] = dict(r2_per_cand=float(np.mean([r2(Ye[:, m], X[:, m]) for m in range(d["M"])])), r2_shared=r2(sh, xs), r2_diff=r2(dif.ravel(), xd.ravel()),
                      rho_shared=float(spearmanr(xs, sh)[0]), rho_difficulty=float(spearmanr(xs, -d["q"][ev].mean(1))[0]))
    u, cnt = np.unique(np.round(tk["zr"][ev], 6), axis=0, return_counts=True)
    occ = dict(K=int(z["cfg_4b"][1]), D=int(z["cfg_4b"][0]), bins_used=len(u), top_share=float(cnt.max() / cnt.sum()), rho_true_vs_difficulty=float(spearmanr(sh, -d["q"][ev].mean(1))[0]))
    # the headline with ZeroRouter's success: same success on both sides
    swap = {}
    for w, Pw in (("ours_success", d["P"]), ("zr_4b", z["P_4b"]), ("zr_pub", z["P_pub"])):
        Hm = curve(d, Pw, Cm, ev); Ho = curve(d, Pw, cost_of(d, z["len_ours"]), ev); Ha = curve(d, Pw, cost_of(d, d["L"]), ev)
        swap[w] = dict(probe_vs_median=100 * saved(Ho, Hm), headroom=100 * saved(Ha, Hm), median_vs_our_median=100 * saved(Hm, H0), probe_vs_our_median=100 * saved(Ho, H0))
    rb = np.random.default_rng(0); bs = []
    for _ in range(NB):
        ii = ev[rb.integers(0, len(ev), len(ev))]; bs.append(100 * saved(curve(d, z["P_4b"], cost_of(d, z["len_ours"]), ii), curve(d, z["P_4b"], Cm, ii)))
    swap["zr_4b"]["probe_vs_median_bs"] = bs
    res["seeds"][s] = dict(saved=sv, acc=acc, occ=occ, swap=swap)
    print(f"set {s}: " + " ".join(f"{a} {v:.1f}" for a, v in sv.items()) + f" | with ZR4B success: probe vs median {swap['zr_4b']['probe_vs_median']:.1f}", flush=True)
    json.dump(res, open(Path(__file__).parent / f"zr_pricing_why_{p_}_pinned.json", "w"), indent=1, default=float)
print("DONE", flush=True)
