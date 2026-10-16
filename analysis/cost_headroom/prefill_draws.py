"""Other prefills on the 10 training sets (NEW_PATH 4.A.112): tab:size (size_sweep_eval.py, manifest split only) redone with the
training-set protocol. Per training set (DRAW=s, train + calibration reshuffled, test kept) and prefill TAG (features extracted for
size_sweep_eval.py: same prompts, same 8 relative layers x {mean, last}): a complete router on that prefill -- success = the paper's
logistic recipe (success_linear, C and Platt on calibration), cost = our ridge recipe -- against
  (a) median pricing with the SAME prefill's success readouts (the row's "vs median"), and
  (b) our router (Qwen3-4B-2507 prefill, success and cost refitted on the same training set): ours vs it.
Point + NBOOT test resamples (rng 0, the same in every set); mean test success AUC and log-length R2 over routes.
Tags: q06b, q17b, q4bhyb (Qwen3 0.6B / 1.7B / 4B hybrid), phi4mini, granite2b, smol17b. Qwen3-8B is deliberately excluded.
Usage: REASON_ROOT=.../reason_pinned OUT_DIR=... python prefill_draws.py Omni|MMLU-Pro TAG S0 S1
"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
POOL_, TAG_, S0, S1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); sys.argv = [sys.argv[0], "prefill", POOL_]
exec(open(Path(__file__).parent / "cost_generalization.py").read().split('MODE, POOL = sys.argv[1], sys.argv[2]')[0])
from baseline_cost_heads import rich
OUT = Path(os.environ.get("OUT_DIR", Path(__file__).parent))
assert TAG_ != "q8b", "Qwen3-8B is not reported (parameter-matched comparisons only)"
ds_ = {"Omni": "omni500", "MMLU-Pro": "mmlupro"}[POOL_]; FEAT = R / "prefill_size_20261005" / TAG_ / f"{ds_}.npz"
r2 = lambda y, yh: float(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())
res = {"pool": POOL_, "tag": TAG_, "seeds": {}}
for s in range(S0, S1):
    os.environ["DRAW"] = str(s); d = load(POOL_); N = len(d["ids"]); tr, ca, ev = d["tr"], d["ca"], d["ev"]; M = d["M"]
    tok_o = fit_predict("ours", view(d, tr), view(d, np.arange(N)))
    dT = dict(d, X=rich(FEAT, d["ids"])); S, Ca, T = view(dT, tr), view(dT, ca), view(dT, np.arange(N))
    Pt = success_linear("X", S, Ca, T); tok_t = fit_predict("ours", S, T); del dT, S, Ca, T
    Cm, Co, Ct = (cost_of(d, x) for x in (np.repeat(d["med"][None], N, 0), tok_o, tok_t))
    rb = np.random.default_rng(0); BS = [ev] + [ev[rb.integers(0, len(ev), len(ev))] for _ in range(NBOOT)]
    rows = []
    for ii in BS:
        Hm_t, Ht, Hm_o, Ho = curve(d, Pt, Cm, ii), curve(d, Pt, Ct, ii), curve(d, d["P"], Cm, ii), curve(d, d["P"], Co, ii)
        rows.append([saved(Ht, Hm_t), saved(Ho, Ht), saved(Ho, Hm_o)])
    rows = np.array(rows) * 100; Y = np.log(d["L"])
    r_ = dict(vs_median=float(rows[0, 0]), vs_median_bs=rows[1:, 0].tolist(), ours_vs_it=float(rows[0, 1]), ours_vs_it_bs=rows[1:, 1].tolist(),
              ours_vs_median=float(rows[0, 2]),
              auc=float(np.mean([auc(d["y0"][ev, k], Pt[ev, k]) for k in range(M)])), auc_ours=float(np.mean([auc(d["y0"][ev, k], d["P"][ev, k]) for k in range(M)])),
              r2=float(np.mean([r2(Y[ev, k], np.log(tok_t[ev, k])) for k in range(M)])), r2_ours=float(np.mean([r2(Y[ev, k], np.log(tok_o[ev, k])) for k in range(M)])))
    res["seeds"][s] = r_
    lo, hi = np.nanpercentile(r_["ours_vs_it_bs"], [2.5, 97.5])
    print(f"  set {s}: {TAG_} vs median (own success) {r_['vs_median']:+5.1f} | ours vs it {r_['ours_vs_it']:+5.1f} [{lo:+.1f}, {hi:+.1f}] | ours vs median "
          f"{r_['ours_vs_median']:+5.1f} | AUC {r_['auc']:.3f} (ours {r_['auc_ours']:.3f}) | R2 {r_['r2']:.2f} (ours {r_['r2_ours']:.2f})", flush=True)
    json.dump(res, open(OUT / f"prefill_draws_{POOL_.lower().replace('-', '')}_{TAG_}_{S0}_{S1}{os.environ.get('RESULT_TAG', '')}.json", "w"), indent=1, default=float)
print("DONE", flush=True)
