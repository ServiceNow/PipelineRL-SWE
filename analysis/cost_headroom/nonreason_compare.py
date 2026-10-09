"""Reasoning vs NON-reasoning pools on the same problems (NEW_PATH 4.A.64 data; TMLR claim C1), pinned, billed.
Reasoning pool: the paper's 5 routes (gpt-oss-20b low/medium, deepseek-v4-flash thinking, gpt-oss-120b medium/high) with the
paper's readouts (tmlr_free_analyses header). Non-reasoning pool: deepseek-v4-flash thinking OFF, llama-3.1-8b, qwen3-30b-a3b-instruct,
llama-3.3-70b, qwen3-235b-a22b-instruct, one draw each, every route pinned to one provider. Same problems, same split, same Qwen3-4B
prefill features; readouts refitted with the paper recipes (success: activation_content_preds --rich --select-C; cost: ridge on
log length, smearing + level match, as probe_model_compare). Realized cost = billed usage_cost (math) or tokens x billed rates fitted
from the math rows' usage_cost (LCB, which records no cost).
On the common test problems (every route of both pools valid):
  spread     per route: accuracy, mean / median output, sd(log output), p90/p10 output, $/1000 queries; cost-readout test R2
  routing    headroom (oracle per-query cost vs median), ours vs median, cost-from-success vs median; paired bootstrap (300)
  mixed      the 10-route union (both pools): same three numbers
  deepseek   thinking on vs off: accuracy, mean output, corr of log output across problems
Usage: REASON_ROOT=.../reason_pinned python nonreason_compare.py Omni|MMLU-Pro|LCB
"""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).parent))
exec(open(Path(__file__).parent / "tmlr_free_analyses.py").read().split("# ---------------------------------------------------------------- 1.")[0]
     .replace('print(f"===== {POOL}', 'print(f"===== reasoning pool {POOL}'))
REALR = Path("/mnt/llmd/results/exps/aristides/reason"); PY = sys.executable
NR = ["nr_ds4off", "nr_llama8", "nr_qw30", "nr_llama70", "nr_qw235"]; ds_ = {"Omni": "omni500", "MMLU-Pro": "mmlupro", "LCB": "lcb"}[POOL]
OUT = REALR / "nonreason_eval_20261009" / ds_; OUT.mkdir(parents=True, exist_ok=True)


def latest_valid(files, rule):
    rows = {}
    for f in files:
        if not f.exists():
            continue
        for l in open(f):
            r = json.loads(l)
            if rule == "lcb":
                good = isinstance(r.get("resolved"), bool) and (str(r.get("full_output") or "").strip() or r.get("finish_reason") == "length")
            else:
                good = r.get("finish_reason") != "error" and not r.get("error") and r.get("completion_tokens")
            if good:
                rows[str(r["problem_id"])] = r
    return rows


# ---- billed $/token per non-reasoning route, fitted on the math rows' usage_cost (cost = a * prompt + b * completion)
rateN, rowsN = [], []
for r_ in NR:
    mr = [r for d in ("math_pool_nonreason", "math_expand_nonreason") for ds2 in ("omni500", "mmlupro")
          for r in latest_valid([REALR / d / ds2 / f"{r_}_d0.jsonl"], "math").values() if r.get("usage_cost") is not None]
    A = np.array([[r["prompt_tokens"], r["completion_tokens"]] for r in mr], float); b = np.array([r["usage_cost"] for r in mr])
    rateN.append(np.maximum(np.linalg.lstsq(A, b, rcond=None)[0], 0) * 1e6)
    if POOL == "LCB":
        files = [REALR / "pool_v2_lcb_nonreason" / f"{r_}_{s}_d0.jsonl" for s in ("train", "eval")]; rowsN.append(latest_valid(files, "lcb"))
    else:
        files = [REALR / d / ds_ / f"{r_}_d0.jsonl" for d in ("math_pool_nonreason", "math_expand_nonreason")]; rowsN.append(latest_valid(files, "math"))
rateN = np.array(rateN)
print("billed $/M (in, out): " + "  ".join(f"{s} {a:.3f}/{b:.3f}" for s, (a, b) in zip(NR, rateN)), flush=True)
n = len(ids); vN = np.zeros((n, 5), bool); qN = np.zeros((n, 5)); LN = np.ones((n, 5)); IN = np.zeros((n, 5)); paidN = np.zeros((n, 5))
for k, rows in enumerate(rowsN):
    for p, r in rows.items():
        if p in idx:
            i = idx[p]; vN[i, k] = True; qN[i, k] = float(bool(r["resolved"])); LN[i, k] = max(r["completion_tokens"], 1); IN[i, k] = r["prompt_tokens"]
            paidN[i, k] = r["usage_cost"] if r.get("usage_cost") is not None else IN[i, k] * rateN[k, 0] / 1e6 + LN[i, k] * rateN[k, 1] / 1e6
for k in range(5):           # a route with no valid draw on a problem: fill the prompt length (only used where invalid rows are excluded)
    IN[~vN[:, k], k] = I[~vN[:, k]].mean(1) if (~vN[:, k]).any() else 0
print("valid per route (train / test): " + "  ".join(f"{s} {vN[tr, k].sum()}/{vN[ev, k].sum()}" for k, s in enumerate(NR)), flush=True)

# ---- success readouts (paper recipe) on the non-reasoning tensors
np.savez(OUT / "tensors.npz", final_outcome=qN[:, :, None] > 0, execution_outcome=qN[:, :, None] > 0, weak_verifier_outcome=qN[:, :, None] > 0,
         valid=vN[:, :, None], prompt_tokens=IN[:, :, None].astype(np.float32), completion_tokens=LN[:, :, None].astype(np.float32),
         problem_ids=np.array(ids), model_slots=np.array(NR), schema_version=3)
(OUT / "split_manifest.json").write_text((old / "split_manifest.json").read_text())
for fn in ("problems.jsonl",):
    if not (OUT / fn).exists():
        (OUT / fn).symlink_to((F / fn).resolve() if (F / fn).exists() else (old / fn).resolve())
subprocess.run([PY, str(Path(__file__).parents[2] / "pipelinerl/swe/scripts/livecodebench/activation_content_preds.py"), "--activations", str(feat),
                "--tensors-dir", str(OUT), "--rich", "--select-C", "--out", str(OUT / "success_preds.jsonl")],
               stdout=open(OUT / "content.log", "w"), stderr=subprocess.STDOUT, check=True)
PN = np.clip(read_predictions(OUT / "success_preds.jsonl", ids, "p_successes", 5), 1e-4, 1 - 1e-4)

# ---- cost readouts: dedicated (prefill ridge) and from-success (ridge on the success logits), both priced at billed rates
X = rich(feat, ids); trv = lambda k: tr[vN[tr, k]]
tokN, tokFS, r2N = np.zeros((n, 5)), np.zeros((n, 5)), []
lgN = np.log(PN / (1 - PN)); FS = np.c_[lgN, lgN ** 2]
for k in range(5):
    t_ = trv(k); y = np.log(LN[:, k]); Xs = StandardScaler().fit(X[t_]).transform(X)
    for out_, Z, alphas in ((tokN, Xs, np.geomspace(1e1, 1e7, 13)), (tokFS, FS, None)):
        m = RidgeCV(alphas=alphas).fit(Z[t_], y[t_]) if alphas is not None else make_pipeline(StandardScaler(), RidgeCV(alphas=np.geomspace(1e-3, 1e4, 15))).fit(Z[t_], y[t_])
        yh = m.predict(Z); o = np.exp(yh) * np.mean(np.exp(y[t_] - yh[t_])); out_[:, k] = o * LN[t_, k].mean() / o[t_].mean()
    e_ = ev[vN[ev, k]]; r2N.append(1 - ((np.log(LN[e_, k]) - np.log(tokN[e_, k])) ** 2).sum() / ((np.log(LN[e_, k]) - np.log(LN[e_, k]).mean()) ** 2).sum())
medN = np.array([np.median(LN[trv(k), k]) for k in range(5)])
cN = lambda tk: IN * rateN[:, 0] / 1e6 + tk * rateN[:, 1] / 1e6
CN_ours, CN_fs, CN_med = cN(tokN), cN(tokFS), cN(np.repeat(medN[None], n, 0))
lgR = np.log(P / (1 - P)); FSR = np.c_[lgR, lgR ** 2]; tokRFS = np.zeros_like(L)
for k in range(M):
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.geomspace(1e-3, 1e4, 15))).fit(FSR[tr], np.log(L[tr, k])); yh = m.predict(FSR)
    o = np.exp(yh) * np.mean(np.exp(np.log(L[tr, k]) - yh[tr])); tokRFS[:, k] = o * L[tr, k].mean() / o[tr].mean()
C_fs = costs(tokRFS, rates)


# ---- frontier arithmetic (same as the suite), parameterized by pool
def front_g(Pm, C, pd, qq, ii):
    pts = []
    for V in VALUES:
        m = (V * Pm[ii] - C[ii]).argmax(1); kk = np.arange(len(ii)); pts.append((pd[ii][kk, m].mean(), qq[ii][kk, m].mean()))
    return hull(pts)


def saved_g(Pm, qq, pd, Ca, Cb, ii):
    Ha, Hb = front_g(Pm, Ca, pd, qq, ii), front_g(Pm, Cb, pd, qq, ii); lo, hi = max(Ha[0][1], Hb[0][1]), min(Ha[-1][1], Hb[-1][1])
    if hi <= lo:
        return np.nan
    T = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 12)
    return 1 - float(np.exp(np.nanmean(np.log([cost_at(Ha, x) / cost_at(Hb, x) for x in T]))))


evc = ev[vN[ev].all(1)]
POOLS_ = {"reasoning": (P, q, paid, C_ours, C_fs, C_med), "nonreasoning": (PN, qN, paidN, CN_ours, CN_fs, CN_med),
          "mixed": tuple(np.concatenate([a, b], 1) for a, b in zip((P, q, paid, C_ours, C_fs, C_med), (PN, qN, paidN, CN_ours, CN_fs, CN_med)))}
res = {"pool": POOL, "n_test_common": int(len(evc)), "rates_nonreasoning_per_M": dict(zip(NR, rateN.tolist())), "routing": {}, "spread": {}}
rng = np.random.default_rng(0); BS = [evc[rng.integers(0, len(evc), len(evc))] for _ in range(300)]
for nm, (Pm, qq, pd, Co, Cf, Cm) in POOLS_.items():
    row = {}
    for arm, Ca in (("headroom", pd), ("ours", Co), ("from_success", Cf)):
        g = saved_g(Pm, qq, pd, Ca, Cm, evc); b = [saved_g(Pm, qq, pd, Ca, Cm, bb) for bb in BS]; row[arm] = [g, *np.nanpercentile(b, [2.5, 97.5]).tolist()]
    row["capture"] = row["ours"][0] / row["headroom"][0]; res["routing"][nm] = row
    print(f"  {nm:<13} n={len(evc)}  headroom {row['headroom'][0]*100:+5.1f}%  ours {row['ours'][0]*100:+5.1f}% [{row['ours'][1]*100:+.1f}, {row['ours'][2]*100:+.1f}]"
          f"  from-success {row['from_success'][0]*100:+5.1f}%  (capture {row['capture']*100:.0f}%)", flush=True)
for nm, (slots_, qq, LL, pd, r2s) in {"reasoning": (slots, q, L, paid, None), "nonreasoning": (NR, qN, LN, paidN, r2N)}.items():
    for k, s in enumerate(slots_):
        x = LL[evc, k]; lx = np.log(x)
        res["spread"][s] = dict(pool=nm, acc=float(qq[evc, k].mean()), mean_out=float(x.mean()), median_out=float(np.median(x)), sd_log_out=float(lx.std()),
                                p90_p10=float(np.percentile(x, 90) / np.percentile(x, 10)), usd_per_1000=float(pd[evc, k].mean() * 1000),
                                cost_r2=None if r2s is None else float(r2s[k]))
        d = res["spread"][s]
        print(f"  {nm[:5]} {s:<10} acc {d['acc']:.3f}  mean out {d['mean_out']:7.0f}  median {d['median_out']:6.0f}  sd(log) {d['sd_log_out']:.2f}  "
              f"p90/p10 {d['p90_p10']:6.1f}  ${d['usd_per_1000']:.3f}/1000" + ("" if d["cost_r2"] is None else f"  cost R2 {d['cost_r2']:+.2f}"), flush=True)
kd, ko = slots.index("dsv4f"), 0
res["deepseek"] = dict(acc_on=float(q[evc, kd].mean()), acc_off=float(qN[evc, ko].mean()), mean_out_on=float(L[evc, kd].mean()), mean_out_off=float(LN[evc, ko].mean()),
                       corr_log_out=float(np.corrcoef(np.log(L[evc, kd]), np.log(LN[evc, ko]))[0, 1]))
print(f"  deepseek thinking on vs off: acc {res['deepseek']['acc_on']:.3f} vs {res['deepseek']['acc_off']:.3f}; mean output {res['deepseek']['mean_out_on']:.0f} vs "
      f"{res['deepseek']['mean_out_off']:.0f}; corr(log output) {res['deepseek']['corr_log_out']:+.2f}", flush=True)
json.dump(res, open(Path(os.environ.get("OUT_DIR", Path(__file__).parent)) / f"nonreason_compare_{ds_}{os.environ.get('RESULT_TAG', '')}.json", "w"), indent=1, default=float)
