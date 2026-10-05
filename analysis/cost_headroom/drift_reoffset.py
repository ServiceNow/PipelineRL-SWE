"""Provider drift on deepseek-v4-flash in the fresh collection (OpenInference, unseen in training, writes ~half the tokens):
can k fresh labelled examples fix it? Both arms get the same k fresh problems' dsv4f lengths:
  ours      one multiplicative offset for dsv4f: exp(delta) = sum L_true / sum L_pred over the k (ratio of means -- the
            cost scale; a mean log ratio is biased low because predictions are means of a right-skewed length distribution)
  median    dsv4f median output length recomputed from the k (other routes unchanged)
k in {0, 10, 50, 200}; the k problems are drawn from the fresh set and EXCLUDED from evaluation (NREP=200 draws of them;
95% intervals over the draws = uncertainty from WHICH k calls are used). Policies
are Codex's deterministic calibration-selected ones (cheapest single V reaching each target on ORIGINAL calibration, chosen
with the original costs, as in evaluate_expanded_calibrated_policies.py); the corrected costs only change decisions at test.
Also reported: realized net-utility gain (ours - median) averaged over the calibration V range (utility_metric_check.py).
Fresh stratum weights and stratified paired bootstrap as in Codex's evaluation.
Usage: python drift_reoffset.py
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from carrot_compare import POOLS, read_predictions
from decompose import MK, R
from evaluate_expanded_calibrated_policies import policy, VALUES
from utility_metric_check import v_range  # noqa (module also runs its own report on import)

NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 200
out = {}
for ds in ("mmlupro", "omni500"):
    label = "MMLU-Pro" if ds == "mmlupro" else "Omni"; name, cost_file = POOLS[label]; old = R / name
    folder = R / "expanded_eval_20261001" / ds; t = np.load(folder / "tensors.npz", allow_pickle=True)
    ids, slots = list(map(str, t["problem_ids"])), list(map(str, t["model_slots"])); index = {p: i for i, p in enumerate(ids)}
    split = json.loads((old / "split_manifest.json").read_text())
    tr, ca = [np.asarray([index[str(p)] for p in split[k + "_problem_ids"]]) for k in ("train", "calibration")]
    n_old = len(np.load(old / "tensors.npz", allow_pickle=True)["problem_ids"]); fresh = np.arange(n_old, len(ids))
    valid = t["valid"].astype(bool); counts = valid.sum(2)
    q = np.where(valid, t["final_outcome"], 0).sum(2) / counts; L = np.where(valid, t["completion_tokens"], 0).sum(2) / counts
    I = np.where(valid, t["prompt_tokens"], 0).sum(2) / counts
    pin = np.array([MK[s][0] for s in slots]) / 1e6; pout = np.array([MK[s][1] for s in slots]) / 1e6; paid = (I * pin + L * pout) * 100
    med_len = np.asarray([np.median(t["completion_tokens"][tr, j][valid[tr, j]]) for j in range(len(slots))])
    learned = read_predictions(folder / "paper_cost_preds.jsonl", ids, "expected_costs", len(slots)) * 100
    learned[:n_old] = read_predictions(old / cost_file, ids[:n_old], "expected_costs", len(slots)) * 100
    p = read_predictions(folder / "success_preds.jsonl", ids, "p_successes", len(slots))
    p[:n_old] = read_predictions(old / "content_preds.jsonl", ids[:n_old], "p_successes", len(slots))
    j = slots.index("dsv4f")
    pred_len = np.maximum((learned / 100 - I * pin) / pout, 1)                       # learned predicted output tokens
    problems = [json.loads(l) for l in (folder / "problems.jsonl").read_text().splitlines()]
    strata = np.asarray([str(x.get("subject", "")) if ds == "mmlupro" else str(round(float(x.get("difficulty", 0)))) for x in problems[n_old:]])
    weights = json.loads((folder / "expansion_manifest.json").read_text())["stratum_weights"]
    med_cost0 = (I * pin + med_len * pout) * 100
    # policies chosen on ORIGINAL calibration with the ORIGINAL costs (deterministic), fixed for all k
    pols = {}
    for target in (0.65, 0.75, 0.85):
        pols[target] = {arm: policy(p[ca], c[ca], q[ca], paid[ca], target, deterministic=True) for arm, c in (("ours", learned), ("median", med_cost0))}
    va, vb = v_range(p, med_cost0, q, ca); vgrid = np.geomspace(va, vb, 25)
    rng = np.random.default_rng(7); res = {}
    print(f"\n===== {label} fresh (n={len(fresh)}); dsv4f predicted/realized mean cost before correction "
          f"{learned[fresh, j].mean() / paid[fresh, j].mean():.2f}")
    for k in (0, 10, 50, 200):
        reps = []
        for rep in range(NREP if k else 1):
            kk = rng.choice(fresh, k, replace=False) if k else np.array([], int)
            ev = np.setdiff1d(fresh, kk); pos = ev - n_old
            Lc = learned.copy(); Mc = med_cost0.copy()
            if k:
                delta = np.log(L[kk, j].sum() / pred_len[kk, j].sum())             # ratio of means (cost scale), not mean log ratio
                Lc[:, j] = (I[:, j] * pin[j] + pred_len[:, j] * np.exp(delta) * pout[j]) * 100
                Mc[:, j] = (I[:, j] * pin[j] + np.median(L[kk, j]) * pout[j]) * 100
            groups = {s: np.flatnonzero(strata[pos] == s) for s in weights}
            mean_fn = lambda x: sum(float(weights[s]) * float(x[groups[s]].mean()) for s in groups if len(groups[s]))
            row = {"calib": Lc[ev, j].mean() / paid[ev, j].mean()}
            r = np.arange(len(ev))
            for target, pl in pols.items():
                if pl["ours"] is None or pl["median"] is None:
                    continue
                mo = (pl["ours"]["V_cents"][0] * p[ev] - Lc[ev]).argmax(1); mm = (pl["median"]["V_cents"][0] * p[ev] - Mc[ev]).argmax(1)
                row[f"sav{target}"] = 1 - mean_fn(paid[ev][r, mo]) / mean_fn(paid[ev][r, mm]); row[f"dacc{target}"] = mean_fn(q[ev][r, mo] - q[ev][r, mm])
            du = np.mean([(V * q[ev][r, (V * p[ev] - Lc[ev]).argmax(1)] - paid[ev][r, (V * p[ev] - Lc[ev]).argmax(1)])
                          - (V * q[ev][r, (V * p[ev] - Mc[ev]).argmax(1)] - paid[ev][r, (V * p[ev] - Mc[ev]).argmax(1)]) for V in vgrid], 0)
            ref = np.mean([mean_fn(paid[ev][r, (V * p[ev] - Mc[ev]).argmax(1)]) for V in vgrid])
            row["util_pct"] = 100 * mean_fn(du) / ref
            if k == 0:                                                               # bootstrap CI only at k=0 and k=50 rep 0
                pass
            reps.append(row)
        m = {key: float(np.mean([x[key] for x in reps if key in x])) for key in reps[0]}
        ci = {key: [float(x) for x in np.percentile([x_[key] for x_ in reps if key in x_], [2.5, 97.5])] for key in reps[0]}
        res[k] = dict(mean=m, ci=ci)
        print(f"  k={k:<4} 95% over draws: calib [{ci['calib'][0]:.2f}, {ci['calib'][1]:.2f}] | " + " | ".join(
            f"savings {tg} [{ci[f'sav{tg}'][0]*100:+.1f}, {ci[f'sav{tg}'][1]*100:+.1f}]" for tg in (0.65, 0.75, 0.85) if f"sav{tg}" in ci)
              + f" | utility [{ci['util_pct'][0]:+.1f}, {ci['util_pct'][1]:+.1f}]")
        print(f"  k={k:<4} dsv4f calib {m['calib']:.2f} | " + " | ".join(
            f"target {tg}: savings {m.get(f'sav{tg}', np.nan)*100:+5.1f}% acc {m.get(f'dacc{tg}', np.nan)*100:+.2f}pt" for tg in (0.65, 0.75, 0.85))
              + f" | utility gain {m['util_pct']:+5.1f}% of spend")
    out[label] = res
json.dump(out, open(Path(__file__).parent / "drift_reoffset.json", "w"), indent=1, default=float)
