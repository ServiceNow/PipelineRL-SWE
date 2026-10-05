"""Live run (NEW_PATH 4.A.56), step 4: what each frozen policy actually spent and scored on 1,000 never-used MMLU-Pro problems.
For each accuracy target, ours (prefill cost readouts) vs median-length pricing (same success readouts, the paper rule):
billed spend per problem, accuracy, cost saved (1 - spend_ours / spend_median) and accuracy difference, paired problem bootstrap
(2,000). Also predicted / realized cost per route (live calibration of the frozen cost readouts, incl. provider drift).
Usage: python live_eval.py
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decompose import R
from billed import RATE
OUT = R / "live_run_20261005"
rate_of = lambda s: RATE["oss120" if "120" in s else ("oss20" if s.startswith("oss20") else "dsv4f")]
D = json.load(open(OUT / "decisions.json")); ids, slots = D["problem_ids"], D["slots"]
calls = {}
for l in open(OUT / "calls.jsonl"):
    r = json.loads(l)
    if r.get("finish_reason") != "error":
        calls[(r["problem_id"], r["route_label"])] = r
tok = np.array(D["tokens_pred"]); Ipred = np.array(D["input_tokens_pred"])
res = dict(n_problems=len(ids), n_calls=len(calls), spent_usd=sum(float(r.get("usage_cost") or 0) for r in calls.values()))
print(f"{len(ids)} live problems, {len(calls)} successful calls, billed ${res['spent_usd']:.3f}")
print("per route: predicted/realized output tokens, predicted/realized billed cost (calls made)")
res["route_calibration"] = {}
for k, s in enumerate(slots):
    rows = [(i, calls[(p, s)]) for i, p in enumerate(ids) if (p, s) in calls]
    if not rows:
        continue
    pt = sum(tok[i, k] for i, _ in rows); rt = sum(r["completion_tokens"] for _, r in rows)
    pc = sum(Ipred[i, k] * rate_of(s)[0] + tok[i, k] * rate_of(s)[1] for i, _ in rows); rc = sum(float(r.get("usage_cost") or 0) for _, r in rows)
    provs = {}
    for _, r in rows:
        provs[r.get("provider")] = provs.get(r.get("provider"), 0) + 1
    res["route_calibration"][s] = dict(n=len(rows), tokens_ratio=pt / rt, cost_ratio=pc / max(rc, 1e-12), acc=float(np.mean([r["resolved"] for _, r in rows])), providers=provs)
    print(f"  {s:<9} n={len(rows):4d}  tokens pred/real {pt/rt:.2f}  cost pred/real {pc/max(rc,1e-12):.2f}  acc {res['route_calibration'][s]['acc']:.3f}  providers {provs}")
rng = np.random.default_rng(0)
def spend(r, mode):
    if mode == "billed":
        return float(r.get("usage_cost") or 0)
    a, b = rate_of(r["route_label"]); return r["prompt_tokens"] * a + r["completion_tokens"] * b   # decision-time rates, realized tokens


for mode in ("billed", "decision_rates"):
  print(f"--- spend = {'billed usage_cost' if mode == 'billed' else 'realized tokens x the billed rates the router used (price drift removed)'}")
  res["targets_" + mode] = {}
  for tg, d in D["decisions"].items():
      arm = {}
      for a in ("ours", "median"):
          ok = [(p, s) in calls for p, s in zip(ids, d[a])]
          arm[a] = (np.array([float(calls[(p, s)]["resolved"]) if (p, s) in calls else np.nan for p, s in zip(ids, d[a])]),
                    np.array([spend(calls[(p, s)], mode) if (p, s) in calls else np.nan for p, s in zip(ids, d[a])]), np.array(ok))
      m = arm["ours"][2] & arm["median"][2]                                  # problems where both arms' calls returned
      qo, co, qm, cm = arm["ours"][0][m], arm["ours"][1][m], arm["median"][0][m], arm["median"][1][m]; n = int(m.sum())
      sv = 1 - co.sum() / cm.sum(); da = qo.mean() - qm.mean(); bsi = [rng.integers(0, n, n) for _ in range(2000)]
      s_bs = [1 - co[b].sum() / cm[b].sum() for b in bsi]; a_bs = [qo[b].mean() - qm[b].mean() for b in bsi]
      same = float(np.mean([x == y for x, y in zip(np.array(d["ours"])[m], np.array(d["median"])[m])]))
      res["targets_" + mode][tg] = dict(n=n, acc_ours=float(qo.mean()), acc_median=float(qm.mean()), spend_ours=float(co.mean()), spend_median=float(cm.mean()),
                                saved=sv, saved_ci=list(np.percentile(s_bs, [2.5, 97.5])), acc_diff=da, acc_diff_ci=list(np.percentile(a_bs, [2.5, 97.5])),
                                same_route_share=same, V=d["V"])
      print(f"target {tg}: n={n}  ours acc {qo.mean()*100:.1f}% spend {co.mean()*1e3:.3f} m$ | median acc {qm.mean()*100:.1f}% spend {cm.mean()*1e3:.3f} m$ | "
            f"saved {sv*100:+.1f}% [{np.percentile(s_bs,2.5)*100:+.1f}, {np.percentile(s_bs,97.5)*100:+.1f}]  acc diff {da*100:+.2f} pp "
            f"[{np.percentile(a_bs,2.5)*100:+.2f}, {np.percentile(a_bs,97.5)*100:+.2f}]  same route {same:.2f}")
json.dump(res, open(Path(__file__).parent / "live_eval.json", "w"), indent=1, default=float)
