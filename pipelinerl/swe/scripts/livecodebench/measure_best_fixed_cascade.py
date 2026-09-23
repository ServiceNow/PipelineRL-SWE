#!/usr/bin/env python3
"""The strongest FIXED baseline: every (k per rung) plan, run cheapest-first with stop-on-success.

This exists because the replay's Zero Router family was too weak to be an honest baseline. It held
8 arms -- three single-model, two two-tier cascades, one single-pass cascade, two best-of-16 -- and
omitted the plans that actually win, which are multi-draw-per-tier. Measured on pool_v2 the gap was
large enough to change the paper's headline: the replay's family reached 62.4% at 0.05c where a
properly enumerated cascade reaches 73.3%, so a reported "+12.9pt over the Zero Router" was really
"+3.0pt over a fixed plan anyone would have tried".

Two rules this enforces, both learned the hard way here:
  * The plan is SELECTED ON TRAIN+CALIBRATION and scored on TEST. Picking it on test is worth up to
    2.8pt of phantom baseline strength, which flatters us when we then beat it.
  * Execution is sequential with stop-on-success on the REAL draws, so the baseline gets the same
    "stop as soon as you win" advantage the adaptive policy has. A pre-committed allocation is a
    strawman: it loses ~4pt to the sequential version.

Reported alongside RoR v1, never instead of it: the cascade is what practitioners deploy, RoR v1 is
the closest prior method, and the two answer different questions.
"""
from __future__ import annotations
import argparse, itertools, json
from pathlib import Path
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tensors-dir", required=True)
    ap.add_argument("--prices", required=True, help="label=USD_per_M, comma separated")
    ap.add_argument("--max-total-draws", type=int, default=20)
    ap.add_argument("--budgets", default="0.05,0.1,0.2,0.4,0.8,1.6")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    T = Path(a.tensors_dir)
    t = np.load(T / "tensors.npz", allow_pickle=True)
    ok = (t["final_outcome"] & t["valid"]).astype(bool); valid = t["valid"].astype(bool)
    slots = [str(s) for s in t["model_slots"]]
    price = {k.split("=")[0]: float(k.split("=")[1]) for k in a.prices.split(",") if k.strip()}
    usd = np.zeros(ok.shape, float)
    for m, s in enumerate(slots):
        usd[:, m, :] = (t["prompt_tokens"][:, m, :] + t["completion_tokens"][:, m, :]) * price[s] / 1e6
    sm = json.loads((T / "split_manifest.json").read_text())
    pids = [str(p) for p in t["problem_ids"]]
    test = set(map(str, sm["test_problem_ids"]))
    fit = set(map(str, sm["train_problem_ids"])) | set(map(str, sm["calibration_problem_ids"]))
    ti = [i for i, p in enumerate(pids) if p in test]
    fi = [i for i, p in enumerate(pids) if p in fit]
    K = [int(valid[:, m, :].sum(1).max()) for m in range(len(slots))]
    plans = [p for p in itertools.product(*[range(k + 1) for k in K])
             if 0 < sum(p) <= a.max_total_draws]

    def run(plan, rows):
        acc, cost = [], []
        for i in rows:
            spent, solved = 0.0, False
            for m in range(len(slots)):                       # cheapest rung first
                for k in [k for k in range(ok.shape[2]) if valid[i, m, k]][: plan[m]]:
                    spent += usd[i, m, k]
                    if ok[i, m, k]:
                        solved = True; break
                if solved:
                    break
            acc.append(solved); cost.append(spent)
        return float(np.mean(cost) * 100), float(np.mean(acc) * 100)

    F = {p: run(p, fi) for p in plans}
    S = {p: run(p, ti) for p in plans}
    H = [(0.0, 0.0, None)]
    for p, (c, acc) in sorted(F.items(), key=lambda kv: kv[1][0]):
        while len(H) >= 2 and (H[-1][1] - H[-2][1]) * (c - H[-1][0]) <= (acc - H[-1][1]) * (H[-1][0] - H[-2][0]):
            H.pop()
        if acc > H[-1][1]:
            H.append((c, acc, p))
    print(f"{len(plans)} plans; {len(fi)} fit / {len(ti)} test problems\n")
    print(f"{'budget':>9}{'plan (chosen on fit)':>34}{'fit acc':>9}{'TEST acc':>10}{'test cost':>11}")
    out = []
    for b in [float(x) for x in a.budgets.split(",")]:
        elig = [(c, acc, p) for c, acc, p in H if c <= b and p is not None]
        if not elig:
            continue
        _, _, p = elig[-1]
        tc, ta = S[p]
        print(f"{b:>8.2f}c{str(dict(zip(slots, p))):>34}{F[p][1]:>8.1f}%{ta:>9.1f}%{tc:>10.3f}c")
        out.append({"budget": b, "plan": dict(zip(slots, p)), "fit_acc": F[p][1],
                    "test_acc": ta, "test_cost": tc})
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
