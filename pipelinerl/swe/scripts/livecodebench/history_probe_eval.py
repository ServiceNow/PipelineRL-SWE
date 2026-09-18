#!/usr/bin/env python3
"""Does re-prefilling the scout on the failed attempts improve every route's next-draw belief?

Predictor-level check before any policy work. Both arms come from the SAME extraction jobs
(launch_history_probe.sh), same readout (mean + last over 8 layers), same estimator:

  A  prompt-only probe (today's method): fitted on the prompt-only examples of train problems,
     Platt-calibrated on calibration problems, then the count update p = theta * k/(k + n_m).
  B  history probe: fitted on every example of train problems (prompt alone, one failure, a
     scout-then-other pair), Platt-calibrated on calibration examples. No count update: whatever a
     failure implies has to come from reading it.
  C  control: A recalibrated on calibration with the failure COUNT on each route as extra
     inputs -- knows which routes failed, never what they produced.
  D  C plus B's prediction: does reading the failure add anything once the counts are known?

Target for route m on an example: the success rate of m's draws on that problem EXCLUDING any
draw in the history (so a failed draw never labels itself). Soft labels are fitted by weighting
each example twice (y=1 with weight r, y=0 with weight 1-r). Scored on test problems only:
Brier against the soft rate, and AUC for "route m solves it on some remaining draw".
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def platt(raw_cal, y_cal, w_cal, raw):
    lo = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    m = LogisticRegression(C=1e6, max_iter=2000).fit(lo(raw_cal)[:, None], y_cal, sample_weight=w_cal)
    return m.predict_proba(lo(raw)[:, None])[:, 1]


def soft_rows(X, r):
    ok = np.isfinite(r)
    X, r = X[ok], r[ok]
    return (np.vstack([X, X]), np.r_[np.ones(len(r)), np.zeros(len(r))], np.r_[r, 1 - r])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/mnt/llmd/results/exps/aristides/reason/history_probe")
    ap.add_argument("--tensors-dir", default="/mnt/llmd/results/exps/aristides/reason/"
                    "lcb_pool64k_full_prepared/tensors_v3")
    ap.add_argument("--variant", default="count", help="prompt/manifest variant")
    ap.add_argument("--act-tag", default="", help="activation file tag (defaults to variant)")
    ap.add_argument("--readouts", default="mean,last",
                    help="comma-separated readouts to concatenate: mean,last,content_last,...")
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--kappa", type=float, default=0.95)
    ap.add_argument("--C", type=float, default=0.05)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    D = Path(a.dir)

    ids, feats = [], []
    for i in range(a.shards):
        z = np.load(D / f"act_{a.act_tag or a.variant}_shard{i}.npz", allow_pickle=True)
        ids += [str(x) for x in z["problem_ids"]]
        feats.append(np.concatenate([z[k].reshape(len(z[k]), -1)
                                     for k in a.readouts.split(",")], axis=1))
    X = np.concatenate(feats).astype(np.float32); del feats
    man = {json.loads(l)["example_id"]: json.loads(l)
           for l in open(D / f"{a.variant}_manifest.jsonl") if l.strip()}
    t = np.load(Path(a.tensors_dir) / "tensors.npz", allow_pickle=True)
    ok = (t["final_outcome"] & t["valid"]).astype(float); valid = t["valid"].astype(bool)
    tp = {str(p): i for i, p in enumerate(t["problem_ids"])}
    slots = [str(s) for s in t["model_slots"]]; M = len(slots)
    sm = json.loads((Path(a.tensors_dir) / "split_manifest.json").read_text())
    split = {**{str(p): "train" for p in sm["train_problem_ids"]},
             **{str(p): "cal" for p in sm["calibration_problem_ids"]},
             **{str(p): "test" for p in sm["test_problem_ids"]}}

    N = len(ids)
    pid = [man[e]["problem_id"] for e in ids]
    hist = [[tuple(h) for h in man[e]["history"]] for e in ids]
    sp = np.array([split.get(p, "none") for p in pid])
    nfail = np.zeros((N, M))
    R = np.full((N, M), np.nan)
    for i in range(N):
        pi = tp[pid[i]]
        used = set(hist[i])
        for s, k in hist[i]:
            nfail[i, slots.index(s)] += 1
        for m, s in enumerate(slots):
            keep = [k for k in range(ok.shape[2]) if valid[pi, m, k] and (s, k) not in used]
            if keep:
                R[i, m] = ok[pi, m, keep].mean()
    is0 = np.array([len(h) == 0 for h in hist])
    print(f"{N} examples: {is0.sum()} prompt-only; splits "
          f"{ {k: int((sp == k).sum()) for k in ('train', 'cal', 'test')} }")

    sc = StandardScaler().fit(X[(sp == "train")])
    Xs = sc.transform(X).astype(np.float32)
    C = a.C / max(1, X.shape[1] // 2560)
    PA = np.zeros((N, M)); PB = np.zeros((N, M))
    row_of_prompt = {pid[i]: i for i in range(N) if is0[i]}
    for m, s in enumerate(slots):
        # A: prompt-only probe, per problem, then the count update
        tr = np.where((sp == "train") & is0)[0]
        Xa, ya, wa = soft_rows(Xs[tr], R[tr, m])
        A = LogisticRegression(C=C, max_iter=3000).fit(Xa, ya, sample_weight=wa)
        theta_raw = A.predict_proba(Xs[[row_of_prompt[p] for p in pid]])[:, 1]
        ca = np.where((sp == "cal") & is0 & np.isfinite(R[:, m]))[0]
        theta = platt(np.r_[theta_raw[ca], theta_raw[ca]], np.r_[np.ones(len(ca)), np.zeros(len(ca))],
                      np.r_[R[ca, m], 1 - R[ca, m]], theta_raw)
        PA[:, m] = theta * a.kappa / (a.kappa + nfail[:, m])
        # B: history probe on every example
        tr = np.where(sp == "train")[0]
        Xb, yb, wb = soft_rows(Xs[tr], R[tr, m])
        B = LogisticRegression(C=C, max_iter=3000).fit(Xb, yb, sample_weight=wb)
        raw = B.predict_proba(Xs)[:, 1]
        cb = np.where((sp == "cal") & np.isfinite(R[:, m]))[0]
        PB[:, m] = platt(np.r_[raw[cb], raw[cb]], np.r_[np.ones(len(cb)), np.zeros(len(cb))],
                         np.r_[R[cb, m], 1 - R[cb, m]], raw)
        print(f"  fitted route {s}")

    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    PC = np.zeros((N, M)); PD = np.zeros((N, M))
    cal_all = np.where(sp == "cal")[0]
    for m in range(M):
        for P_, F in ((PC, np.c_[lg(PA[:, m]), nfail]), (PD, np.c_[lg(PA[:, m]), nfail, lg(PB[:, m])])):
            j = cal_all[np.isfinite(R[cal_all, m])]
            r = R[j, m]
            f = LogisticRegression(C=1e4, max_iter=5000).fit(
                np.vstack([F[j], F[j]]), np.r_[np.ones(len(j)), np.zeros(len(j))],
                sample_weight=np.r_[r, 1 - r])
            P_[:, m] = f.predict_proba(F)[:, 1]
    ARMS = {"A": PA, "C": PC, "B": PB, "D": PD}

    def case(i):
        h = hist[i]
        if not h: return "no history"
        if len(h) == 1: return f"one {h[0][0]} failure"
        return f"scout + {h[1][0]} failures"
    cases = defaultdict(list)
    for i in np.where(sp == "test")[0]:
        cases[case(i)].append(i)
        if hist[i]:
            cases["ALL with history"].append(i)
    lines = []
    hdr = (f"{'history (test problems)':26s} {'route':7s} {'n':>5s} | Brier "
           + " ".join(f"{k:>6s}" for k in ARMS) + " | AUC " + " ".join(f"{k:>5s}" for k in ARMS))
    print(hdr); lines.append(hdr)
    for c in sorted(cases):
        idx = np.array(cases[c])
        for m, s in enumerate(slots):
            j = idx[np.isfinite(R[idx, m])]
            if len(j) < 20: continue
            r = R[j, m]
            yb_ = (r > 0).astype(int)
            br = [np.mean((P_[j, m] - r) ** 2) for P_ in ARMS.values()]
            au = [roc_auc_score(yb_, P_[j, m]) if 0 < yb_.mean() < 1 else np.nan
                  for P_ in ARMS.values()]
            ln = (f"{c:26s} {s:7s} {len(j):5d} | Brier " + " ".join(f"{x:6.4f}" for x in br)
                  + " | AUC " + " ".join(f"{x:5.3f}" for x in au))
            print(ln); lines.append(ln)
    if a.out:
        Path(a.out).write_text("\n".join(lines) + "\n")
        np.savez(Path(a.out).with_suffix(".npz"), ids=np.array(ids), PA=PA, PB=PB, PC=PC, PD=PD,
                 R=R, split=sp)


if __name__ == "__main__":
    main()
