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
    ap.add_argument("--export-dir", default="", help=(
        "write replay inputs: content_preds_A.jsonl (prompt-only probe, Platt), history_preds.jsonl "
        "(B on prompt-only and single-failure examples, with the re-prefill's cost) and "
        "history_recal.json (C and D coefficients)"))
    ap.add_argument("--depth-buckets", default="0,1,2,3", help=(
        "calibrate the history head separately by TOTAL failures so far (bucket edges; the last "
        "is 'or more'). Pooled calibration is dominated by deep states, while tight budgets act at "
        "depth 0-1 -- the states where the deep probe was worse than the decay."))
    ap.add_argument("--apply-dir", default="", help=(
        "apply the fitted B heads to a second activation set (e.g. every reachable replay state "
        "from build_deep_state_prompts.py) and write its history_preds.jsonl into --export-dir"))
    ap.add_argument("--apply-variant", default="state")
    ap.add_argument("--apply-act-tag", default="statejudge")
    ap.add_argument("--apply-shards", type=int, default=12)
    ap.add_argument("--scout-usd-per-token", type=float, default=0.278e-6)
    ap.add_argument("--chars-per-token", type=float, default=3.2)
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
    PA = np.zeros((N, M)); PB = np.zeros((N, M)); THETA = np.zeros((N, M))
    HEAD: dict[int, object] = {}; PLATT: dict[int, object] = {}; BUCKET: dict[int, object] = {}
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
        THETA[:, m] = theta
        PA[:, m] = theta * a.kappa / (a.kappa + nfail[:, m])
        # B: history probe on every example
        tr = np.where(sp == "train")[0]
        Xb, yb, wb = soft_rows(Xs[tr], R[tr, m])
        B = LogisticRegression(C=C, max_iter=3000).fit(Xb, yb, sample_weight=wb)
        raw = B.predict_proba(Xs)[:, 1]
        cb = np.where((sp == "cal") & np.isfinite(R[:, m]))[0]
        PB[:, m] = platt(np.r_[raw[cb], raw[cb]], np.r_[np.ones(len(cb)), np.zeros(len(cb))],
                         np.r_[R[cb, m], 1 - R[cb, m]], raw)
        HEAD[m] = B
        # Depth-bucketed calibration: one Platt map per bucket of total failures so far. A pooled
        # map is dominated by deep states, while tight budgets act at depth 0-1 -- exactly where
        # the deep probe was worse than the count decay.
        _edges = [int(x) for x in a.depth_buckets.split(",") if x.strip()]
        _bucket = lambda d: max(i for i, e in enumerate(_edges) if d >= e)
        _dep = np.array([_bucket(len(h)) for h in hist])
        _lo = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
        _maps = {}
        for _b in sorted(set(_dep)):
            _j = np.where((sp == "cal") & (_dep == _b) & np.isfinite(R[:, m]))[0]
            if len(_j) < 40:
                _j = cb                       # too few calibration examples: pooled fit
            _r = R[_j, m]
            _maps[_b] = LogisticRegression(C=1e6, max_iter=2000).fit(
                _lo(np.r_[raw[_j], raw[_j]])[:, None],
                np.r_[np.ones(len(_j)), np.zeros(len(_j))],
                sample_weight=np.r_[_r, 1 - _r])
        for _b in sorted(set(_dep)):
            _sel = _dep == _b
            PB[_sel, m] = _maps[_b].predict_proba(_lo(raw[_sel])[:, None])[:, 1]
        PLATT[m] = (lambda mp: (lambda q, d: mp[min(d, max(mp))].predict_proba(
            _lo(q)[:, None])[:, 1]))(_maps)
        BUCKET[m] = _bucket
        print(f"  fitted route {s}")

    if a.apply_dir:
        # Same scaler, same fitted B heads, same Platt maps: only the inputs change.
        AD = Path(a.apply_dir)
        aids, afe = [], []
        for i in range(a.apply_shards):
            z = np.load(AD / f"act_{a.apply_act_tag}_shard{i}.npz", allow_pickle=True)
            aids += [str(x) for x in z["problem_ids"]]
            afe.append(np.concatenate([z[k].reshape(len(z[k]), -1)
                                       for k in a.readouts.split(",")], axis=1))
        AX = sc.transform(np.concatenate(afe).astype(np.float32)).astype(np.float32); del afe
        aman = {json.loads(l)["example_id"]: json.loads(l)
                for l in open(AD / f"{a.apply_variant}_manifest.jsonl") if l.strip()}
        adep = np.array([BUCKET[0](sum(aman[e].get("counts") or [0])) for e in aids])
        APB = np.zeros((len(aids), M))
        for m in range(M):
            raw_a = HEAD[m].predict_proba(AX)[:, 1]
            for _b in sorted(set(adep)):
                _sel = adep == _b
                APB[_sel, m] = PLATT[m](raw_a[_sel], int(_b))
        plen = {}
        for i in range(a.apply_shards):
            for l in open(AD / f"{a.apply_variant}_shard{i}.jsonl"):
                if l.strip():
                    r_ = json.loads(l); plen[r_["problem_id"]] = len(r_["prompt"])
        E = Path(a.export_dir or a.apply_dir); E.mkdir(parents=True, exist_ok=True)
        with open(E / "history_preds.jsonl", "w") as f:
            for i, eid in enumerate(aids):
                f.write(json.dumps({"example_id": eid, "p": [float(x) for x in APB[i]],
                                    "prefill_usd": plen[eid] / a.chars_per_token
                                    * a.scout_usd_per_token}) + "\n")
        print(f"applied to {len(aids)} states -> {E / 'history_preds.jsonl'}")

    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    PC = np.zeros((N, M)); PD = np.zeros((N, M))
    cal_all = np.where(sp == "cal")[0]
    COEF = {"C": {}, "D": {}}
    for m in range(M):
        # 0/1 "this route has failed" rather than the count: the fitting examples have at most
        # one failure per route, and a linear term in the COUNT extrapolates to n = 5-6 in the
        # replay -- on TACO some fitted weights were positive, so beliefs ROSE with every failure
        # and the policy never stopped. Repeated failures are left to the count decay in PA.
        failed = (nfail > 0).astype(float)
        for name, P_, F in (("C", PC, np.c_[lg(PA[:, m]), failed]),
                            ("D", PD, np.c_[lg(PA[:, m]), failed, lg(PB[:, m])])):
            j = cal_all[np.isfinite(R[cal_all, m])]
            r = R[j, m]
            f = LogisticRegression(C=1e4, max_iter=5000).fit(
                np.vstack([F[j], F[j]]), np.r_[np.ones(len(j)), np.zeros(len(j))],
                sample_weight=np.r_[r, 1 - r])
            P_[:, m] = f.predict_proba(F)[:, 1]
            _c = f.coef_[0]
            COEF[name][slots[m]] = {"a": float(f.intercept_[0]), "b": float(_c[0]),
                                    "w": [float(x) for x in _c[1:1 + M]],
                                    "d": float(_c[1 + M]) if name == "D" else 0.0,
                                    "indicator": True}
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
    if a.export_dir:
        E = Path(a.export_dir); E.mkdir(parents=True, exist_ok=True)
        plen = {}
        for i in range(a.shards):
            for l in open(D / f"{a.variant}_shard{i}.jsonl"):
                if l.strip():
                    r_ = json.loads(l); plen[r_["problem_id"]] = len(r_["prompt"])
        with open(E / "content_preds_A.jsonl", "w") as f:
            for i in np.where(is0)[0]:
                f.write(json.dumps({"problem_id": pid[i],
                                    "p_successes": [float(x) for x in THETA[i]]}) + "\n")
        # --apply-dir writes the deep-state table into this same file; do not clobber it.
        with open(E / ("history_preds_fit.jsonl" if a.apply_dir else "history_preds.jsonl"), "w") as f:
            for i in range(N):
                if len(hist[i]) <= 1:
                    usd = plen[ids[i]] / a.chars_per_token * a.scout_usd_per_token
                    f.write(json.dumps({"example_id": ids[i], "p": [float(x) for x in PB[i]],
                                        "prefill_usd": usd}) + "\n")
        (E / "history_recal.json").write_text(json.dumps(COEF, indent=1))
        print(f"exported replay inputs -> {E}")
    if a.out:
        Path(a.out).write_text("\n".join(lines) + "\n")
        np.savez(Path(a.out).with_suffix(".npz"), ids=np.array(ids), PA=PA, PB=PB, PC=PC, PD=PD,
                 R=R, split=sp)


if __name__ == "__main__":
    main()
