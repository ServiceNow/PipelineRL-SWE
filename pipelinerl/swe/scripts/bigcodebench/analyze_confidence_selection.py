#!/usr/bin/env python3
"""Is a generator's own answer-token confidence a verifier-free selector? (BCB logprob pilot)

The 4B judge is near chance at picking among a STRONG model's attempts (BCB pick-1-of-4 on
deepseek-v4-flash: 52.8% judge vs 52.6% random vs 60.9% perfect), which is what capped judged
multi-sampling. The generator's own token confidence is free -- it comes back with the draw -- and
may scale with the generator rather than with a fixed 4B reader. Providers return logprobs for the
answer channel only, so every score here is computed on the answer (the code), never the reasoning.

Scores per attempt, all "higher = more confident":
  mean_lp      mean token logprob (length-normalised sequence likelihood)
  tok_conf     DeepConf token confidence, -(1/k) sum_j log p_j over the top-k, averaged
  low_group    DeepConf lowest group confidence: min over sliding windows of mean tok_conf
  tail_conf    tok_conf over the last 256 tokens
  min_lp       the single least likely token
  consilience  C_final - 2 * C_initial (first/last 20% after skipping 5%), per 2608.09898

Reported per rung: within-problem AUC (mixed problems), pick-1-of-k accuracy against random and
perfect selection, the share of failures where a correct attempt was held but lost ("selection
misses"), and across-problem AUC -- the stopping decision needs that one, the pick does not.
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score


def scores(lp: list) -> dict[str, float] | None:
    if not lp:
        return None
    tok = np.array([t[1] for t in lp if t[1] is not None], dtype=float)
    conf = np.array([-np.mean(t[2]) for t in lp if t[2]], dtype=float)
    if len(tok) < 5 or len(conf) < 5:
        return None
    w = min(32, len(conf))
    windows = np.convolve(conf, np.ones(w) / w, mode="valid")
    s = max(1, int(0.05 * len(conf)))
    body = conf[s:]
    k = max(1, int(0.2 * len(body)))
    return {"mean_lp": float(tok.mean()), "tok_conf": float(conf.mean()),
            "low_group": float(windows.min()), "tail_conf": float(conf[-256:].mean()),
            "min_lp": float(tok.min()),
            "consilience": float(body[-k:].mean() - 2.0 * body[:k].mean())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot-dir", required=True)
    ap.add_argument("--routes", default="oss20lo,oss20md,dsv4f,oss120md")
    ap.add_argument("--min-draws", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    D = Path(a.pilot_dir)
    for route in a.routes.split(","):
        by = defaultdict(list)
        n_rows = n_lp = 0
        providers = defaultdict(int)
        for f in sorted(D.glob(f"{route}_*_d*.jsonl")):
            for line in open(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("finish_reason") == "error":
                    continue
                n_rows += 1
                providers[r.get("provider") or "?"] += 1
                sc = scores(r.get("logprobs") or [])
                if sc is None:
                    continue
                n_lp += 1
                by[r["problem_id"]].append((bool(r["resolved"]), sc))
        probs = {p: v for p, v in by.items() if len(v) >= a.min_draws}
        if not probs:
            print(f"\n== {route}: no problems with >= {a.min_draws} scored draws yet ({n_rows} rows)")
            continue
        allrows = [x for v in probs.values() for x in v]
        y_all = np.array([x[0] for x in allrows])
        top = ", ".join(f"{k} {v}" for k, v in sorted(providers.items(), key=lambda kv: -kv[1])[:4])
        print(f"\n== {route}: {len(probs)} problems, {n_rows} draws ({n_lp} with logprobs), "
              f"pass@1 {y_all.mean()*100:.1f}%   providers: {top}")
        keys = list(allrows[0][1])
        print(f"   {'score':<12}{'withinAUC':>10}{'acrossAUC':>10}{'pick@2':>8}{'pick@4':>8}{'miss@4':>8}")
        base = {}
        for k in [2, 4]:
            pk, pa, pr = [], [], []
            for v in probs.values():
                for _ in range(20):
                    S = rng.choice(len(v), k, replace=False)
                    y = np.array([v[i][0] for i in S])
                    pa.append(y.max()); pr.append(y[0])
            base[k] = (np.mean(pa) * 100, np.mean(pr) * 100)
        for key in keys:
            within = []
            for v in probs.values():
                y = np.array([x[0] for x in v]); s = np.array([x[1][key] for x in v])
                if 0 < y.sum() < len(y):
                    within.append(roc_auc_score(y, s))
            across = roc_auc_score(y_all, [x[1][key] for x in allrows]) if 0 < y_all.sum() < len(y_all) else float("nan")
            pick = {}; miss = []
            for k in [2, 4]:
                acc = []
                for v in probs.values():
                    for _ in range(20):
                        S = rng.choice(len(v), k, replace=False)
                        y = np.array([v[i][0] for i in S]); s = np.array([v[i][1][key] for i in S])
                        acc.append(y[np.argmax(s)])
                        if k == 4 and y.any():
                            miss.append(not y[np.argmax(s)])
                pick[k] = np.mean(acc) * 100
            print(f"   {key:<12}{np.mean(within):>10.3f}{across:>10.3f}{pick[2]:>7.1f}%{pick[4]:>7.1f}%{np.mean(miss)*100:>7.1f}%")
        print(f"   {'random':<12}{0.5:>10.3f}{'':>10}{base[2][1]:>7.1f}%{base[4][1]:>7.1f}%")
        print(f"   {'perfect':<12}{1.0:>10.3f}{'':>10}{base[2][0]:>7.1f}%{base[4][0]:>7.1f}%{0.0:>7.1f}%")


if __name__ == "__main__":
    main()
