#!/usr/bin/env python3
"""SWE test-writer pilot, step 3: go/no-go statistics.

Selection uses only label-free information: a writer's script counts as VALID if it exits non-zero on
the unpatched repo (it reproduces something). Among the pool patches that applied, submit one that the
script passes (exit 0); ties are broken uniformly and scored exactly. Gold-patch results are reported
as a diagnostic (a valid script should pass gold) and never used for selection.

Reported:
  1. per writer: valid rate, gold-pass rate among valid scripts
  2. patch-selection accuracy: random pick / each writer / all-valid-writers vote / per-instance oracle
     writer (optimistic: chosen on the same patches) / perfect
  3. does writer quality track instance difficulty (#pool models resolving)?  SWT-bench says it
     should not; on LCB it did (NEW_PATH 2.8)
  4. self-family: false-accept rate of a writer's script on wrong patches from its OWN family vs others
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

FAMILY = {"oss20": "gpt-oss", "oss120": "gpt-oss", "qwen4b": "qwen", "qwen30": "qwen", "qcoder30": "qwen",
          "gemini": "gemini", "opus": "claude", "dsv4f": "deepseek", "devstral": "mistral"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exec", required=True)
    ap.add_argument("--patch-runs", required=True)
    ap.add_argument("--results-root", default="/mnt/llmd/results/exps/aristides/reason")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    runs = json.loads(Path(a.patch_runs).read_text())
    truth = {k: {json.loads(l)["instance_id"]: bool(json.loads(l)["resolved"]) for l in
                 open(Path(a.results_root) / f"opus_verified_daytona_eval_{r}/predictions/predictions_opus_verified.results.jsonl")}
             for k, r in runs.items()}
    rows = [json.loads(l) for l in open(a.exec)]
    rows = [r for r in rows if not r.get("error") and r["base"]]
    writers = sorted({w for r in rows for w in r["base"]})
    print(f"{len(rows)} instances executed; writers {writers}")

    # 1. validity
    print("\n1. validity (label-free: script fails on the unpatched repo) and gold agreement")
    for w in writers:
        base = [r["base"].get(w) for r in rows]
        valid = [c not in (0, None) for c in base]
        gold = [r["patches"].get("gold", {}).get(w) == 0 for r, v in zip(rows, valid) if v]
        print(f"   {w:<9} valid {np.mean(valid)*100:5.1f}%   passes gold | valid {np.mean(gold)*100 if gold else float('nan'):5.1f}%"
              f"   (timeouts on base: {sum(c == 124 for c in base)})")

    # 2. selection among pool patches
    def pick(r, passes):
        """expected accuracy of submitting a patch the rule accepts (uniform among accepted; if none, uniform among all)"""
        cands = [p for p in runs if r["apply"].get(p)]
        if not cands:
            return None
        acc = [p for p in cands if passes(p)]
        pool = acc if acc else cands
        return float(np.mean([truth[p][r["instance_id"]] for p in pool]))

    def writer_rule(w):
        def rule(r):
            if r["base"].get(w) in (0, None):
                return lambda p: True                      # invalid script: no information
            return lambda p: r["patches"].get(p, {}).get(w) == 0
        return rule

    def vote_rule(r):
        valid = [w for w in writers if r["base"].get(w) not in (0, None)]
        if not valid:
            return lambda p: True
        score = {p: sum(r["patches"].get(p, {}).get(w) == 0 for w in valid) for p in runs}
        top = max(score.values())
        return lambda p: score[p] == top

    res = defaultdict(list)
    for r in rows:
        iid = r["instance_id"]
        base_acc = pick(r, lambda p: True)
        if base_acc is None:
            continue
        res["random"].append(base_acc)
        res["perfect"].append(float(any(truth[p][iid] for p in runs if r["apply"].get(p))))
        per = {w: pick(r, writer_rule(w)(r)) for w in writers}
        for w in writers:
            res[w].append(per[w])
        res["vote (all valid writers)"].append(pick(r, vote_rule(r)))
        res["oracle writer (optimistic)"].append(max(per.values()))
    print(f"\n2. patch selection over {len(res['random'])} instances (pool patches that applied)")
    R = np.array(res["random"])
    for k in ["random", *writers, "vote (all valid writers)", "oracle writer (optimistic)", "perfect"]:
        x = np.array(res[k]); d = x - R
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        print(f"   {k:<28} {x.mean()*100:5.1f}%   vs random {d.mean()*100:+5.1f} [{np.percentile(bs,2.5)*100:+.1f},{np.percentile(bs,97.5)*100:+.1f}]")
    best_fixed = max(writers, key=lambda w: np.mean(res[w]))
    d = np.array(res["oracle writer (optimistic)"]) - np.array(res[best_fixed])
    print(f"   oracle - best fixed ({best_fixed}): {d.mean()*100:+.1f}   (optimistic: chosen on the scored patches)")

    # 3. writer quality vs instance difficulty
    print("\n3. does writer validity-with-gold-pass track difficulty (#pool models resolving)?  Spearman rho")
    diff = [sum(truth[p][r["instance_id"]] for p in runs) for r in rows]
    for w in writers:
        good = [float(r["base"].get(w) not in (0, None) and r["patches"].get("gold", {}).get(w) == 0) for r in rows]
        rho, pv = spearmanr(diff, good)
        print(f"   {w:<9} good-script rate {np.mean(good)*100:5.1f}%   rho vs #resolving {rho:+.2f} (p={pv:.2f})")
    G = np.array([[float(r["base"].get(w) not in (0, None) and r["patches"].get("gold", {}).get(w) == 0) for w in writers] for r in rows])
    if len(writers) > 1:
        c = np.corrcoef(G.T)
        print("   writer-writer correlation of 'good script' across instances (low = complementary):")
        for i, w in enumerate(writers):
            print("     " + w.ljust(9) + " ".join(f"{c[i, j]:+.2f}" for j in range(len(writers))))

    # 4. self-family false accepts on WRONG patches
    print("\n4. false-accept rate on wrong pool patches (valid scripts only): own family vs other families")
    for w in writers:
        own, other = [], []
        for r in rows:
            if r["base"].get(w) in (0, None):
                continue
            for p in runs:
                if r["apply"].get(p) and not truth[p][r["instance_id"]]:
                    fa = r["patches"].get(p, {}).get(w) == 0
                    (own if FAMILY.get(p) == FAMILY.get(w) else other).append(fa)
        print(f"   {w:<9} own-family {np.mean(own)*100 if own else float('nan'):5.1f}% (n={len(own)})   "
              f"other {np.mean(other)*100 if other else float('nan'):5.1f}% (n={len(other)})")


if __name__ == "__main__":
    main()
