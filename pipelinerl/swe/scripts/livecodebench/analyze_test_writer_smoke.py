"""Test-writer smoke-test analysis (NEW_PATH.md 2.7): the four go/no-go statistics.

1. Executability  - share of candidate-executions with a fully-running suite (per writer).
2. Discrimination - per writer, AUC of suite score separating correct/incorrect candidates,
                    stratified by difficulty; plus CodeT-style selection accuracy.
3. Crossover      - per-problem argmax test-writer; regret of the globally best fixed writer vs
                    the per-problem oracle, problem-level bootstrap CI.
4. Correlation    - per (writer, generator slot): alpha = P(suite fully passes | candidate wrong);
                    same-family vs cross-family alpha contrast.

Outputs: stdout summary + smoke_stats.json + smoke_report.md in <out-dir>/analysis/.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

FAMILY = {
    "qwen4b": "qwen",
    "oss20lo": "gpt-oss", "oss20md": "gpt-oss", "oss120md": "gpt-oss", "oss120hi": "gpt-oss",
    "dsv4f": "deepseek",
}
COST_PER_SUITE_C = {"oss20lo": 0.011, "oss20md": 0.042, "dsv4f": 0.158, "oss120md": 0.106}


def auc(scores, labels):
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    pairs = list(zip(scores, labels))
    order = sorted(range(len(pairs)), key=lambda k: pairs[k][0])
    rank = [0.0] * len(pairs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and pairs[order[j + 1]][0] == pairs[order[i]][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            rank[order[k]] = avg
        i = j + 1
    sum_pos = sum(rank[k] for k in range(len(pairs)) if pairs[k][1] == 1)
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def load_verdicts(verdicts_dir, writer):
    rows = []
    with open(verdicts_dir / ("verdicts_%s.jsonl" % writer)) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def full_pass(r):
    return r["suite_ok"] and r["n_ran"] > 0 and r["n_case_pass"] == r["n_ran"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--verdicts-dir", required=True)
    ap.add_argument("--out-dir", default=None, help="default: <verdicts-dir>/analysis")
    ap.add_argument("--writers", default="qwen4b,oss20lo,oss20md,dsv4f,oss120md")
    ap.add_argument("--min-cands", type=int, default=3)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pass-field", default="per_case", choices=["per_case", "per_case_loose"])
    args = ap.parse_args()
    writers = [w.strip() for w in args.writers.split(",")]
    out_dir = Path(args.out_dir or (Path(args.verdicts_dir) / "analysis"))
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    difficulty = {}
    with open(Path(args.pool_dir) / "problems.jsonl") as f:
        for line in f:
            p = json.loads(line)
            difficulty[p["problem_id"]] = p.get("difficulty")

    verdicts = {}
    for w in writers:
        rows = load_verdicts(Path(args.verdicts_dir), w)
        d = {}
        for r in rows:
            d[(r["problem_id"], r["slot"], r["draw_index"])] = r
        verdicts[w] = d

    report = []
    stats = {}

    def say(line=""):
        print(line)
        report.append(line)

    # ---------- STAT 1: executability ----------
    stat1 = {}
    for w in writers:
        rows = verdicts[w].values()
        n = len(rows)
        ok = sum(1 for r in rows if r["suite_ok"])
        errs = defaultdict(int)
        for r in rows:
            if not r["suite_ok"] and r["suite_err"]:
                errs[r["suite_err"]] += 1
        top = sorted(errs.items(), key=lambda kv: -kv[1])[:3]
        stat1[w] = {"n_rows": n, "frac_ok": round(ok / n, 4) if n else None, "top_err": dict(top)}
    stats["stat1"] = stat1
    say("## STAT 1 - executability")
    for w in writers:
        s = stat1[w]
        say("- %s: %.4f suite_ok over %d rows; top errors: %s" % (w, s["frac_ok"] or 0, s["n_rows"], s["top_err"]))
    say("")
    # ---------- STAT 2: discrimination ----------
    sc = {}  # writer -> key -> (score, truth, pid); fully-executed suites only
    for w in writers:
        d = {}
        for key, r in verdicts[w].items():
            if not r["suite_ok"] or r["n_ran"] == 0:
                continue
            d[(r["problem_id"], r["slot"], r["draw_index"])] = (
                r["n_case_pass"] / r["n_ran"], r["truth"], r["problem_id"])
        sc[w] = d

    stat2 = {}
    for w in writers:
        pts = list(sc[w].values())
        entry = {"n_cands": len(pts)}
        if pts:
            scores = [p[0] for p in pts]
            labels = [1 if p[1] else 0 for p in pts]
            a = auc(scores, labels)
            entry["auc"] = round(a, 4) if a is not None else None
            by_diff = defaultdict(list)
            for s, t, pid in pts:
                by_diff[difficulty.get(pid, "?")].append((s, t))
            for d in sorted(by_diff):
                xs = by_diff[d]
                a = auc([s for s, _ in xs], [1 if t else 0 for _, t in xs])
                entry["auc_" + d] = round(a, 4) if a is not None else None
                entry["n_" + d] = len(xs)
            by_pid = defaultdict(list)
            for s, t, pid in pts:
                by_pid[pid].append((s, t))
            sels = {}
            for pid, lst in by_pid.items():
                if len(lst) < args.min_cands:
                    continue
                lst = sorted(lst, key=lambda st: (-st[0], rng.random()))
                sels[pid] = 1 if lst[0][1] else 0
            entry["selection_acc"] = round(mean(sels.values()), 4) if sels else None
            entry["n_selection_problems"] = len(sels)
        stat2[w] = entry
    stats["stat2"] = stat2
    say("## STAT 2 - discrimination")
    for w in writers:
        e = stat2[w]
        diffs = " ".join("%s=%s(n=%d)" % (k.replace("auc_", ""), e.get(k), e.get("n_" + k.replace("auc_", ""), 0))
                         for k in sorted(e) if k.startswith("auc_"))
        say("- %s: AUC=%s selection_acc=%s over %d problems (n_cands=%d)" % (
            w, e.get("auc"), e.get("selection_acc"), e.get("n_selection_problems", 0), e["n_cands"]))
        say("    by difficulty: %s" % diffs)
    say("")
    # ---------- STAT 3: crossover ----------
    by_pid_w = {}
    for w in writers:
        d = defaultdict(list)
        for s, t, pid in sc[w].values():
            d[pid].append((s, t))
        by_pid_w[w] = d
    usable = []
    for pid in by_pid_w[writers[0]]:
        if all(len(by_pid_w[w].get(pid, [])) >= args.min_cands for w in writers):
            usable.append(pid)

    sel = {}
    for pid in usable:
        for w in writers:
            lst = sorted(by_pid_w[w][pid], key=lambda st: (-st[0], rng.random()))
            sel[(w, pid)] = 1 if lst[0][1] else 0

    stat3 = {"n_usable_problems": len(usable)}

    def fixed_of(w, pid_list):
        return mean(sel[(w, pid)] for pid in pid_list)

    def oracle_of(pid_list):
        return mean(max(sel[(w, pid)] for w in writers) for pid in pid_list)

    if usable:
        fixed_means = {}
        for w in writers:
            fixed_means[w] = fixed_of(w, usable)
        best_fixed = max(fixed_means, key=fixed_means.get)
        regs = []
        for _ in range(args.bootstrap):
            sample = [rng.choice(usable) for _ in usable]
            fb = max(writers, key=lambda w: fixed_of(w, sample))
            regs.append(oracle_of(sample) - fixed_of(w=fb, pid_list=sample))
        regs.sort()
        lo = regs[int(0.025 * args.bootstrap)]
        hi = regs[int(0.975 * args.bootstrap)]
        n_cross = 0
        for pid in usable:
            best_w = writers[0]
            best_val = -1.0
            for w in writers:
                v = sel[(w, pid)]
                if v > best_val:
                    best_val = v
                    best_w = w
            if best_w != best_fixed:
                n_cross += 1
        stat3 = {
            "n_usable_problems": len(usable),
            "fixed_means": {w: round(v, 4) for w, v in fixed_means.items()},
            "best_fixed": best_fixed,
            "oracle_mean": round(oracle_of(usable), 4),
            "regret_of_best_fixed": round(oracle_of(usable) - fixed_means[best_fixed], 4),
            "regret_ci": [round(lo, 4), round(hi, 4)],
            "crossover_frac": round(n_cross / len(usable), 4),
        }
    stats["stat3"] = stat3
    say("## STAT 3 - crossover")
    if usable:
        s3 = stat3
        for w in writers:
            say("- fixed %s: %.4f" % (w, s3["fixed_means"][w]))
        say("- best fixed: %s ; per-problem oracle: %s ; regret of best-fixed: %s (CI %s) ; crossover_frac=%s" % (
            s3["best_fixed"], s3["oracle_mean"], s3["regret_of_best_fixed"], s3["regret_ci"], s3["crossover_frac"]))
    else:
        say("- no problems usable (need >= min_cands scored candidates for every writer)")
    say("")
    # ---------- STAT 4: correlation ----------
    stat4 = {}
    for w in writers:
        fam_w = FAMILY.get(w)
        by_slot = defaultdict(list)
        for r in verdicts[w].values():
            if not r["suite_ok"] or r["n_ran"] == 0:
                continue
            by_slot[r["slot"]].append((full_pass(r), r["truth"]))
        pairs = {}
        for slot in sorted(by_slot):
            lst = by_slot[slot]
            wrong = [p for p, t in lst if not t]
            right = [p for p, t in lst if t]
            a = mean(wrong) if wrong else None
            b = 1 - mean(right) if right else None
            pairs_stat = {
                "alpha": round(a, 4) if a is not None else None,
                "n_wrong": len(wrong),
                "beta": round(b, 4) if b is not None else None,
                "n_right": len(right),
                "family": FAMILY.get(slot),
            }
            pairs[slot] = pairs_stat
        same = []
        cross = []
        for slot, v in pairs.items():
            if v["alpha"] is None:
                continue
            if v["family"] == fam_w:
                same.append(v["alpha"])
            else:
                cross.append(v["alpha"])
        stat4[w] = {
            "writer_family": fam_w,
            "pairs": pairs,
            "alpha_same_family": round(mean(same), 4) if same else None,
            "alpha_cross_family": round(mean(cross), 4) if cross else None,
        }
    stats["stat4"] = stat4
    say("## STAT 4 - alpha by (writer, generator slot): P(suite fully passes | candidate wrong)")
    for w in writers:
        s4 = stat4[w]
        say("- %s [%s same-fam vs %s cross-fam]" % (
            w, s4["alpha_same_family"], s4["alpha_cross_family"]))
        for slot, v in s4["pairs"].items():
            say("    %s (fam=%s): alpha=%s n_wrong=%d beta=%s n_right=%d" % (
                slot, v["family"], v["alpha"], v["n_wrong"], v["beta"], v["n_right"]))
    say("")

    with open(out_dir / "smoke_stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    with open(out_dir / "smoke_report.md", "w") as f:
        f.write("\n".join(report) + "\n")
    say("saved -> %s" % out_dir)


if __name__ == "__main__":
    main()
