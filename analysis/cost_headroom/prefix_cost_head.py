"""Partial-generation cost predictor on CodeContests: does the first 512 tokens of reasoning predict how long a route
will run, where the 4B prefill (log-R2 0.18-0.43) could not?

Features of a reasoning prefix (text only): whether the route FINISHED inside the cap (then its length is known), the
prefix's token and char counts, hesitation markers ("wait", "hmm", "actually"), plan structure (enumerated steps),
difficulty words ("trivial", "tricky", "brute force", "dp", ...), complexity talk ("O(", "10^5"), and whether the
answer has started. Per route, ridge on TRAIN (target: log mean output tokens), same post-processing as the other
heads (smearing, train-level match), market prices. Written as cost_preds_<method>.jsonl + .overhead.json (the
prefix cost, charged in decompose.py):
  prefix_own    each route's OWN prefix + the 4B probe's prediction   (overhead: all 5 prefixes, no continuation credit)
  prefix_cheap  gpt-oss-20b-low's prefix only + probe, for every route  (overhead: that one prefix)
  prefix_alone  own prefix, no probe                                   (what the prefix knows by itself)
"""
import json, re, sys, numpy as np
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).parent))
from decompose import MK, R

WORDS = ["wait", "hmm", "actually", "but ", "however", "maybe", "not sure", "?", "let's", "we need", "observe", "note that",
         "case", "brute", "dp", "dynamic programming", "greedy", "binary search", "graph", "tree", "dfs", "bfs", "segment",
         "modulo", "prime", "o(", "10^5", "10^9", "n^2", "edge", "tricky", "hard", "complex", "simple", "easy", "trivial",
         "straightforward", "just ", "compute", "formula", "proof", "prove", "example", "sample", "```", "def ", "input()"]


def feats(r):
    t = (r.get("reasoning") or "") + "\n" + (r.get("content") or ""); lo = t.lower(); ntok = max(r.get("completion_tokens", 0), 1)
    return [float(r.get("finish_reason") == "stop"), np.log1p(ntok), np.log1p(len(t)), len(t) / ntok, float(bool(r.get("content"))),
            len(re.findall(r"^\s*(?:\d+[.)]|[-*])\s", t, flags=re.M)), t.count("\n")] + [lo.count(w) / ntok * 100 for w in WORDS]


def main():
    name = "cc_tensors"; P = R / "cc_prefixes"
    D = R / name; t = np.load(D / "tensors.npz", allow_pickle=True)
    S = [str(s) for s in t["model_slots"]]; pids = [str(p) for p in t["problem_ids"]]; pi = {p: i for i, p in enumerate(pids)}
    v = t["valid"].astype(bool); ct = t["completion_tokens"].astype(float); pt = t["prompt_tokens"].astype(float)
    n = v.sum(2); outm = np.where(n > 0, np.where(v, ct, 0).sum(2) / np.maximum(n, 1), np.nan)
    inp = np.nanmean(np.where(v, pt, np.nan), 2)
    sp = json.load(open(D / "split_manifest.json")); tr = np.array([pi[str(p)] for p in sp["train_problem_ids"]])
    te = np.array([pi[str(p)] for p in sp["test_problem_ids"]])
    pre = {s: {json.loads(l)["problem_id"]: json.loads(l) for l in open(P / f"{s}.jsonl") if not json.loads(l).get("error")} for s in S}
    F = {s: np.array([feats(pre[s].get(p, {})) for p in pids]) for s in S}
    have = {s: np.array([p in pre[s] for p in pids]) for s in S}
    pcost = {s: np.array([(pre[s][p]["prompt_tokens"] * MK[s][0] + pre[s][p]["completion_tokens"] * MK[s][1]) / 1e6 * 100
                          if p in pre[s] else 0.0 for p in pids]) for s in S}
    lcf = {json.loads(l)["problem_id"]: json.loads(l)["expected_costs"][:len(S)] for l in open(D / "cost_preds_market.jsonl")}
    LC = np.array([lcf[p] for p in pids])
    print(f"prefixes: " + ", ".join(f"{s} {have[s].sum()} (finished inside cap {F[s][have[s], 0].mean()*100:.0f}%, "
                                     f"mean cost {pcost[s].mean():.4f}c)" for s in S))
    rep = {}
    for meth in ("prefix_own", "prefix_cheap", "prefix_alone"):
        C = np.zeros((len(pids), len(S))); r2 = []
        for m, s in enumerate(S):
            pin, pout = MK[s][0] / 1e6, MK[s][1] / 1e6
            probe = np.log(np.maximum((LC[:, m] - np.nan_to_num(inp[:, m]) * pin) / pout, 1.0))
            src = "oss20lo" if meth == "prefix_cheap" else s
            X = F[src] if meth == "prefix_alone" else np.c_[F[src], probe]
            y = np.log(np.maximum(outm[:, m], 1.0)); a = np.isfinite(outm[:, m]) & have[src]; trm = tr[a[tr]]
            sc = StandardScaler().fit(X[trm]); Xs = sc.transform(X)
            yh = RidgeCV(alphas=np.geomspace(1e-2, 1e4, 13)).fit(Xs[trm], y[trm]).predict(Xs)
            yh = np.where(have[src], yh, probe)                         # no prefix -> fall back to the probe
            smear = np.mean(np.exp(y[trm] - yh[trm])); out_tok = np.exp(yh) * smear
            out_tok *= np.nanmean(outm[trm, m]) / out_tok[trm].mean()
            C[:, m] = np.nan_to_num(inp[:, m], nan=np.nanmean(inp[:, m])) * pin + out_tok * pout
            tt = te[a[te]]; r2.append(1 - ((y[tt] - np.log(out_tok[tt])) ** 2).sum() / ((y[tt] - y[tt].mean()) ** 2).sum())
        oh = sum(pcost.values()) if meth != "prefix_cheap" else pcost["oss20lo"]
        with open(D / f"cost_preds_{meth}.jsonl", "w") as f:
            for i, p in enumerate(pids):
                f.write(json.dumps({"problem_id": p, "expected_costs": [float(x) for x in C[i]]}) + "\n")
        json.dump({p: float(oh[i]) for i, p in enumerate(pids)}, open(D / f"cost_preds_{meth}.overhead.json", "w"))
        rep[meth] = dict(zip(S, map(float, r2)))
        print(f"{meth:<13} test log-output R2: " + "  ".join(f"{s} {x:+.2f}" for s, x in zip(S, r2)) + f"   overhead {oh.mean():.4f}c/problem")
    probe_r2 = []
    for m, s in enumerate(S):
        y = np.log(np.maximum(outm[:, m], 1.0)); tt = te[np.isfinite(outm[te, m])]
        yo = np.log(np.maximum((LC[:, m] - np.nan_to_num(inp[:, m]) * MK[s][0] / 1e6) / (MK[s][1] / 1e6), 1.0))
        probe_r2.append(1 - ((y[tt] - yo[tt]) ** 2).sum() / ((y[tt] - y[tt].mean()) ** 2).sum())
    print(f"{'probe (ref)':<13} test log-output R2: " + "  ".join(f"{s} {x:+.2f}" for s, x in zip(S, probe_r2)))
    json.dump(rep, open(D / "prefix_cost_heads_r2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
