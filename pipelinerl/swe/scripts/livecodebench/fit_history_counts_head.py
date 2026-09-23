#!/usr/bin/env python3
"""Belief head that reads the LATEST FAILURE semantically and the rest of the history numerically.

Why this split. Conditioning the PROMPT on the whole trajectory needs one prefill per reachable
trajectory: on the recollected pool that is ~2,321 states per problem, 791,606 in total, about 66
GPU-hours -- infeasible. But the history has two parts with very different costs:

  * what the failures LOOKED like -- semantic, needs the model to read code. Only the most recent
    attempt is affordable: one prefill per (problem, failed draw), ~34 per problem, ~1 GPU-hour.
  * how MANY there were and on which routes -- a short integer vector, free. It goes to the head
    as features, not to the language model as text.

So the probe reads the newest failed attempt, and the head sees the full failure counts. This also
removes the hand-tuned Beta-Bernoulli decay: the head LEARNS the response to repeated failure from
data instead of us choosing kappa, which is what the count decay was standing in for.

Training states are the reachable (counts, last-failure) pairs, and crucially the prefill is SHARED
by every state with the same last failure -- only the cheap count features vary -- so a large
training set costs nothing extra to extract.

Labels are leakage-free: for a state whose history is a set of failed draws, route m's label is
whether m succeeds among its draws NOT already consumed by that history.
"""
from __future__ import annotations
import argparse, itertools, json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


def load_acts(d: Path, tag: str, shards: int, readouts: str):
    ids, feats = [], []
    for i in range(shards):
        f = d / f"act_{tag}_shard{i}.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        ids += [str(x) for x in z["problem_ids"]]
        feats.append(np.concatenate([z[k].reshape(len(z[k]), -1)
                                     for k in readouts.split(",")], axis=1))
    if not ids:
        raise SystemExit(f"no activation shards found in {d} for tag {tag}")
    return ids, np.concatenate(feats).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts-dir", required=True)
    ap.add_argument("--variant", default="code")
    ap.add_argument("--act-tag", default="pv2code")
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--readouts", default="last")
    ap.add_argument("--tensors-dir", required=True)
    ap.add_argument("--pca", type=int, default=192)
    ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--max-states-per-problem", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scout-usd-per-token", type=float, default=0.278e-6)
    ap.add_argument("--chars-per-token", type=float, default=3.2)
    a = ap.parse_args()

    PD = Path(a.prompts_dir)
    ids, X = load_acts(PD, a.act_tag, a.shards, a.readouts)
    man = {r["example_id"]: r for r in
           (json.loads(l) for l in open(PD / f"{a.variant}_manifest.jsonl") if l.strip())}
    T = Path(a.tensors_dir)
    t = np.load(T / "tensors.npz", allow_pickle=True)
    ok = (t["final_outcome"] & t["valid"]).astype(bool); valid = t["valid"].astype(bool)
    slots = [str(s) for s in t["model_slots"]]; M = len(slots)
    pidx = {str(p): i for i, p in enumerate(t["problem_ids"])}
    sm = json.loads((T / "split_manifest.json").read_text())
    split = {**{str(p): "train" for p in sm["train_problem_ids"]},
             **{str(p): "cal" for p in sm["calibration_problem_ids"]},
             **{str(p): "test" for p in sm["test_problem_ids"]}}

    row_of = {e: i for i, e in enumerate(ids)}
    mu, sd = X.mean(0), X.std(0) + 1e-6
    tr_rows = [row_of[e] for e in ids if split.get(man[e]["problem_id"]) == "train"]
    pca = PCA(n_components=min(a.pca, len(tr_rows) - 1), random_state=0).fit((X[tr_rows] - mu) / sd)
    Z = pca.transform((X - mu) / sd).astype(np.float32)
    print(f"{len(ids)} prefills -> PCA {Z.shape[1]}; {len(tr_rows)} train prefills")

    # Reachable (counts, last-failure) states. The prefill is keyed by the last failure only.
    rng = np.random.default_rng(a.seed)
    rows = []          # (example_id, counts, split, labels[M], mask[M])
    for pid, pi in pidx.items():
        fails = {mi: [k for k in range(ok.shape[2]) if valid[pi, mi, k] and not ok[pi, mi, k]]
                 for mi in range(M)}
        cand = []
        for counts in itertools.product(*[range(len(fails[mi]) + 1) for mi in range(M)]):
            if sum(counts) == 0 or sum(counts) > 8:
                continue
            for r in range(M):
                if counts[r] >= 1:
                    cand.append((counts, r, fails[r][counts[r] - 1]))
        if len(cand) > a.max_states_per_problem:
            cand = [cand[i] for i in rng.choice(len(cand), a.max_states_per_problem, replace=False)]
        for counts, r, draw in cand:
            eid = f"{pid}||{slots[r]}{draw}"
            if eid not in row_of:
                continue
            used = {(mi, k) for mi in range(M) for k in fails[mi][:counts[mi]]}
            lab, msk = np.zeros(M), np.zeros(M, bool)
            for mi in range(M):
                rem = [k for k in range(ok.shape[2]) if valid[pi, mi, k] and (mi, k) not in used]
                if rem:
                    msk[mi] = True; lab[mi] = float(any(ok[pi, mi, k] for k in rem))
            rows.append((eid, np.array(counts, float), split.get(pid, "none"), lab, msk))
    print(f"{len(rows)} training states ({len(rows)/max(1,len(pidx)):.0f} per problem)")

    def feats(sel):
        Zi = Z[[row_of[e] for e, *_ in sel]]
        C = np.array([c for _, c, *_ in sel])
        last = np.zeros((len(sel), M))
        for i, (e, *_ ) in enumerate(sel):
            last[i, slots.index(e.split("||")[1].rstrip("0123456789"))] = 1
        return np.hstack([Zi, C, np.log1p(C), C.sum(1, keepdims=True), last])

    heads, cal = {}, {}
    for mi, s in enumerate(slots):
        tr = [r for r in rows if r[2] == "train" and r[4][mi]]
        ca = [r for r in rows if r[2] == "cal" and r[4][mi]]
        if len(tr) < 50:
            continue
        ytr = np.array([r[3][mi] for r in tr])
        clf = LogisticRegression(C=a.C, max_iter=3000).fit(feats(tr), ytr)
        heads[s] = clf
        if ca:
            p = clf.predict_proba(feats(ca))[:, 1]
            y = np.array([r[3][mi] for r in ca])
            o = np.argsort(p); rk = np.empty(len(p)); rk[o] = np.arange(1, len(p) + 1)
            n1, n0 = y.sum(), (1 - y).sum()
            auc = (rk[y > 0].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)
            lo = (p < 0.02).mean()
            print(f"  {s:<10} train {len(tr):>6}  cal AUC {auc:.3f}  base {y.mean()*100:>5.1f}%  "
                  f"beliefs below 2%: {lo*100:>5.2f}%")
            cal[s] = float(auc)

    # emit per-state beliefs for the replay, keyed as <pid>||<slot><draw>
    plen = {}
    for i in range(a.shards):
        f = PD / f"{a.variant}_shard{i}.jsonl"
        if f.exists():
            for l in open(f):
                if l.strip():
                    r = json.loads(l); plen[r["problem_id"]] = len(r["prompt"])
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    by_eid = {}
    for r in rows:
        by_eid.setdefault(r[0], r)
    sel = list(by_eid.values())
    F = feats(sel)
    P = np.full((len(sel), M), np.nan)
    for mi, s in enumerate(slots):
        if s in heads:
            P[:, mi] = heads[s].predict_proba(F)[:, 1]
    with open(out, "w") as fh:
        for i, (e, *_rest) in enumerate(sel):
            fh.write(json.dumps({"example_id": e,
                                 "p": [float(x) if np.isfinite(x) else 0.0 for x in P[i]],
                                 "prefill_usd": plen.get(e.split("||")[0], 3000)
                                 / a.chars_per_token * a.scout_usd_per_token}) + "\n")
    print(f"wrote {len(sel)} state beliefs -> {out}")


if __name__ == "__main__":
    main()
