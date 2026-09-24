#!/usr/bin/env python3
"""Judge head: P(this specific attempt is correct | problem, attempt), with set-aware features.

The verifier-free policy holds several attempts and must pick one. Scoring each in isolation
throws away the most useful signal available: whether the attempts AGREE. Two independently
sampled solutions converging is evidence both are right; three mutually disagreeing is evidence
the problem is hard and the next draw will not help either.

Doing that by re-prefilling on (problem + all held attempts) is combinatorial -- the same
explosion that made deep-history conditioning cost 66 GPU-hours. But the comparative signal is
already in the per-attempt embeddings, so it costs nothing extra:

    score(i | held set S) = head([ e_i, mean_{j!=i} e_j, max_j cos(e_i,e_j), mean_j cos(e_i,e_j),
                                   |S|, exact-code-match rate with S, route one-hot ])

permutation-invariant over S, linear in |S|. Reported as an ablation ladder -- independent judge,
+ agreement features, + set pooling -- so it is visible which part earns its keep.
"""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts-dir", required=True)
    ap.add_argument("--act-tag", default="judge")
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--readouts", default="last")
    ap.add_argument("--tensors-dir", required=True)
    ap.add_argument("--pca", type=int, default=128)
    ap.add_argument("--C", type=float, default=0.01)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    PD = Path(a.prompts_dir)
    ids, feats = [], []
    for i in range(a.shards):
        f = PD / f"act_{a.act_tag}_shard{i}.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        ids += [str(x) for x in z["problem_ids"]]
        feats.append(np.concatenate([z[k].reshape(len(z[k]), -1)
                                     for k in a.readouts.split(",")], axis=1))
    if not ids:
        raise SystemExit(f"no activation shards for tag {a.act_tag} in {PD}")
    X = np.concatenate(feats).astype(np.float32)
    man = {r["example_id"]: r for r in
           (json.loads(l) for l in open(PD / "judge_manifest.jsonl") if l.strip())}
    code = {}
    for i in range(a.shards):
        f = PD / f"judge_shard{i}.jsonl"
        if f.exists():
            for l in open(f):
                if l.strip():
                    r = json.loads(l)
                    seg = r["prompt"].split("```python")
                    code[r["problem_id"]] = hashlib.md5(
                        (seg[-1] if len(seg) > 1 else "").encode()).hexdigest()
    T = Path(a.tensors_dir)
    sm = json.loads((T / "split_manifest.json").read_text())
    grp = {**{str(p): 0 for p in sm["train_problem_ids"]},
           **{str(p): 1 for p in sm["calibration_problem_ids"]},
           **{str(p): 2 for p in sm["test_problem_ids"]}}
    slots = [str(s) for s in np.load(T / "tensors.npz", allow_pickle=True)["model_slots"]]
    row = {e: i for i, e in enumerate(ids)}
    tr_rows = [row[e] for e in ids if grp.get(man[e]["problem_id"], 3) == 0]
    mu, sd = X[tr_rows].mean(0), X[tr_rows].std(0) + 1e-6
    Z = PCA(n_components=min(a.pca, len(tr_rows) - 1), random_state=0)\
        .fit((X[tr_rows] - mu) / sd).transform((X - mu) / sd).astype(np.float32)
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    by_prob = {}
    for e in ids:
        by_prob.setdefault(man[e]["problem_id"], []).append(e)
    print(f"{len(ids)} judged attempts over {len(by_prob)} problems, PCA {Z.shape[1]}")

    def build(sel, mode):
        out = []
        for e in sel:
            i = row[e]; pid = man[e]["problem_id"]
            peers = [q for q in by_prob[pid] if q != e]
            f = [Z[i]]
            if mode != "independent":
                pj = [row[q] for q in peers]
                sims = Zn[pj] @ Zn[i] if pj else np.array([0.0])
                same = np.mean([code.get(q) == code.get(e) for q in peers]) if peers else 0.0
                f.append(np.array([sims.max(), sims.mean(), len(peers), same]))
            if mode == "set":
                f.append(Z[[row[q] for q in peers]].mean(0) if peers else np.zeros(Z.shape[1]))
            out.append(np.concatenate(f))
        return np.array(out)

    tr = [e for e in ids if grp.get(man[e]["problem_id"], 3) == 0]
    te = [e for e in ids if grp.get(man[e]["problem_id"], 3) == 2]
    ytr = np.array([man[e]["correct"] for e in tr], float)
    yte = np.array([man[e]["correct"] for e in te], float)
    print(f"  train {len(tr)} ({ytr.mean()*100:.1f}% correct) / test {len(te)} ({yte.mean()*100:.1f}%)")
    best = None
    for mode in ["independent", "agreement", "set"]:
        clf = LogisticRegression(C=a.C, max_iter=3000).fit(build(tr, mode), ytr)
        p = clf.predict_proba(build(te, mode))[:, 1]
        o = np.argsort(p); rk = np.empty(len(p)); rk[o] = np.arange(1, len(p) + 1)
        yb = yte > 0.5
        auc = (rk[yb].sum() - yb.sum() * (yb.sum() + 1) / 2) / (yb.sum() * (~yb).sum())
        print(f"  {mode:<12} test AUC {auc:.3f}   mean pred {p.mean()*100:5.1f}% vs actual "
              f"{yte.mean()*100:5.1f}%   ({p.mean()*100-yte.mean()*100:+.1f})")
        if best is None or auc > best[0]:
            best = (auc, mode, clf)
    _, mode, clf = best
    print(f"  -> using '{mode}'")
    allp = clf.predict_proba(build(ids, mode))[:, 1]
    with open(a.out, "w") as f:
        for e, p in zip(ids, allp):
            f.write(json.dumps({"example_id": e, "p_correct": float(p)}) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
