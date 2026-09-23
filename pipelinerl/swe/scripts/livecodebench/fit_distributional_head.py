#!/usr/bin/env python3
"""A belief head that emits a DISTRIBUTION over a route's per-draw success rate.

Why (PAPER_OUTLINE 3b-lxxxiii): a logistic head + Platt cannot reach the tail. Among states with
history only 0.2% of its beliefs fall below 2%, against 5.2% for the count decay, while the give-up
test at tight budgets fires below ~0.3-2%. So a read-the-history policy with no decay is +16% at
the 80-84% targets and -60% at 50%. A head that predicts P(q) rather than E[q] has tail resolution
by construction (mass at q~0 => belief ~0), distinguishes a fluke from a never, and gives the value
of n more draws in closed form: P(no success in n) = sum_g w_g (1-q_g)^n.

Model, per route m: spike-and-slab over q in [0,1]
    pi0 = sigmoid(u_m . z)                      P(this route never solves this problem)
    a   = softplus(v_m . z) + eps,  b = softplus(w_m . z) + eps      Beta(a, b) otherwise
    p(next draw succeeds) = (1 - pi0) * a / (a + b)
z is PCA of the same prefill features (fit on train only). Trained by the MARGINAL likelihood of
the draws NOT already in the state's history -- s successes, f failures:
    L = log[ pi0 * 1{s = 0} + (1 - pi0) * B(a + s, b + f) / B(a, b) ]
which rewards tail mass exactly when everything failed, and handles unequal draw counts. The
predicted distribution is then shrunk toward the pool-level one with a coefficient fitted on
calibration.

Writes history_preds.jsonl (mean belief per route, plus the distribution's parameters) for the
replay, for whichever activation set is passed with --apply-*.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from scipy.special import betaln
from sklearn.decomposition import PCA


def load_acts(d: Path, tag: str, shards: int, readouts: str):
    ids, feats = [], []
    for i in range(shards):
        z = np.load(d / f"act_{tag}_shard{i}.npz", allow_pickle=True)
        ids += [str(x) for x in z["problem_ids"]]
        feats.append(np.concatenate([z[k].reshape(len(z[k]), -1)
                                     for k in readouts.split(",")], axis=1))
    return ids, np.concatenate(feats).astype(np.float32)


def counts_from(manifest: dict, ok: np.ndarray, valid: np.ndarray, pi: int, m: int, slot: str):
    """Successes and failures among this route's draws that are NOT in the state's history."""
    used = {tuple(h) for h in (manifest.get("history") or [])}
    n_hist = manifest.get("counts", [0, 0, 0])
    keep = [k for k in range(ok.shape[2]) if valid[pi, m, k] and (slot, k) not in used]
    if used:
        s = sum(int(ok[pi, m, k]) for k in keep)
        return s, len(keep) - s
    # state manifests carry counts only: drop that many failed draws from this route
    fails = [k for k in range(ok.shape[2]) if valid[pi, m, k] and not ok[pi, m, k]]
    drop = set(fails[: int(n_hist[m])])
    keep = [k for k in range(ok.shape[2]) if valid[pi, m, k] and k not in drop]
    s = sum(int(ok[pi, m, k]) for k in keep)
    return s, len(keep) - s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--variant", default="deep")
    ap.add_argument("--act-tag", default="deepjudge")
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--readouts", default="last")
    ap.add_argument("--tensors-dir", required=True)
    ap.add_argument("--pca", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--apply-dir", default="")
    ap.add_argument("--apply-variant", default="state")
    ap.add_argument("--apply-act-tag", default="statejudge")
    ap.add_argument("--apply-shards", type=int, default=12)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--select-on-tail", action="store_true", default=True,
                    help="choose the shrinkage on all-fail states (the tail the stop rule reads) "
                         "rather than on average marginal likelihood")
    ap.add_argument("--select-on-bulk", dest="select_on_tail", action="store_false")
    ap.add_argument("--scout-usd-per-token", type=float, default=0.278e-6)
    ap.add_argument("--chars-per-token", type=float, default=3.2)
    a = ap.parse_args()
    D = Path(a.dir)

    ids, X = load_acts(D, a.act_tag, a.shards, a.readouts)
    man = {json.loads(l)["example_id"]: json.loads(l)
           for l in open(D / f"{a.variant}_manifest.jsonl") if l.strip()}
    t = np.load(Path(a.tensors_dir) / "tensors.npz", allow_pickle=True)
    ok = (t["final_outcome"] & t["valid"]).astype(bool); valid = t["valid"].astype(bool)
    tp = {str(p): i for i, p in enumerate(t["problem_ids"])}
    slots = [str(s) for s in t["model_slots"]]; M = len(slots)
    sm = json.loads((Path(a.tensors_dir) / "split_manifest.json").read_text())
    split = {**{str(p): "train" for p in sm["train_problem_ids"]},
             **{str(p): "cal" for p in sm["calibration_problem_ids"]},
             **{str(p): "test" for p in sm["test_problem_ids"]}}
    pid = [man[e]["problem_id"] for e in ids]
    sp = np.array([split.get(p, "none") for p in pid])
    SF = np.zeros((len(ids), M, 2))
    for i, e in enumerate(ids):
        for m, slot in enumerate(slots):
            SF[i, m] = counts_from(man[e], ok, valid, tp[pid[i]], m, slot)

    tr = np.where(sp == "train")[0]; ca = np.where(sp == "cal")[0]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    pca = PCA(n_components=a.pca, random_state=0).fit(((X[tr] - mu) / sd))
    Z = pca.transform((X - mu) / sd).astype(np.float32)
    print(f"{len(ids)} examples, PCA {a.pca} dims, {len(tr)} train / {len(ca)} cal")

    def fit_route(m: int):
        z = torch.tensor(Z[tr]); s = torch.tensor(SF[tr, m, 0]); f = torch.tensor(SF[tr, m, 1])
        W = torch.zeros(Z.shape[1] + 1, 3, requires_grad=True)
        with torch.no_grad():                      # start at the pool's own rates
            r = (SF[tr, m, 0].sum() + 1) / (SF[tr, m].sum() + 2)
            W[-1, 0] = -1.0; W[-1, 1] = np.log(np.exp(max(r, 1e-3) * 4) - 1)
            W[-1, 2] = np.log(np.exp(max(1 - r, 1e-3) * 4) - 1)
        opt = torch.optim.Adam([W], lr=a.lr)
        zb = torch.cat([z, torch.ones(len(z), 1)], 1)
        for ep in range(a.epochs):
            o = zb @ W
            pi0 = torch.sigmoid(o[:, 0])
            al = torch.nn.functional.softplus(o[:, 1]) + 1e-3
            be = torch.nn.functional.softplus(o[:, 2]) + 1e-3
            # log B(a+s, b+f) - log B(a, b)
            lb = (torch.lgamma(al + s) + torch.lgamma(be + f) - torch.lgamma(al + be + s + f)
                  - (torch.lgamma(al) + torch.lgamma(be) - torch.lgamma(al + be)))
            slab = torch.log(torch.clamp(1 - pi0, 1e-9)) + lb
            spike = torch.where(s > 0, torch.full_like(slab, -1e9), torch.log(torch.clamp(pi0, 1e-9)))
            ll = torch.logsumexp(torch.stack([spike, slab]), 0)
            loss = -ll.mean() + a.l2 * (W[:-1] ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
            if ep % 100 == 0:
                print(f"  route {slots[m]} epoch {ep:4d} nll {-ll.mean().item():.4f}")
        return W.detach()

    Ws = [fit_route(m) for m in range(M)]

    def predict(Zmat: np.ndarray):
        out = np.zeros((len(Zmat), M, 3))
        zb = torch.cat([torch.tensor(Zmat), torch.ones(len(Zmat), 1)], 1)
        for m in range(M):
            o = (zb @ Ws[m]).numpy()
            out[:, m, 0] = 1 / (1 + np.exp(-o[:, 0]))
            out[:, m, 1] = np.log1p(np.exp(o[:, 1])) + 1e-3
            out[:, m, 2] = np.log1p(np.exp(o[:, 2])) + 1e-3
        return out

    # shrink toward the pool-level distribution; coefficient chosen on calibration by marginal LL
    P = predict(Z)
    prior = np.array([[(SF[tr, m, 0].sum() + 1) / (SF[tr, m].sum() + 2)] for m in range(M)])
    def mean_of(Pm):
        return (1 - Pm[..., 0]) * Pm[..., 1] / (Pm[..., 1] + Pm[..., 2])
    def cal_ll(lam: float) -> float:
        tot = 0.0
        for m in range(M):
            pi0 = (1 - lam) * P[ca, m, 0]
            al = (1 - lam) * P[ca, m, 1] + lam * prior[m] * 4
            be = (1 - lam) * P[ca, m, 2] + lam * (1 - prior[m]) * 4
            s, f = SF[ca, m, 0], SF[ca, m, 1]
            lb = betaln(al + s, be + f) - betaln(al, be)
            ll = np.logaddexp(np.where(s > 0, -1e9, np.log(np.clip(pi0, 1e-9, 1))),
                              np.log(np.clip(1 - pi0, 1e-9, 1)) + lb)
            tot += ll.mean()
        return tot
    def cal_tail_ll(lam: float) -> float:
        """Marginal LL restricted to ALL-FAIL route-states -- the tail the give-up test reads.

        Average marginal likelihood is the wrong selector here: shrinking toward the pool prior
        raises it by improving the bulk while capping pi0 at (1-lam) of its fitted value, which is
        precisely the mass the stop rule needs (3b-lxxxiii). Scoring only the states where every
        remaining draw failed makes the criterion care about the thing the policy uses.
        """
        tot = 0.0
        for m in range(M):
            s_, f_ = SF[ca, m, 0], SF[ca, m, 1]
            sel = (s_ == 0) & (f_ > 0)
            if sel.sum() < 10:
                continue
            pi0 = (1 - lam) * P[ca, m, 0][sel]
            al = (1 - lam) * P[ca, m, 1][sel] + lam * prior[m] * 4
            be = (1 - lam) * P[ca, m, 2][sel] + lam * (1 - prior[m]) * 4
            lb = betaln(al, be + f_[sel]) - betaln(al, be)
            tot += np.logaddexp(np.log(np.clip(pi0, 1e-9, 1)),
                                np.log(np.clip(1 - pi0, 1e-9, 1)) + lb).mean()
        return tot

    grid = np.linspace(0, 0.9, 19)
    lam_bulk = max(grid, key=cal_ll)
    lam = max(grid, key=cal_tail_ll) if a.select_on_tail else lam_bulk
    print(f"shrinkage toward the pool prior: lambda = {lam:.2f} "
          f"({'all-fail tail LL' if a.select_on_tail else 'calibration marginal LL'}; "
          f"bulk criterion would pick {lam_bulk:.2f})")

    def shrink(Pm):
        Q = Pm.copy()
        for m in range(M):
            Q[:, m, 0] = (1 - lam) * Pm[:, m, 0]
            Q[:, m, 1] = (1 - lam) * Pm[:, m, 1] + lam * prior[m] * 4
            Q[:, m, 2] = (1 - lam) * Pm[:, m, 2] + lam * (1 - prior[m]) * 4
        return Q

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for name, (I, Zm) in {"fit": (ids, Z)}.items():
        Q = shrink(predict(Zm)); mean = mean_of(Q)
        np.savez(out / f"dist_{name}.npz", ids=np.array(I), params=Q, mean=mean,
                 split=sp if name == "fit" else None, sf=SF if name == "fit" else None)
    print("wrote", out / "dist_fit.npz")

    if a.apply_dir:
        AD = Path(a.apply_dir)
        aids, AX = load_acts(AD, a.apply_act_tag, a.apply_shards, a.readouts)
        AZ = pca.transform((AX - mu) / sd).astype(np.float32); del AX
        AQ = shrink(predict(AZ)); AM = mean_of(AQ)
        plen = {}
        for i in range(a.apply_shards):
            for l in open(AD / f"{a.apply_variant}_shard{i}.jsonl"):
                if l.strip():
                    r_ = json.loads(l); plen[r_["problem_id"]] = len(r_["prompt"])
        with open(out / "history_preds.jsonl", "w") as fh:
            for i, eid in enumerate(aids):
                fh.write(json.dumps({
                    "example_id": eid, "p": [float(x) for x in AM[i]],
                    "params": [[float(x) for x in AQ[i, m]] for m in range(M)],
                    "prefill_usd": plen[eid] / a.chars_per_token * a.scout_usd_per_token}) + "\n")
        np.savez(out / "dist_apply.npz", ids=np.array(aids), params=AQ, mean=AM)
        print(f"applied to {len(aids)} states -> {out / 'history_preds.jsonl'}")


if __name__ == "__main__":
    main()
