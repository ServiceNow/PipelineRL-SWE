#!/usr/bin/env python3
"""Fine-tune a small code encoder end-to-end to predict per-route success from a routing state.

Why this can win where the frozen version does not. Frozen jina-embeddings-v2-base-code is trained
for RETRIEVAL, and still lands within 0.01-0.02 AUC of a 20,480-d Qwen3-4B prefill on two of three
rungs (it loses badly only on gpt-oss-20b-low). So most of the gap is plausibly objective mismatch,
not capacity: nothing in contrastive retrieval training asks "will a 20B model solve this". At 137M
the encoder can be trained on that question directly, which a frozen 4B prefill cannot be cheaply --
and it is ~50x cheaper per state, which matters because the history arm pays one encode per
decision, not one per problem.

Trained on the same states, labels and split as the frozen comparison, so the delta is the training
objective and nothing else. Multi-task: one shared encoder, one head per route, masked where a
route has no remaining draws. Count features are concatenated at the head, exactly as in
fit_history_counts_head.py, so the encoder is only asked for the semantic part.
"""
from __future__ import annotations
import argparse, itertools, json, math, time
from pathlib import Path
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts-dir", required=True)
    ap.add_argument("--variant", default="code")
    ap.add_argument("--tensors-dir", required=True)
    ap.add_argument("--model", default="jinaai/jina-embeddings-v2-base-code")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--max-states-per-problem", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", choices=["rate", "any"], default="rate", help=(
        "rate: each route's success RATE over its remaining draws (soft BCE target) -- the per-draw "
        "belief the replay consumes. any: 'any remaining draw succeeds', the union label that made the "
        "4B history head 24-38pt overconfident; kept only to reproduce the old run."))
    ap.add_argument("--preds-out", default="", help=(
        "write history_preds.jsonl (replay --history-preds format) for every one-failed-draw state "
        "'pid||<slot><draw>': counts = that single failure, same states the 4B history probe covers"))
    ap.add_argument("--entry-states", action="store_true", help=(
        "also train on / predict the problem-only entry state 'pid||'"))
    ap.add_argument("--prefill-usd", type=float, default=4.6875e-05 / 29, help=(
        "per-state prefill charge; default scales the 4B probe's charge by the 4B/137M parameter ratio"))
    a = ap.parse_args()
    import torch, torch.nn as nn
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    PD = Path(a.prompts_dir); T = Path(a.tensors_dir)
    text = {}
    for f in sorted(PD.glob(f"{a.variant}_shard*.jsonl")):
        for line in open(f):
            if line.strip():
                r = json.loads(line); text[str(r["problem_id"])] = r["prompt"]
    t = np.load(T / "tensors.npz", allow_pickle=True)
    ok = (t["final_outcome"] & t["valid"]).astype(bool); valid = t["valid"].astype(bool)
    slots = [str(s) for s in t["model_slots"]]; M = len(slots)
    pidx = {str(p): i for i, p in enumerate(t["problem_ids"])}
    sm = json.loads((T / "split_manifest.json").read_text())
    grp = {**{str(p): 0 for p in sm["train_problem_ids"]},
           **{str(p): 1 for p in sm["calibration_problem_ids"]},
           **{str(p): 2 for p in sm["test_problem_ids"]}}
    rng = np.random.default_rng(a.seed); rows = []
    for pid, pi in pidx.items():
        fails = {m: [k for k in range(ok.shape[2]) if valid[pi, m, k] and not ok[pi, m, k]]
                 for m in range(M)}
        cand = []
        for counts in itertools.product(*[range(len(fails[m]) + 1) for m in range(M)]):
            if sum(counts) == 0 or sum(counts) > 8:
                continue
            for r in range(M):
                if counts[r] >= 1:
                    cand.append((counts, r, fails[r][counts[r] - 1]))
        if len(cand) > a.max_states_per_problem:
            cand = [cand[i] for i in rng.choice(len(cand), a.max_states_per_problem, replace=False)]
        if a.entry_states and f"{pid}||" in text:
            # the ENTRY state (problem only, no failures): lets the 137M make the first routing
            # decision too, instead of borrowing the 4B probe's entry belief
            lab, msk = np.zeros(M, np.float32), np.zeros(M, np.float32)
            for m in range(M):
                ks = [k for k in range(ok.shape[2]) if valid[pi, m, k]]
                if ks:
                    msk[m] = 1.0
                    lab[m] = (float(np.mean([ok[pi, m, k] for k in ks])) if a.label == "rate"
                              else float(any(ok[pi, m, k] for k in ks)))
            rows.append((f"{pid}||", np.zeros(M, np.float32), grp.get(pid, 3), lab, msk, -1))
        for counts, r, dr in cand:
            eid = f"{pid}||{slots[r]}{dr}"
            if eid not in text:
                continue
            used = {(m, k) for m in range(M) for k in fails[m][:counts[m]]}
            lab, msk = np.zeros(M, np.float32), np.zeros(M, np.float32)
            for m in range(M):
                rem = [k for k in range(ok.shape[2]) if valid[pi, m, k] and (m, k) not in used]
                if rem:
                    msk[m] = 1.0
                    lab[m] = (float(np.mean([ok[pi, m, k] for k in rem])) if a.label == "rate"
                              else float(any(ok[pi, m, k] for k in rem)))
            rows.append((eid, np.array(counts, np.float32), grp.get(pid, 3), lab, msk, r))
    tr = [r for r in rows if r[2] == 0]
    va = [r for r in rows if r[2] == 1]
    te = [r for r in rows if r[2] == 2]
    print(f"{len(rows)} states: {len(tr)} train / {len(va)} calibration / {len(te)} test", flush=True)
    if not tr or not va or not te:
        raise ValueError("train, calibration, and test splits must all contain routing states")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    enc = AutoModel.from_pretrained(a.model, trust_remote_code=True).to(dev)
    D = enc.config.hidden_size
    head = nn.Sequential(nn.Linear(D + 2 * M + 1 + M, 256), nn.ReLU(), nn.Linear(256, M)).to(dev)
    opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": a.lr},
                             {"params": head.parameters(), "lr": a.head_lr}])
    bce = nn.BCEWithLogitsLoss(reduction="none")

    def batch(sel):
        e = tok([text[r[0]] for r in sel], padding=True, truncation=True,
                max_length=a.max_len, return_tensors="pt").to(dev)
        C = torch.tensor(np.array([r[1] for r in sel]), device=dev)
        last = torch.zeros(len(sel), M, device=dev)
        for i, r in enumerate(sel):
            if r[5] >= 0:
                last[i, r[5]] = 1
        side = torch.cat([C, torch.log1p(C), C.sum(1, keepdim=True), last], 1)
        return e, side, (torch.tensor(np.array([r[3] for r in sel]), device=dev),
                         torch.tensor(np.array([r[4] for r in sel]), device=dev))

    def forward(e, side):
        h = enc(**e).last_hidden_state
        m = e["attention_mask"].unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return head(torch.cat([pooled, side], 1))

    def auc_score(p, y):
        y = np.asarray(y, dtype=bool)
        n_pos = int(y.sum()); n_neg = len(y) - n_pos
        if n_pos == 0 or n_neg == 0:
            return None
        order = np.argsort(p, kind="mergesort")
        sorted_p = np.asarray(p)[order]
        ranks = np.empty(len(p), dtype=np.float64)
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and sorted_p[j] == sorted_p[i]:
                j += 1
            ranks[order[i:j]] = ((i + 1) + j) / 2.0
            i = j
        return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    def evaluate(sel):
        enc.eval(); head.eval(); P, Y, MK = [], [], []
        with torch.no_grad():
            for i in range(0, len(sel), a.batch):
                e, side, (y, msk) = batch(sel[i:i + a.batch])
                P.append(torch.sigmoid(forward(e, side)).cpu().numpy())
                Y.append(y.cpu().numpy()); MK.append(msk.cpu().numpy())
        if not P:
            return {}, float("nan")
        P, Y, MK = np.concatenate(P), np.concatenate(Y), np.concatenate(MK)
        Pc = np.clip(P, 1e-6, 1 - 1e-6)
        evaluate.bce = float((-(Y * np.log(Pc) + (1 - Y) * np.log(1 - Pc)) * MK).sum() / max(MK.sum(), 1))
        scores = {}
        for m, slot in enumerate(slots):
            keep = MK[:, m] > 0
            score = auc_score(P[keep, m], Y[keep, m] >= 0.5)
            if score is not None:
                scores[slot] = score
        macro = float(np.mean(list(scores.values()))) if scores else float("nan")
        return scores, macro

    def show_scores(scores):
        return "  ".join(f"{s} {scores[s]:.3f}" if s in scores else f"{s} n/a" for s in slots)

    best_auc = -float("inf")
    best_epoch = None
    best_encoder = best_head = None
    steps_per_epoch = math.ceil(len(tr) / a.batch)
    for ep in range(a.epochs):
        enc.train(); head.train(); perm = rng.permutation(len(tr)); tot = 0.0
        epoch_start = time.perf_counter()
        for step, i in enumerate(range(0, len(perm), a.batch), start=1):
            sel = [tr[j] for j in perm[i:i + a.batch]]
            e, side, (y, msk) = batch(sel)
            loss = (bce(forward(e, side), y) * msk).sum() / msk.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
            if step == 1 or step % 25 == 0 or step == steps_per_epoch:
                elapsed = time.perf_counter() - epoch_start
                remaining = (steps_per_epoch - step) + (a.epochs - ep - 1) * steps_per_epoch
                eta = elapsed / step * remaining
                print(f"  epoch {ep + 1}/{a.epochs} step {step}/{steps_per_epoch} "
                      f"loss {tot / step:.4f} elapsed {elapsed / 60:.1f}m "
                      f"run_eta {eta / 60:.1f}m", flush=True)

        cal_scores, cal_auc = evaluate(va)
        epoch_elapsed = time.perf_counter() - epoch_start
        print(f"epoch {ep + 1}: train_bce {tot / max(1, steps_per_epoch):.4f}; "
              f"calibration macro AUC {cal_auc:.4f} ({show_scores(cal_scores)}); "
              f"elapsed {epoch_elapsed / 60:.1f}m", flush=True)
        # a soft per-draw target is a calibration problem: select on calibration BCE, not AUC
        crit = -evaluate.bce if a.label == "rate" else cal_auc
        print(f"  calibration soft BCE {evaluate.bce:.4f}", flush=True)
        if np.isfinite(crit) and crit > best_auc:
            best_auc, best_epoch = crit, ep + 1
            best_encoder = {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}
            best_head = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            print(f"  selected epoch {best_epoch} as current best on calibration", flush=True)

    if best_epoch is None:
        raise RuntimeError("calibration split did not produce a usable macro AUC")
    enc.load_state_dict(best_encoder); head.load_state_dict(best_head)
    test_scores, test_auc = evaluate(te)
    print(f"selected epoch {best_epoch}; final test macro AUC {test_auc:.4f} "
          f"({show_scores(test_scores)})", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_name": a.model,
        "encoder_state_dict": best_encoder,
        "head_state_dict": best_head,
        "slots": slots,
        "hidden_size": D,
        "max_len": a.max_len,
        "pooling": "attention_masked_mean",
        "best_epoch": best_epoch,
        "calibration_macro_auc": best_auc,
        "test_macro_auc": test_auc,
        "test_route_auc": test_scores,
        "train_args": vars(a),
    }, out)
    print("wrote full encoder and head checkpoint", out, flush=True)
    if a.preds_out:
        import re
        slot_re = re.compile("^(" + "|".join(sorted(map(re.escape, slots), key=len, reverse=True)) + r")(\d+)$")
        states = []
        if a.entry_states:
            states += [(f"{pid}||", np.zeros(M, np.float32), 3, np.zeros(M, np.float32),
                        np.zeros(M, np.float32), -1) for pid in pidx if f"{pid}||" in text]
        for eid in text:
            mt = slot_re.match(eid.partition("||")[2])
            if mt is None:
                continue
            r = slots.index(mt.group(1)); counts = np.zeros(M, np.float32); counts[r] = 1
            states.append((eid, counts, 3, np.zeros(M, np.float32), np.zeros(M, np.float32), r))
        enc.eval(); head.eval()
        with torch.no_grad(), open(a.preds_out, "w") as f:
            for i in range(0, len(states), a.batch):
                sel = states[i:i + a.batch]
                e, side, _ = batch(sel)
                for st, p in zip(sel, torch.sigmoid(forward(e, side)).cpu().numpy()):
                    f.write(json.dumps({"example_id": st[0], "p": [float(x) for x in p],
                                        "prefill_usd": a.prefill_usd}) + "\n")
        print(f"wrote {len(states)} history predictions to {a.preds_out}", flush=True)


if __name__ == "__main__":
    main()
