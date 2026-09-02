#!/usr/bin/env python3
"""Can the scout's own activations tell us when to give up?

The abstention gate is the last live lever: oracle stopping is ~63% of headroom, but the
gate's response to predictor quality is sharply convex -- flat below AUC 0.85, +2.4pt at
0.90 -- and every affordable channel has landed at 0.73-0.76 conditional:

    trained `nothing` head                     0.7335
    post-scout abstention predictor            0.7629
    cheap observable features                  0.8591 uncond / 0.7491 conditional
    scout failure-mode taxonomy                0.8280 uncond / 0.7030 conditional
    token entropy / logprobs                   0.45-0.60 (at or below chance)
    verbalized self-assessment                 0.5700
    kNN retrieval over embeddings              0.6690

Activations are the one channel never tried here. "LLMs Encode Their Failures"
(arXiv 2602.09924) shows pre-generation activation probes predict policy-specific success
well enough to route a pool below the best single model's cost, so the mechanism is real --
but for *its own* success, on math, single-draw. The question here is different and harder:
whether a 4B model's internal state predicts POOL SOLVABILITY, i.e. whether anything in the
portfolio will solve the problem at any depth. That is the quantity abstention needs, and
because failures are ~89% shared across this pool it is largely a difficulty question, which
is exactly what activations plausibly encode better than surface text.

Two feature points, both cheap:
  PRE   hidden state at the last prompt token, before generating anything. One forward pass.
  POST  hidden state at the last token of the scout's completed attempt. The generation is
        already paid for under the mandatory-scout protocol.

Reported per layer, and both unconditionally and CONDITIONAL on the scout having failed --
the latter being the only set where an abstention decision exists. Every predictor in the
list above collapses toward 0.75 under that conditioning, and this project's own
methodological finding is that the unconditional number is inflated by cases carrying no
decision. The bar to beat is 0.90, not 0.76.

Label hygiene: the pool-solvability label is contaminated by the 4096-token cap -- the
truncation audit found 24.2% of "impossible" problems fall at 32k. If the audit output is
supplied, those problems are relabelled solvable before any probe is fit.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SCOUT_SYSTEM = (
    "You are an expert competitive programmer. "
    "Solve the problem and write a complete, correct Python solution. "
    "Output only Python code with no explanation."
)


def build_labels(tensors_dir: Path, audit_dir: str = ""):
    t = np.load(tensors_dir / "tensors.npz", allow_pickle=True)
    problems = {str(json.loads(l)["problem_id"]): json.loads(l)
                for l in open(tensors_dir / "problems.jsonl") if l.strip()}
    pids = [str(p) for p in t["problem_ids"]]
    ok = t["final_outcome"] & t["valid"]
    solvable = {p: bool(ok[i].any()) for i, p in enumerate(pids)}
    scout_failed = {p: bool(not ok[i, 0, :].any()) for i, p in enumerate(pids)}
    n_flipped = 0
    if audit_dir:
        for f in glob.glob(f"{audit_dir}/oss120_32k_*_d*.jsonl"):
            for line in open(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                pid = str(r["problem_id"])
                if r.get("resolved") and pid in solvable and not solvable[pid]:
                    solvable[pid] = True
                    n_flipped += 1
        logger.info("truncation audit relabelled %d problems as solvable", n_flipped)
    return pids, problems, solvable, scout_failed, n_flipped


def extract(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pipelinerl.swe.scripts.livecodebench.collect_lcb_trajectories import (
        load_lcb, make_prompt, problem_id,
    )

    rows = {problem_id(r): r for r in load_lcb(min_date=args.min_date,
                                               release_version=args.release_version)}
    scout_text: dict[str, str] = {}
    for pat in args.scout_draw or []:
        for f in glob.glob(pat):
            for line in open(f):
                if line.strip():
                    r = json.loads(line)
                    scout_text.setdefault(str(r["problem_id"]), r.get("full_output") or "")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", output_hidden_states=True)
    model.eval()
    n_layers = model.config.num_hidden_layers
    layers = sorted({int(round(f * n_layers)) for f in (0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)})
    logger.info("model has %d layers; probing %s", n_layers, layers)

    pids = sorted(set(rows) & (set(scout_text) if scout_text else set(rows)))
    if args.limit:
        pids = pids[: args.limit]
    pre = np.zeros((len(pids), len(layers), model.config.hidden_size), dtype=np.float32)
    post = np.zeros_like(pre)
    with torch.no_grad():
        for i, pid in enumerate(pids):
            msgs = [{"role": "system", "content": SCOUT_SYSTEM},
                    {"role": "user", "content": make_prompt(rows[pid])}]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for name, text, store in (("pre", prompt, pre),
                                      ("post", prompt + (scout_text.get(pid) or ""), post)):
                if name == "post" and not scout_text:
                    continue
                ids = tok(text, return_tensors="pt", truncation=True,
                          max_length=args.max_len).to("cuda")
                hs = model(**ids).hidden_states
                for j, L in enumerate(layers):
                    store[i, j] = hs[L][0, -1, :].float().cpu().numpy()
            if (i + 1) % 50 == 0:
                logger.info("%d/%d", i + 1, len(pids))
    out = Path(args.activations)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, pre=pre, post=post, problem_ids=np.array(pids),
                        layers=np.array(layers), has_post=bool(scout_text))
    logger.info("wrote %s", out)


def auc(scores, labels) -> float:
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    if y.all() or not y.any():
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def probe(args) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    d = np.load(args.activations, allow_pickle=True)
    pids = [str(p) for p in d["problem_ids"]]
    layers = list(d["layers"])
    tp, problems, solvable, scout_failed, n_flip = build_labels(Path(args.tensors_dir), args.audit_dir)
    keep = [i for i, p in enumerate(pids) if p in solvable]
    pids = [pids[i] for i in keep]
    split = np.array([problems[p].get("source_temporal_split") for p in pids])
    y = np.array([solvable[p] for p in pids])
    failed = np.array([scout_failed[p] for p in pids])
    tr, te = split == "train", split == "eval"
    print(f"{len(pids)} problems  train {tr.sum()} test {te.sum()}  "
          f"solvable {100*y.mean():.1f}%  (audit relabelled {n_flip})")
    print(f"test subset where the scout failed (decision is live): {int((te & failed).sum())}\n")
    print(f"{'features':<8} {'layer':>6} {'AUC uncond':>11} {'AUC | scout failed':>19}")
    best = {}
    for name in (["pre", "post"] if bool(d["has_post"]) else ["pre"]):
        X = d[name][keep]
        for j, L in enumerate(layers):
            sc = StandardScaler().fit(X[tr, j])
            clf = LogisticRegression(max_iter=2000, C=args.C).fit(sc.transform(X[tr, j]), y[tr])
            s = clf.predict_proba(sc.transform(X[te, j]))[:, 1]
            a_all = auc(s, y[te])
            m = failed[te]
            a_cond = auc(s[m], y[te][m]) if m.sum() > 10 else float("nan")
            print(f"{name:<8} {L:6d} {a_all:11.4f} {a_cond:19.4f}")
            if not np.isnan(a_cond) and a_cond > best.get(name, (0, 0))[1]:
                best[name] = (L, a_cond)
    print("\nbar to clear: 0.90 conditional (the gate is flat below 0.85).")
    for name, (L, a) in best.items():
        verdict = "CLEARS THE BAR" if a >= 0.90 else ("beats every prior channel" if a > 0.7629
                                                      else "no better than what exists")
        print(f"  best {name}: layer {L}, conditional AUC {a:.4f} -> {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["extract", "probe"], required=True)
    ap.add_argument("--activations", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--scout-draw", action="append", help="glob of scout draw files for POST features")
    ap.add_argument("--tensors-dir", default="")
    ap.add_argument("--audit-dir", default="", help="truncation-audit dir, to de-contaminate labels")
    ap.add_argument("--min-date", default="2023-09-01")
    ap.add_argument("--release-version", default="release_v6")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--C", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.phase == "extract":
        extract(args)
    else:
        if not args.tensors_dir:
            raise SystemExit("--probe needs --tensors-dir")
        probe(args)


if __name__ == "__main__":
    main()
