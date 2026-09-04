#!/usr/bin/env python3
"""Does a pool member's OWN activations beat the cheap scout's at predicting its success?

This is the baseline the project has never been able to run. arXiv 2602.09924 routes with
`argmax_i (p_i(x) - lambda*c_i)` -- our myopic rule -- but states plainly that "this rule
requires training separate probes for each model in the pool", and it does not charge for
those forward passes. Our experts are API-served, so their hidden states were unreachable and
the comparison was never made: every cross-model number here came from probing the 4B scout.

Self-hosting the experts closes that. Extract pre-generation last-token states from EVERY
pool member, then fill the full transfer matrix

    A[i][j] = AUC( probe fit on model i's activations -> model j's success )

whose DIAGONAL is 2602.09924's method run faithfully and whose SCOUT ROW is ours. The gap
between them is the price of replacing K probes with one, and the prefill cost of the
diagonal is the price of the method it replaces.

Pre-registered reading, since either outcome is publishable and the framing must not follow
the result:
  - diagonal ~= scout row  -> per-model probes buy nothing, one cheap probe replaces the pool,
    and the shared-difficulty account of these representations is supported.
  - diagonal >> scout row  -> the signal is model-specific after all; the claim weakens to a
    cost/quality tradeoff and must be reported as one.

Phases split by environment on purpose. `extract` needs transformers>=4.56 for gpt-oss and
runs under vllm-env; `matrix` needs scikit-learn and runs under pipeline-rl. They communicate
through .npz files, so neither environment has to satisfy the other's pins.
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

# Identical text for scout and experts in collect_lcb_trajectories, so the only thing that
# differs between routes is the model's own chat template -- which is what we want, since
# the probe must see exactly what the model would see.
SYSTEM = (
    "You are an expert competitive programmer. "
    "Solve the problem and write a complete, correct Python solution. "
    "Output only Python code with no explanation."
)
LAYER_FRACTIONS = (0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def build_prompts(args) -> None:
    """Materialise the user prompts once, where lcb_runner is importable.

    The extraction phase runs under vllm-env, which has transformers>=4.56 for gpt-oss but
    not the LiveCodeBench runner. Rather than reconcile two dependency sets, the prompt text
    is frozen to a file here and the GPU job reads it -- so extraction needs only torch,
    transformers and numpy, and every route provably embeds byte-identical problem text.
    """
    from pipelinerl.swe.scripts.livecodebench.collect_lcb_trajectories import (
        load_lcb, make_prompt, problem_id,
    )
    if args.problems_file:
        # TACO problems are built into this script's record shape, so make_prompt applies
        # unchanged and the probe sees exactly the text the model was given.
        src = [json.loads(l) for l in open(args.problems_file) if l.strip()]
        rows = {}
        for r in src:
            rows[problem_id(r)] = r
            pid_raw = str(r.get("question_id") or r.get("problem_id") or "")
            if pid_raw:
                rows.setdefault(pid_raw, r)
    else:
        rows = {problem_id(r): r for r in load_lcb(min_date=args.min_date,
                                                   release_version=args.release_version)}
    keep = None
    if args.problem_ids_file:
        keep = {l.strip() for l in open(args.problem_ids_file) if l.strip()}
    pids = sorted(p for p in rows if keep is None or p in keep)
    missing = (keep - set(rows)) if keep else set()
    if missing:
        raise SystemExit(f"{len(missing)} requested problem ids absent from LCB: "
                         f"{sorted(missing)[:5]}")
    out = Path(args.prompts_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for pid in pids:
            f.write(json.dumps({"problem_id": pid, "prompt": make_prompt(rows[pid])}) + "\n")
    logger.info("wrote %d prompts -> %s", len(pids), out)


def extract(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    recs = [json.loads(l) for l in open(args.prompts_file) if l.strip()]
    if args.limit:
        recs = recs[: args.limit]
    pids = [r["problem_id"] for r in recs]
    prompts = {r["problem_id"]: r["prompt"] for r in recs}
    logger.info("%d problems to embed with %s", len(pids), args.model)

    tok = AutoTokenizer.from_pretrained(args.model)
    # device_map="auto" shards across whatever GPUs the job was given. gpt-oss ships MXFP4;
    # without triton_kernels transformers dequantizes to bf16, which is why the 120b job asks
    # for four cards rather than one.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", output_hidden_states=True)
    model.eval()
    n_layers = model.config.num_hidden_layers
    layers = sorted({int(round(f * n_layers)) for f in LAYER_FRACTIONS})
    logger.info("%d layers; probing %s; hidden %d", n_layers, layers, model.config.hidden_size)

    # FOUR readouts per layer, from ONE forward pass. The last prompt token is not comparable
    # across models: Qwen's chat template ends on a newline after "<|im_start|>assistant",
    # gpt-oss harmony ends on the word "assistant" mid-header. Reading position N-1 therefore
    # compares different things, and could by itself explain the cross-model result. The
    # content readouts cut the scaffolding off at the same SEMANTIC position in both models,
    # and the means remove position dependence altogether. The forward pass is the expensive
    # part; extra aggregations of hidden states we already computed are free.
    H = model.config.hidden_size
    reads = {k: np.zeros((len(pids), len(layers), H), dtype=np.float32)
             for k in ("last", "content_last", "mean", "content_mean")}
    n_no_offsets = 0
    with torch.no_grad():
        for i, pid in enumerate(pids):
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompts[pid]}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = tok(text, return_tensors="pt", truncation=True, max_length=args.max_len,
                      return_offsets_mapping=tok.is_fast)
            offsets = enc.pop("offset_mapping", None)
            ids = {k: v.to(model.device) for k, v in enc.items()}
            hs = model(**ids).hidden_states
            n_tok = ids["input_ids"].shape[1]
            # index of the last token belonging to the user's problem text
            c_idx = n_tok - 1
            if offsets is not None:
                end_char = text.rfind(prompts[pid])
                if end_char >= 0:
                    end_char += len(prompts[pid])
                    starts = offsets[0, :, 0].tolist()
                    cand = [k for k, a in enumerate(starts) if a < end_char]
                    if cand:
                        c_idx = max(cand)
            else:
                n_no_offsets += 1
            for j, L in enumerate(layers):
                h = hs[L][0]
                reads["last"][i, j] = h[-1, :].float().cpu().numpy()
                reads["content_last"][i, j] = h[c_idx, :].float().cpu().numpy()
                reads["mean"][i, j] = h.mean(0).float().cpu().numpy()
                reads["content_mean"][i, j] = h[: c_idx + 1].mean(0).float().cpu().numpy()
            if (i + 1) % 50 == 0:
                logger.info("%d/%d", i + 1, len(pids))
    if n_no_offsets:
        logger.warning("%d prompts had no offset mapping; content readouts fell back to the "
                       "last token for those", n_no_offsets)
    out = Path(args.activations)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, pre=reads["last"], problem_ids=np.array(pids),
                        layers=np.array(layers), model=args.model,
                        route_label=args.route_label, **reads)
    logger.info("wrote %s (readouts: %s)", out, sorted(reads))


def auc(scores, labels) -> float:
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    if y.all() or not y.any():
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def build_labels(tensors_dir: Path, audit_dir: str):
    t = np.load(tensors_dir / "tensors.npz", allow_pickle=True)
    pids = [str(p) for p in t["problem_ids"]]
    slots = [str(s) for s in t["model_slots"]]
    ok = t["final_outcome"] & t["valid"]
    labels = {slot: {p: bool(ok[i, mi, :].any()) for i, p in enumerate(pids)}
              for mi, slot in enumerate(slots)}
    labels["pool"] = {p: bool(ok[i].any()) for i, p in enumerate(pids)}
    n_flip = 0
    if audit_dir:
        # The 4096 cap manufactured "impossible" problems; probing them would fit a
        # collection artifact. Relabel before any probe is fit, exactly as the scout run did.
        for f in glob.glob(f"{audit_dir}/oss120_32k_*_d*.jsonl"):
            for line in open(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                pid = str(r["problem_id"])
                if r.get("resolved") and pid in labels["pool"] and not labels["pool"][pid]:
                    labels["pool"][pid] = True
                    if pid in labels.get("oss120", {}):
                        labels["oss120"][pid] = True
                    n_flip += 1
        logger.info("truncation audit relabelled %d problems solvable", n_flip)
    prob_split = {str(json.loads(l)["problem_id"]): json.loads(l).get("source_temporal_split")
                  for l in open(tensors_dir / "problems.jsonl") if l.strip()}
    return labels, prob_split, slots


def matrix(args) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    acts: dict[str, dict] = {}
    for spec in args.activations_file:
        label, _, path = spec.partition("=")
        d = np.load(path, allow_pickle=True)
        key = args.readout if args.readout in d.files else "pre"
        acts[label] = {"pids": [str(p) for p in d["problem_ids"]],
                       "pre": d[key], "layers": list(d["layers"])}
        logger.info("%s: %s, layers %s", label, acts[label]["pre"].shape, acts[label]["layers"])

    labels, prob_split, slots = build_labels(Path(args.tensors_dir), args.audit_dir)
    common = sorted(set.intersection(*[set(a["pids"]) for a in acts.values()])
                    & set(labels["pool"]) & set(prob_split))
    split = np.array([prob_split[p] for p in common])
    tr, te = split == "train", split == "eval"
    print(f"{len(common)} problems common to every route; train {tr.sum()} test {te.sum()}")

    scout_ok = np.array([labels["scout"][p] for p in common])
    targets = [s for s in slots] + ["pool"]

    def scores_for(src: str, tgt: str, layer_frac: float):
        a = acts[src]
        L = min(range(len(a["layers"])),
                key=lambda j: abs(a["layers"][j] / max(a["layers"]) - layer_frac))
        idx = {p: i for i, p in enumerate(a["pids"])}
        X = np.stack([a["pre"][idx[p], L] for p in common])
        y = np.array([labels[tgt][p] for p in common])
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=3000, C=args.C).fit(sc.transform(X[tr]), y[tr])
        return clf.predict_proba(sc.transform(X[te]))[:, 1], y

    for frac in [float(x) for x in args.layer_fracs.split(",")]:
        print(f"\n=== last-token pre-generation state at ~{frac:.0%} depth ===")
        print(f"{'activations from':<20}" + "".join(f"{'-> ' + t:>14}" for t in targets))
        for src in acts:
            row = f"{src:<20}"
            for tgt in targets:
                if tgt not in labels:
                    row += f"{'--':>14}"
                    continue
                s, y = scores_for(src, tgt, frac)
                mark = "*" if src == tgt else " "
                row += f"{auc(s, y[te]):13.4f}{mark}"
            print(row)
        print("* = own activations predicting own success: arXiv 2602.09924's method.")
        print(f"{'':<20}" + "  scout row = ours (one probe, already paid for by the protocol)")

        print(f"\n{'conditional on the scout having failed':<44}"
              + "".join(f"{'-> ' + t:>14}" for t in targets))
        m = te & ~scout_ok
        for src in acts:
            row = f"{src:<44}"
            for tgt in targets:
                if tgt not in labels:
                    row += f"{'--':>14}"
                    continue
                s, y = scores_for(src, tgt, frac)
                sub = s[(~scout_ok)[te]]
                row += f"{auc(sub, y[m]):14.4f}"
            print(row)
        print(f"(n = {int(m.sum())} test problems where a decision actually exists)")

    if args.cost_json:
        c = json.loads(Path(args.cost_json).read_text())
        print("\nprobe cost per problem, USD:")
        own = sum(c.get(s, 0.0) for s in slots if s in acts)
        print(f"  per-candidate probing (diagonal, one prefill per pool member): {own:.6f}")
        print(f"  scout-only probing (ours):                                    {c.get('scout',0.0):.6f}")
        if c.get("scout"):
            print(f"  ratio: {own / c['scout']:.1f}x")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["prompts", "extract", "matrix"], required=True)
    ap.add_argument("--model")
    ap.add_argument("--route-label", default="")
    ap.add_argument("--activations")
    ap.add_argument("--activations-file", action="append", metavar="LABEL=PATH")
    ap.add_argument("--problem-ids-file")
    ap.add_argument("--problems-file", default="",
                    help="Pre-built problems JSONL (TACO), bypassing the LiveCodeBench download.")
    ap.add_argument("--prompts-file")
    ap.add_argument("--tensors-dir")
    ap.add_argument("--audit-dir", default="")
    ap.add_argument("--cost-json", default="")
    ap.add_argument("--min-date", default="2023-09-01")
    ap.add_argument("--release-version", default="release_v6")
    ap.add_argument("--layer-fracs", default="0.5,1.0")
    ap.add_argument("--readout", default="pre",
                    choices=["pre", "last", "content_last", "mean", "content_mean"],
                    help="Which pooled readout to score. 'last' is the original last-prompt-"
                         "token; the others are the chat-template controls.")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--C", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.phase == "prompts":
        if not args.prompts_file:
            raise SystemExit("prompts needs --prompts-file")
        build_prompts(args)
    elif args.phase == "extract":
        if not (args.model and args.activations and args.prompts_file):
            raise SystemExit("extract needs --model, --activations and --prompts-file")
        extract(args)
    else:
        if not (args.activations_file and args.tensors_dir):
            raise SystemExit("matrix needs --activations-file and --tensors-dir")
        matrix(args)


if __name__ == "__main__":
    main()
