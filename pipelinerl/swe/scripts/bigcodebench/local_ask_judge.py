#!/usr/bin/env python3
"""Control for the strong-reader judge: ask Qwen3-4B the SAME zero-shot question, locally.

The dsv4f reader differs from our 4B judge in two ways at once -- a stronger reader, and asking
instead of a trained linear probe. Asking the 4B itself separates them: 4B-probe vs 4B-ask is
the method, 4B-ask vs dsv4f-ask is the reader. Qwen3-4B is not served with logprobs on
OpenRouter, so this is one prefill per attempt on a GPU, reading the next-token logits for
"Yes" and "No". Output rows match strong_reader_judge.py.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipelinerl.swe.scripts.bigcodebench.strong_reader_judge import messages


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tensors-dir", required=True)
    ap.add_argument("--tasks-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--split", default="test")
    ap.add_argument("--draws-per-route", default="oss20lo=8,oss20md=8,dsv4f=8,oss120md=6")
    ap.add_argument("--max-code-chars", type=int, default=12000)
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    tasks = {json.loads(l)["task_id"]: json.loads(l)["instruct_prompt"] for l in open(a.tasks_file)}
    test = set(json.loads((Path(a.tensors_dir) / "split_manifest.json").read_text())[f"{a.split}_problem_ids"])
    caps = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in a.draws_per_route.split(",")}
    rows = [r for r in map(json.loads, open(Path(a.tensors_dir) / "draw_records.jsonl"))
            if r["problem_id"] in test and r["model_slot"] in caps and r["draw_index"] < caps[r["model_slot"]]
            and (r.get("code") or "").strip()]
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id = tok.encode("No", add_special_tokens=False)[0]
    texts = [tok.apply_chat_template(messages(tasks[r["problem_id"]], r["code"], a.max_code_chars),
                                     tokenize=False, add_generation_prompt=True) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: len(texts[i]))   # length-bucketed batches
    out = open(a.out, "w")
    with torch.no_grad():
        for b in range(0, len(order), a.batch_size):
            idx = order[b:b + a.batch_size]
            enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True).to("cuda")
            logits = model(**enc).logits[:, -1, :].float()
            pair = torch.softmax(logits[:, [yes_id, no_id]], dim=-1)[:, 0].tolist()
            for i, p in zip(idx, pair):
                r = rows[i]
                out.write(json.dumps({"example_id": f"{r['problem_id']}||{r['model_slot']}{r['draw_index']}",
                                      "problem_id": r["problem_id"], "slot": r["model_slot"],
                                      "draw": r["draw_index"], "correct": bool(r["final_outcome"]),
                                      "p_yes": p}) + "\n")
            if b % (50 * a.batch_size) == 0:
                print(f"{b + len(idx)}/{len(rows)}", flush=True)
    out.close()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
