#!/usr/bin/env python3
"""Math pools for the cost-predictability rule (NEW_PATH 4.A.6): MATH-500 (levels 1-5) and Omni-MATH-500
(difficulty 1-10), the same five open reasoning routes as LCB / CodeContests / BCB.

One row per (problem, route, draw): {problem_id, dataset, route_label, model, draw, difficulty, resolved,
prompt_tokens, completion_tokens, finish_reason, provider, reasoning, content, error}. The REASONING TEXT is
kept (the earlier pools kept only its length), so partial-generation cost predictors can be tested offline.
Graded with math_verify against the reference answer (\\boxed{} extraction). Resumable per output file.
Output: <out>/<dataset>/<route>_d<k>.jsonl

--pilot N: a stratified sample of N problems per dataset (by level / rounded difficulty), seed 0.
"""
from __future__ import annotations
import argparse, asyncio, json, random
from collections import defaultdict
from pathlib import Path
import aiohttp

ROUTES = {  # label: (OpenRouter id, extra body, temperature, top_p) -- as in the CodeContests pool
    "oss20lo": ("openai/gpt-oss-20b", {"reasoning": {"effort": "low"}}, 1.0, 1.0),
    "oss20md": ("openai/gpt-oss-20b", {"reasoning": {"effort": "medium"}}, 1.0, 1.0),
    "dsv4f": ("deepseek/deepseek-v4-flash", {"reasoning": {"enabled": True}}, 0.7, 0.95),
    "oss120md": ("openai/gpt-oss-120b", {"reasoning": {"effort": "medium"}}, 1.0, 1.0),
    "oss120hi": ("openai/gpt-oss-120b", {"reasoning": {"effort": "high"}}, 1.0, 1.0),
}
IGNORE = ["Parasail", "AkashML"]
PROMPT = "Solve the following math problem. Reason step by step, then put your final answer within \\boxed{{}}.\n\n{problem}"


def load(name):
    from datasets import load_dataset
    if name == "math500":
        d = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return [{"problem_id": f"math500_{i}", "problem": r["problem"], "answer": r["answer"], "difficulty": float(r["level"]),
                 "subject": r["subject"]} for i, r in enumerate(d)]
    d = load_dataset("reliable-agents/Omni-MATH-500", split="test")
    return [{"problem_id": f"omni500_{i}", "problem": r["problem"], "answer": r["answer"], "difficulty": float(r["difficulty"]),
             "subject": str(r["domain"])[:120]} for i, r in enumerate(d)]


def stratified(tasks, n, seed=0):
    rng = random.Random(seed); by = defaultdict(list)
    for t in tasks:
        by[round(t["difficulty"])].append(t)
    keys = sorted(by); out = []
    while len(out) < n and any(by.values()):
        for k in keys:
            if by[k] and len(out) < n:
                out.append(by[k].pop(rng.randrange(len(by[k]))))
    return out


def grade(pred_text, gold):
    from math_verify import parse, verify
    try:
        g = parse(f"${gold}$") or parse(gold)
        p = parse(pred_text)
        return bool(p) and bool(g) and bool(verify(g, p))
    except Exception:
        return False


async def call(session, key, route, prompt, sem, max_tokens):
    model, extra, temp, top_p = ROUTES[route]
    body = {"model": model, "max_tokens": max_tokens, "temperature": temp, "top_p": top_p,
            "messages": [{"role": "user", "content": prompt}],
            "provider": {"ignore": IGNORE, "require_parameters": True}, **extra}
    err = None
    for attempt in range(4):
        try:
            async with sem, session.post("https://openrouter.ai/api/v1/chat/completions", json=body,
                                         headers={"Authorization": f"Bearer {key}"},
                                         timeout=aiohttp.ClientTimeout(total=3600)) as r:
                r.raise_for_status(); d = await r.json()
            ch = d["choices"][0]; msg = ch["message"]; u = d.get("usage", {})
            return dict(content=msg.get("content") or "", reasoning=msg.get("reasoning") or "",
                        prompt_tokens=u.get("prompt_tokens", 0), completion_tokens=u.get("completion_tokens", 0),
                        finish_reason=ch.get("finish_reason"), provider=d.get("provider"), error=None)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:200]; await asyncio.sleep(5 * (attempt + 1))
    return dict(content="", reasoning="", prompt_tokens=0, completion_tokens=0, finish_reason="error", provider=None, error=err)


async def run(a):
    key = Path(a.api_key_file).read_text().strip(); sem = asyncio.Semaphore(a.concurrency)
    draws = {kv.split(":")[0]: int(kv.split(":")[1]) for kv in a.routes.split(",")}
    async with aiohttp.ClientSession() as session:
        jobs = []
        for ds in a.datasets.split(","):
            tasks = load(ds)
            if a.pilot:
                tasks = stratified(tasks, a.pilot)
            out = Path(a.out_dir) / ds; out.mkdir(parents=True, exist_ok=True)
            (out / "problems.jsonl").write_text("".join(json.dumps({k: t[k] for k in ("problem_id", "difficulty", "subject", "answer")}
                                                                   | {"problem_statement": t["problem"]}) + "\n" for t in tasks))
            for route, k in draws.items():
                for d in range(k):
                    jobs.append((ds, tasks, route, d, out / f"{route}_d{d}.jsonl"))

        async def one_file(ds, tasks, route, d, path):
            done = set()
            if path.exists():
                done = {json.loads(l)["problem_id"] for l in open(path) if json.loads(l).get("finish_reason") != "error"}
            todo = [t for t in tasks if t["problem_id"] not in done]

            fh = open(path, "a")                  # each row is written the moment it returns: a killed job loses nothing

            async def one(t):
                r = await call(session, key, route, PROMPT.format(problem=t["problem"]), sem, a.max_tokens)
                r.update(problem_id=t["problem_id"], dataset=ds, route_label=route, model=ROUTES[route][0], draw=d,
                         difficulty=t["difficulty"], resolved=grade(r["content"], t["answer"]) if r["content"] else False)
                fh.write(json.dumps(r) + "\n"); fh.flush()
                return r
            rows = await asyncio.gather(*[one(t) for t in todo])
            fh.close()
            ok = [r for r in rows if r["finish_reason"] != "error"]
            print(f"{ds} {route} d{d}: {len(ok)}/{len(rows)} ok, acc {sum(r['resolved'] for r in ok)/max(len(ok),1):.2f}, "
                  f"mean out {sum(r['completion_tokens'] for r in ok)/max(len(ok),1):.0f} tok, "
                  f"capped {sum(r['finish_reason'] == 'length' for r in ok)}", flush=True)
        await asyncio.gather(*[one_file(*j) for j in jobs])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--datasets", default="math500,omni500")
    ap.add_argument("--routes", default="oss20lo:1,oss20md:1,dsv4f:1,oss120md:1,oss120hi:1", help="label:draws,...")
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--api-key-file", default="/home/toolkit/.secrets/openrouter_api_key")
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--max-tokens", type=int, default=64000)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
