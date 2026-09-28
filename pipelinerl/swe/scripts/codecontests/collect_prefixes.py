#!/usr/bin/env python3
"""Partial-generation cost predictor, data: the first K output tokens of every route on every CodeContests problem.

Same system prompt, user prompt and sampling settings as the CodeContests pool (collect_cc_expert.py), but capped at
--max-tokens (default 512), keeping the REASONING prefix (the pool kept only its length). The label -- each
problem's typical total output length per route -- already exists in the pool, so no full generations are needed.
A prefix is a fresh draw, not the prefix of a labelled draw; it predicts the problem's EXPECTED length.
Output: <out>/<route>.jsonl rows {problem_id, route_label, reasoning, content, prompt_tokens, completion_tokens,
finish_reason, provider, error}. Resumable.
"""
from __future__ import annotations
import argparse, asyncio, json
from pathlib import Path
import aiohttp
from pipelinerl.swe.scripts.codecontests.collect_cc_expert import SYSTEM

ROUTES = {  # as in the CodeContests pool: (OpenRouter id, reasoning body, temperature, top_p)
    "oss20lo": ("openai/gpt-oss-20b", {"reasoning": {"effort": "low"}}, 1.0, 1.0),
    "oss20md": ("openai/gpt-oss-20b", {"reasoning": {"effort": "medium"}}, 1.0, 1.0),
    "dsv4f": ("deepseek/deepseek-v4-flash", {"reasoning": {"enabled": True}}, 0.7, 0.95),
    "oss120md": ("openai/gpt-oss-120b", {"reasoning": {"effort": "medium"}}, 1.0, 1.0),
    "oss120hi": ("openai/gpt-oss-120b", {"reasoning": {"effort": "high"}}, 1.0, 1.0),
}
IGNORE = ["Parasail", "AkashML"]


async def call(session, key, route, user, sem, max_tokens):
    model, extra, temp, top_p = ROUTES[route]
    body = {"model": model, "max_tokens": max_tokens, "temperature": temp, "top_p": top_p,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            "provider": {"ignore": IGNORE, "require_parameters": True}, **extra}
    err = None
    for attempt in range(4):
        try:
            async with sem, session.post("https://openrouter.ai/api/v1/chat/completions", json=body,
                                         headers={"Authorization": f"Bearer {key}"},
                                         timeout=aiohttp.ClientTimeout(total=600)) as r:
                r.raise_for_status(); d = await r.json()
            ch = d["choices"][0]; msg = ch["message"]; u = d.get("usage", {})
            return dict(reasoning=msg.get("reasoning") or "", content=msg.get("content") or "",
                        prompt_tokens=u.get("prompt_tokens", 0), completion_tokens=u.get("completion_tokens", 0),
                        finish_reason=ch.get("finish_reason"), provider=d.get("provider"), error=None)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:200]; await asyncio.sleep(5 * (attempt + 1))
    return dict(reasoning="", content="", prompt_tokens=0, completion_tokens=0, finish_reason="error", provider=None, error=err)


async def run(a):
    key = Path(a.api_key_file).read_text().strip(); sem = asyncio.Semaphore(a.concurrency)
    tasks = [json.loads(l) for l in open(a.tasks_file)]
    keep = {str(p) for p in json.load(open(a.problem_ids))} if a.problem_ids else None
    tasks = [t for t in tasks if keep is None or t["problem_id"] in keep]
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    async with aiohttp.ClientSession() as session:
        async def one_route(route):
            path = out / f"{route}.jsonl"
            done = {json.loads(l)["problem_id"] for l in open(path) if not json.loads(l).get("error")} if path.exists() else set()
            todo = [t for t in tasks if t["problem_id"] not in done]

            async def one(t):
                r = await call(session, key, route, t["prompt"], sem, a.max_tokens)
                r.update(problem_id=t["problem_id"], route_label=route); return r
            rows = await asyncio.gather(*[one(t) for t in todo])
            with open(path, "a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            ok = [r for r in rows if not r["error"]]
            print(f"{route}: {len(ok)}/{len(rows)} ok; mean out {sum(r['completion_tokens'] for r in ok)/max(len(ok),1):.0f} tok; "
                  f"reasoning text present {sum(bool(r['reasoning']) for r in ok)}; mean reasoning chars "
                  f"{sum(len(r['reasoning']) for r in ok)/max(len(ok),1):.0f}; finished early {sum(r['finish_reason'] == 'stop' for r in ok)}",
                  flush=True)
        await asyncio.gather(*[one_route(r) for r in a.routes.split(",")])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks-file", default="/mnt/llmd/results/exps/aristides/reason/cc_pool/cc_tasks.jsonl")
    ap.add_argument("--problem-ids", default="", help="JSON list restricting the problems (default: all tasks)")
    ap.add_argument("--out-dir", default="/mnt/llmd/results/exps/aristides/reason/cc_prefixes")
    ap.add_argument("--routes", default=",".join(ROUTES))
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--api-key-file", default="/home/toolkit/.secrets/openrouter_api_key")
    ap.add_argument("--concurrency", type=int, default=64)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
