#!/usr/bin/env python3
"""Per-model SELF-ESTIMATED token budgets (TALE / SelfBudgeter style) for routing: each candidate model is asked, in a
short capped call, how many tokens it would generate to solve the problem at its route's reasoning effort. Model-specific
by construction -- the between-model cost differences the prompt-only probe cannot see.
Estimation calls run at LOW reasoning effort (cheap); the prompt names the route's own effort. Output per route:
<out>/<route>.jsonl rows {problem_id, route_label, estimate, content, prompt_tokens, completion_tokens, error}.
"""
from __future__ import annotations
import argparse, asyncio, json, random, re
from pathlib import Path
import aiohttp

ROUTES = {  # route: (model, estimation-call body, effort the ESTIMATE is for)
    "oss20lo": ("openai/gpt-oss-20b", {"reasoning": {"effort": "low"}}, "low"),
    "oss20md": ("openai/gpt-oss-20b", {"reasoning": {"effort": "low"}}, "medium"),
    "dsv4f": ("deepseek/deepseek-v4-flash", {"reasoning": {"enabled": True}}, ""),
    "oss120md": ("openai/gpt-oss-120b", {"reasoning": {"effort": "low"}}, "medium"),
    "oss120hi": ("openai/gpt-oss-120b", {"reasoning": {"effort": "low"}}, "high"),
}
ASK = ("\n\n---\nDo NOT solve the problem. Estimate how many tokens you would generate in total (all of your reasoning plus the "
       "final code) to solve it correctly{effort}. Reply with a single integer and nothing else.")


async def call(session, key, route, system, user, sem):
    model, extra, effort = ROUTES[route]
    body = {"model": model, "max_tokens": 1500, "temperature": 0.7,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user + ASK.format(effort=f" when reasoning with {effort} reasoning effort" if effort else "")}],
            "provider": {"ignore": ["Parasail", "AkashML"], "require_parameters": True}, **extra}
    for attempt in range(4):
        try:
            async with sem, session.post("https://openrouter.ai/api/v1/chat/completions", json=body,
                                         headers={"Authorization": f"Bearer {key}"}, timeout=aiohttp.ClientTimeout(total=300)) as r:
                r.raise_for_status(); d = await r.json()
            msg = d["choices"][0]["message"]; txt = (msg.get("content") or "").strip()
            nums = [int(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", txt)]
            u = d.get("usage", {})
            return dict(estimate=nums[-1] if nums else None, content=txt[:200], prompt_tokens=u.get("prompt_tokens", 0),
                        completion_tokens=u.get("completion_tokens", 0), error=None)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:200]; await asyncio.sleep(3 * (attempt + 1))
    return dict(estimate=None, content="", prompt_tokens=0, completion_tokens=0, error=err)


async def run(a):
    key = Path(a.api_key_file).read_text().strip(); sem = asyncio.Semaphore(a.concurrency)
    prompts = {json.loads(l)["problem_id"]: json.loads(l)["prompt"] for l in open(a.prompts)}
    ids = sorted(prompts); random.Random(0).shuffle(ids); ids = ids[: a.n] if a.n else ids
    system = Path(a.system_prompt_file).read_text() if a.system_prompt_file else ""
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as session:
        async def one_route(route):
            path = out / f"{route}.jsonl"
            done = {json.loads(l)["problem_id"] for l in open(path) if not json.loads(l).get("error")} if path.exists() else set()
            fh = open(path, "a")

            async def one(pid):
                r = await call(session, key, route, system, prompts[pid], sem); r.update(problem_id=pid, route_label=route)
                fh.write(json.dumps(r) + "\n"); fh.flush(); return r
            rows = await asyncio.gather(*[one(p) for p in ids if p not in done]); fh.close()
            ok = [r for r in rows if r["estimate"] is not None]
            print(f"{route}: {len(ok)}/{len(rows)} parsed; median estimate {sorted(r['estimate'] for r in ok)[len(ok)//2] if ok else None}", flush=True)
        await asyncio.gather(*[one_route(r) for r in ROUTES])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--system-prompt-file", default=""); ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--api-key-file", default="/home/toolkit/.secrets/openrouter_api_key"); ap.add_argument("--concurrency", type=int, default=128)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
