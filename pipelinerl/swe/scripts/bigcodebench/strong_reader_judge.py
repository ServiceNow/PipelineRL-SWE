#!/usr/bin/env python3
"""A strong cheap model as a READ-ONLY judge: P(Yes) from one output token, reasoning off.

A judge call is almost all input (problem + code); with max_tokens=1 its price is set by the
reader's INPUT price, not its size. deepseek-v4-flash reads at $0.055/M -- about what we already
charge the Qwen3-4B judge ($0.05/M) -- so this is a much stronger reader at the same price, and
nothing is executed, so it cannot collapse back into failure counting.

Scores existing pool draws (draw_records.jsonl), so it compares against the 4B judge on exactly
the same attempts. Writes one row per attempt: example_id "<pid>||<slot><draw>", p_yes, usage.
"""
from __future__ import annotations
import argparse, asyncio, json, math
from pathlib import Path
import aiohttp

SYSTEM = ("You are an expert Python code reviewer. You will see a programming task and a candidate "
          "solution. Decide whether the solution is fully correct: it would pass a thorough hidden "
          "unit-test suite for the task. Answer with exactly one word: Yes or No.")


def messages(task: str, code: str, max_code_chars: int) -> list[dict]:
    """The exact question every reader is asked -- shared so reader comparisons differ only in the reader."""
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Task:\n{task}\n\nCandidate solution:\n```python\n"
                                        f"{code[:max_code_chars]}\n```\n\nIs this solution correct?"}]


def p_yes(top: list[dict]) -> float | None:
    """Normalised P(Yes) over the Yes/No mass in the first token's top logprobs."""
    yes = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower() == "yes")
    no = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower() == "no")
    return yes / (yes + no) if yes + no > 0 else None


async def score(session, sem, key, a, row, task) -> dict:
    body = {"model": a.model, "max_tokens": 1, "temperature": 0.0, "logprobs": True, "top_logprobs": 5,
            "reasoning": {"enabled": False}, "provider": {"require_parameters": True},
            "messages": messages(task, row["code"], a.max_code_chars)}
    out = {"example_id": f"{row['problem_id']}||{row['model_slot']}{row['draw_index']}",
           "problem_id": row["problem_id"], "slot": row["model_slot"], "draw": row["draw_index"],
           "correct": bool(row["final_outcome"])}
    for attempt in range(4):
        try:
            async with sem, session.post("https://openrouter.ai/api/v1/chat/completions", json=body,
                                         headers={"Authorization": f"Bearer {key}"},
                                         timeout=aiohttp.ClientTimeout(total=120)) as r:
                r.raise_for_status()
                d = await r.json()
            ch = d["choices"][0]
            content = ((ch.get("logprobs") or {}).get("content") or [])
            out.update(p_yes=p_yes(content[0]["top_logprobs"]) if content else None,
                       answer=ch["message"].get("content"), provider=d.get("provider"),
                       prompt_tokens=d.get("usage", {}).get("prompt_tokens"),
                       completion_tokens=d.get("usage", {}).get("completion_tokens"))
            return out
        except Exception as e:                     # transport/provider error: retry, then record
            out["error"] = f"{type(e).__name__}: {e}"[:200]
            await asyncio.sleep(2 * (attempt + 1))
    return out


async def run(a) -> None:
    key = Path(a.api_key_file).read_text().strip()
    tasks = {json.loads(l)["task_id"]: json.loads(l)["instruct_prompt"] for l in open(a.tasks_file)}
    test = set(json.loads((Path(a.tensors_dir) / "split_manifest.json").read_text())[f"{a.split}_problem_ids"])
    caps = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in a.draws_per_route.split(",")}
    rows = [r for r in map(json.loads, open(Path(a.tensors_dir) / "draw_records.jsonl"))
            if r["problem_id"] in test and r["model_slot"] in caps and r["draw_index"] < caps[r["model_slot"]]
            and (r.get("code") or "").strip()]
    if a.limit:
        rows = rows[:a.limit]
    out = Path(a.out)
    done = {json.loads(l)["example_id"] for l in open(out)} if out.exists() else set()
    rows = [r for r in rows if f"{r['problem_id']}||{r['model_slot']}{r['draw_index']}" not in done]
    print(f"{len(rows)} attempts to score ({len(done)} already done)", flush=True)
    sem = asyncio.Semaphore(a.concurrency)
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(rows), 200):
            res = await asyncio.gather(*[score(session, sem, key, a, r, tasks[r["problem_id"]])
                                         for r in rows[i:i + 200]])
            with open(out, "a") as f:
                for x in res:
                    f.write(json.dumps(x) + "\n")
            ok = [x for x in res if x.get("p_yes") is not None]
            print(f"{i + len(res)}/{len(rows)} scored, {len(ok)} with P(yes)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tensors-dir", required=True)
    ap.add_argument("--tasks-file", required=True)
    ap.add_argument("--api-key-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="deepseek/deepseek-v4-flash")
    ap.add_argument("--split", default="test")
    ap.add_argument("--draws-per-route", default="oss20lo=8,oss20md=8,dsv4f=8,oss120md=6")
    ap.add_argument("--max-code-chars", type=int, default=12000)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
