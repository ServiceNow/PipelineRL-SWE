#!/usr/bin/env python3
"""Provider-routing pilot (NEW_PATH 4.A.39): deepseek-v4-flash PINNED to one provider at a time, so every problem has one draw
from every pinned provider. Same generation settings as the dsv4f route (temperature 0.7, top_p 0.95, reasoning enabled,
require_parameters), plus provider.only=[P] and allow_fallbacks=false. The other routes are reused from existing collections.
Datasets:
  mmlupro  2,000 problems sampled (seed 20261002) from the 6,500 fresh expansion tasks; graded like collect_expansion.py
  apps     all 1,000 APPS tasks (apps_tasks.jsonl); graded by executing the extracted code against the tests (cc_grader)
  cc       all 700 CodeContests tasks (cc_pool/cc_tasks.jsonl); graded like apps
Every row records provider, generation_id and the BILLED usage cost; a shared spend guard stops all workers.
Output: <out>/<dataset>/dsv4f_pin_<Provider>.jsonl (rows appended as they return; resumable).
Usage: python collect_provider_pilot.py --out DIR --providers StreamLake,GMICloud,DigitalOcean --budget-usd 18
"""
from __future__ import annotations
import argparse, asyncio, json, random, re, sys
from pathlib import Path
import aiohttp

sys.path.insert(0, str(Path(__file__).parent))
from collect_math_pool import PROMPT, _grade_task, IGNORE  # noqa: E402
from pipelinerl.swe.scripts.codecontests.cc_grader import grade as cc_grade  # noqa: E402
from pipelinerl.swe.scripts.livecodebench.collect_lcb_trajectories import extract_code  # noqa: E402
from pipelinerl.swe.scripts.codecontests.collect_cc_expert import SYSTEM as CC_SYSTEM  # noqa: E402

REPO = Path(__file__).resolve().parents[4]
R = Path("/mnt/llmd/results/exps/aristides/reason")


def tasks_for(ds, n_mmlu):
    if ds == "mmlupro":
        rows = [json.loads(l) for l in (REPO / "analysis/cost_headroom/expansion_20261001/mmlupro_tasks.jsonl").read_text().splitlines()]
        rows = sorted(rows, key=lambda t: t["problem_id"]); random.Random(20261002).shuffle(rows)
        return rows[:n_mmlu]
    if ds == "cc":                                    # CodeContests (700; second coding pool, NEW_PATH 4.A.55): same stdin/stdout grader as APPS
        return [json.loads(l) for l in (R / "cc_pool" / "cc_tasks.jsonl").read_text().splitlines()]
    return [json.loads(l) for l in (R / "apps_tasks.jsonl").read_text().splitlines()]


async def call(session, key, ds, task, provider, sem):
    if ds == "mmlupro":
        msgs = [{"role": "user", "content": task.get("prompt") or PROMPT.format(problem=task["problem"])}]; max_tokens = 64000
    else:
        msgs = [{"role": "system", "content": CC_SYSTEM}, {"role": "user", "content": task["prompt"]}]; max_tokens = 128000
    body = {"model": "deepseek/deepseek-v4-flash", "messages": msgs, "max_tokens": max_tokens, "temperature": 0.7, "top_p": 0.95,
            "reasoning": {"enabled": True},
            "provider": {"only": [provider], "allow_fallbacks": False, "require_parameters": True}}
    err = None
    for attempt in range(4):
        try:
            async with sem, session.post("https://openrouter.ai/api/v1/chat/completions", json=body,
                                         headers={"Authorization": f"Bearer {key}"}, timeout=aiohttp.ClientTimeout(total=3600)) as r:
                r.raise_for_status(); d = await r.json()
            ch = d["choices"][0]; msg = ch["message"]; u = d.get("usage", {})
            return dict(content=msg.get("content") or "", reasoning=msg.get("reasoning") or "", prompt_tokens=u.get("prompt_tokens", 0),
                        completion_tokens=u.get("completion_tokens", 0), finish_reason=ch.get("finish_reason"), provider=d.get("provider"),
                        generation_id=d.get("id"), usage_cost=u.get("cost"), error=None)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:200]; await asyncio.sleep(10 * (attempt + 1))
    return dict(content="", reasoning="", prompt_tokens=0, completion_tokens=0, finish_reason="error", provider=None, usage_cost=None, error=err)


async def run(a):
    key = Path(a.api_key_file).read_text().strip(); spent = {"usd": 0.0, "stop": False}
    jobs = []
    for ds in a.datasets.split(","):
        tasks = tasks_for(ds, a.n_mmlu); od = Path(a.out) / ds; od.mkdir(parents=True, exist_ok=True)
        (od / "problem_ids.json").write_text(json.dumps([t["problem_id"] for t in tasks]))
        for prov in a.providers.split(","):
            path = od / f"dsv4f_pin_{re.sub(r'[^A-Za-z0-9]', '', prov)}.jsonl"; done = set()
            if path.exists():
                for l in path.read_text().splitlines():
                    try: r = json.loads(l)
                    except json.JSONDecodeError: continue
                    spent["usd"] += float(r.get("usage_cost") or 0)
                    if r.get("finish_reason") != "error": done.add(r["problem_id"])
            jobs.append((ds, prov, path, [t for t in tasks if t["problem_id"] not in done]))
    print(json.dumps({"event": "start", "already_spent": spent["usd"], "todo": {f"{d}/{p}": len(t) for d, p, _, t in jobs}}), flush=True)
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as session:
        async def one_job(ds, prov, path, todo):
            sem = asyncio.Semaphore(a.concurrency); fh = open(path, "a"); n_ok = n_err = 0

            async def one(t):
                nonlocal n_ok, n_err
                if spent["stop"]:
                    return
                r = await call(session, key, ds, t, prov, sem)
                if ds == "mmlupro":
                    ok = _grade_task(t, r) if r["finish_reason"] != "error" else False
                else:
                    code = extract_code(r["content"]) or extract_code(r["reasoning"][-20000:]) if r["finish_reason"] != "error" else ""
                    g = await asyncio.to_thread(cc_grade, code, [tuple(x) for x in t["tests"]], t["timeout_s"]) if code else None
                    ok = bool(g and g.ok); r["code"] = code
                r.update(problem_id=t["problem_id"], dataset=ds, route_label=f"dsv4f_pin_{prov}", pinned_provider=prov, resolved=bool(ok))
                if ds in ("apps", "cc"):
                    r.pop("reasoning", None)                          # keep files small; reasoning length is completion_tokens
                spent["usd"] += float(r.get("usage_cost") or 0)
                if spent["usd"] > a.budget_usd:
                    spent["stop"] = True
                n_err += r["finish_reason"] == "error"; n_ok += r["finish_reason"] != "error"
                fh.write(json.dumps(r) + "\n"); fh.flush()
            await asyncio.gather(*[one(t) for t in todo]); fh.close()
            print(json.dumps({"event": "done", "dataset": ds, "provider": prov, "ok": n_ok, "errors": n_err, "spent_total": round(spent["usd"], 4)}), flush=True)
        await asyncio.gather(*[one_job(*j) for j in jobs])
    print(json.dumps({"event": "finished", "spent_usd": spent["usd"], "budget_stop": spent["stop"]}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True); ap.add_argument("--providers", default="StreamLake,GMICloud,DigitalOcean")
    ap.add_argument("--datasets", default="mmlupro,apps"); ap.add_argument("--n-mmlu", type=int, default=2000)
    ap.add_argument("--budget-usd", type=float, default=18.0); ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--api-key-file", default="/home/toolkit/.secrets/openrouter_api_key")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
