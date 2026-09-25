#!/usr/bin/env python3
"""SWE test-writer pilot, step 1: each writer model writes ONE reproduction script per instance.

Script mode (as in SWT-bench's reproduction-script track): a standalone `python repro.py` run from the
repo root that exits 1 while the issue is present and 0 once it is fixed. Unlike competitive
programming, writing it needs an understanding of the bug, not a fix -- SWT-bench found test
generation and repair success statistically independent per instance -- which is the whole premise.

Context given to every writer (identical across writers, so writer comparisons are fair): the issue
text plus the pre-fix contents of the files the gold patch touches. That is ORACLE localisation -- a
pilot simplification that removes retrieval as a confound; it reveals where the bug is, not the fix.
File contents are fetched from GitHub at the instance's base commit.

Output: <out-dir>/scripts_<writer>.jsonl, one row per instance, resumable.
"""
from __future__ import annotations
import argparse, asyncio, json, random, re, urllib.request
from pathlib import Path
import aiohttp

WRITERS = {  # label: (OpenRouter id, extra body) -- ids verified against /api/v1/models 2026-09-25
    "oss20": ("openai/gpt-oss-20b", {"reasoning": {"effort": "medium"}}),
    "oss120": ("openai/gpt-oss-120b", {"reasoning": {"effort": "medium"}}),
    "dsv4f": ("deepseek/deepseek-v4-flash", {"reasoning": {"enabled": True}}),
    "qcoder30": ("qwen/qwen3-coder-30b-a3b-instruct", {}),
    "devstral": ("mistralai/devstral-2512", {}),
}
IGNORE = ["Parasail", "AkashML"]  # gpt-oss tool_calls serving artifact (PAPER_OUTLINE 3b-lxxxv)

PROMPT = """You are writing a bug-reproduction script for the repository {repo}.

<issue>
{issue}
</issue>

Relevant source files, before any fix:
{files}

Write ONE standalone Python script that reproduces this issue. It will be run from the repository
root (the package is installed in development mode) as `python repro.py`.
- Exit with status 1 if the issue is PRESENT (the buggy behaviour described in the issue occurs).
- Exit with status 0 if the issue is FIXED (the correct behaviour occurs).
- Do not modify any repository files. Configure anything the library needs inline (for Django, call
  django.conf.settings.configure(...) and django.setup() before importing models).
- Print a one-line diagnostic of what you observed.
Return only the script, in a single ```python code block."""


def fetch(repo: str, commit: str, path: str, max_chars: int) -> str:
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
    try:
        txt = urllib.request.urlopen(url, timeout=30).read().decode("utf-8", "replace")
    except Exception as e:
        return f"<could not fetch {path}: {type(e).__name__}>"
    return txt if len(txt) <= max_chars else txt[:max_chars] + "\n# ... (truncated)"


def gold_files(patch: str) -> list[str]:
    return sorted(set(re.findall(r"^diff --git a/(\S+) b/", patch, flags=re.M)))


def extract(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text or "", flags=re.S)
    return max(blocks, key=len).strip() if blocks else ""


async def call(session, key, model, extra, prompt, sem, max_tokens):
    body = {"model": model, "max_tokens": max_tokens, "temperature": 0.7,
            "messages": [{"role": "user", "content": prompt}],
            "provider": {"ignore": IGNORE, "require_parameters": True}, **extra}
    for attempt in range(4):
        try:
            async with sem, session.post("https://openrouter.ai/api/v1/chat/completions", json=body,
                                         headers={"Authorization": f"Bearer {key}"},
                                         timeout=aiohttp.ClientTimeout(total=1800)) as r:
                r.raise_for_status()
                d = await r.json()
            msg = d["choices"][0]["message"]
            content = msg.get("content") or ""
            if "```" not in content and "```" in (msg.get("reasoning") or ""):
                content = msg["reasoning"]                    # answer landed on the reasoning channel
            u = d.get("usage", {})
            return content, u.get("prompt_tokens", 0), u.get("completion_tokens", 0), d.get("provider"), None
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:200]
            await asyncio.sleep(3 * (attempt + 1))
    return "", 0, 0, None, err


async def run(a) -> None:
    from datasets import load_dataset
    ds = {r["instance_id"]: r for r in load_dataset("princeton-nlp/SWE-bench_Verified", split="test")}
    ids = json.loads(Path(a.instances_file).read_text())
    key = Path(a.api_key_file).read_text().strip()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    ctx_path = out / "contexts.jsonl"
    ctx = {json.loads(l)["instance_id"]: json.loads(l)["files"] for l in open(ctx_path)} if ctx_path.exists() else {}
    with open(ctx_path, "a") as f:
        for iid in ids:
            if iid in ctx:
                continue
            inst = ds[iid]
            ctx[iid] = {p: fetch(inst["repo"], inst["base_commit"], p, a.max_file_chars) for p in gold_files(inst["patch"])}
            f.write(json.dumps({"instance_id": iid, "files": ctx[iid]}) + "\n")
    sem = asyncio.Semaphore(a.concurrency)
    async with aiohttp.ClientSession() as session:
        for w in a.writers.split(","):
            model, extra = WRITERS[w]
            path = out / f"scripts_{w}.jsonl"
            done = {json.loads(l)["instance_id"] for l in open(path)} if path.exists() else set()
            todo = [i for i in ids if i not in done]

            async def one(iid):
                inst = ds[iid]
                files = "\n".join(f"<file path=\"{p}\">\n{t}\n</file>" for p, t in ctx[iid].items())
                prompt = PROMPT.format(repo=inst["repo"], issue=inst["problem_statement"], files=files)
                text, pt, ct, prov, err = await call(session, key, model, extra, prompt, sem, a.max_tokens)
                return {"instance_id": iid, "writer": w, "model": model, "script": extract(text),
                        "prompt_tokens": pt, "completion_tokens": ct, "provider": prov, "error": err}

            rows = await asyncio.gather(*[one(i) for i in todo])
            with open(path, "a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            n_ok = sum(bool(r["script"]) for r in rows)
            print(f"{w}: {n_ok}/{len(rows)} scripts extracted "
                  f"({sum(r['prompt_tokens'] for r in rows)/1e6:.2f}M in, "
                  f"{sum(r['completion_tokens'] for r in rows)/1e6:.2f}M out)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances-file", required=True, help="JSON list of instance_ids")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--writers", default=",".join(WRITERS))
    ap.add_argument("--api-key-file", default="/home/toolkit/.secrets/openrouter_api_key")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=32000)
    ap.add_argument("--max-file-chars", type=int, default=40000)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
