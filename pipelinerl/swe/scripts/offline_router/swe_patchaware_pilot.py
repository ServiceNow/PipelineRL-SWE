#!/usr/bin/env python3
"""Track B, patch-aware test pilot: does a writer that SEES the candidate patch write a sharper check?

All tests so far were ISSUE-ONLY: one reproduction script per instance, written from the issue (+ oracle-localised
files), run on every candidate. Their false-accept rate on wrong patches is 12-25%, and with those tests no policy
beats routing once to gpt-oss-120b below ~52%. A PATCH-AWARE test is written per candidate, from the issue + files +
that candidate's diff, so it can target what the patch changed (missed cases, wrong branch). Risk: it rubber-stamps
the patch it was shown. Maestro Order (2606.23983) assumes verifiers with fixed (beta, alpha) and never measures real
ones; here we measure beta (pass | correct) and alpha (pass | wrong) for real writers, paired against issue-only.

Sample: every one of the 368 Verified instances whose ladder (oss20 x3, qwen30 x3, oss120) has >= 1 correct and
>= 1 wrong candidate (166); one correct (C) and one wrong (W) candidate drawn at random (seed 0).
Writers: oss20 (cheapest), dsv4f (best issue-only writer). Modes:
  aware    "a candidate patch was proposed, it may be wrong; write the reproduction script for the ISSUE"
  critic   "review this patch; write a script that exits 1 unless the issue is FULLY fixed"
One script per (instance, candidate, writer, mode) = 166 x 2 x 2 x 2 = 1328. Every script runs in Daytona on the
unpatched repo (validity, label-free), on C, on W and on gold -- so each test is scored on its OWN candidate
(the deployable check) and on the OTHER candidate (rubber-stamp probe: alpha_own >> alpha_cross = rubber-stamping).
Issue-only baseline for the same C/W comes from the existing pass matrix (same writers, same candidates).

Subcommands: sample | generate | execute | analyze.
PRE-REGISTERED go criterion (decided before any data): for dsv4f, some patch-aware mode cuts the false-accept rate
alpha (valid & pass on a wrong candidate) by >= 1/3 relative to issue-only, with the paired 95% CI on the
difference excluding 0, while beta (valid & pass on a correct candidate) drops by <= 5pt. Otherwise: no juice.
"""
from __future__ import annotations
import argparse, asyncio, json, os, random
from pathlib import Path
import numpy as np

R = Path("/mnt/llmd/results/exps/aristides/reason")
PILOT = R / "swe_testwriter_pilot"
LADDER = ["oss20", "oss20_d1", "oss20_d2", "qwen30", "qwen30_d1", "qwen30_d2", "oss120"]
PRICE = {"oss20": (0.018, 0.09), "dsv4f": (0.04704, 0.09408)}  # $/M in, out

MODE_TEXT = {
    "aware": """A candidate patch has been proposed for this issue. It may or may not be correct.
<candidate_patch>
{patch}
</candidate_patch>

Write ONE standalone Python script that checks whether the issue is fixed. It will be run from the repository
root (the package is installed in development mode) as `python repro.py`, with or without the patch applied.
Test the behaviour the ISSUE asks for; do not merely mirror the patch's implementation -- an incorrect patch
must fail your script.""",
    "critic": """A candidate patch has been proposed for this issue. Your job is to find out whether it REALLY fixes it.
<candidate_patch>
{patch}
</candidate_patch>

Look for ways this patch could fail to fully resolve the issue: cases the issue describes that it misses, wrong
behaviour on the inputs the issue mentions, or breaking the code path it touches. Write ONE standalone Python
script that exercises those cases. It will be run from the repository root (the package is installed in
development mode) as `python repro.py`, with or without the patch applied.""",
}
TAIL = """
- Exit with status 1 if the issue is PRESENT or only partially fixed (any buggy behaviour described in the issue occurs).
- Exit with status 0 only if the issue is FULLY fixed (the correct behaviour occurs).
- Do not modify any repository files. Configure anything the library needs inline (for Django, call
  django.conf.settings.configure(...) and django.setup() before importing models).
- Print a one-line diagnostic of what you observed.
Return only the script, in a single ```python code block."""
HEAD = """You are writing a check for a bug report in the repository {repo}.

<issue>
{issue}
</issue>

Relevant source files, before any fix:
{files}

"""


def candidate_patches() -> dict:
    """{(instance_id, cid): diff} for every ladder candidate (pool first draws + redraws)."""
    runs = json.loads((PILOT / "patch_runs.json").read_text())
    out = {}
    for k, r in runs.items():
        f = R / f"opus_verified_daytona_eval_{r}/predictions/predictions_opus_verified.jsonl"
        for l in open(f):
            x = json.loads(l); out[(x["instance_id"], k)] = x.get("model_patch") or ""
    for f in (PILOT / "redraws").glob("predictions_*_d*.jsonl"):
        if f.name.endswith(".results.jsonl"):
            continue
        cid = f.stem.replace("predictions_", "")
        for l in open(f):
            x = json.loads(l); out[(x["instance_id"], cid)] = x.get("model_patch") or ""
    return out


def cmd_sample(a):
    recs = [json.loads(l) for l in open(a.pass_matrix)]
    pat = candidate_patches(); rng = random.Random(0); rows = []
    for r in recs:
        cands = [c for c in r["candidates"] if c["cid"] in LADDER and pat.get((r["instance_id"], c["cid"]), "").strip()]
        C = [c for c in cands if c["correct"]]; W = [c for c in cands if not c["correct"]]
        if C and W:
            c, w = rng.choice(C), rng.choice(W)
            rows.append({"instance_id": r["instance_id"], "C": c["cid"], "W": w["cid"],
                         "patch_C": pat[(r["instance_id"], c["cid"])], "patch_W": pat[(r["instance_id"], w["cid"])]})
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / "sample.jsonl", "w") as f:
        for x in rows:
            f.write(json.dumps(x) + "\n")
    print(f"{len(rows)} instances sampled -> {out/'sample.jsonl'}")


async def _generate(a):
    import aiohttp
    from datasets import load_dataset
    from swe_testwriter_generate import WRITERS, call, extract
    ds = {r["instance_id"]: r for r in load_dataset("princeton-nlp/SWE-bench_Verified", split="test")}
    out = Path(a.out_dir)
    sample = [json.loads(l) for l in open(out / "sample.jsonl")]
    if a.limit:
        sample = sample[: a.limit]
    ctx = {json.loads(l)["instance_id"]: json.loads(l)["files"] for l in open(PILOT / "scripts" / "contexts.jsonl")}
    key = Path(a.api_key_file).read_text().strip()
    sem = asyncio.Semaphore(a.concurrency)
    async with aiohttp.ClientSession() as session:
        for w in a.writers.split(","):
            model, extra = WRITERS[w]
            for mode in a.modes.split(","):
                path = out / f"scripts_{w}_{mode}.jsonl"
                done = {(json.loads(l)["instance_id"], json.loads(l)["role"]) for l in open(path)} if path.exists() else set()
                todo = [(s, role) for s in sample for role in ("C", "W") if (s["instance_id"], role) not in done]

                async def one(s, role):
                    inst = ds[s["instance_id"]]
                    files = "\n".join(f"<file path=\"{p}\">\n{t}\n</file>" for p, t in ctx[s["instance_id"]].items())
                    prompt = (HEAD.format(repo=inst["repo"], issue=inst["problem_statement"], files=files)
                              + MODE_TEXT[mode].format(patch=s[f"patch_{role}"]) + TAIL)
                    text, pt, ct, prov, err = await call(session, key, model, extra, prompt, sem, a.max_tokens)
                    return {"instance_id": s["instance_id"], "role": role, "cid": s[role], "writer": w, "mode": mode,
                            "script": extract(text), "prompt_tokens": pt, "completion_tokens": ct,
                            "provider": prov, "error": err}

                rows = await asyncio.gather(*[one(s, r) for s, r in todo])
                with open(path, "a") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
                usd = sum(r["prompt_tokens"] * PRICE[w][0] + r["completion_tokens"] * PRICE[w][1] for r in rows) / 1e6
                print(f"{w}/{mode}: {sum(bool(r['script']) for r in rows)}/{len(rows)} scripts, ${usd:.3f}", flush=True)


async def _execute(a):
    from daytona import AsyncDaytona
    from datasets import load_dataset
    from swe_testwriter_execute import one
    ds = {r["instance_id"]: r for r in load_dataset("princeton-nlp/SWE-bench_Verified", split="test")}
    out = Path(a.out_dir)
    sample = {json.loads(l)["instance_id"]: json.loads(l) for l in open(out / "sample.jsonl")}
    scripts = {}
    for f in out.glob("scripts_*_*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            scripts.setdefault(r["instance_id"], {})[f"{r['writer']}__{r['mode']}__{r['role']}"] = r["script"]
    ex = out / "exec.jsonl"
    done = {json.loads(l)["instance_id"] for l in open(ex) if not json.loads(l).get("error")} if ex.exists() else set()
    todo = [i for i in sample if i in scripts and i not in done]
    print(f"{len(todo)} instances to execute ({len(done)} done)", flush=True)
    sem = asyncio.Semaphore(a.concurrency)
    async with AsyncDaytona() as daytona:
        tasks = [one(daytona, sem, i, scripts[i],
                     {"C": sample[i]["patch_C"], "W": sample[i]["patch_W"], "gold": ds[i]["patch"]}, a.timeout)
                 for i in todo]
        with open(ex, "a") as f:
            for n, t in enumerate(asyncio.as_completed(tasks), 1):
                r = await t
                f.write(json.dumps(r) + "\n"); f.flush()
                print(f"[{n}/{len(todo)}] {r['instance_id']} err={r.get('error', '')[:80]}", flush=True)


def cmd_analyze(a):
    out = Path(a.out_dir)
    sample = {json.loads(l)["instance_id"]: json.loads(l) for l in open(out / "sample.jsonl")}
    ex = {}
    for l in open(out / "exec.jsonl"):
        r = json.loads(l)
        if not r.get("error"):
            ex[r["instance_id"]] = r
    pm = {json.loads(l)["instance_id"]: json.loads(l) for l in open(a.pass_matrix)}
    cost = {}
    for f in out.glob("scripts_*_*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            cost[(r["writer"], r["mode"])] = cost.get((r["writer"], r["mode"]), []) + [
                (r["prompt_tokens"] * PRICE[r["writer"]][0] + r["completion_tokens"] * PRICE[r["writer"]][1]) / 1e4]
    ids = sorted(i for i in ex if i in sample)
    applied = [i for i in ids if ex[i]["apply"].get("C") and ex[i]["apply"].get("W")]
    print(f"{len(ids)} instances executed; {len(applied)} with both candidates applying (analysis set)\n")
    rng = np.random.default_rng(0)

    def acc(code):            # the deployable verdict: script exists, FAILS on base (valid), exits 0 on the patch
        return code == 0

    def arms(w):
        """per instance: (beta, alpha) indicators for issue-only and each patch-aware mode, own and cross."""
        res = {}
        for i in applied:
            s = sample[i]; e = ex[i]
            t = next((x for x in pm[i]["tests"] if x["writer"] == w), None)
            iv = t is not None and t["valid"] is not False
            res.setdefault("issue-only", []).append((iv and t["passes"].get(s["C"], False), iv and t["passes"].get(s["W"], False)))
            for mode in a.modes.split(","):
                lc, lw = f"{w}__{mode}__C", f"{w}__{mode}__W"
                vC = e["base"].get(lc) not in (0, None); vW = e["base"].get(lw) not in (0, None)
                pc = lambda lab, p: acc(e["patches"].get(p, {}).get(lab))
                res.setdefault(f"{mode} own", []).append((vC and pc(lc, "C"), vW and pc(lw, "W")))
                res.setdefault(f"{mode} cross", []).append((vW and pc(lw, "C"), vC and pc(lc, "W")))
                res.setdefault(f"{mode} valid", []).append((vC, vW))
                res.setdefault(f"{mode} gold", []).append((vC and pc(lc, "gold"), vW and pc(lw, "gold")))
        return {k: np.array(v, float) for k, v in res.items()}

    report = {}
    for w in a.writers.split(","):
        A = arms(w); base = A["issue-only"]
        print(f"== {w}: beta = P(valid & pass | correct), alpha = P(valid & pass | wrong); Lambda = beta/alpha")
        print(f"   {'arm':<14}{'beta':>7}{'alpha':>7}{'Lambda':>8}   d_beta vs issue [95% CI]    d_alpha vs issue [95% CI]   write cost/candidate")
        for k in ["issue-only"] + [f"{m} {x}" for m in a.modes.split(",") for x in ("own", "cross")]:
            v = A[k]; b, al = v[:, 0].mean(), v[:, 1].mean()
            line = f"   {k:<14}{b*100:6.1f}%{al*100:6.1f}%{b/max(al,1e-9):8.2f}"
            if k != "issue-only":
                d = v - base
                bs = np.array([d[rng.integers(0, len(d), len(d))].mean(0) for _ in range(4000)])
                lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
                line += (f"   {d[:,0].mean()*100:+5.1f} [{lo[0]*100:+5.1f},{hi[0]*100:+5.1f}]"
                         f"      {d[:,1].mean()*100:+5.1f} [{lo[1]*100:+5.1f},{hi[1]*100:+5.1f}]")
                m = k.split()[0]
                if k.endswith("own"):
                    line += f"   {np.median(cost.get((w, m), [np.nan])):.3f}c"
                    report[(w, m)] = dict(beta=b, alpha=al, d_beta=d[:, 0].mean(), d_alpha=d[:, 1].mean(),
                                          d_alpha_ci=[lo[1], hi[1]], alpha_issue=base[:, 1].mean())
            print(line)
        for m in a.modes.split(","):
            v = A[f"{m} valid"]; g = A[f"{m} gold"]
            print(f"   {m}: valid {v.mean()*100:.0f}% (from C {v[:,0].mean()*100:.0f}%, from W {v[:,1].mean()*100:.0f}%); "
                  f"passes gold {g.mean()*100:.0f}%")
        # pair selection: pick the candidate whose OWN test accepts it (tie -> coin flip)
        for k in ["issue-only"] + [f"{m} own" for m in a.modes.split(",")]:
            v = A[k]; p = np.where(v[:, 0] == v[:, 1], 0.5, v[:, 0])
            print(f"   pair selection ({k}): picks the correct one {p.mean()*100:.1f}% (random 50%)")
        print()
    print("PRE-REGISTERED go criterion (dsv4f): alpha down >= 1/3 relative, CI on d_alpha excludes 0, beta down <= 5pt")
    for (w, m), r in report.items():
        if w != "dsv4f":
            continue
        go = (r["d_alpha"] <= -r["alpha_issue"] / 3 and r["d_alpha_ci"][1] < 0 and r["d_beta"] >= -0.05)
        print(f"   dsv4f {m}: alpha {r['alpha_issue']*100:.1f}% -> {r['alpha']*100:.1f}%, d_beta {r['d_beta']*100:+.1f}pt -> {'GO' if go else 'no go'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["sample", "generate", "execute", "analyze"])
    ap.add_argument("--out-dir", default=str(R / "swe_patchaware_pilot"))
    ap.add_argument("--pass-matrix", default=str(R / "trackB/pass_matrix_swe_verified.jsonl"))
    ap.add_argument("--writers", default="oss20,dsv4f")
    ap.add_argument("--modes", default="aware,critic")
    ap.add_argument("--limit", type=int, default=0, help="generate for the first N sampled instances only")
    ap.add_argument("--api-key-file", default="/home/toolkit/.secrets/openrouter_api_key")
    ap.add_argument("--concurrency", type=int, default=12, help="API concurrency (generate) / Daytona sandboxes (execute, keep <= 3)")
    ap.add_argument("--max-tokens", type=int, default=32000)
    ap.add_argument("--timeout", type=int, default=120)
    a = ap.parse_args()
    if a.cmd == "sample":
        cmd_sample(a)
    elif a.cmd == "generate":
        asyncio.run(_generate(a))
    elif a.cmd == "execute":
        if not os.environ.get("DAYTONA_API_KEY"):
            for line in open("/home/toolkit/PipelineRL-SWE/.env"):
                if line.startswith("DAYTONA_API_KEY="):
                    os.environ["DAYTONA_API_KEY"] = line.split("=", 1)[1].strip().strip("'\"")
        asyncio.run(_execute(a))
    else:
        cmd_analyze(a)


if __name__ == "__main__":
    main()
