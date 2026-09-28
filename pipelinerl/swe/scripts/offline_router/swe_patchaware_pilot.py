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
THE QUESTION (2026-09-28): does seeing the patch UNLOCK per-candidate test-writer selection? Issue-only tests are
written once per instance, so writer choice could only depend on the issue, and it did not pay (pilot -0.6pt).
A patch-aware check is per candidate, so the best writer may depend on the patch -- most plausibly through
same-family rubber-stamping (a writer too lenient on its own family's patches).
Writers from three families: oss20 (gpt-oss), dsv4f (deepseek), qcoder30 (qwen); patches are gpt-oss or qwen.
Mode `aware` ("a candidate patch was proposed, it may be wrong; write the check for the ISSUE"; `critic` kept as
an option). TWO independent scripts per (candidate, writer): 166 x 2 x 3 x 2 = 1992. Every script runs in Daytona
on the unpatched repo (validity, label-free), on C, on W and on gold.
Split-draw ceiling: choose the writer per candidate with labels on one draw, score it on the other draw -- a
per-candidate writer advantage that is STABLE, not a lucky accept (the in-sample oracle's flaw).
PRE-REGISTERED go criterion: the split-draw ceiling beats EVERY fixed writer (incl. the best) at pair selection by
>= +5pt at equal writing cost, or is <= 0.75x the cost at equal accuracy, bootstrap CI excluding 0. Else park.
Also reported: false-accept rate on wrong patches by writer x generator family, patch-aware vs issue-only.

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
PRICE = {"oss20": (0.018, 0.09), "dsv4f": (0.04704, 0.09408), "qcoder30": (0.07, 0.28)}  # $/M in, out
FAMILY = {"oss20": "gpt-oss", "oss120": "gpt-oss", "qwen30": "qwen", "qcoder30": "qwen", "dsv4f": "deepseek"}

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
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as session:
        for w in a.writers.split(","):
            model, extra = WRITERS[w]
            for mode in a.modes.split(","):
                path = out / f"scripts_{w}_{mode}.jsonl"
                done = {(json.loads(l)["instance_id"], json.loads(l)["role"], json.loads(l).get("draw", 0)) for l in open(path)} if path.exists() else set()
                todo = [(s, role, d) for s in sample for role in ("C", "W") for d in range(a.draws)
                        if (s["instance_id"], role, d) not in done]

                async def one(s, role, d):
                    inst = ds[s["instance_id"]]
                    files = "\n".join(f"<file path=\"{p}\">\n{t}\n</file>" for p, t in ctx[s["instance_id"]].items())
                    prompt = (HEAD.format(repo=inst["repo"], issue=inst["problem_statement"], files=files)
                              + MODE_TEXT[mode].format(patch=s[f"patch_{role}"]) + TAIL)
                    text, pt, ct, prov, err = await call(session, key, model, extra, prompt, sem, a.max_tokens)
                    return {"instance_id": s["instance_id"], "role": role, "cid": s[role], "writer": w, "mode": mode, "draw": d,
                            "script": extract(text), "prompt_tokens": pt, "completion_tokens": ct,
                            "provider": prov, "error": err}

                rows = await asyncio.gather(*[one(s, r, d) for s, r, d in todo])
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
            scripts.setdefault(r["instance_id"], {})[f"{r['writer']}__{r['mode']}__{r['role']}__d{r.get('draw', 0)}"] = r["script"]
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
    """Split-draw test of per-candidate writer choice + the same-family rubber-stamp effect."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from trackB_oracle_writer_ceiling import hull, cost_at
    out = Path(a.out_dir); mode = a.modes.split(",")[0]; Wr = a.writers.split(",")
    sample = {json.loads(l)["instance_id"]: json.loads(l) for l in open(out / "sample.jsonl")}
    ex = {json.loads(l)["instance_id"]: json.loads(l) for l in open(out / "exec.jsonl")}
    ex = {k: v for k, v in ex.items() if not v.get("error")}
    pm = {json.loads(l)["instance_id"]: json.loads(l) for l in open(a.pass_matrix)}
    wcost = {}
    for w in Wr:
        for l in open(out / f"scripts_{w}_{mode}.jsonl"):
            r = json.loads(l)
            wcost[(r["instance_id"], r["role"], w, r.get("draw", 0))] = (
                r["prompt_tokens"] * PRICE[w][0] + r["completion_tokens"] * PRICE[w][1]) / 1e4   # cents
    ids = sorted(i for i in ex if i in sample and ex[i]["apply"].get("C") and ex[i]["apply"].get("W"))
    n = len(ids)
    print(f"{len(ex)} instances executed; {n} with both candidates applying (analysis set); writers {Wr}, mode {mode}\n")
    # V[i, role, w, d] = verdict CORRECT (accept a correct candidate / reject a wrong one); K = cost (cents)
    # ACC[i, role, w, d] = accepted (valid & exit 0 on its own candidate)
    ACC = np.zeros((n, 2, len(Wr), 2)); K = np.zeros_like(ACC); VAL = np.zeros_like(ACC)
    for ii, i in enumerate(ids):
        e = ex[i]
        for ri, role in enumerate(("C", "W")):
            for wi, w in enumerate(Wr):
                for d in range(2):
                    lab = f"{w}__{mode}__{role}__d{d}"
                    valid = e["base"].get(lab) not in (0, None)
                    ACC[ii, ri, wi, d] = valid and e["patches"].get(role, {}).get(lab) == 0
                    VAL[ii, ri, wi, d] = valid
                    K[ii, ri, wi, d] = wcost.get((i, role, w, d), np.nan)
    K = np.where(np.isfinite(K), K, np.nanmedian(K))
    V = np.stack([ACC[:, 0], 1 - ACC[:, 1]], 1)            # correct verdicts
    # ---- 1. per-writer reliability (both draws pooled) + draw-to-draw stability
    print("per writer (patch-aware; both draws): valid, beta = P(accept | correct), alpha = P(accept | wrong), median write cost")
    for wi, w in enumerate(Wr):
        agree = (V[:, :, wi, 0] == V[:, :, wi, 1]).mean()
        print(f"   {w:<9} valid {VAL[:, :, wi].mean()*100:4.0f}%  beta {ACC[:, 0, wi].mean()*100:5.1f}%  alpha {ACC[:, 1, wi].mean()*100:5.1f}%  "
              f"Lambda {ACC[:, 0, wi].mean()/max(ACC[:, 1, wi].mean(),1e-9):5.2f}  cost {np.median(K[:, :, wi]):.3f}c  "
              f"draw1-draw2 verdict agreement {agree*100:.0f}%")
    # ---- 2. split-draw: choose the writer PER CANDIDATE on draw d (with labels), score on the other draw
    def pair_score(v):                                     # v[:, 2] correct-verdict indicators (C, W)
        acC, rjW = v[:, 0], v[:, 1]                         # both right -> correct pick; both wrong -> wrong; else tie
        return np.where((acC == 1) & (rjW == 1), 1.0, np.where((acC == 0) & (rjW == 0), 0.0, 0.5))
    mus = np.r_[0.0, np.geomspace(0.01, 100, 50)]

    def frontiers(idx):
        v, k = V[idx], K[idx]
        fixed = []
        for wi in range(len(Wr)):
            for d in range(2):
                fixed.append((k[:, :, wi, d].sum(1).mean(), pair_score(v[:, :, wi, d]).mean(),
                               v[:, :, wi, d].mean()))
        fx = [(np.mean([f[0] for f in fixed[2*wi:2*wi+2]]), np.mean([f[1] for f in fixed[2*wi:2*wi+2]])) for wi in range(len(Wr))]
        orc = []
        for mu in mus:
            sc, co = [], []
            for d in range(2):                               # choose on draw d, score on draw 1-d; average both ways
                u = v[:, :, :, d] - mu * k[:, :, :, d]         # [n, 2, W]
                ch = u.argmax(2)
                vv = np.take_along_axis(v[:, :, :, 1 - d], ch[..., None], 2)[..., 0]
                kk = np.take_along_axis(k[:, :, :, 1 - d], ch[..., None], 2)[..., 0]
                sc.append(pair_score(vv).mean()); co.append(kk.sum(1).mean())
            orc.append((np.mean(co), np.mean(sc)))
        return hull(fx), hull(orc), fx
    hf, ho, fx = frontiers(np.arange(n))
    rng = np.random.default_rng(0)
    B = [frontiers(rng.integers(0, n, n)) for _ in range(a.boot)]
    print("\npair selection (pick the candidate whose own check accepts it; ties 1/2), cost = two checks, cents:")
    for wi, w in enumerate(Wr):
        print(f"   always {w:<9} {fx[wi][1]*100:5.1f}% @ {fx[wi][0]:.3f}c")
    print(f"   split-draw per-candidate writer (labels on one draw, scored on the other): max {ho[-1][1]*100:.1f}% @ {ho[-1][0]:.3f}c")
    go = False
    for wi, w in enumerate(Wr):
        c0, s0 = fx[wi]
        # accuracy of the oracle hull AT the fixed writer's cost (interpolated)
        def acc_at(h, c):
            if not h or c < h[0][0]:
                return np.nan
            for (c1, a1), (c2, a2) in zip(h, h[1:]):
                if c1 <= c <= c2:
                    return a1 + (a2 - a1) * (c - c1) / (c2 - c1)
            return h[-1][1]
        g = acc_at(ho, c0) - s0
        gb = np.array([acc_at(b[1], b[2][wi][0]) - b[2][wi][1] for b in B]); gb = gb[np.isfinite(gb)]
        cr = cost_at(ho, s0) / c0
        cb = np.array([cost_at(b[1], b[2][wi][1]) / b[2][wi][0] for b in B]); cb = cb[np.isfinite(cb)]
        lo, hi = np.percentile(gb, [2.5, 97.5]); clo, chi = np.percentile(cb, [2.5, 97.5])
        print(f"   vs always {w:<9} at equal cost: {g*100:+.1f}pt [{lo*100:+.1f},{hi*100:+.1f}]   "
              f"at equal accuracy: cost x{cr:.2f} [{clo:.2f},{chi:.2f}]")
        if (g >= 0.05 and lo > 0) or (cr <= 0.75 and chi < 1):
            go = True
    best = max(range(len(Wr)), key=lambda wi: fx[wi][1])
    print(f"\nPRE-REGISTERED go (vs every fixed writer incl. the best, {Wr[best]}): split-draw ceiling >= +5pt at equal cost "
          f"or <= 0.75x cost at equal accuracy, CI excluding 0 -> {'GO (per-candidate writer choice has room)' if go else 'NO GO'}")
    # ---- 3. same-family rubber-stamp: alpha on WRONG candidates by writer x generator family, aware vs issue-only
    print("\nfalse accepts on WRONG candidates, by writer and the patch's generator family (patch-aware, both draws | issue-only):")
    for wi, w in enumerate(Wr):
        cells = []
        for fam in ("gpt-oss", "qwen"):
            sel = [ii for ii, i in enumerate(ids) if FAMILY[sample[i]["W"].split("_d")[0]] == fam]
            aw = ACC[sel, 1, wi].mean() if sel else np.nan
            io = []
            for ii in sel:
                i = ids[ii]; t = next((x for x in pm[i]["tests"] if x["writer"] == w), None)
                io.append(bool(t is not None and t["valid"] is not False and t["passes"].get(sample[i]["W"], False)))
            same = "SAME" if FAMILY[w] == fam else "other"
            cells.append(f"{fam} patches ({same}, n={len(sel)}): {aw*100:4.1f}% | {np.mean(io)*100:4.1f}%")
        print(f"   {w:<9} " + "   ".join(cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["sample", "generate", "execute", "analyze"])
    ap.add_argument("--out-dir", default=str(R / "swe_patchaware_pilot"))
    ap.add_argument("--pass-matrix", default=str(R / "trackB/pass_matrix_swe_verified.jsonl"))
    ap.add_argument("--writers", default="oss20,dsv4f,qcoder30")
    ap.add_argument("--modes", default="aware")
    ap.add_argument("--draws", type=int, default=2, help="independent scripts per (candidate, writer, mode)")
    ap.add_argument("--limit", type=int, default=0, help="generate for the first N sampled instances only")
    ap.add_argument("--api-key-file", default="/home/toolkit/.secrets/openrouter_api_key")
    ap.add_argument("--concurrency", type=int, default=12, help="API concurrency (generate) / Daytona sandboxes (execute, keep <= 3)")
    ap.add_argument("--max-tokens", type=int, default=32000)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--boot", type=int, default=1000)
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
