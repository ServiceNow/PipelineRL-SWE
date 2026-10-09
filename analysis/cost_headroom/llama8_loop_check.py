"""Are Llama-3.1-8B's repetition loops (NEW_PATH 4.A.71 correction: 9-12% of its math answers run to the 16k cap, ~80% of those
loop) the model, the provider or the sampling? Problems: 300 math problems where nr_llama8 LOOPED (zlib ratio of the last 4k chars of
its answer < .12) and 150 controls where it stopped normally. Arms (one draw each, card sampling T 0.6 / top_p 0.9 unless stated):
  l8_redraw   DeepInfra again, same settings, 16k cap       (is looping a property of the problem or of the draw?)
  l8_novita   Novita, same settings, 14k cap (its maximum)  (provider)
  l8_reppen   DeepInfra, repetition_penalty 1.1, 16k cap    (sampling)
Usage: python llama8_loop_check.py build OUT      (local: needs the HF dataset cache; writes OUT/tasks_<ds>.jsonl + manifest.json)
       python llama8_loop_check.py run OUT BUDGET (eai: collects all arms with collect_second_family's runner, spend guard BUDGET)
       python llama8_loop_check.py report OUT     (loop / cap / accuracy per arm and group)
"""
import argparse, asyncio, json, sys, zlib
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]; R = Path("/mnt/llmd/results/exps/aristides/reason")
sys.path.insert(0, str(REPO / "pipelinerl/swe/scripts/math_pool"))
import collect_math_pool as cmp  # noqa: E402
import collect_second_family as csf  # noqa: E402

cmp.ROUTES["l8_redraw"] = ("meta-llama/llama-3.1-8b-instruct", {"provider": {"only": ["DeepInfra"], "allow_fallbacks": False, "require_parameters": True}}, 0.6, 0.9)
cmp.ROUTES["l8_novita"] = ("meta-llama/llama-3.1-8b-instruct", {"provider": {"only": ["Novita"], "allow_fallbacks": False, "require_parameters": True}}, 0.6, 0.9)
cmp.ROUTES["l8_reppen"] = ("meta-llama/llama-3.1-8b-instruct", {"provider": {"only": ["DeepInfra"], "allow_fallbacks": False, "require_parameters": True},
                                                             "repetition_penalty": 1.1}, 0.6, 0.9)
loops = lambda t: len(t) > 500 and len(zlib.compress(t[-4000:].encode())) / len(t[-4000:].encode()) < .12


def build(out):
    rows = {}
    for f in [*R.glob("math_pool_nonreason/*/nr_llama8_d0.jsonl"), *R.glob("math_expand_nonreason/*/nr_llama8_d0.jsonl")]:
        for l in open(f):
            if l.strip():
                r = json.loads(l); rows[(f.parent.name, r["problem_id"])] = r            # latest row per (dataset, problem)
    loop = [k for k, r in rows.items() if loops(r.get("content") or "")]
    ctrl = [k for k, r in rows.items() if r.get("finish_reason") == "stop" and not loops(r.get("content") or "")]
    rng = np.random.default_rng(0)
    pick = {"loop": [loop[i] for i in rng.choice(len(loop), min(300, len(loop)), replace=False)],
            "control": [ctrl[i] for i in rng.choice(len(ctrl), 150, replace=False)]}
    tasks = {}
    for ds in ("omni500", "mmlupro"):
        allt = {t["problem_id"]: t for t in cmp.load(ds)}
        for f in (REPO / "analysis/cost_headroom/expansion_20261001").glob(f"{ds}_tasks*.jsonl"):
            allt.update({t["problem_id"]: t for t in map(json.loads, f.read_text().splitlines())})
        tasks[ds] = allt
    Path(out).mkdir(parents=True, exist_ok=True); man = {}
    for ds in ("omni500", "mmlupro"):
        sel = [tasks[ds][pid] for g in pick for (d, pid) in pick[g] if d == ds]
        Path(out, f"tasks_{ds}.jsonl").write_text("".join(json.dumps(t) + "\n" for t in sel))
    for g, ks in pick.items():
        for d, pid in ks:
            r = rows[(d, pid)]; man[f"{d}|{pid}"] = dict(group=g, orig_tokens=r.get("completion_tokens"), orig_finish=r.get("finish_reason"),
                                                          orig_resolved=bool(r.get("resolved")), orig_loop=loops(r.get("content") or ""))
    Path(out, "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"loop rows {len(loop)} / {len(rows)}; picked {len(pick['loop'])} loop + {len(pick['control'])} control; "
          + ", ".join(f"{ds} {sum(1 for k in man if k.startswith(ds))}" for ds in ("omni500", "mmlupro")))


def run(out, budget):
    for routes, mt in (("l8_redraw,l8_reppen", 16000), ("l8_novita", 14000)):
        a = argparse.Namespace(out=out, routes=routes, datasets="omni500,mmlupro", budget_usd=budget, concurrency=12, max_tokens=mt,
                               pilot=0, tasks_file=str(Path(out) / "tasks_{ds}.jsonl"), api_key_file="/home/toolkit/.secrets/openrouter_api_key")
        asyncio.run(csf.run(a))


def report(out):
    man = json.loads(Path(out, "manifest.json").read_text()); res = {}
    for arm in ("l8_redraw", "l8_novita", "l8_reppen"):
        rs = {}
        for ds in ("omni500", "mmlupro"):
            f = Path(out, ds, f"{arm}_d0.jsonl")
            if f.exists():
                rs.update({f"{ds}|{json.loads(l)['problem_id']}": json.loads(l) for l in f.read_text().splitlines() if l.strip()})
        for g in ("loop", "control"):
            ks = [k for k, m in man.items() if m["group"] == g and k in rs and rs[k].get("finish_reason") != "error"]
            if not ks:
                continue
            x = [rs[k] for k in ks]
            res[f"{arm}|{g}"] = dict(n=len(ks), loop=float(np.mean([loops(r.get("content") or "") for r in x])),
                                     capped=float(np.mean([r.get("finish_reason") == "length" for r in x])),
                                     acc=float(np.mean([bool(r.get("resolved")) for r in x])),
                                     orig_acc=float(np.mean([man[k]["orig_resolved"] for k in ks])),
                                     median_tokens=float(np.median([r.get("completion_tokens") or 0 for r in x])),
                                     providers=sorted({r.get("provider") for r in x if r.get("provider")}))
            print(f"  {arm:<10} {g:<8} n={len(ks):3d}  loops {res[f'{arm}|{g}']['loop']:5.1%}  capped {res[f'{arm}|{g}']['capped']:5.1%}  "
                  f"acc {res[f'{arm}|{g}']['acc']:5.1%} (original draw {res[f'{arm}|{g}']['orig_acc']:5.1%})  median out "
                  f"{res[f'{arm}|{g}']['median_tokens']:.0f}  {res[f'{arm}|{g}']['providers']}", flush=True)
    Path(out, "report.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"build": lambda: build(sys.argv[2]), "run": lambda: run(sys.argv[2], float(sys.argv[3])), "report": lambda: report(sys.argv[2])}[sys.argv[1]]()
