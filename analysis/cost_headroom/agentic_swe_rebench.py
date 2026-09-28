"""Agentic cost variance from public trajectories: SWE-rebench July 2026 (ibragim-bad/swe_rebench_07_2026_trajectories;
111 tasks x 17 participants x 5 runs, per-run cost_usd / steps / tokens / resolved). No API spend.

Questions:
 1. Is agentic cost "model-dominated, q-independent" (SWE-Router's assumption)? Variance decomposition of log cost into
    participant, task, participant x task, and run-to-run noise (two-way random effects on the balanced 17 x 111 x 5 grid).
 2. Per participant: spread of per-task mean cost (p90/p10), and the ICC of log cost across runs = the between-task
    share = the ceiling for ANY pre-generation cost predictor.
 3. Headroom in one-shot routing among the 13 standalone models (common scaffold): cross-fitted on runs. Success rate and
    per-task cost are estimated from runs {0,1,2}; routing is scored on the realised outcome and cost of runs {3,4}.
    ORACLE-cost arm (per-task mean cost from runs 0-2) vs CONSTANT arm (the model's median per-task cost from runs 0-2),
    both with the same per-task success estimate -> value of knowing per-task cost when success is known.
"""
import json, numpy as np, collections
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from decompose import hull, cost_at

D = Path("/mnt/llmd/results/exps/aristides/reason/swe_rebench_traj")
# OpenRouter list prices $/M (input, cache read, output), fetched 2026-09-28; the standalone models share one scaffold whose
# "input" INCLUDES cached tokens. A cache-read price of 0 = not listed -> cached tokens priced at the full input rate.
PRICE = {"Qwen/Qwen3.5-35B-A3B": (0.163, 0, 1.3), "Qwen/Qwen3.6-27B": (0.32, 0, 3.2), "Qwen/Qwen3.6-35B-A3B": (0.15, 0.05, 1.0),
         "claude-fable-5": (10, 1, 50), "claude-opus-5": (5, 0.5, 25), "claude-sonnet-5": (2, 0.2, 10),
         "deepseek/deepseek-v4-pro": (0.783, 0.065, 1.566), "gpt-5.6-luna": (0.2, 0.02, 1.2), "gpt-5.6-sol": (2, 0.2, 10),
         "minimax/minimax-m3": (0.3, 0.06, 1.2), "x-ai/grok-4.5": (2, 0.3, 6), "xiaomi/mimo-v2.5-pro": (0.435, 0.004, 0.87),
         "z-ai/glm-5.2": (0.65, 0.121, 2.042)}
rows = json.load(open(D / "light.json"))
key = lambda p: p["key"] if p.get("type") == "agentic_system" else p["model"]
parts = sorted({key(r["participant"]) for r in rows}); tasks = sorted({r["instance_id"] for r in rows})
kind = {key(r["participant"]): r["participant"].get("type") for r in rows}
pi, ti = {p: i for i, p in enumerate(parts)}, {t: i for i, t in enumerate(tasks)}
C = np.full((len(parts), len(tasks), 5), np.nan); S = np.full_like(C, np.nan); ST = np.full_like(C, np.nan)
for r in rows:
    i, j, k = pi[key(r["participant"])], ti[r["instance_id"]], int(r["run"]["index"])
    u = r["usage"] or {}; ev = r["evaluation"] or {}; tk = u.get("tokens") or {}; m = r["participant"].get("model")
    if r["participant"].get("type") != "agentic_system" and m in PRICE and tk.get("input") is not None:
        pin, pcache, pout = PRICE[m]
        cached = tk.get("cached_input") or 0; unc = max((tk.get("input") or 0) - cached, 0)
        C[i, j, k] = (unc * pin + cached * (pcache or pin) + (tk.get("output") or 0) * pout) / 1e6
    elif u.get("cost_usd") is not None:                       # agentic systems: provider/agent-reported dollars
        C[i, j, k] = u["cost_usd"]
    ST[i, j, k] = u.get("steps") or np.nan
    S[i, j, k] = float(bool(ev.get("resolved")))
print(f"{len(rows)} runs, {len(parts)} participants, {len(tasks)} tasks; runs with cost {np.isfinite(C).sum()}")

# 1. variance decomposition of log cost (balanced where finite; simple moment estimates)
L = np.log(np.where(C > 0, C, np.nan))
grand = np.nanmean(L); pm = np.nanmean(L, (1, 2)); tm = np.nanmean(L, (0, 2)); cell = np.nanmean(L, 2)
v_part = np.nanvar(pm); v_task = np.nanvar(tm)
inter = cell - pm[:, None] - tm[None, :] + grand; v_int = np.nanvar(inter)
v_run = np.nanmean(np.nanvar(L, 2))
tot = v_part + v_task + v_int + v_run
print("\n1. log-cost variance shares (approx.): " + "  ".join(f"{n} {v / tot * 100:.0f}%" for n, v in
      (("participant", v_part), ("task", v_task), ("participant x task", v_int), ("run-to-run", v_run))))
print("   => within a participant, the task (+ task x participant) share is what a per-query cost predictor could use")

# 2. per participant
print("\n2. per participant: resolve rate, mean $/run, per-task cost p90/p10, ICC of log cost across runs, mean steps")
for i, p in enumerate(parts):
    m = np.nanmean(C[i], 1); ok = np.isfinite(m) & (m > 0)
    if ok.sum() < 10:
        print(f"   {p:<22} {kind[p][:8]:<8} resolve {np.nanmean(S[i])*100:5.1f}%  (no cost data)"); continue
    lv = L[i]; within = np.nanmean(np.nanvar(lv, 1)); total = np.nanvar(lv)
    print(f"   {p:<22} {kind[p][:8]:<8} resolve {np.nanmean(S[i])*100:5.1f}%  ${np.nanmean(C[i]):.3f}  p90/p10 "
          f"{np.percentile(m[ok], 90) / np.percentile(m[ok], 10):5.1f}  ICC {1 - within / total:.2f}  steps {np.nanmean(ST[i]):5.1f}")

# 3. cross-fitted headroom among the standalone models
import os
SUBSET = os.environ.get("SUBSET", "")                     # "open" = open-weight standalone models only (narrower price gaps)
OPEN = {"Qwen/Qwen3.5-35B-A3B", "Qwen/Qwen3.6-27B", "Qwen/Qwen3.6-35B-A3B", "deepseek/deepseek-v4-pro", "minimax/minimax-m3",
        "xiaomi/mimo-v2.5-pro", "z-ai/glm-5.2"}
M = [i for i, p in enumerate(parts) if kind[p] != "agentic_system" and (SUBSET != "open" or p in OPEN)]
print(f"\nrouting pool ({SUBSET or 'all standalone'}): " + ", ".join(parts[i] for i in M) +
      f"; mean $/run range {min(np.nanmean(C[i]) for i in M):.3f}-{max(np.nanmean(C[i]) for i in M):.3f}")
fit, ev = [0, 1, 2], [3, 4]
Pf = np.nanmean(S[M][:, :, fit], 2).T; Cf = np.nanmean(C[M][:, :, fit], 2).T          # [task, model]
Qe = np.nanmean(S[M][:, :, ev], 2).T; Ce = np.nanmean(C[M][:, :, ev], 2).T
const = np.nanmedian(Cf, 0)[None, :].repeat(len(tasks), 0)
avail = np.isfinite(Cf) & np.isfinite(Ce) & np.isfinite(Pf)
VS = np.geomspace(1e-4, 1e4, 400)


def frontier(Cest, ii):
    pts = []
    for V in VS:
        U = np.where(avail[ii], Pf[ii] * V - Cest[ii], -np.inf); ch = U.argmax(1); r = np.arange(len(ii))
        pts.append((np.nanmean(Ce[ii][r, ch]), np.nanmean(Qe[ii][r, ch])))
    return hull(pts)


def headroom(ii):
    Ho, Hc = frontier(np.where(avail, Cf, 1e9), ii), frontier(np.where(avail, const, 1e9), ii)
    lo, hi = max(Ho[0][1], Hc[0][1]), min(Ho[-1][1], Hc[-1][1]); T = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 10)
    r = np.array([cost_at(Ho, x) / cost_at(Hc, x) for x in T]); return 1 - float(np.exp(np.nanmean(np.log(r)))), (lo, hi)


h, band = headroom(np.arange(len(tasks)))
rng = np.random.default_rng(0); B = [headroom(rng.integers(0, len(tasks), len(tasks)))[0] for _ in range(1000)]
print(f"\n3. one-shot routing among {len(M)} standalone models (cross-fitted on runs): HEADROOM of per-task cost knowledge vs "
      f"each model's median cost = {h*100:.1f}% [{np.percentile(B, 2.5)*100:.1f}, {np.percentile(B, 97.5)*100:.1f}] "
      f"over accuracy {band[0]*100:.0f}-{band[1]*100:.0f}%")
json.dump({"parts": parts, "headroom": h, "headroom_ci": [float(np.percentile(B, 2.5)), float(np.percentile(B, 97.5))],
           "shares": {"participant": v_part / tot, "task": v_task / tot, "interaction": v_int / tot, "run": v_run / tot}},
          open(f"analysis/cost_headroom/agentic_swe_rebench{('_' + SUBSET) if SUBSET else ''}.json", "w"), indent=1, default=float)
