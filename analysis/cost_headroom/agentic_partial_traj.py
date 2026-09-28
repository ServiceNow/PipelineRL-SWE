"""Can a PARTIAL agent trajectory predict the cost of the run -- and of other models on the same task?
SWE-rebench July 2026 trajectories (111 tasks x 13 standalone models x 5 runs); costs priced from tokens as in
agentic_swe_rebench.py. Features after the first k agent steps (k = 3, 5, 10, 20), from the event trace only:
tool calls by tool, assistant-message chars, tool-result chars (context-growth proxy), tool errors, elapsed seconds.
  own    predict log final cost of THIS run from its own first k steps (per model; runs that finished within k steps are
         excluded -- their cost is known, scoring them would inflate R2)
  cross  a cheap open scout (MiMo-V2.5-Pro) runs k steps on the task; its features predict each OTHER model's log mean cost
         on that task (the routing-relevant question: explore cheaply, then price the candidates)
Grouped 5-fold CV by TASK (runs of one task never straddle train/test). Baselines: model median (R2 <= 0) and the
statement prefill (agentic_predictability.py: R2 ~ 0).
"""
import gzip, json, glob, numpy as np, collections, sys
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
sys.path.insert(0, str(Path(__file__).parent))
from agentic_swe_rebench import PRICE

D = Path("/mnt/llmd/results/exps/aristides/reason/swe_rebench_traj")
KS = (3, 5, 10, 20)
TOOLS = ["bash_terminal", "str_replace_editor", "file_editor", "edit", "view", "search", "submit", "think"]


def feats(events, k):
    steps = 0; f = collections.Counter(); t = 0.0
    for e in events:
        typ = e.get("type")
        if typ == "assistant_message":
            steps += 1
            if steps > k:
                break
            f["asst_chars"] += len(str(e.get("content") or ""))
        elif typ == "tool_call":
            tool = str(e.get("tool") or "other"); f["calls"] += 1; f["tool_" + (tool if tool in TOOLS else "other")] += 1
            inp = str(e.get("input") or ""); f["pytest"] += "pytest" in inp or "test" in inp; f["grep"] += ("grep" in inp) + ("find " in inp)
        elif typ == "tool_result":
            f["result_chars"] += len(str(e.get("content") or e.get("output") or "")); f["errors"] += e.get("status") == "error"
        t += float(e.get("duration_ms") or 0) / 1000
    f["elapsed"] = t
    return [np.log1p(f[x]) for x in ("asst_chars", "calls", "result_chars", "errors", "pytest", "grep", "elapsed")] + \
           [f["tool_" + x] for x in TOOLS + ["other"]], steps > k      # (features, still running after k steps)


rows = []
for fpath in sorted(glob.glob(str(D / "trajectories/*/run_*.jsonl.gz"))):
    for l in gzip.open(fpath, "rt"):
        r = json.loads(l); p = r["participant"]
        if p.get("type") == "agentic_system" or p.get("model") not in PRICE:
            continue
        tk = (r["usage"] or {}).get("tokens") or {}
        if tk.get("input") is None:
            continue
        pin, pc, pout = PRICE[p["model"]]; cached = tk.get("cached_input") or 0
        cost = ((max(tk["input"] - cached, 0)) * pin + cached * (pc or pin) + (tk.get("output") or 0) * pout) / 1e6
        rows.append({"model": p["model"], "task": r["instance_id"], "run": r["run"]["index"], "cost": cost,
                     "F": {k: feats(r["events"], k) for k in KS}})
models = sorted({r["model"] for r in rows}); tasks = sorted({r["task"] for r in rows}); ti = {t: i for i, t in enumerate(tasks)}
print(f"{len(rows)} standalone runs, {len(models)} models, {len(tasks)} tasks")


def cv_r2(X, y, groups):
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        pred[te] = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, min_samples_leaf=10).fit(X[tr], y[tr]).predict(X[te])
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


print("\nOWN run: log final cost from the run's first k steps (runs still going at step k only); mean over models")
for k in KS:
    r2s, alive = [], []
    for m in models:
        rs = [r for r in rows if r["model"] == m and r["F"][k][1]]
        alive.append(len(rs) / max(1, sum(r["model"] == m for r in rows)))
        if len(rs) < 60:
            continue
        X = np.array([r["F"][k][0] for r in rs]); y = np.log([r["cost"] for r in rs]); g = [ti[r["task"]] for r in rs]
        r2s.append(cv_r2(X, y, g))
    print(f"   k={k:>2}: R2 {np.mean(r2s):+.2f} (models {len(r2s)}, range {min(r2s):+.2f}..{max(r2s):+.2f}); share of runs still going {np.mean(alive)*100:.0f}%")

SCOUT = "xiaomi/mimo-v2.5-pro"
print(f"\nCROSS-model: scout {SCOUT} runs k steps (run 0); predict each other model's log mean cost on the task")
Y = collections.defaultdict(dict)
for r in rows:
    Y[r["model"]].setdefault(r["task"], []).append(r["cost"])
scout = {r["task"]: r for r in rows if r["model"] == SCOUT and r["run"] == 0}
for k in KS:
    r2s = []
    for m in models:
        if m == SCOUT:
            continue
        ts = [t for t in tasks if t in scout and t in Y[m]]
        X = np.array([scout[t]["F"][k][0] + [float(scout[t]["F"][k][1])] for t in ts]); y = np.log([np.mean(Y[m][t]) for t in ts])
        r2s.append(cv_r2(X, y, [ti[t] for t in ts]))
    print(f"   k={k:>2}: R2 {np.mean(r2s):+.2f} (range {min(r2s):+.2f}..{max(r2s):+.2f})")
