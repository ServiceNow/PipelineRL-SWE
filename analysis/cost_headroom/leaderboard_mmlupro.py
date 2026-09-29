"""Population data for ZeroRouter's stage 1 on MMLU-Pro: per-question correctness of Open LLM Leaderboard models on OUR 1000
MMLU-Pro problems (the paper's own data source). Streams each model's samples_leaderboard_mmlu_pro_*.jsonl (~378 MB) and keeps only
{question_id: correct} for our problems, so disk use is small (the transfer is not).
Model choice (--n): not merged, not flagged, available on the hub, one model per base model (no near-duplicates), spread evenly over
MMLU-PRO Raw by stratifying into --n score bins and taking the most-liked model in each bin (then filling gaps from neighbours).
Usage: python leaderboard_mmlupro.py --n 50 [--test-one]
Output: <R>/leaderboard_mmlupro/models.json, <R>/leaderboard_mmlupro/outcomes/<model>.json
"""
import argparse, json, sys, time, numpy as np, pandas as pd, requests
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from decompose import R

O = R / "leaderboard_mmlupro"; (O / "outcomes").mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=50); ap.add_argument("--test-one", action="store_true")
a = ap.parse_args()
TOK = (Path.home() / ".cache/huggingface/token").read_text().strip()      # sent only to huggingface.co
HDR = {"Authorization": f"Bearer {TOK}"}
ours = {int(json.loads(l)["problem_id"].split("_")[1]) for l in open(R / "math_pool" / "mmlupro" / "problems.jsonl")}


def choose(n):
    d = pd.read_parquet(O / "contents.parquet")
    d = d[(~d["Merged"].astype(bool)) & (~d["Flagged"].astype(bool)) & (d["Available on the hub"].astype(bool))].copy()
    d["base"] = d["Base Model"].fillna(d["fullname"]).astype(str)
    d = d.sort_values("Hub ❤️", ascending=False).drop_duplicates("base").drop_duplicates("fullname")
    d = d[d["MMLU-PRO Raw"].notna()]
    edges = np.quantile(d["MMLU-PRO Raw"], np.linspace(0, 1, n + 1)); pick = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        c = d[(d["MMLU-PRO Raw"] >= lo) & (d["MMLU-PRO Raw"] <= hi) & (~d["fullname"].isin(pick))]
        if len(c):
            pick.append(c.iloc[0]["fullname"])
    return [(f, float(d[d["fullname"] == f]["MMLU-PRO Raw"].iloc[0])) for f in pick]


def fetch(fullname):
    repo = f"open-llm-leaderboard/{fullname.replace('/', '__')}-details"
    tree = requests.get(f"https://huggingface.co/api/datasets/{repo}/tree/main/{fullname.replace('/', '__')}", headers=HDR, timeout=60).json()
    files = sorted(x["path"] for x in tree if isinstance(x, dict) and "samples_leaderboard_mmlu_pro_" in x["path"])
    if not files:
        return None
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{files[-1]}"                    # latest run
    out = {}
    with requests.get(url, stream=True, headers=HDR, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            x = json.loads(line); qid = x.get("doc", {}).get("question_id")
            if qid in ours:
                out[int(qid)] = float(x.get("acc", x.get("metrics", {}).get("acc", np.nan)))
    return out


models = choose(a.n)
json.dump(models, open(O / f"models_{a.n}.json", "w"), indent=1)
print(f"{len(models)} models chosen; MMLU-PRO raw from {models[0][1]:.3f} to {models[-1][1]:.3f}", flush=True)
for f, score in (models[:1] if a.test_one else models):
    dst = O / "outcomes" / f"{f.replace('/', '__')}.json"
    if dst.exists():
        continue
    t0 = time.time()
    try:
        res = fetch(f)
    except Exception as e:
        print(f"{f}: FAILED {type(e).__name__}: {e}"[:200], flush=True); continue
    if res is None:
        print(f"{f}: no MMLU-Pro samples file", flush=True); continue
    json.dump({"model": f, "leaderboard_mmlupro_raw": score, "outcomes": res}, open(dst, "w"))
    print(f"{f}: {len(res)}/{len(ours)} of our questions, acc {np.mean(list(res.values())):.3f} (board {score:.3f}), {time.time()-t0:.0f}s", flush=True)
print("ALL DONE", flush=True)
