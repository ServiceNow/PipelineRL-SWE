"""Re-price the ORIGINAL pools' test-set comparison at billed rates (NEW_PATH 4.A.49). The original pools' headline (learned cost vs
median-length pricing, matched-accuracy band shared with the oracle, as decompose.pool) used the list prices we assumed (MK).
Here every cost -- realized per draw, median-rule prediction, learned prediction (its token forecast backed out at MK) -- is priced
at each model's effective billed rate from the fresh collection (billed.RATE). The original calls carry no usage_cost, so
realized cost = tokens x billed rate. Paired problem bootstrap (500, seed 0, as decompose). First line per pool reproduces MK.
Usage: python reprice_original.py
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from decompose import MK, R, VS, hull, cost_at
from billed import RATE

POOLS = [("LCB", "pool_v2_tensors_5rung", "cost_preds_probe.jsonl"), ("Omni", "omni500_tensors", "cost_preds_probe_thinking.jsonl"),
         ("MMLU-Pro", "mmlupro_tensors", "cost_preds_probe_instruct.jsonl")]
if len(sys.argv) > 1:                       # e.g. "AIME:aime_tensors:cost_preds_probe_instruct.jsonl" (NEW_PATH 4.A.53) -> only that pool
    POOLS = [tuple(a.split(":")) for a in sys.argv[1:]]
fam = lambda s: "oss120" if "120" in s else ("oss20" if s.startswith("oss20") else "dsv4f")
out = {}
for label, name, cfile in POOLS:
    D = R / name; t = np.load(D / "tensors.npz", allow_pickle=True); S = list(map(str, t["model_slots"])); M = len(S)
    pids = list(map(str, t["problem_ids"])); pi = {p: i for i, p in enumerate(pids)}
    v = t["valid"].astype(bool); ok = (t["final_outcome"] & t["valid"]).astype(float)
    pt, ct = t["prompt_tokens"].astype(float), t["completion_tokens"].astype(float); n = v.sum(2); avail = n > 0
    Q = np.where(avail, (ok * v).sum(2) / np.maximum(n, 1), 0)
    sp = json.load(open(D / "split_manifest.json")); tr, te = [np.array([pi[str(p)] for p in sp[f"{k}_problem_ids"]]) for k in ("train", "test")]
    lp = {json.loads(l)["problem_id"]: json.loads(l)["p_successes"][:M] for l in open(D / "content_preds.jsonl")}; P = np.array([lp[p] for p in pids])
    lc = {json.loads(l)["problem_id"]: json.loads(l)["expected_costs"][:M] for l in open(D / cfile)}; LC_mk = np.array([lc[p] for p in pids]) * 100
    inp = np.nanmean(np.where(v, pt, np.nan), 2); inp = np.where(np.isnan(inp), np.nanmean(inp[tr], 0), inp)
    med = np.array([np.median(ct[tr, m][v[tr, m]]) for m in range(M)])
    mk = np.array([MK[s] for s in S]) / 1e6; tok = np.maximum((LC_mk / 100 - inp * mk[:, 0]) / mk[:, 1], 1)
    res = {}
    for scheme, rt in (("list (paper)", mk), ("billed", np.array([RATE[fam(s)] for s in S]))):
        real = (pt * rt[None, :, 0, None] + ct * rt[None, :, 1, None]) * 100
        Cr = np.where(avail, (real * v).sum(2) / np.maximum(n, 1), 1e9)
        PC = (inp * rt[:, 0] + med * rt[:, 1]) * 100
        LC = LC_mk if scheme.startswith("list") else (inp * rt[:, 0] + tok * rt[:, 1]) * 100
        arms = {"paper": PC, "learned": LC, "ORACLE": Cr}
        curves = {}
        for a, C in arms.items():
            ch = np.where(avail[te][None], P[te][None] * VS[:, None, None] - C[te][None], -np.inf).argmax(2)
            curves[a] = (np.take_along_axis(np.broadcast_to(Q[te], (len(VS),) + Q[te].shape), ch[..., None], 2)[..., 0],
                         np.take_along_axis(np.broadcast_to(Cr[te], (len(VS),) + Cr[te].shape), ch[..., None], 2)[..., 0])

        def summary(ii):
            H = {a: hull(list(zip(c[1][:, ii].mean(1), c[0][:, ii].mean(1)))) for a, c in curves.items()}
            lo = max(h[0][1] for h in H.values()); hi = min(h[-1][1] for h in H.values()); T = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 12)
            base = np.array([cost_at(H["paper"], x) for x in T])
            return {a: 1 - float(np.exp(np.nanmean(np.log(np.array([cost_at(H[a], x) for x in T]) / base)))) for a in ("learned", "ORACLE")}
        g = summary(np.arange(len(te))); rng = np.random.default_rng(0); B = [summary(rng.integers(0, len(te), len(te))) for _ in range(500)]
        ci = {a: np.percentile([b[a] for b in B], [2.5, 97.5]) for a in g}
        res[scheme] = dict(learned=[g["learned"], *ci["learned"]], headroom=[g["ORACLE"], *ci["ORACLE"]])
        print(f"{label:<9} {scheme:<13} learned saves {g['learned']*100:5.1f}% [{ci['learned'][0]*100:.1f}, {ci['learned'][1]*100:.1f}]   "
              f"oracle headroom {g['ORACLE']*100:5.1f}% [{ci['ORACLE'][0]*100:.1f}, {ci['ORACLE'][1]*100:.1f}]", flush=True)
    out[label] = res
json.dump(out, open(Path(__file__).parent / ("reprice_original.json" if len(sys.argv) == 1 else f"reprice_{POOLS[0][0].lower()}.json"), "w"), indent=1, default=float)
