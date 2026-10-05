"""Free preview: does removing provider variability (deepseek-v4-flash PINNED to StreamLake) help or hurt our edge? (NEW_PATH 4.A.58)
Uses the StreamLake-pinned dsv4f draws already collected in the provider pilot (provider_pilot_20261002). No API spend.

APPS (all 1,000; train/cal/test as apps_tensors): two worlds that differ ONLY in the dsv4f column --
  unpinned = dsv4f as collected (OpenRouter's provider mix); pinned = dsv4f pinned to StreamLake (same problems, same settings).
  Each world refits, with identical code, (a) success readouts: activation_content_preds.py --rich --select-C (paper command);
  (b) our cost readouts: StandardScaler -> RidgeCV(1e1..1e7) on log output, smearing, level match (paper recipe);
  (c) cost from success: RidgeCV of log output on the five success logits, same post-processing; (d) training-median length.
  Realized cost = realized tokens x a FIXED per-model rate (billed.py RATE), so provider PRICE variation is removed in both worlds.
  Price scenario A: dsv4f at the pooled billed rate in both worlds (isolates label/length noise). Scenario B: pinned world at
  StreamLake's own fitted billed rate (what pinning would really cost).
MMLU-Pro (2,000 pinned fresh problems; test-side only, readouts frozen from the original unpinned train): realized dsv4f outcome
  unpinned vs pinned; plus both arms re-offset for dsv4f from the 300 FIT problems (ours: ratio of means; median: FIT median).
Metrics: cost saved by ours at matched accuracy vs median / vs cost-from-success (12 targets, interior 90% of the shared band),
headroom (oracle per-problem cost vs median), dsv4f test log-length R2 and success AUC; paired problem bootstrap (300).
Usage: python pin_preview.py
"""
import json, subprocess, sys, shutil
from pathlib import Path
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).parent))
from carrot_compare import read_predictions
from decompose import MK, R, hull, cost_at
from billed import RATE
from baseline_cost_heads import rich

REPO = Path(__file__).resolve().parents[2]; OUT = R / "pin_preview_20261005"; OUT.mkdir(exist_ok=True)
PIL = R / "provider_pilot_20261002"; PROV = "StreamLake"; VALUES = np.geomspace(1e-7, 1, 300); NB = 300
fam = lambda s: "oss120" if "120" in s else ("oss20" if s.startswith("oss20") else "dsv4f")


def pinned_rows(ds):
    rows = {}
    for l in open(PIL / ds / f"dsv4f_pin_{PROV}.jsonl"):
        r = json.loads(l)
        if r.get("finish_reason") != "error" and r.get("completion_tokens"):
            rows[r["problem_id"]] = r
    return rows


def own_rate(rows):
    X = np.array([[r["prompt_tokens"], r["completion_tokens"]] for r in rows.values() if r.get("usage_cost") is not None], float)
    y = np.array([r["usage_cost"] for r in rows.values() if r.get("usage_cost") is not None])
    return np.linalg.lstsq(X, y, rcond=None)[0]


def front(P, C, q, paid, ii):
    pts = []
    for V in VALUES:
        m = (V * P[ii] - C[ii]).argmax(1); k = np.arange(len(ii)); pts.append((paid[ii][k, m].mean(), q[ii][k, m].mean()))
    return hull(pts)


def saved(a, b, q, paid, ii):
    Ha, Hb = front(*a, q, paid, ii), front(*b, q, paid, ii); lo, hi = max(Ha[0][1], Hb[0][1]), min(Ha[-1][1], Hb[-1][1])
    if hi <= lo:
        return np.nan
    T = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 12)
    return 1 - float(np.exp(np.nanmean(np.log([cost_at(Ha, x) / cost_at(Hb, x) for x in T]))))


def ci(fn, ii, rng):
    g = fn(ii); bs = [fn(ii[rng.integers(0, len(ii), len(ii))]) for _ in range(NB)]
    return [g, *np.nanpercentile(bs, [2.5, 97.5])]


def fmt(x):
    return f"{x[0]*100:+6.1f} [{x[1]*100:+.1f}, {x[2]*100:+.1f}]"


def post(yh, Y, L, tr):
    sm = np.mean(np.exp(Y[tr] - yh[tr])); tok = np.exp(yh) * sm
    return tok * L[tr].mean() / tok[tr].mean()


out = {}
# ------------------------------------------------------------------ APPS: refit everything in both worlds
T0 = R / "apps_tensors"; t = np.load(T0 / "tensors.npz", allow_pickle=True)
ids = list(map(str, t["problem_ids"])); slots = list(map(str, t["model_slots"])); j = slots.index("dsv4f"); M = len(slots)
pins = pinned_rows("apps"); keep = np.array([p in pins for p in ids]); print(f"APPS: {keep.sum()}/{len(ids)} problems have a pinned {PROV} draw")
sp = json.loads((T0 / "split_manifest.json").read_text()); idx = {p: i for i, p in enumerate(ids)}
tr_all, te_all = [np.array([idx[str(p)] for p in sp[k]]) for k in ("train_problem_ids", "test_problem_ids")]
tr, te = tr_all[keep[tr_all]], te_all[keep[te_all]]
X = rich(R / "apps_probe" / "instruct.npz", ids); sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X); del X
rate_pin = own_rate(pins); print(f"  {PROV} fitted billed rate: in {rate_pin[0]*1e6:.3f} / out {rate_pin[1]*1e6:.3f} $/M (pooled dsv4f {RATE['dsv4f'][1]*1e6:.3f})")
worlds = {}
for world in ("unpinned", "pinned"):
    arr = {k: t[k].copy() for k in t.files}
    if world == "pinned":
        for p, i in idx.items():
            if p in pins:
                r = pins[p]; arr["final_outcome"][i, j, 0] = bool(r["resolved"]); arr["valid"][i, j, 0] = True
                arr["completion_tokens"][i, j, 0] = r["completion_tokens"]; arr["prompt_tokens"][i, j, 0] = r["prompt_tokens"]
                if "execution_outcome" in arr:
                    arr["execution_outcome"][i, j, 0] = bool(r["resolved"])
    D = OUT / f"apps_{world}"; D.mkdir(exist_ok=True); np.savez(D / "tensors.npz", **arr)
    for f in ("split_manifest.json", "problems.jsonl"):
        shutil.copy(T0 / f, D / f)
    spf = D / "success_preds.jsonl"
    if not spf.exists():
        subprocess.run([sys.executable, str(REPO / "pipelinerl/swe/scripts/livecodebench/activation_content_preds.py"), "--activations",
                        str(R / "apps_probe" / "instruct.npz"), "--rich", "--tensors-dir", str(D), "--select-C", "--out", str(spf)],
                       check=True, cwd=REPO, stdout=subprocess.DEVNULL)
    P = np.clip(read_predictions(spf, ids, "p_successes", M), 1e-4, 1 - 1e-4)
    v = arr["valid"].astype(bool); q = np.where(v, arr["final_outcome"], 0).sum(2) / np.maximum(v.sum(2), 1)
    L = np.maximum(np.where(v, arr["completion_tokens"], 0).sum(2) / np.maximum(v.sum(2), 1), 1); I = np.where(v, arr["prompt_tokens"], 0).sum(2) / np.maximum(v.sum(2), 1)
    Y = np.log(L); lg = np.log(P / (1 - P))
    tok = np.stack([post(RidgeCV(alphas=np.geomspace(1e1, 1e7, 13)).fit(Xs[tr], Y[tr, k]).predict(Xs), Y[:, k], L[:, k], tr) for k in range(M)], 1)
    tfs = np.stack([post(RidgeCV(alphas=np.geomspace(1e-3, 1e3, 13)).fit(lg[tr], Y[tr, k]).predict(lg), Y[:, k], L[:, k], tr) for k in range(M)], 1)
    med = np.array([np.median(L[tr, k]) for k in range(M)])
    r2 = 1 - ((Y[te, j] - np.log(tok[te, j])) ** 2).sum() / ((Y[te, j] - Y[te, j].mean()) ** 2).sum()
    auc = roc_auc_score(q[te, j] > .5, P[te, j]) if 0 < (q[te, j] > .5).mean() < 1 else np.nan
    worlds[world] = dict(P=P, q=q, L=L, I=I, tok=tok, tfs=tfs, med=med, r2=float(r2), auc=float(auc), acc=float(q[te, j].mean()))
    print(f"  {world}: dsv4f test acc {q[te, j].mean():.3f}, mean out {L[te, j].mean():.0f}, log-length R2 {r2:.3f}, success AUC {auc:.3f}", flush=True)
res = {}
for scen in ("A_same_price", "B_own_price"):
    print(f"\n  APPS scenario {scen}: cost saved by ours at matched accuracy [95% CI], test n={len(te)}")
    res[scen] = {}
    for world, w in worlds.items():
        rates = np.array([RATE[fam(s)] for s in slots]);
        if world == "pinned" and scen == "B_own_price":
            rates[j] = rate_pin
        paid = w["I"] * rates[:, 0] + w["L"] * rates[:, 1]
        cst = lambda tk: w["I"] * rates[:, 0] + tk * rates[:, 1]
        ours, medn, fs, orc = (w["P"], cst(w["tok"])), (w["P"], cst(w["med"][None])), (w["P"], cst(w["tfs"])), (w["P"], paid)
        rng = np.random.default_rng(0)
        row = dict(vs_median=ci(lambda ii: saved(ours, medn, w["q"], paid, ii), te, rng),
                   vs_from_success=ci(lambda ii: saved(ours, fs, w["q"], paid, ii), te, rng),
                   headroom=ci(lambda ii: saved(orc, medn, w["q"], paid, ii), te, rng),
                   dsv4f_r2=w["r2"], dsv4f_auc=w["auc"], dsv4f_acc=w["acc"])
        res[scen][world] = row
        print(f"    {world:<9} ours vs median {fmt(row['vs_median'])}  vs cost-from-success {fmt(row['vs_from_success'])}  headroom {fmt(row['headroom'])}", flush=True)
out["APPS"] = res

# ------------------------------------------------------------------ MMLU-Pro: test-side only (readouts frozen, trained unpinned)
F = R / "expanded_eval_20261001" / "mmlupro"; t = np.load(F / "tensors.npz", allow_pickle=True)
ids = list(map(str, t["problem_ids"])); slots = list(map(str, t["model_slots"])); j = slots.index("dsv4f"); M = len(slots); idx = {p: i for i, p in enumerate(ids)}
old = R / "mmlupro_tensors"; spo = json.loads((old / "split_manifest.json").read_text()); tr = np.array([idx[str(p)] for p in spo["train_problem_ids"]])
pins = pinned_rows("mmlupro"); cand = [p for p in json.load(open(PIL / "mmlupro" / "problem_ids.json")) if p in pins and p in idx]
v = t["valid"].astype(bool); ok = (v[:, :, 0]).all(1)
cand = np.array([idx[p] for p in cand if ok[idx[p]]]); rng0 = np.random.default_rng(0); perm = rng0.permutation(len(cand)); fit, ev = cand[perm[:300]], cand[perm[300:]]
q0 = t["final_outcome"][:, :, 0].astype(float); L0 = np.maximum(t["completion_tokens"][:, :, 0], 1).astype(float); I0 = t["prompt_tokens"][:, :, 0].astype(float)
P = np.clip(read_predictions(F / "success_preds.jsonl", ids, "p_successes", M), 1e-4, 1 - 1e-4)
learned = read_predictions(F / "paper_cost_preds.jsonl", ids, "expected_costs", M)
asg = np.array([[MK[s][0] / 1e6, MK[s][1] / 1e6] for s in slots]); Ihat = np.where(v, t["prompt_tokens"], 0).sum(2) / np.maximum(v.sum(2), 1)
tok = np.maximum((learned - Ihat * asg[:, 0]) / asg[:, 1], 1)
Lt = np.where(v, t["completion_tokens"], 0); med = np.array([np.median(t["completion_tokens"][tr, k][v[tr, k]]) for k in range(M)])
rate_pin = own_rate(pins); print(f"\nMMLU-Pro: {len(cand)} pinned fresh problems (fit {len(fit)}, eval {len(ev)}); {PROV} out {rate_pin[1]*1e6:.3f} $/M")
res = {}
for scen in ("A_same_price", "B_own_price"):
    print(f"  MMLU-Pro scenario {scen}: cost saved by ours at matched accuracy [95% CI], eval n={len(ev)}")
    res[scen] = {}
    for world in ("unpinned", "pinned", "pinned_reoffset"):
        q, L, I = q0.copy(), L0.copy(), I0.copy()
        if world != "unpinned":
            for i in cand:
                r = pins[ids[i]]; q[i, j] = float(bool(r["resolved"])); L[i, j] = r["completion_tokens"]; I[i, j] = r["prompt_tokens"]
        rates = np.array([RATE[fam(s)] for s in slots])
        if world != "unpinned" and scen == "B_own_price":
            rates[j] = rate_pin
        tk, md = tok.copy(), med.copy()
        if world == "pinned_reoffset":                       # both arms get the same 300 FIT calls
            tk[:, j] *= L[fit, j].mean() / tok[fit, j].mean(); md[j] = np.median(L[fit, j])
        paid = I * rates[:, 0] + L * rates[:, 1]
        ours, medn, orc = (P, I * rates[:, 0] + tk * rates[:, 1]), (P, I * rates[:, 0] + md[None] * rates[:, 1]), (P, paid)
        rng = np.random.default_rng(0)
        r2 = 1 - ((np.log(L[ev, j]) - np.log(tk[ev, j])) ** 2).sum() / ((np.log(L[ev, j]) - np.log(L[ev, j]).mean()) ** 2).sum()
        row = dict(vs_median=ci(lambda ii: saved(ours, medn, q, paid, ii), ev, rng), headroom=ci(lambda ii: saved(orc, medn, q, paid, ii), ev, rng),
                   dsv4f_r2=float(r2), dsv4f_acc=float(q[ev, j].mean()), dsv4f_pred_over_real=float(tk[ev, j].mean() / L[ev, j].mean()))
        res[scen][world] = row
        print(f"    {world:<16} ours vs median {fmt(row['vs_median'])}  headroom {fmt(row['headroom'])}  dsv4f acc {row['dsv4f_acc']:.3f} "
              f"R2 {r2:.3f} pred/real {row['dsv4f_pred_over_real']:.2f}", flush=True)
out["MMLU-Pro"] = res
json.dump(out, open(Path(__file__).parent / "pin_preview.json", "w"), indent=1, default=float)
