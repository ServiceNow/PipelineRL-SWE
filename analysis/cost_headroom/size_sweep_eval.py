"""Prefill-size sweep (NEW_PATH 4.A.50): how small can the prefill model be? Each model's prefill feeds BOTH readouts, so every row
is a complete router at that size. Same family (Qwen3), same prompts (original solving prompts + system prompt), same 8 relative
layers x {mean, last}; readouts fitted on ORIGINAL train (success: + calibration C selection) with the paper's recipes:
  success   activation_content_preds.py --rich --select-C (binomial logistic + its calibration)
  cost      reconstruct_paper_cost_heads.py recipe: StandardScaler -> RidgeCV(1e1..1e7) on log mean output, smearing, level match
Rows: Qwen3-0.6B, 1.7B, 4B (hybrid; anchor for the size trend), 8B (all hybrid thinking-mode releases); off-family Phi-4-mini-instruct
(3.8B, Microsoft), Granite-3.3-2B-instruct (2.5B, IBM), SmolLM2-1.7B-Instruct (HF); plus the paper's
Qwen3-4B-2507 (Instruct on MMLU-Pro, Thinking on Omni) refitted through this same code as a check.
Evaluated on the FRESH problems at billed prices (realized = usage_cost):
  fresh success AUC (mean over routes) and log loss; fresh log-length R2 (mean over routes)
  cost saved at matched accuracy by the row's router vs (a) median pricing with the SAME row's success, (b) the paper's 4B-2507 router;
  (b) also with each row's encoder pass priced (ENC_RATE $/M input tokens, scaled with parameters from the 4B's 0.03 upper bound).
Paired problem bootstrap (300). Usage: python size_sweep_eval.py
"""
import glob, json, os, subprocess, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from carrot_compare import read_predictions
from decompose import R, hull, cost_at
from baseline_cost_heads import rich
from billed import RATE

S = R / "prefill_size_20261005"; REPO = Path(__file__).resolve().parents[2]
ROWS = {"q06b": ("Qwen3-0.6B", 0.6), "q17b": ("Qwen3-1.7B", 1.7), "q4bhyb": ("Qwen3-4B", 4.0), "q8b": ("Qwen3-8B", 8.0),
        "phi4mini": ("Phi-4-mini-instruct", 3.8), "granite2b": ("Granite-3.3-2B-instruct", 2.5), "smol17b": ("SmolLM2-1.7B-Instruct", 1.7),
        "paper": ("Qwen3-4B-2507 (paper)", 4.0)}
ENC_RATE = {k: 0.03 * b / 4.0 for k, (_, b) in ROWS.items()}
VALUES = np.geomspace(1e-7, 1, 300)
rate_of = lambda s: RATE["oss120" if "120" in s else ("oss20" if s.startswith("oss20") else "dsv4f")]
only = sys.argv[1].split(",") if len(sys.argv) > 1 else list(ROWS)
out = {}
for ds in ("omni500", "mmlupro"):
    label = "MMLU-Pro" if ds == "mmlupro" else "Omni"; F = R / "expanded_eval_20261001" / ds
    t = np.load(F / "tensors.npz", allow_pickle=True); ids, slots = list(map(str, t["problem_ids"])), list(map(str, t["model_slots"]))
    idx = {p: i for i, p in enumerate(ids)}; M = len(slots)
    sp = json.loads((F / "split_manifest.json").read_text()); tr = np.array([idx[str(p)] for p in sp["train_problem_ids"]])
    old = R / ("mmlupro_tensors" if ds == "mmlupro" else "omni500_tensors"); n_old = len(np.load(old / "tensors.npz", allow_pickle=True)["problem_ids"])
    v = t["valid"].astype(bool); cnt = np.maximum(v.sum(2), 1)
    q = np.where(v, t["final_outcome"], 0).sum(2) / cnt; L = np.where(v, t["completion_tokens"], 0).sum(2) / cnt; I = np.where(v, t["prompt_tokens"], 0).sum(2) / cnt
    rates = np.array([rate_of(s) for s in slots]); paid = I * rates[:, 0] + L * rates[:, 1]
    for f in glob.glob(f"{R}/math_expand_20261001/{ds}/*_d0.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if r.get("finish_reason") != "error" and r.get("usage_cost") is not None and r["problem_id"] in idx and r["route_label"] in slots:
                paid[idx[r["problem_id"]], slots.index(r["route_label"])] = r["usage_cost"]
    fr = np.arange(n_old, len(ids)); fr = fr[(v[fr].sum(2) > 0).all(1)]
    Y = np.log(np.maximum(L, 1)); y1 = (t["final_outcome"][:, :, 0] & t["valid"][:, :, 0]).astype(int)
    med = np.array([np.median(t["completion_tokens"][tr, k][v[tr, k]]) for k in range(M)])
    router = {}
    for tag in only:
        feat = F / "prefill_combined.npz" if tag == "paper" else S / tag / f"{ds}.npz"
        if not feat.exists():
            print(f"  {label}: {tag} features missing, skipped"); continue
        sp_out = S / f"readouts{os.environ.get('RESULT_TAG', '')}" / f"{ds}_{tag}_success.jsonl"; sp_out.parent.mkdir(parents=True, exist_ok=True)   # tagged cache: pinned labels never reuse unpinned readouts
        if not sp_out.exists():
            subprocess.run([sys.executable, str(REPO / "pipelinerl/swe/scripts/livecodebench/activation_content_preds.py"), "--activations", str(feat),
                            "--rich", "--tensors-dir", str(F), "--select-C", "--out", str(sp_out)], check=True, cwd=REPO, stdout=subprocess.DEVNULL)
        P = np.clip(read_predictions(sp_out, ids, "p_successes", M), 1e-4, 1 - 1e-4)
        X = rich(feat, ids); sc = StandardScaler().fit(X[tr]); Xs = sc.transform(X); del X
        tok = np.zeros_like(L)
        for k in range(M):
            m = RidgeCV(alphas=np.geomspace(1e1, 1e7, 13)).fit(Xs[tr], Y[tr, k]); yh = m.predict(Xs)
            sm = np.mean(np.exp(Y[tr, k] - yh[tr])); tok[:, k] = np.exp(yh) * sm * L[tr, k].mean() / (np.exp(yh[tr]) * sm).mean()
        del Xs
        r2 = np.mean([1 - ((Y[fr, k] - np.log(tok[fr, k])) ** 2).sum() / ((Y[fr, k] - Y[fr, k].mean()) ** 2).sum() for k in range(M)])
        aucs = np.mean([roc_auc_score(y1[fr, k], P[fr, k]) for k in range(M)])
        ll = -np.mean(y1[fr] * np.log(P[fr]) + (1 - y1[fr]) * np.log(1 - P[fr]))
        router[tag] = dict(P=P, C=I * rates[:, 0] + tok * rates[:, 1], Cmed=I * rates[:, 0] + med[None] * rates[:, 1], auc=aucs, ll=ll, r2=r2)
        print(f"  {label}: fitted {tag}: fresh success AUC {aucs:.3f}, log loss {ll:.3f}, log-length R2 {r2:.3f}", flush=True)
    enc_in = I.mean(1)

    def front(P, C, ii, enc=0.0):
        pts = []
        for V in VALUES:
            m = (V * P[ii] - C[ii]).argmax(1); pts.append((paid[ii][np.arange(len(ii)), m].mean() + enc, q[ii][np.arange(len(ii)), m].mean()))
        return hull(pts)

    def saved(a, b, ii, ea=0.0, eb=0.0):
        Ha, Hb = front(*a, ii, ea), front(*b, ii, eb); lo, hi = max(Ha[0][1], Hb[0][1]), min(Ha[-1][1], Hb[-1][1])
        T = np.linspace(lo + .05 * (hi - lo), hi - .05 * (hi - lo), 12)
        return 1 - float(np.exp(np.nanmean(np.log([cost_at(Ha, x) / cost_at(Hb, x) for x in T]))))
    rng = np.random.default_rng(0); BS = [fr[rng.integers(0, len(fr), len(fr))] for _ in range(300)]
    res = {}
    print(f"\n===== {label} fresh n={len(fr)}, billed prices; cost saved at matched accuracy [95% CI]")
    for tag, rt in router.items():
        own, ref = (rt["P"], rt["C"]), (rt["P"], rt["Cmed"])
        g_med = saved(own, ref, fr); b_med = [saved(own, ref, bb) for bb in BS]
        row = dict(auc=rt["auc"], logloss=rt["ll"], r2=rt["r2"], max_acc=front(*own, fr)[-1][1], vs_median=[g_med, *np.percentile(b_med, [2.5, 97.5])])
        if "paper" in router and tag != "paper":
            pp = (router["paper"]["P"], router["paper"]["C"])
            g = saved(own, pp, fr); b = [saved(own, pp, bb) for bb in BS]
            e_own, e_pp = ENC_RATE[tag] * enc_in[fr].mean() / 1e6, ENC_RATE["paper"] * enc_in[fr].mean() / 1e6
            row["vs_paper"] = [g, *np.percentile(b, [2.5, 97.5])]; row["vs_paper_enc"] = saved(own, pp, fr, e_own, e_pp)
        res[tag] = row
        vp = (f" | vs paper 4B router {row['vs_paper'][0]*100:+6.1f}% [{row['vs_paper'][1]*100:+.1f}, {row['vs_paper'][2]*100:+.1f}] (enc priced {row['vs_paper_enc']*100:+.1f}%)"
              if "vs_paper" in row else "")
        print(f"  {ROWS[tag][0]:<22} AUC {rt['auc']:.3f} R2 {rt['r2']:.2f} max acc {row['max_acc']*100:.1f}% | vs median (own success) "
              f"{g_med*100:+6.1f}% [{np.percentile(b_med,2.5)*100:+.1f}, {np.percentile(b_med,97.5)*100:+.1f}]{vp}", flush=True)
    out[label] = res
json.dump(out, open(Path(__file__).parent / f"size_sweep_eval{os.environ.get('RESULT_TAG', '')}.json", "w"), indent=1, default=float)
