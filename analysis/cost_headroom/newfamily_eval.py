"""New routed families on the three main sets (NEW_PATH 4.A.76): Qwen3-32B (SiliconFlow), GLM-4.7-flash (Novita / Cloudflare on the
4.A.48 MMLU-Pro rows), Nemotron-3-super-120B (DekaLLM); one draw each, reasoning on. The five original routes keep the paper's
readouts; each new route gets readouts from the SAME frozen prefill features with the paper's recipes (success: L2 logistic, C by
calibration AUC, Platt on calibration; cost: RidgeCV log length + smearing + train-mean level). Billed prices: new routes' usage_cost
(effective billed $/token fitted on their rows where a call has no cost). Test = problems where all eight routes are valid.
  routing     all8 (5 + 3) and newpool (gpt-oss-20b x2 + the three new families): ours vs median / mean / cost-from-success / difficulty
              bins, and the headroom oracle; paired bootstrap (300)
  onboarding  (in newpool, where the new families carry the top end; and in all8, where they are dominated) hold out ONE new family (a model the pool has never seen; no sibling), onboard it from k = 5/10/20/50 problems (20 draws,
              same anchors in every arm): ours (shared level + offset; logistic in shared difficulty) vs ZeroRouter onboarding (IRT on the
              other 7 routes, D x K chosen on calibration) vs CARROT-style kNN vs naive; gain vs the full 8-route pool at median pricing;
              ours - ZR with a paired bootstrap (resample b <-> draw b mod 20)
  transfer    the new routes' readouts trained on Omni-MATH (500-problem pool, Instruct prefill: the encoder MMLU-Pro uses) and applied
              to MMLU-Pro; old routes in-domain. New routes: in-domain readouts / transferred readouts / transferred success + median
              pricing. AUC and R2 of the transferred new readouts. (The reverse needs Instruct features for the Omni test set: not
              available.)
Usage: REASON_ROOT=.../reason_pinned RESULT_TAG=_pinned OUT_DIR=... python newfamily_eval.py [LCB,Omni,MMLU-Pro]
"""
import glob, json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
_src = open(Path(__file__).parent / "cost_generalization.py").read()
exec(_src.split('MODE, POOL = sys.argv[1], sys.argv[2]')[0])
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from zr_dimsweep import fit_stage1, fit_theta
R0 = Path("/mnt/llmd/results/exps/aristides/reason"); OUT = Path(os.environ.get("OUT_DIR", Path(__file__).parent))
NEWR = ["qw32", "glm47f", "nemo120"]; KS, NDRAW = [5, 10, 20, 50], 20


def files_for(P_, r):
    if P_ == "LCB":
        return sorted(glob.glob(str(R0 / "pool_v2_lcb_second_family" / f"{r}_*_d0.jsonl")))
    if P_ in ("Omni", "Omni500"):
        return [str(R0 / "second_family_20261009" / "omni500" / f"{r}_d0.jsonl")]
    return [str(R0 / ("second_family_20261009" if r == "nemo120" else "second_family_20261002") / "mmlupro" / f"{r}_d0.jsonl")]


def build(P_):
    d = load(P_); N = len(d["ids"]); idx = {p: i for i, p in enumerate(d["ids"])}; M5 = d["M"]
    y = np.full((N, 3), np.nan); Ln = np.full((N, 3), np.nan); In = np.full((N, 3), np.nan); pc = np.full((N, 3), np.nan)
    for j, r in enumerate(NEWR):
        last = {}
        for f in files_for(P_, r):
            for l in open(f):
                if l.strip():
                    x = json.loads(l); last[str(x["problem_id"])] = x
        for pid, x in last.items():
            if pid in idx and x.get("finish_reason") not in ("error", "") and (x.get("completion_tokens") or 0) > 0:
                i = idx[pid]; y[i, j] = float(bool(x.get("resolved"))); Ln[i, j] = x["completion_tokens"]; In[i, j] = x.get("prompt_tokens") or 0
                pc[i, j] = x["usage_cost"] if x.get("usage_cost") is not None else np.nan
    okn = np.isfinite(y)
    rate_n = np.array([np.linalg.lstsq(np.c_[In[m, j], Ln[m, j]], pc[m, j], rcond=None)[0] for j in range(3) for m in [okn[:, j] & np.isfinite(pc[:, j])]])
    pc = np.where(np.isfinite(pc), pc, In * rate_n[:, 0] + Ln * rate_n[:, 1])
    S = dict(d=d, N=N, slots=d["slots"] + NEWR, q=np.c_[d["q"], np.nan_to_num(y)], L=np.c_[d["L"], np.nan_to_num(Ln, nan=1.0)],
             I=np.c_[d["I"], np.nan_to_num(In)], paid=np.c_[d["paid"], np.nan_to_num(pc)], rates=np.r_[d["rates"], rate_n], okn=okn, y=y, Ln=Ln)
    allok = okn.all(1); S["tr"], S["ca"] = d["tr"][allok[d["tr"]]], d["ca"][allok[d["ca"]]]; S["ev"] = d["ev"][allok[d["ev"]]]
    T = view(d, np.arange(N)); tok5 = fit_predict("ours", view(d, d["tr"]), T); X = d["X"]
    P3, tok3 = np.zeros((N, 3)), np.zeros((N, 3))
    for j in range(3):
        trj, caj = d["tr"][okn[d["tr"], j]], d["ca"][okn[d["ca"], j]]
        one = lambda ii: dict(X=X[ii], y0=y[ii, j][:, None].astype(int), succ=y[ii, j][:, None], n=np.ones((len(ii), 1)))
        P3[:, j] = success_linear("X", one(trj), one(caj), dict(X=X))[:, 0]
        tok3[:, j] = fit_predict("ours", dict(X=X[trj], Y=np.log(Ln[trj, j])[:, None]), dict(X=X))[:, 0]
    S["P"] = np.c_[d["P"], P3]; S["tok"] = np.c_[tok5, tok3]; S["X"] = X
    S["succ"] = np.c_[d["succ"], np.nan_to_num(y)]; S["n"] = np.c_[d["n"], okn.astype(float)]
    if len(S["ev"]): print(f"===== {P_}: test {len(S['ev'])} (all 8 valid), train {len(S['tr'])}, cal {len(S['ca'])}; new routes valid "
          + " ".join(f"{r} {okn[:, j].mean():.2f}" for j, r in enumerate(NEWR)) + "; billed out $/M "
          + " ".join(f"{r} {rate_n[j, 1]*1e6:.3f}" for j, r in enumerate(NEWR)) + "; test acc " + " ".join(f"{s} {S['q'][S['ev'], k].mean():.2f}" for k, s in enumerate(S["slots"])), flush=True)
    ev = S["ev"]; Y = np.log(S["L"])
    if len(ev) == 0:                                  # Omni500 (transfer source): its test problems were not collected for new routes
        return S
    print("  new readouts on test: AUC " + " ".join(f"{r} {auc(y[ev, j].astype(int), P3[ev, j]):.3f}" for j, r in enumerate(NEWR))
          + "   log-length R2 " + " ".join(f"{r} {1 - ((Y[ev, 5 + j] - np.log(tok3[ev, j])) ** 2).sum() / ((Y[ev, 5 + j] - Y[ev, 5 + j].mean()) ** 2).sum():+.2f}" for j, r in enumerate(NEWR)), flush=True)
    return S


def curve8(S, Pm, C, ii, cols):
    m = (VALUES[:, None, None] * Pm[ii][:, cols][None] - C[ii][:, cols][None]).argmax(2); r = np.arange(len(ii))[None]
    return hull(list(zip(S["paid"][ii][:, cols][r, m].mean(1), S["q"][ii][:, cols][r, m].mean(1))))


def save8(S, Pa, Ca, Pb, Cb, ii, cols):
    return saved(curve8(S, Pa, Ca, ii, cols), curve8(S, Pb, Cb, ii, cols))


def costs8(S, tk):
    return S["I"] * S["rates"][:, 0] + tk * S["rates"][:, 1]


def routing(S):
    tr, ev, N = S["tr"], S["ev"], S["N"]; Y = np.log(S["L"]); lg = np.log(S["P"] / (1 - S["P"])); out = {}
    for pname, cols in (("all8", list(range(8))), ("newpool", [0, 1, 5, 6, 7])):
        tk = {"ours": S["tok"], "median": np.repeat(np.median(S["L"][tr], 0)[None], N, 0), "mean": np.repeat(S["L"][tr].mean(0)[None], N, 0)}
        FS = np.c_[lg[:, cols], lg[:, cols] ** 2]; fs = np.zeros((N, 8)); bins_ = np.zeros((N, 8))
        dbar = lg[:, cols].mean(1); e = np.quantile(dbar[tr], np.linspace(0, 1, 11)[1:-1]); b = np.searchsorted(e, dbar)
        for k in range(8):
            mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.geomspace(1e-3, 1e4, 15))).fit(FS[tr], Y[tr, k]); yh = mdl.predict(FS)
            fs[:, k] = np.exp(yh) * np.mean(np.exp(Y[tr, k] - yh[tr])); fs[:, k] *= S["L"][tr, k].mean() / fs[tr, k].mean()
            bins_[:, k] = np.array([S["L"][tr][b[tr] == j, k].mean() if (b[tr] == j).any() else S["L"][tr, k].mean() for j in range(10)])[b]
        tk["fromsuccess"], tk["bins"] = fs, bins_
        C = {a: costs8(S, x) for a, x in tk.items()} | {"oracle": S["paid"]}
        rb = np.random.default_rng(0); BS = [ev[rb.integers(0, len(ev), len(ev))] for _ in range(300)]; res = {}
        for a in C:
            vm = save8(S, S["P"], C[a], S["P"], C["median"], ev, cols)
            d_ = [save8(S, S["P"], C["ours"], S["P"], C[a], bb, cols) for bb in BS] if a not in ("ours", "median") else None
            res[a] = dict(vs_median=vm, ours_vs_it=None if d_ is None else [save8(S, S["P"], C["ours"], S["P"], C[a], ev, cols), *np.nanpercentile(d_, [2.5, 97.5]).tolist()])
        out[pname] = res
        print(f"  routing {pname:<7}: " + "  ".join(f"{a} {r['vs_median']*100:+.1f}" for a, r in res.items())
              + "   | ours vs: " + "  ".join(f"{a} {r['ours_vs_it'][0]*100:+.1f} [{r['ours_vs_it'][1]*100:+.1f}, {r['ours_vs_it'][2]*100:+.1f}]" for a, r in res.items() if r["ours_vs_it"]), flush=True)
    return out


def onboarding(S, allc):
    """allc: the pool's columns. In all8 the new families are dominated (removing one changes nothing), so the informative pool is
    newpool, where they carry the top end."""
    tr, ca, ev, N = S["tr"], S["ca"], S["ev"], S["N"]; lg = np.log(S["P"] / (1 - S["P"])); LT = np.log(S["tok"]); M = 8
    C_full = costs8(S, S["tok"]); C_med = costs8(S, np.repeat(np.median(S["L"][tr], 0)[None], N, 0))
    H0 = {nm: curve8(S, S["P"], C_med, ii, allc) for nm, ii in (("te", ev), ("ca", ca))}
    gain = lambda Pm, C, ii, h0: saved(curve8(S, Pm, C, ii, allc), h0)
    pca = PCA(256, random_state=0).fit(S["X"][tr]); Z = pca.transform(S["X"]); Z /= Z[tr].std(0) + 1e-6
    out = {"full": gain(S["P"], C_full, ev, H0["te"])}; rng = np.random.default_rng(0)
    BS = [ev[rng.integers(0, len(ev), len(ev))] for _ in range(200)]; H0b = [curve8(S, S["P"], C_med, ii, allc) for ii in BS]
    for h in range(5, 8):
        others = [m for m in allc if m != h]; lvl, dbar = LT[:, others].mean(1), lg[:, others].mean(1); trh = tr
        put = lambda ph, ch: (np.c_[S["P"][:, :h], np.clip(ph, 1e-4, 1 - 1e-4)[:, None], S["P"][:, h + 1:]], np.c_[C_full[:, :h], ch[:, None], C_full[:, h + 1:]])
        lat = {}
        for D in (1, 5):
            la, bb, th = fit_stage1(S["succ"][tr][:, others], S["n"][tr][:, others], D, 0)
            pr = RidgeCV(alphas=np.geomspace(1, 1e5, 11)).fit(Z[tr], np.c_[la, bb]).predict(Z)
            A, B = np.exp(pr[:, :D]), pr[:, D:]; A[tr], B[tr] = np.exp(la), bb; s = (A * B).sum(1)
            lat[D] = (A, B, {K: np.searchsorted(np.quantile(s[tr], np.linspace(0, 1, K + 1)[1:-1]), s) for K in (5, 10, 20)})
        Lh, Ih, rh = S["L"][:, h], S["I"][:, h], S["rates"][h]
        res_h = {"without": gain(np.c_[S["P"][:, :h], np.full((N, 1), 1e-4), S["P"][:, h + 1:]], np.c_[C_full[:, :h], np.full((N, 1), 1e9), C_full[:, h + 1:]], ev, H0["te"])}
        for k in KS:
            draws = []
            for _ in range(NDRAW):
                an = rng.choice(trh, k, replace=False); yk = S["q"][an, h]; o = {}
                off = np.mean(np.log(Lh[an]) - lvl[an]); sm = np.mean(np.exp(np.log(Lh[an]) - (lvl[an] + off)))
                ph = LogisticRegression(C=1.0).fit(dbar[an][:, None], yk.astype(int)).predict_proba(dbar[:, None])[:, 1] if len(set(yk)) == 2 else np.full(N, yk.mean())
                o["ours"] = put(ph, Ih * rh[0] + np.exp(lvl + off) * sm * rh[1])
                o["naive"] = put(np.full(N, np.clip(yk.mean(), .02, .98)), Ih * rh[0] + np.median(Lh[an]) * rh[1])
                for D, (A, B, bins) in lat.items():
                    tn = fit_theta(A[an], B[an], yk, np.ones(k), D); pz = sig((A * (tn[None] - B)).sum(1))
                    for K, bn in bins.items():
                        tab = np.array([Lh[an][bn[an] == j].mean() if (bn[an] == j).any() else Lh[an].mean() for j in range(K)])
                        o[("zr", D, K)] = put(pz, Ih * rh[0] + tab[bn] * rh[1])
                d2 = (Z[an] ** 2).sum(1)[None] - 2 * Z @ Z[an].T; order = np.argsort(d2, 1)
                for nn in (1, 3, 5, 10):
                    if nn <= k:
                        nb = an[order[:, :nn]]; o[("knn", nn)] = put(np.clip(S["q"][nb, h].mean(1), .02, .98), Ih * rh[0] + Lh[nb].mean(1) * rh[1])
                draws.append(o)
            cz = max([(D, K) for D in (1, 5) for K in (5, 10, 20)], key=lambda c: np.nanmean([gain(*o_[("zr",) + c], ca, H0["ca"]) for o_ in draws]))
            ck = max([nn for nn in (1, 3, 5, 10) if nn <= k], key=lambda c: np.nanmean([gain(*o_[("knn", c)], ca, H0["ca"]) for o_ in draws]))
            arms = {"ours": lambda o_: o_["ours"], "zr": lambda o_: o_[("zr",) + cz], "knn": lambda o_: o_[("knn", ck)], "naive": lambda o_: o_["naive"]}
            pt = {a: np.nanmean([gain(*f(o_), ev, H0["te"]) for o_ in draws]) for a, f in arms.items()}
            bo = np.array([gain(*arms["ours"](draws[b % NDRAW]), ii, H0b[b]) - gain(*arms["zr"](draws[b % NDRAW]), ii, H0b[b]) for b, ii in enumerate(BS)]) * 100
            res_h[str(k)] = dict(pt, ours_minus_zr=[(pt["ours"] - pt["zr"]) * 100, *np.nanpercentile(bo, [2.5, 97.5]).tolist()], zr_cfg=list(cz), knn_nn=ck)
            print(f"  onboard [{len(allc)} routes] {S['slots'][h]:<8} k={k:<3} ours {pt['ours']*100:+5.1f}  zr {pt['zr']*100:+5.1f}  knn {pt['knn']*100:+5.1f}  naive {pt['naive']*100:+5.1f}"
                  f"   ours - zr {res_h[str(k)]['ours_minus_zr'][0]:+5.1f} [{res_h[str(k)]['ours_minus_zr'][1]:+.1f}, {res_h[str(k)]['ours_minus_zr'][2]:+.1f}]"
                  f"   (full {out['full']*100:+.1f}, without it {res_h['without']*100:+.1f})", flush=True)
        out[S["slots"][h]] = res_h
    return out


def transfer(Ssrc, Stg):
    """new routes' readouts trained on the source pool's train split, applied to the target; old routes in-domain on the target"""
    tr, ca, X = Ssrc["d"]["tr"], Ssrc["d"]["ca"], Stg["X"]; ev, N = Stg["ev"], Stg["N"]; P3, tok3, med3 = np.zeros((N, 3)), np.zeros((N, 3)), np.zeros(3)
    for j in range(3):
        trj, caj = tr[Ssrc["okn"][tr, j]], ca[Ssrc["okn"][ca, j]]
        one = lambda ii: dict(X=Ssrc["X"][ii], y0=Ssrc["y"][ii, j][:, None].astype(int), succ=Ssrc["y"][ii, j][:, None], n=np.ones((len(ii), 1)))
        P3[:, j] = success_linear("X", one(trj), one(caj), dict(X=X))[:, 0]
        tok3[:, j] = fit_predict("ours", dict(X=Ssrc["X"][trj], Y=np.log(Ssrc["Ln"][trj, j])[:, None]), dict(X=X))[:, 0]; med3[j] = np.median(Ssrc["Ln"][trj, j])
    allc = list(range(8)); C_in = costs8(Stg, Stg["tok"]); P_x = np.c_[Stg["P"][:, :5], P3]
    C_x = costs8(Stg, np.c_[Stg["tok"][:, :5], tok3]); C_xmed = costs8(Stg, np.c_[Stg["tok"][:, :5], np.repeat(med3[None], N, 0)])
    C_med = costs8(Stg, np.repeat(np.median(Stg["L"][Stg["tr"]], 0)[None], N, 0)); H0 = curve8(Stg, Stg["P"], C_med, ev, allc)
    res = {"in_domain": saved(curve8(Stg, Stg["P"], C_in, ev, allc), H0), "transferred": saved(curve8(Stg, P_x, C_x, ev, allc), H0),
           "transferred_success_median_cost": saved(curve8(Stg, P_x, C_xmed, ev, allc), H0),
           "auc_transferred": [auc(Stg["y"][ev, j].astype(int), P3[ev, j]) for j in range(3)], "auc_in": [auc(Stg["y"][ev, j].astype(int), Stg["P"][ev, 5 + j]) for j in range(3)],
           "r2_transferred": [r2_routes(dict(L=Stg["Ln"][:, [j]]), tok3[:, [j]], ev)[0] for j in range(3)],
           "r2_in": [r2_routes(dict(L=Stg["Ln"][:, [j]]), Stg["tok"][:, [5 + j]], ev)[0] for j in range(3)]}
    print(f"  transfer of the new routes' readouts Omni500 -> MMLU-Pro (all 8 routes, saving vs median): in-domain {res['in_domain']*100:+.1f}  "
          f"transferred {res['transferred']*100:+.1f}  transferred success + source median {res['transferred_success_median_cost']*100:+.1f}; "
          f"AUC in/xfer " + " ".join(f"{a:.3f}/{b:.3f}" for a, b in zip(res["auc_in"], res["auc_transferred"]))
          + "  R2 in/xfer " + " ".join(f"{a:+.2f}/{b:+.2f}" for a, b in zip(res["r2_in"], res["r2_transferred"])), flush=True)
    return res


res = {}; built = {}
for P_ in (sys.argv[1] if len(sys.argv) > 1 else "LCB,Omni,MMLU-Pro").split(","):
    S = build(P_); built[P_] = S; res[P_] = dict(n_test=int(len(S["ev"])), routing=routing(S), onboarding_newpool=onboarding(S, [0, 1, 5, 6, 7]), onboarding_all8=onboarding(S, list(range(8))))
    json.dump(res, open(OUT / f"newfamily_eval{os.environ.get('RESULT_TAG', '')}.json", "w"), indent=1, default=float)
if "MMLU-Pro" in built:
    res["transfer_omni500_to_mmlupro"] = transfer(build("Omni500"), built["MMLU-Pro"])
json.dump(res, open(OUT / f"newfamily_eval{os.environ.get('RESULT_TAG', '')}.json", "w"), indent=1, default=float)
print("DONE", flush=True)
