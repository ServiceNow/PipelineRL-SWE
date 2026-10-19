"""ZeroRouter as published (arXiv 2601.06220) on the 10 training sets, next to the paper's strengthened stand-in (NEW_PATH 4.A.118).
Both share stage 1 (their Eq. 1: D-dim 2PL IRT on the training outcomes, MAP; zr_dimsweep.fit_stage1), their own success model
(sigmoid(alpha^T (theta - b))) and their bin pricing (Eqs. 8-10: K quantile bins of s = alpha^T b on the training queries, per-model
mean training length per bin); training queries keep their stage-1 positions. They differ only in stage 2, the reader that places a
NEW query in the latent:
  zr_pub     their encoder: fine-tuned distilbert-base-uncased [CLS] + 11 linguistic features -> fusion trunk -> heads for
             (log alpha, b) (zr_encoder.Net), their settings (40 epochs, batch 32, AdamW lr 3e-5, max 512 tokens), MSE onto the
             standardized stage-1 targets; epoch chosen on a held-out 10% of the training queries (calibration queries have no stage-1
             targets)
  zr_4b      the paper's stand-in: frozen Qwen3-4B prefill -> PCA(256) -> ridge (tmlr_strengthen.zr_paper)
(D, K) in {1, 5} x {5, 10, 20} chosen for each by the saving vs median on the calibration problems, as in tmlr_strengthen.
Also zr_pub-pricing / zr_4b-pricing: their bin pricing with OUR success probes (the tab:allest-style cost-only comparison).
Per training set (DRAW=s; success readouts refitted as tmlr_strengthen.py `draws`): ours, the four ZeroRouter arms, oracle; saving vs
median and ours vs the arm, point + NBOOT test resamples (rng 0, the same in every set), for draws_aggregate-style pooling.
Usage: (GPU) REASON_ROOT=.../reason_pinned RESULT_TAG=_pinned NBOOT=300 OUT_DIR=... python zr_faithful_draws.py POOL S0 S1
"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
POOL_, S0, S1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); sys.argv = [sys.argv[0], "zrpub", POOL_]
exec(open(Path(__file__).parent / "cost_generalization.py").read().split('MODE, POOL = sys.argv[1], sys.argv[2]')[0])
_ts = open(Path(__file__).parent / "tmlr_strengthen.py").read()
exec(_ts[_ts.index("def zr_paper"):_ts.index('if MODE_ == "deploy"')])                  # zr_paper (the stand-in), unchanged
_ze = open(Path(__file__).parent / "zr_encoder.py").read()
exec(_ze[_ze.index("def syllables"):_ze.index("assert torch.cuda.is_available()")])      # feats, Net: their encoder, unchanged
from transformers import AutoTokenizer
assert torch.cuda.is_available(), "GPU job"
OUT = Path(os.environ.get("OUT_DIR", Path(__file__).parent)); TAG = os.environ.get("RESULT_TAG", "")
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def texts_of(pool):
    name, _, ds = POOLS[pool]; F = R / "expanded_eval_20261001" / ds if ds else R / name
    return [str(json.loads(l)["problem_statement"]) for l in open(F / "problems.jsonl")]


def stage2_pub(enc, Fz, tr, la, b, D, seed=0):
    """their stage-2 encoder fitted on the training queries' stage-1 positions; returns (log alpha, b) for every query"""
    torch.manual_seed(seed); N = len(Fz); tgt = np.c_[la, b].astype(np.float32); mu, sd = tgt.mean(0), tgt.std(0) + 1e-6
    T = torch.tensor((tgt - mu) / sd).cuda(); Ft = torch.tensor(Fz).cuda(); row = {int(i): j for j, i in enumerate(tr)}
    net = Net(D, b.mean(0)).cuda(); opt = torch.optim.AdamW(net.parameters(), lr=3e-5)
    rng = np.random.default_rng(seed); perm = rng.permutation(tr); hold, fit_ = perm[: len(tr) // 10], perm[len(tr) // 10:]

    def predict(ii):
        net.eval(); out = []
        with torch.no_grad():
            for j in range(0, len(ii), 128):
                s = ii[j:j + 128]; a_, b_ = net(enc["input_ids"][s].cuda(), enc["attention_mask"][s].cuda(), Ft[s]); out.append(torch.cat([a_, b_], 1).cpu().numpy())
        net.train(); return np.concatenate(out) * sd + mu
    hz = np.array([row[int(i)] for i in hold]); best = (np.inf, None, -1)
    for ep in range(40):
        order = rng.permutation(fit_)
        for j in range(0, len(order), 32):
            s = order[j:j + 32]; a_, b_ = net(enc["input_ids"][s].cuda(), enc["attention_mask"][s].cuda(), Ft[s])
            tz = T[[row[int(i)] for i in s]]; loss = ((torch.cat([a_, b_], 1) - tz) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        hm = float((((predict(hold) - tgt[hz]) / sd) ** 2).mean())
        if hm < best[0]:
            best = (hm, {k: v.detach().clone() for k, v in net.state_dict().items()}, ep)
    net.load_state_dict(best[1]); pr = predict(np.arange(N))
    return pr[:, :D], pr[:, D:], best[2]


def zr_from_positions(d, ca, H0, LA, B, th, D, tr, la, bb):
    """their success model and bin pricing from latent positions (training queries at their stage-1 positions); K by calibration"""
    A = np.exp(LA).copy(); Bq = B.copy(); A[tr] = np.exp(la); Bq[tr] = bb
    Pz = np.clip(sig((A[:, None, :] * (th[None] - Bq[:, None, :])).sum(-1)), 1e-4, 1 - 1e-4); sc = (A * Bq).sum(1); Ltr = d["L"][tr]; out = []
    for K in (5, 10, 20):
        e_ = np.quantile(sc[tr], np.linspace(0, 1, K + 1)[1:-1]); bn = np.searchsorted(e_, sc)
        tab = np.array([[Ltr[bn[tr] == j, k].mean() if (bn[tr] == j).any() else Ltr[:, k].mean() for j in range(K)] for k in range(d["M"])])
        out.append((saved(curve(d, Pz, cost_of(d, tab.T[bn]), ca), H0), Pz, tab.T[bn], (D, K)))
    return out


texts = texts_of(POOL_); enc = tok(texts, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
Fraw = np.array([feats(x) for x in texts], np.float32)
res = {"pool": POOL_, "seeds": {}}
for s in range(S0, S1):
    os.environ["DRAW"] = str(s); d = load(POOL_); N = len(d["ids"]); tr, ca, ev = d["tr"], d["ca"], d["ev"]; assert len(texts) == N
    S, T = view(d, tr), view(d, np.arange(N)); Fz = (Fraw - Fraw[tr].mean(0)) / (Fraw[tr].std(0) + 1e-6)
    H0c = curve(d, d["P"], cost_of(d, np.repeat(d["med"][None], N, 0)), ca); cands = []; eps = {}
    for D in (1, 5):
        la, bb, th = fit_stage1(S["succ"], S["n"], D, 0); LA, B, eps[D] = stage2_pub(enc, Fz, tr, la, bb, D)
        cands += zr_from_positions(d, ca, H0c, LA, B, th, D, tr, la, bb)
    best = max(cands, key=lambda c: -np.inf if np.isnan(c[0]) else c[0]); Ppub, tkpub, cfg_pub = best[1], best[2], best[3]
    P4b, tk4b, cfg_4b = zr_paper(d, S)
    tk = {"ours": fit_predict("ours", S, T), "zr_pub": tkpub, "zr_pub-pricing": tkpub, "zr_4b": tk4b, "zr_4b-pricing": tk4b, "oracle": d["L"]}
    Ps = {a: d["P"] for a in tk}; Ps["zr_pub"] = Ppub; Ps["zr_4b"] = P4b
    C = {a: cost_of(d, x) for a, x in tk.items()}; Cm = cost_of(d, np.repeat(d["med"][None], N, 0))
    rb = np.random.default_rng(0); BS = [ev] + [ev[rb.integers(0, len(ev), len(ev))] for _ in range(NBOOT)]
    Hs = [{a: curve(d, Ps[a], C[a], ii) for a in C} | {"median": curve(d, d["P"], Cm, ii)} for ii in BS]
    r_ = {}
    for a in C:
        vm = np.array([saved(H[a], H["median"]) for H in Hs]) * 100; vd = np.array([saved(H["ours"], H[a]) for H in Hs]) * 100
        r_[a] = dict(vs_median=float(vm[0]), vs_median_bs=vm[1:].tolist(), ours_vs_it=float(vd[0]), ours_vs_it_bs=vd[1:].tolist(),
                     auc=[float(auc(d["y0"][ev, m], Ps[a][ev, m])) for m in range(d["M"])])
    r_["_cfg"] = dict(zr_pub=list(cfg_pub), zr_4b=list(cfg_4b), epochs={str(k): v for k, v in eps.items()}); r_["_n"] = dict(train=len(tr), calibration=len(ca), test=len(ev))
    res["seeds"][s] = r_
    print(f"  set {s}: vs median " + "  ".join(f"{a} {r_[a]['vs_median']:+5.1f}" for a in C) + f" | cfg pub {cfg_pub} 4b {cfg_4b} epochs {eps}", flush=True)
    print("        ours vs it " + "  ".join(f"{a} {r_[a]['ours_vs_it']:+5.1f}" for a in C if a != "ours")
          + " | AUC " + "  ".join(f"{a} {np.mean(r_[a]['auc']):.3f}" for a in ("ours", "zr_pub", "zr_4b")), flush=True)
    json.dump(res, open(OUT / f"zr_faithful_draws_{POOL_.lower().replace('-', '')}_{S0}_{S1}{TAG}.json", "w"), indent=1, default=float)
print("DONE", flush=True)
