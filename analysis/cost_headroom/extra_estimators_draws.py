"""Two further cost estimators on the 10 training sets (NEW_PATH 4.A.120), cost-only (same success probes for every arm):
  rtr       Route-To-Reason (arXiv 2505.19435, Sec. 3): frozen all-mpnet-base-v2 query embedding q (mean-pooled, L2-normalized, 384
            tokens, as sentence-transformers); each route = (model j, effort k) gets z_j = [E(d_j); e_j], z_k = [E(d_k); e_k] with E the
            same encoder on a fixed text description and e_* learnable; one shared MLP (hidden 768) on [q; z_j; z_k] predicts output
            length, squared error on length (standardized), Adam lr 1e-3, batch 32, up to 100 epochs, early stopping on calibration loss
            (the paper: "test loss"); learnable embedding size 64 (not given). Their strategies are prompting strategies; our efforts
            play that role.
  rtr-log   the same network on log length (our target), to separate the architecture from the target scale
  scx       SCX-style (arXiv 2609.02292): one expected output length per QUERY, not per route: ridge on our prefill onto the
            route-centered mean log length, plus each route's training mean; then the usual smearing and level match
All arms post-processed like every estimator (level matched to the route's training mean; log-target arms smeared).
Per training set (DRAW=s): saving vs median and ours vs the arm, point + NBOOT test resamples (rng 0); test log-length R2 per route.
Usage: (GPU) REASON_ROOT=.../reason_pinned RESULT_TAG=_pinned NBOOT=300 OUT_DIR=... python extra_estimators_draws.py POOL S0 S1
"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
POOL_, S0, S1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); sys.argv = [sys.argv[0], "extra", POOL_]
exec(open(Path(__file__).parent / "cost_generalization.py").read().split('MODE, POOL = sys.argv[1], sys.argv[2]')[0])
from transformers import AutoModel, AutoTokenizer
assert torch.cuda.is_available(), "GPU job"
OUT = Path(os.environ.get("OUT_DIR", Path(__file__).parent)); TAG = os.environ.get("RESULT_TAG", "")
ENC = "sentence-transformers/all-mpnet-base-v2"
ROUTE = {"oss20lo": ("gpt-oss-20b", "low"), "oss20md": ("gpt-oss-20b", "medium"), "dsv4f": ("deepseek-v4-flash", "default"),
         "oss120md": ("gpt-oss-120b", "medium"), "oss120hi": ("gpt-oss-120b", "high")}
DESC = {"gpt-oss-20b": "gpt-oss-20b: OpenAI open-weight mixture-of-experts reasoning model, 21B parameters (3.6B active).",
        "gpt-oss-120b": "gpt-oss-120b: OpenAI open-weight mixture-of-experts reasoning model, 117B parameters (5.1B active).",
        "deepseek-v4-flash": "deepseek-v4-flash: DeepSeek open-weight mixture-of-experts reasoning model, fast variant, thinking enabled.",
        "low": "Low reasoning effort: a short chain of thought before answering.",
        "medium": "Medium reasoning effort: a moderate chain of thought before answering.",
        "high": "High reasoning effort: a long, thorough chain of thought before answering.",
        "default": "Default reasoning effort of the model: thinking enabled with the provider's default budget."}


def texts_of(pool):
    name, _, ds = POOLS[pool]; F = R / "expanded_eval_20261001" / ds if ds else R / name
    return [str(json.loads(l)["problem_statement"]) for l in open(F / "problems.jsonl")]


def encode(texts):
    tok = AutoTokenizer.from_pretrained(ENC); m = AutoModel.from_pretrained(ENC).cuda().eval(); out = []
    with torch.no_grad():
        for j in range(0, len(texts), 128):
            b = tok(texts[j:j + 128], truncation=True, max_length=384, padding=True, return_tensors="pt").to("cuda")
            h = m(**b).last_hidden_state; e = (h * b["attention_mask"][..., None]).sum(1) / b["attention_mask"].sum(1, keepdim=True)
            out.append(torch.nn.functional.normalize(e, dim=1).cpu().numpy())
    return np.concatenate(out)


class RTR(torch.nn.Module):
    def __init__(s, Dm, Dk, emb=64):
        super().__init__()
        s.Dm, s.Dk = torch.nn.Parameter(torch.tensor(Dm), requires_grad=False), torch.nn.Parameter(torch.tensor(Dk), requires_grad=False)
        s.em, s.ek = torch.nn.Embedding(len(Dm), emb), torch.nn.Embedding(len(Dk), emb)
        s.mlp = torch.nn.Sequential(torch.nn.Linear(768 * 3 + 2 * emb, 768), torch.nn.ReLU(), torch.nn.Linear(768, 1))

    def forward(s, q, j, k):
        return s.mlp(torch.cat([q, s.Dm[j], s.em(j), s.Dk[k], s.ek(k)], 1))[:, 0]


def rtr(d, Q, log_target, seed=0):
    """tokens [N, M] for every problem; trained on (problem, route) pairs of the training problems, early-stopped on calibration"""
    torch.manual_seed(seed); M = d["M"]; tr, ca = d["tr"], d["ca"]
    models = sorted({ROUTE[s][0] for s in d["slots"]}); efforts = sorted({ROUTE[s][1] for s in d["slots"]})
    jm = np.array([models.index(ROUTE[s][0]) for s in d["slots"]]); jk = np.array([efforts.index(ROUTE[s][1]) for s in d["slots"]])
    Dm, Dk = encode([DESC[x] for x in models]), encode([DESC[x] for x in efforts])
    Y = np.log(d["L"]) if log_target else d["L"]; mu, sd = Y[tr].mean(), Y[tr].std() + 1e-6
    pairs = lambda ii: (np.repeat(ii, M), np.tile(np.arange(M), len(ii)))
    Qt = torch.tensor(Q, dtype=torch.float32).cuda(); Yt = torch.tensor((Y - mu) / sd, dtype=torch.float32).cuda()
    jmt, jkt = torch.tensor(jm).cuda(), torch.tensor(jk).cuda()
    net = RTR(Dm, Dk).cuda(); opt = torch.optim.Adam(net.parameters(), lr=1e-3); rng = np.random.default_rng(seed)
    pi, pr = pairs(tr); ci, cr = pairs(ca); best, bad = (np.inf, None), 0
    f = lambda i, r: net(Qt[i], jmt[r], jkt[r])
    for ep in range(100):
        net.train(); o = rng.permutation(len(pi))
        for b in range(0, len(o), 32):
            s = o[b:b + 32]; i, r = torch.tensor(pi[s]).cuda(), torch.tensor(pr[s]).cuda()
            loss = ((f(i, r) - Yt[i, r]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            i, r = torch.tensor(ci).cuda(), torch.tensor(cr).cuda(); cl = float(((f(i, r) - Yt[i, r]) ** 2).mean())
        if cl < best[0]:
            best, bad = (cl, {k: v.clone() for k, v in net.state_dict().items()}), 0
        else:
            bad += 1
            if bad >= 10:
                break
    net.load_state_dict(best[1]); net.eval(); N = len(Q); ai, ar = pairs(np.arange(N))
    with torch.no_grad():
        yh = np.concatenate([f(torch.tensor(ai[b:b + 4096]).cuda(), torch.tensor(ar[b:b + 4096]).cuda()).cpu().numpy() for b in range(0, len(ai), 4096)])
    yh = (yh * sd + mu).reshape(N, M); tk = np.zeros((N, M)); Ytr = np.log(d["L"][tr])
    for m in range(M):
        if log_target:                                                                    # smearing + level match, as every estimator
            sm = np.mean(np.exp(Ytr[:, m] - yh[tr, m])); es, et = np.exp(yh[tr, m]) * sm, np.exp(yh[:, m]) * sm
        else:                                                                             # their raw-length target: level match only
            es, et = np.maximum(yh[tr, m], 1), np.maximum(yh[:, m], 1)
        tk[:, m] = et * d["L"][tr, m].mean() / es.mean()
    return tk, ep + 1


def scx(d):
    """one expected length per query (route-centered mean log length) read from our prefill, plus each route's training mean"""
    tr = d["tr"]; Y = np.log(d["L"]); c = Y[tr].mean(0); ybar = (Y - c).mean(1)
    sc = StandardScaler().fit(d["X"][tr]); Xs, Xa = sc.transform(d["X"][tr]), sc.transform(d["X"])
    z = RidgeCV(alphas=np.geomspace(1e1, 1e7, 13)).fit(Xs, ybar[tr]).predict(Xa); tk = np.zeros_like(d["L"])
    for m in range(d["M"]):
        yh = c[m] + z; sm = np.mean(np.exp(Y[tr, m] - yh[tr])); es, et = np.exp(yh[tr]) * sm, np.exp(yh) * sm
        tk[:, m] = et * d["L"][tr, m].mean() / es.mean()
    return tk


Q = encode(texts_of(POOL_)); res = {"pool": POOL_, "seeds": {}}
for s in range(S0, S1):
    os.environ["DRAW"] = str(s); d = load(POOL_); N = len(d["ids"]); tr, ca, ev = d["tr"], d["ca"], d["ev"]; assert len(Q) == N
    S, T = view(d, tr), view(d, np.arange(N)); eps = {}
    tk = {"ours": fit_predict("ours", S, T), "scx": scx(d)}
    tk["rtr"], eps["rtr"] = rtr(d, Q, False); tk["rtr-log"], eps["rtr-log"] = rtr(d, Q, True); tk["oracle"] = d["L"]
    C = {a: cost_of(d, x) for a, x in tk.items()}; Cm = cost_of(d, np.repeat(d["med"][None], N, 0))
    rb = np.random.default_rng(0); BS = [ev] + [ev[rb.integers(0, len(ev), len(ev))] for _ in range(NBOOT)]
    Hs = [{a: curve(d, d["P"], C[a], ii) for a in C} | {"median": curve(d, d["P"], Cm, ii)} for ii in BS]
    r_ = {}
    for a in C:
        vm = np.array([saved(H[a], H["median"]) for H in Hs]) * 100; vd = np.array([saved(H["ours"], H[a]) for H in Hs]) * 100
        r_[a] = dict(vs_median=float(vm[0]), vs_median_bs=vm[1:].tolist(), ours_vs_it=float(vd[0]), ours_vs_it_bs=vd[1:].tolist(),
                     auc=[float(auc(d["y0"][ev, m], d["P"][ev, m])) for m in range(d["M"])], r2=r2_routes(d, tk[a], ev))
    r_["_epochs"] = eps; r_["_n"] = dict(train=len(tr), calibration=len(ca), test=len(ev)); res["seeds"][s] = r_
    print(f"  set {s}: vs median " + "  ".join(f"{a} {r_[a]['vs_median']:+5.1f}" for a in C) + f" | epochs {eps}", flush=True)
    print("        ours vs it " + "  ".join(f"{a} {r_[a]['ours_vs_it']:+5.1f}" for a in C if a != "ours")
          + " | R2 " + "  ".join(f"{a} {np.mean(r_[a]['r2']):.2f}" for a in C if a != "oracle"), flush=True)
    json.dump(res, open(OUT / f"extra_estimators_draws_{POOL_.lower().replace('-', '')}_{S0}_{S1}{TAG}.json", "w"), indent=1, default=float)
print("DONE", flush=True)
