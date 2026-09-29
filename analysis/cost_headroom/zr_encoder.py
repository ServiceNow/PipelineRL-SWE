"""ZeroRouter's OWN stage-2 encoder on our pools (fidelity check for 4.A.32/4.A.33): fine-tuned DistilBERT ([CLS], final layer)
+ 11 linguistic features -> fusion trunk -> difficulty head (residual from the mean b) and discrimination head -> (log alpha, b).
Paper settings: distilbert-base-uncased, 40 epochs, batch 32, lr 3e-5. The 11 features are not listed in the paper beyond
"readability scores, parse tree depth"; we use: chars, words, sentences, mean word length, mean sentence length, syllables per word,
Flesch reading ease, Flesch-Kincaid grade, digit share, math/code symbol share, max bracket nesting depth (parse-depth proxy).
Loss (not stated in the paper): MSE onto the stage-1 targets. Targets: stage-1 IRT (zr_dimsweep.fit_stage1) on the 5 pool
models' TRAIN outcomes, D in {1, 5}, seed 0; the epoch is chosen on calibration MSE. Writes <pool>/zr_encoder_D<D>.npz with
predicted (log alpha, b) for every problem plus the stage-1 theta, for offline routing evaluation (zr_repro-style).
Usage: python zr_encoder.py   (GPU)
"""
import json, re, sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from decompose import R
from zr_dimsweep import fit_stage1

POOLS = [("pool_v2_tensors_5rung", "problems.jsonl"), ("omni500_tensors", None), ("mmlupro_tensors", None)]
TEXT_SRC = {"omni500_tensors": R / "math_pool" / "omni500" / "problems.jsonl", "mmlupro_tensors": R / "math_pool" / "mmlupro" / "problems.jsonl"}


def syllables(w):
    w = w.lower(); g = re.findall(r"[aeiouy]+", w); s = len(g) - (1 if w.endswith("e") and len(g) > 1 else 0)
    return max(s, 1)


def feats(x):
    words = re.findall(r"[A-Za-z]+", x); sents = max(len(re.findall(r"[.!?]+", x)), 1); nw = max(len(words), 1)
    syl = sum(syllables(w) for w in words) / nw; wl = sum(len(w) for w in words) / nw; sl = nw / sents
    depth = best = 0
    for ch in x:
        depth += ch in "([{"; depth -= ch in ")]}"; best = max(best, depth)
    return [np.log1p(len(x)), np.log1p(nw), np.log1p(sents), wl, sl, syl, 206.835 - 1.015 * sl - 84.6 * syl,
            0.39 * sl + 11.8 * syl - 15.59, sum(c.isdigit() for c in x) / max(len(x), 1),
            sum(c in "=+-*/^_\\$<>{}[]()" for c in x) / max(len(x), 1), best]


class Net(torch.nn.Module):
    def __init__(self, D, bmean):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained("distilbert-base-uncased")
        self.ps = torch.nn.Linear(768, 256); self.pt = torch.nn.Linear(11, 64)
        self.trunk = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(320, 256), torch.nn.ReLU(), torch.nn.Linear(256, 256), torch.nn.ReLU())
        self.hb = torch.nn.Sequential(torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Linear(128, D))
        self.ha = torch.nn.Sequential(torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Linear(128, D))
        self.register_buffer("bmean", torch.tensor(bmean, dtype=torch.float32))

    def forward(self, ids, mask, f):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        z = self.trunk(torch.cat([self.ps(h), self.pt(f)], 1))
        return self.ha(z), self.bmean + self.hb(z)


assert torch.cuda.is_available(), "GPU job"
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
for name, _ in POOLS:
    D_ = R / name; t = np.load(D_ / "tensors.npz", allow_pickle=True)
    pids = [str(p) for p in t["problem_ids"]]; pi = {p: i for i, p in enumerate(pids)}
    src = TEXT_SRC.get(name, D_ / "problems.jsonl")
    txt = {json.loads(l)["problem_id"]: json.loads(l)["problem_statement"] for l in open(src)}
    text = [txt[p] for p in pids]
    v = t["valid"].astype(bool); okd = (t["final_outcome"] & t["valid"]).astype(bool); n = v.sum(2); succ = (okd & v).sum(2)
    sp = json.load(open(D_ / "split_manifest.json")); idx = {k: np.array([pi[str(p)] for p in sp[f"{k}_problem_ids"]]) for k in ("train", "calibration", "test")}
    tr, cal = idx["train"], idx["calibration"]
    F = np.array([feats(x) for x in text], np.float32); F = (F - F[tr].mean(0)) / (F[tr].std(0) + 1e-6)
    enc = tok(text, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
    for D in (1, 5):
        torch.manual_seed(0)
        la, b, th = fit_stage1(succ[tr], n[tr], D, 0)
        tgt = np.zeros((len(pids), 2 * D), np.float32); tgt[tr] = np.c_[la, b]
        mu, sd = tgt[tr].mean(0), tgt[tr].std(0) + 1e-6
        net = Net(D, b.mean(0)).cuda(); opt = torch.optim.AdamW(net.parameters(), lr=3e-5)
        Ft, T = torch.tensor(F).cuda(), torch.tensor((tgt - mu) / sd).cuda()

        def predict(ii):
            net.eval(); out = []
            with torch.no_grad():
                for j in range(0, len(ii), 64):
                    s = ii[j:j + 64]; a_, b_ = net(enc["input_ids"][s].cuda(), enc["attention_mask"][s].cuda(), Ft[s])
                    out.append(torch.cat([a_, b_], 1).cpu().numpy())
            net.train(); return np.concatenate(out) * sd + mu
        # calibration targets: stage-1 positions of CALIBRATION queries are not fitted (not in train), so pick the epoch on a
        # held-out 10% of TRAIN
        rng = np.random.default_rng(0); perm = rng.permutation(tr); hold, fit_ = perm[: len(tr) // 10], perm[len(tr) // 10:]
        best = (np.inf, None, -1)
        for ep in range(40):
            order = rng.permutation(fit_)
            for j in range(0, len(order), 32):
                s = order[j:j + 32]
                a_, b_ = net(enc["input_ids"][s].cuda(), enc["attention_mask"][s].cuda(), Ft[s])
                loss = ((torch.cat([a_, b_], 1) - T[s]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
            hm = float((((predict(hold) - tgt[hold]) / sd) ** 2).mean())
            if hm < best[0]:
                best = (hm, predict(np.arange(len(pids))), ep)
        print(f"{name} D={D}: best epoch {best[2]}, held-out-train MSE (z units) {best[0]:.3f}", flush=True)
        pr = best[1]
        np.savez(D_ / f"zr_encoder_D{D}.npz", log_alpha=pr[:, :D], b=pr[:, D:], theta=th, train_log_alpha=la, train_b=b, problem_ids=np.array(pids))
print("ALL DONE")
