"""FrugalGPT-style answer scorer in the code setting: fine-tune jina-embeddings-v2-base-code (137M, 8k context) end-to-end,
ONE scorer per tier (route), to predict whether one attempt is correct from problem + attempt code -- the same text the
frozen 4B judge probe reads (pool_v2_judge_full/judge_shard*.jsonl, "... Is this solution correct?").
Mean-pooled encoder -> Linear(768, 1), BCE. Train on the tier's TRAIN-problem attempts, pick the epoch on CALIBRATION
log-loss, predict every attempt once. Writes <judge dir>/judge_preds_ft137m.jsonl {example_id, p_correct} and prints test AUC
(overall and within-problem) per tier next to the 4B probe's.
Usage: python finetune_judge_reader.py [--judge-dir ...] [--pool pool_v2_tensors_5rung] [--max-len 2048]
"""
import argparse, collections, glob, json, sys, numpy as np, torch
from pathlib import Path
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).parent))
from decompose import R

ap = argparse.ArgumentParser()
ap.add_argument("--judge-dir", default=str(R / "pool_v2_judge_full")); ap.add_argument("--pool", default="pool_v2_tensors_5rung")
ap.add_argument("--max-len", type=int, default=2048); ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--batch", type=int, default=16); ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--head-lr", type=float, default=1e-3)
a = ap.parse_args()
assert torch.cuda.is_available(), "GPU job only"
J = Path(a.judge_dir)
text = {}
for f in sorted(glob.glob(str(J / "judge_shard*.jsonl"))):
    for l in open(f):
        r = json.loads(l); text[r["problem_id"]] = r["prompt"]            # key = example_id ("<problem>||<slot><draw>")
man = [json.loads(l) for l in open(J / "judge_manifest.jsonl")]
sp = json.load(open(R / a.pool / "split_manifest.json"))
grp = {**{str(p): "train" for p in sp["train_problem_ids"]}, **{str(p): "calibration" for p in sp["calibration_problem_ids"]},
       **{str(p): "test" for p in sp["test_problem_ids"]}}
probe = {json.loads(l)["example_id"]: json.loads(l)["p_correct"] for l in open(J / "judge_preds_causal.jsonl")}

from transformers import AutoModel, AutoTokenizer
tok = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v2-base-code")


def within_auc(rows, score):
    byp = collections.defaultdict(list)
    for r in rows:
        byp[r["problem_id"]].append((score[r["example_id"]], r["correct"]))
    w = [1.0 if x > y else 0.5 if x == y else 0.0 for v in byp.values() for x, c1 in v if c1 for y, c2 in v if not c2]
    return float(np.mean(w)) if w else float("nan")


out = {}
for slot in sorted({r["slot"] for r in man}):
    torch.manual_seed(0); np.random.seed(0)
    rows = [r for r in man if r["slot"] == slot and r["example_id"] in text]
    split = {k: [r for r in rows if grp[str(r["problem_id"])] == k] for k in ("train", "calibration", "test")}
    enc = AutoModel.from_pretrained("jinaai/jina-embeddings-v2-base-code", trust_remote_code=True).cuda()
    head = torch.nn.Linear(enc.config.hidden_size, 1).cuda()
    opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": a.lr}, {"params": head.parameters(), "lr": a.head_lr}], weight_decay=0.01)

    def forward(batch):
        b = tok([text[r["example_id"]] for r in batch], truncation=True, max_length=a.max_len, padding=True, return_tensors="pt").to("cuda")
        h = enc(**b).last_hidden_state; m = b["attention_mask"].unsqueeze(-1).float()
        return head((h * m).sum(1) / m.sum(1)).squeeze(-1)

    def predict(rs):
        enc.eval(); ps = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(rs), 32):
                ps.append(torch.sigmoid(forward(rs[i:i + 32]).float()).cpu().numpy())
        enc.train(); return np.concatenate(ps) if ps else np.zeros(0)

    best = (np.inf, None, -1)
    for ep in range(a.epochs):
        perm = np.random.permutation(len(split["train"]))
        for i in range(0, len(perm), a.batch):
            batch = [split["train"][j] for j in perm[i:i + a.batch]]
            y = torch.tensor([float(r["correct"]) for r in batch], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = forward(batch).float()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), 1.0); opt.step()
        pc = np.clip(predict(split["calibration"]), 1e-4, 1 - 1e-4); yc = np.array([r["correct"] for r in split["calibration"]], float)
        ll = float(-(yc * np.log(pc) + (1 - yc) * np.log(1 - pc)).mean())
        print(f"{slot} epoch {ep}: calibration log-loss {ll:.4f}", flush=True)
        if ll < best[0]:
            best = (ll, predict(rows), ep)
    score = {r["example_id"]: float(p) for r, p in zip(rows, best[1])}
    out.update(score)
    te = split["test"]; y = np.array([r["correct"] for r in te], int)
    print(f"{slot}: best epoch {best[2]}; TEST n={len(te)} AUC 137M {roc_auc_score(y, [score[r['example_id']] for r in te]):.3f} "
          f"(4B probe {roc_auc_score(y, [probe[r['example_id']] for r in te]):.3f}); within-problem 137M {within_auc(te, score):.3f} "
          f"(4B probe {within_auc(te, probe):.3f})", flush=True)
    del enc, head, opt; torch.cuda.empty_cache()
with open(J / "judge_preds_ft137m.jsonl", "w") as f:
    for k, p in out.items():
        f.write(json.dumps({"example_id": k, "p_correct": p}) + "\n")
print("ALL DONE")
