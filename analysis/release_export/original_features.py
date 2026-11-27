"""Write the paper's original features in the release layout (internal; reads the pinned roots on /mnt).

The re-extracted activations (scoutroute.extract) differ slightly from the originals, enough to move single-split numbers;
the original ones reproduce the paper's summaries exactly. Writes, per set, <out>/<set>/scout.npz (problem_ids, layers, model,
system_prompt, mean, last), jina.npz and qwen3emb4b.npz (problem_ids, emb), prompts.jsonl (copied from the repo's features/),
and <out>/scouts/<tag>/<set>/scout.npz for the other scouts. Success probes: run `python -m scoutroute.probes` afterwards.
Usage: python original_features.py <repo> <out>
"""
import shutil
import sys
from pathlib import Path

import numpy as np

P = Path("/mnt/llmd/results/exps/aristides/reason_pinned"); R = Path("/mnt/llmd/results/exps/aristides/reason")
REPO, OUT = Path(sys.argv[1]), Path(sys.argv[2])

# release set: (activation file, directory holding the jina / qwen3emb4b embeddings, tensors dir for the jina .npy ids or None)
SETS = {
    "lcb": ("pv2_scout_prefill_1756715297/scout.npz", "pool_v2_tensors_5rung", True),
    "apps": ("apps_probe/instruct.npz", "apps_tensors", True),
    "bcb": ("bcb_scout_prefill.npz", "bcb_tensors_5r", True),
    "cc": ("cc_pool/scout_prefill.npz", "cc_tensors", True),
    "omni": ("omni500_probe/instruct.npz", "omni500_tensors", True),
    "mmlupro": ("mmlupro_probe/instruct.npz", "mmlupro_tensors", True),
    "aime": ("aime_probe/instruct.npz", "aime_tensors", True),
    "supergpqa": ("supergpqa_probe/instruct.npz", "supergpqa_tensors", True),
    "bbeh": ("bbeh_probe/instruct.npz", "bbeh_tensors", True),
    "omni_expanded": ("expanded_eval_20261001/omni500/prefill_combined.npz", "expanded_eval_20261001/omni500", False),
    "mmlupro_expanded": ("expanded_eval_20261001/mmlupro/prefill_combined.npz", "expanded_eval_20261001/mmlupro", False),
}
SCOUT_SETS = {"lcb": "lcb", "omni_expanded": "omni500", "mmlupro_expanded": "mmlupro"}
KEEP = ("problem_ids", "layers", "model", "system_prompt", "mean", "last")


def orig(rel):
    """the pinned root's copy first (what the paper's runs read), else the raw root"""
    for root in (P, R):
        if (root / rel).exists():
            return root / rel
    raise FileNotFoundError(rel)


def activations(src, dst):
    z = np.load(src, allow_pickle=True); dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **{k: z[k] for k in KEEP}); print(dst, z["mean"].shape, flush=True)


def main():
    for s, (act, emb, base) in SETS.items():
        d = OUT / s; d.mkdir(parents=True, exist_ok=True)
        activations(orig(act), d / "scout.npz")
        shutil.copy(REPO / "features" / s / "prompts.jsonl", d / "prompts.jsonl")
        if base:
            ids = np.load(orig(f"{emb}/tensors.npz"), allow_pickle=True)["problem_ids"]
            jina = np.load(orig(f"{emb}/emb_jina_code.npy"))
        else:
            z = np.load(orig(f"{emb}/text_embeddings.npz"), allow_pickle=True); ids, jina = z["problem_ids"], z["jina"]
        assert len(ids) == len(jina)
        np.savez(d / "jina.npz", problem_ids=np.array([str(p) for p in ids]), emb=jina.astype(np.float32))
        try:
            q = np.load(orig(f"{emb}/qwen3emb4b.npz"), allow_pickle=True)
            np.savez(d / "qwen3emb4b.npz", problem_ids=q["problem_ids"], emb=q["emb"])
        except FileNotFoundError:
            print(s, "no qwen3emb4b (none in the release either)", flush=True)
    for f in sorted((REPO / "features" / "scouts").glob("*/*/scout.npz")):
        tag, s = f.parent.parent.name, f.parent.name
        activations(orig(f"prefill_size_20261005/{tag}/{SCOUT_SETS[s]}.npz"), OUT / "scouts" / tag / s / "scout.npz")


if __name__ == "__main__":
    main()
