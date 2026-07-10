"""
Experiment B: kNN retrieval abstention.

No learned classification head. We embed every task with the 8B encoder and use
k nearest neighbours from the training set (by cosine similarity) as the
abstention signal: score(task) = mean 120B-success rate of the k closest
training tasks.

Two variants:
  - BASE: raw Qwen3-Embedding-8B (no LoRA)
  - LORA: best trained LoRA checkpoint (epoch 9)

This tells us whether the embedding geometry already captures task difficulty
(pre-trained) or whether our fine-tuning improved it.

Outputs a table comparing kNN variants vs trained-head baselines.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

DATASET_DIR = (
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_smith_train1500_real_labels_4route_1780639659/collect"
)
LORA_CHECKPOINT = (
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_reward_bce_r32_qkvo_mlp_10epoch_1781112916/"
    "train_qwen3_embedding_8b_lora_reward_bce_10epoch/checkpoints/epoch_0009"
)
PS_PREDS_PATH = (
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_reward_bce_r32_qkvo_mlp_10epoch_1781112916/"
    "train_qwen3_embedding_8b_lora_reward_bce_10epoch/eval_predictions.jsonl"
)
IO_PREDS_PATH = (
    "/mnt/llmd/results/exps/aristides/reason/"
    "offline_router_swe_smith_train1500_real_4route_qwen3_embedding_8b_lora_reward_bce_r32_qkvo_mlp_input_only_10epoch_1781112942/"
    "train_qwen3_embedding_8b_lora_reward_bce_10epoch/eval_predictions.jsonl"
)
MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
MAX_SEQ_LENGTH = 24000
ROUTE_LABELS = [
    "scout:Qwen/Qwen3-4B-Instruct-2507",
    "solver:openai/gpt-oss-20b",
    "solver:Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "solver:openai/gpt-oss-120b",
]
OUT_PATH = "/home/toolkit/PipelineRL-SWE/analysis/knn_retrieval_results.json"


def _last_token_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    left_padded = attn_mask[:, 0].sum() == 0
    if not left_padded:
        return last_hidden[:, -1]
    seq_lens = attn_mask.sum(dim=1) - 1
    return last_hidden[torch.arange(last_hidden.shape[0], device=last_hidden.device), seq_lens]


def build_input_text(row: dict, route_labels: list[str]) -> str | None:
    """Replicate the post_primary reward input text used during training."""
    prompt_text = row.get("prompt_text")
    primary_output_text = row.get("primary_output_text")
    if not isinstance(prompt_text, str) or not isinstance(primary_output_text, str):
        return None
    route_legend = "\n".join(f"{idx}: {label}" for idx, label in enumerate(route_labels))
    return (
        "Predict the realized proxy rewards for each model route.\n"
        "The proxy reward is computed after the repair run by comparing the route's patch against the gold patch.\n"
        "Only the primary model attempt is shown. Use it as context for all route predictions.\n\n"
        "[Route Order]\n"
        f"{route_legend}\n\n"
        "[Original Repair Prompt]\n"
        f"{prompt_text}\n\n"
        "[Primary Model Attempt]\n"
        f"{primary_output_text}"
    )


def build_input_text_base(row: dict, route_labels: list[str]) -> str | None:
    """For the base (no-LoRA) model: use standard Qwen3-Embedding instruction format."""
    prompt_text = row.get("prompt_text")
    primary_output_text = row.get("primary_output_text")
    if not isinstance(prompt_text, str) or not isinstance(primary_output_text, str):
        return None
    route_legend = "\n".join(f"{idx}: {label}" for idx, label in enumerate(route_labels))
    task = (
        "Predict the probability that each SWE repair model route produces a patch that passes the tests. "
        "Use the route identities and the available repair evidence to estimate per-route success."
    )
    body = (
        "[Route Order]\n"
        f"{route_legend}\n\n"
        "[Original Repair Prompt]\n"
        f"{prompt_text}\n\n"
        "[Primary Model Attempt]\n"
        f"{primary_output_text}"
    )
    return f"Instruct: {task}\nQuery: {body}"


def encode_texts(
    texts: list[str],
    tokenizer,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 1,
    max_seq_length: int = MAX_SEQ_LENGTH,
    desc: str = "Encoding",
) -> np.ndarray:
    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc=desc):
            batch_texts = texts[start : start + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt",
            ).to(device)
            outputs = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                return_dict=True,
            )
            pooled = _last_token_pool(outputs.last_hidden_state, enc["attention_mask"])
            emb = F.normalize(pooled.float(), p=2, dim=1)
            all_embeddings.append(emb.cpu().numpy())
    return np.vstack(all_embeddings)


def knn_scores(
    train_emb: np.ndarray,
    eval_emb: np.ndarray,
    train_labels: np.ndarray,
    k: int,
) -> np.ndarray:
    """Cosine similarity kNN: returns mean label of k nearest training neighbours."""
    sims = eval_emb @ train_emb.T  # (N_eval, N_train)
    scores = []
    for row in sims:
        idx = np.argpartition(row, -k)[-k:]
        scores.append(train_labels[idx].mean())
    return np.array(scores)


def load_parquet_split(split_dir: str) -> list[dict]:
    dfs = [pd.read_parquet(p) for p in Path(split_dir).glob("*.parquet")]
    df = pd.concat(dfs)
    rows = df.to_dict(orient="records")
    # convert numpy arrays to lists
    for row in rows:
        for k, v in row.items():
            if hasattr(v, "tolist"):
                row[k] = v.tolist()
    return rows


def load_trained_head_roc(pred_path: str, eval_ids: list[str], eval_labels: np.ndarray) -> float:
    preds = {}
    with open(pred_path) as f:
        for line in f:
            d = json.loads(line)
            preds[d["problem_id"]] = float(np.mean(d["pred_rewards"]))
    scores = np.array([preds.get(pid, 0.5) for pid in eval_ids])
    return float(roc_auc_score(eval_labels, scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-base", action="store_true", help="Skip base (no-LoRA) model to save time")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load data ──────────────────────────────────────────────────────────────
    print("Loading data...")
    train_rows = load_parquet_split(f"{DATASET_DIR}/train")
    eval_rows = load_parquet_split(f"{DATASET_DIR}/eval")
    print(f"  train={len(train_rows)}  eval={len(eval_rows)}")

    train_labels = np.array([int(r["route_successes"][3]) for r in train_rows])
    eval_labels = np.array([int(r["route_successes"][3]) for r in eval_rows])
    eval_ids = [str(r["problem_id"]) for r in eval_rows]

    print(f"  Train 120B success rate: {train_labels.mean():.3f}")
    print(f"  Eval  120B success rate: {eval_labels.mean():.3f}")

    results = {}

    # ── trained-head baselines ─────────────────────────────────────────────────
    print("\nComputing trained-head baselines...")
    ps_roc = load_trained_head_roc(PS_PREDS_PATH, eval_ids, eval_labels)
    io_roc = load_trained_head_roc(IO_PREDS_PATH, eval_ids, eval_labels)
    print(f"  PS trained head ROC AUC: {ps_roc:.4f}")
    print(f"  IO trained head ROC AUC: {io_roc:.4f}")
    results["ps_trained_head"] = ps_roc
    results["io_trained_head"] = io_roc

    # PS+IO ensemble
    ps_preds, io_preds = {}, {}
    for path, d in [(PS_PREDS_PATH, ps_preds), (IO_PREDS_PATH, io_preds)]:
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                d[rec["problem_id"]] = float(np.mean(rec["pred_rewards"]))
    ens_scores = np.array([(ps_preds.get(pid, 0.5) + io_preds.get(pid, 0.5)) / 2 for pid in eval_ids])
    ens_roc = float(roc_auc_score(eval_labels, ens_scores))
    print(f"  PS+IO ensemble ROC AUC:  {ens_roc:.4f}")
    results["ps_io_ensemble"] = ens_roc

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")

    k_values = [1, 3, 5, 10, 20, 50]

    # ── BASE model kNN ─────────────────────────────────────────────────────────
    if not args.skip_base:
        print(f"\n{'='*60}")
        print("BASE model (no LoRA):")
        base_model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16,
                                               attn_implementation="flash_attention_2").to(device)

        train_texts_base = [build_input_text_base(r, ROUTE_LABELS) or "" for r in train_rows]
        eval_texts_base = [build_input_text_base(r, ROUTE_LABELS) or "" for r in eval_rows]

        train_emb_base = encode_texts(train_texts_base, tokenizer, base_model, device, desc="BASE train")
        eval_emb_base = encode_texts(eval_texts_base, tokenizer, base_model, device, desc="BASE eval")

        results["base_knn"] = {}
        for k in k_values:
            scores = knn_scores(train_emb_base, eval_emb_base, train_labels, k)
            roc = float(roc_auc_score(eval_labels, scores))
            sp = float(spearmanr(scores, eval_labels).statistic)
            print(f"  k={k:3d}: ROC AUC={roc:.4f}  Spearman={sp:.3f}")
            results["base_knn"][k] = {"roc_auc": roc, "spearman": sp}

        del base_model
        torch.cuda.empty_cache()

    # ── LoRA-tuned model kNN ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("LoRA-tuned model (epoch 9):")
    try:
        from peft import PeftModel
        base_for_lora = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16,
                                                  attn_implementation="flash_attention_2")
        lora_model = PeftModel.from_pretrained(base_for_lora, LORA_CHECKPOINT).to(device)
        lora_model = lora_model.merge_and_unload()
    except Exception as e:
        print(f"  Failed to load LoRA model: {e}")
        print("  Falling back to loading safetensors directly...")
        from safetensors.torch import load_file
        lora_model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16,
                                               attn_implementation="flash_attention_2")
        # Load only encoder weights from checkpoint
        ckpt_path = Path(LORA_CHECKPOINT) / "model.safetensors"
        if not ckpt_path.exists():
            ckpt_path = Path(LORA_CHECKPOINT)
        state = load_file(str(ckpt_path))
        encoder_state = {k.removeprefix("encoder."): v for k, v in state.items() if k.startswith("encoder.")}
        missing, unexpected = lora_model.load_state_dict(encoder_state, strict=False)
        print(f"  Loaded encoder weights: {len(encoder_state)} keys, missing={len(missing)}, unexpected={len(unexpected)}")
        lora_model = lora_model.to(device)

    # Use same input format as training (no Instruct: prefix)
    train_texts_lora = [build_input_text(r, ROUTE_LABELS) or "" for r in train_rows]
    eval_texts_lora = [build_input_text(r, ROUTE_LABELS) or "" for r in eval_rows]

    train_emb_lora = encode_texts(train_texts_lora, tokenizer, lora_model, device, desc="LORA train")
    eval_emb_lora = encode_texts(eval_texts_lora, tokenizer, lora_model, device, desc="LORA eval")

    results["lora_knn"] = {}
    for k in k_values:
        scores = knn_scores(train_emb_lora, eval_emb_lora, train_labels, k)
        roc = float(roc_auc_score(eval_labels, scores))
        sp = float(spearmanr(scores, eval_labels).statistic)
        print(f"  k={k:3d}: ROC AUC={roc:.4f}  Spearman={sp:.3f}")
        results["lora_knn"][k] = {"roc_auc": roc, "spearman": sp}

    # ── summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'Method':<40} {'ROC AUC':>8}")
    print(f"{'IO trained head':<40} {io_roc:>8.4f}")
    print(f"{'PS trained head':<40} {ps_roc:>8.4f}")
    print(f"{'PS+IO ensemble':<40} {ens_roc:>8.4f}")
    if not args.skip_base:
        best_base = max(results["base_knn"].values(), key=lambda x: x["roc_auc"])
        best_base_k = max(results["base_knn"], key=lambda k: results["base_knn"][k]["roc_auc"])
        print(f"{'BASE kNN (best k)':<40} {best_base['roc_auc']:>8.4f}  (k={best_base_k})")
    best_lora = max(results["lora_knn"].values(), key=lambda x: x["roc_auc"])
    best_lora_k = max(results["lora_knn"], key=lambda k: results["lora_knn"][k]["roc_auc"])
    print(f"{'LoRA kNN (best k)':<40} {best_lora['roc_auc']:>8.4f}  (k={best_lora_k})")

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {OUT_PATH}")


if __name__ == "__main__":
    main()
