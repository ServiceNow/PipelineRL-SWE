"""GPU: score the FRESH expansion problems with the already-trained Intern-Decision LoRA success predictors (NEW_PATH 4.A.43).
Requests are built exactly as in intern_decision_finetune.data(): same template, grading rule, route descriptions and training-accuracy
priors (original train split); only the problem text differs. Adapter = best_adapter.pt (selected on original calibration). No fresh
labels are used. Writes intern_finetune_20261001/<pool>/fresh_predictions.npz (problem_ids, model_slots, p_successes).
Usage: python intern_fresh_infer.py --pool Omni|MMLU-Pro
"""
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from intern_decision_finetune import data, prepare, FROZEN
from intern_decision_pilot import MODEL, REVISION, TEMPERATURE
from jev_pilot import GRADING, ROUTES

R = Path("/mnt/llmd/results/exps/aristides/reason")
ap = argparse.ArgumentParser(); ap.add_argument("--pool", choices=["Omni", "MMLU-Pro"], required=True); a = ap.parse_args()
import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
assert torch.cuda.is_available(), "GPU job"
m = prepare(); d = data(a.pool)
template = d["requests"][0]["questions"]; priors = d["q"][d["tr"]].mean(0)
ds = "omni500" if a.pool == "Omni" else "mmlupro"
probs = [json.loads(l) for l in (R / "expanded_eval_20261001" / ds / "problems.jsonl").read_text().splitlines()]
old_ids = set(d["ids"]); fresh = [r for r in probs if r["problem_id"] not in old_ids]
reqs = [{"state": {"problem": str(r["problem_statement"]), "grading_rule": GRADING[a.pool],
                   "routes": {s: {"description": ROUTES[s], "training_accuracy": round(float(priors[j]), 4)} for j, s in enumerate(d["slots"])}},
         "questions": template} for r in fresh]
ck = snapshot_download(MODEL, revision=REVISION, token=False)
spec = importlib.util.spec_from_file_location("intern_inf", Path(ck) / "inference.py"); mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float32"
eng = mod.DecisionEngine(checkpoint=ck, temperature=TEMPERATURE, max_length=m["max_length"], device="cuda", dtype=dtype)
lin = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"}
targets = [n for n, l in eng.backend.model.named_modules() if isinstance(l, torch.nn.Linear) and ".language_model." in n and n.rsplit(".", 1)[-1] in lin]
model = get_peft_model(eng.backend.model, LoraConfig(r=m["lora_rank"], lora_alpha=m["lora_alpha"], lora_dropout=m["lora_dropout"], target_modules=targets, bias="none"))
out = R / "intern_finetune_20261001" / a.pool.lower().replace("-", "_")
set_peft_model_state_dict(model, torch.load(out / "best_adapter.pt", map_location="cpu", weights_only=True)); model.eval()
no, yes = [eng.tokenizer.encode(s, add_special_tokens=False)[0] for s in ["A", "B"]]
P = []
with torch.no_grad():
    for k, r in enumerate(reqs):
        comp, batch, pos = eng.backend.encode(mod.validate_request(r))
        if list(comp.fields) != d["slots"]: raise ValueError("Route order mismatch")
        lg = model(**batch.to("cuda"), use_cache=False, logits_to_keep=pos.to("cuda")).logits[0]
        P.append(torch.sigmoid((lg[:, yes].float() - lg[:, no].float()) / TEMPERATURE).cpu().numpy())
        if k % 500 == 0: print(k, len(reqs), flush=True)
np.savez_compressed(out / "fresh_predictions.npz", problem_ids=np.array([r["problem_id"] for r in fresh]), model_slots=np.array(d["slots"]), p_successes=np.asarray(P))
print("ALL DONE", len(P))
