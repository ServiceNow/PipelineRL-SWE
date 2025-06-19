#!/usr/bin/env python3

import json
import random
from datasets import load_dataset
from swebench.harness.modal_eval.run_evaluation_modal import run_instances_modal

# Load your predictions
with open('/home/toolkit/research-now-reasoner/pipelinerl/preds_rl.jsonl', 'r') as f:
    predictions = [json.loads(line) for line in f]

# Group predictions by instance_id
instance_groups = {}
for pred in predictions:
    instance_id = pred['instance_id']
    if instance_id not in instance_groups:
        instance_groups[instance_id] = []
    instance_groups[instance_id].append(pred)

unique_instance_ids = list(instance_groups.keys())
print(f"Found {len(predictions)} total predictions")
print(f"Found {len(unique_instance_ids)} unique instance IDs")

# Randomly select one prediction from each group and validate patches
deduplicated_predictions = []
for instance_id, group in instance_groups.items():
    selected_pred = random.choice(group)
    
    # Validate patch format
    patch = selected_pred.get('model_patch', '')
    if patch and not patch.endswith('\n'):
        print(f"Warning: Fixing patch for {instance_id} - adding missing newline")
        selected_pred['model_patch'] = patch + '\n'
    
    deduplicated_predictions.append(selected_pred)
    if len(group) > 1:
        print(f"Instance {instance_id}: randomly selected 1 from {len(group)} predictions")

print(f"After deduplication: {len(deduplicated_predictions)} predictions")

# Convert predictions to dict format
predictions_dict = {pred['instance_id']: pred for pred in deduplicated_predictions}

# Load the dataset to get the instances
print("Loading SWE-bench Lite dataset...")
full_dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

# Filter dataset for instances that we have predictions for
target_instances = [
    item for item in full_dataset 
    if item['instance_id'] in predictions_dict
]

print(f"Found {len(target_instances)} instances in dataset matching our predictions")

# Run evaluation in the cloud
print("Starting Modal evaluation...")
results = run_instances_modal(
    predictions=predictions_dict,
    instances=target_instances,
    full_dataset=list(full_dataset),
    run_id="dedup_eval_run",
    timeout=600  # 10 minutes timeout
)

print("Evaluation completed!")
print("Results:", results)
print(f"Evaluated {len(target_instances)} unique instances")