#!/usr/bin/env bash
# Track A head-to-head (NEW_PATH 4.A.6): literature cost predictors (MixLLM-style embedding ensemble, prompt-feature GBM,
# plain ridge on our probe, each route's own prefill on LCB) through the same headroom/capture decomposition.
# GPU eai job (the text embedding runs on the GPU; everything else is sklearn). Commit and push first.
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason
O=${R}/cost_head_to_head; NAME="cost_h2h_$(date -u +%Y%m%d_%H%M%S)"; mkdir -p ${O}
cat > ${O}/run.sh <<EOS
#!/usr/bin/env bash
set -uo pipefail
export HF_HOME=/home/toolkit/.cache/huggingface HF_HUB_OFFLINE=1
B=analysis/cost_headroom/baseline_cost_heads.py
OWN=${R}/pool_activations_readouts_1788468245
python \${B} pool_v2_tensors_5rung ${R}/pv2_scout_prefill_1756715297/scout.npz \
  --own oss20lo=\${OWN}/oss20.npz,oss20md=\${OWN}/oss20.npz,oss120md=\${OWN}/oss120.npz,oss120hi=\${OWN}/oss120.npz > ${O}/lcb.log 2>&1
python \${B} cc_tensors ${R}/cc_pool/scout_prefill.npz > ${O}/cc.log 2>&1
python \${B} bcb_tensors_5r ${R}/bcb_scout_prefill.npz > ${O}/bcb.log 2>&1
python \${B} swesmith_reason_tensors ${R}/swesmith_costhead/scout_prefill.npz > ${O}/swesmith.log 2>&1
python \${B} taco_tensors_ha ${R}/taco_activations_1788500841/scout.npz > ${O}/taco.log 2>&1
A=""
for p in pool_v2_tensors_5rung cc_tensors bcb_tensors_5r swesmith_reason_tensors taco_tensors_ha; do
  for m in market mixllm gbm probe ownprefill; do   # skip a predictor whose step failed rather than lose the table
    [ -f ${R}/\$p/cost_preds_\$m.jsonl ] && [ \$m != ownprefill -o \$p == pool_v2_tensors_5rung ] && A="\$A \$p:cost_preds_\$m.jsonl:market"
  done
done
python analysis/cost_headroom/decompose.py --out head_to_head \$A > ${O}/decompose.log 2>&1
echo ALL DONE >> ${O}/decompose.log
EOS
chmod +x ${O}/run.sh
if [ "${SUBMIT:-0}" != "1" ]; then echo "dry run:"; cat ${O}/run.sh; exit 0; fi
make job JOB_NAME="${NAME}" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda GPU=1 GPU_MEM=0 CPU=16 CPU_MEM=64 SNAPSHOT=1 \
  COMMAND="bash ${O}/run.sh"
echo "submitted ${NAME} -> ${O}"
