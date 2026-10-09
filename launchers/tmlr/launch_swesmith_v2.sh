#!/usr/bin/env bash
# TMLR (NEW_PATH 4.A.76): SWE-Smith one-shot reasoning pool, scaled 500 -> 1,432 instances and deepseek-v4-flash PINNED to StreamLake.
# gpt-oss routes: the 500 plan-D draws are reused (copied), the other 932 drawn now (unpinned, as every gpt-oss route in the paper);
# dsv4f: all 1,432 redrawn pinned (the plan-D dsv4f draws were unpinned). Then SEARCH/REPLACE -> diffs (light converter) and Daytona
# labels at concurrency 8 (<= 10 sandboxes total; nothing else uses Daytona now). Reports reuse the plan-D run ids for gpt-oss, so the
# 500 already-labelled draws are skipped. Spend guard $5 per generation job (est. ~$5 total). Usage: SUBMIT=1 bash launch_swesmith_v2.sh
set -euo pipefail
R=/mnt/llmd/results/exps/aristides/reason; V1=$R/swesmith_reasoning_pool; W=$R/swesmith_reasoning_pool_v2; mkdir -p $W/full $W/patches
COLL="$R/offline_router_swe_smith_train1500_real_labels_4route_1780639659/collect/*"; PY=/home/toolkit/.conda/envs/pipeline-rl/bin/python3
[ -s $W/full/ids.json ] || $PY -c "
import pandas as pd, glob, json
df = pd.concat([pd.read_parquet(f, columns=['problem_id']) for f in sorted(glob.glob('$COLL/*.parquet'))]).drop_duplicates('problem_id')
json.dump(sorted(df.problem_id), open('$W/full/ids.json', 'w'))"
for r in oss20lo oss20md oss120md oss120hi; do [ -s $W/full/predictions_${r}_d0.jsonl ] || cp $V1/full/predictions_${r}_d0.jsonl $W/full/; done
cat > $W/run_gen_oss.sh <<EOS
#!/usr/bin/env bash
cd /home/toolkit/PipelineRL-SWE
$PY -m pipelinerl.swe.scripts.offline_router.swe_draw_patches --instances-file $W/full/ids.json --out-dir $W/full --collection-dir "$COLL" \
  --models oss20lo,oss20md,oss120md,oss120hi --first-draw 0 --draws 1 --raw --concurrency 12 --budget-usd 5 > $W/gen_oss.log 2>&1
echo EXIT \$? >> $W/gen_oss.log; touch $W/GEN_OSS_DONE
EOS
mkdir -p $W/full_dsv4f
cat > $W/run_gen_dsv4f.sh <<EOS
#!/usr/bin/env bash
cd /home/toolkit/PipelineRL-SWE; export OPENROUTER_PIN_PROVIDER=StreamLake
$PY -m pipelinerl.swe.scripts.offline_router.swe_draw_patches --instances-file $W/full/ids.json --out-dir $W/full_dsv4f --collection-dir "$COLL" \
  --models dsv4f --first-draw 0 --draws 1 --raw --concurrency 12 --budget-usd 5 > $W/gen_dsv4f.log 2>&1
echo EXIT \$? >> $W/gen_dsv4f.log; cp $W/full_dsv4f/predictions_dsv4f_d0.jsonl $W/full/; touch $W/GEN_DSV4F_DONE
EOS
cat > $W/run_label.sh <<EOS
#!/usr/bin/env bash
cd /home/toolkit/PipelineRL-SWE
until [ -e $W/GEN_OSS_DONE ] && [ -e $W/GEN_DSV4F_DONE ]; do sleep 120; done
cp $W/full/predictions_*_d0.jsonl $W/patches/
$PY pipelinerl/swe/scripts/offline_router/convert_swesmith_patches_light.py --predictions-dir $W/patches > $W/convert.log 2>&1
set -a; . /home/toolkit/PipelineRL-SWE/.env; set +a
for r in oss20lo oss20md oss120md oss120hi dsv4f; do
  run=swesmith_reason_v2_predictions_\${r}_d0; [ \$r = dsv4f ] && run=swesmith_reason_pinned_predictions_dsv4f_d0
  $PY pipelinerl/swe/scripts/offline_router/run_swesmith_eval_daytona.py --predictions_path $W/patches/predictions_\${r}_d0.jsonl \
    --run_id \$run --concurrency 8 > $W/eval_\$r.log 2>&1
done
echo ALL DONE >> $W/eval_done.log
EOS
chmod +x $W/run_*.sh
[[ ${SUBMIT:-0} == 1 ]] || { echo "dry run: scripts in $W"; exit 0; }
sub() { make job JOB_NAME="$1" ENV=pipeline-rl CONDA_EXE=/opt/conda/bin/conda GPU=0 GPU_MEM=0 CPU=$3 CPU_MEM=$4 SNAPSHOT=1 COMMAND="$2" 2>&1 | grep -oE "QUEUING|RUNNING|[Ee]rror.*" | head -1 || true; sleep 35; }
echo -n "swev2_genoss "; sub swev2_genoss "bash $W/run_gen_oss.sh" 4 16
echo -n "swev2_gendsv4f "; sub swev2_gendsv4f "bash $W/run_gen_dsv4f.sh" 4 16
echo -n "swev2_label "; sub swev2_label "bash $W/run_label.sh" 8 32
