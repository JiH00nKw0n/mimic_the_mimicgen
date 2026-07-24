#!/bin/bash
# Generate robomimic train configs for the complete motivation_new tasks and
# launch training at concurrency 8 (rollout disabled -> eval via frozen resets).
exec >> /home/ubuntu/mnew_train_setup.log 2>&1
M=/home/ubuntu/mimicgen_jihoonkwon/mimic_the_mimicgen/motivation
V=/home/ubuntu/mimicgen_jihoonkwon/robosuite_mimicgen/venv/bin/python
NV=/home/ubuntu/mimicgen_jihoonkwon/robosuite_mimicgen/venv/lib/python3.10/site-packages/nvidia
CFG=$M/configs/experiments
BASE=/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_ic/e2_train_cfgs/e2_coffee_baseline_seed101.json
OUTCFG=/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_new/e2_train_cfgs
RESULTS=/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_new/e2_results
ARMS=/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_new/e2_arms
mkdir -p "$OUTCFG" "$RESULTS"

echo "[setup $(date -u +%H:%M:%S)] generating train configs"
for t in square threading coffee three_piece_assembly stack stack_three hammer_cleanup; do
  [ -f "$ARMS/${t}_N2/arms_manifest.json" ] || { echo "  skip $t (no manifest)"; continue; }
  sed -e "s/^task:.*/task: $t/" -e "s/^variant:.*/variant: N2/" "$CFG/e2_square.yaml" > "$CFG/e2_new_$t.yaml"
  PYTHONPATH=$M "$V" "$M/scripts/c_make_train_configs.py" \
    --arms-root "$ARMS" --base-config "$BASE" \
    --experiment-config "$CFG/e2_new_$t.yaml" \
    --out-dir "$OUTCFG" --results-dir "$RESULTS"
done

echo "[setup $(date -u +%H:%M:%S)] launching training (concurrency 8)"
setsid nohup env PYTHONPATH=$M LD_LIBRARY_PATH=$NV/cu13/lib \
  "$V" "$M/scripts/c_train_all.py" \
  --launch-lists $OUTCFG/launch_*.txt --results-dir "$RESULTS" --concurrency 8 \
  </dev/null >> /home/ubuntu/mnew_train.log 2>&1 &
echo "[setup $(date -u +%H:%M:%S)] training launched pid $!"
