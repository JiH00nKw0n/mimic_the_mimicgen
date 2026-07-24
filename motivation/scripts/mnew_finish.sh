#!/bin/bash
# Autonomous overnight finisher for motivation_new E2:
#   (bg) mug: arms -> train config -> train -> frozen reset
#   frozen resets for the 7 trained tasks -> eval them
#   then eval mug -> aggregate all -> E2_ALL_DONE
exec >> /home/ubuntu/mnew_finish.log 2>&1
M=/home/ubuntu/mimicgen_jihoonkwon/mimic_the_mimicgen/motivation
V=/home/ubuntu/mimicgen_jihoonkwon/robosuite_mimicgen/venv/bin/python
NV=/home/ubuntu/mimicgen_jihoonkwon/robosuite_mimicgen/venv/lib/python3.10/site-packages/nvidia
A=/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_new/e2_arms
R=/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_new/e2_results
CFG=$M/configs/experiments
OUTCFG=/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_new/e2_train_cfgs
BASE=/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_ic/e2_train_cfgs/e2_coffee_baseline_seed101.json
export PYTHONPATH=$M
export LD_LIBRARY_PATH=$NV/cu13/lib
log() { echo "[finish $(date -u +%H:%M:%S)] $*"; }
TRAINED="square threading coffee three_piece_assembly stack stack_three hammer_cleanup"

pkill -9 -f c_make_frozen_resets 2>/dev/null; sleep 2

# --- mug sub-pipeline (background) ---
(
  log "mug: arms"
  "$V" /home/ubuntu/mnew_arms.py mug_cleanup
  if [ -f "$A/mug_cleanup_N2/arms_manifest.json" ]; then
    log "mug: train config + train"
    sed -e "s/^task:.*/task: mug_cleanup/" -e "s/^variant:.*/variant: N2/" "$CFG/e2_square.yaml" > "$CFG/e2_new_mug_cleanup.yaml"
    "$V" "$M/scripts/c_make_train_configs.py" --arms-root "$A" --base-config "$BASE" \
       --experiment-config "$CFG/e2_new_mug_cleanup.yaml" --out-dir "$OUTCFG" --results-dir "$R"
    "$V" "$M/scripts/c_train_all.py" --launch-lists "$OUTCFG/launch_mug_cleanup.txt" --results-dir "$R" --concurrency 4
    log "mug: frozen reset"
    "$V" "$M/scripts/c_make_frozen_resets.py" --arms-root "$A" --tasks mug_cleanup:N2 --num-resets 200 --seed 7
  else
    log "mug: arms produced no manifest, skipping mug"
  fi
  touch /home/ubuntu/mug_ready
) &

# --- frozen resets for the 7 trained tasks (foreground) ---
log "frozen resets: 7 trained tasks"
"$V" "$M/scripts/c_make_frozen_resets.py" --arms-root "$A" \
  --tasks square:N2 threading:N2 coffee:N2 three_piece_assembly:N2 stack:N2 stack_three:N2 hammer_cleanup:N2 \
  --num-resets 200 --seed 7
log "eval: 7 trained tasks"
"$V" /home/ubuntu/mnew_eval.py $TRAINED

# --- wait for mug, eval it ---
log "waiting for mug sub-pipeline"
for i in $(seq 1 240); do [ -f /home/ubuntu/mug_ready ] && break; sleep 60; done
if [ -f "$A/mug_cleanup_N2/frozen_resets.hdf5" ]; then
  log "eval: mug"
  "$V" /home/ubuntu/mnew_eval.py mug_cleanup
fi

# --- aggregate everything ---
log "aggregate all"
"$V" /home/ubuntu/mnew_eval.py --aggregate $TRAINED mug_cleanup
touch /home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_new/E2_ALL_DONE
log "ALL DONE"
