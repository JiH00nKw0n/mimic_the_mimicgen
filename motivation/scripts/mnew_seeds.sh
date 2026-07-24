#!/bin/bash
# A: grow motivation_new E2 from 2 -> 6 dataset seeds (add 103-106).
#   add arm-seed filter keys -> train configs -> train (101/102 auto-skip)
#   -> eval all 6 seeds -> far-bin + paired McNemar over 6 seeds.
exec >> /home/ubuntu/mnew_seeds.log 2>&1
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
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MUJOCO_GL=egl
ALLSEEDS=101,102,103,104,105,106
TASKS="square threading coffee three_piece_assembly stack stack_three mug_cleanup hammer_cleanup"
log(){ echo "[seeds $(date -u +%H:%M:%S)] $*"; }

log "STEP 1 add seeds 103-106"
MNEW_SEEDS=103,104,105,106 $V /home/ubuntu/mnew_addseeds.py $TASKS

log "STEP 2 train configs"
LAUNCHES=""
for t in $TASKS; do
  sed -e "s/^task:.*/task: $t/" -e "s/^variant:.*/variant: N2/" $CFG/e2_square.yaml > $CFG/e2_new_$t.yaml
  $V $M/scripts/c_make_train_configs.py --arms-root $A --base-config $BASE \
     --experiment-config $CFG/e2_new_$t.yaml --out-dir $OUTCFG --results-dir $R
  LAUNCHES="$LAUNCHES $OUTCFG/launch_$t.txt"
done

log "STEP 3 train (concurrency 8, OMP 1; 101/102 skip)"
$V $M/scripts/c_train_all.py --launch-lists $LAUNCHES --results-dir $R --concurrency 8

log "STEP 4 eval all 6 seeds (101/102 cached)"
MNEW_SEEDS_ALL=$ALLSEEDS $V /home/ubuntu/mnew_eval.py $TASKS

log "STEP 5 far-bin + paired, 6 seeds"
MNEW_SEEDS_ALL=$ALLSEEDS $V /home/ubuntu/mnew_farbin.py $TASKS
touch /home/ubuntu/SEEDS_DONE
log "ALL DONE"
