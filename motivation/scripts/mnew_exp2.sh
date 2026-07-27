#!/bin/bash
# Experiment 2: coverage-vs-density (near_only) + source-quality causal (hidgr/lodgr).
# New arms from existing D2 pool, seeds 201-203. train (existing 101-106 skip) -> eval new arms.
exec >> /home/ubuntu/mnew_exp2.log 2>&1
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
TASKS="square coffee three_piece_assembly threading stack"
log(){ echo "[exp2 $(date -u +%H:%M:%S)] $*"; }

log "STEP1 build new arms (near_only + hidgr/lodgr)"
$V /home/ubuntu/mnew_newarms.py $TASKS

log "STEP2 train configs"
LAUNCHES=""
for t in $TASKS; do
  sed -e "s/^task:.*/task: $t/" -e "s/^variant:.*/variant: N2/" $CFG/e2_square.yaml > $CFG/e2_new_$t.yaml
  $V $M/scripts/c_make_train_configs.py --arms-root $A --base-config $BASE \
     --experiment-config $CFG/e2_new_$t.yaml --out-dir $OUTCFG --results-dir $R
  LAUNCHES="$LAUNCHES $OUTCFG/launch_$t.txt"
done

log "STEP3 train (conc 8, OMP1; existing 101-106 skip)"
$V $M/scripts/c_train_all.py --launch-lists $LAUNCHES --results-dir $R --concurrency 8

log "STEP4 eval new arms (201-203, no aggregate overwrite)"
MNEW_ARMS=near_only,hidgr_only,lodgr_only MNEW_SEEDS_ALL=201,202,203 MNEW_NOAGG=1 \
  $V /home/ubuntu/mnew_eval.py $TASKS

touch /home/ubuntu/EXP2_DONE
log "EXP2 DONE"
