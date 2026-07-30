#!/bin/bash
# motivation_controlled: build arms -> train (39 new runs) -> eval controlled arms.
exec >> /home/ubuntu/ctrl_run.log 2>&1
M=/home/ubuntu/mimicgen_jihoonkwon/mimic_the_mimicgen/motivation
V=/home/ubuntu/mimicgen_jihoonkwon/robosuite_mimicgen/venv/bin/python
NV=/home/ubuntu/mimicgen_jihoonkwon/robosuite_mimicgen/venv/lib/python3.10/site-packages/nvidia
C=/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_controlled
CA=$C/arms; CR=$C/results; OUTCFG=$C/train_cfgs; CFG=$M/configs/experiments
BASE=/home/ubuntu/mimicgen_jihoonkwon/experiments/motivation_ic/e2_train_cfgs/e2_coffee_baseline_seed101.json
export PYTHONPATH=$M
export LD_LIBRARY_PATH=$NV/cu13/lib
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MUJOCO_GL=egl
ARMS=A_nearheavy,A_farheavy,B_far,B_nearpad,C2_hi,C2_mid,C_hi,C_mid,C_lo,D_1,D_2
log(){ echo "[ctrl $(date -u +%H:%M:%S)] $*"; }

log "STEP1 build arms (idempotent)"
$V /home/ubuntu/ctrl_build_arms.py all

log "STEP2 train configs"
LAUNCHES=""
for t in stack threading; do
  sed -e "s/^task:.*/task: $t/" -e "s/^variant:.*/variant: N2/" $CFG/e2_square.yaml > $CFG/e2_ctrl_$t.yaml
  $V $M/scripts/c_make_train_configs.py --arms-root $CA --base-config $BASE \
     --experiment-config $CFG/e2_ctrl_$t.yaml --out-dir $OUTCFG --results-dir $CR
  LAUNCHES="$LAUNCHES $OUTCFG/launch_$t.txt"
done

log "STEP3 train (conc 8, OMP1) — 39 runs"
$V $M/scripts/c_train_all.py --launch-lists $LAUNCHES --results-dir $CR --concurrency 8

log "STEP4 eval controlled arms (seeds 301-306, no aggregate)"
MNEW_A=$CA MNEW_R=$CR MNEW_ARMS=$ARMS MNEW_SEEDS_ALL=301,302,303,304,305,306 MNEW_NOAGG=1 \
  $V /home/ubuntu/mnew_eval.py stack threading

touch /home/ubuntu/CTRL_DONE
log "CTRL DONE"
