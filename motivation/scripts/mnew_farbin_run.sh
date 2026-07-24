#!/bin/bash
# d_eval for the 7 remaining tasks (square already done), 4-way parallel,
# then far-bin + paired-McNemar over all 8 tasks.
exec >> /home/ubuntu/mnew_farbin.log 2>&1
M=/home/ubuntu/mimicgen_jihoonkwon/mimic_the_mimicgen/motivation
V=/home/ubuntu/mimicgen_jihoonkwon/robosuite_mimicgen/venv/bin/python
NV=/home/ubuntu/mimicgen_jihoonkwon/robosuite_mimicgen/venv/lib/python3.10/site-packages/nvidia
export PYTHONPATH=$M
export LD_LIBRARY_PATH=$NV/cu13/lib
export MUJOCO_GL=egl
cd $M
echo "[farrun $(date -u +%H:%M:%S)] deval start (4-way parallel)"
TASKS="threading coffee three_piece_assembly stack stack_three hammer_cleanup mug_cleanup"
run_one() { stdbuf -oL "$V" /home/ubuntu/mnew_deval.py "$1" > /home/ubuntu/deval_$1.log 2>&1; echo "[farrun $(date -u +%H:%M:%S)] deval done: $1"; }
export -f run_one; export V M
printf "%s\n" $TASKS | xargs -P 4 -I{} bash -c 'run_one "$@"' _ {}
echo "[farrun $(date -u +%H:%M:%S)] all deval done -> far-bin analysis"
$V /home/ubuntu/mnew_farbin.py square $TASKS
touch /home/ubuntu/FARBIN_DONE
echo "[farrun $(date -u +%H:%M:%S)] DONE"
