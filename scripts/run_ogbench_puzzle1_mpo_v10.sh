#!/bin/bash

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export TMPDIR=${TMPDIR:-/data/jzn/qc_clean_runs/tmp}
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR"

beta=1.0
temp=0.001
q_clip=0.15

project_name=qc_online
save_dir=${SAVE_DIR:-/data/jzn/qc_clean_runs/exp}
env_name=puzzle-3x3-play-singletask-task1-v0
run_group=puzzle1_QC-MPO-beta$beta-temp$temp-v10

# IPPO
for seed in 42 43 44
do
  python main.py \
    --offline_steps=500000 \
    --start_training=5000 \
    --agent=agents/acmpo.py \
    --agent.beta=$beta \
    --agent.temperature=$temp \
    --env_name=$env_name \
    --save_dir=$save_dir \
    --horizon_length=5 \
    --project_name=$project_name \
    --run_group=$run_group \
    --exp_name=seed-$seed-$run_group \
    --seed=$seed \
    --debug=False \
    --agent.q_loss_clip=$q_clip \
    --agent.q_agg="mean" \
    --online_steps=500000 \
    --eval_interval=10000 \
    --save_interval=100000
done
