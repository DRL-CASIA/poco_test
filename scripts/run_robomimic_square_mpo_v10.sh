#!/bin/bash

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
export TMPDIR=${TMPDIR:-/data/jzn/qc_clean_runs/tmp}
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR"

beta=1.0
temp=0.001
utd=1
use_bc_online=${1:-False}  # Second argument, default False
q_loss_clip=0.08         

project_name=qc_off2on
save_dir=${SAVE_DIR:-/data/jzn/qc_clean_runs/exp}
env_name=square-mh-low_dim

# Add bc_online suffix if enabled
if [ "$use_bc_online" = "True" ]; then
  run_group=square_QC-MPO-final-bc_online-clip$q_loss_clip
else
  run_group=square_QC-MPO-final-clip$q_loss_clip
fi

# QC-FQL-MPO
for seed in 42 43 44
do
  python main.py \
    --start_training=5000 \
    --agent=agents/acmpo.py \
    --agent.beta=$beta \
    --agent.temperature=$temp \
    --agent.use_bc_online=$use_bc_online \
    --env_name=$env_name \
    --save_dir=$save_dir \
    --horizon_length=5 \
    --project_name=$project_name \
    --run_group=$run_group \
    --exp_name=seed-$seed-$run_group \
    --seed=$seed \
    --debug=False  \
    --offline_steps=300000 \
    --utd_ratio=$utd \
    --agent.q_loss_clip=$q_loss_clip \
    --agent.q_agg="mean" \
    --save_interval=100000
done
